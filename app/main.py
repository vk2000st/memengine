import asyncio
import secrets
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import bcrypt
import litellm
import structlog
from cryptography.fernet import Fernet, InvalidToken
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from sqlalchemy import case, func, literal, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import get_settings
from app.models.db import (
    Agent, AuditLog, Company, Memory, MemoryCandidate,
    PipelineTrace, TraceReport, get_db, get_engine, get_session_factory, utcnow,
)
from app.schemas.memory import (
    AgentCreate, AgentOut,
    AgentMemoryCount, CategoryCount, MemoryDetail, TimelineEvent, UserProfile, UserSummary,
    AnalyticsOverview, RecentTrace,
    AuditLogOut,
    CandidateOut,
    CompanyCreate, CompanyCreated, CompanyOut,
    MemoryAddAccepted, MemoryAddRequest,
    MemoryOut,
    MemorySearchRequest, MemorySearchResponse, MemorySearchResult,
    PlaygroundChatRequest, PlaygroundChatResponse, PlaygroundSessionOut,
    TraceOut, TraceReportCreate, TraceReportOut, TraceStatusOut,
)
from app.services.extraction.pipeline import run_pipeline, run_search

log = structlog.get_logger()
settings = get_settings()

# In-memory rate limiters for public playground endpoints: IP → list[request_time]
_playground_rate: dict[str, list[datetime]] = defaultdict(list)
_chat_rate: dict[str, list[datetime]] = defaultdict(list)
_PLAYGROUND_WINDOW = timedelta(hours=1)


def _playground_rate_ok(ip: str) -> bool:
    now = datetime.now(timezone.utc)
    cutoff = now - _PLAYGROUND_WINDOW
    _playground_rate[ip] = [t for t in _playground_rate[ip] if t > cutoff]
    if len(_playground_rate[ip]) >= 20:
        return False
    _playground_rate[ip].append(now)
    return True


def _chat_rate_ok(ip: str) -> bool:
    now = datetime.now(timezone.utc)
    cutoff = now - _PLAYGROUND_WINDOW
    _chat_rate[ip] = [t for t in _chat_rate[ip] if t > cutoff]
    if len(_chat_rate[ip]) >= 10:
        return False
    _chat_rate[ip].append(now)
    return True


# ── Encryption helpers ───────────────────────────────────────────────────────

def _fernet() -> Fernet:
    if not settings.encryption_key:
        raise RuntimeError("ENCRYPTION_KEY is not configured")
    return Fernet(settings.encryption_key.encode())


def encrypt_api_key(raw: str) -> str:
    return _fernet().encrypt(raw.encode()).decode()


def decrypt_api_key(token: str) -> str | None:
    try:
        return _fernet().decrypt(token.encode()).decode()
    except (InvalidToken, Exception):
        return None


# ── Email helper ─────────────────────────────────────────────────────────────

async def _send_report_email(
    company_name: str,
    agent_slug: str,
    agent_instructions: str,
    user_id: str,
    trace_id: uuid.UUID,
    session_id: str | None,
    created_at: datetime,
    user_messages: list[str],
    reason: str,
    note: str | None,
    persisted_count: int,
    rejected_count: int,
) -> None:
    if not settings.postmark_server_token:
        log.warning("postmark_not_configured_skipping_report_email")
        return

    conversation = "\n".join(f"- {m}" for m in user_messages) or "(no user messages)"
    body = f"""New trace report from {company_name}

Reason: {reason}
Note: {note or "none"}

Company: {company_name}
Agent: {agent_slug}
User: {user_id}
Trace ID: {trace_id}
Session: {session_id or "none"}
Time: {created_at.isoformat()}

User said:
{conversation}

Extraction instructions:
{agent_instructions}

Pipeline result: {persisted_count} persisted, {rejected_count} rejected

View trace: https://redorb.tech/dashboard/traces?id={trace_id}
"""

    def _send() -> None:
        from postmarker.core import PostmarkClient
        client = PostmarkClient(server_token=settings.postmark_server_token)
        client.emails.send(
            From=settings.postmark_from,
            To=settings.report_email_to,
            Subject=f"New trace report — {company_name} / {agent_slug}",
            TextBody=body,
        )

    try:
        await asyncio.to_thread(_send)
        log.info("report_email_sent", trace_id=str(trace_id), to=settings.report_email_to)
    except Exception as e:
        log.error("report_email_failed", trace_id=str(trace_id), error=str(e))


_qdrant: QdrantClient | None = None


def get_qdrant() -> QdrantClient:
    return _qdrant


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _qdrant

    # Qdrant
    _qdrant = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        timeout=10,
    )
    try:
        _qdrant.get_collection(settings.qdrant_collection)
    except Exception:
        _qdrant.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=settings.embedding_dim, distance=Distance.COSINE),
        )
        log.info("qdrant_collection_created", name=settings.qdrant_collection)

    log.info("startup_complete")
    yield
    log.info("shutdown")


app = FastAPI(
    title="MemEngine",
    version="1.0.0",
    description="Production-grade AI memory engine",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Auth dependency ──────────────────────────────────────────────────────────

async def get_company(
    x_api_key: str = Header(..., alias="X-API-Key"),
    db: AsyncSession = Depends(get_db),
) -> Company:
    if not x_api_key or len(x_api_key) < 8:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    prefix = x_api_key[:8]
    result = await db.execute(
        select(Company).where(Company.api_key_prefix == prefix, Company.is_active == True)
    )
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    if not bcrypt.checkpw(x_api_key.encode(), company.api_key_hash.encode()):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    return company


async def get_agent_by_slug(
    agent_slug: str,
    company: Company,
    db: AsyncSession,
) -> Agent:
    result = await db.execute(
        select(Agent).where(
            Agent.company_id == company.id,
            Agent.slug == agent_slug,
            Agent.is_active == True,
        )
    )
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_slug}' not found")
    return agent


# ── Error handler ────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    log.error("unhandled_exception", path=request.url.path, error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ── Company routes ───────────────────────────────────────────────────────────

@app.post("/companies", response_model=CompanyCreated, status_code=201, tags=["Companies"])
async def create_company(payload: CompanyCreate, db: AsyncSession = Depends(get_db)):
    """Create a new tenant company and return its API key (shown once)."""
    raw_key = "mem_" + secrets.token_urlsafe(32)
    hashed = bcrypt.hashpw(raw_key.encode(), bcrypt.gensalt()).decode()
    company = Company(
        name=payload.name,
        email=payload.email,
        api_key_hash=hashed,
        api_key_prefix=raw_key[:8],
        api_key_encrypted=encrypt_api_key(raw_key),
    )
    db.add(company)
    await db.commit()
    await db.refresh(company)
    log.info("company_created", id=str(company.id), name=company.name)
    return CompanyCreated(
        id=company.id,
        name=company.name,
        email=company.email,
        api_key_prefix=company.api_key_prefix,
        is_active=company.is_active,
        created_at=company.created_at,
        api_key=raw_key,
    )


@app.get("/companies/me", response_model=CompanyOut, tags=["Companies"])
async def get_company_me(company: Company = Depends(get_company)):
    """Return the authenticated company, including the decrypted API key if available."""
    api_key = decrypt_api_key(company.api_key_encrypted) if company.api_key_encrypted else None
    return CompanyOut(
        id=company.id,
        name=company.name,
        email=company.email,
        api_key_prefix=company.api_key_prefix,
        api_key=api_key,
        is_active=company.is_active,
        created_at=company.created_at,
    )


@app.post("/companies/me/regenerate-key", response_model=CompanyOut, tags=["Companies"])
async def regenerate_api_key(
    company: Company = Depends(get_company),
    db: AsyncSession = Depends(get_db),
):
    """
    Rotate the API key. The old key is immediately invalidated.
    Returns the new key in plaintext — store it; this is the only time it appears
    outside of GET /companies/me.
    """
    raw_key = "mem_" + secrets.token_urlsafe(32)
    company.api_key_hash = bcrypt.hashpw(raw_key.encode(), bcrypt.gensalt()).decode()
    company.api_key_prefix = raw_key[:8]
    company.api_key_encrypted = encrypt_api_key(raw_key)
    db.add(company)
    await db.commit()
    await db.refresh(company)
    log.info("api_key_regenerated", company_id=str(company.id))
    return CompanyOut(
        id=company.id,
        name=company.name,
        email=company.email,
        api_key_prefix=company.api_key_prefix,
        api_key=raw_key,
        is_active=company.is_active,
        created_at=company.created_at,
    )


@app.get("/companies/by-email", response_model=CompanyOut, tags=["Companies"])
async def get_company_by_email(email: str, db: AsyncSession = Depends(get_db)):
    """Look up a company by email. Used by the onboarding flow to check for an existing account."""
    result = await db.execute(
        select(Company).where(Company.email == email, Company.is_active == True)
    )
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(status_code=404, detail="No account found for that email")
    return company


# ── Agent routes ─────────────────────────────────────────────────────────────

@app.post("/agents", response_model=AgentOut, status_code=201, tags=["Agents"])
async def create_agent(
    payload: AgentCreate,
    company: Company = Depends(get_company),
    db: AsyncSession = Depends(get_db),
):
    existing = await db.execute(
        select(Agent).where(Agent.company_id == company.id, Agent.slug == payload.agent_slug)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail=f"Agent slug '{payload.agent_slug}' already exists")

    agent = Agent(
        company_id=company.id,
        slug=payload.agent_slug,
        name=payload.name,
        extraction_instructions=payload.extraction_instructions,
    )
    db.add(agent)
    await db.commit()
    await db.refresh(agent)
    log.info("agent_created", id=str(agent.id), slug=agent.slug)
    return agent


@app.get("/agents", response_model=list[AgentOut], tags=["Agents"])
async def list_agents(
    company: Company = Depends(get_company),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Agent).where(Agent.company_id == company.id, Agent.is_active == True)
    )
    return result.scalars().all()


@app.get("/agents/{agent_slug}", response_model=AgentOut, tags=["Agents"])
async def get_agent(
    agent_slug: str,
    company: Company = Depends(get_company),
    db: AsyncSession = Depends(get_db),
):
    return await get_agent_by_slug(agent_slug, company, db)


# ── Memory routes ────────────────────────────────────────────────────────────

async def _run_pipeline_background(
    trace_id: uuid.UUID,
    agent_id: uuid.UUID,
    messages: list[dict],
    user_id: str,
    session_id: str | None,
    extra_metadata: dict,
) -> None:
    """Background task: run the full extraction pipeline after the HTTP response is sent."""
    factory = get_session_factory()
    async with factory() as db:
        trace_row = await db.execute(select(PipelineTrace).where(PipelineTrace.id == trace_id))
        trace = trace_row.scalar_one()
        agent_row = await db.execute(select(Agent).where(Agent.id == agent_id))
        agent = agent_row.scalar_one()
        try:
            await run_pipeline(
                trace=trace,
                messages=messages,
                agent=agent,
                user_id=user_id,
                session_id=session_id,
                extra_metadata=extra_metadata,
                db=db,
                qdrant_client=_qdrant,
            )
            log.info("bg_pipeline_complete", trace_id=str(trace_id))
        except Exception as e:
            trace.status = "failed"
            trace.error = str(e)
            trace.completed_at = utcnow()
            await db.commit()
            log.error("bg_pipeline_failed", trace_id=str(trace_id), error=str(e))


@app.post("/memory/add", response_model=MemoryAddAccepted, status_code=202, tags=["Memory"])
async def add_memory(
    payload: MemoryAddRequest,
    background_tasks: BackgroundTasks,
    company: Company = Depends(get_company),
    db: AsyncSession = Depends(get_db),
):
    """
    Enqueue the extraction pipeline and return immediately.
    Poll GET /trace/{trace_id}/status to check completion.
    """
    agent = await get_agent_by_slug(payload.agent_slug, company, db)

    trace = PipelineTrace(
        company_id=company.id,
        agent_id=agent.id,
        user_id=payload.user_id,
        session_id=payload.session_id,
        input_messages=[m.model_dump() for m in payload.messages],
        status="processing",
    )
    db.add(trace)
    await db.flush()
    await db.commit()  # persisted before the background task starts

    background_tasks.add_task(
        _run_pipeline_background,
        trace_id=trace.id,
        agent_id=agent.id,
        messages=[m.model_dump() for m in payload.messages],
        user_id=payload.user_id,
        session_id=payload.session_id,
        extra_metadata=payload.metadata,
    )

    log.info("memory_add_enqueued", trace_id=str(trace.id), agent=agent.slug, user=payload.user_id)
    return MemoryAddAccepted(trace_id=trace.id, status="processing")


@app.post("/memory/search", response_model=MemorySearchResponse, tags=["Memory"])
async def search_memory(
    payload: MemorySearchRequest,
    company: Company = Depends(get_company),
    db: AsyncSession = Depends(get_db),
    qdrant: QdrantClient = Depends(get_qdrant),
):
    """Search memories scoped to the authenticated company. agent_slug and user_id are optional filters."""
    agent = None
    if payload.agent_slug:
        agent = await get_agent_by_slug(payload.agent_slug, company, db)

    results, rewritten_query = await run_search(
        query=payload.query,
        company_id=company.id,
        agent=agent,
        user_id=payload.user_id,
        limit=payload.limit,
        session_id=payload.session_id,
        db=db,
        qdrant_client=qdrant,
    )

    # Batch-resolve agent slugs from the unique agent_ids in the result set
    agent_slug_map: dict[uuid.UUID, str] = {}
    agent_ids = {m.agent_id for m, _, _ in results}
    if agent_ids:
        slug_rows = (await db.execute(
            select(Agent.id, Agent.slug).where(Agent.id.in_(agent_ids))
        )).all()
        agent_slug_map = {row.id: row.slug for row in slug_rows}

    search_results = [
        MemorySearchResult(
            **_memory_to_schema(m).model_dump(),
            similarity_score=score,
            retrieval_reason=reason,
            agent_slug=agent_slug_map.get(m.agent_id),
        )
        for m, score, reason in results
    ]

    return MemorySearchResponse(results=search_results, rewritten_query=rewritten_query)


@app.delete("/memory/{memory_id}", status_code=204, tags=["Memory"])
async def delete_memory(
    memory_id: uuid.UUID,
    company: Company = Depends(get_company),
    db: AsyncSession = Depends(get_db),
    qdrant: QdrantClient = Depends(get_qdrant),
):
    """Soft-delete a memory."""
    result = await db.execute(
        select(Memory).where(Memory.id == memory_id, Memory.company_id == company.id)
    )
    memory = result.scalar_one_or_none()
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    if memory.deleted_at:
        raise HTTPException(status_code=409, detail="Memory already deleted")

    memory.deleted_at = utcnow()
    db.add(AuditLog(
        company_id=company.id,
        agent_id=memory.agent_id,
        entity_type="memory",
        entity_id=memory.id,
        action="deleted",
        actor="api",
        details={},
    ))

    if memory.qdrant_id:
        try:
            qdrant.set_payload(
                collection_name=settings.qdrant_collection,
                payload={"deleted": True},
                points=[memory.qdrant_id],
            )
        except Exception:
            pass

    await db.commit()


# ── Observability routes ─────────────────────────────────────────────────────

@app.get("/trace/{trace_id}/status", response_model=TraceStatusOut, tags=["Observability"])
async def get_trace_status(
    trace_id: uuid.UUID,
    company: Company = Depends(get_company),
    db: AsyncSession = Depends(get_db),
):
    """Poll pipeline status for an async /memory/add call."""
    result = await db.execute(
        select(PipelineTrace).where(
            PipelineTrace.id == trace_id,
            PipelineTrace.company_id == company.id,
        )
    )
    trace = result.scalar_one_or_none()
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")

    persisted_count: int = (await db.execute(
        select(func.count(MemoryCandidate.id)).where(
            MemoryCandidate.trace_id == trace_id,
            MemoryCandidate.final_decision == "persist",
        )
    )).scalar_one()

    rejected_count: int = (await db.execute(
        select(func.count(MemoryCandidate.id)).where(
            MemoryCandidate.trace_id == trace_id,
            MemoryCandidate.final_decision == "reject",
        )
    )).scalar_one()

    return TraceStatusOut(
        trace_id=trace.id,
        status=trace.status,
        persisted_count=persisted_count,
        rejected_count=rejected_count,
        error=trace.error,
    )


@app.post("/trace/{trace_id}/report", response_model=TraceReportOut, status_code=201, tags=["Observability"])
async def report_trace(
    trace_id: uuid.UUID,
    payload: TraceReportCreate,
    company: Company = Depends(get_company),
    db: AsyncSession = Depends(get_db),
):
    """Flag a pipeline trace for review and send an email notification."""
    trace_row = await db.execute(
        select(PipelineTrace).where(
            PipelineTrace.id == trace_id,
            PipelineTrace.company_id == company.id,
        )
    )
    trace = trace_row.scalar_one_or_none()
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")

    agent_row = await db.execute(select(Agent).where(Agent.id == trace.agent_id))
    agent = agent_row.scalar_one()

    persisted_count: int = (await db.execute(
        select(func.count(MemoryCandidate.id)).where(
            MemoryCandidate.trace_id == trace_id,
            MemoryCandidate.final_decision == "persist",
        )
    )).scalar_one()

    rejected_count: int = (await db.execute(
        select(func.count(MemoryCandidate.id)).where(
            MemoryCandidate.trace_id == trace_id,
            MemoryCandidate.final_decision == "reject",
        )
    )).scalar_one()

    report = TraceReport(
        trace_id=trace_id,
        company_id=company.id,
        reason=payload.reason,
        note=payload.note,
    )
    db.add(report)
    await db.commit()
    await db.refresh(report)
    log.info("trace_report_created", report_id=str(report.id), trace_id=str(trace_id))

    user_messages = [
        m["content"]
        for m in (trace.input_messages or [])
        if m.get("role") == "user"
    ]

    await _send_report_email(
        company_name=company.name,
        agent_slug=agent.slug,
        agent_instructions=agent.extraction_instructions,
        user_id=trace.user_id,
        trace_id=trace_id,
        session_id=trace.session_id,
        created_at=trace.created_at,
        user_messages=user_messages,
        reason=payload.reason,
        note=payload.note,
        persisted_count=persisted_count,
        rejected_count=rejected_count,
    )

    return TraceReportOut(success=True, report_id=report.id)


@app.get("/trace/{trace_id}", response_model=TraceOut, tags=["Observability"])
async def get_trace(
    trace_id: uuid.UUID,
    company: Company = Depends(get_company),
    db: AsyncSession = Depends(get_db),
):
    """Get the full pipeline trace including all candidates and LLM decisions."""
    result = await db.execute(
        select(PipelineTrace)
        .where(PipelineTrace.id == trace_id, PipelineTrace.company_id == company.id)
        .options(selectinload(PipelineTrace.candidates))
    )
    trace = result.scalar_one_or_none()
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    return TraceOut.model_validate(trace)


@app.get("/memory/{memory_id}/audit", response_model=list[AuditLogOut], tags=["Observability"])
async def get_memory_audit(
    memory_id: uuid.UUID,
    company: Company = Depends(get_company),
    db: AsyncSession = Depends(get_db),
):
    """Get full audit trail for a memory."""
    result = await db.execute(
        select(Memory).where(Memory.id == memory_id, Memory.company_id == company.id)
    )
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Memory not found")

    logs = await db.execute(
        select(AuditLog)
        .where(AuditLog.entity_id == memory_id, AuditLog.entity_type == "memory")
        .order_by(AuditLog.created_at)
    )
    return [AuditLogOut.model_validate(l) for l in logs.scalars().all()]


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "version": "1.0.0"}


# ── Analytics routes ─────────────────────────────────────────────────────────

@app.get("/analytics/overview", response_model=AnalyticsOverview, tags=["Analytics"])
async def analytics_overview(
    agent_slug: str | None = None,
    since: datetime | None = None,
    company: Company = Depends(get_company),
    db: AsyncSession = Depends(get_db),
):
    """
    Aggregate memory stats for the dashboard.
    Both agent_slug and since are optional filters.
    recent_traces always returns the last 10 regardless of filters.
    """
    # Resolve optional agent filter
    agent_id: uuid.UUID | None = None
    if agent_slug:
        agent_row = await db.execute(
            select(Agent).where(Agent.company_id == company.id, Agent.slug == agent_slug)
        )
        agent = agent_row.scalar_one_or_none()
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_slug}' not found")
        agent_id = agent.id

    # ── total_memories: active (not deleted) for this company [+ agent] ──────
    total_stmt = select(func.count(Memory.id)).where(
        Memory.company_id == company.id,
        Memory.deleted_at.is_(None),
    )
    if agent_id:
        total_stmt = total_stmt.where(Memory.agent_id == agent_id)
    total_memories: int = (await db.execute(total_stmt)).scalar_one()

    # ── memories_added: same filter + created_at >= since ────────────────────
    added_stmt = total_stmt  # inherits company + agent + deleted_at filters
    if since:
        added_stmt = added_stmt.where(Memory.created_at >= since)
    memories_added: int = (await db.execute(added_stmt)).scalar_one()

    # ── recent_traces: last 10 for this company, with per-candidate counts ───
    traces_stmt = (
        select(
            PipelineTrace.id,
            PipelineTrace.user_id,
            PipelineTrace.created_at,
            Agent.slug.label("agent_slug"),
            func.count(MemoryCandidate.id).label("candidates_total"),
            func.count(MemoryCandidate.id)
                .filter(MemoryCandidate.final_decision == "persist")
                .label("persisted_count"),
            func.count(MemoryCandidate.id)
                .filter(MemoryCandidate.final_decision == "reject")
                .label("rejected_count"),
        )
        .join(Agent, PipelineTrace.agent_id == Agent.id)
        .outerjoin(MemoryCandidate, MemoryCandidate.trace_id == PipelineTrace.id)
        .where(PipelineTrace.company_id == company.id)
        .group_by(PipelineTrace.id, PipelineTrace.user_id, PipelineTrace.created_at, Agent.slug)
        .order_by(PipelineTrace.created_at.desc())
        .limit(10)
    )
    traces_rows = (await db.execute(traces_stmt)).all()
    recent_traces = [
        RecentTrace(
            trace_id=row.id,
            agent_slug=row.agent_slug,
            user_id=row.user_id,
            candidates_total=row.candidates_total or 0,
            persisted_count=row.persisted_count or 0,
            rejected_count=row.rejected_count or 0,
            created_at=row.created_at,
        )
        for row in traces_rows
    ]

    return AnalyticsOverview(
        total_memories=total_memories,
        memories_added=memories_added,
        total_searches=0,
        recent_traces=recent_traces,
    )


# ── User routes ──────────────────────────────────────────────────────────────

@app.get("/users", response_model=list[UserSummary], tags=["Users"])
async def list_users(
    agent_slug: str | None = None,
    last_active_days: int | None = None,
    limit: int = 20,
    offset: int = 0,
    company: Company = Depends(get_company),
    db: AsyncSession = Depends(get_db),
):
    """List all users for the company with aggregate memory stats."""
    # Resolve optional agent filter
    agent_id: uuid.UUID | None = None
    if agent_slug:
        row = await db.execute(
            select(Agent).where(Agent.company_id == company.id, Agent.slug == agent_slug)
        )
        agent = row.scalar_one_or_none()
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_slug}' not found")
        agent_id = agent.id

    extra = [Memory.agent_id == agent_id] if agent_id else []

    # Aggregate per user_id
    stats_stmt = (
        select(
            Memory.user_id,
            func.count(Memory.id).label("total_memories"),
            func.max(Memory.created_at).label("last_memory_at"),
            func.count(func.distinct(Memory.session_id)).label("session_count"),
            func.mode().within_group(Memory.memory_type).label("top_category"),
        )
        .where(Memory.company_id == company.id, Memory.deleted_at.is_(None), *extra)
        .group_by(Memory.user_id)
        .order_by(func.max(Memory.created_at).desc())
    )

    if last_active_days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=last_active_days)
        stats_stmt = stats_stmt.having(func.max(Memory.created_at) >= cutoff)

    stats_stmt = stats_stmt.limit(limit).offset(offset)
    stats_rows = (await db.execute(stats_stmt)).all()

    # Fetch agent slugs for each returned user in one query
    user_ids = [r.user_id for r in stats_rows]
    agents_by_user: dict[str, list[str]] = defaultdict(list)
    if user_ids:
        agent_rows = (await db.execute(
            select(Memory.user_id, Agent.slug)
            .join(Agent, Memory.agent_id == Agent.id)
            .where(
                Memory.company_id == company.id,
                Memory.user_id.in_(user_ids),
                Memory.deleted_at.is_(None),
            )
            .distinct()
        )).all()
        for r in agent_rows:
            if r.slug not in agents_by_user[r.user_id]:
                agents_by_user[r.user_id].append(r.slug)

    return [
        UserSummary(
            user_id=r.user_id,
            total_memories=r.total_memories,
            last_memory_at=r.last_memory_at,
            session_count=r.session_count,
            top_category=r.top_category,
            agents=agents_by_user.get(r.user_id, []),
        )
        for r in stats_rows
    ]


@app.get("/users/{user_id}/profile", response_model=UserProfile, tags=["Users"])
async def get_user_profile(
    user_id: str,
    agent_slug: str | None = None,
    company: Company = Depends(get_company),
    db: AsyncSession = Depends(get_db),
):
    """Full memory profile for a single user, optionally filtered to one agent."""
    agent_id: uuid.UUID | None = None
    if agent_slug:
        row = await db.execute(
            select(Agent).where(Agent.company_id == company.id, Agent.slug == agent_slug)
        )
        agent = row.scalar_one_or_none()
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_slug}' not found")
        agent_id = agent.id

    mem_filter = [
        Memory.company_id == company.id,
        Memory.user_id == user_id,
        *([Memory.agent_id == agent_id] if agent_id else []),
    ]
    trace_filter = [
        PipelineTrace.company_id == company.id,
        PipelineTrace.user_id == user_id,
        *([PipelineTrace.agent_id == agent_id] if agent_id else []),
    ]

    # ── Basic stats ───────────────────────────────────────────────────────────
    stats = (await db.execute(
        select(
            func.min(Memory.created_at).label("first_seen"),
            func.max(Memory.created_at).label("last_active"),
            func.count(Memory.id).filter(Memory.deleted_at.is_(None)).label("total_memories"),
            func.count(func.distinct(Memory.session_id)).label("session_count"),
        ).where(*mem_filter)
    )).one()

    if stats.first_seen is None:
        raise HTTPException(status_code=404, detail="User not found")

    # ── Agents breakdown ──────────────────────────────────────────────────────
    agent_rows = (await db.execute(
        select(Agent.slug, func.count(Memory.id).label("memory_count"))
        .join(Memory, Memory.agent_id == Agent.id)
        .where(Memory.company_id == company.id, Memory.user_id == user_id, Memory.deleted_at.is_(None))
        .group_by(Agent.slug)
        .order_by(func.count(Memory.id).desc())
    )).all()

    # ── Categories breakdown ──────────────────────────────────────────────────
    cat_rows = (await db.execute(
        select(Memory.memory_type.label("category"), func.count(Memory.id).label("count"))
        .where(*mem_filter, Memory.deleted_at.is_(None))
        .group_by(Memory.memory_type)
        .order_by(func.count(Memory.id).desc())
    )).all()

    # ── Memories (with trace_id + confirmed_by via correlated subqueries) ─────
    trace_id_sq = (
        select(MemoryCandidate.trace_id)
        .where(MemoryCandidate.memory_id == Memory.id)
        .limit(1)
        .correlate(Memory)
        .scalar_subquery()
    )
    confirmed_sq = (
        select(func.count(func.distinct(PipelineTrace.session_id)))
        .join(MemoryCandidate, MemoryCandidate.trace_id == PipelineTrace.id)
        .where(
            or_(
                MemoryCandidate.memory_id == Memory.id,
                MemoryCandidate.dedup_target_id == Memory.id,
            )
        )
        .correlate(Memory)
        .scalar_subquery()
    )

    mem_rows = (await db.execute(
        select(
            Memory.id,
            Memory.content,
            Memory.memory_type,
            Memory.session_id,
            Memory.deleted_at,
            Memory.created_at,
            Memory.updated_at,
            Agent.slug.label("agent_slug"),
            trace_id_sq.label("trace_id"),
            confirmed_sq.label("confirmed_by"),
        )
        .join(Agent, Memory.agent_id == Agent.id)
        .where(*mem_filter)
        .order_by(Memory.created_at.desc())
    )).all()

    # ── Rejected count ────────────────────────────────────────────────────────
    rejected_count: int = (await db.execute(
        select(func.count(MemoryCandidate.id))
        .join(PipelineTrace, MemoryCandidate.trace_id == PipelineTrace.id)
        .where(*trace_filter, MemoryCandidate.final_decision == "reject")
    )).scalar_one()

    # ── Timeline ──────────────────────────────────────────────────────────────
    # Persisted memories → "added" or "updated" (based on dedup_decision)
    persisted_events = (await db.execute(
        select(
            Memory.created_at.label("at"),
            case(
                (MemoryCandidate.dedup_decision == "update", "updated"),
                else_="added",
            ).label("event_type"),
            Memory.content.label("memory_text"),
            Agent.slug.label("agent_slug"),
            Memory.session_id,
        )
        .join(Agent, Memory.agent_id == Agent.id)
        .outerjoin(MemoryCandidate, MemoryCandidate.memory_id == Memory.id)
        .where(*mem_filter, Memory.deleted_at.is_(None))
    )).all()

    # Soft-deleted memories → "deleted" event at deleted_at
    deleted_events = (await db.execute(
        select(
            Memory.deleted_at.label("at"),
            literal("deleted").label("event_type"),
            Memory.content.label("memory_text"),
            Agent.slug.label("agent_slug"),
            Memory.session_id,
        )
        .join(Agent, Memory.agent_id == Agent.id)
        .where(*mem_filter, Memory.deleted_at.isnot(None))
    )).all()

    # Rejected candidates → "rejected" event at trace created_at
    rejected_events = (await db.execute(
        select(
            PipelineTrace.created_at.label("at"),
            literal("rejected").label("event_type"),
            MemoryCandidate.content.label("memory_text"),
            Agent.slug.label("agent_slug"),
            PipelineTrace.session_id,
        )
        .join(PipelineTrace, MemoryCandidate.trace_id == PipelineTrace.id)
        .join(Agent, PipelineTrace.agent_id == Agent.id)
        .where(*trace_filter, MemoryCandidate.final_decision == "reject")
    )).all()

    timeline = sorted(
        [*persisted_events, *deleted_events, *rejected_events],
        key=lambda r: r.at,
        reverse=True,
    )

    return UserProfile(
        user_id=user_id,
        first_seen=stats.first_seen,
        last_active=stats.last_active,
        total_memories=stats.total_memories,
        session_count=stats.session_count,
        agents=[AgentMemoryCount(slug=r.slug, memory_count=r.memory_count) for r in agent_rows],
        categories=[CategoryCount(category=r.category, count=r.count) for r in cat_rows],
        memories=[
            MemoryDetail(
                memory_id=r.id,
                memory_text=r.content,
                category=r.memory_type,
                agent_slug=r.agent_slug,
                session_id=r.session_id,
                confirmed_by=r.confirmed_by or 0,
                status="deleted" if r.deleted_at else "active",
                created_at=r.created_at,
                updated_at=r.updated_at,
                trace_id=r.trace_id,
            )
            for r in mem_rows
        ],
        rejected_count=rejected_count,
        timeline=[
            TimelineEvent(
                at=r.at,
                event_type=r.event_type,
                memory_text=r.memory_text,
                agent_slug=r.agent_slug,
                session_id=r.session_id,
            )
            for r in timeline
        ],
    )


# ── Playground routes ────────────────────────────────────────────────────────

@app.post("/playground/session", response_model=PlaygroundSessionOut, tags=["Playground"])
async def create_playground_session(request: Request):
    """
    Public endpoint — no auth required. Rate limited: 20 calls per IP per hour.

    Pre-setup (run once against the live API, then set env vars):
      1. POST /companies  {"name": "Playground Demo"}
         → copy api_key  → DEMO_API_KEY in env
      2. POST /agents  {"agent_slug": "support-bot", "name": "Support Bot",
                        "extraction_instructions": "..."}  (auth with DEMO_API_KEY)
         → set DEMO_AGENT_SLUG=support-bot in env

    Demo users are ephemeral — no PII is stored. Each call returns a fresh user_id.
    Sessions expire after 2 turns (enforced by the frontend).
    """
    client_ip = request.headers.get("X-Forwarded-For", request.client.host or "unknown").split(",")[0].strip()

    if not _playground_rate_ok(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Max 20 playground sessions per IP per hour.",
        )

    if not settings.demo_api_key:
        raise HTTPException(status_code=503, detail="Playground is not configured on this server.")

    user_id = "demo_" + uuid.uuid4().hex[:8]

    return PlaygroundSessionOut(
        session_id=user_id,
        user_id=user_id,
        agent_slug=settings.demo_agent_slug,
        api_key=settings.demo_api_key,
    )


_SUPPORT_SYSTEM = (
    "You are a helpful customer support assistant for a SaaS product. "
    "Keep responses concise — 2 to 3 sentences max. "
    "If memories are provided about this user, use them naturally in your response "
    "without saying 'I remember' or 'Based on my memory'. Just use the context naturally."
)
_CHAT_MODEL = "openai/gpt-4o-mini"


@app.post("/playground/chat", response_model=PlaygroundChatResponse, tags=["Playground"])
async def playground_chat(payload: PlaygroundChatRequest, request: Request):
    """
    Public endpoint — no auth required. Rate limited: 10 calls per IP per hour.
    Calls gpt-4o-mini via LiteLLM. Returns a mock response if OPENAI_API_KEY is not set.
    """
    client_ip = request.headers.get("X-Forwarded-For", request.client.host or "unknown").split(",")[0].strip()
    if not _chat_rate_ok(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Max 10 chat messages per IP per hour.")

    valid_memories = [m for m in payload.memories if m.get("content")]
    memories_used = len(valid_memories)

    system = _SUPPORT_SYSTEM
    if valid_memories:
        context = "\n".join(m["content"] for m in valid_memories)
        system += f"\n\nWhat you know about this user:\n{context}"

    if not settings.openai_api_key:
        return PlaygroundChatResponse(
            response="Hi! This is a mock response — OPENAI_API_KEY is not configured. Set it to enable chat.",
            memories_used=memories_used,
        )

    result = await litellm.acompletion(
        model=_CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": payload.message},
        ],
        max_tokens=200,
        api_key=settings.openai_api_key,
    )
    text = result.choices[0].message.content or ""
    return PlaygroundChatResponse(response=text, memories_used=memories_used)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _memory_to_schema(m: Memory) -> MemoryOut:
    return MemoryOut.model_validate(m)
