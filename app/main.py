import secrets
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import bcrypt
import structlog
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import get_settings
from app.models.db import (
    Agent, AuditLog, Company, Memory, MemoryCandidate,
    PipelineTrace, get_db, get_engine, utcnow,
)
from app.schemas.memory import (
    AgentCreate, AgentOut,
    AuditLogOut,
    CandidateOut,
    CompanyCreate, CompanyCreated, CompanyOut,
    MemoryAddRequest, MemoryAddResponse,
    MemoryOut,
    MemorySearchRequest, MemorySearchResponse, MemorySearchResult,
    TraceOut,
)
from app.services.extraction.pipeline import run_pipeline, run_search

log = structlog.get_logger()
settings = get_settings()

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
        api_key_hash=hashed,
        api_key_prefix=raw_key[:8],
    )
    db.add(company)
    await db.commit()
    await db.refresh(company)
    log.info("company_created", id=str(company.id), name=company.name)
    return CompanyCreated(
        id=company.id,
        name=company.name,
        api_key_prefix=company.api_key_prefix,
        is_active=company.is_active,
        created_at=company.created_at,
        api_key=raw_key,
    )


@app.get("/companies/me", response_model=CompanyOut, tags=["Companies"])
async def get_company_me(company: Company = Depends(get_company)):
    return company


# ── Agent routes ─────────────────────────────────────────────────────────────

@app.post("/agents", response_model=AgentOut, status_code=201, tags=["Agents"])
async def create_agent(
    payload: AgentCreate,
    company: Company = Depends(get_company),
    db: AsyncSession = Depends(get_db),
):
    existing = await db.execute(
        select(Agent).where(Agent.company_id == company.id, Agent.slug == payload.slug)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail=f"Agent slug '{payload.slug}' already exists")

    agent = Agent(
        company_id=company.id,
        slug=payload.slug,
        name=payload.name,
        extraction_instructions=payload.extraction_instructions,
        config=payload.config,
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


@app.get("/agents/{slug}", response_model=AgentOut, tags=["Agents"])
async def get_agent(
    slug: str,
    company: Company = Depends(get_company),
    db: AsyncSession = Depends(get_db),
):
    return await get_agent_by_slug(slug, company, db)


# ── Memory routes ────────────────────────────────────────────────────────────

@app.post("/memory/add", response_model=MemoryAddResponse, tags=["Memory"])
async def add_memory(
    payload: MemoryAddRequest,
    company: Company = Depends(get_company),
    db: AsyncSession = Depends(get_db),
    qdrant: QdrantClient = Depends(get_qdrant),
):
    """Run the extraction pipeline on a conversation and persist resulting memories."""
    agent = await get_agent_by_slug(payload.slug, company, db)

    trace = PipelineTrace(
        company_id=company.id,
        agent_id=agent.id,
        user_id=payload.user_id,
        session_id=payload.session_id,
        input_messages=[m.model_dump() for m in payload.messages],
        status="processing",
    )
    db.add(trace)
    await db.flush()  # get trace.id before pipeline starts

    try:
        persisted_memories, persisted_candidates, rejected_candidates = await run_pipeline(
            trace=trace,
            messages=[m.model_dump() for m in payload.messages],
            agent=agent,
            user_id=payload.user_id,
            session_id=payload.session_id,
            extra_metadata=payload.metadata,
            db=db,
            qdrant_client=qdrant,
        )
    except Exception as e:
        trace.status = "failed"
        trace.error = str(e)
        trace.completed_at = utcnow()
        await db.commit()
        log.error("pipeline_failed", trace_id=str(trace.id), error=str(e))
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")

    log.info(
        "pipeline_complete",
        trace_id=str(trace.id),
        persisted=len(persisted_memories),
        rejected=len(rejected_candidates),
    )

    return MemoryAddResponse(
        trace_id=trace.id,
        persisted=[_memory_to_schema(m) for m in persisted_memories],
        rejected=[CandidateOut.model_validate(c) for c in rejected_candidates],
        candidates=[CandidateOut.model_validate(c) for c in (persisted_candidates + rejected_candidates)],
    )


@app.post("/memory/search", response_model=MemorySearchResponse, tags=["Memory"])
async def search_memory(
    payload: MemorySearchRequest,
    company: Company = Depends(get_company),
    db: AsyncSession = Depends(get_db),
    qdrant: QdrantClient = Depends(get_qdrant),
):
    """Search memories for a user using semantic similarity."""
    agent = await get_agent_by_slug(payload.slug, company, db)

    results, rewritten_query = await run_search(
        query=payload.query,
        agent=agent,
        user_id=payload.user_id,
        limit=payload.limit,
        session_id=payload.session_id,
        db=db,
        qdrant_client=qdrant,
    )

    search_results = [
        MemorySearchResult(
            **_memory_to_schema(m).model_dump(),
            similarity_score=score,
            retrieval_reason=reason,
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


# ── Helpers ──────────────────────────────────────────────────────────────────

def _memory_to_schema(m: Memory) -> MemoryOut:
    return MemoryOut.model_validate(m)
