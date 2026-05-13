"""
Full extraction pipeline: extract → classify → deduplicate → decide → persist.
Every LLM call is logged onto the MemoryCandidate.llm_responses dict.
"""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import litellm
from qdrant_client.http.models import FieldCondition, Filter, MatchValue
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import get_settings
from app.models.db import Agent, AuditLog, Memory, MemoryCandidate, PipelineTrace, utcnow

PROMPTS_DIR = Path(__file__).parent.parent.parent.parent / "prompts"


def _load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.txt").read_text()


def _make_retrieval_reason(score: float, category: str) -> str:
    if score >= 0.8:
        return f"Strong semantic match on {category} memory"
    if score >= 0.6:
        return f"Good semantic match on {category} memory"
    if score >= 0.4:
        return f"Moderate semantic match on {category} memory"
    return f"Weak semantic match on {category} memory"


def _fmt(template: str, **kwargs: Any) -> str:
    for key, value in kwargs.items():
        template = template.replace("{" + key + "}", str(value))
    return template


settings = get_settings()


async def _llm_call(prompt: str, step: str) -> tuple[str, dict]:
    """Call LiteLLM and return (content, usage_dict)."""
    response = await litellm.acompletion(
        model=settings.litellm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    content = response.choices[0].message.content or ""
    usage = {
        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        "total_tokens": response.usage.total_tokens if response.usage else 0,
    }
    return content, usage


async def _embed(text: str) -> list[float]:
    response = await litellm.aembedding(
        model=settings.embedding_model,
        input=[text],
    )
    return response.data[0]["embedding"]


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
    return json.loads(raw)


async def _extract_step(
    messages: list[dict],
    agent: Agent,
    trace: PipelineTrace,
    db: AsyncSession,
) -> tuple[list[dict], int]:
    """Returns list of {content, rationale} dicts and total tokens used."""
    user_messages = [m for m in messages if m.get("role") == "user"]
    conversation = "\n".join(m["content"] for m in user_messages)
    prompt = _fmt(
        _load_prompt("extract"),
        extraction_instructions=agent.extraction_instructions,
        messages=conversation,
    )
    raw, usage = await _llm_call(prompt, "extract")
    parsed = _parse_json(raw)
    candidates = parsed.get("candidates", [])
    return candidates, usage["total_tokens"]


async def _classify_step(
    content: str,
    agent: Agent,
    candidate: MemoryCandidate,
    db: AsyncSession,
) -> tuple[str, float, int]:
    """Returns (memory_type, importance_score, tokens)."""
    prompt = _fmt(
        _load_prompt("classify"),
        content=content,
        extraction_instructions=agent.extraction_instructions,
    )
    raw, usage = await _llm_call(prompt, "classify")
    parsed = _parse_json(raw)
    candidate.llm_responses = {
        **candidate.llm_responses,
        "classify": {"raw": raw, "parsed": parsed, "usage": usage},
    }
    return parsed["memory_type"], float(parsed["importance_score"]), usage["total_tokens"]


async def _search_similar_memories(
    content: str,
    agent: Agent,
    user_id: str,
    qdrant_client: Any,
    limit: int = 5,
) -> list[dict]:
    """Search Qdrant for similar memories for this user+agent."""
    try:
        embedding = await _embed(content)
        results = qdrant_client.query_points(
            collection_name=settings.qdrant_collection,
            query=embedding,
            limit=limit,
            query_filter=Filter(must=[
                FieldCondition(key="agent_id", match=MatchValue(value=str(agent.id))),
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                FieldCondition(key="deleted", match=MatchValue(value=False)),
            ]),
            with_payload=True,
        )
        return [
            {
                "memory_id": r.payload.get("memory_id"),
                "content": r.payload.get("content", ""),
                "score": r.score,
            }
            for r in results.points
        ]
    except Exception:
        return []


async def _dedup_step(
    content: str,
    similar: list[dict],
    candidate: MemoryCandidate,
    db: AsyncSession,
) -> tuple[str, str | None, int]:
    """Returns (decision, target_memory_id_str, tokens)."""
    if not similar:
        candidate.llm_responses = {
            **candidate.llm_responses,
            "deduplicate": {"decision": "new", "target_memory_id": None, "skipped": "no_similar"},
        }
        return "new", None, 0

    existing_str = "\n".join(
        f"[{i+1}] ID={m['memory_id']} (similarity={m['score']:.2f}): {m['content']}"
        for i, m in enumerate(similar)
    )
    prompt = _fmt(
        _load_prompt("deduplicate"),
        candidate_content=content,
        existing_memories=existing_str,
    )
    raw, usage = await _llm_call(prompt, "deduplicate")
    parsed = _parse_json(raw)
    candidate.llm_responses = {
        **candidate.llm_responses,
        "deduplicate": {"raw": raw, "parsed": parsed, "usage": usage},
    }
    return parsed["decision"], parsed.get("target_memory_id"), usage["total_tokens"]


async def _decide_step(
    content: str,
    memory_type: str,
    importance_score: float,
    dedup_decision: str,
    agent: Agent,
    candidate: MemoryCandidate,
    db: AsyncSession,
) -> tuple[str, str | None, int]:
    """Returns (decision, rejection_reason, tokens)."""
    prompt = _fmt(
        _load_prompt("decide"),
        content=content,
        memory_type=memory_type,
        importance_score=importance_score,
        dedup_decision=dedup_decision,
        extraction_instructions=agent.extraction_instructions,
    )
    raw, usage = await _llm_call(prompt, "decide")
    parsed = _parse_json(raw)
    candidate.llm_responses = {
        **candidate.llm_responses,
        "decide": {"raw": raw, "parsed": parsed, "usage": usage},
    }
    return parsed["decision"], parsed.get("rejection_reason"), usage["total_tokens"]


async def _persist_memory(
    candidate: MemoryCandidate,
    agent: Agent,
    user_id: str,
    session_id: str | None,
    dedup_decision: str,
    dedup_target_id: str | None,
    extra_metadata: dict,
    db: AsyncSession,
    qdrant_client: Any,
) -> Memory:
    """Save memory to Postgres + Qdrant, handle supersede if update."""
    mem_uuid = uuid.uuid4()

    # Supersede old memory if this is an update
    if dedup_decision == "update" and dedup_target_id:
        try:
            old_id = uuid.UUID(dedup_target_id)
            result = await db.execute(select(Memory).where(Memory.id == old_id))
            old_mem = result.scalar_one_or_none()
            if old_mem and not old_mem.deleted_at:
                old_mem.deleted_at = utcnow()
                old_mem.superseded_by_id = mem_uuid
                db.add(AuditLog(
                    company_id=agent.company_id,
                    agent_id=agent.id,
                    entity_type="memory",
                    entity_id=old_id,
                    action="superseded",
                    actor="system",
                    details={"superseded_by": str(mem_uuid), "candidate_id": str(candidate.id)},
                ))
                # Soft-delete in Qdrant
                try:
                    qdrant_client.set_payload(
                        collection_name=settings.qdrant_collection,
                        payload={"deleted": True},
                        points=[old_mem.qdrant_id],
                    )
                except Exception:
                    pass
        except (ValueError, Exception):
            pass

    # Embed and upsert to Qdrant
    qdrant_point_id = str(uuid.uuid4())
    embedding = await _embed(candidate.content)
    try:
        qdrant_client.upsert(
            collection_name=settings.qdrant_collection,
            points=[{
                "id": qdrant_point_id,
                "vector": embedding,
                "payload": {
                    "memory_id": str(mem_uuid),
                    "agent_id": str(agent.id),
                    "company_id": str(agent.company_id),
                    "user_id": user_id,
                    "session_id": session_id,
                    "content": candidate.content,
                    "memory_type": candidate.memory_type,
                    "importance_score": candidate.importance_score,
                    "deleted": False,
                },
            }],
        )
    except Exception:
        qdrant_point_id = None

    memory = Memory(
        id=mem_uuid,
        company_id=agent.company_id,
        agent_id=agent.id,
        user_id=user_id,
        session_id=session_id,
        content=candidate.content,
        memory_type=candidate.memory_type or "fact",
        importance_score=candidate.importance_score or 0.5,
        metadata_={**extra_metadata, "trace_id": str(candidate.trace_id)},
        qdrant_id=qdrant_point_id,
    )
    db.add(memory)

    db.add(AuditLog(
        company_id=agent.company_id,
        agent_id=agent.id,
        entity_type="memory",
        entity_id=mem_uuid,
        action="created",
        actor="system",
        details={
            "candidate_id": str(candidate.id),
            "dedup_decision": dedup_decision,
            "importance_score": candidate.importance_score,
        },
    ))

    candidate.memory_id = mem_uuid
    return memory


async def run_pipeline(
    trace: PipelineTrace,
    messages: list[dict],
    agent: Agent,
    user_id: str,
    session_id: str | None,
    extra_metadata: dict,
    db: AsyncSession,
    qdrant_client: Any,
) -> tuple[list[Memory], list[MemoryCandidate], list[MemoryCandidate]]:
    """
    Run the full 4-step pipeline.
    Returns (persisted_memories, persisted_candidates, rejected_candidates).
    """
    total_tokens = 0
    llm_calls = 0

    # ── Step 1: Extract ──────────────────────────────────────────────────────
    raw_candidates, tokens = await _extract_step(messages, agent, trace, db)
    total_tokens += tokens
    llm_calls += 1

    all_candidates: list[MemoryCandidate] = []
    persisted_memories: list[Memory] = []
    persisted_candidates: list[MemoryCandidate] = []
    rejected_candidates: list[MemoryCandidate] = []

    for raw in raw_candidates:
        content = raw.get("content", "").strip()
        if not content:
            continue

        candidate = MemoryCandidate(
            trace_id=trace.id,
            content=content,
            llm_responses={"extract": {"rationale": raw.get("rationale", "")}},
        )
        db.add(candidate)
        all_candidates.append(candidate)

        # ── Step 2: Classify ─────────────────────────────────────────────────
        try:
            mem_type, importance, tokens = await _classify_step(content, agent, candidate, db)
            total_tokens += tokens
            llm_calls += 1
            candidate.memory_type = mem_type
            candidate.importance_score = importance
        except Exception as e:
            candidate.memory_type = "fact"
            candidate.importance_score = 0.5
            candidate.llm_responses = {**candidate.llm_responses, "classify_error": str(e)}

        # ── Step 3: Deduplicate ───────────────────────────────────────────────
        try:
            similar = await _search_similar_memories(content, agent, user_id, qdrant_client)
            dedup_decision, dedup_target_id, tokens = await _dedup_step(content, similar, candidate, db)
            total_tokens += tokens
            if tokens > 0:
                llm_calls += 1
            candidate.dedup_decision = dedup_decision
            candidate.dedup_target_id = uuid.UUID(dedup_target_id) if dedup_target_id else None
        except Exception as e:
            dedup_decision = "new"
            dedup_target_id = None
            candidate.dedup_decision = "new"
            candidate.llm_responses = {**candidate.llm_responses, "dedup_error": str(e)}

        # ── Step 4: Decide ───────────────────────────────────────────────────
        try:
            decision, rejection_reason, tokens = await _decide_step(
                content,
                candidate.memory_type or "fact",
                candidate.importance_score or 0.5,
                dedup_decision,
                agent,
                candidate,
                db,
            )
            total_tokens += tokens
            llm_calls += 1
        except Exception as e:
            decision = "persist" if (candidate.importance_score or 0) >= 0.3 else "reject"
            rejection_reason = None if decision == "persist" else "decide_step_error"
            candidate.llm_responses = {**candidate.llm_responses, "decide_error": str(e)}

        candidate.final_decision = decision
        candidate.rejection_reason = rejection_reason

        if decision == "persist":
            try:
                memory = await _persist_memory(
                    candidate, agent, user_id, session_id,
                    dedup_decision, str(dedup_target_id) if dedup_target_id else None,
                    extra_metadata, db, qdrant_client,
                )
                persisted_memories.append(memory)
                persisted_candidates.append(candidate)
            except Exception as e:
                candidate.final_decision = "reject"
                candidate.rejection_reason = f"persist_error: {e}"
                rejected_candidates.append(candidate)
        else:
            rejected_candidates.append(candidate)

    # ── Finalize trace ───────────────────────────────────────────────────────
    trace.status = "completed"
    trace.completed_at = utcnow()
    trace.llm_calls_total = llm_calls
    trace.tokens_used = total_tokens

    await db.commit()

    # Refresh candidates to get generated IDs
    for c in all_candidates:
        await db.refresh(c)
    for m in persisted_memories:
        await db.refresh(m)

    return persisted_memories, persisted_candidates, rejected_candidates


async def run_search(
    query: str,
    company_id: uuid.UUID,
    agent: Agent | None,
    user_id: str | None,
    limit: int,
    session_id: str | None,
    db: AsyncSession,
    qdrant_client: Any,
) -> list[tuple[Memory, float, str]]:
    """
    Search memories scoped to company. agent and user_id are optional filters.
    Returns [(memory, score, reason)].
    """
    # Build Qdrant filter — company is always required, agent/user are optional
    embedding = await _embed(query)
    must_filters = [
        FieldCondition(key="company_id", match=MatchValue(value=str(company_id))),
        FieldCondition(key="deleted", match=MatchValue(value=False)),
    ]
    if agent:
        must_filters.append(FieldCondition(key="agent_id", match=MatchValue(value=str(agent.id))))
    if user_id:
        must_filters.append(FieldCondition(key="user_id", match=MatchValue(value=user_id)))
    if session_id:
        must_filters.append(FieldCondition(key="session_id", match=MatchValue(value=session_id)))

    search_results = qdrant_client.query_points(
        collection_name=settings.qdrant_collection,
        query=embedding,
        limit=limit,
        query_filter=Filter(must=must_filters),
        with_payload=True,
    )

    results = []
    for r in search_results.points:
        mem_id_str = r.payload.get("memory_id")
        if not mem_id_str:
            continue
        try:
            mem_id = uuid.UUID(mem_id_str)
        except ValueError:
            continue

        result = await db.execute(
            select(Memory).where(
                Memory.id == mem_id,
                Memory.company_id == company_id,
                Memory.deleted_at.is_(None),
            )
        )
        memory = result.scalar_one_or_none()
        if not memory:
            continue

        reason = _make_retrieval_reason(r.score, memory.memory_type)

        results.append((memory, r.score, reason))

    return results
