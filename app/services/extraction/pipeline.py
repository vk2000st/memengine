"""
Full extraction pipeline: extract → classify → deduplicate → decide → persist.
Every LLM call is logged onto the MemoryCandidate.llm_responses dict.
"""
import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import litellm
from fastembed import TextEmbedding
from qdrant_client.http.models import FieldCondition, Filter, MatchValue
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import get_settings
from app.models.db import Agent, AuditLog, Memory, MemoryCandidate, PipelineTrace, utcnow

PROMPTS_DIR = Path(__file__).parent.parent.parent.parent / "prompts"
_fastembed_model = None


def _get_embed_model() -> TextEmbedding:
    global _fastembed_model
    if _fastembed_model is None:
        _fastembed_model = TextEmbedding(model_name="BAAI/bge-large-en-v1.5")
    return _fastembed_model


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
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        lambda: list(_get_embed_model().embed([text]))[0].tolist(),
    )
    return result


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


async def _extract_classify_step(
    messages: list[dict],
    agent: Agent,
) -> tuple[list[dict], int]:
    """Returns (candidates_list, tokens). Combines extraction and classification in one LLM call."""
    conversation = "\n".join(f"{m['role']}: {m['content']}" for m in messages if m.get("content"))
    prompt = _fmt(
        _load_prompt("extract_classify"),
        messages=conversation,
        extraction_instructions=agent.extraction_instructions,
    )
    try:
        raw, usage = await _llm_call(prompt, "extract_classify")
        parsed = _parse_json(raw)
        return parsed.get("candidates", []), usage["total_tokens"]
    except Exception:
        return [], 0


async def _dedup_decide_step(
    candidates: list[dict],
    existing_memories: list[dict],
    agent: Agent,
) -> tuple[list[dict], int]:
    """Returns (decisions_list, tokens). Combines dedup and decide in one LLM call."""
    if existing_memories:
        existing_str = "\n".join(
            f"[{i+1}] ID={m['memory_id']}: {m['content']}"
            for i, m in enumerate(existing_memories)
        )
    else:
        existing_str = "None"

    candidates_str = "\n".join(
        f"[{i}] {c['content']} (type={c['memory_type']}, score={c['importance_score']})"
        for i, c in enumerate(candidates)
    )

    prompt = _fmt(
        _load_prompt("dedup_decide"),
        candidates=candidates_str,
        existing_memories=existing_str,
        extraction_instructions=agent.extraction_instructions,
    )
    try:
        raw, usage = await _llm_call(prompt, "dedup_decide")
        parsed = _parse_json(raw)
        return parsed.get("decisions", []), usage["total_tokens"]
    except Exception:
        return [], 0


async def _fetch_existing_memories(
    agent: Agent,
    user_id: str,
    db: AsyncSession,
    qdrant_client: Any,
    limit: int = 50,
) -> list[dict]:
    """Fetch all non-deleted memories for agent+user via Qdrant scroll, then hydrate from Postgres."""
    try:
        loop = asyncio.get_running_loop()
        scroll_result = await loop.run_in_executor(
            None,
            lambda: qdrant_client.scroll(
                collection_name=settings.qdrant_collection,
                scroll_filter=Filter(must=[
                    FieldCondition(key="agent_id", match=MatchValue(value=str(agent.id))),
                    FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                    FieldCondition(key="deleted", match=MatchValue(value=False)),
                ]),
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
        )
        points = scroll_result[0]

        mem_ids = []
        for point in points:
            mem_id_str = point.payload.get("memory_id")
            if not mem_id_str:
                continue
            try:
                mem_ids.append(uuid.UUID(mem_id_str))
            except ValueError:
                continue

        if not mem_ids:
            return []

        result = await db.execute(
            select(Memory).where(
                Memory.id.in_(mem_ids),
                Memory.deleted_at.is_(None),
            )
        )
        memories_db = result.scalars().all()

        return [
            {
                "memory_id": str(m.id),
                "memory_type": m.memory_type,
                "content": m.content,
            }
            for m in memories_db
        ]
    except Exception:
        return []


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

    # Flush so the new memory row exists before we reference it via FK
    await db.flush()

    # Supersede old memory if this is an update (must happen after flush)
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
    Run the 2-step pipeline: extract+classify → dedup+decide.
    Returns (persisted_memories, persisted_candidates, rejected_candidates).
    """
    all_candidates: list[MemoryCandidate] = []
    persisted_memories: list[Memory] = []
    persisted_candidates: list[MemoryCandidate] = []
    rejected_candidates: list[MemoryCandidate] = []

    # ── Step 1: Extract + Classify ───────────────────────────────────────────
    t0 = time.monotonic()
    raw_candidates, tokens_step1 = await _extract_classify_step(messages, agent)
    extract_classify_ms = int((time.monotonic() - t0) * 1000)

    for raw in raw_candidates:
        content = raw.get("content", "").strip()
        if not content:
            continue
        importance = float(raw.get("importance_score", 0.0))
        if importance < 0.3:
            continue

        candidate = MemoryCandidate(
            trace_id=trace.id,
            content=content,
            memory_type=raw.get("memory_type", "fact"),
            importance_score=importance,
            llm_responses={
                "extract_classify": {
                    "rationale": raw.get("rationale", ""),
                    "memory_type": raw.get("memory_type"),
                    "importance_score": importance,
                    "classify_reasoning": raw.get("classify_reasoning", ""),
                    "is_structured": raw.get("is_structured", False),
                    "relation_label": raw.get("relation_label"),
                    "object_value": raw.get("object_value"),
                }
            },
        )
        db.add(candidate)
        setattr(candidate, '_is_structured', raw.get("is_structured", False))
        setattr(candidate, '_relation_label', raw.get("relation_label"))
        setattr(candidate, '_object_value', raw.get("object_value"))
        all_candidates.append(candidate)

    # Early exit if nothing passed the importance filter
    if not all_candidates:
        trace.status = "completed"
        trace.completed_at = utcnow()
        trace.llm_calls_total = 1
        trace.tokens_used = tokens_step1
        trace.pipeline_timing = {"extract_classify_ms": extract_classify_ms, "total_ms": extract_classify_ms}
        await db.commit()
        return persisted_memories, persisted_candidates, rejected_candidates

    # ── Step 2: Fetch existing memories ─────────────────────────────────────
    await db.flush()
    t1 = time.monotonic()
    existing_memories = await _fetch_existing_memories(agent, user_id, db, qdrant_client, limit=50)
    fetch_memories_ms = int((time.monotonic() - t1) * 1000)

    # ── Step 3: Dedup + Decide ───────────────────────────────────────────────
    t2 = time.monotonic()
    candidates_dicts = [
        {
            "content": c.content,
            "memory_type": c.memory_type or "fact",
            "importance_score": c.importance_score or 0.5,
        }
        for c in all_candidates
    ]

    decisions, tokens_step3 = await _dedup_decide_step(candidates_dicts, existing_memories, agent)
    dedup_decide_ms = int((time.monotonic() - t2) * 1000)

    # Build index → decision map; default to persist if decisions are missing
    decision_map: dict[int, dict] = {d["candidate_index"]: d for d in decisions}

    t3 = time.monotonic()
    for idx, candidate in enumerate(all_candidates):
        decision = decision_map.get(idx)

        if decision is None:
            # Fallback: persist if score qualifies
            action = "persist"
            dedup_decision = "new"
            dedup_target_id_str = None
            rejection_reason = None
        else:
            action = decision.get("action", "persist")
            supersedes_id = decision.get("supersedes_id")
            rejection_category = decision.get("rejection_category")
            rejection_reason_raw = decision.get("rejection_reason")

            if action == "persist":
                dedup_decision = "new"
                dedup_target_id_str = None
                rejection_reason = None
            elif action == "update":
                dedup_decision = "update"
                dedup_target_id_str = supersedes_id
                rejection_reason = None
            elif action == "duplicate":
                dedup_decision = "duplicate"
                dedup_target_id_str = None
                rejection_reason = "duplicate"
            else:  # reject
                dedup_decision = "new"
                dedup_target_id_str = None
                rejection_reason = rejection_category or rejection_reason_raw or "rejected"

        candidate.dedup_decision = dedup_decision
        candidate.dedup_target_id = None
        if dedup_target_id_str:
            try:
                candidate.dedup_target_id = uuid.UUID(dedup_target_id_str)
            except ValueError:
                pass

        candidate.llm_responses = {
            **candidate.llm_responses,
            "dedup_decide": decision or {"action": "persist", "fallback": True},
        }

        if action in ("persist", "update"):
            candidate.final_decision = "persist"
            candidate.rejection_reason = None
            try:
                memory = await _persist_memory(
                    candidate, agent, user_id, session_id,
                    dedup_decision, dedup_target_id_str,
                    extra_metadata, db, qdrant_client,
                )
                persisted_memories.append(memory)
                persisted_candidates.append(candidate)
            except Exception as e:
                candidate.final_decision = "reject"
                candidate.rejection_reason = f"persist_error: {e}"
                rejected_candidates.append(candidate)
        else:
            candidate.final_decision = "reject"
            candidate.rejection_reason = rejection_reason
            rejected_candidates.append(candidate)

    persist_ms = int((time.monotonic() - t3) * 1000)

    # ── Step 4: Finalize trace ───────────────────────────────────────────────
    trace.status = "completed"
    trace.completed_at = utcnow()
    trace.llm_calls_total = 2
    trace.tokens_used = tokens_step1 + tokens_step3
    trace.pipeline_timing = {
        "extract_classify_ms": extract_classify_ms,
        "fetch_memories_ms": fetch_memories_ms,
        "dedup_decide_ms": dedup_decide_ms,
        "persist_ms": persist_ms,
        "total_ms": extract_classify_ms + fetch_memories_ms + dedup_decide_ms + persist_ms,
    }

    await db.commit()

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
        score_threshold=0.3,
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
