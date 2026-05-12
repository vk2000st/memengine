from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Memory:
    id: str
    agent_id: str
    user_id: str
    content: str
    memory_type: str
    importance_score: float
    metadata: dict[str, Any]
    created_at: str
    updated_at: str
    session_id: str | None = None


@dataclass
class Candidate:
    id: str
    content: str
    llm_responses: dict[str, Any]
    memory_type: str | None = None
    importance_score: float | None = None
    dedup_decision: str | None = None
    dedup_target_id: str | None = None
    final_decision: str | None = None
    rejection_reason: str | None = None
    memory_id: str | None = None


@dataclass
class AddAccepted:
    trace_id: str
    status: str


@dataclass
class TraceStatus:
    trace_id: str
    status: str
    persisted_count: int
    rejected_count: int
    error: str | None


@dataclass
class AddResult:
    trace_id: str
    persisted: list[Memory]
    rejected: list[Candidate]
    candidates: list[Candidate]


@dataclass
class SearchResult:
    id: str
    agent_id: str
    user_id: str
    content: str
    memory_type: str
    importance_score: float
    similarity_score: float
    retrieval_reason: str
    metadata: dict[str, Any]
    created_at: str
    updated_at: str
    session_id: str | None = None


@dataclass
class SearchResponse:
    results: list[SearchResult]
    rewritten_query: str


@dataclass
class Trace:
    id: str
    agent_id: str
    user_id: str
    status: str
    llm_calls_total: int
    tokens_used: int
    candidates: list[Candidate]
    created_at: str
    session_id: str | None = None
    error: str | None = None
    completed_at: str | None = None


@dataclass
class AuditEntry:
    id: str
    entity_type: str
    entity_id: str
    action: str
    actor: str
    details: dict[str, Any]
    created_at: str


@dataclass
class Agent:
    id: str
    company_id: str
    agent_slug: str
    name: str
    extraction_instructions: str
    is_active: bool
    created_at: str
    updated_at: str
