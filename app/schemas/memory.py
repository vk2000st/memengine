import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ── Auth / Company ──────────────────────────────────────────────────────────

class CompanyCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    email: str | None = Field(None, max_length=255)


class CompanyOut(BaseModel):
    id: uuid.UUID
    name: str
    email: str | None = None
    api_key_prefix: str
    api_key: str | None = None  # populated by /me (decrypted) and on creation
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class CompanyCreated(CompanyOut):
    api_key: str  # non-optional on creation


# ── Agent ───────────────────────────────────────────────────────────────────

class AgentCreate(BaseModel):
    slug: str = Field(..., pattern=r"^[a-z0-9_-]+$", min_length=1, max_length=100)
    name: str = Field(..., min_length=1, max_length=255)
    extraction_instructions: str = Field(..., min_length=10)
    config: dict[str, Any] = Field(default_factory=dict)


class AgentOut(BaseModel):
    id: uuid.UUID
    company_id: uuid.UUID
    slug: str
    name: str
    extraction_instructions: str
    config: dict[str, Any]
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ── Memory ───────────────────────────────────────────────────────────────────

class MemoryOut(BaseModel):
    id: uuid.UUID
    agent_id: uuid.UUID
    user_id: str
    session_id: str | None
    content: str
    memory_type: str
    importance_score: float
    metadata: dict[str, Any] = Field(default_factory=dict, validation_alias="metadata_")
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True, "populate_by_name": True}


class MemorySearchResult(MemoryOut):
    similarity_score: float
    retrieval_reason: str
    agent_slug: str | None = None


# ── Pipeline input ──────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str = Field(..., pattern=r"^(system|user|assistant)$")
    content: str


class MemoryAddRequest(BaseModel):
    agent_slug: str
    user_id: str = Field(..., min_length=1, max_length=255)
    session_id: str | None = Field(None, max_length=255)
    messages: list[Message] = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemorySearchRequest(BaseModel):
    agent_slug: str | None = None
    user_id: str | None = Field(None, max_length=255)
    query: str = Field(..., min_length=1)
    limit: int = Field(10, ge=1, le=50)
    session_id: str | None = None
    metadata_filter: dict[str, Any] = Field(default_factory=dict)


# ── Pipeline trace output ────────────────────────────────────────────────────

class LLMResponses(BaseModel):
    extract: dict[str, Any] | None = None
    classify: dict[str, Any] | None = None
    deduplicate: dict[str, Any] | None = None
    decide: dict[str, Any] | None = None


class CandidateOut(BaseModel):
    id: uuid.UUID
    content: str
    memory_type: str | None
    importance_score: float | None
    dedup_decision: str | None
    dedup_target_id: uuid.UUID | None
    final_decision: str | None
    rejection_reason: str | None
    memory_id: uuid.UUID | None
    llm_responses: dict[str, Any]

    model_config = {"from_attributes": True}


class TraceOut(BaseModel):
    id: uuid.UUID
    agent_id: uuid.UUID
    user_id: str
    session_id: str | None
    status: str
    error: str | None
    llm_calls_total: int
    tokens_used: int
    created_at: datetime
    completed_at: datetime | None
    candidates: list[CandidateOut]
    messages: list[dict] = Field(default_factory=list, validation_alias="input_messages")

    model_config = {"from_attributes": True, "populate_by_name": True}


class MemoryAddAccepted(BaseModel):
    trace_id: uuid.UUID
    status: str = "processing"


class MemoryAddResponse(BaseModel):
    trace_id: uuid.UUID
    persisted: list[MemoryOut]
    rejected: list[CandidateOut]
    candidates: list[CandidateOut]


class MemorySearchResponse(BaseModel):
    results: list[MemorySearchResult]
    rewritten_query: str


class TraceStatusOut(BaseModel):
    trace_id: uuid.UUID
    status: str
    persisted_count: int
    rejected_count: int
    error: str | None


class AuditLogOut(BaseModel):
    id: uuid.UUID
    entity_type: str
    entity_id: uuid.UUID
    action: str
    actor: str
    details: dict[str, Any]
    created_at: datetime


# ── Users ────────────────────────────────────────────────────────────────────

class UserSummary(BaseModel):
    user_id: str
    total_memories: int
    agents: list[str]
    last_memory_at: datetime | None
    session_count: int
    top_category: str | None


class AgentMemoryCount(BaseModel):
    slug: str
    memory_count: int


class CategoryCount(BaseModel):
    category: str
    count: int


class MemoryDetail(BaseModel):
    memory_id: uuid.UUID
    memory_text: str
    category: str
    agent_slug: str
    session_id: str | None
    confirmed_by: int
    status: str  # "active" | "deleted"
    created_at: datetime
    updated_at: datetime
    trace_id: uuid.UUID | None


class TimelineEvent(BaseModel):
    at: datetime
    event_type: str  # "added" | "updated" | "rejected" | "deleted"
    memory_text: str
    agent_slug: str
    session_id: str | None


class UserProfile(BaseModel):
    user_id: str
    first_seen: datetime | None
    last_active: datetime | None
    total_memories: int
    session_count: int
    agents: list[AgentMemoryCount]
    categories: list[CategoryCount]
    memories: list[MemoryDetail]
    rejected_count: int
    timeline: list[TimelineEvent]


# ── Analytics ─────────────────────────────────────────────────────────────────

class RecentTrace(BaseModel):
    trace_id: uuid.UUID
    agent_slug: str
    user_id: str
    candidates_total: int
    persisted_count: int
    rejected_count: int
    created_at: datetime


class AnalyticsOverview(BaseModel):
    total_memories: int
    memories_added: int
    total_searches: int
    recent_traces: list[RecentTrace]

    model_config = {"from_attributes": True}
