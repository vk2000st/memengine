from __future__ import annotations

from typing import Any

import httpx

from .exceptions import AuthError, ConflictError, MemEngineError, NotFoundError, ValidationError
from .models import AddResult, Agent, AuditEntry, Candidate, Memory, SearchResponse, SearchResult, Trace


class MemEngine:
    """
    Synchronous client for the MemEngine API.

    Usage::

        # First-time setup: register a company
        client = MemEngine("http://localhost:8000")
        api_key = client.register("Acme Corp")

        # Regular use with a saved key
        client = MemEngine("http://localhost:8000", api_key="mem_xxx...")
    """

    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 120.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._http = httpx.Client(base_url=self._base_url, timeout=timeout)

    # ── Internal helpers ────────────────────────────────────────────────────

    def _auth_headers(self) -> dict[str, str]:
        if not self._api_key:
            raise AuthError(401, "No API key set. Call register() first or pass api_key to the constructor.")
        return {"X-API-Key": self._api_key}

    def _raise(self, response: httpx.Response) -> None:
        if response.is_success:
            return
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        code = response.status_code
        if code == 401:
            raise AuthError(code, detail)
        if code == 404:
            raise NotFoundError(code, detail)
        if code == 409:
            raise ConflictError(code, detail)
        if code == 422:
            raise ValidationError(code, str(detail))
        raise MemEngineError(code, detail)

    # ── Company / auth ──────────────────────────────────────────────────────

    def register(self, company_name: str) -> str:
        """
        Create a new tenant company and set the API key on this client.

        Returns the API key — store it securely; it is shown only once.
        """
        response = self._http.post("/companies", json={"name": company_name})
        self._raise(response)
        self._api_key = response.json()["api_key"]
        return self._api_key

    # ── Agent management ────────────────────────────────────────────────────

    def create_agent(
        self,
        slug: str,
        name: str,
        extraction_instructions: str,
        config: dict[str, Any] | None = None,
    ) -> Agent:
        """Create a new agent under the authenticated company."""
        payload: dict[str, Any] = {
            "slug": slug,
            "name": name,
            "extraction_instructions": extraction_instructions,
        }
        if config:
            payload["config"] = config
        response = self._http.post("/agents", json=payload, headers=self._auth_headers())
        self._raise(response)
        return _parse_agent(response.json())

    def list_agents(self) -> list[Agent]:
        """List all active agents for the authenticated company."""
        response = self._http.get("/agents", headers=self._auth_headers())
        self._raise(response)
        return [_parse_agent(a) for a in response.json()]

    def get_agent(self, slug: str) -> Agent:
        """Fetch a single agent by slug."""
        response = self._http.get(f"/agents/{slug}", headers=self._auth_headers())
        self._raise(response)
        return _parse_agent(response.json())

    # ── Memory ──────────────────────────────────────────────────────────────

    def add(
        self,
        agent_slug: str,
        user_id: str,
        messages: list[dict[str, str]],
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AddResult:
        """
        Run the extraction pipeline on a conversation and persist memories.

        :param agent_slug: Slug of the agent to use for extraction.
        :param user_id: Stable identifier for the end user.
        :param messages: List of ``{"role": "user"|"assistant", "content": "..."}`` dicts.
        :param session_id: Optional session grouping key.
        :param metadata: Arbitrary key/value pairs attached to persisted memories.
        :returns: :class:`AddResult` with ``persisted``, ``rejected``, ``candidates``, and ``trace_id``.
        """
        payload: dict[str, Any] = {"agent_slug": agent_slug, "user_id": user_id, "messages": messages}
        if session_id is not None:
            payload["session_id"] = session_id
        if metadata:
            payload["metadata"] = metadata

        response = self._http.post("/memory/add", json=payload, headers=self._auth_headers())
        self._raise(response)
        data = response.json()
        return AddResult(
            trace_id=data["trace_id"],
            persisted=[_parse_memory(m) for m in data["persisted"]],
            rejected=[_parse_candidate(c) for c in data["rejected"]],
            candidates=[_parse_candidate(c) for c in data["candidates"]],
        )

    def search(
        self,
        agent_slug: str,
        user_id: str,
        query: str,
        limit: int = 10,
        session_id: str | None = None,
    ) -> SearchResponse:
        """
        Search memories for a user using semantic similarity.

        :returns: :class:`SearchResponse` with ``results`` (ranked) and ``rewritten_query``.
        """
        payload: dict[str, Any] = {"agent_slug": agent_slug, "user_id": user_id, "query": query, "limit": limit}
        if session_id is not None:
            payload["session_id"] = session_id

        response = self._http.post("/memory/search", json=payload, headers=self._auth_headers())
        self._raise(response)
        data = response.json()
        return SearchResponse(
            results=[_parse_search_result(r) for r in data["results"]],
            rewritten_query=data["rewritten_query"],
        )

    def list_memories(
        self,
        agent_slug: str,
        user_id: str,
        limit: int = 50,
        session_id: str | None = None,
    ) -> list[SearchResult]:
        """
        Return the most relevant stored memories for a user.

        The API exposes memories via semantic search only (no dedicated list endpoint).
        This method issues a broad recall query to approximate a full listing.
        Increase ``limit`` (max 50) to retrieve more results.
        """
        resp = self.search(
            agent_slug=agent_slug,
            user_id=user_id,
            query="user information preferences goals habits facts",
            limit=limit,
            session_id=session_id,
        )
        return resp.results

    def get_memory(self, memory_id: str) -> list[AuditEntry]:
        """
        Return the full audit trail for a memory (created, updated, deleted events).

        The API does not expose a GET /memory/{id} endpoint; the audit log
        is the authoritative record of a memory's lifecycle.
        """
        response = self._http.get(f"/memory/{memory_id}/audit", headers=self._auth_headers())
        self._raise(response)
        return [_parse_audit_entry(e) for e in response.json()]

    def delete_memory(self, memory_id: str) -> None:
        """Soft-delete a memory by ID."""
        response = self._http.delete(f"/memory/{memory_id}", headers=self._auth_headers())
        self._raise(response)

    # ── Observability ───────────────────────────────────────────────────────

    def get_trace(self, trace_id: str) -> Trace:
        """Return the full pipeline trace for an :meth:`add` call."""
        response = self._http.get(f"/trace/{trace_id}", headers=self._auth_headers())
        self._raise(response)
        return _parse_trace(response.json())

    def list_rejected(self, trace_id: str) -> list[Candidate]:
        """
        Return all candidates that were rejected during a pipeline run.

        Fetches the trace and filters for ``final_decision == "reject"``.
        """
        trace = self.get_trace(trace_id)
        return [c for c in trace.candidates if c.final_decision == "reject"]

    # ── Misc ────────────────────────────────────────────────────────────────

    def health(self) -> dict[str, str]:
        """Check API health. Does not require an API key."""
        response = self._http.get("/health")
        self._raise(response)
        return response.json()

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._http.close()

    def __enter__(self) -> "MemEngine":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ── Parsers ─────────────────────────────────────────────────────────────────

def _parse_memory(d: dict[str, Any]) -> Memory:
    return Memory(
        id=d["id"],
        agent_id=d["agent_id"],
        user_id=d["user_id"],
        session_id=d.get("session_id"),
        content=d["content"],
        memory_type=d["memory_type"],
        importance_score=d["importance_score"],
        metadata=d.get("metadata", {}),
        created_at=d["created_at"],
        updated_at=d["updated_at"],
    )


def _parse_candidate(d: dict[str, Any]) -> Candidate:
    return Candidate(
        id=d["id"],
        content=d["content"],
        memory_type=d.get("memory_type"),
        importance_score=d.get("importance_score"),
        dedup_decision=d.get("dedup_decision"),
        dedup_target_id=d.get("dedup_target_id"),
        final_decision=d.get("final_decision"),
        rejection_reason=d.get("rejection_reason"),
        memory_id=d.get("memory_id"),
        llm_responses=d.get("llm_responses", {}),
    )


def _parse_search_result(d: dict[str, Any]) -> SearchResult:
    return SearchResult(
        id=d["id"],
        agent_id=d["agent_id"],
        user_id=d["user_id"],
        session_id=d.get("session_id"),
        content=d["content"],
        memory_type=d["memory_type"],
        importance_score=d["importance_score"],
        similarity_score=d["similarity_score"],
        retrieval_reason=d["retrieval_reason"],
        metadata=d.get("metadata", {}),
        created_at=d["created_at"],
        updated_at=d["updated_at"],
    )


def _parse_trace(d: dict[str, Any]) -> Trace:
    return Trace(
        id=d["id"],
        agent_id=d["agent_id"],
        user_id=d["user_id"],
        session_id=d.get("session_id"),
        status=d["status"],
        error=d.get("error"),
        llm_calls_total=d["llm_calls_total"],
        tokens_used=d["tokens_used"],
        created_at=d["created_at"],
        completed_at=d.get("completed_at"),
        candidates=[_parse_candidate(c) for c in d.get("candidates", [])],
    )


def _parse_audit_entry(d: dict[str, Any]) -> AuditEntry:
    return AuditEntry(
        id=d["id"],
        entity_type=d["entity_type"],
        entity_id=d["entity_id"],
        action=d["action"],
        actor=d["actor"],
        details=d.get("details", {}),
        created_at=d["created_at"],
    )


def _parse_agent(d: dict[str, Any]) -> Agent:
    return Agent(
        id=d["id"],
        company_id=d["company_id"],
        slug=d["slug"],
        name=d["name"],
        extraction_instructions=d["extraction_instructions"],
        config=d.get("config", {}),
        is_active=d["is_active"],
        created_at=d["created_at"],
        updated_at=d["updated_at"],
    )
