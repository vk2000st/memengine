# MemEngine Python SDK

Python client for the [MemEngine](https://github.com/vk2000st/memengine) AI memory API.

## Installation

```bash
pip install git+https://github.com/vk2000st/memengine.git#subdirectory=sdk
```

> **PyPI package coming soon.** Install directly from GitHub for now.

For local development from a clone:

```bash
pip install -e ./sdk                # from the repo root
```

## Quick Start

```python
from memengine import MemEngine

# 1. Register a new company (first time only)
client = MemEngine("http://localhost:8000")
api_key = client.register("Acme Corp")
print(f"Save this key: {api_key}")

# 2. Re-use a saved key
client = MemEngine("http://localhost:8000", api_key="mem_xxx...")

# 3. Create an agent
client.create_agent(
    slug="support-bot",
    name="Support Bot",
    extraction_instructions=(
        "Extract facts about the user's product issues, preferences, "
        "and account details. Ignore pleasantries."
    ),
)

# 4. Add memories from a conversation
result = client.add(
    slug="support-bot",
    user_id="user-42",
    messages=[
        {"role": "user", "content": "I'm on the Pro plan and prefer email notifications."},
        {"role": "assistant", "content": "Got it, I'll remember that."},
    ],
)
print(f"Persisted {len(result.persisted)} memories (trace: {result.trace_id})")

# 5. Search memories
resp = client.search(slug="support-bot", user_id="user-42", query="notification preferences")
for r in resp.results:
    print(f"  [{r.similarity_score:.2f}] {r.content}")
```

## API Reference

### `MemEngine(base_url, api_key=None, timeout=30.0)`

Constructor. `api_key` is optional only if you call `register()` next.

Use as a context manager to ensure the HTTP connection is closed:

```python
with MemEngine("http://localhost:8000", api_key="mem_xxx...") as client:
    result = client.add(...)
```

---

### `register(company_name) -> str`

Create a new tenant company. Sets `api_key` on the client and returns it.
**The key is shown only once — store it securely.**

```python
client = MemEngine("http://localhost:8000")
api_key = client.register("My Company")
# persist api_key somewhere safe
```

---

### `add(slug, user_id, messages, session_id=None, metadata=None) -> AddResult`

Run the extraction pipeline on a conversation. Returns an `AddResult`:

| Field | Type | Description |
|---|---|---|
| `trace_id` | `str` | UUID for the pipeline run |
| `persisted` | `list[Memory]` | Memories saved to the store |
| `rejected` | `list[Candidate]` | Candidates the pipeline dropped |
| `candidates` | `list[Candidate]` | All candidates (persisted + rejected) |

```python
result = client.add(
    slug="my-agent",
    user_id="user-123",
    messages=[
        {"role": "user", "content": "I hate spicy food and live in Berlin."},
    ],
    session_id="chat-session-1",        # optional
    metadata={"source": "chat"},        # optional
)

for memory in result.persisted:
    print(memory.content, memory.memory_type, memory.importance_score)
```

---

### `search(slug, user_id, query, limit=10, session_id=None) -> SearchResponse`

Semantic search over stored memories. Returns a `SearchResponse`:

| Field | Type | Description |
|---|---|---|
| `results` | `list[SearchResult]` | Ranked results |
| `rewritten_query` | `str` | Query rewritten for better recall |

Each `SearchResult` extends `Memory` with `similarity_score` and `retrieval_reason`.

```python
resp = client.search(
    slug="my-agent",
    user_id="user-123",
    query="dietary restrictions",
    limit=5,
)
print("Rewritten:", resp.rewritten_query)
for r in resp.results:
    print(f"  [{r.similarity_score:.3f}] {r.content}  — {r.retrieval_reason}")
```

---

### `list_memories(slug, user_id, limit=50, session_id=None) -> list[SearchResult]`

Approximate listing of all memories for a user. Internally issues a broad
semantic search — increase `limit` (max 50) to retrieve more results.

```python
memories = client.list_memories(slug="my-agent", user_id="user-123")
for m in memories:
    print(m.content)
```

> **Note:** The API exposes memories via search only. A dedicated list endpoint
> is planned for a future release.

---

### `get_memory(memory_id) -> list[AuditEntry]`

Return the full lifecycle audit trail for a memory (created, superseded, deleted).

```python
trail = client.get_memory("8f03e3a2-...")
for entry in trail:
    print(entry.action, entry.actor, entry.created_at)
```

---

### `delete_memory(memory_id) -> None`

Soft-delete a memory by ID.

```python
client.delete_memory("8f03e3a2-...")
```

---

### `get_trace(trace_id) -> Trace`

Fetch the full pipeline trace for an `add()` call.

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Trace UUID |
| `status` | `str` | `completed` or `failed` |
| `llm_calls_total` | `int` | Number of LLM calls made |
| `tokens_used` | `int` | Total tokens consumed |
| `candidates` | `list[Candidate]` | All extraction candidates |

```python
trace = client.get_trace(result.trace_id)
print(f"Status: {trace.status} | LLM calls: {trace.llm_calls_total} | Tokens: {trace.tokens_used}")
```

---

### `list_rejected(trace_id) -> list[Candidate]`

Return candidates that were rejected during a pipeline run. Each `Candidate`
has a `rejection_reason` explaining why it was dropped.

```python
rejected = client.list_rejected(result.trace_id)
for c in rejected:
    print(f"  REJECTED: {c.content!r}  — {c.rejection_reason}")
```

---

## Error Handling

All methods raise a subclass of `MemEngineError` on failure:

| Exception | HTTP status | When |
|---|---|---|
| `AuthError` | 401 | Invalid or missing API key |
| `NotFoundError` | 404 | Memory, agent, or trace not found |
| `ConflictError` | 409 | Duplicate agent slug |
| `ValidationError` | 422 | Invalid request payload |
| `MemEngineError` | any | Unexpected server error |

```python
from memengine import MemEngine, NotFoundError, AuthError

try:
    trace = client.get_trace("nonexistent-id")
except NotFoundError as e:
    print(f"Not found: {e.detail}")
except AuthError as e:
    print(f"Auth failed — check your API key")
```

---

## Full Example

```python
from memengine import MemEngine

BASE_URL = "http://localhost:8000"
API_KEY = "mem_xxx..."   # from register()

client = MemEngine(BASE_URL, api_key=API_KEY)

# Set up an agent
client.create_agent(
    slug="assistant",
    name="Personal Assistant",
    extraction_instructions=(
        "Extract long-term facts about the user: preferences, goals, "
        "relationships, constraints, and skills. Skip small talk."
    ),
)

# Ingest a conversation
result = client.add(
    slug="assistant",
    user_id="alice",
    messages=[
        {"role": "user", "content": "I'm training for a marathon and I'm vegetarian."},
        {"role": "assistant", "content": "I'll keep that in mind!"},
    ],
)
print(f"Persisted {len(result.persisted)} memories")

# Retrieve relevant context before the next conversation
resp = client.search(slug="assistant", user_id="alice", query="health and diet")
context = "\n".join(f"- {r.content}" for r in resp.results)
print("User context:\n", context)

# Inspect the pipeline trace
trace = client.get_trace(result.trace_id)
print(f"Pipeline used {trace.llm_calls_total} LLM calls and {trace.tokens_used} tokens")

# See what was rejected and why
for c in client.list_rejected(result.trace_id):
    print(f"  Dropped: {c.content!r}  ({c.rejection_reason})")
```
