# MemEngine

Production-grade AI memory engine. Extracts, classifies, deduplicates, and persists meaningful memories from conversations, with full LLM-decision tracing and semantic search.

## Architecture

```
POST /memory/add  →  extract → classify → deduplicate → decide → persist
POST /memory/search  →  rewrite query → embed → Qdrant → retrieval reason
GET  /trace/{id}  →  full pipeline trace with every LLM response
GET  /memory/{id}/audit  →  complete audit trail
```

**Stack:** FastAPI · SQLAlchemy async · PostgreSQL · Qdrant · LiteLLM · bcrypt

## Quick Start

```bash
cp .env.example .env
# Set OPENAI_API_KEY (or any LiteLLM-supported provider) in .env

docker compose up -d
```

The API is at `http://localhost:8000`. Docs at `http://localhost:8000/docs`.

## Setup (without Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Start Postgres + Qdrant however you prefer, then:
psql $DATABASE_URL -f migrations/001_initial_schema.sql

cp .env.example .env  # edit with your values
uvicorn app.main:app --reload
```

## Usage

### 1. Create a company (tenant)

```bash
curl -X POST http://localhost:8000/companies \
  -H "Content-Type: application/json" \
  -d '{"name": "Acme Corp"}'
```

Returns an `api_key` — store it, it won't be shown again.

### 2. Create an agent

```bash
curl -X POST http://localhost:8000/agents \
  -H "X-API-Key: <your-api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "slug": "support-bot",
    "name": "Support Bot",
    "extraction_instructions": "Extract user preferences, reported issues, product feedback, and any personal context relevant to customer support. Focus on facts that would help personalize future interactions."
  }'
```

### 3. Add memories from a conversation

```bash
curl -X POST http://localhost:8000/memory/add \
  -H "X-API-Key: <your-api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_slug": "support-bot",
    "user_id": "user_123",
    "session_id": "session_abc",
    "messages": [
      {"role": "user", "content": "I prefer email over phone calls for support"},
      {"role": "assistant", "content": "Got it, we will reach out by email."},
      {"role": "user", "content": "Also I am in the EU so GDPR matters to me"}
    ]
  }'
```

Response includes full trace: candidates, per-candidate LLM decisions, persisted/rejected split.

### 4. Search memories

```bash
curl -X POST http://localhost:8000/memory/search \
  -H "X-API-Key: <your-api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_slug": "support-bot",
    "user_id": "user_123",
    "query": "how does this user prefer to be contacted?"
  }'
```

Each result includes a `retrieval_reason` explaining why it matched.

### 5. Inspect a trace

```bash
curl http://localhost:8000/trace/<trace_id> \
  -H "X-API-Key: <your-api-key>"
```

### 6. Memory audit trail

```bash
curl http://localhost:8000/memory/<memory_id>/audit \
  -H "X-API-Key: <your-api-key>"
```

## Data Model

| Table | Purpose |
|---|---|
| `companies` | Top-level tenant; holds bcrypt-hashed API key |
| `agents` | Named memory agents with plain-English extraction instructions |
| `memories` | Persisted memories (soft delete only via `deleted_at`) |
| `pipeline_traces` | One per `/memory/add` call; tracks status + token usage |
| `memory_candidates` | Per-candidate record with full LLM response at each step |
| `audit_logs` | Immutable log of every create/update/delete/supersede |

## Pipeline Steps

1. **Extract** — LLM reads the conversation + agent instructions, outputs memory candidates
2. **Classify** — Each candidate gets a type (`fact`, `preference`, `goal`, `event`, `relationship`, `skill`, `constraint`) and importance score (0–1)
3. **Deduplicate** — Candidate is embedded, similar memories searched in Qdrant, LLM decides `new / update / duplicate`
4. **Decide** — Final `persist / reject` with reason; updates supersede old memories

## Configuration

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://...` | Async Postgres URL |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |
| `LITELLM_MODEL` | `openai/gpt-4o-mini` | Any LiteLLM model string |
| `EMBEDDING_MODEL` | `openai/text-embedding-3-small` | Embedding model |
| `EMBEDDING_DIM` | `1536` | Must match embedding model output |
| `DEDUP_SIMILARITY_THRESHOLD` | `0.85` | Cosine similarity for dedup search |

Any LiteLLM-supported model works: `anthropic/claude-3-5-haiku-20241022`, `groq/llama3-8b-8192`, etc.
