-- MemEngine initial schema
-- Run with: psql $DATABASE_URL -f migrations/001_initial_schema.sql

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ── Companies ────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS companies (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            VARCHAR(255) NOT NULL,
    api_key_hash    VARCHAR(255) NOT NULL,
    api_key_prefix  VARCHAR(12)  NOT NULL,
    is_active       BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_companies_api_key_prefix ON companies (api_key_prefix);

-- ── Agents ───────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS agents (
    id                       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id               UUID         NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    slug                     VARCHAR(100) NOT NULL,
    name                     VARCHAR(255) NOT NULL,
    extraction_instructions  TEXT         NOT NULL,
    config                   JSONB        NOT NULL DEFAULT '{}',
    is_active                BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at               TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at               TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_agent_company_slug UNIQUE (company_id, slug)
);

CREATE INDEX IF NOT EXISTS ix_agents_company_id ON agents (company_id);

-- ── Memories ─────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS memories (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id       UUID         NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    agent_id         UUID         NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    user_id          VARCHAR(255) NOT NULL,
    session_id       VARCHAR(255),
    content          TEXT         NOT NULL,
    memory_type      VARCHAR(50)  NOT NULL,
    importance_score FLOAT        NOT NULL DEFAULT 0.5,
    metadata         JSONB        NOT NULL DEFAULT '{}',
    qdrant_id        VARCHAR(36),
    superseded_by_id UUID         REFERENCES memories(id),
    deleted_at       TIMESTAMPTZ,
    created_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_memories_agent_user         ON memories (agent_id, user_id);
CREATE INDEX IF NOT EXISTS ix_memories_agent_user_session ON memories (agent_id, user_id, session_id);
CREATE INDEX IF NOT EXISTS ix_memories_deleted_at         ON memories (deleted_at) WHERE deleted_at IS NULL;

-- ── Pipeline Traces ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS pipeline_traces (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id      UUID         NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    agent_id        UUID         NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    user_id         VARCHAR(255) NOT NULL,
    session_id      VARCHAR(255),
    input_messages  JSONB        NOT NULL,
    status          VARCHAR(20)  NOT NULL DEFAULT 'processing',
    error           TEXT,
    llm_calls_total INTEGER      NOT NULL DEFAULT 0,
    tokens_used     INTEGER      NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    completed_at    TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS ix_pipeline_traces_company    ON pipeline_traces (company_id);
CREATE INDEX IF NOT EXISTS ix_pipeline_traces_agent_user ON pipeline_traces (agent_id, user_id);

-- ── Memory Candidates ─────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS memory_candidates (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trace_id         UUID        NOT NULL REFERENCES pipeline_traces(id) ON DELETE CASCADE,
    content          TEXT        NOT NULL,
    memory_type      VARCHAR(50),
    importance_score FLOAT,
    dedup_decision   VARCHAR(20),      -- new | update | duplicate
    dedup_target_id  UUID,             -- existing memory being updated/replaced
    final_decision   VARCHAR(20),      -- persist | reject
    rejection_reason TEXT,
    memory_id        UUID        REFERENCES memories(id),
    llm_responses    JSONB       NOT NULL DEFAULT '{}',
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_memory_candidates_trace_id  ON memory_candidates (trace_id);
CREATE INDEX IF NOT EXISTS ix_memory_candidates_memory_id ON memory_candidates (memory_id);

-- ── Audit Logs ────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS audit_logs (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id  UUID        NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    agent_id    UUID,
    entity_type VARCHAR(50) NOT NULL,
    entity_id   UUID        NOT NULL,
    action      VARCHAR(50) NOT NULL,
    actor       VARCHAR(50) NOT NULL DEFAULT 'system',
    details     JSONB       NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_audit_logs_entity    ON audit_logs (entity_type, entity_id);
CREATE INDEX IF NOT EXISTS ix_audit_logs_company   ON audit_logs (company_id);
CREATE INDEX IF NOT EXISTS ix_audit_logs_created   ON audit_logs (created_at);

-- ── updated_at trigger ───────────────────────────────────────────────────────

CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'companies_updated_at') THEN
        CREATE TRIGGER companies_updated_at BEFORE UPDATE ON companies FOR EACH ROW EXECUTE FUNCTION set_updated_at();
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'agents_updated_at') THEN
        CREATE TRIGGER agents_updated_at BEFORE UPDATE ON agents FOR EACH ROW EXECUTE FUNCTION set_updated_at();
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'memories_updated_at') THEN
        CREATE TRIGGER memories_updated_at BEFORE UPDATE ON memories FOR EACH ROW EXECUTE FUNCTION set_updated_at();
    END IF;
END $$;
