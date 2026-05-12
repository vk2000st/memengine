-- Trace reports: flag a pipeline trace for human review
CREATE TABLE IF NOT EXISTS trace_reports (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    trace_id    UUID        NOT NULL REFERENCES pipeline_traces(id) ON DELETE CASCADE,
    company_id  UUID        NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    reason      TEXT        NOT NULL,
    note        TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_trace_reports_trace_id   ON trace_reports (trace_id);
CREATE INDEX IF NOT EXISTS ix_trace_reports_company_id ON trace_reports (company_id);
