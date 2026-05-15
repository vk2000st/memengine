import uuid
import logging
from arq import cron
from arq.connections import RedisSettings
from sqlalchemy import select

from app.core.config import get_settings
from app.models.db import Agent, PipelineTrace, get_session_factory, utcnow
from app.services.extraction.pipeline import run_pipeline

log = logging.getLogger(__name__)
settings = get_settings()

# Qdrant client — reuse same pattern as main.py
from qdrant_client import QdrantClient
_qdrant = QdrantClient(url=settings.qdrant_url)


async def run_pipeline_job(
    ctx: dict,
    trace_id: str,
    agent_id: str,
    messages: list[dict],
    user_id: str,
    session_id: str | None,
    extra_metadata: dict,
) -> None:
    """ARQ job: run the full extraction pipeline."""
    log.info(f"arq_job_start trace_id={trace_id}")
    factory = get_session_factory()
    async with factory() as db:
        trace_row = await db.execute(select(PipelineTrace).where(PipelineTrace.id == uuid.UUID(trace_id)))
        trace = trace_row.scalar_one()
        agent_row = await db.execute(select(Agent).where(Agent.id == uuid.UUID(agent_id)))
        agent = agent_row.scalar_one()
        try:
            await run_pipeline(
                trace=trace,
                messages=messages,
                agent=agent,
                user_id=user_id,
                session_id=session_id,
                extra_metadata=extra_metadata,
                db=db,
                qdrant_client=_qdrant,
            )
            log.info(f"arq_job_complete trace_id={trace_id}")
        except Exception as e:
            trace.status = "failed"
            trace.error = str(e)
            trace.completed_at = utcnow()
            await db.commit()
            log.error(f"arq_job_failed trace_id={trace_id} error={e}")


class WorkerSettings:
    functions = [run_pipeline_job]
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
    max_jobs = 10
    job_timeout = 120
    keep_result = 0
