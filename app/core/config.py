from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Database
    database_url: str = "postgresql+asyncpg://memengine:memengine@postgres:5432/memengine"

    # Redis
    redis_url: str = "redis://localhost:6379"

    # Qdrant
    qdrant_url: str = "http://qdrant:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "memories"

    # LLM
    litellm_model: str = "openai/gpt-4o-mini"
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # Embeddings
    embedding_model: str = "openai/text-embedding-3-small"
    embedding_dim: int = 1536

    # Security
    secret_key: str = "change-me"
    encryption_key: str = ""

    # App
    app_env: str = "development"
    log_level: str = "INFO"

    # Postmark (for trace reports)
    postmark_server_token: str = ""
    postmark_from: str = ""
    report_email_to: str = "vibhanshu.karn@gmail.com"

    # Playground demo
    demo_api_key: str = ""
    demo_agent_slug: str = "support-bot"

    # Pipeline tunables
    dedup_similarity_threshold: float = 0.85
    dedup_candidate_limit: int = 5
    min_importance_to_persist: float = 0.3


@lru_cache
def get_settings() -> Settings:
    return Settings()
