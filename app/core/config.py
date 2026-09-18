from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "local"
    auth_enabled: bool = True
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/rag_center"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    langfuse_enabled: bool = False
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    model_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model_api_key: str = ""
    embedding_model: str = "qwen3.7-text-embedding"
    embedding_dimensions: int = 1536
    embedding_batch_size: int = Field(
        default=20,
        ge=1,
        le=20,
        description="Maximum number of document chunks sent per embedding request.",
    )

    llm_provider: str = "openai_compatible"
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    llm_timeout_seconds: int = 60
    query_rewrite_enabled: bool = False
    query_rewrite_timeout_ms: int = 2000

    vector_store: str = "pgvector"
    keyword_search_provider: str = "elasticsearch"
    elasticsearch_url: str = "http://localhost:19200"
    elasticsearch_index: str = "rag_chunks"
    retrieval_mode: str = "vector"
    hybrid_fusion: str = "rrf"
    hybrid_rrf_k: int = 60
    hybrid_vector_top_k: int = 20
    hybrid_bm25_top_k: int = 20
    hybrid_top_n: int = 20
    chunk_size: int = 800
    chunk_overlap: int = 100
    table_max_rows_per_chunk: int = Field(
        default=10,
        ge=1,
        description="Maximum number of table data rows stored in one chunk.",
    )
    top_k: int = 5

    rerank_enabled: bool = False
    rerank_provider: str = "llm"
    rerank_top_n: int = 5
    rerank_max_candidates: int = 20
    rerank_chunk_max_chars: int = 1000
    rerank_temperature: float = 0.0

    log_level: str = "INFO"
    log_dir: str = "logs"
    log_max_bytes: int = 10 * 1024 * 1024
    log_backup_count: int = 5
    log_rotation_when: str = "midnight"
    log_rotation_interval: int = 1
    log_console_color: bool = True
    log_payload_max_length: int = 2000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
