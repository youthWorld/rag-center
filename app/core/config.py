from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "local"
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/rag_center"

    model_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model_api_key: str = ""
    embedding_model: str = "qwen3.7-text-embedding"
    embedding_dimensions: int = 1536

    vector_store: str = "pgvector"
    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k: int = 5

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
