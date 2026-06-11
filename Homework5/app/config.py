from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central app configuration.

    Values are loaded from:
    1. environment variables
    2. .env file
    3. defaults below
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "document_chunks"

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Chunking
    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 50

    # LLM / OpenRouter
    # Optional for now, because indexing should work without an LLM key.
    openrouter_api_key: str | None = None
    openrouter_model: str = "openai/gpt-4o-mini"
    redis_url: str = "redis://localhost:6379/0"

    # Semantic cache
    qdrant_cache_collection: str = "semantic_cache"
    semantic_cache_threshold: float = 0.92
    semantic_cache_ttl_seconds: int = 3600

    usage_db_path: str = "data/usage.sqlite3"

    # Langfuse
    langfuse_enabled: bool = True
settings = Settings()