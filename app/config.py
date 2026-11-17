import os
from typing import Optional

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))


class Settings:
    """Application settings loaded from environment variables or safe defaults."""

    # LLM / app
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    TOP_K: int = int(os.getenv("TOP_K", "8"))
    FETCH_K: int = int(os.getenv("FETCH_K", "20"))
    MULTI_QUERY: bool = os.getenv("MULTI_QUERY", "true").lower() in {"1", "true", "yes", "on"}
    STREAMING: bool = os.getenv("STREAMING", "true").lower() in {"1", "true", "yes", "on"}
    MAX_HISTORY_TOKENS: int = int(os.getenv("MAX_HISTORY_TOKENS", "2000"))

    # Redis / caching / session / rate limiting
    REDIS_URL: Optional[str] = os.getenv("REDIS_URL")
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))
    TEMP_STATE_PREFIX: str = os.getenv("TEMP_STATE_PREFIX", "app:state:")
    TEMP_STATE_TTL: int = int(os.getenv("TEMP_STATE_TTL", "3600"))
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    RATE_LIMIT_WINDOW_SECONDS: int = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
    QUEUE_NAME: str = os.getenv("QUEUE_NAME", "app:jobs")

    # File system
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "uploads")
    CHROMA_DIR: str = os.getenv("CHROMA_DIR", "chroma_db")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    MAX_UPLOAD_BYTES: int = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "8"))
    CHROMA_BATCH_SIZE: int = int(os.getenv("CHROMA_BATCH_SIZE", "16"))


settings = Settings()