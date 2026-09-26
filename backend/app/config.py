from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):

    app_name: str = "Intelligent Document Intelligence"
    app_env: str = "development"
    debug: bool = True

    groq_api_key: str
    groq_model: str = "openai/gpt-oss-120b"

    sarvam_api_key: str | None = None
    sarvam_stt_model: str = "saaras:v2.5"

    qdrant_url: str = "local"
    qdrant_api_key: str | None = None
    document_qdrant_collection: str = "document_chunks"

    embedding_model: str = (
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    embedding_device: str = "cpu"

    dense_top_k: int = 8
    pageindex_max_sources: int = 8
    max_context_chunks: int = 8
    min_retrieval_score: float = 0.20

    temperature: float = 0.0

    pageindex_enabled: bool = True

    pageindex_mode: str = "auto"

    pageindex_api_key: str | None = None

    pageindex_index_model: str = "groq/openai/gpt-oss-20b"
    pageindex_chat_model: str = "groq/openai/gpt-oss-120b"

    pageindex_storage_path: str = str(
        BASE_DIR / "data" / "pageindex"
    )

    documents_dir: str = str(
        BASE_DIR / "data" / "documents"
    )

    qdrant_local_path: str = str(
        BASE_DIR / "data" / "qdrant"
    )

    ocr_enabled: bool = True
    ocr_language: str = "eng"

    tesseract_cmd: str | None = None

    max_upload_size_mb: int = 50

    cors_origins: str = (
        "http://localhost:5173,"
        "http://127.0.0.1:5173"
    )

    conversation_max_messages: int = 12

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    


@lru_cache
def get_settings() -> Settings:
    return Settings()