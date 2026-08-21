from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central application configuration.

    Values are loaded from environment variables / .env.
    """

    app_name: str = Field(
        default="Voice RAG",
        validation_alias="APP_NAME",
    )

    app_env: str = Field(
        default="development",
        validation_alias="APP_ENV",
    )

    debug: bool = Field(
        default=False,
        validation_alias="DEBUG",
    )

    groq_api_key: str = Field(
        validation_alias="GROQ_API_KEY",
    )

    groq_model: str = Field(
        default="grok-4.5",
        validation_alias="GROQ_MODEL",
    )


    sarvam_api_key: str = Field(
        validation_alias="SARVAM_API_KEY",
    )

    sarvam_stt_model: str = Field(
        default="saaras:v3",
        validation_alias="SARVAM_STT_MODEL",
    )


    qdrant_url: str = Field(
        default="http://localhost:6333",
        validation_alias="QDRANT_URL",
    )

    qdrant_api_key: str | None = Field(
        default=None,
        validation_alias="QDRANT_API_KEY",
    )

    qdrant_collection: str = Field(
        default="msmarco_xi",
        validation_alias="QDRANT_COLLECTION",
    )


    embedding_model: str = Field(
        default=(
            "sentence-transformers/"
            "paraphrase-multilingual-mpnet-base-v2"
        ),
        validation_alias="EMBEDDING_MODEL",
    )

    embedding_device: str = Field(
        default="cpu",
        validation_alias="EMBEDDING_DEVICE",
    )


    dense_top_k: int = Field(
        default=20,
        validation_alias="DENSE_TOP_K",
    )

    sparse_top_k: int = Field(
        default=20,
        validation_alias="SPARSE_TOP_K",
    )

    rrf_k: int = Field(
        default=60,
        validation_alias="RRF_K",
    )

    rerank_top_k: int = Field(
        default=5,
        validation_alias="RERANK_TOP_K",
    )

    min_retrieval_score: float = Field(
        default=0.20,
        validation_alias="MIN_RETRIEVAL_SCORE",
    )

    max_context_chunks: int = Field(
        default=5,
        validation_alias="MAX_CONTEXT_CHUNKS",
    )

    temperature: float = Field(
        default=0.0,
        validation_alias="TEMPERATURE",
    )



    host: str = Field(
        default="0.0.0.0",
        validation_alias="HOST",
    )

    port: int = Field(
        default=8000,
        validation_alias="PORT",
    )


    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Cached settings instance.

    This prevents repeatedly parsing environment variables
    throughout the application.
    """

    return Settings()