from pydantic import BaseModel, Field


class RAGRequest(BaseModel):
    """
    Input to the RAG pipeline.
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User's natural-language question",
    )

    language: str | None = Field(
        default=None,
        description="Optional language code such as en, hi, mr, ta",
    )


class Source(BaseModel):
    """
    Source returned by the retrieval layer.
    """

    chunk_id: str

    score: float

    text: str

    language: str | None = None


class LatencyMetrics(BaseModel):
    """
    Per-stage latency measurements.
    """

    validation_ms: float = 0.0
    embedding_ms: float = 0.0
    dense_retrieval_ms: float = 0.0
    sparse_retrieval_ms: float = 0.0
    fusion_ms: float = 0.0
    reranking_ms: float = 0.0
    generation_ms: float = 0.0
    guardrail_ms: float = 0.0
    total_ms: float = 0.0


class RAGResponse(BaseModel):
    """
    Final response returned by /rag.
    """

    answer: str

    grounded: bool

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    refused: bool = False

    refusal_reason: str | None = None

    sources: list[Source] = Field(
        default_factory=list,
    )

    latency_ms: LatencyMetrics