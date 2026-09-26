from __future__ import annotations

from pydantic import BaseModel, Field


class RAGRequest(BaseModel):

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
    )

    language: str | None = None

    conversation_id: str = "default"


class Source(BaseModel):

    source_id: int

    document_id: str | None = None

    filename: str | None = None

    page: int | None = None

    section: str | None = None

    text: str

    source_type: str | None = None

    retrieval_backend: str | None = None

    score: float | None = None

    citation: dict | None = None


class LatencyMetrics(BaseModel):

    retrieval_ms: float = 0

    generation_ms: float = 0

    total_ms: float = 0


class RAGResponse(BaseModel):

    success: bool

    answer: str

    grounded: bool

    sources: list[Source]

    latency: LatencyMetrics

    conversation_id: str