from __future__ import annotations

import sys
from pathlib import Path

from qdrant_client import QdrantClient

# Allow imports when running from backend/
sys.path.append(
    str(Path(__file__).resolve().parents[2])
)

from app.config import get_settings
from app.services.embeddings import (
    get_embedding_service,
)


class RetrievalService:
    """
    Dense vector retrieval using Qdrant.
    """

    def __init__(self) -> None:

        settings = get_settings()

        self.collection_name = (
            settings.qdrant_collection
        )

        self.top_k = (
            settings.dense_top_k
        )

        self.embedding_service = (
            get_embedding_service()
        )

        # ----------------------------------------------------
        # Local Qdrant
        #
        # Same directory used by build_index.py
        # ----------------------------------------------------

        qdrant_path = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "qdrant"
        )

        self.client = QdrantClient(
            path=str(qdrant_path)
        )

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query: str,
        top_k: int | None = None,
        language: str | None = None,
    ) -> list[dict]:

        if not query or not query.strip():
            return []

        query = query.strip()

        if top_k is None:
            top_k = self.top_k

        # ----------------------------------------------------
        # Query embedding
        # ----------------------------------------------------

        query_vector = (
            self.embedding_service.embed_text(
                query
            )
        )

        # ----------------------------------------------------
        # Optional language filtering
        # ----------------------------------------------------

        query_filter = None

        if language:

            from qdrant_client.models import (
                FieldCondition,
                Filter,
                MatchValue,
            )

            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="language",
                        match=MatchValue(
                            value=language
                        ),
                    )
                ]
            )

        # ----------------------------------------------------
        # Qdrant search
        # ----------------------------------------------------

        results = self.client.query_points(
            collection_name=(
                self.collection_name
            ),

            query=query_vector,

            query_filter=query_filter,

            limit=top_k,

            with_payload=True,
        )

        # ----------------------------------------------------
        # Normalize result
        # ----------------------------------------------------

        documents = []

        for result in results.points:

            payload = (
                result.payload
                or {}
            )

            documents.append(
                {
                    "chunk_id": payload.get(
                        "chunk_id",
                        "",
                    ),

                    "parent_id": payload.get(
                        "parent_id",
                        "",
                    ),

                    "text": payload.get(
                        "text",
                        "",
                    ),

                    "language": payload.get(
                        "language",
                    ),

                    "query": payload.get(
                        "query",
                    ),

                    "answer": payload.get(
                        "answer",
                    ),

                    "strategy": payload.get(
                        "strategy",
                    ),

                    "is_selected": payload.get(
                        "is_selected",
                        False,
                    ),

                    "score": float(
                        result.score
                    ),

                    "metadata": payload.get(
                        "metadata",
                        {},
                    ),
                }
            )

        return documents

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self) -> None:

        self.client.close()


# ============================================================
# SINGLETON
# ============================================================

_retrieval_service: RetrievalService | None = None


def get_retrieval_service() -> RetrievalService:

    global _retrieval_service

    if _retrieval_service is None:

        _retrieval_service = (
            RetrievalService()
        )

    return _retrieval_service