from __future__ import annotations

from functools import lru_cache
from typing import Any

from app.config import get_settings
from app.services.document_qdrant import (
    get_document_qdrant_service,
)
from app.services.pageindex_service import (
    get_pageindex_service,
)


class DocumentRetrievalService:

    def __init__(self):

        self.settings = get_settings()

        self.pageindex = (
            get_pageindex_service()
        )

        self.qdrant = (
            get_document_qdrant_service()
        )

    def retrieve(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
    ) -> list[dict[str, Any]]:

        results = []

        pageindex_results = (
            self.pageindex.retrieve(
                query=query,
                history=history,
                max_sources=(
                    self.settings
                    .pageindex_max_sources
                ),
            )
        )

        results.extend(
            pageindex_results
        )

        qdrant_results = (
            self.qdrant.search(
                query=query,
                limit=(
                    self.settings
                    .dense_top_k
                ),
            )
        )

        results.extend(
            qdrant_results
        )

        results = self._deduplicate(
            results
        )

        results.sort(
            key=lambda item: (
                item.get(
                    "score",
                    0,
                )
            ),
            reverse=True,
        )

        return results[
            : self.settings.max_context_chunks
        ]

    def _deduplicate(
        self,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        output = []

        seen = set()

        for item in results:

            key = (
                item.get(
                    "filename"
                ),
                item.get(
                    "page"
                ),
                item.get(
                    "text",
                    "",
                )[:200],
            )

            if key in seen:
                continue

            seen.add(key)

            output.append(item)

        return output


@lru_cache
def get_document_retrieval_service():

    return DocumentRetrievalService()