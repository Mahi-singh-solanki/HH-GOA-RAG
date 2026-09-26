from __future__ import annotations

from pathlib import Path
from typing import Any

from app.services.document_chunker import (
    chunk_processed_document,
)
from app.services.document_qdrant import (
    get_document_qdrant_service,
)
from app.services.pageindex_service import (
    get_pageindex_service,
)


class DocumentIndexer:

    def __init__(self):

        self.pageindex = (
            get_pageindex_service()
        )

        self.qdrant = (
            get_document_qdrant_service()
        )

    def index(
        self,
        file_path: Path,
        processed_document: dict[str, Any],
        original_filename: str,
    ) -> dict[str, Any]:

        file_type = processed_document[
            "file_type"
        ]

        if (
            file_type == "pdf"
            and self.pageindex
            .should_use_for_pdf(
                processed_document
            )
        ):

            try:

                result = (
                    self.pageindex
                    .index_document(
                        file_path,
                        original_filename,
                    )
                )

                return {
                    "status": "indexed",
                    **result,
                }

            except Exception as exc:

                print(
                    "PageIndex failed:",
                    repr(exc),
                )
        
        chunks = (
            chunk_processed_document(
                processed_document
            )
        )

        count = (
            self.qdrant.index_chunks(
                chunks
            )
        )

        return {
            "status": "indexed",
            "index_type": "qdrant",
            "chunks": count,
            "fallback": (
                file_type == "pdf"
            ),
        }


_indexer = None


def get_document_indexer() -> DocumentIndexer:

    global _indexer

    if _indexer is None:
        _indexer = DocumentIndexer()

    return _indexer