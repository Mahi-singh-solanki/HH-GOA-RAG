from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.config import get_settings
from app.services.embeddings import (
    get_embedding_service,
)


class DocumentQdrantService:

    def __init__(self):

        settings = get_settings()

        self.settings = settings

        if settings.qdrant_url.lower() == "local":

            Path(
                settings.qdrant_local_path
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

            self.client = QdrantClient(
                path=settings.qdrant_local_path
            )

        else:

            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key
                or None,
            )

        self.collection = (
            settings.document_qdrant_collection
        )

        self.embedding_service = (
            get_embedding_service()
        )

        self.ensure_collection()

    def ensure_collection(self):

        existing = [
            collection.name
            for collection in (
                self.client
                .get_collections()
                .collections
            )
        ]

        if self.collection in existing:
            return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=self.embedding_service.dimension,
                distance=Distance.COSINE,
            ),
        )

    def index_chunks(
        self,
        chunks: list[dict[str, Any]],
        batch_size: int = 32,
    ) -> int:

        if not chunks:
            return 0

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        vectors = (
            self.embedding_service.embed_texts(
                texts,
                batch_size=batch_size,
            )
        )

        points = []

        for chunk, vector in zip(
            chunks,
            vectors,
        ):

            payload = {
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "filename": chunk["filename"],
                "file_type": chunk["file_type"],
                "page": chunk["page"],
                "section": chunk["section"],
                "text": chunk["text"],
                "source_type": chunk["source_type"],
                "ocr_used": chunk["ocr_used"],
                "chunk_index": chunk["chunk_index"],
                "retrieval_backend": "qdrant",
            }

            points.append(
                PointStruct(
                    id=chunk["chunk_id"],
                    vector=vector,
                    payload=payload,
                )
            )

            if len(points) >= batch_size:

                self.client.upsert(
                    collection_name=self.collection,
                    points=points,
                )

                points = []

        if points:

            self.client.upsert(
                collection_name=self.collection,
                points=points,
            )

        return len(chunks)

    def search(
        self,
        query: str,
        limit: int = 8,
    ) -> list[dict[str, Any]]:

        try:

            query_vector = (
                self.embedding_service
                .embed_text(query)
            )

            response = self.client.query_points(
                collection_name=self.collection,
                query=query_vector,
                limit=limit,
                with_payload=True,
            )

        except Exception:

            return []

        results = []

        for point in response.points:

            payload = point.payload or {}

            results.append(
                {
                    "chunk_id": payload.get(
                        "chunk_id",
                        str(point.id),
                    ),
                    "document_id": payload.get(
                        "document_id"
                    ),
                    "filename": payload.get(
                        "filename"
                    ),
                    "page": payload.get(
                        "page"
                    ),
                    "section": payload.get(
                        "section"
                    ),
                    "text": payload.get(
                        "text",
                        "",
                    ),
                    "source_type": payload.get(
                        "source_type",
                        "qdrant",
                    ),
                    "retrieval_backend": "qdrant",
                    "score": float(
                        point.score or 0
                    ),
                    "ocr_used": payload.get(
                        "ocr_used",
                        False,
                    ),
                }
            )

        return results

    def delete_document(
        self,
        document_id: str,
    ) -> None:

        self.client.delete(
            collection_name=self.collection,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(
                            value=document_id
                        ),
                    )
                ]
            ),
        )


@lru_cache
def get_document_qdrant_service():
    return DocumentQdrantService()