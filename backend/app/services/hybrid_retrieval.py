from __future__ import annotations

from app.services.retrieval import (
    get_retrieval_service,
)
import time
from app.services.bm25 import (
    get_bm25_service,
)


# ============================================================
# CONFIG
# ============================================================

DENSE_TOP_K = 10
BM25_TOP_K = 10

FINAL_TOP_K = 5

RRF_K = 60


# ============================================================
# RRF
# ============================================================

def reciprocal_rank_fusion(
    dense_results: list[dict],
    bm25_results: list[dict],
    top_k: int = FINAL_TOP_K,
) -> list[dict]:
    """
    Reciprocal Rank Fusion.

    RRF score:

        1 / (k + rank)

    A document appearing in both retrieval systems receives
    a higher combined score.
    """

    scores: dict[str, float] = {}

    documents: dict[str, dict] = {}

    # --------------------------------------------------------
    # Dense results
    # --------------------------------------------------------

    for rank, result in enumerate(
        dense_results,
        start=1,
    ):

        chunk_id = result.get(
            "chunk_id"
        )

        if not chunk_id:
            continue

        scores[chunk_id] = (
            scores.get(chunk_id, 0.0)
            +
            1.0 / (
                RRF_K + rank
            )
        )

        documents[chunk_id] = result

    # --------------------------------------------------------
    # BM25 results
    # --------------------------------------------------------

    for rank, result in enumerate(
        bm25_results,
        start=1,
    ):

        chunk_id = result.get(
            "chunk_id"
        )

        if not chunk_id:
            continue

        scores[chunk_id] = (
            scores.get(chunk_id, 0.0)
            +
            1.0 / (
                RRF_K + rank
            )
        )

        # Prefer BM25's actual passage text when this
        # document wasn't already returned by Qdrant.
        if chunk_id not in documents:

            documents[chunk_id] = result

    # --------------------------------------------------------
    # Sort by fused score
    # --------------------------------------------------------

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    results = []

    for chunk_id, rrf_score in ranked:

        document = dict(
            documents[chunk_id]
        )

        document["rrf_score"] = (
            rrf_score
        )

        results.append(
            document
        )

        if len(results) >= top_k:
            break

    return results


# ============================================================
# HYBRID RETRIEVAL SERVICE
# ============================================================

class HybridRetrievalService:

    def __init__(self):

        self.dense = (
            get_retrieval_service()
        )

        self.bm25 = (
            get_bm25_service()
        )

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
    self,
    query: str,
    top_k: int = FINAL_TOP_K,
    language: str | None = None,
) -> list[dict]:

        if not query or not query.strip():
            return []

    # ========================================================
    # DENSE
    # ========================================================

        dense_start = time.perf_counter()

        dense_results = self.dense.search(
        query=query,
        top_k=DENSE_TOP_K,
        language=language,
        )

        dense_ms = (
        time.perf_counter()
        - dense_start
        ) * 1000

    # ========================================================
    # BM25
    # ========================================================

        bm25_start = time.perf_counter()

        bm25_results = self.bm25.search(
        query=query,
        top_k=BM25_TOP_K,
        language=language,
    )

        bm25_ms = (
        time.perf_counter()
        - bm25_start
    ) * 1000

    # ========================================================
    # RRF
    # ========================================================

        rrf_start = time.perf_counter()

        final_results = reciprocal_rank_fusion(
        dense_results=dense_results,
        bm25_results=bm25_results,
        top_k=top_k,
    )

        rrf_ms = (
        time.perf_counter()
        - rrf_start
    ) * 1000

        print(
        f"[RETRIEVAL] "
        f"dense={dense_ms:.2f}ms | "
        f"bm25={bm25_ms:.2f}ms | "
        f"rrf={rrf_ms:.2f}ms"
    )

        return final_results

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):

        self.dense.close()

        self.bm25.close()


# ============================================================
# SINGLETON
# ============================================================

_hybrid_service: (
    HybridRetrievalService | None
) = None


def get_hybrid_retrieval_service():

    global _hybrid_service

    if _hybrid_service is None:

        _hybrid_service = (
            HybridRetrievalService()
        )

    return _hybrid_service