from __future__ import annotations

import time
from typing import Any

from app.services.hybrid_retrieval import (
    get_hybrid_retrieval_service,
)
from app.services.guardrails import (
    get_guardrail_service,
)
from app.services.llm import (
    get_llm_service,
)


# ============================================================
# CONFIG
# ============================================================

RETRIEVAL_TOP_K = 5

MIN_RETRIEVAL_SCORE = 0.0


# ============================================================
# RAG SERVICE
# ============================================================

class RAGService:

    def __init__(self):

        self.retriever = (
            get_hybrid_retrieval_service()
        )

        self.llm = (
            get_llm_service()
        )
        self.guardrails = (
    get_guardrail_service()
)   

    # ========================================================
    # RUN
    # ========================================================

    def run(
        self,
        query: str,
        language: str | None = None,
    ) -> dict[str, Any]:

        total_start = time.perf_counter()

        # ----------------------------------------------------
        # Basic input validation
        # ----------------------------------------------------

        query = (
            query.strip()
            if query
            else ""
        )
        guardrail = self.guardrails.check(
    query
)

        if not guardrail.allowed:

            return {
        "success": False,

        "answer": (
            "I can't process this query."
        ),

        "grounded": False,

        "sources": [],

        "guardrail": {
            "blocked": True,
            "reason": guardrail.reason,
        },

        "latency": {
            "retrieval_ms": 0,
            "generation_ms": 0,
            "total_ms": (
                (time.perf_counter() - total_start)
                * 1000
            ),
        },
    }
        if not query:

            return {
                "success": False,
                "answer": (
                    "Please provide a question."
                ),
                "grounded": False,
                "sources": [],
            }

        # ----------------------------------------------------
        # Retrieval
        # ----------------------------------------------------

        retrieval_start = (
            time.perf_counter()
        )

        documents = (
            self.retriever.search(
                query=query,
                top_k=RETRIEVAL_TOP_K,
                language=language,
            )
        )

        retrieval_latency = (
            time.perf_counter()
            - retrieval_start
        )

        # ----------------------------------------------------
        # No context
        # ----------------------------------------------------

        if not documents:

            total_latency = (
                time.perf_counter()
                - total_start
            )

            return {
                "success": True,

                "answer": (
                    "I don't have enough "
                    "information in the "
                    "retrieved context."
                ),
                "guardrail": {
    "blocked": False,
    "reason": None,
},

                "grounded": False,

                "sources": [],

                "latency": {
                    "retrieval_ms": (
                        retrieval_latency
                        * 1000
                    ),

                    "total_ms": (
                        total_latency
                        * 1000
                    ),
                },
            }

        # ----------------------------------------------------
        # Remove unusable documents
        # ----------------------------------------------------

        documents = [
            document
            for document in documents
            if document.get(
                "text",
                "",
            ).strip()
        ]

        # ----------------------------------------------------
        # Generate answer
        # ----------------------------------------------------

        generation_start = (
            time.perf_counter()
        )

        llm_result = (
            self.llm.generate(
                query=query,
                documents=documents,
            )
        )

        generation_latency = (
            time.perf_counter()
            - generation_start
        )

        # ----------------------------------------------------
        # Final latency
        # ----------------------------------------------------

        total_latency = (
            time.perf_counter()
            - total_start
        )

        # ----------------------------------------------------
        # Source metadata
        # ----------------------------------------------------

        sources = []

        for document in documents:

            sources.append(
                {
                    "chunk_id": document.get(
                        "chunk_id"
                    ),

                    "parent_id": document.get(
                        "parent_id"
                    ),

                    "language": document.get(
                        "language"
                    ),

                    "score": document.get(
                        "score"
                    ),

                    "rrf_score": document.get(
                        "rrf_score"
                    ),

                    "strategy": document.get(
                        "strategy"
                    ),

                    "text": document.get(
                        "text",
                        "",
                    )[:500],
                }
            )

        # ----------------------------------------------------
        # Return
        # ----------------------------------------------------

        return {
            "success": True,

            "answer": llm_result.get(
                "answer",
                "Unable to generate answer.",
            ),

            "grounded": llm_result.get(
                "grounded",
                False,
            ),
            "guardrail": {
    "blocked": False,
    "reason": None,
},

            "sources": sources,

            "latency": {
                "retrieval_ms": (
                    retrieval_latency
                    * 1000
                ),

                "generation_ms": (
                    generation_latency
                    * 1000
                ),

                "total_ms": (
                    total_latency
                    * 1000
                ),
            },
        }


# ============================================================
# SINGLETON
# ============================================================

_rag_service: RAGService | None = None


def get_rag_service() -> RAGService:

    global _rag_service

    if _rag_service is None:

        _rag_service = RAGService()

    return _rag_service