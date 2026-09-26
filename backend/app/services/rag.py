from __future__ import annotations

import time
from collections import defaultdict
from functools import lru_cache
from threading import Lock
from typing import Any

from app.config import get_settings
from app.services.document_retrieval import (
    get_document_retrieval_service,
)
from app.services.guardrails import (
    validate_query,
)
from app.services.llm import (
    get_llm_service,
)


class RAGService:

    def __init__(self):

        self.settings = get_settings()

        self.retrieval = (
            get_document_retrieval_service()
        )

        self.llm = get_llm_service()

        self.conversations = defaultdict(list)

        self.lock = Lock()

    def _get_history(
        self,
        conversation_id: str,
    ) -> list[dict[str, str]]:

        with self.lock:

            return list(
                self.conversations[
                    conversation_id
                ]
            )

    def _save_turn(
        self,
        conversation_id: str,
        query: str,
        answer: str,
    ):

        with self.lock:

            history = (
                self.conversations[
                    conversation_id
                ]
            )

            history.append(
                {
                    "role": "user",
                    "content": query,
                }
            )

            history.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )

            max_messages = (
                self.settings
                .conversation_max_messages
            )

            self.conversations[
                conversation_id
            ] = history[
                -max_messages:
            ]

    def _build_retrieval_query(
        self,
        query: str,
        history: list[dict[str, str]],
    ) -> str:

        stripped = query.strip()

        if (
            len(stripped.split()) >= 5
            or not history
        ):
            return stripped

        previous_user_messages = [
            item["content"]
            for item in history
            if item["role"] == "user"
        ]

        if not previous_user_messages:
            return stripped

        previous = (
            previous_user_messages[-1]
        )

        return (
            f"Previous question: {previous}\n"
            f"Follow-up question: {stripped}"
        )

    def _normalize_sources(
        self,
        sources: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        normalized = []

        for index, source in enumerate(
            sources,
            start=1,
        ):

            normalized.append(
                {
                    **source,
                    "source_id": index,
                    "text": source.get(
                        "text",
                        "",
                    )[:5000],
                }
            )

        return normalized

    def run(
        self,
        query: str,
        language: str | None = None,
        conversation_id: str = "default",
    ) -> dict[str, Any]:

        total_start = time.perf_counter()

        validation = validate_query(
            query
        )

        if not validation.get(
            "valid",
            False,
        ):

            return {
                "success": False,
                "answer": validation.get(
                    "reason",
                    "Invalid query.",
                ),
                "grounded": False,
                "sources": [],
                "latency": {
                    "retrieval_ms": 0,
                    "generation_ms": 0,
                    "total_ms": (
                        time.perf_counter()
                        - total_start
                    )
                    * 1000,
                },
                "conversation_id": (
                    conversation_id
                ),
            }

        history = self._get_history(
            conversation_id
        )

        retrieval_query = (
            self._build_retrieval_query(
                query,
                history,
            )
        )

        retrieval_start = time.perf_counter()

        sources = (
            self.retrieval.retrieve(
                query=retrieval_query,
                history=history,
            )
        )

        retrieval_ms = (
            time.perf_counter()
            - retrieval_start
        ) * 1000

        generation_start = time.perf_counter()

        result = self.llm.generate(
            query=query,
            sources=sources,
            history=history,
        )

        generation_ms = (
            time.perf_counter()
            - generation_start
        ) * 1000

        used_source_ids = set(
            result.get(
                "used_sources",
                [],
            )
        )

        normalized_sources = (
            self._normalize_sources(
                sources
            )
        )

        selected_sources = [
            source
            for source in normalized_sources
            if source["source_id"]
            in used_source_ids
        ]

        if not selected_sources:

            selected_sources = (
                normalized_sources
            )

        answer = result.get(
            "answer",
            "No answer generated.",
        )

        self._save_turn(
            conversation_id,
            query,
            answer,
        )

        total_ms = (
            time.perf_counter()
            - total_start
        ) * 1000

        return {
            "success": True,
            "answer": answer,
            "grounded": bool(
                result.get(
                    "grounded",
                    False,
                )
            ),
            "sources": selected_sources,
            "latency": {
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "total_ms": total_ms,
            },
            "conversation_id": (
                conversation_id
            ),
        }


@lru_cache
def get_rag_service():

    return RAGService()