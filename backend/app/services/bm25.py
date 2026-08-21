from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_FILE = (
    BASE_DIR
    / "data"
    / "msmarco_xi"
    / "processed.jsonl"
)

INDEX_FILE = (
    BASE_DIR
    / "data"
    / "bm25"
    / "index.json"
)

DEFAULT_TOP_K = 10


# ============================================================
# TEXT PROCESSING
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize text while preserving English + Hindi characters.
    """

    if not text:
        return ""

    text = str(text).lower().strip()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


def tokenize(text: str) -> list[str]:
    """
    Lightweight multilingual tokenizer.

    Supports:
        English
        Hindi / Devanagari
        numbers
    """

    text = normalize_text(text)

    if not text:
        return []

    tokens = re.findall(
        r"[\w\u0900-\u097F]+",
        text,
        flags=re.UNICODE,
    )

    return tokens


# ============================================================
# BM25 SERVICE
# ============================================================

class BM25Service:

    def __init__(
        self,
        index_file: Path = INDEX_FILE,
    ):

        self.index_file = index_file

        self.documents: list[dict[str, Any]] = []

        self.bm25: BM25Okapi | None = None

        self._load_index()

    # ========================================================
    # LOAD
    # ========================================================

    def _load_index(self) -> None:

        if not self.index_file.exists():
            raise FileNotFoundError(
            "\nBM25 index not found.\n"
            f"Expected:\n{self.index_file}\n\n"
            "Run build_bm25.py first."
            )

        with self.index_file.open(
        "r",
        encoding="utf-8",
        ) as file:

            data = json.load(file)

        self.documents = data["documents"]

        tokenized_documents = data[
        "tokenized_documents"
        ]

        self.bm25 = BM25Okapi(
        tokenized_documents
        )

        print(
        f"BM25 loaded: "
        f"{len(self.documents):,} documents"
        )

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        language: str | None = None,
    ) -> list[dict]:

        if not query or not query.strip():

            return []

        if self.bm25 is None:

            return []

        query_tokens = tokenize(
            query
        )

        if not query_tokens:

            return []

        # ----------------------------------------------------
        # BM25 scores
        # ----------------------------------------------------

        scores = self.bm25.get_scores(
            query_tokens
        )

        # ----------------------------------------------------
        # Sort highest score first
        # ----------------------------------------------------

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        results = []

        for index in ranked_indices:

            document = self.documents[
                index
            ]

            # ------------------------------------------------
            # Optional language filtering
            # ------------------------------------------------

            if language:

                if (
                    document.get(
                        "language"
                    )
                    != language
                ):
                    continue

            score = float(
                scores[index]
            )

            # Ignore zero-score documents.
            if score <= 0:
                continue

            results.append(
                {
                    "chunk_id": document.get(
                        "chunk_id",
                        "",
                    ),

                    "parent_id": document.get(
                        "parent_id",
                        "",
                    ),

                    "text": document.get(
                        "text",
                        "",
                    ),

                    "language": document.get(
                        "language",
                    ),

                    "query": document.get(
                        "query",
                    ),

                    "answer": document.get(
                        "answer",
                    ),

                    "is_selected": document.get(
                        "is_selected",
                        False,
                    ),

                    "strategy": "bm25",

                    "score": score,
                }
            )

            if len(results) >= top_k:

                break

        return results

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self) -> None:
        pass


# ============================================================
# SINGLETON
# ============================================================

_bm25_service: BM25Service | None = None


def get_bm25_service() -> BM25Service:

    global _bm25_service

    if _bm25_service is None:

        _bm25_service = BM25Service()

    return _bm25_service