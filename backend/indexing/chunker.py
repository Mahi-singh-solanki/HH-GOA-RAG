from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


# ============================================================
# CONFIGURATION
# ============================================================

SEMANTIC_TARGET_WORDS = 180
SEMANTIC_MAX_WORDS = 260
SEMANTIC_OVERLAP_SENTENCES = 1

FIXED_CHUNK_WORDS = 220
FIXED_OVERLAP_WORDS = 40

MIN_CHUNK_WORDS = 20


# ============================================================
# DATA STRUCTURE
# ============================================================

@dataclass
class Chunk:
    chunk_id: str
    text: str
    parent_id: str
    strategy: str

    language: str | None = None

    query: str | None = None
    answer: str | None = None

    is_selected: bool = False

    metadata: dict[str, Any] | None = None


# ============================================================
# TEXT UTILITIES
# ============================================================

def normalize_text(text: str) -> str:

    if not text:
        return ""

    text = str(text)

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def word_count(text: str) -> int:

    return len(
        normalize_text(text).split()
    )


def split_sentences(text: str) -> list[str]:
    """
    Lightweight multilingual sentence splitting.

    Handles:
        English: . ! ?
        Hindi:    । ॥
    """

    text = normalize_text(text)

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?।॥])\s+",
        text,
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def make_chunk_id(
    parent_id: str,
    language: str,
    strategy: str,
    index: int,
) -> str:

    return (
        f"{parent_id}"
        f"_{language}"
        f"_{strategy}"
        f"_{index}"
    )


# ============================================================
# STRATEGY 1
# SEMANTIC / SENTENCE-AWARE
# ============================================================

def semantic_chunk(
    text: str,
    parent_id: str,
    language: str,
    query: str,
    answer: str,
    is_selected: bool,
) -> list[Chunk]:

    sentences = split_sentences(text)

    if not sentences:
        return []

    chunks: list[Chunk] = []

    current: list[str] = []
    current_words = 0

    chunk_index = 0

    for sentence in sentences:

        sentence_words = word_count(
            sentence
        )

        # --------------------------------------------
        # Extremely long sentence
        # --------------------------------------------

        if sentence_words > SEMANTIC_MAX_WORDS:

            if current:

                chunk_text = " ".join(
                    current
                )

                chunks.append(
                    Chunk(
                        chunk_id=make_chunk_id(
                            parent_id,
                            language,
                            "semantic",
                            chunk_index,
                        ),
                        text=chunk_text,
                        parent_id=parent_id,
                        strategy="semantic",
                        language=language,
                        query=query,
                        answer=answer,
                        is_selected=is_selected,
                    )
                )

                chunk_index += 1

                current = []
                current_words = 0

            # Let fixed-window splitting handle this
            # sentence later.
            large_chunks = fixed_window_chunk(
                sentence,
                parent_id,
                language,
                query,
                answer,
                is_selected,
            )

            for large_chunk in large_chunks:

                large_chunk.chunk_id = (
                    make_chunk_id(
                        parent_id,
                        language,
                        "fixed_fallback",
                        chunk_index,
                    )
                )

                chunks.append(
                    large_chunk
                )

                chunk_index += 1

            continue

        # --------------------------------------------
        # Add sentence
        # --------------------------------------------

        if (
            current_words + sentence_words
            <= SEMANTIC_MAX_WORDS
        ):

            current.append(sentence)

            current_words += sentence_words

        else:

            # ----------------------------------------
            # Save current chunk
            # ----------------------------------------

            if current:

                chunk_text = " ".join(
                    current
                )

                chunks.append(
                    Chunk(
                        chunk_id=make_chunk_id(
                            parent_id,
                            language,
                            "semantic",
                            chunk_index,
                        ),
                        text=chunk_text,
                        parent_id=parent_id,
                        strategy="semantic",
                        language=language,
                        query=query,
                        answer=answer,
                        is_selected=is_selected,
                    )
                )

                chunk_index += 1

            # ----------------------------------------
            # Sentence overlap
            # ----------------------------------------

            overlap = current[
                -SEMANTIC_OVERLAP_SENTENCES:
            ]

            current = overlap + [
                sentence
            ]

            current_words = sum(
                word_count(s)
                for s in current
            )

    # --------------------------------------------
    # Final chunk
    # --------------------------------------------

    if current:

        chunk_text = " ".join(
            current
        )

        chunks.append(
            Chunk(
                chunk_id=make_chunk_id(
                    parent_id,
                    language,
                    "semantic",
                    chunk_index,
                ),
                text=chunk_text,
                parent_id=parent_id,
                strategy="semantic",
                language=language,
                query=query,
                answer=answer,
                is_selected=is_selected,
            )
        )

    return [
        chunk
        for chunk in chunks
        if word_count(chunk.text)
        >= MIN_CHUNK_WORDS
    ]


# ============================================================
# STRATEGY 2
# FIXED WINDOW
# ============================================================

def fixed_window_chunk(
    text: str,
    parent_id: str,
    language: str,
    query: str,
    answer: str,
    is_selected: bool,
    chunk_size: int = FIXED_CHUNK_WORDS,
    overlap: int = FIXED_OVERLAP_WORDS,
) -> list[Chunk]:

    words = normalize_text(
        text
    ).split()

    if not words:
        return []

    if overlap >= chunk_size:

        raise ValueError(
            "overlap must be smaller "
            "than chunk_size"
        )

    chunks: list[Chunk] = []

    step = (
        chunk_size - overlap
    )

    start = 0
    index = 0

    while start < len(words):

        chunk_words = words[
            start:start + chunk_size
        ]

        chunk_text = " ".join(
            chunk_words
        )

        if (
            word_count(chunk_text)
            >= MIN_CHUNK_WORDS
        ):

            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(
                        parent_id,
                        language,
                        "fixed",
                        index,
                    ),
                    text=chunk_text,
                    parent_id=parent_id,
                    strategy="fixed",
                    language=language,
                    query=query,
                    answer=answer,
                    is_selected=is_selected,
                )
            )

            index += 1

        start += step

    return chunks


# ============================================================
# STRATEGY 3
# QUESTION-AWARE
# ============================================================

def question_aware_chunk(
    passage: str,
    query: str,
    answer: str,
    parent_id: str,
    language: str,
    is_selected: bool,
) -> list[Chunk]:
    """
    MSMARCO-specific strategy.

    The query is included in the embedding text because
    MSMARCO is fundamentally a query-passage retrieval dataset.
    """

    passage = normalize_text(
        passage
    )

    query = normalize_text(
        query
    )

    answer = normalize_text(
        answer
    )

    if not passage:
        return []

    base_chunks = semantic_chunk(
        text=passage,
        parent_id=parent_id,
        language=language,
        query=query,
        answer=answer,
        is_selected=is_selected,
    )

    result: list[Chunk] = []

    for index, chunk in enumerate(
        base_chunks
    ):

        enriched_text = (
            f"Question: {query}\n\n"
            f"Passage: {chunk.text}"
        )

        result.append(
            Chunk(
                chunk_id=make_chunk_id(
                    parent_id,
                    language,
                    "question_aware",
                    index,
                ),
                text=enriched_text,
                parent_id=parent_id,
                strategy="question_aware",
                language=language,
                query=query,
                answer=answer,
                is_selected=is_selected,
                metadata={
                    "original_passage": chunk.text,
                },
            )
        )

    return result


# ============================================================
# STRATEGY SELECTION
# ============================================================

def choose_strategy(
    text: str,
    query: str | None = None,
) -> str:

    text = normalize_text(
        text
    )

    if not text:
        return "fixed"

    # MSMARCO query + passage
    if query and query.strip():
        return "question_aware"

    # Very long / malformed text
    if word_count(text) > 1500:
        return "fixed"

    # Normal text
    return "semantic"


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def chunk_document(
    text: str,
    parent_id: str,
    language: str,
    query: str,
    answer: str,
    is_selected: bool,
) -> list[Chunk]:

    strategy = choose_strategy(
        text=text,
        query=query,
    )

    if strategy == "question_aware":

        return question_aware_chunk(
            passage=text,
            query=query,
            answer=answer,
            parent_id=parent_id,
            language=language,
            is_selected=is_selected,
        )

    if strategy == "semantic":

        return semantic_chunk(
            text=text,
            parent_id=parent_id,
            language=language,
            query=query,
            answer=answer,
            is_selected=is_selected,
        )

    return fixed_window_chunk(
        text=text,
        parent_id=parent_id,
        language=language,
        query=query,
        answer=answer,
        is_selected=is_selected,
    )