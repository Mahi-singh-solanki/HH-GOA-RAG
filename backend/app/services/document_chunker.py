from __future__ import annotations

import re
from typing import Any
from uuid import uuid5, NAMESPACE_URL


def normalize_text(text: str) -> str:

    text = text.replace("\x00", " ")

    lines = []

    for line in text.splitlines():

        line = " ".join(line.split())

        if line:
            lines.append(line)

    return "\n".join(lines)


def detect_section(
    text: str,
    fallback: str | None = None,
) -> str | None:

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        words = line.split()

        if len(words) <= 12:

            if (
                line.isupper()
                or line.startswith("#")
                or not re.search(
                    r"[.!?]$",
                    line,
                )
            ):
                return line.lstrip("#").strip()

    return fallback


def split_words(
    text: str,
    chunk_size: int = 350,
    overlap: int = 60,
) -> list[str]:

    words = text.split()

    if not words:
        return []

    if len(words) <= chunk_size:
        return [text]

    chunks = []

    start = 0

    while start < len(words):

        end = min(
            start + chunk_size,
            len(words),
        )

        chunk = " ".join(
            words[start:end]
        )

        chunks.append(chunk)

        if end >= len(words):
            break

        start = max(
            end - overlap,
            start + 1,
        )

    return chunks


def chunk_processed_document(
    document: dict[str, Any],
    chunk_size: int = 350,
    overlap: int = 60,
) -> list[dict[str, Any]]:

    document_id = document["document_id"]
    filename = document["filename"]
    file_type = document["file_type"]

    chunks = []

    for page in document.get("pages", []):

        page_number = page.get(
            "page_number",
            1,
        )

        text = normalize_text(
            page.get("text", "")
        )

        if not text:
            continue

        section = (
            page.get("section")
            or detect_section(text)
        )

        page_chunks = split_words(
            text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk_index, chunk_text in enumerate(
            page_chunks
        ):

            raw_id = (
                f"{document_id}:"
                f"{page_number}:"
                f"{chunk_index}"
            )

            chunk_id = str(
                uuid5(
                    NAMESPACE_URL,
                    raw_id,
                )
            )

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "filename": filename,
                    "file_type": file_type,
                    "page": page_number,
                    "section": section,
                    "text": chunk_text,
                    "source_type": page.get(
                        "source",
                        file_type,
                    ),
                    "ocr_used": page.get(
                        "ocr_used",
                        False,
                    ),
                    "chunk_index": chunk_index,
                }
            )

    return chunks