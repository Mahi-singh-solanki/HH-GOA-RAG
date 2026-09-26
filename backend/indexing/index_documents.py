from __future__ import annotations

from pathlib import Path

from app.config import get_settings
from app.services.document_indexer import (
    get_document_indexer,
)
from app.services.document_processor import (
    process_document,
)


def main():

    settings = get_settings()

    documents_dir = Path(
        settings.documents_dir
    )

    indexer = (
        get_document_indexer()
    )

    for file_path in documents_dir.iterdir():

        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in {
            ".pdf",
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".docx",
            ".pptx",
            ".xlsx",
            ".csv",
            ".txt",
            ".md",
        }:
            continue

        if file_path.name.endswith(
            ".json"
        ):
            continue

        print(
            f"Processing: {file_path.name}"
        )

        processed = process_document(
            file_path=file_path,
            original_filename=file_path.name,
        )

        result = indexer.index(
            file_path=file_path,
            processed_document=processed,
            original_filename=file_path.name,
        )

        print(
            f"Indexed: {result}"
        )


if __name__ == "__main__":
    main()