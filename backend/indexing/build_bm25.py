from __future__ import annotations

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = (
    BASE_DIR
    / "data"
    / "msmarco_xi"
    / "processed.jsonl"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "bm25"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "index.json"
)


# ============================================================
# TEXT PROCESSING
# ============================================================

def normalize_text(text: str) -> str:

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

    text = normalize_text(text)

    if not text:
        return []

    return re.findall(
        r"[\w\u0900-\u097F]+",
        text,
        flags=re.UNICODE,
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_documents() -> list[dict]:

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"\nDataset not found:\n"
            f"{DATA_FILE.resolve()}\n"
        )

    documents = []

    print(
        f"Loading:\n{DATA_FILE.resolve()}"
    )

    with DATA_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            record = json.loads(
                line
            )

            parent_id = str(
                record.get("id", "")
            )

            passages = record.get(
                "passages",
                {},
            )

            # =================================================
            # ENGLISH PASSAGES
            # =================================================

            english_passages = (
                passages.get(
                    "english",
                    [],
                )
            )

            english_selected = (
                passages.get(
                    "is_selected",
                    [],
                )
            )

            english_query = record.get(
                "english_query",
                "",
            )

            english_answer = record.get(
                "english_answer",
                "",
            )

            for index, passage in enumerate(
                english_passages
            ):

                passage = str(
                    passage
                ).strip()

                if not passage:
                    continue

                selected = (
                    bool(
                        english_selected[index]
                    )
                    if index
                    < len(english_selected)
                    else False
                )

                documents.append(
                    {
                        "chunk_id": (
                            f"{parent_id}"
                            f"_en_bm25_{index}"
                        ),

                        "parent_id": parent_id,

                        "text": passage,

                        "language": "en",

                        "query": english_query,

                        "answer": english_answer,

                        "is_selected": selected,
                    }
                )

            # =================================================
            # HINDI PASSAGES
            # =================================================

            hindi_passages = (
                passages.get(
                    "hindi",
                    [],
                )
            )

            hindi_query = record.get(
                "query",
                "",
            )

            hindi_answer = record.get(
                "answer",
                "",
            )

            for index, passage in enumerate(
                hindi_passages
            ):

                passage = str(
                    passage
                ).strip()

                if not passage:
                    continue

                selected = (
                    bool(
                        english_selected[index]
                    )
                    if index
                    < len(english_selected)
                    else False
                )

                documents.append(
                    {
                        "chunk_id": (
                            f"{parent_id}"
                            f"_hi_bm25_{index}"
                        ),

                        "parent_id": parent_id,

                        "text": passage,

                        "language": "hi",

                        "query": hindi_query,

                        "answer": hindi_answer,

                        "is_selected": selected,
                    }
                )

    return documents


# ============================================================
# BUILD INDEX
# ============================================================

def build_index(
    documents: list[dict],
) -> None:

    if not documents:

        raise RuntimeError(
            "No documents found."
        )

    print(
        f"\nDocuments: "
        f"{len(documents):,}"
    )

    print(
        "Tokenizing documents..."
    )

    tokenized_documents = [
        tokenize(
            document["text"]
        )
        for document in documents
    ]

    print(
        "Building BM25..."
    )

    # --------------------------------------------------------
    # This validates the corpus and calculates BM25 statistics.
    # --------------------------------------------------------

    BM25Okapi(
        tokenized_documents
    )

    # --------------------------------------------------------
    # Save documents + tokenized corpus.
    #
    # We don't need to serialize the BM25 Python object.
    # BM25Service rebuilds it very quickly from these tokens.
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = {
        "documents": documents,

        "tokenized_documents": (
            tokenized_documents
        ),
    }

    print(
        f"\nSaving index to:\n"
        f"{OUTPUT_FILE.resolve()}"
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
        )

    print(
        "\nBM25 index created successfully."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MSMARCO-XI BM25 INDEX BUILDER")
    print("=" * 70)

    documents = load_documents()

    english_count = sum(
        1
        for document in documents
        if document["language"] == "en"
    )

    hindi_count = sum(
        1
        for document in documents
        if document["language"] == "hi"
    )

    print(
        f"\nEnglish documents: "
        f"{english_count:,}"
    )

    print(
        f"Hindi documents: "
        f"{hindi_count:,}"
    )

    build_index(
        documents
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()