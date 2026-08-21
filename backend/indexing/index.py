from __future__ import annotations
import uuid
import json
import sys
from pathlib import Path
from typing import Iterator

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

# Allow imports from backend/
sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

from app.config import get_settings
from app.services.embeddings import (
    get_embedding_service,
)
from indexing.chunker import (
    Chunk,
    chunk_document,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = Path(
    "data/msmarco_xi/processed.jsonl"
)

QDRANT_PATH = Path(
    "data/qdrant"
)

COLLECTION_NAME = "msmarco_xi"

EMBEDDING_BATCH_SIZE = 64

UPSERT_BATCH_SIZE = 128


# ============================================================
# DATA LOADER
# ============================================================
def make_qdrant_id(chunk_id: str) -> str:
    """
    Convert our human-readable chunk ID into a deterministic UUID.

    The same chunk_id always produces the same UUID, which means
    rerunning the indexer updates the same Qdrant point instead
    of creating duplicates.
    """

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"msmarco-xi:{chunk_id}",
        )
    )
def load_records(
    file_path: Path,
) -> Iterator[dict]:

    """
    Stream JSONL records.

    We intentionally do not load the entire dataset into RAM.
    """

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            yield json.loads(line)


# ============================================================
# CHUNK GENERATOR
# ============================================================

def generate_chunks(
    records: Iterator[dict],
) -> Iterator[Chunk]:

    """
    Convert MSMARCO-XI records into searchable chunks.

    Every language is indexed independently.

    Example:

        English question
        +
        English passage

        Hindi question
        +
        Hindi passage
    """

    for record in records:

        parent_id = str(
            record["id"]
        )

        hindi_query = record.get(
            "query",
            "",
        )

        english_query = record.get(
            "english_query",
            "",
        )

        hindi_answer = record.get(
            "answer",
            "",
        )

        english_answer = record.get(
            "english_answer",
            "",
        )

        passages = record.get(
            "passages",
            {},
        )

        hindi_passages = passages.get(
            "hindi",
            [],
        )

        english_passages = passages.get(
            "english",
            [],
        )

        selected = passages.get(
            "is_selected",
            [],
        )

        # ====================================================
        # ENGLISH
        # ====================================================

        for index, passage in enumerate(
            english_passages
        ):

            if not passage:
                continue

            is_selected = (
                bool(selected[index])
                if index < len(selected)
                else False
            )

            chunks = chunk_document(
                text=passage,

                parent_id=parent_id,

                language="en",

                query=english_query,

                answer=english_answer,

                is_selected=is_selected,
            )

            for chunk in chunks:

                yield chunk

        # ====================================================
        # HINDI
        # ====================================================

        for index, passage in enumerate(
            hindi_passages
        ):

            if not passage:
                continue

            is_selected = (
                bool(selected[index])
                if index < len(selected)
                else False
            )

            chunks = chunk_document(
                text=passage,

                parent_id=parent_id,

                language="hi",

                query=hindi_query,

                answer=hindi_answer,

                is_selected=is_selected,
            )

            for chunk in chunks:

                yield chunk


# ============================================================
# QDRANT
# ============================================================

def create_qdrant_collection(
    client: QdrantClient,
    dimension: int,
) -> None:

    collections = (
        client.get_collections()
        .collections
    )

    existing_names = {
        collection.name
        for collection in collections
    }

    if COLLECTION_NAME in existing_names:

        print(
            f"Collection "
            f"'{COLLECTION_NAME}' "
            f"already exists."
        )

        return

    print(
        f"Creating collection "
        f"'{COLLECTION_NAME}'..."
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,

        vectors_config=VectorParams(
            size=dimension,
            distance=Distance.COSINE,
        ),
    )

    print("Collection created.")


# ============================================================
# BATCH INDEXING
# ============================================================

def index_chunks(
    client: QdrantClient,
    chunks: Iterator[Chunk],
    embedding_service,
) -> tuple[int, int]:

    """
    Embed chunks in batches and upload them to Qdrant.
    """

    chunk_batch: list[Chunk] = []

    total_chunks = 0

    total_points = 0

    batch_number = 0

    for chunk in chunks:

        chunk_batch.append(
            chunk
        )

        if len(chunk_batch) < (
            EMBEDDING_BATCH_SIZE
        ):
            continue

        batch_number += 1

        total_chunks += len(
            chunk_batch
        )

        print(
            f"\nEmbedding batch "
            f"{batch_number}"
            f" | chunks={len(chunk_batch)}"
            f" | total={total_chunks:,}"
        )

        # ----------------------------------------------------
        # Generate embeddings
        # ----------------------------------------------------

        texts = [
            chunk.text
            for chunk in chunk_batch
        ]

        vectors = (
            embedding_service.embed_texts(
                texts,
                batch_size=EMBEDDING_BATCH_SIZE,
            )
        )

        # ----------------------------------------------------
        # Create Qdrant points
        # ----------------------------------------------------

        points = []

        for chunk, vector in zip(
            chunk_batch,
            vectors,
        ):

            payload = {
                "chunk_id": chunk.chunk_id,

                "parent_id": chunk.parent_id,

                "text": chunk.text,

                "language": chunk.language,

                "query": chunk.query,

                "answer": chunk.answer,

                "strategy": chunk.strategy,

                "is_selected": (
                    chunk.is_selected
                ),

                "metadata": (
                    chunk.metadata
                    or {}
                ),
            }

            points.append(
                PointStruct(
                    id=make_qdrant_id(
            chunk.chunk_id
        ),

                    vector=vector,

                    payload=payload,
                )
            )

        # ----------------------------------------------------
        # Upload to Qdrant
        # ----------------------------------------------------

        for start in range(
            0,
            len(points),
            UPSERT_BATCH_SIZE,
        ):

            batch = points[
                start:
                start + UPSERT_BATCH_SIZE
            ]

            client.upsert(
                collection_name=(
                    COLLECTION_NAME
                ),

                points=batch,

                wait=True,
            )

            total_points += len(
                batch
            )

        chunk_batch.clear()

    # ========================================================
    # Remaining chunks
    # ========================================================

    if chunk_batch:

        batch_number += 1

        total_chunks += len(
            chunk_batch
        )

        print(
            f"\nEmbedding final batch "
            f"{batch_number}"
            f" | chunks={len(chunk_batch)}"
        )

        texts = [
            chunk.text
            for chunk in chunk_batch
        ]

        vectors = (
            embedding_service.embed_texts(
                texts,
                batch_size=EMBEDDING_BATCH_SIZE,
            )
        )

        points = []

        for chunk, vector in zip(
            chunk_batch,
            vectors,
        ):

            payload = {
                "chunk_id": chunk.chunk_id,

                "parent_id": chunk.parent_id,

                "text": chunk.text,

                "language": chunk.language,

                "query": chunk.query,

                "answer": chunk.answer,

                "strategy": chunk.strategy,

                "is_selected": (
                    chunk.is_selected
                ),

                "metadata": (
                    chunk.metadata
                    or {}
                ),
            }

            points.append(
                PointStruct(
                    id=make_qdrant_id(
            chunk.chunk_id
        ),

                    vector=vector,

                    payload=payload,
                )
            )

        for start in range(
            0,
            len(points),
            UPSERT_BATCH_SIZE,
        ):

            batch = points[
                start:
                start + UPSERT_BATCH_SIZE
            ]

            client.upsert(
                collection_name=(
                    COLLECTION_NAME
                ),

                points=batch,

                wait=True,
            )

            total_points += len(
                batch
            )

    return (
        total_chunks,
        total_points,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MSMARCO-XI VECTOR INDEX BUILDER")
    print("=" * 70)

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"\nProcessed dataset not found:\n"
            f"{DATA_FILE.resolve()}\n\n"
            f"Run load_dataset.py first."
        )

    # --------------------------------------------------------
    # Qdrant directory
    # --------------------------------------------------------

    QDRANT_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"\nDataset:"
        f"\n{DATA_FILE.resolve()}"
    )

    print(
        f"\nQdrant:"
        f"\n{QDRANT_PATH.resolve()}"
    )

    # --------------------------------------------------------
    # Embedding model
    # --------------------------------------------------------

    print(
        "\nLoading embedding model..."
    )

    embedding_service = (
        get_embedding_service()
    )

    dimension = (
        embedding_service.dimension
    )

    print(
        f"Embedding model:"
        f"\n{embedding_service.model_name}"
    )

    print(
        f"Embedding dimension: "
        f"{dimension}"
    )

    # --------------------------------------------------------
    # Qdrant
    # --------------------------------------------------------

    print(
        "\nOpening Qdrant..."
    )

    client = QdrantClient(
        path=str(
            QDRANT_PATH
        )
    )

    create_qdrant_collection(
        client=client,

        dimension=dimension,
    )

    # --------------------------------------------------------
    # Generate chunks
    # --------------------------------------------------------

    print(
        "\nGenerating chunks..."
    )

    records = load_records(
        DATA_FILE
    )

    chunks = generate_chunks(
        records
    )

    # --------------------------------------------------------
    # Index
    # --------------------------------------------------------

    total_chunks, total_points = (
        index_chunks(
            client=client,

            chunks=chunks,

            embedding_service=(
                embedding_service
            ),
        )
    )

    # --------------------------------------------------------
    # Final information
    # --------------------------------------------------------

    collection_info = (
        client.get_collection(
            COLLECTION_NAME
        )
    )

    print()
    print("=" * 70)
    print("INDEXING COMPLETE")
    print("=" * 70)

    print(
        f"Chunks processed : "
        f"{total_chunks:,}"
    )

    print(
        f"Points indexed   : "
        f"{total_points:,}"
    )

    print(
        f"Qdrant points    : "
        f"{collection_info.points_count:,}"
    )

    print(
        f"\nCollection:"
        f"\n{COLLECTION_NAME}"
    )

    client.close()


if __name__ == "__main__":
    main()