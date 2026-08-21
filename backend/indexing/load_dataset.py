from pathlib import Path
import json

import duckdb


# ============================================================
# CONFIG
# ============================================================

PARQUET_FILE = Path(
    "data/msmarco_xi/train/hintrain.parquet"
)

OUTPUT_FILE = Path(
    "data/msmarco_xi/processed.jsonl"
)

MAX_RECORDS = 20_000


# ============================================================
# MAIN
# ============================================================

def main():

    if not PARQUET_FILE.exists():
        raise FileNotFoundError(
            f"Parquet file not found:\n"
            f"{PARQUET_FILE.resolve()}"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("MSMARCO-XI DUCKDB PROCESSOR")
    print("=" * 70)

    print(
        f"\nInput:\n"
        f"{PARQUET_FILE.resolve()}"
    )

    print(
        f"\nOutput:\n"
        f"{OUTPUT_FILE.resolve()}"
    )

    print(
        f"\nMaximum records: "
        f"{MAX_RECORDS:,}"
    )

    # --------------------------------------------------------
    # Create DuckDB connection
    # --------------------------------------------------------

    con = duckdb.connect()

    # --------------------------------------------------------
    # First inspect the schema
    # --------------------------------------------------------

    print("\nInspecting Parquet schema...")

    schema = con.execute(
        f"""
        DESCRIBE
        SELECT *
        FROM read_parquet(
            '{PARQUET_FILE.as_posix()}'
        )
        """
    ).fetchall()

    print("\nColumns:")

    for column in schema:
        print(
            f"  {column[0]:25} "
            f"{column[1]}"
        )

    # --------------------------------------------------------
    # Extract data
    # --------------------------------------------------------
    #
    # We use DuckDB's struct/list operators instead of
    # converting the nested Arrow column through PyArrow.
    #
    # --------------------------------------------------------

    print("\nExtracting records...")

    query = f"""
        SELECT
            query_id,
            query,
            "Eng_Query",
            "Answer",
            "Eng_Answer",
            query_type,

            passages.English_passages
                AS english_passages,

            passages.Translated_passages
                AS hindi_passages,

            passages.is_selected
                AS is_selected

        FROM read_parquet(
            '{PARQUET_FILE.as_posix()}'
        )

        WHERE
            query IS NOT NULL
            AND "Eng_Query" IS NOT NULL
            AND passages IS NOT NULL

        LIMIT {MAX_RECORDS}
    """

    result = con.execute(
        query
    ).fetchall()

    print(
        f"Rows retrieved: "
        f"{len(result):,}"
    )

    # --------------------------------------------------------
    # Write JSONL
    # --------------------------------------------------------

    print("\nWriting JSONL...")

    written = 0

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as output:

        for row in result:

            (
                query_id,
                hindi_query,
                english_query,
                hindi_answer,
                english_answer,
                query_type,
                english_passages,
                hindi_passages,
                is_selected,
            ) = row

            # --------------------------------------------
            # Convert values
            # --------------------------------------------

            hindi_query = (
                str(hindi_query).strip()
                if hindi_query
                else ""
            )

            english_query = (
                str(english_query).strip()
                if english_query
                else ""
            )

            hindi_answer = (
                str(hindi_answer).strip()
                if hindi_answer
                else ""
            )

            english_answer = (
                str(english_answer).strip()
                if english_answer
                else ""
            )

            # --------------------------------------------
            # Convert LIST values
            # --------------------------------------------

            english_passages = (
                list(english_passages)
                if english_passages
                else []
            )

            hindi_passages = (
                list(hindi_passages)
                if hindi_passages
                else []
            )

            is_selected = (
                list(is_selected)
                if is_selected
                else []
            )

            # --------------------------------------------
            # Clean passages
            # --------------------------------------------

            english_passages = [
                str(p).strip()
                for p in english_passages
                if p
                and str(p).strip()
            ]

            hindi_passages = [
                str(p).strip()
                for p in hindi_passages
                if p
                and str(p).strip()
            ]

            # --------------------------------------------
            # Skip unusable records
            # --------------------------------------------

            if not hindi_query:
                continue

            if not english_query:
                continue

            if not english_passages:
                continue

            if not hindi_passages:
                continue

            # --------------------------------------------
            # Create normalized record
            # --------------------------------------------

            record = {
                "id": str(query_id),

                "language": "hi",

                "query": hindi_query,

                "english_query": english_query,

                "answer": hindi_answer,

                "english_answer": english_answer,

                "query_type": (
                    str(query_type).strip()
                    if query_type
                    else ""
                ),

                "passages": {
                    "english": english_passages,

                    "hindi": hindi_passages,

                    "is_selected": [
                        int(x)
                        for x in is_selected
                        if x is not None
                    ],
                },
            }

            output.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            written += 1

            if written % 100 == 0:

                print(
                    f"\rWritten: "
                    f"{written:,}/"
                    f"{MAX_RECORDS:,}",
                    end="",
                )

    print()

    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"Records written: "
        f"{written:,}"
    )

    print(
        f"\nOutput:"
        f"\n{OUTPUT_FILE.resolve()}"
    )

    con.close()


if __name__ == "__main__":
    main()