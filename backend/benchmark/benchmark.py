from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import requests


# ============================================================
# CONFIG
# ============================================================

API_URL = "http://127.0.0.1:8000/rag"

OUTPUT_FILE = Path(
    "benchmark_results.json"
)

NUMBER_OF_QUERIES = 5


TEST_QUERIES = [
    "What is the capital of India?",
    "भारत की राजधानी क्या है?",
    "What is artificial intelligence?",
    "कंप्यूटर क्या है?",
    "What is machine learning?",
    "भारत का मौसम कैसा है?",
    "What is the meaning of artificial intelligence?",
    "What is a computer?",
    "What is climate?",
    "भारत में कितने राज्य हैं?",
    "What is the internet?",
    "What is a database?",
    "डेटाबेस क्या है?",
    "What is technology?",
    "What is science?",
    "विज्ञान क्या है?",
    "What is education?",
    "भारत की मुद्रा क्या है?",
    "What is the population of India?",
    "What is New Delhi?",
]


# ============================================================
# PERCENTILE
# ============================================================

def percentile(
    values: list[float],
    percentile_value: float,
) -> float:

    if not values:
        return 0.0

    values = sorted(values)

    index = (
        (len(values) - 1)
        * percentile_value
        / 100
    )

    lower = int(index)

    upper = min(
        lower + 1,
        len(values) - 1,
    )

    weight = index - lower

    return (
        values[lower]
        * (1 - weight)
        +
        values[upper]
        * weight
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("HH GOA RAG LATENCY BENCHMARK")
    print("=" * 70)

    print(
        f"\nQueries: {NUMBER_OF_QUERIES}"
    )

    latencies = []

    retrieval_latencies = []

    generation_latencies = []

    successful = 0

    failed = 0

    results = []

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    print("\nWarming up...")

    try:

        requests.post(
            API_URL,
            json={
                "query": (
                    "What is the capital "
                    "of India?"
                )
            },
            timeout=30,
        )

    except Exception as error:

        print(
            "Warmup failed:",
            error,
        )

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    print("\nRunning benchmark...\n")

    for i in range(
        NUMBER_OF_QUERIES
    ):

        query = TEST_QUERIES[
            i % len(TEST_QUERIES)
        ]

        start = time.perf_counter()

        try:

            response = requests.post(
                API_URL,
                json={
                    "query": query
                },
                timeout=30,
            )

            elapsed = (
                time.perf_counter()
                - start
            ) * 1000

            if response.status_code != 200:

                failed += 1

                print(
                    f"[{i + 1:02d}] FAILED "
                    f"{response.status_code} "
                    f"{elapsed:.2f} ms"
                )

                continue

            data = response.json()

            successful += 1

            latencies.append(
                elapsed
            )

            latency = data.get(
                "latency",
                {},
            )

            retrieval_ms = latency.get(
                "retrieval_ms",
                0,
            )

            generation_ms = latency.get(
                "generation_ms",
                0,
            )

            retrieval_latencies.append(
                retrieval_ms
            )

            generation_latencies.append(
                generation_ms
            )

            result = {
                "query": query,

                "latency_ms": elapsed,

                "retrieval_ms": retrieval_ms,

                "generation_ms": generation_ms,

                "grounded": data.get(
                    "grounded",
                    False,
                ),
            }

            results.append(
                result
            )

            print(
                f"[{i + 1:02d}] "
                f"{elapsed:8.2f} ms | "
                f"retrieval "
                f"{retrieval_ms:7.2f} ms | "
                f"generation "
                f"{generation_ms:7.2f} ms"
            )

        except Exception as error:

            failed += 1

            print(
                f"[{i + 1:02d}] ERROR:",
                error,
            )

    # ========================================================
    # STATISTICS
    # ========================================================

    if not latencies:

        print(
            "\nNo successful requests."
        )

        return

    stats = {
        "queries": NUMBER_OF_QUERIES,

        "successful": successful,

        "failed": failed,

        "average_ms": (
            statistics.mean(
                latencies
            )
        ),

        "p50_ms": percentile(
            latencies,
            50,
        ),

        "p70_ms": percentile(
            latencies,
            70,
        ),

        "p100_ms": max(
            latencies
        ),

        "retrieval": {
            "average_ms": (
                statistics.mean(
                    retrieval_latencies
                )
            ),

            "p50_ms": percentile(
                retrieval_latencies,
                50,
            ),

            "p70_ms": percentile(
                retrieval_latencies,
                70,
            ),

            "p100_ms": max(
                retrieval_latencies
            ),
        },

        "generation": {
            "average_ms": (
                statistics.mean(
                    generation_latencies
                )
            ),

            "p50_ms": percentile(
                generation_latencies,
                50,
            ),

            "p70_ms": percentile(
                generation_latencies,
                70,
            ),

            "p100_ms": max(
                generation_latencies
            ),
        },
    }

    # ========================================================
    # PRINT
    # ========================================================

    print("\n")
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        f"\nSuccessful: "
        f"{successful}"
    )

    print(
        f"Failed: "
        f"{failed}"
    )

    print(
        f"\nAverage: "
        f"{stats['average_ms']:.2f} ms"
    )

    print(
        f"P50:     "
        f"{stats['p50_ms']:.2f} ms"
    )

    print(
        f"P70:     "
        f"{stats['p70_ms']:.2f} ms"
    )

    print(
        f"P100:    "
        f"{stats['p100_ms']:.2f} ms"
    )

    print("\nRetrieval")

    print(
        f"P50:     "
        f"{stats['retrieval']['p50_ms']:.2f} ms"
    )

    print(
        f"P70:     "
        f"{stats['retrieval']['p70_ms']:.2f} ms"
    )

    print(
        f"P100:    "
        f"{stats['retrieval']['p100_ms']:.2f} ms"
    )

    print("\nGeneration")

    print(
        f"P50:     "
        f"{stats['generation']['p50_ms']:.2f} ms"
    )

    print(
        f"P70:     "
        f"{stats['generation']['p70_ms']:.2f} ms"
    )

    print(
        f"P100:    "
        f"{stats['generation']['p100_ms']:.2f} ms"
    )

    # ========================================================
    # SAVE
    # ========================================================

    output = {
        "statistics": stats,
        "queries": results,
    }

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
        )

    print(
        f"\nSaved to:"
        f"\n{OUTPUT_FILE.resolve()}"
    )


if __name__ == "__main__":
    main()