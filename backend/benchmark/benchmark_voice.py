from __future__ import annotations

import statistics
import time
from pathlib import Path

import requests


# ============================================================
# CONFIG
# ============================================================

API_URL = "http://127.0.0.1:8000/voice"

AUDIO_FILE = Path("test_audio.wav")

NUMBER_OF_QUERIES = 5


# ============================================================
# PERCENTILE
# ============================================================

def percentile(
    values: list[float],
    p: float,
) -> float:

    if not values:
        return 0.0

    values = sorted(values)

    index = (
        (len(values) - 1)
        * p
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
    print("HH GOA VOICE RAG LATENCY BENCHMARK")
    print("=" * 70)

    if not AUDIO_FILE.exists():

        raise FileNotFoundError(
            f"Audio file not found:\n"
            f"{AUDIO_FILE.resolve()}"
        )

    audio_bytes = AUDIO_FILE.read_bytes()

    print(
        f"\nAudio: {AUDIO_FILE}"
    )

    print(
        f"Size: {len(audio_bytes) / 1024:.2f} KB"
    )

    print(
        f"Requests: {NUMBER_OF_QUERIES}"
    )

    total_latencies = []
    stt_latencies = []
    rag_latencies = []

    successful = 0
    failed = 0

    # ========================================================
    # WARMUP
    # ========================================================

    print("\nWarming up...")

    try:

        requests.post(
            API_URL,
            files={
                "audio": (
                    AUDIO_FILE.name,
                    audio_bytes,
                    "audio/wav",
                )
            },
            timeout=60,
        )

    except Exception as error:

        print(
            "Warmup failed:",
            error,
        )

    # ========================================================
    # BENCHMARK
    # ========================================================

    print("\nRunning benchmark...\n")

    for i in range(
        NUMBER_OF_QUERIES
    ):

        start = time.perf_counter()

        try:

            response = requests.post(
                API_URL,
                files={
                    "audio": (
                        AUDIO_FILE.name,
                        audio_bytes,
                        "audio/wav",
                    )
                },
                timeout=60,
            )

            total_ms = (
                time.perf_counter()
                - start
            ) * 1000

            if response.status_code != 200:

                failed += 1

                print(
                    f"[{i + 1:02d}] FAILED "
                    f"{response.status_code} "
                    f"{total_ms:.2f} ms"
                )

                continue

            data = response.json()

            latency = data.get(
                "latency",
                {},
            )

            stt_ms = float(
                latency.get(
                    "stt_ms",
                    0,
                )
            )

            rag_ms = float(
                latency.get(
                    "rag_ms",
                    latency.get(
                        "total_ms",
                        0,
                    ),
                )
            )

            successful += 1

            total_latencies.append(
                total_ms
            )

            stt_latencies.append(
                stt_ms
            )

            rag_latencies.append(
                rag_ms
            )

            print(
                f"[{i + 1:02d}] "
                f"TOTAL={total_ms:8.2f} ms | "
                f"STT={stt_ms:8.2f} ms | "
                f"RAG={rag_ms:8.2f} ms"
            )

        except Exception as error:

            failed += 1

            print(
                f"[{i + 1:02d}] ERROR:",
                error,
            )

    # ========================================================
    # RESULTS
    # ========================================================

    if not total_latencies:

        print(
            "\nNo successful requests."
        )

        return

    print("\n")
    print("=" * 70)
    print("VOICE RESULTS")
    print("=" * 70)

    print(
        f"\nSuccessful: "
        f"{successful}"
    )

    print(
        f"Failed: "
        f"{failed}"
    )

    # ========================================================
    # TOTAL
    # ========================================================

    print("\nTOTAL VOICE PIPELINE")

    print(
        f"Average: "
        f"{statistics.mean(total_latencies):.2f} ms"
    )

    print(
        f"P50:     "
        f"{percentile(total_latencies, 50):.2f} ms"
    )

    print(
        f"P70:     "
        f"{percentile(total_latencies, 70):.2f} ms"
    )

    print(
        f"P100:    "
        f"{max(total_latencies):.2f} ms"
    )

    # ========================================================
    # STT
    # ========================================================

    print("\nSPEECH-TO-TEXT")

    print(
        f"Average: "
        f"{statistics.mean(stt_latencies):.2f} ms"
    )

    print(
        f"P50:     "
        f"{percentile(stt_latencies, 50):.2f} ms"
    )

    print(
        f"P70:     "
        f"{percentile(stt_latencies, 70):.2f} ms"
    )

    print(
        f"P100:    "
        f"{max(stt_latencies):.2f} ms"
    )

    # ========================================================
    # RAG
    # ========================================================

    print("\nRAG")

    print(
        f"Average: "
        f"{statistics.mean(rag_latencies):.2f} ms"
    )

    print(
        f"P50:     "
        f"{percentile(rag_latencies, 50):.2f} ms"
    )

    print(
        f"P70:     "
        f"{percentile(rag_latencies, 70):.2f} ms"
    )

    print(
        f"P100:    "
        f"{max(rag_latencies):.2f} ms"
    )

    # ========================================================
    # TARGET
    # ========================================================

    p50_total = percentile(
        total_latencies,
        50,
    )

    print("\nTARGET")

    print(
        f"P50 < 200 ms: "
        f"{'YES' if p50_total < 200 else 'NO'}"
    )

    print(
        f"Actual P50: "
        f"{p50_total:.2f} ms"
    )


if __name__ == "__main__":
    main()