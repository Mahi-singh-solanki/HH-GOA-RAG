from __future__ import annotations

import time
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException,UploadFile, File
from pydantic import BaseModel, Field

from app.services.rag import get_rag_service

from app.services.stt import get_stt_service
# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="HH Goa Voice RAG",
    version="1.0.0",
)


# ============================================================
# REQUEST
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class RAGRequest(BaseModel):

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
    )

    language: str | None = Field(
        default=None,
        description="hi or en",
    )


# ============================================================
# RESPONSE
# ============================================================

class RAGResponse(BaseModel):

    success: bool

    answer: str

    grounded: bool

    sources: list[dict]

    latency: dict


# ============================================================
# HEALTH
# ============================================================

@app.get("/")
def root():

    return {
        "status": "ok",
        "service": "HH Goa Voice RAG",
    }


# ============================================================
# RAG
# ============================================================

@app.post(
    "/rag",
    response_model=RAGResponse,
)
def rag(
    request: RAGRequest,
):

    start = time.perf_counter()

    try:

        rag_service = (
            get_rag_service()
        )

        result = rag_service.run(
            query=request.query,
            language=request.language,
        )

        # ----------------------------------------------------
        # Add API latency
        # ----------------------------------------------------

        api_latency = (
            time.perf_counter()
            - start
        ) * 1000

        result["latency"][
            "api_ms"
        ] = api_latency

        return result

    except Exception as error:

        print(
            "RAG ERROR:",
            repr(error),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to process "
                "RAG request."
            ),
        )

@app.post("/voice")
async def voice_rag(
    audio: UploadFile = File(...)
):
    start = time.perf_counter()

    # ========================================================
    # Validate file
    # ========================================================

    if not audio.filename:
        raise HTTPException(
            status_code=400,
            detail="Audio filename is missing."
        )

    # Basic extension validation
    allowed_extensions = {
        ".wav",
        ".mp3",
        ".m4a",
        ".webm",
        ".ogg",
    }

    extension = ""

    if "." in audio.filename:
        extension = (
            "."
            + audio.filename.rsplit(
                ".",
                1
            )[1].lower()
        )

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported audio format. "
                "Use WAV, MP3, M4A, WEBM or OGG."
            )
        )

    # ========================================================
    # Read audio
    # ========================================================

    audio_bytes = await audio.read()

    if not audio_bytes:
        raise HTTPException(
            status_code=400,
            detail="Empty audio file."
        )

    # ========================================================
    # Speech-to-text
    # ========================================================

    stt_start = time.perf_counter()

    try:

        stt = get_stt_service()

        transcription = stt.transcribe(
            audio_bytes=audio_bytes,
            filename=audio.filename,
        )
        stt_ms = (
    time.perf_counter() - stt_start
) *1000

    except Exception as error:

        print(
            "STT ERROR:",
            repr(error)
        )

        raise HTTPException(
            status_code=502,
            detail="Speech-to-text failed."
        )

    stt_latency = (
        time.perf_counter()
        - stt_start
    ) * 1000

    transcript = transcription.get(
        "text",
        ""
    ).strip()

    if not transcript:

        raise HTTPException(
            status_code=422,
            detail="No speech detected."
        )

    # ========================================================
    # RAG
    # ========================================================

    rag_start = time.perf_counter()

    try:

        rag = get_rag_service()

        result = rag.run(
            query=transcript
        )
        rag_ms = (
    time.perf_counter() - rag_start
) * 1000


    except Exception as error:

        print(
            "RAG ERROR:",
            repr(error)
        )

        raise HTTPException(
            status_code=500,
            detail="RAG processing failed."
        )

    rag_latency = (
        time.perf_counter()
        - rag_start
    ) * 1000

    # ========================================================
    # Total latency
    # ========================================================

    total_latency = (
        time.perf_counter()
        - start
    ) * 1000
    print(
    f"[VOICE] "
    f"STT={stt_ms:.2f}ms | "
    f"RAG={rag_ms:.2f}ms | "
    f"TOTAL={total_latency:.2f}ms"
)

    # ========================================================
    # Response
    # ========================================================

    result["transcript"] = transcript

    result["latency"]["stt_ms"] = (
        stt_latency
    )

    result["latency"]["rag_ms"] = (
        rag_latency
    )

    result["latency"]["total_voice_ms"] = (
        total_latency
    )

    return result