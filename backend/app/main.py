from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import (
    CORSMiddleware,
)
from fastapi.openapi.utils import get_openapi

from app.config import get_settings
from app.rag import (
    RAGRequest,
    RAGResponse,
)
from app.services.document_indexer import (
    get_document_indexer,
)
from app.services.document_processor import (
    SUPPORTED_EXTENSIONS,
    ensure_directories,
    process_document,
)
from app.services.rag import (
    get_rag_service,
)
from app.services.stt import (
    get_stt_service,
)


settings = get_settings()

app = FastAPI(
    title="Intelligent Document Intelligence",
    version="2.0.0",
    description=(
        "Multi-format document processing, "
        "PageIndex + Qdrant retrieval, "
        "OCR and evidence-grounded answers."
    ),
)


def custom_openapi():

    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    schemas = (
        schema
        .get("components", {})
        .get("schemas", {})
    )

    for schema_item in schemas.values():

        properties = (
            schema_item
            .get("properties", {})
        )

        for prop in properties.values():

            if "items" in prop:

                items = prop["items"]

                if (
                    items.get(
                        "contentMediaType"
                    )
                    == "application/octet-stream"
                ):

                    items.pop(
                        "contentMediaType",
                        None,
                    )

                    items["format"] = "binary"

            elif (
                prop.get(
                    "contentMediaType"
                )
                == "application/octet-stream"
            ):

                prop.pop(
                    "contentMediaType",
                    None,
                )

                prop["format"] = "binary"

    app.openapi_schema = schema

    return schema


app.openapi = custom_openapi


cors_origins = [
    origin.strip()
    for origin in settings.cors_origins.split(",")
    if origin.strip()
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():

    return {
        "status": "ok",
        "service": (
            "Intelligent Document "
            "Intelligence"
        ),
        "version": "2.0.0",
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "pageindex_enabled": (
            settings.pageindex_enabled
        ),
        "pageindex_mode": (
            settings.pageindex_mode
        ),
        "qdrant": (
            settings.qdrant_url
        ),
    }


@app.post(
    "/rag",
    response_model=RAGResponse,
)
def rag(
    request: RAGRequest,
):

    try:

        service = get_rag_service()

        return service.run(
            query=request.query,
            language=request.language,
            conversation_id=(
                request.conversation_id
            ),
        )

    except Exception as exc:

        print(
            "RAG ERROR:",
            repr(exc),
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


@app.post("/voice")
async def voice_rag(
    audio: UploadFile = File(...),
    conversation_id: str = "default",
):

    start = time.perf_counter()

    if not audio.filename:

        raise HTTPException(
            status_code=400,
            detail=(
                "Audio filename is missing."
            ),
        )

    allowed_extensions = {
        ".wav",
        ".mp3",
        ".m4a",
        ".webm",
        ".ogg",
    }

    extension = Path(
        audio.filename
    ).suffix.lower()

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported audio format."
            ),
        )

    audio_bytes = await audio.read()

    if not audio_bytes:

        raise HTTPException(
            status_code=400,
            detail="Empty audio file.",
        )

    stt_start = time.perf_counter()

    try:

        stt = get_stt_service()

        transcription = stt.transcribe(
            audio_bytes=audio_bytes,
            filename=audio.filename,
        )

    except Exception as exc:

        print(
            "STT ERROR:",
            repr(exc),
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "Speech-to-text failed."
            ),
        )

    stt_ms = (
        time.perf_counter()
        - stt_start
    ) * 1000

    transcript = (
        transcription
        .get("text", "")
        .strip()
    )

    if not transcript:

        raise HTTPException(
            status_code=422,
            detail="No speech detected.",
        )

    rag_start = time.perf_counter()

    try:

        rag_service = (
            get_rag_service()
        )

        result = rag_service.run(
            query=transcript,
            conversation_id=(
                conversation_id
            ),
        )

    except Exception as exc:

        print(
            "VOICE RAG ERROR:",
            repr(exc),
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    rag_ms = (
        time.perf_counter()
        - rag_start
    ) * 1000

    total_ms = (
        time.perf_counter()
        - start
    ) * 1000

    result["transcript"] = transcript

    result["latency"]["stt_ms"] = stt_ms
    result["latency"]["rag_ms"] = rag_ms
    result["latency"]["total_voice_ms"] = (
        total_ms
    )

    return result


@app.post("/upload")
async def upload_documents(
    files: Annotated[
        list[UploadFile],
        File(...),
    ],
):

    ensure_directories()

    indexer = (
        get_document_indexer()
    )

    results = []

    for file in files:

        if not file.filename:
            continue

        extension = (
            Path(
                file.filename
            ).suffix.lower()
        )

        if extension not in SUPPORTED_EXTENSIONS:

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported file type: "
                    f"{extension}. "
                    f"Supported: "
                    f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
                ),
            )

        content = await file.read()

        max_bytes = (
            settings.max_upload_size_mb
            * 1024
            * 1024
        )

        if len(content) > max_bytes:

            raise HTTPException(
                status_code=413,
                detail=(
                    f"File exceeds "
                    f"{settings.max_upload_size_mb} MB."
                ),
            )

        document_id = uuid4().hex

        safe_name = Path(
            file.filename
        ).name

        stored_filename = (
            f"{document_id}_{safe_name}"
        )

        file_path = (
            Path(settings.documents_dir)
            / stored_filename
        )

        try:

            file_path.write_bytes(
                content
            )

            processed = process_document(
                file_path=file_path,
                original_filename=(
                    file.filename
                ),
            )

            index_result = (
                indexer.index(
                    file_path=file_path,
                    processed_document=processed,
                    original_filename=(
                        file.filename
                    ),
                )
            )

            results.append(
                {
                    "document_id": (
                        processed[
                            "document_id"
                        ]
                    ),
                    "filename": (
                        file.filename
                    ),
                    "stored_filename": (
                        stored_filename
                    ),
                    "file_type": (
                        extension.lstrip(".")
                    ),
                    "ocr_used": (
                        processed[
                            "processing"
                        ][
                            "ocr_used"
                        ]
                    ),
                    "pages": len(
                        processed[
                            "pages"
                        ]
                    ),
                    "index": index_result,
                    "status": "indexed",
                }
            )

        except Exception as exc:

            if file_path.exists():
                file_path.unlink()

            json_path = (
                file_path.with_suffix(
                    ".json"
                )
            )

            if json_path.exists():
                json_path.unlink()

            print(
                "UPLOAD ERROR:",
                repr(exc),
            )

            raise HTTPException(
                status_code=500,
                detail=(
                    f"Failed to process "
                    f"{file.filename}: "
                    f"{str(exc)}"
                ),
            )

    return {
        "status": "success",
        "documents": results,
    }