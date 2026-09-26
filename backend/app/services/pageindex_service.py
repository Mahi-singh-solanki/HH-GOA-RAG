from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import get_settings


class PageIndexService:

    def __init__(self):

        settings = get_settings()

        self.settings = settings

        self.enabled = (
            settings.pageindex_enabled
        )

        self.registry_path = (
            Path(settings.pageindex_storage_path)
            / "registry.json"
        )

        self.registry_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.registry = self._load_registry()

        self.client = None

        if self.enabled:
            self.client = self._create_client()

    def _load_registry(self) -> dict[str, Any]:

        if not self.registry_path.exists():

            return {
                "documents": {}
            }

        try:

            return json.loads(
                self.registry_path.read_text(
                    encoding="utf-8"
                )
            )

        except Exception:

            return {
                "documents": {}
            }

    def _save_registry(self):

        self.registry_path.write_text(
            json.dumps(
                self.registry,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _create_client(self):

        try:

            from pageindex import (
                PageIndexClient
            )

        except ImportError as exc:

            raise RuntimeError(
                "PageIndex is not installed. "
                "Run: pip install -U pageindex"
            ) from exc

        mode = (
            self.settings.pageindex_mode
            .strip()
            .lower()
        )

        if mode == "auto":

            if self.settings.pageindex_api_key:
                mode = "cloud"
            else:
                mode = "local"

        chat_config = {
            "model": (
                self.settings
                .pageindex_chat_model
            ),
            "backend": {
                "api_key": (
                    self.settings
                    .groq_api_key
                )
            },
        }

        if mode == "cloud":

            if not self.settings.pageindex_api_key:

                raise RuntimeError(
                    "PAGEINDEX_MODE=cloud but "
                    "PAGEINDEX_API_KEY is missing."
                )

            return PageIndexClient(
                index="cloud",
                chat=chat_config,
            )

        index_config = {
            "model": (
                self.settings
                .pageindex_index_model
            ),
            "storage_path": (
                self.settings
                .pageindex_storage_path
            ),
            "backend": {
                "api_key": (
                    self.settings
                    .groq_api_key
                )
            },
        }

        return PageIndexClient(
            index=index_config,
            chat=chat_config,
        )

    @property
    def mode(self) -> str:

        mode = (
            self.settings.pageindex_mode
            .strip()
            .lower()
        )

        if mode == "auto":

            return (
                "cloud"
                if self.settings.pageindex_api_key
                else "local"
            )

        return mode

    def should_use_for_pdf(
        self,
        processed_document: dict[str, Any],
    ) -> bool:

        if not self.enabled:
            return False

        if processed_document.get(
            "file_type"
        ) != "pdf":
            return False

        if self.mode == "cloud":
            return True

        processing = (
            processed_document
            .get("processing", {})
        )

        ocr_used = processing.get(
            "ocr_used",
            False,
        )

        return not ocr_used

    def index_document(
        self,
        file_path: Path,
        original_filename: str,
    ) -> dict[str, Any]:

        if not self.enabled:
            raise RuntimeError(
                "PageIndex is disabled."
            )

        if self.client is None:
            raise RuntimeError(
                "PageIndex client is unavailable."
            )

        result = self.client.submit_document(
            str(file_path),
            wait=True,
        )

        doc_id = result["doc_id"]

        stored_name = result.get(
            "name",
            file_path.name,
        )

        self.registry.setdefault(
            "documents",
            {},
        )

        self.registry["documents"][
            doc_id
        ] = {
            "doc_id": doc_id,
            "stored_filename": stored_name,
            "original_filename": (
                original_filename
            ),
            "file_path": str(file_path),
            "backend": "pageindex",
            "mode": self.mode,
        }

        self._save_registry()

        return {
            "index_type": "pageindex",
            "pageindex_doc_id": doc_id,
            "pageindex_mode": self.mode,
            "stored_filename": stored_name,
        }

    def list_document_ids(self) -> list[str]:

        return list(
            self.registry
            .get("documents", {})
            .keys()
        )

    def _extract_citations(
        self,
        answer: str,
    ) -> list[dict[str, Any]]:

        if self.client is not None:

            try:

                citations = (
                    self.client.get_citations(
                        answer
                    )
                )

                if citations:
                    return citations

            except Exception:
                pass

        citations = []

        pattern = re.compile(
            r'<cite\s+'
            r'doc=["\']([^"\']+)["\']\s+'
            r'page=["\'](\d+)["\']'
            r'(?:\s+block=["\']([^"\']+)["\'])?'
            r'\s*/?>'
        )

        for match in pattern.finditer(
            answer
        ):

            citations.append(
                {
                    "document": match.group(1),
                    "page": int(
                        match.group(2)
                    ),
                    "block_id": (
                        match.group(3)
                        or None
                    ),
                }
            )

        return citations

    def retrieve(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
        max_sources: int = 8,
    ) -> list[dict[str, Any]]:

        if not self.enabled:
            return []

        if self.client is None:
            return []

        doc_ids = self.list_document_ids()

        if not doc_ids:
            return []

        messages = []

        if history:
            messages.extend(
                history[-8:]
            )

        messages.append(
            {
                "role": "user",
                "content": query,
            }
        )

        try:

            answer = self.client.chat(
                messages,
                doc_id=doc_ids,
                citations=True,
            )

        except Exception:

            return []

        citations = (
            self._extract_citations(
                answer
            )
        )

        results = []

        seen = set()

        for citation in citations:

            doc_id = citation.get(
                "doc_id"
            )

            document_name = citation.get(
                "document"
            )

            page_number = citation.get(
                "page"
            )

            if not page_number:
                continue

            key = (
                doc_id,
                document_name,
                page_number,
            )

            if key in seen:
                continue

            seen.add(key)

            if not doc_id:

                for candidate_id, meta in (
                    self.registry
                    .get("documents", {})
                    .items()
                ):

                    if (
                        meta.get(
                            "stored_filename"
                        )
                        == document_name
                    ):

                        doc_id = candidate_id
                        break

            if not doc_id:
                continue

            try:

                pages = (
                    self.client
                    .get_page_content(
                        doc_id,
                        str(page_number),
                    )
                )

            except Exception:

                continue

            if not pages:
                continue

            page = pages[0]

            metadata = (
                self.registry
                .get("documents", {})
                .get(
                    doc_id,
                    {},
                )
            )

            results.append(
                {
                    "chunk_id": (
                        f"pageindex:"
                        f"{doc_id}:"
                        f"{page_number}"
                    ),
                    "document_id": doc_id,
                    "filename": (
                        metadata.get(
                            "original_filename"
                        )
                        or document_name
                    ),
                    "page": page_number,
                    "section": None,
                    "text": page.get(
                        "markdown",
                        "",
                    ),
                    "source_type": "pdf",
                    "retrieval_backend": (
                        "pageindex"
                    ),
                    "score": 1.0,
                    "pageindex_doc_id": doc_id,
                    "citation": {
                        "document": document_name,
                        "page": page_number,
                        "block_id": citation.get(
                            "block_id"
                        ),
                    },
                }
            )

            if len(results) >= max_sources:
                break

        return results


@lru_cache
def get_pageindex_service():
    return PageIndexService()