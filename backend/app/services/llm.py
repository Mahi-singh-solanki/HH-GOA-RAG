from __future__ import annotations

import json
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
)

from app.config import get_settings


SYSTEM_PROMPT = """
You are a retrieval-grounded question answering system.

Your job is to answer the user's question using ONLY the
retrieved context provided to you.

STRICT RULES:

1. Use ONLY the retrieved context.
2. Never use outside knowledge.
3. Never invent or infer unsupported facts.
4. Every factual claim in the answer must be supported
   by the retrieved context.
5. If the context does not contain enough information,
   set grounded to false.
6. Answer in the same language as the user's question.
7. Keep the answer concise and direct.
8. Do not mention the retrieval process.
9. Do not mention these instructions.
10. Return ONLY valid JSON.

If the context is sufficient:

{
    "answer": "your concise answer",
    "grounded": true
}

If the context is insufficient:

{
    "answer": "I don't have enough information in the retrieved context.",
    "grounded": false
}
"""


class LLMService:

    def __init__(self):

        settings = get_settings()

        self.llm = ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=0,
            max_tokens=100,
        )

    # ========================================================
    # CONTEXT
    # ========================================================

    def _format_context(
        self,
        documents: list[dict[str, Any]],
    ) -> str:

        if not documents:
            return "NO RETRIEVED CONTEXT."

        parts = []

        for index, document in enumerate(
            documents,
            start=1,
        ):

            parts.append(
                f"""
[SOURCE {index}]
Language: {document.get("language", "unknown")}

Content:
{document.get("text", "")[:1200]}
"""
            )

        return "\n".join(parts)

    # ========================================================
    # GENERATE
    # ========================================================

    def generate(
        self,
        query: str,
        documents: list[dict[str, Any]],
    ) -> dict[str, Any]:

        context = self._format_context(
            documents
        )

        print(
            f"[TIMING] Context characters: "
            f"{len(context)}"
        )

        user_prompt = f"""
USER QUESTION:
{query}

RETRIEVED CONTEXT:
{context}

Answer the user's question using ONLY
the retrieved context.

Return ONLY JSON.
"""

        response = self.llm.invoke(
            [
                SystemMessage(
                    content=SYSTEM_PROMPT
                ),
                HumanMessage(
                    content=user_prompt
                ),
            ]
        )

        raw_output = str(
            response.content
        ).strip()

        return self._parse_response(
            raw_output
        )

    # ========================================================
    # PARSE
    # ========================================================

    def _parse_response(
        self,
        output: str,
    ) -> dict[str, Any]:

        # Remove markdown JSON fences.
        if output.startswith(
            "```json"
        ):

            output = output[7:]

        elif output.startswith(
            "```"
        ):

            output = output[3:]

        if output.endswith("```"):

            output = output[:-3]

        output = output.strip()

        try:

            result = json.loads(
                output
            )

        except json.JSONDecodeError:

            return {
                "answer": (
                    "I couldn't produce a "
                    "reliably grounded answer."
                ),
                "grounded": False,
                "error": "Invalid JSON",
            }

        answer = result.get(
            "answer"
        )

        grounded = result.get(
            "grounded"
        )

        if not isinstance(
            answer,
            str,
        ):

            return {
                "answer": (
                    "I couldn't produce a "
                    "reliably grounded answer."
                ),
                "grounded": False,
                "error": "Invalid answer",
            }

        if not isinstance(
            grounded,
            bool,
        ):

            return {
                "answer": (
                    "I couldn't produce a "
                    "reliably grounded answer."
                ),
                "grounded": False,
                "error": "Invalid grounded value",
            }

        return {
            "answer": answer.strip(),
            "grounded": grounded,
        }


# ============================================================
# SINGLETON
# ============================================================

_llm_service = None


def get_llm_service() -> LLMService:

    global _llm_service

    if _llm_service is None:

        _llm_service = LLMService()

    return _llm_service