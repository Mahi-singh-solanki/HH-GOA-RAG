from __future__ import annotations

import re


# ============================================================
# CONFIG
# ============================================================

MAX_QUERY_LENGTH = 500


# Queries that are obviously not useful for this RAG.
# Keep this lightweight because it runs on every request.
BLOCKED_PATTERNS = [
    r"\b(ignore|disregard)\s+(all|the|previous)\s+instructions\b",
    r"\bsystem\s+prompt\b",
    r"\breveal\s+(your|the)\s+(prompt|instructions)\b",
]


# ============================================================
# RESULT
# ============================================================

class GuardrailResult:

    def __init__(
        self,
        allowed: bool,
        reason: str = "",
    ):

        self.allowed = allowed
        self.reason = reason


# ============================================================
# GUARDRAIL
# ============================================================

class GuardrailService:

    def check(
        self,
        query: str,
    ) -> GuardrailResult:

        # ----------------------------------------------------
        # Empty
        # ----------------------------------------------------

        if not query or not query.strip():

            return GuardrailResult(
                allowed=False,
                reason="empty_query",
            )

        query = query.strip()

        # ----------------------------------------------------
        # Length
        # ----------------------------------------------------

        if len(query) > MAX_QUERY_LENGTH:

            return GuardrailResult(
                allowed=False,
                reason="query_too_long",
            )

        # ----------------------------------------------------
        # Repeated garbage characters
        # ----------------------------------------------------

        if re.fullmatch(
            r"[\W_]+",
            query,
            flags=re.UNICODE,
        ):

            return GuardrailResult(
                allowed=False,
                reason="invalid_query",
            )

        # ----------------------------------------------------
        # Prompt injection patterns
        # ----------------------------------------------------

        query_lower = query.lower()

        for pattern in BLOCKED_PATTERNS:

            if re.search(
                pattern,
                query_lower,
            ):

                return GuardrailResult(
                    allowed=False,
                    reason="prompt_injection",
                )

        # ----------------------------------------------------
        # Otherwise allow
        # ----------------------------------------------------

        return GuardrailResult(
            allowed=True,
        )


# ============================================================
# SINGLETON
# ============================================================

_guardrail_service = None


def get_guardrail_service():

    global _guardrail_service

    if _guardrail_service is None:

        _guardrail_service = (
            GuardrailService()
        )

    return _guardrail_service