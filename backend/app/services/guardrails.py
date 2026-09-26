from __future__ import annotations

import re


BLOCKED_PATTERNS = [
    r"ignore previous instructions",
    r"ignore all previous instructions",
    r"system prompt",
    r"reveal your prompt",
    r"developer message",
    r"jailbreak",
]


def validate_query(
    query: str,
) -> dict:

    if not query:

        return {
            "valid": False,
            "reason": "Query is empty.",
        }

    query = query.strip()

    if not query:

        return {
            "valid": False,
            "reason": "Query is empty.",
        }

    if len(query) > 2000:

        return {
            "valid": False,
            "reason": (
                "Query is too long."
            ),
        }

    lowered = query.lower()

    for pattern in BLOCKED_PATTERNS:

        if re.search(
            pattern,
            lowered,
        ):

            return {
                "valid": False,
                "reason": (
                    "The query contains "
                    "an unsupported instruction."
                ),
            }

    return {
        "valid": True,
    }