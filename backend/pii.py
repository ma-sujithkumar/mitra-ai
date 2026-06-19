from __future__ import annotations

import re


def column_matches_pii(column_name: str, pii_patterns: list[str]) -> bool:
    # A column is PII-suspect when its name matches any configured pattern.
    return any(
        re.search(pattern, column_name) is not None
        for pattern in pii_patterns
    )


def match_pii_columns(
    column_names: list[str],
    pii_patterns: list[str],
) -> list[str]:
    # Deterministic, name-based PII detection shared by the validator's PII check
    # and the metadata drop step, so the regex logic lives in one place.
    return [
        column_name
        for column_name in column_names
        if column_matches_pii(column_name=column_name, pii_patterns=pii_patterns)
    ]
