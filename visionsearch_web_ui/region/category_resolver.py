"""Conservative query-to-label resolution for region ranking."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from math import isfinite


def _normalize_label(label: str) -> str:
    return re.sub(r"\s+", " ", label.strip().lower().replace("_", " "))


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", _normalize_label(text))


def _contains_phrase(query_tokens: list[str], phrase: str) -> bool:
    phrase_tokens = _tokens(phrase)
    if not phrase_tokens or len(phrase_tokens) > len(query_tokens):
        return False
    width = len(phrase_tokens)
    return any(
        query_tokens[start : start + width] == phrase_tokens
        for start in range(len(query_tokens) - width + 1)
    )


def _category_for_label(
    label: str,
    aliases: Mapping[str, set[str]],
) -> str:
    normalized = _normalize_label(label)
    for category, members in aliases.items():
        if normalized == _normalize_label(category) or any(
            normalized == _normalize_label(member) for member in members
        ):
            return _normalize_label(category)
    return normalized


def resolve_query_category(
    query: str | None,
    known_labels: Iterable[str],
    aliases: Mapping[str, set[str]],
) -> str | None:
    """Resolve one unambiguous query category present in detector metadata.

    Labels outside the observed detector vocabulary are deliberately ignored;
    unsupported or ambiguous queries therefore retain semantic-only ranking.
    """
    if not query or not query.strip():
        return None

    query_tokens = _tokens(query)
    if not query_tokens:
        return None

    resolved: set[str] = set()
    observed_labels = {
        _normalize_label(label)
        for label in known_labels
        if label and _normalize_label(label) not in ("whole", "")
        and not _normalize_label(label).startswith("grid ")
    }

    for label in observed_labels:
        category = _category_for_label(label, aliases)
        phrases = {label, category}
        phrases.update(
            member
            for member in aliases.get(category, set())
        )
        if any(_contains_phrase(query_tokens, phrase) for phrase in phrases):
            resolved.add(category)

    return next(iter(resolved)) if len(resolved) == 1 else None


def category_adjustment(
    row: Mapping[str, object],
    query_category: str | None,
    known_labels: Iterable[str],
    aliases: Mapping[str, set[str]],
    *,
    min_confidence: float,
    bonus: float,
    penalty: float,
) -> float:
    """Return a soft adjustment only for confident, known YOLO detections."""
    if query_category is None or row.get("crop_type") != "yolo":
        return 0.0

    label = row.get("class_label")
    confidence = row.get("confidence")
    if not isinstance(label, str) or not label.strip():
        return 0.0
    if not isinstance(confidence, (int, float)) or not isfinite(confidence):
        return 0.0
    if confidence < min_confidence:
        return 0.0

    normalized_label = _normalize_label(label)
    if normalized_label in {"unknown", "other", "none", "n a"}:
        return 0.0
    normalized_known_labels = {
        _normalize_label(known_label)
        for known_label in known_labels
        if known_label
    }
    if (
        normalized_label not in normalized_known_labels
        or re.fullmatch(r"\d+", normalized_label)
    ):
        return 0.0

    if _category_for_label(normalized_label, aliases) == query_category:
        return float(bonus)
    return -float(penalty)
