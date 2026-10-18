"""Extract searchable attributes from natural-language queries."""

import re
from collections import OrderedDict


COLORS = {
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "pink",
    "black",
    "white",
    "gray",
    "grey",
    "brown",
    "beige",
    "gold",
    "silver",
    "teal",
    "navy",
    "maroon",
    "cyan",
    "magenta",
}

SCENES = {
    "beach",
    "forest",
    "city",
    "mountain",
    "ocean",
    "park",
    "room",
    "indoor",
    "outdoor",
    "sky",
    "desert",
    "waterfall",
    "street",
    "garden",
    "office",
    "kitchen",
    "living room",
    "bedroom",
    "classroom",
}

STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "with",
    "in",
    "on",
    "at",
    "of",
    "for",
    "to",
    "from",
    "by",
    "is",
    "are",
    "it",
    "this",
    "that",
    "these",
    "those",
    "be",
    "as",
    "into",
    "our",
    "your",
}


def _normalize_token(token: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", token.lower()).strip()


def extract_attributes(query: str) -> dict:
    """Return a small, query-oriented attribute dictionary.

    The structure is intentionally lightweight so callers can use it without
    needing the full search backend or domain-specific NLP libraries.
    """
    if not isinstance(query, str):
        raise TypeError("Query must be a string")

    normalized = query.strip()
    if not normalized:
        return {
            "query": "",
            "colors": [],
            "scenes": [],
            "objects": [],
            "keywords": [],
            "modifiers": {},
        }

    tokens = [
        _normalize_token(part)
        for part in re.split(r"\s+", normalized)
        if _normalize_token(part)
    ]
    lower_query = normalized.lower()

    matched_colors = sorted({token for token in tokens if token in COLORS}, key=lambda x: tokens.index(x))
    matched_scenes = sorted({token for token in tokens if token in SCENES}, key=lambda x: tokens.index(x))

    phrase_scenes = []
    for scene in sorted(SCENES, key=len, reverse=True):
        if scene in lower_query:
            phrase_scenes.append(scene)
    matched_scenes = sorted(set(matched_scenes) | set(phrase_scenes), key=lambda value: lower_query.index(value) if value in lower_query else len(lower_query))

    objects = []
    for token in tokens:
        if token in STOPWORDS or token in COLORS or token in SCENES:
            continue
        if len(token) <= 2:
            continue
        objects.append(token)

    keywords = list(OrderedDict.fromkeys(objects + matched_colors + matched_scenes))

    modifiers = {
        "color": matched_colors,
        "scene": matched_scenes,
    }

    if not objects and not matched_colors and not matched_scenes:
        modifiers["general"] = [normalized]

    return {
        "query": normalized,
        "colors": matched_colors,
        "scenes": matched_scenes,
        "objects": objects,
        "keywords": keywords,
        "modifiers": modifiers,
    }
