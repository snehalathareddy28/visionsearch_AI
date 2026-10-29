
"""Separate query terms into objects, colors, actions, and scene terms."""
import re
from collections import OrderedDict


COLORS = {
    "red", "blue", "green", "yellow", "orange", "purple", "pink",
    "black", "white", "gray", "grey", "brown", "beige", "gold",
    "silver", "teal", "navy", "maroon", "cyan", "magenta",
}

SCENES = {
    "beach", "forest", "city", "mountain", "ocean", "park", "room",
    "indoor", "outdoor", "sky", "desert", "waterfall", "street",
    "garden", "office", "kitchen", "living room", "bedroom", "classroom",
}

ACTIONS = {
    "sitting", "sit", "standing", "stand", "running", "run", "walking",
    "walk", "jumping", "jump", "sleeping", "sleep", "eating", "eat",
    "playing", "play", "flying", "fly", "swimming", "swim", "driving",
    "drive", "riding", "ride", "lying", "lie", "barking", "bark",
}


SCENE_ELEMENTS = {
    "grass", "tree", "trees", "cloud", "clouds",
    "road", "path", "sand", "snow", "water", "field",
    "building", "buildings", "table", "chair", "chairs",
    "carpet",
}

STOPWORDS = {
    "a", "an", "the", "and", "or", "with", "in", "on", "at", "of",
    "for", "to", "from", "by", "is", "are", "it", "this", "that",
    "these", "those", "be", "as", "into", "our", "your",
    "photo", "picture", "image", "images",
}


def _normalize_token(token: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", token.lower()).strip()


def extract_attributes(query: str) -> dict:
    """Separate query terms into objects, colors, actions, and scene terms."""

    if not isinstance(query, str):
        raise TypeError("Query must be a string")

    normalized = query.strip()
    if not normalized:
        return {
            "query": "",
            "colors": [],
            "scenes": [],
            "objects": [],
            "actions": [],
            "scene_elements": [],
            "keywords": [],
            "modifiers": {},
        }

    tokens = [
        _normalize_token(part)
        for part in re.split(r"\s+", normalized)
        if _normalize_token(part)
    ]

    lower_query = normalized.lower()

    matched_colors = list(OrderedDict.fromkeys(
        token for token in tokens if token in COLORS
    ))

    matched_scenes = list(OrderedDict.fromkeys(
        token for token in tokens if token in SCENES
    ))

    for scene in sorted(SCENES, key=len, reverse=True):
        if scene in lower_query and scene not in matched_scenes:
            matched_scenes.append(scene)

    matched_actions = list(OrderedDict.fromkeys(
        token for token in tokens if token in ACTIONS
    ))

    matched_scene_elements = list(OrderedDict.fromkeys(
        token for token in tokens if token in SCENE_ELEMENTS
    ))

    excluded_terms = (
        STOPWORDS | COLORS | SCENES | ACTIONS | SCENE_ELEMENTS
    )

    objects = list(OrderedDict.fromkeys(
        token for token in tokens
        if token not in excluded_terms and len(token) > 2
    ))

    keywords = list(OrderedDict.fromkeys(
        objects + matched_colors + matched_scenes
        + matched_actions + matched_scene_elements
    ))

    modifiers = {
        "color": matched_colors,
        "scene": matched_scenes,
        "action": matched_actions,
    }

    if not objects and not matched_colors and not matched_scenes:
        modifiers["general"] = [normalized]

    return {
        "query": normalized,
        "colors": matched_colors,
        "scenes": matched_scenes,
        "objects": objects,
        "actions": matched_actions,
        "scene_elements": matched_scene_elements,
        "keywords": keywords,
        "modifiers": modifiers,
    }
