
"""FAISS retrieval and attribute-aware ranking for CLIP embeddings."""

from pathlib import Path

import faiss
import numpy as np

from services.attribute_service import extract_attributes
from models.clip_model import ClipModel, load_clip_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = PROJECT_ROOT / "index" / "faiss.index"
PATHS_PATH = PROJECT_ROOT / "index" / "image_paths.txt"

DEFAULT_SIMILARITY_THRESHOLD = 0.20
CATEGORY_BONUS = 0.04
ATTRIBUTE_WEIGHT = 0.30

# Map query objects to related dataset folder names.
CATEGORY_ALIASES = {
    "flower": {
        "flower",
        "flowers",
        "lotus",
        "sunflower",
        "water",
        "lilly",
        "lily",
        "water_lilly",
        "waterlily",
        "water_lily",
    },
    "car": {
        "car",
        "cars",
        "car_side",
        "automobile",
        "vehicle",
        "vehicles",
    },
    "dog": {"dog", "dogs"},
    "cat": {"cat", "cats"},
}


def load_index(index_path=INDEX_PATH, paths_path=PATHS_PATH):
    """Load the FAISS index and corresponding image paths."""

    index_path = Path(index_path)
    paths_path = Path(paths_path)

    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at '{index_path}'. "
            "Build an index from Dataset Management first."
        )

    if not paths_path.exists():
        raise FileNotFoundError(
            f"Image paths file not found at '{paths_path}'. "
            "Rebuild the image index."
        )

    index = faiss.read_index(str(index_path))
    paths = paths_path.read_text(encoding="utf-8").splitlines()

    if index.ntotal != len(paths):
        raise ValueError(
            f"Index contains {index.ntotal} vectors, "
            f"but the paths file contains {len(paths)} paths."
        )

    return index, paths


def search(
    query_embedding,
    top_k=5,
    index_path=INDEX_PATH,
    paths_path=PATHS_PATH,
    threshold=DEFAULT_SIMILARITY_THRESHOLD,
):
    """Load the index and return similarity-ranked results."""

    index, paths = load_index(index_path, paths_path)

    return search_loaded(
        index=index,
        paths=paths,
        query_embedding=query_embedding,
        top_k=top_k,
        threshold=threshold,
    )


def search_loaded(
    index,
    paths,
    query_embedding,
    top_k=5,
    threshold=DEFAULT_SIMILARITY_THRESHOLD,
):
    """Search an already-loaded FAISS index by similarity."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    if index.ntotal == 0:
        return []

    query = np.asarray(query_embedding, dtype=np.float32)

    if query.ndim == 1:
        query = query.reshape(1, -1)

    if query.shape != (1, index.d):
        raise ValueError(
            f"Expected query embedding shape (1, {index.d}), "
            f"got {query.shape}."
        )

    scores, ids = index.search(query, index.ntotal)
    results = []

    for score, index_id in zip(scores[0], ids[0]):
        if index_id < 0 or score < threshold:
            continue

        results.append(
            {
                "image_path": paths[int(index_id)],
                "score": float(score),
                "faiss_id": int(index_id),
            }
        )

    results.sort(
        key=lambda result: result["score"],
        reverse=True,
    )

    results = results[:top_k]

    for rank, result in enumerate(results, start=1):
        result["rank"] = rank

    return results


def search_by_text(
    query: str,
    limit: int = 20,
    clip_model: ClipModel | None = None,
):
    """Search using full-query and attribute-focused CLIP prompts."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must not be empty")

    if limit < 1:
        raise ValueError("Limit must be at least 1")

    model = clip_model or load_clip_model()
    query = query.strip()

    query_embedding = model.encode_text(query)
    attributes = extract_attributes(query)

    objects = attributes.get("objects", [])
    colors = attributes.get("colors", [])
    actions = attributes.get("actions", [])
    scenes = attributes.get("scenes", [])
    scene_elements = attributes.get("scene_elements", [])

    prompts = []

    if objects:
        prompts.append(f"a photo of {' '.join(objects)}")

    if objects and colors:
        prompts.append(
            f"a photo of {' '.join(colors)} {' '.join(objects)}"
        )

    if objects and actions:
        prompts.append(
            f"a photo of {' '.join(objects)} {' '.join(actions)}"
        )

    surroundings = scenes + scene_elements

    if objects and surroundings:
        prompts.append(
            f"a photo of {' '.join(objects)} "
            f"in {' '.join(surroundings)}"
        )

    # Remove duplicate prompts while preserving their order.
    prompts = list(dict.fromkeys(prompts))

    attribute_embeddings = [
        model.encode_text(prompt) for prompt in prompts
    ]

    index, paths = load_index()

    return search_text_loaded(
        query=query,
        index=index,
        paths=paths,
        query_embedding=query_embedding,
        attribute_embeddings=attribute_embeddings,
        top_k=limit,
    )


def search_text_loaded(
    query: str,
    index,
    paths,
    query_embedding,
    top_k: int = 5,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    attribute_embeddings=None,
):
    """Rank images using semantic similarity and attribute-aware scoring."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    if not isinstance(query, str):
        raise TypeError("Query must be a string")

    if index.ntotal == 0:
        return []

    # Retrieve candidates using the complete query.
    candidates = search_loaded(
        index=index,
        paths=paths,
        query_embedding=query_embedding,
        top_k=index.ntotal,
        threshold=float("-inf"),
    )

    attributes = extract_attributes(query)

    query_objects = {
        str(term).lower().replace("_", " ").strip()
        for term in attributes.get("objects", [])
        if str(term).strip()
    }

    expanded_objects = set(query_objects)

    for obj in query_objects:
        expanded_objects.update(CATEGORY_ALIASES.get(obj, set()))

    # Split category names into words for folder matching.
    normalized_objects = {
        word
        for obj in expanded_objects
        for word in obj.replace("_", " ").split()
    }

    # Calculate candidate similarity for each attribute prompt.
    attribute_score_maps = []

    for embedding in attribute_embeddings or []:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)

        if vector.size != index.d:
            continue

        norm = float(np.linalg.norm(vector))

        if norm == 0:
            continue

        vector = vector / norm

        scores, ids = index.search(
            vector.reshape(1, -1),
            index.ntotal,
        )

        score_map = {
            int(image_id): float(score)
            for score, image_id in zip(scores[0], ids[0])
            if image_id >= 0
        }

        attribute_score_maps.append(score_map)

    for result in candidates:
        image_id = result["faiss_id"]

        folder = Path(result["image_path"]).parent.name.lower()
        normalized_folder = folder.replace("_", " ")
        folder_words = set(normalized_folder.split())

        category_match = bool(normalized_objects & folder_words)

        if not category_match:
            category_match = any(
                alias.replace("_", " ") in normalized_folder
                for alias in expanded_objects
            )

        result["category_match"] = category_match

        matching_scores = [
            score_map[image_id]
            for score_map in attribute_score_maps
            if image_id in score_map
        ]

        # Average attribute scores instead of allowing one prompt
        # to dominate the entire ranking.
        ranking_score = result["score"]

        if matching_scores:
            attribute_score = sum(matching_scores) / len(matching_scores)

            ranking_score = (
                (1.0 - ATTRIBUTE_WEIGHT) * result["score"]
                + ATTRIBUTE_WEIGHT * attribute_score
            )

        # Category information is a small bonus, not a hard priority.
        if category_match:
            ranking_score += CATEGORY_BONUS

        result["ranking_score"] = float(ranking_score)

    # Filter using the full-query similarity threshold.
    results = [
        result
        for result in candidates
        if result["score"] >= threshold
    ]

    # The combined score determines ranking; category matches
    # receive only the small bonus applied above.
    results.sort(
        key=lambda result: (
            result["ranking_score"],
            result["score"],
        ),
        reverse=True,
    )

    results = results[:top_k]

    for rank, result in enumerate(results, start=1):
        result["rank"] = rank
        result.pop("ranking_score", None)

    return results


def search_by_image(
    image,
    limit: int = 20,
    clip_model: ClipModel | None = None,
):
    """Search images using an uploaded reference image."""

    if limit < 1:
        raise ValueError("Limit must be at least 1")

    model = clip_model or load_clip_model()
    query_embedding = model.encode_image(image)

    return search(
        query_embedding=query_embedding,
        top_k=limit,
    )
