"""FAISS retrieval for normalized CLIP embeddings."""

from pathlib import Path

import faiss
import numpy as np
from PIL import Image

from models.clip_model import ClipModel, load_clip_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = PROJECT_ROOT / "index" / "faiss.index"
PATHS_PATH = PROJECT_ROOT / "index" / "image_paths.txt"


def load_index(index_path=INDEX_PATH, paths_path=PATHS_PATH):
    """Load a FAISS index and its line-aligned image paths."""

    index_path = Path(index_path)
    paths_path = Path(paths_path)

    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at '{index_path}'. "
            f"Build an index from Dataset Management first. "
            f"Project root: '{PROJECT_ROOT}'."
        )

    if not paths_path.exists():
        raise FileNotFoundError(
            f"Image paths file not found at '{paths_path}'. "
            "Rebuild the image index."
        )

    index = faiss.read_index(str(index_path))

    paths = paths_path.read_text(
        encoding="utf-8"
    ).splitlines()

    if index.ntotal != len(paths):
        raise ValueError(
            f"Index contains {index.ntotal} vectors "
            f"but paths file contains {len(paths)} paths"
        )

    return index, paths


def search(
    query_embedding,
    top_k=5,
    index_path=INDEX_PATH,
    paths_path=PATHS_PATH,
    threshold=0.25,
):
    """Return ranked search results."""

    index, paths = load_index(
        index_path,
        paths_path
    )

    return search_loaded(
        index,
        paths,
        query_embedding,
        top_k,
        threshold
    )


def search_loaded(
    index,
    paths,
    query_embedding,
    top_k=5,
    threshold=0.25,
):
    """Search an already-loaded FAISS index."""

    if top_k < 1:
        raise ValueError(
            "top_k must be at least 1"
        )

    if index.ntotal == 0:
        return []

    query = np.asarray(
        query_embedding,
        dtype=np.float32
    )

    if query.ndim == 1:
        query = query[None, :]

    if query.shape != (1, index.d):
        raise ValueError(
            f"Expected query embedding shape "
            f"(1, {index.d}), got {query.shape}"
        )

    scores, ids = index.search(
        query,
        min(top_k, index.ntotal)
    )

    results = []

    for score, index_id in zip(
        scores[0],
        ids[0]
    ):

        if index_id < 0:
            continue

        if score < threshold:
            continue

        results.append(
            {
                "rank": len(results) + 1,
                "image_path": paths[index_id],
                "score": float(score),
                "faiss_id": int(index_id),
            }
        )

    return results


def search_by_text(
    query: str,
    limit: int = 20,
    clip_model: ClipModel | None = None,
):
    """Search images using a text query."""

    model = clip_model or load_clip_model()

    query_embedding = model.encode_text(query)

    return search(
        query_embedding,
        limit
    )


def search_by_image(
    image,
    limit: int = 20,
    clip_model: ClipModel | None = None,
):
    """Search images using an uploaded image."""

    model = clip_model or load_clip_model()

    query_embedding = model.encode_image(image)

    return search(
        query_embedding,
        limit
    )