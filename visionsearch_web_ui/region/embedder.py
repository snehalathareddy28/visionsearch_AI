"""
embedder.py — Batch CLIP embedding of image crops.

Responsibilities
----------------
- Accept a list of (PIL crop, RegionBox) pairs and a ClipModel.
- Process them in batches of ``config.EMBED_BATCH_SIZE`` with a tqdm bar.
- Return a numpy array of L2-normalised embeddings aligned with the input list.

Design note: batch processing matters even on CPU because the CLIP
processor can be called once with multiple images, reducing Python
overhead and allowing torch to parallelise ops across cores.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from visionsearch_web_ui.region import config

if TYPE_CHECKING:
    from models.clip_model import ClipModel
    from visionsearch_web_ui.region.region_splitter import RegionBox

logger = logging.getLogger(__name__)


def embed_crops(
    crops: list[Image.Image],
    clip_model: "ClipModel",
    *,
    batch_size: int = config.EMBED_BATCH_SIZE,
    show_progress: bool = True,
) -> np.ndarray:
    """Embed a list of PIL crops with CLIP in batches.

    Parameters
    ----------
    crops : list[PIL.Image.Image]
        Pre-cropped images (any size; the CLIP processor handles resizing).
    clip_model : ClipModel
        The loaded CLIP model wrapper.
    batch_size : int
        How many crops to pass to the model at once.
    show_progress : bool
        Whether to show a tqdm progress bar.

    Returns
    -------
    numpy.ndarray, shape (N, 512)
        L2-normalised float32 embeddings.  Row *i* corresponds to
        ``crops[i]``.

    Raises
    ------
    ValueError
        If *crops* is empty.
    """
    if not crops:
        raise ValueError("crops list must not be empty")
    if batch_size < 1:
        raise ValueError("batch_size must be greater than zero")

    try:
        from tqdm import tqdm
    except ImportError:  # tqdm is already a project dependency but guard anyway
        tqdm = None  # type: ignore[assignment]

    all_vectors: list[np.ndarray] = []
    total = len(crops)

    batches = [crops[i : i + batch_size] for i in range(0, total, batch_size)]

    iterator = (
        tqdm(batches, desc="Embedding crops", unit="batch")
        if (show_progress and tqdm is not None)
        else batches
    )

    for batch in iterator:
        encode_images = getattr(clip_model, "encode_images", None)
        if callable(encode_images):
            vectors = np.asarray(encode_images(batch), dtype=np.float32)
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)
            if vectors.ndim != 2 or vectors.shape[0] != len(batch):
                raise ValueError(
                    "CLIP batch encoder must return one 2D embedding per input image"
                )
            all_vectors.extend(vectors)
        else:
            # Keep compatibility with model wrappers that only expose encode_image.
            for crop in batch:
                vec = np.asarray(clip_model.encode_image(crop), dtype=np.float32)
                if vec.ndim == 2:
                    vec = vec[0]
                all_vectors.append(vec)

    embedding_matrix = np.stack(all_vectors, axis=0)  # (N, 512)

    # Renormalise the full matrix once to make inner products == cosine sim.
    norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True).clip(min=1e-12)
    embedding_matrix = embedding_matrix / norms

    logger.debug("Embedded %d crops → shape %s", total, embedding_matrix.shape)
    return embedding_matrix


def embed_single(image: Image.Image, clip_model: "ClipModel") -> np.ndarray:
    """Embed a single PIL image and return a shape-(512,) float32 vector."""
    result = embed_crops([image], clip_model, show_progress=False)
    return result[0]
