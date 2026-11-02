"""
config.py — All tunable knobs for region-based retrieval.

Every value can be overridden with an environment variable of the same name,
prefixed with VISIONSEARCH_. For example:
    VISIONSEARCH_YOLO_MODEL=yolov8s.pt
"""

from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]  # project root


def _env(name: str, default: str) -> str:
    """Read an env var with the VISIONSEARCH_ prefix, falling back to default."""
    return os.environ.get(f"VISIONSEARCH_{name}", default)


# ---------------------------------------------------------------------------
# YOLO detector
# ---------------------------------------------------------------------------

#: Model weights file name (ultralytics downloads on first use).
YOLO_MODEL: str = _env("YOLO_MODEL", "yolov8n.pt")

#: Minimum detection confidence to keep a box.
YOLO_CONF_THRESHOLD: float = float(_env("YOLO_CONF_THRESHOLD", "0.35"))

#: Maximum number of detections kept per image (by descending confidence).
YOLO_MAX_DETECTIONS: int = int(_env("YOLO_MAX_DETECTIONS", "8"))

#: Minimum crop dimension in pixels (both width AND height must meet this).
YOLO_MIN_CROP_PX: int = int(_env("YOLO_MIN_CROP_PX", "64"))

#: Fractional padding added around each detected bbox before cropping.
YOLO_PAD_FRACTION: float = float(_env("YOLO_PAD_FRACTION", "0.05"))

# ---------------------------------------------------------------------------
# Grid / tile fallback
# ---------------------------------------------------------------------------

#: Number of rows and columns in the grid fallback.
GRID_ROWS: int = int(_env("GRID_ROWS", "3"))
GRID_COLS: int = int(_env("GRID_COLS", "3"))

#: Fractional overlap between adjacent tiles (0.0 = no overlap).
GRID_OVERLAP: float = float(_env("GRID_OVERLAP", "0.20"))

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

#: Number of crops embedded in a single CLIP forward pass.
EMBED_BATCH_SIZE: int = int(_env("EMBED_BATCH_SIZE", "8"))

# ---------------------------------------------------------------------------
# FAISS / index
# ---------------------------------------------------------------------------

#: Directory that stores all index artefacts.
INDEX_DIR: Path = _ROOT / "index"

#: Path to the region FAISS index (separate from the whole-image index).
REGION_FAISS_PATH: Path = INDEX_DIR / "region_faiss.index"

#: Path to the whole-image FAISS index used only by region retrieval.
REGION_WHOLE_FAISS_PATH: Path = INDEX_DIR / "region_whole_faiss.index"

#: Path to the SQLite metadata database.
REGION_DB_PATH: Path = INDEX_DIR / "region_metadata.db"

# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

#: Default search scope: "whole" | "regions" | "both"
DEFAULT_SEARCH_SCOPE: str = _env("DEFAULT_SEARCH_SCOPE", "both")

#: Weight given to the whole-image score when scope == "both".
#: Region best-score weight = 1 - WHOLE_IMAGE_WEIGHT.
WHOLE_IMAGE_WEIGHT: float = float(_env("WHOLE_IMAGE_WEIGHT", "0.5"))

#: Default number of results returned.
DEFAULT_TOP_K: int = int(_env("DEFAULT_TOP_K", "10"))

#: Small category compatibility bonus applied after raw similarity ranking.
REGION_CATEGORY_BONUS: float = float(_env("REGION_CATEGORY_BONUS", "0.08"))

#: Small penalty for mismatched query/category labels to keep CLIP similarity dominant.
REGION_CATEGORY_PENALTY: float = float(_env("REGION_CATEGORY_PENALTY", "0.03"))

#: Keep the original ranking unless category-aware ranking is explicitly enabled.
REGION_CATEGORY_AWARE_ENABLED: bool = (
    _env("REGION_CATEGORY_AWARE_ENABLED", "0").strip().lower()
    in ("1", "true", "yes", "on")
)

#: Ignore uncertain YOLO detections when applying category compatibility.
REGION_CATEGORY_MIN_CONFIDENCE: float = float(
    _env("REGION_CATEGORY_MIN_CONFIDENCE", "0.6")
)

CATEGORY_ALIASES: dict[str, set[str]] = {
    "dog": {"dog", "dogs", "puppy", "puppies"},
    "cat": {"cat", "cats", "kitten", "kittens"},
    "car": {"car", "cars", "automobile", "vehicle", "vehicles"},
    "flower": {"flower", "flowers", "lotus", "sunflower", "lily", "lilies", "waterlily", "water_lily"},
    "person": {"person", "people", "man", "woman", "girl", "boy"},
}

if not 0.0 <= REGION_CATEGORY_MIN_CONFIDENCE <= 1.0:
    raise ValueError("REGION_CATEGORY_MIN_CONFIDENCE must be in [0, 1]")
if REGION_CATEGORY_BONUS < 0.0 or REGION_CATEGORY_PENALTY < 0.0:
    raise ValueError("Region category bonus and penalty must be non-negative")

# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

#: Whether to cache cropped images to disk (False = generate on the fly).
CACHE_CROPS: bool = _env("CACHE_CROPS", "0").strip() in ("1", "true", "yes")

#: Directory used when CACHE_CROPS is True.
CROP_CACHE_DIR: Path = _ROOT / "crop_cache"
