"""
region_splitter.py — Tile-grid fallback and bbox utility helpers.

Responsibilities
----------------
- Compute an overlapping tile grid when YOLO finds no objects (or when
  the user explicitly requests "grid mode").
- Pad, clamp, and validate bounding boxes.
- Convert raw (x1, y1, x2, y2) tuples to PIL crops.

All geometry is done in **pixel coordinates** (floats are rounded to int
only when cropping).  Exported types use plain tuples so there is no
dependency on ultralytics or numpy in this module.
"""

from __future__ import annotations

from typing import NamedTuple

from PIL import Image

from visionsearch_web_ui.region import config


class RegionBox(NamedTuple):
    """A single region of an image ready to be embedded.

    Attributes
    ----------
    x1, y1, x2, y2 : float
        Pixel coordinates of the crop, clamped to image bounds.
    class_label : str
        Human-readable class name, e.g. ``"dog"`` or ``"grid_1_2"``.
    confidence : float
        Detection confidence (0.0 for grid tiles).
    crop_type : str
        Either ``"yolo"`` or ``"grid"``.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    class_label: str
    confidence: float
    crop_type: str


# ---------------------------------------------------------------------------
# Bbox helpers
# ---------------------------------------------------------------------------


def pad_bbox(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_w: int,
    img_h: int,
    pad_fraction: float = config.YOLO_PAD_FRACTION,
) -> tuple[float, float, float, float]:
    """Add a fractional border around a bbox and clamp to image bounds.

    Parameters
    ----------
    x1, y1, x2, y2 : float
        Raw bbox coordinates in pixel space.
    img_w, img_h : int
        Image width and height in pixels.
    pad_fraction : float
        Fraction of the bbox width/height to add on each side.

    Returns
    -------
    tuple[float, float, float, float]
        Padded, clamped (x1, y1, x2, y2).
    """
    w = x2 - x1
    h = y2 - y1
    pad_x = w * pad_fraction
    pad_y = h * pad_fraction

    x1 = max(0.0, x1 - pad_x)
    y1 = max(0.0, y1 - pad_y)
    x2 = min(float(img_w), x2 + pad_x)
    y2 = min(float(img_h), y2 + pad_y)

    return x1, y1, x2, y2


def clamp_bbox(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float]:
    """Clamp a bbox strictly within image bounds without padding."""
    return (
        max(0.0, min(x1, float(img_w))),
        max(0.0, min(y1, float(img_h))),
        max(0.0, min(x2, float(img_w))),
        max(0.0, min(y2, float(img_h))),
    )


def is_valid_bbox(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    min_px: int = config.YOLO_MIN_CROP_PX,
) -> bool:
    """Return True if the bbox is large enough to embed meaningfully."""
    return (x2 - x1) >= min_px and (y2 - y1) >= min_px


# ---------------------------------------------------------------------------
# Grid splitter
# ---------------------------------------------------------------------------


def grid_regions(
    img_w: int,
    img_h: int,
    rows: int = config.GRID_ROWS,
    cols: int = config.GRID_COLS,
    overlap: float = config.GRID_OVERLAP,
) -> list[RegionBox]:
    """Split an image into overlapping tiles and return RegionBox objects.

    The tile at row *r*, column *c* (0-indexed) is labelled
    ``"grid_{r}_{c}"``.

    Parameters
    ----------
    img_w, img_h : int
        Image dimensions in pixels.
    rows, cols : int
        Number of tiles in each axis.
    overlap : float
        Fractional overlap between adjacent tiles (0 = no overlap).

    Returns
    -------
    list[RegionBox]
        One RegionBox per tile that passes the minimum-size check.
    """
    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must each be at least 1")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")

    tile_w = img_w / cols
    tile_h = img_h / rows
    step_x = tile_w * (1.0 - overlap)
    step_y = tile_h * (1.0 - overlap)

    boxes: list[RegionBox] = []
    for r in range(rows):
        for c in range(cols):
            x1 = c * step_x
            y1 = r * step_y
            x2 = x1 + tile_w
            y2 = y1 + tile_h
            x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, img_w, img_h)

            if not is_valid_bbox(x1, y1, x2, y2):
                continue  # tile too small for this image

            boxes.append(
                RegionBox(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    class_label=f"grid_{r}_{c}",
                    confidence=0.0,
                    crop_type="grid",
                )
            )

    return boxes


# ---------------------------------------------------------------------------
# Crop helper
# ---------------------------------------------------------------------------


def crop_region(image: Image.Image, box: RegionBox) -> Image.Image:
    """Return a PIL crop for the given RegionBox.

    Parameters
    ----------
    image : PIL.Image.Image
        The full parent image (must already be in RGB mode).
    box : RegionBox
        The region to crop.

    Returns
    -------
    PIL.Image.Image
        Cropped image in RGB mode.
    """
    region = image.crop(
        (int(box.x1), int(box.y1), int(box.x2), int(box.y2))
    )
    return region.convert("RGB")
