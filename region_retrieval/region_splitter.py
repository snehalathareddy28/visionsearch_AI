"""Bounding-box validation, grid splitting, and in-memory image crops."""

from __future__ import annotations

import math
from dataclasses import dataclass

from PIL import Image

from region_retrieval.config import DEFAULT_SETTINGS, RegionSettings


@dataclass(frozen=True)
class RegionBox:
    """A region in pixel coordinates, with optional detector metadata."""

    x1: float
    y1: float
    x2: float
    y2: float
    class_label: str
    confidence: float
    crop_type: str


def _validate_dimensions(image_width: int, image_height: int) -> None:
    if (
        not isinstance(image_width, int)
        or isinstance(image_width, bool)
        or not isinstance(image_height, int)
        or isinstance(image_height, bool)
        or image_width <= 0
        or image_height <= 0
    ):
        raise ValueError("image dimensions must be positive integers")


def _validate_coordinates(
    x1: float, y1: float, x2: float, y2: float
) -> None:
    if not all(math.isfinite(float(value)) for value in (x1, y1, x2, y2)):
        raise ValueError("bounding-box coordinates must be finite")


def clamp_bbox(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """Clamp finite bbox coordinates to the image's pixel boundaries."""
    _validate_dimensions(image_width, image_height)
    _validate_coordinates(x1, y1, x2, y2)
    return (
        max(0.0, min(float(x1), float(image_width))),
        max(0.0, min(float(y1), float(image_height))),
        max(0.0, min(float(x2), float(image_width))),
        max(0.0, min(float(y2), float(image_height))),
    )


def pad_bbox(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_width: int,
    image_height: int,
    padding_fraction: float = DEFAULT_SETTINGS.padding_fraction,
) -> tuple[float, float, float, float]:
    """Expand a valid box by a fraction of its size and clamp it to the image."""
    _validate_coordinates(x1, y1, x2, y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bounding box must have positive width and height")
    if not math.isfinite(padding_fraction) or padding_fraction < 0.0:
        raise ValueError("padding_fraction must be finite and non-negative")

    padding_x = (x2 - x1) * padding_fraction
    padding_y = (y2 - y1) * padding_fraction
    return clamp_bbox(
        x1 - padding_x,
        y1 - padding_y,
        x2 + padding_x,
        y2 + padding_y,
        image_width,
        image_height,
    )


def is_valid_bbox(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    min_crop_px: int = DEFAULT_SETTINGS.min_crop_px,
) -> bool:
    """Return whether a finite box has positive area and meets minimum size."""
    if min_crop_px < 1:
        raise ValueError("min_crop_px must be at least 1")
    try:
        _validate_coordinates(x1, y1, x2, y2)
    except (TypeError, ValueError):
        return False
    return x2 > x1 and y2 > y1 and (x2 - x1) >= min_crop_px and (y2 - y1) >= min_crop_px


def grid_regions(
    image_width: int,
    image_height: int,
    rows: int = DEFAULT_SETTINGS.grid_rows,
    cols: int = DEFAULT_SETTINGS.grid_cols,
    overlap: float = DEFAULT_SETTINGS.grid_overlap,
    min_crop_px: int = DEFAULT_SETTINGS.min_crop_px,
) -> list[RegionBox]:
    """Split an image into a row/column grid with overlap and no duplicate tiles."""
    _validate_dimensions(image_width, image_height)
    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must each be at least 1")
    if not math.isfinite(overlap) or not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    if min_crop_px < 1:
        raise ValueError("min_crop_px must be at least 1")

    tile_width = image_width / (cols - (cols - 1) * overlap)
    tile_height = image_height / (rows - (rows - 1) * overlap)
    step_x = tile_width * (1.0 - overlap)
    step_y = tile_height * (1.0 - overlap)

    boxes: list[RegionBox] = []
    seen_pixel_boxes: set[tuple[int, int, int, int]] = set()
    for row in range(rows):
        for col in range(cols):
            x1, y1, x2, y2 = clamp_bbox(
                col * step_x,
                row * step_y,
                col * step_x + tile_width,
                row * step_y + tile_height,
                image_width,
                image_height,
            )
            pixel_box = (
                math.floor(x1),
                math.floor(y1),
                math.ceil(x2),
                math.ceil(y2),
            )
            if pixel_box in seen_pixel_boxes:
                continue
            seen_pixel_boxes.add(pixel_box)
            if not is_valid_bbox(*pixel_box, min_crop_px=min_crop_px):
                continue
            boxes.append(
                RegionBox(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    class_label=f"grid_{row}_{col}",
                    confidence=0.0,
                    crop_type="grid",
                )
            )
    return boxes


def crop_region(image: Image.Image, box: RegionBox) -> Image.Image:
    """Return an RGB crop from an image, clamping the box to image bounds."""
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL Image")
    image_width, image_height = image.size
    x1, y1, x2, y2 = clamp_bbox(
        box.x1, box.y1, box.x2, box.y2, image_width, image_height
    )
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bounding box is empty or outside the image")

    pixel_box = (
        max(0, math.floor(x1)),
        max(0, math.floor(y1)),
        min(image_width, math.ceil(x2)),
        min(image_height, math.ceil(y2)),
    )
    if pixel_box[2] <= pixel_box[0] or pixel_box[3] <= pixel_box[1]:
        raise ValueError("bounding box does not produce a non-empty crop")
    return image.convert("RGB").crop(pixel_box)
