"""YOLOv8 region detection with grid and automatic fallback modes."""

from __future__ import annotations

import logging
import math
from functools import lru_cache
from typing import Any, Literal

from PIL import Image

from region_retrieval.config import DEFAULT_SETTINGS, RegionSettings
from region_retrieval.region_splitter import (
    RegionBox,
    clamp_bbox,
    grid_regions,
    is_valid_bbox,
    pad_bbox,
)

logger = logging.getLogger(__name__)
SplitMode = Literal["yolo", "grid", "auto"]


@lru_cache(maxsize=4)
def _load_yolo(model_name: str) -> Any:
    """Load a configured Ultralytics YOLO model once per model name."""
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "YOLO mode requires the 'ultralytics' package; install the project dependencies."
        ) from error
    logger.info("Loading YOLO model %s", model_name)
    return YOLO(model_name)


def get_yolo(settings: RegionSettings = DEFAULT_SETTINGS) -> Any:
    """Return the cached YOLO model selected by settings."""
    return _load_yolo(settings.yolo_model)


def _as_list(value: Any) -> list[Any]:
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _label_for(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    try:
        return str(names[class_id])
    except (IndexError, KeyError, TypeError):
        return str(class_id)


def _run_yolo(
    image: Image.Image,
    model: Any,
    settings: RegionSettings,
) -> list[RegionBox]:
    results = model(
        image,
        conf=settings.confidence_threshold,
        verbose=False,
    )
    image_width, image_height = image.size
    detections: list[tuple[float, float, float, float, float, int, Any]] = []

    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        coordinates = _as_list(boxes.xyxy)
        confidences = _as_list(boxes.conf)
        classes = _as_list(boxes.cls)
        if not (len(coordinates) == len(confidences) == len(classes)):
            raise RuntimeError("YOLO returned inconsistent box metadata lengths")
        names = getattr(result, "names", {})
        for coords, confidence, class_id in zip(coordinates, confidences, classes):
            if len(coords) != 4:
                raise RuntimeError("YOLO returned a bounding box without four coordinates")
            x1, y1, x2, y2 = (float(value) for value in coords)
            confidence_value = float(confidence)
            class_value = int(class_id)
            if (
                not all(math.isfinite(v) for v in (x1, y1, x2, y2, confidence_value))
                or confidence_value < settings.confidence_threshold
                or x2 <= x1
                or y2 <= y1
            ):
                continue
            detections.append(
                (x1, y1, x2, y2, confidence_value, class_value, names)
            )

    detections.sort(key=lambda detection: detection[4], reverse=True)
    regions: list[RegionBox] = []
    for x1, y1, x2, y2, confidence, class_id, names in detections:
        padded = pad_bbox(
            x1,
            y1,
            x2,
            y2,
            image_width,
            image_height,
            settings.padding_fraction,
        )
        bounded = clamp_bbox(*padded, image_width, image_height)
        if not is_valid_bbox(*bounded, min_crop_px=settings.min_crop_px):
            continue
        regions.append(
            RegionBox(
                x1=bounded[0],
                y1=bounded[1],
                x2=bounded[2],
                y2=bounded[3],
                class_label=_label_for(names, class_id),
                confidence=confidence,
                crop_type="yolo",
            )
        )
        if len(regions) == settings.max_detections:
            break
    return regions


def detect_regions(
    image: Image.Image,
    *,
    split_mode: SplitMode = "auto",
    model: Any | None = None,
    settings: RegionSettings = DEFAULT_SETTINGS,
) -> list[RegionBox]:
    """Return YOLO regions, grid tiles, or automatic YOLO-with-grid-fallback regions."""
    if split_mode not in ("yolo", "grid", "auto"):
        raise ValueError("split_mode must be 'yolo', 'grid', or 'auto'")
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL Image")
    image = image.convert("RGB")
    image_width, image_height = image.size
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")

    if split_mode == "grid":
        return grid_regions(
            image_width,
            image_height,
            rows=settings.grid_rows,
            cols=settings.grid_cols,
            overlap=settings.grid_overlap,
            min_crop_px=settings.min_crop_px,
        )

    try:
        regions = _run_yolo(image, model if model is not None else get_yolo(settings), settings)
    except Exception:
        if split_mode == "yolo":
            raise
        logger.exception("YOLO failed in automatic mode; falling back to grid regions")
        regions = []

    if regions or split_mode == "yolo":
        return regions
    logger.info("YOLO produced no valid regions; using grid fallback")
    return grid_regions(
        image_width,
        image_height,
        rows=settings.grid_rows,
        cols=settings.grid_cols,
        overlap=settings.grid_overlap,
        min_crop_px=settings.min_crop_px,
    )
