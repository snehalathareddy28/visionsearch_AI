"""
detector.py — YOLOv8 object detector wrapper.

Responsibilities
----------------
- Load ``yolov8n.pt`` (or a configurable model) once per process.
- Run inference on a PIL image and return a list of RegionBox objects.
- Apply confidence threshold, max-detection cap, minimum crop size, and
  fractional padding — all driven by ``config.py``.
- Fall back to the grid splitter when no objects are detected or when
  ``force_grid=True`` is passed.

This module intentionally does **not** import numpy or FAISS so it can be
tested in isolation with a lightweight mock.
"""

from __future__ import annotations

import logging
import math
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image

from visionsearch_web_ui.region import config
from visionsearch_web_ui.region.region_splitter import (
    RegionBox,
    clamp_bbox,
    grid_regions,
    is_valid_bbox,
    pad_bbox,
)

logger = logging.getLogger(__name__)
SplitMode = Literal["yolo", "grid", "auto"]


@dataclass(frozen=True)
class DetectionOutcome:
    """Detection regions and whether automatic mode had to use grid tiles."""

    regions: list[RegionBox]
    requested_mode: SplitMode
    used_mode: Literal["yolo", "grid"]
    fallback_reason: str | None = None


# ---------------------------------------------------------------------------
# Model loader (singleton per process)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_yolo(model_name: str):
    """Load and cache a YOLO model.

    The first call downloads weights from the Ultralytics CDN if they are
    not already present in the working directory.
    """
    from ultralytics import YOLO  # deferred import — ultralytics is optional

    logger.info("Loading YOLO model: %s", model_name)
    model = YOLO(model_name)
    checkpoint = Path(getattr(model, "ckpt_path", model_name))
    if not checkpoint.is_file():
        raise RuntimeError(f"YOLO checkpoint was not found after loading: {checkpoint}")
    logger.info(
        "YOLO model loaded: requested=%s checkpoint=%s checkpoint_exists=%s",
        model_name,
        checkpoint.resolve(),
        checkpoint.is_file(),
    )
    return model


def get_yolo():
    """Return the process-wide cached YOLO model instance."""
    return _load_yolo(config.YOLO_MODEL)


# ---------------------------------------------------------------------------
# Detection pipeline
# ---------------------------------------------------------------------------


def detect_regions(
    image: Image.Image,
    *,
    force_grid: bool = False,
    split_mode: SplitMode | None = None,
    conf_threshold: float = config.YOLO_CONF_THRESHOLD,
    max_detections: int = config.YOLO_MAX_DETECTIONS,
    min_crop_px: int = config.YOLO_MIN_CROP_PX,
    pad_fraction: float = config.YOLO_PAD_FRACTION,
) -> list[RegionBox]:
    """Return object regions or grid regions while preserving legacy callers."""
    mode: SplitMode = "grid" if force_grid else (split_mode or "auto")
    return detect_regions_with_status(
        image,
        split_mode=mode,
        conf_threshold=conf_threshold,
        max_detections=max_detections,
        min_crop_px=min_crop_px,
        pad_fraction=pad_fraction,
    ).regions


def detect_regions_with_status(
    image: Image.Image,
    *,
    split_mode: SplitMode = "auto",
    conf_threshold: float = config.YOLO_CONF_THRESHOLD,
    max_detections: int = config.YOLO_MAX_DETECTIONS,
    min_crop_px: int = config.YOLO_MIN_CROP_PX,
    pad_fraction: float = config.YOLO_PAD_FRACTION,
) -> DetectionOutcome:
    """Detect regions and report the requested/actual mode and fallback reason.

    ``yolo`` is strict and propagates model-loading or inference failures.
    ``auto`` logs those failures and reports an explicit grid fallback.
    """
    if split_mode not in ("yolo", "grid", "auto"):
        raise ValueError("split_mode must be 'yolo', 'grid', or 'auto'")
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL image")
    image = image.convert("RGB")
    img_w, img_h = image.size
    if img_w <= 0 or img_h <= 0:
        raise ValueError("image dimensions must be positive")

    if split_mode == "grid":
        return DetectionOutcome(
            regions=grid_regions(img_w, img_h),
            requested_mode=split_mode,
            used_mode="grid",
        )

    try:
        boxes = _run_yolo(
            image,
            conf_threshold=conf_threshold,
            max_detections=max_detections,
            min_crop_px=min_crop_px,
            pad_fraction=pad_fraction,
            img_w=img_w,
            img_h=img_h,
        )
    except Exception as error:
        if split_mode == "yolo":
            logger.exception("YOLO detection failed in strict YOLO mode.")
            raise RuntimeError(f"YOLO detection failed: {error}") from error
        logger.exception("YOLO inference failed; automatic mode is using grid tiles.")
        return DetectionOutcome(
            regions=grid_regions(img_w, img_h),
            requested_mode=split_mode,
            used_mode="grid",
            fallback_reason="YOLO failed; grid tiles were used instead.",
        )

    if boxes:
        return DetectionOutcome(
            regions=boxes,
            requested_mode=split_mode,
            used_mode="yolo",
        )

    if split_mode == "yolo":
        return DetectionOutcome(
            regions=[],
            requested_mode=split_mode,
            used_mode="yolo",
            fallback_reason="YOLO completed but found no valid objects.",
        )

    logger.info("YOLO found no valid regions; automatic mode is using grid tiles.")
    return DetectionOutcome(
        regions=grid_regions(img_w, img_h),
        requested_mode=split_mode,
        used_mode="grid",
        fallback_reason="YOLO found no valid objects; grid tiles were used instead.",
    )


def _run_yolo(
    image: Image.Image,
    *,
    conf_threshold: float,
    max_detections: int,
    min_crop_px: int,
    pad_fraction: float,
    img_w: int,
    img_h: int,
) -> list[RegionBox]:
    """Run YOLO inference and convert results to RegionBox objects."""
    model = get_yolo()

    # verbose=False suppresses per-image YOLO console output.
    results = model(image, conf=conf_threshold, verbose=False)

    raw_boxes: list[tuple[float, float, float, float, float, str]] = []

    for result in results:
        if result.boxes is None:
            continue
        boxes_xyxy = result.boxes.xyxy.cpu().tolist()
        confs = result.boxes.conf.cpu().tolist()
        cls_ids = result.boxes.cls.cpu().tolist()
        names = result.names  # dict {int: str}

        for (x1, y1, x2, y2), conf, cls_id in zip(boxes_xyxy, confs, cls_ids):
            label = names.get(int(cls_id), str(int(cls_id)))
            coordinates = (float(x1), float(y1), float(x2), float(y2))
            confidence = float(conf)
            logger.info(
                "YOLO detection class=%s confidence=%.4f bbox_xyxy=%s",
                label,
                confidence,
                coordinates,
            )
            if not all(math.isfinite(value) for value in (*coordinates, confidence)):
                logger.warning("Ignoring non-finite YOLO output for class %s", label)
                continue
            raw_boxes.append((*coordinates, confidence, label))

    # Sort by confidence descending, then cap.
    raw_boxes.sort(key=lambda b: b[4], reverse=True)
    raw_boxes = raw_boxes[:max_detections]

    boxes: list[RegionBox] = []
    for x1, y1, x2, y2, conf, label in raw_boxes:
        x1, y1, x2, y2 = pad_bbox(x1, y1, x2, y2, img_w, img_h, pad_fraction)
        x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, img_w, img_h)
        if not is_valid_bbox(x1, y1, x2, y2, min_crop_px):
            continue
        boxes.append(
            RegionBox(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                class_label=label,
                confidence=conf,
                crop_type="yolo",
            )
        )

    return boxes
