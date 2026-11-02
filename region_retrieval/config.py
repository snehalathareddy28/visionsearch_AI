"""Configuration for the independent region retrieval module."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

PACKAGE_DIR = Path(__file__).resolve().parent


def _configured_path(value: str | None, default: Path) -> Path:
    path = Path(value).expanduser() if value else default
    if not path.is_absolute():
        path = PACKAGE_DIR / path
    return path.resolve()


@dataclass(frozen=True)
class RegionSettings:
    """Validated detector, splitter, and storage settings."""

    yolo_model: str = "yolov8n.pt"
    confidence_threshold: float = 0.35
    max_detections: int = 8
    min_crop_px: int = 64
    padding_fraction: float = 0.05
    grid_rows: int = 3
    grid_cols: int = 3
    grid_overlap: float = 0.20
    index_dir: Path = PACKAGE_DIR / "index"
    database_path: Path = PACKAGE_DIR / "database" / "metadata.sqlite"
    cache_dir: Path = PACKAGE_DIR / "cache"

    def __post_init__(self) -> None:
        if not self.yolo_model.strip():
            raise ValueError("yolo_model cannot be empty")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if self.max_detections < 1:
            raise ValueError("max_detections must be at least 1")
        if self.min_crop_px < 1:
            raise ValueError("min_crop_px must be at least 1")
        if self.padding_fraction < 0.0:
            raise ValueError("padding_fraction cannot be negative")
        if self.grid_rows < 1 or self.grid_cols < 1:
            raise ValueError("grid_rows and grid_cols must each be at least 1")
        if not 0.0 <= self.grid_overlap < 1.0:
            raise ValueError("grid_overlap must be in [0, 1)")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> RegionSettings:
        """Create settings using `REGION_RETRIEVAL_` environment variables."""
        env = os.environ if environ is None else environ

        def value(name: str, default: str) -> str:
            return env.get(f"REGION_RETRIEVAL_{name}", default)

        return cls(
            yolo_model=value("YOLO_MODEL", "yolov8n.pt"),
            confidence_threshold=float(value("CONFIDENCE_THRESHOLD", "0.35")),
            max_detections=int(value("MAX_DETECTIONS", "8")),
            min_crop_px=int(value("MIN_CROP_PX", "64")),
            padding_fraction=float(value("PADDING_FRACTION", "0.05")),
            grid_rows=int(value("GRID_ROWS", "3")),
            grid_cols=int(value("GRID_COLS", "3")),
            grid_overlap=float(value("GRID_OVERLAP", "0.20")),
            index_dir=_configured_path(
                env.get("REGION_RETRIEVAL_INDEX_DIR"), PACKAGE_DIR / "index"
            ),
            database_path=_configured_path(
                env.get("REGION_RETRIEVAL_DATABASE_PATH"),
                PACKAGE_DIR / "database" / "metadata.sqlite",
            ),
            cache_dir=_configured_path(
                env.get("REGION_RETRIEVAL_CACHE_DIR"), PACKAGE_DIR / "cache"
            ),
        )


DEFAULT_SETTINGS = RegionSettings.from_env()
