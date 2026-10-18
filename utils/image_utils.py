"""Image loading and display helpers."""

from pathlib import Path

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(directory):
    directory = Path(directory)
    return sorted(path for path in directory.rglob("*") if path.suffix.lower() in SUPPORTED_EXTENSIONS)
