import hashlib
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
PUBLIC_SHARING_ENABLED = os.getenv(
    "VISIONSEARCH_PUBLIC_SHARING_ENABLED", ""
).strip().lower() in {"1", "true", "yes"}
PUBLIC_BASE_URL = os.getenv("VISIONSEARCH_PUBLIC_BASE_URL", "").strip().rstrip("/")
LOCAL_API_BASE_URL = os.getenv(
    "VISIONSEARCH_API_BASE_URL", "http://127.0.0.1:8000"
).strip().rstrip("/")


def image_id_for_path(image_path: Path) -> str | None:
    """Create a stable opaque ID for a supported image inside the project."""
    try:
        resolved_path = image_path.resolve()
        relative_path = resolved_path.relative_to(ROOT.resolve())
    except (OSError, ValueError):
        return None

    if (
        not resolved_path.is_file()
        or resolved_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES
    ):
        return None

    return hashlib.sha256(relative_path.as_posix().encode("utf-8")).hexdigest()
