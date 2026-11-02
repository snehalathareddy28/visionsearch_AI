"""
api_regions.py — FastAPI router for region-based retrieval endpoints.

Mount this router into the existing api.py with:
    app.include_router(region_router)

Endpoints
---------
POST /index/build            Kick off a full region index build.
POST /index/add              Add specific images to the region index.
GET  /index/stats            Return FAISS + SQLite statistics.
POST /search/text            Text-to-region search.
POST /search/image/detect    Upload an image → get detected crop thumbnails.
POST /search/image           Image-to-region search (whole / crop-id / bbox).
"""

from __future__ import annotations

import io
import hashlib
import logging
import math
import re
import uuid
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Response as FastAPIResponse, UploadFile
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel, Field, field_validator

from visionsearch_web_ui.region import config
from visionsearch_web_ui.sharing import PUBLIC_SHARING_ENABLED, image_id_for_path
from visionsearch_web_ui.region.index_manager import (
    add_images_to_region_index,
    build_region_index,
    get_index_stats,
)
from visionsearch_web_ui.region.region_splitter import crop_region, RegionBox
from visionsearch_web_ui.region.search_service import (
    detect_query_regions,
    invalidate_region_index_cache,
    search_regions,
)

logger = logging.getLogger(__name__)

_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def _resolve_request_id(value: Optional[str]) -> str:
    if value is None:
        return uuid.uuid4().hex
    if not _REQUEST_ID_PATTERN.fullmatch(value):
        raise HTTPException(
            status_code=400,
            detail="request_id must contain only letters, numbers, '.', '_' or '-'.",
        )
    return value
region_router = APIRouter(prefix="/api/region", tags=["Region Search"])

# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class BuildIndexRequest(BaseModel):
    image_dir: str = Field(
        default="dataset/caltech-101-selected",
        description="Path to the image directory, relative to the project root.",
    )
    force_rebuild: bool = Field(
        default=False,
        description="Wipe and rebuild the index from scratch.",
    )
    split_mode: str = Field(
        default="auto",
        description="'auto' = YOLO + grid fallback; 'yolo' = strict YOLO; 'grid' = tiles only.",
    )

    @field_validator("split_mode")
    @classmethod
    def validate_split_mode(cls, v: str) -> str:
        if v not in ("auto", "yolo", "grid"):
            raise ValueError("split_mode must be 'auto', 'yolo', or 'grid'")
        return v


class AddImagesRequest(BaseModel):
    image_paths: list[str] = Field(description="Absolute paths to image files to index.")
    split_mode: str = Field(default="auto", pattern="^(auto|yolo|grid)$")


class TextSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=50)
    search_scope: str = Field(default=config.DEFAULT_SEARCH_SCOPE)
    whole_weight: float = Field(default=config.WHOLE_IMAGE_WEIGHT, ge=0.0, le=1.0)
    split_mode: str = Field(default="auto")
    request_id: Optional[str] = Field(default=None, max_length=128)

    @field_validator("search_scope")
    @classmethod
    def validate_scope(cls, v: str) -> str:
        if v not in ("whole", "regions", "both"):
            raise ValueError("search_scope must be 'whole', 'regions', or 'both'")
        return v

    @field_validator("split_mode")
    @classmethod
    def validate_split_mode(cls, v: str) -> str:
        if v not in ("auto", "yolo", "grid"):
            raise ValueError("split_mode must be 'auto', 'yolo', or 'grid'")
        return v


class RegionResult(BaseModel):
    name: str
    image_url: str
    share_image_id: Optional[str]
    share_url: Optional[str]
    score: float
    matched_bbox: Optional[list[float]]
    class_label: Optional[str]
    match_type: str
    confidence: float
    rank: int


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parents[2]


def _get_model():
    """Reuse the model cached by the existing api.py module."""
    try:
        from visionsearch_web_ui import api as main_api
        return main_api.get_model()
    except Exception:
        from models.clip_model import load_clip_model
        return load_clip_model()


def _to_region_results(raw: list[dict]) -> list[RegionResult]:
    """Convert search_service output to Pydantic response models."""
    out = []
    for item in raw:
        ppath = Path(item["parent_path"])
        try:
            resolved_path = ppath.resolve()
            rel = resolved_path.relative_to(_ROOT.resolve()).as_posix()
            image_url = f"/image/{quote(rel, safe='/')}"
        except (OSError, ValueError):
            resolved_path = ppath
            image_url = f"/image/{ppath.name}"

        share_image_id = image_id_for_path(resolved_path)
        out.append(
            RegionResult(
                name=ppath.name,
                image_url=image_url,
                share_image_id=share_image_id,
                share_url=(
                    f"/api/share/images/{share_image_id}"
                    if PUBLIC_SHARING_ENABLED and share_image_id
                    else None
                ),
                score=item["score"],
                matched_bbox=item.get("matched_bbox"),
                class_label=item.get("class_label"),
                match_type=item.get("match_type", "whole"),
                confidence=item.get("confidence", 0.0),
                rank=item["rank"],
            )
        )
    return out


def _read_upload_image(file: UploadFile) -> Image.Image:
    """Read and validate an uploaded image file."""
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload a valid image file.")
    try:
        return Image.open(io.BytesIO(file.file.read())).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not open the uploaded image.")


# ---------------------------------------------------------------------------
# /index routes
# ---------------------------------------------------------------------------


@region_router.post("/index/build", summary="Build or rebuild the region index")
def api_build_index(body: BuildIndexRequest):
    """Runs the full region-indexing pipeline. This may take several minutes."""
    image_dir = (_ROOT / body.image_dir).resolve()
    if not image_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"Image directory not found: {image_dir}")

    try:
        model = _get_model()
        summary = build_region_index(
            image_dir,
            model,
            force_rebuild=body.force_rebuild,
            split_mode=body.split_mode,
        )
        invalidate_region_index_cache()
        return {"status": "ok", **summary}
    except Exception as exc:
        logger.exception("Region index build failed.")
        raise HTTPException(status_code=500, detail=str(exc))


@region_router.post("/index/add", summary="Incrementally add images to the region index")
def api_add_images(body: AddImagesRequest):
    """Add a list of images without rebuilding the entire index."""
    paths = [Path(p) for p in body.image_paths]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"These image files do not exist: {[str(p) for p in missing[:5]]}"
        )

    try:
        model = _get_model()
        summary = add_images_to_region_index(
            paths,
            model,
            split_mode=body.split_mode,
        )
        invalidate_region_index_cache()
        return {"status": "ok", **summary}
    except Exception as exc:
        logger.exception("Region index add failed.")
        raise HTTPException(status_code=500, detail=str(exc))


@region_router.get("/index/stats", summary="Region index statistics")
def api_index_stats():
    """Return FAISS vector count, SQLite summary, and file sizes."""
    return get_index_stats()


# ---------------------------------------------------------------------------
# /search routes
# ---------------------------------------------------------------------------


@region_router.post("/search/text", summary="Text-to-region search", response_model=list[RegionResult])
def api_text_search(body: TextSearchRequest, response: FastAPIResponse):
    """Embed the text query with CLIP and search the region index."""
    request_id = _resolve_request_id(body.request_id)
    response.headers["X-Request-ID"] = request_id
    model = _get_model()
    try:
        embedding = model.encode_text(body.query.strip())
    except Exception as exc:
        logger.exception("Region text query embedding failed request_id=%s", request_id)
        raise HTTPException(status_code=500, detail=f"Embedding failed: {exc}")

    try:
        raw = search_regions(
            embedding,
            search_scope=body.search_scope,
            top_k=body.top_k,
            whole_weight=body.whole_weight,
            region_crop_type=body.split_mode if body.split_mode in ("yolo", "grid") else None,
            query_label=body.query.strip(),
            query_dimensions=None,
            request_id=request_id,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    logger.info(
        "Region text search response request_id=%s query_label=%s result_count=%d",
        request_id,
        body.query.strip(),
        len(raw),
    )

    return _to_region_results(raw)


@region_router.post("/search/image/detect", summary="Detect regions in an uploaded image")
async def api_detect(
    file: UploadFile = File(...),
    split_mode: str = Form(default="yolo"),
):
    """Detect regions in an uploaded image and return mode/fallback metadata.

    The ``region_id`` values in the response can be passed to
    ``POST /api/region/search/image`` to search with a specific crop.
    """
    if split_mode not in ("yolo", "grid", "auto"):
        raise HTTPException(
            status_code=422,
            detail="split_mode must be 'yolo', 'grid', or 'auto'.",
        )
    image = _read_upload_image(file)
    try:
        result = detect_query_regions(image, split_mode=split_mode)
    except Exception as exc:
        logger.exception("Region detection failed.")
        if split_mode == "yolo":
            raise HTTPException(
                status_code=503,
                detail=(
                    "YOLO detection is unavailable. Check that Ultralytics and "
                    f"the configured weights are installed and loadable: {exc}"
                ),
            ) from exc
        raise HTTPException(
            status_code=500,
            detail=f"Region detection failed in {split_mode} mode: {exc}",
        ) from exc
    result["total"] = len(result["regions"])
    logger.info(
        "Region detection request split_mode=%s used_mode=%s fallback=%s "
        "count=%d classes=%s",
        result["requested_mode"],
        result["used_mode"],
        result["fallback"],
        result["total"],
        [
            {
                "class": region["class_label"],
                "confidence": region["confidence"],
                "bbox": region["bbox"],
            }
            for region in result["regions"]
        ],
    )
    return result


@region_router.post("/search/image", summary="Image-to-region search", response_model=list[RegionResult])
async def api_image_search(
    response: FastAPIResponse,
    file: UploadFile = File(...),
    top_k: int = Form(default=10),
    search_scope: str = Form(default=config.DEFAULT_SEARCH_SCOPE),
    whole_weight: float = Form(default=config.WHOLE_IMAGE_WEIGHT),
    region_id: Optional[int] = Form(default=None),
    bbox_x1: Optional[float] = Form(default=None),
    bbox_y1: Optional[float] = Form(default=None),
    bbox_x2: Optional[float] = Form(default=None),
    bbox_y2: Optional[float] = Form(default=None),
    query_label: Optional[str] = Form(default=None),
    query_crop_type: Optional[str] = Form(default=None),
    request_id: Optional[str] = Form(default=None, max_length=128),
    split_mode: str = Form(default="auto"),
):
    """Search the region index using an uploaded image.

    Three query modes (mutually exclusive, checked in order):
    1. **Custom bbox** — user draws a box; ``bbox_x1/y1/x2/y2`` are set.
    2. **Crop by region_id** — YOLO/grid is run; crop ``region_id`` is used.
    3. **Whole image** — the full uploaded image is embedded.
    """
    if search_scope not in ("whole", "regions", "both"):
        raise HTTPException(status_code=400, detail="Invalid search_scope.")
    if split_mode not in ("yolo", "grid", "auto"):
        raise HTTPException(
            status_code=422,
            detail="split_mode must be 'yolo', 'grid', or 'auto'.",
        )
    if query_crop_type not in (None, "yolo", "grid"):
        raise HTTPException(
            status_code=422,
            detail="query_crop_type must be 'yolo', 'grid', or omitted.",
        )
    request_id = _resolve_request_id(request_id)
    response.headers["X-Request-ID"] = request_id

    image = _read_upload_image(file)
    img_w, img_h = image.size
    model = _get_model()
    bbox_values = (bbox_x1, bbox_y1, bbox_x2, bbox_y2)
    if any(value is not None for value in bbox_values) and not all(
        value is not None for value in bbox_values
    ):
        raise HTTPException(status_code=400, detail="All four bounding-box coordinates are required.")
    if region_id is not None and region_id < 0:
        raise HTTPException(status_code=400, detail="region_id must be non-negative.")

    # Determine query region
    query_region_id = region_id
    query_box = None
    if all(value is not None for value in bbox_values):
        from visionsearch_web_ui.region.region_splitter import clamp_bbox
        coordinates = (bbox_x1, bbox_y1, bbox_x2, bbox_y2)
        if not all(math.isfinite(value) for value in coordinates if value is not None):
            raise HTTPException(status_code=400, detail="Bounding-box coordinates must be finite.")
        x1, y1, x2, y2 = clamp_bbox(
            bbox_x1,
            bbox_y1,
            bbox_x2,
            bbox_y2,
            img_w,
            img_h,
        )
        if x2 <= x1 or y2 <= y1:
            raise HTTPException(status_code=400, detail="Bounding box must have positive width and height.")
        query_box = RegionBox(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            class_label=query_label or "custom_bbox",
            confidence=1.0 if query_label == "custom_bbox" else 0.0,
            crop_type=query_crop_type or (split_mode if split_mode in ("yolo", "grid") else "yolo"),
        )
        query_image = crop_region(image, query_box)
    elif region_id is not None:
        from visionsearch_web_ui.region.detector import detect_regions
        from visionsearch_web_ui.region.region_splitter import crop_region as _crop
        try:
            boxes = detect_regions(image, split_mode=split_mode)
        except Exception as exc:
            logger.exception("Query image detection failed in %s mode.", split_mode)
            if split_mode == "yolo":
                raise HTTPException(
                    status_code=503,
                    detail=f"YOLO detection is unavailable: {exc}",
                ) from exc
            raise HTTPException(
                status_code=500,
                detail=f"Region detection failed in {split_mode} mode: {exc}",
            ) from exc
        if region_id < 0 or region_id >= len(boxes):
            raise HTTPException(
                status_code=400,
                detail=f"region_id {region_id} is out of range (detected {len(boxes)} regions).",
            )
        query_box = boxes[region_id]
        if query_crop_type is not None and query_box.crop_type != query_crop_type:
            raise HTTPException(
                status_code=409,
                detail="The selected region no longer matches the detected crop type. Detect regions again.",
            )
        query_image = _crop(image, query_box)
    else:
        query_image = image

    crop_dimensions = query_image.size
    object_label = query_label or (query_box.class_label if query_box else "whole_image")
    region_crop_type = (
        query_box.crop_type if query_box is not None
        else query_crop_type or (split_mode if split_mode in ("yolo", "grid") else None)
    )
    logger.info(
        "Region image query request_id=%s region_id=%s object_label=%s crop_dimensions=%sx%s "
        "bbox=%s crop_type=%s split_mode=%s",
        request_id,
        query_region_id,
        object_label,
        crop_dimensions[0],
        crop_dimensions[1],
        [query_box.x1, query_box.y1, query_box.x2, query_box.y2] if query_box else None,
        region_crop_type,
        split_mode,
    )
    try:
        embedding = model.encode_image(query_image)
    except Exception as exc:
        logger.exception(
            "Region image query embedding failed request_id=%s object_label=%s",
            request_id,
            object_label,
        )
        raise HTTPException(status_code=500, detail=f"Embedding failed: {exc}")
    embedding_array = np.asarray(embedding, dtype=np.float32)
    embedding_fingerprint = hashlib.sha256(embedding_array.tobytes()).hexdigest()[:12]
    embedding_norm = float(np.linalg.norm(embedding_array))
    logger.info(
        "Region image query embedded request_id=%s object_label=%s embedding_dimension=%d "
        "embedding_norm=%.8f embedding_fingerprint=%s",
        request_id,
        object_label,
        embedding_array.size,
        embedding_norm,
        embedding_fingerprint,
    )

    try:
        raw = search_regions(
            embedding_array,
            search_scope=search_scope,
            top_k=max(1, min(top_k, 50)),
            whole_weight=whole_weight,
            region_crop_type=region_crop_type,
            query_region_id=query_region_id,
            query_label=object_label,
            query_dimensions=crop_dimensions,
            request_id=request_id,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    logger.info(
        "Region image search response request_id=%s region_id=%s object_label=%s "
        "result_count=%d top_parent_paths=%s",
        request_id,
        query_region_id,
        object_label,
        len(raw),
        [item["parent_path"] for item in raw[:10]],
    )

    return _to_region_results(raw)


# ---------------------------------------------------------------------------
# Crop serving (on-the-fly from bbox)
# ---------------------------------------------------------------------------

@region_router.get(
    "/crop",
    summary="Serve a crop image by parent path + bbox",
    response_class=Response,
)
def serve_crop(
    parent_path: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
):
    """Generate and return a JPEG crop of the specified region on the fly."""
    p = Path(parent_path).resolve()
    try:
        p.relative_to(_ROOT.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid parent path.")
    if not p.is_file():
        raise HTTPException(status_code=404, detail="Parent image not found.")

    try:
        image = Image.open(p).convert("RGB")
        img_w, img_h = image.size
        box = RegionBox(
            x1=max(0.0, x1), y1=max(0.0, y1),
            x2=min(float(img_w), x2), y2=min(float(img_h), y2),
            class_label="crop", confidence=0.0, crop_type="yolo",
        )
        crop = crop_region(image, box)
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=85)
        return Response(content=buf.getvalue(), media_type="image/jpeg")
    except Exception as exc:
        logger.exception("Crop serving failed for %s.", p)
        raise HTTPException(status_code=500, detail=str(exc))
