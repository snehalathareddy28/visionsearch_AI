"""
index_manager.py — Build, update, and query the region FAISS index.

Responsibilities
----------------
- Build a region index from scratch (build_region_index).
- Incrementally add new images (add_images_to_region_index).
- Skip duplicates using SHA-256 file hashes stored in SQLite.
- Store whole-image + crop embeddings in a single FAISS IndexFlatIP.
- Persist the FAISS file and SQLite database atomically.

CLI usage (from project root)
------------------------------
    python -m visionsearch_web_ui.region.index_manager \\
        --image-dir dataset/caltech-101-selected

Flags
-----
--image-dir PATH       Directory of images to index (default: config)
--force                Rebuild from scratch even if the index already exists
--cache-crops          Save crop images to disk (default: False)
--split-mode auto|grid auto = YOLO first, grid fallback; grid = always grid
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import faiss
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from visionsearch_web_ui.region import config
from visionsearch_web_ui.region.detector import detect_regions
from visionsearch_web_ui.region.embedder import embed_crops
from visionsearch_web_ui.region.metadata_db import RegionMetadataDB, sha256_file
from visionsearch_web_ui.region.region_splitter import crop_region

if TYPE_CHECKING:
    from models.clip_model import ClipModel

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 512  # CLIP ViT-B/32 output dimension


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _open_or_create_faiss(path: Path) -> faiss.IndexFlatIP:
    """Load an existing FAISS index or create a fresh one."""
    if path.exists():
        index = faiss.read_index(str(path))
        if index.d != EMBEDDING_DIM:
            raise ValueError(
                f"Existing FAISS index has dimension {index.d}, expected {EMBEDDING_DIM}."
            )
        logger.info("Loaded existing region FAISS index (%d vectors).", index.ntotal)
        return index
    logger.info("Creating new region FAISS index (dim=%d).", EMBEDDING_DIM)
    return faiss.IndexFlatIP(EMBEDDING_DIM)


def _save_faiss(index: faiss.IndexFlatIP, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        faiss.write_index(index, str(temporary_path))
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Core indexing logic for a single image
# ---------------------------------------------------------------------------


def index_one_image(
    image_path: Path,
    clip_model: "ClipModel",
    whole_faiss_index: faiss.IndexFlatIP,
    region_faiss_index: faiss.IndexFlatIP,
    db: RegionMetadataDB,
    *,
    force_grid: bool = False,
    split_mode: str | None = None,
    cache_crops: bool = config.CACHE_CROPS,
) -> dict:
    """Index one image into separate whole-image and region FAISS indexes.

    Parameters
    ----------
    image_path : Path
        Absolute path to the image file.
    clip_model : ClipModel
        Loaded CLIP model wrapper.
    whole_faiss_index, region_faiss_index : faiss.IndexFlatIP
        Separate in-memory FAISS indexes.
    db : RegionMetadataDB
        The open metadata database.
    force_grid : bool
        Skip YOLO; always use grid splitting.
    cache_crops : bool
        Whether to save crop images to ``config.CROP_CACHE_DIR``.

    Returns
    -------
    dict
        Keys: ``added_vectors`` (int), ``num_regions`` (int), ``skipped`` (bool).
    """
    try:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        logger.warning("Cannot open image %s: %s", image_path, exc)
        return {"added_vectors": 0, "num_regions": 0, "skipped": True}

    file_hash = sha256_file(image_path)
    img_w, img_h = image.size

    mode = "grid" if force_grid else (split_mode or "auto")
    region_boxes = detect_regions(image, split_mode=mode)

    crops = [crop_region(image, box) for box in region_boxes]

    if cache_crops and crops:
        cache_root = config.CROP_CACHE_DIR / image_path.stem
        cache_root.mkdir(parents=True, exist_ok=True)
        for idx, (crop, box) in enumerate(zip(crops, region_boxes)):
            crop_name = f"{idx:03d}_{box.class_label}_{box.crop_type}.jpg"
            crop.save(cache_root / crop_name)

    embeddings = embed_crops([image, *crops], clip_model, show_progress=False)
    whole_vec = embeddings[0]
    whole_vec = np.asarray(whole_vec, dtype=np.float32).reshape(-1)
    whole_norm = np.linalg.norm(whole_vec)
    if whole_vec.shape != (EMBEDDING_DIM,) or not np.isfinite(whole_vec).all() or whole_norm <= 1e-12:
        raise ValueError(f"CLIP returned an invalid whole-image embedding for {image_path}")
    whole_vec /= whole_norm

    region_embeddings = embeddings[1:]
    if region_embeddings.shape != (len(crops), EMBEDDING_DIM) or not np.isfinite(region_embeddings).all():
        raise ValueError(f"CLIP returned invalid region embeddings for {image_path}")

    parent_id = db.add_parent_image(image_path, file_hash, img_w, img_h)
    whole_faiss_id = int(whole_faiss_index.ntotal)
    if db.get_next_faiss_id("whole") != whole_faiss_id:
        raise RuntimeError("Whole-image FAISS IDs do not match SQLite metadata.")
    whole_faiss_index.add(whole_vec.reshape(1, -1))
    db.add_region_vector(
        db.get_next_vector_id(),
        parent_id,
        image_path,
        None,
        is_whole_image=True,
        index_type="whole",
        faiss_id=whole_faiss_id,
        file_hash=file_hash,
        image_width=img_w,
        image_height=img_h,
    )

    if crops:
        norms = np.linalg.norm(region_embeddings, axis=1, keepdims=True)
        if np.any(norms <= 1e-12):
            raise ValueError(f"CLIP returned an empty region embedding for {image_path}")
        region_embeddings /= norms
        region_start_id = int(region_faiss_index.ntotal)
        if db.get_next_faiss_id("region") != region_start_id:
            raise RuntimeError("Region FAISS IDs do not match SQLite metadata.")
        region_faiss_index.add(region_embeddings)
        for offset, box in enumerate(region_boxes):
            db.add_region_vector(
                db.get_next_vector_id(),
                parent_id,
                image_path,
                box,
                index_type="region",
                faiss_id=region_start_id + offset,
                file_hash=file_hash,
                image_width=img_w,
                image_height=img_h,
            )

    return {
        "added_vectors": 1 + len(crops),
        "whole_vectors": 1,
        "region_vectors": len(crops),
        "num_regions": len(crops),
        "skipped": False,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_region_index(
    image_dir: str | Path,
    clip_model: "ClipModel",
    *,
    force_rebuild: bool = False,
    force_grid: bool = False,
    split_mode: str | None = None,
    faiss_path: Path = config.REGION_FAISS_PATH,
    whole_faiss_path: Path = config.REGION_WHOLE_FAISS_PATH,
    db_path: Path = config.REGION_DB_PATH,
    cache_crops: bool = config.CACHE_CROPS,
) -> dict:
    """Index all images in *image_dir* and persist the region index.

    Stores whole-image and region vectors separately. Skips duplicate image
    content by SHA-256 while preserving incremental indexing.
    Set ``force_rebuild=True`` to wipe and rebuild from scratch.

    Returns
    -------
    dict
        Summary with ``indexed``, ``skipped``, ``total_vectors``.
    """
    from utils.image_utils import list_images

    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise ValueError(f"Image directory does not exist: {image_dir}")
    image_paths = list_images(image_dir)

    if not image_paths:
        raise ValueError(f"No supported images found in {image_dir}")

    if force_rebuild:
        for path in (faiss_path, whole_faiss_path, db_path):
            if path.exists():
                path.unlink()
        logger.info("Wiped existing region index for rebuild.")

    region_faiss_index = _open_or_create_faiss(faiss_path)
    whole_faiss_index = _open_or_create_faiss(whole_faiss_path)

    indexed_count = 0
    skipped_count = 0
    region_vectors_added = 0

    with RegionMetadataDB(db_path) as db:
        counts = db.get_vector_counts()
        if counts["whole"] != whole_faiss_index.ntotal:
            raise RuntimeError(
                f"Whole-image FAISS/SQLite mismatch: {whole_faiss_index.ntotal} "
                f"vectors in {whole_faiss_path}, {counts['whole']} metadata rows in {db_path}."
            )
        if counts["region"] != region_faiss_index.ntotal:
            raise RuntimeError(
                f"Region FAISS/SQLite mismatch: {region_faiss_index.ntotal} "
                f"vectors in {faiss_path}, {counts['region']} metadata rows in {db_path}."
            )
        known_hashes = db.get_indexed_hashes()

        for image_path in tqdm(image_paths, desc="Building region index", unit="img"):
            # Dedup: skip if SHA-256 already known.
            try:
                fhash = sha256_file(image_path)
            except OSError as exc:
                logger.warning("Cannot read %s: %s", image_path, exc)
                skipped_count += 1
                continue

            if fhash in known_hashes:
                skipped_count += 1
                continue

            result = index_one_image(
                image_path,
                clip_model,
                whole_faiss_index,
                region_faiss_index,
                db,
                force_grid=force_grid,
                split_mode=split_mode,
                cache_crops=cache_crops,
            )

            if result["skipped"]:
                skipped_count += 1
            else:
                indexed_count += 1
                region_vectors_added += result["region_vectors"]
                known_hashes.add(fhash)

    _save_faiss(whole_faiss_index, whole_faiss_path)
    _save_faiss(region_faiss_index, faiss_path)

    return {
        "indexed": indexed_count,
        "skipped": skipped_count,
        "whole_image_vectors": int(whole_faiss_index.ntotal),
        "region_vectors": int(region_faiss_index.ntotal),
        "total_vectors": int(whole_faiss_index.ntotal + region_faiss_index.ntotal),
        "region_vectors_added": region_vectors_added,
        "whole_faiss_path": str(whole_faiss_path),
        "region_faiss_path": str(faiss_path),
        "db_path": str(db_path),
    }


def add_images_to_region_index(
    image_paths: list[Path],
    clip_model: "ClipModel",
    *,
    force_grid: bool = False,
    split_mode: str | None = None,
    faiss_path: Path = config.REGION_FAISS_PATH,
    whole_faiss_path: Path = config.REGION_WHOLE_FAISS_PATH,
    db_path: Path = config.REGION_DB_PATH,
) -> dict:
    """Add a list of images incrementally to an existing region index.

    Creates the index if it does not yet exist.
    """
    region_faiss_index = _open_or_create_faiss(faiss_path)
    whole_faiss_index = _open_or_create_faiss(whole_faiss_path)
    indexed_count = 0
    skipped_count = 0
    region_vectors_added = 0

    with RegionMetadataDB(db_path) as db:
        counts = db.get_vector_counts()
        if counts["whole"] != whole_faiss_index.ntotal or counts["region"] != region_faiss_index.ntotal:
            raise RuntimeError("Region FAISS indexes and SQLite metadata are inconsistent.")
        known_hashes = db.get_indexed_hashes()

        for image_path in tqdm(image_paths, desc="Adding region images", unit="img"):
            try:
                fhash = sha256_file(image_path)
            except OSError:
                skipped_count += 1
                continue

            if fhash in known_hashes:
                skipped_count += 1
                continue

            result = index_one_image(
                image_path,
                clip_model,
                whole_faiss_index,
                region_faiss_index,
                db,
                force_grid=force_grid,
                split_mode=split_mode,
            )

            if result["skipped"]:
                skipped_count += 1
            else:
                indexed_count += 1
                region_vectors_added += result["region_vectors"]
                known_hashes.add(fhash)

    _save_faiss(whole_faiss_index, whole_faiss_path)
    _save_faiss(region_faiss_index, faiss_path)

    return {
        "indexed": indexed_count,
        "skipped": skipped_count,
        "whole_image_vectors": int(whole_faiss_index.ntotal),
        "region_vectors": int(region_faiss_index.ntotal),
        "total_vectors": int(whole_faiss_index.ntotal + region_faiss_index.ntotal),
        "region_vectors_added": region_vectors_added,
        "whole_faiss_path": str(whole_faiss_path),
        "region_faiss_path": str(faiss_path),
        "db_path": str(db_path),
    }


def get_index_stats(
    faiss_path: Path = config.REGION_FAISS_PATH,
    whole_faiss_path: Path = config.REGION_WHOLE_FAISS_PATH,
    db_path: Path = config.REGION_DB_PATH,
) -> dict:
    """Return both FAISS index counts and the SQLite consistency status."""
    missing = [
        str(path)
        for path in (whole_faiss_path, faiss_path, db_path)
        if not path.is_file()
    ]
    if missing:
        return {
            "status": "not_built",
            "database_status": "missing" if not db_path.is_file() else "present",
            "missing_artifacts": missing,
            "whole_faiss_path": str(whole_faiss_path),
            "region_faiss_path": str(faiss_path),
            "database_path": str(db_path),
        }

    whole_index = faiss.read_index(str(whole_faiss_path))
    region_index = faiss.read_index(str(faiss_path))
    with RegionMetadataDB(db_path) as db:
        db_stats = db.stats()
        vector_counts = db.get_vector_counts()
        integrity = db.conn.execute("PRAGMA integrity_check").fetchone()[0]

    whole_matches = vector_counts["whole"] == whole_index.ntotal
    regions_match = vector_counts["region"] == region_index.ntotal
    return {
        "status": "ready" if whole_matches and regions_match and integrity == "ok" else "inconsistent",
        "database_status": integrity,
        "whole_faiss_path": str(whole_faiss_path),
        "region_faiss_path": str(faiss_path),
        "database_path": str(db_path),
        "whole_image_vectors": int(whole_index.ntotal),
        "whole_faiss_vectors": int(whole_index.ntotal),
        "region_faiss_vectors": int(region_index.ntotal),
        "whole_faiss_size_mb": round(whole_faiss_path.stat().st_size / (1024 * 1024), 3),
        "region_faiss_size_mb": round(faiss_path.stat().st_size / (1024 * 1024), 3),
        "faiss_vectors_match_metadata": whole_matches and regions_match,
        **db_stats,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Build the VisionSearch region index.")
    parser.add_argument(
        "--image-dir",
        default=str(Path(__file__).resolve().parents[2] / "dataset" / "caltech-101-selected"),
        help="Directory of images to index",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild from scratch")
    parser.add_argument("--cache-crops", action="store_true", help="Save crop images to disk")
    parser.add_argument(
        "--split-mode",
        choices=["auto", "yolo", "grid"],
        default="auto",
        help="auto = YOLO with grid fallback; yolo = strict object detection; grid = only tiles",
    )
    args = parser.parse_args()

    from models.clip_model import load_clip_model

    model = load_clip_model()
    summary = build_region_index(
        args.image_dir,
        model,
        force_rebuild=args.force,
        split_mode=args.split_mode,
        cache_crops=args.cache_crops,
    )
    print(
        f"\nRegion index built.\n"
        f"  Indexed: {summary['indexed']} images\n"
        f"  Skipped (duplicate or error): {summary['skipped']}\n"
        f"  Whole-image vectors: {summary['whole_image_vectors']}\n"
        f"  Region vectors: {summary['region_vectors']}\n"
        f"  Whole-image FAISS: {summary['whole_faiss_path']}\n"
        f"  Region FAISS: {summary['region_faiss_path']}\n"
        f"  SQLite metadata: {summary['db_path']}\n"
    )
