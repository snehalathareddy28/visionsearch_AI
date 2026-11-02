"""
search_service.py — Region-aware FAISS search for VisionSearch AI.

Search scopes
-------------
"whole"   — query only whole-image vectors (is_whole_image = 1).
"regions" — query only crop vectors (is_whole_image = 0).
"both"    — query all vectors; combine best-region score and whole-image
             score with configurable weights.

Returned result dict (one per unique parent image)
---------------------------------------------------
{
  "parent_path": str,
  "score": float,          # combined / best score
  "matched_bbox": [x1,y1,x2,y2] | None,
  "class_label": str | None,
  "match_type": "whole" | "yolo" | "grid",
  "confidence": float,
  "rank": int,
}
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import faiss
import numpy as np

from visionsearch_web_ui.region import config
from visionsearch_web_ui.region.category_resolver import (
    category_adjustment,
    resolve_query_category,
)
from visionsearch_web_ui.region.metadata_db import RegionMetadataDB

if TYPE_CHECKING:
    from models.clip_model import ClipModel

logger = logging.getLogger(__name__)

_faiss_index: faiss.IndexFlatIP | None = None
_faiss_path_loaded: str | None = None
_loaded_indexes: dict[str, faiss.IndexFlatIP] = {}
_loaded_index_signatures: dict[str, tuple[int, int]] = {}


def _index_file_signature(faiss_path: Path) -> tuple[int, int]:
    """Return file metadata used to detect an index replaced at the same path."""
    stat = faiss_path.stat()
    return stat.st_mtime_ns, stat.st_size


# ---------------------------------------------------------------------------
# Index loading
# ---------------------------------------------------------------------------


def load_region_index(
    faiss_path: Path = config.REGION_FAISS_PATH,
) -> faiss.IndexFlatIP:
    """Load (and cache in-process) the region FAISS index."""
    global _faiss_index, _faiss_path_loaded
    path_str = str(faiss_path)
    if not faiss_path.is_file():
        raise FileNotFoundError(
            f"Region FAISS index not found at {faiss_path}. "
            "Run: python -m visionsearch_web_ui.region.index_manager"
        )
    signature = _index_file_signature(faiss_path)
    if (
        _faiss_index is None
        or _faiss_path_loaded != path_str
        or _loaded_index_signatures.get(path_str) != signature
    ):
        _faiss_index = faiss.read_index(path_str)
        _faiss_path_loaded = path_str
        _loaded_indexes[path_str] = _faiss_index
        _loaded_index_signatures[path_str] = signature
        logger.info("Loaded region FAISS index (%d vectors).", _faiss_index.ntotal)
    return _faiss_index


def load_whole_image_index(
    faiss_path: Path = config.REGION_WHOLE_FAISS_PATH,
) -> faiss.IndexFlatIP:
    """Load (and cache) the region retrieval whole-image FAISS index."""
    path_str = str(faiss_path)
    if not faiss_path.is_file():
        raise FileNotFoundError(
            f"Whole-image region index is missing at {faiss_path}. "
            "Build the region indexes with "
            "`python -m visionsearch_web_ui.region.index_manager`."
        )
    signature = _index_file_signature(faiss_path)
    if (
        path_str not in _loaded_indexes
        or _loaded_index_signatures.get(path_str) != signature
    ):
        _loaded_indexes[path_str] = faiss.read_index(path_str)
        _loaded_index_signatures[path_str] = signature
        logger.info(
            "Loaded whole-image region FAISS index (%d vectors).",
            _loaded_indexes[path_str].ntotal,
        )
    return _loaded_indexes[path_str]


def invalidate_region_index_cache() -> None:
    """Force the next search to reload the index from disk (after re-indexing)."""
    global _faiss_index, _faiss_path_loaded, _loaded_indexes, _loaded_index_signatures
    _faiss_index = None
    _faiss_path_loaded = None
    _loaded_indexes = {}
    _loaded_index_signatures = {}


# ---------------------------------------------------------------------------
# Core search
# ---------------------------------------------------------------------------


def _raw_faiss_search(
    query_vec: np.ndarray,
    faiss_index: faiss.IndexFlatIP,
    k: int,
) -> list[tuple[int, float]]:
    """Return up to *k* (vector_id, score) pairs from a FAISS search."""
    if faiss_index.ntotal == 0:
        return []
    k = min(k, faiss_index.ntotal)
    scores, ids = faiss_index.search(query_vec.reshape(1, -1), k)
    return [
        (int(vid), float(s))
        for s, vid in zip(scores[0], ids[0])
        if vid >= 0
    ]


def search_regions(
    query_embedding: np.ndarray,
    *,
    search_scope: str = config.DEFAULT_SEARCH_SCOPE,
    top_k: int = config.DEFAULT_TOP_K,
    whole_weight: float = config.WHOLE_IMAGE_WEIGHT,
    region_crop_type: str | None = None,
    request_id: str | None = None,
    query_region_id: int | None = None,
    query_label: str | None = None,
    query_dimensions: tuple[int, int] | None = None,
    faiss_path: Path = config.REGION_FAISS_PATH,
    db_path: Path = config.REGION_DB_PATH,
    whole_faiss_path: Path | None = None,
) -> list[dict]:
    """Search the region index and return grouped, ranked results.

    Parameters
    ----------
    query_embedding : np.ndarray
        L2-normalised float32 CLIP embedding, shape (512,) or (1, 512).
    search_scope : str
        One of ``"whole"``, ``"regions"``, or ``"both"``.
    top_k : int
        Maximum number of unique parent images to return.
    whole_weight : float
        Weight for the whole-image score when ``search_scope == "both"``.
        Region best-score weight = ``1 - whole_weight``.

    Returns
    -------
    list[dict]
        Up to *top_k* result dicts, each describing the best-matching region
        of a unique parent image, sorted by descending score.
    """
    if search_scope not in ("whole", "regions", "both"):
        raise ValueError(f"search_scope must be 'whole', 'regions', or 'both'; got '{search_scope}'")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if not 0.0 <= whole_weight <= 1.0:
        raise ValueError("whole_weight must be in [0, 1]")
    if region_crop_type not in (None, "yolo", "grid"):
        raise ValueError("region_crop_type must be 'yolo', 'grid', or None")

    use_default_indexes = (
        faiss_path == config.REGION_FAISS_PATH
        and db_path == config.REGION_DB_PATH
    )
    if whole_faiss_path is None and use_default_indexes:
        whole_faiss_path = config.REGION_WHOLE_FAISS_PATH
    embedding_dimension = np.asarray(query_embedding).size
    embedding_array = np.ascontiguousarray(query_embedding, dtype=np.float32)
    embedding_norm = float(np.linalg.norm(embedding_array))
    embedding_fingerprint = hashlib.sha256(embedding_array.tobytes()).hexdigest()[:12]
    logger.info(
        "Region search query request_id=%s region_id=%s object_label=%s crop_dimensions=%s "
        "embedding_dimension=%d embedding_norm=%.8f embedding_fingerprint=%s "
        "search_scope=%s region_crop_type=%s "
        "faiss_indexes=%s",
        request_id,
        query_region_id,
        query_label,
        query_dimensions,
        embedding_dimension,
        embedding_norm,
        embedding_fingerprint,
        search_scope,
        region_crop_type,
        {
            "whole": str(whole_faiss_path)
            if whole_faiss_path is not None and search_scope in ("whole", "both")
            else None,
            "region": str(faiss_path)
            if search_scope in ("regions", "both")
            else None,
        },
    )
    if whole_faiss_path is not None:
        return _search_separate_indexes(
            query_embedding,
            search_scope=search_scope,
            top_k=top_k,
            whole_weight=whole_weight,
            region_crop_type=region_crop_type,
            request_id=request_id,
            query_region_id=query_region_id,
            query_label=query_label,
            query_dimensions=query_dimensions,
            region_faiss_path=faiss_path,
            whole_faiss_path=whole_faiss_path,
            db_path=db_path,
        )

    query_vec = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(query_vec)
    if norm > 1e-12:
        query_vec = query_vec / norm

    faiss_index = load_region_index(faiss_path)

    if region_crop_type is None:
        candidate_k = min(faiss_index.ntotal, top_k * 20)
        raw_hits = _raw_faiss_search(query_vec, faiss_index, candidate_k)
    else:
        target_parents = top_k if search_scope in ("regions", "whole") else top_k * 20
        raw_hits = _filtered_legacy_hits(
            query_vec,
            faiss_index,
            db_path=db_path,
            crop_type=region_crop_type,
            target_parents=target_parents,
            search_scope=search_scope,
        )

    if not raw_hits:
        return []

    vector_ids = [vid for vid, _ in raw_hits]
    score_map: dict[int, float] = {vid: s for vid, s in raw_hits}

    # Fetch metadata for all hit vector IDs.
    with RegionMetadataDB(db_path) as db:
        meta_rows: list[dict] = []
        for vid in vector_ids:
            row = db.get_vector(vid)
            if row is not None:
                meta_rows.append(row)
        known_labels = (
            db.get_yolo_labels() if config.REGION_CATEGORY_AWARE_ENABLED else set()
        )

    query_category = (
        resolve_query_category(query_label, known_labels, config.CATEGORY_ALIASES)
        if config.REGION_CATEGORY_AWARE_ENABLED
        else None
    )

    _log_candidate_hits(
        query_region_id,
        query_label,
        query_dimensions,
        search_scope,
        region_crop_type,
        request_id,
        [
            (row["faiss_id"], score_map[row["vector_id"]], row)
            for row in meta_rows
            if row["vector_id"] in score_map
        ],
    )

    if not meta_rows:
        return []

    # ---------- Group by parent image and combine scores ----------
    # parent_path → {"whole_score": float, "best_region": dict, "region_score": float}
    grouped: dict[int, dict] = {}

    for row in meta_rows:
        vid = row["vector_id"]
        score = score_map.get(vid, 0.0)
        ppath = row["parent_path"]
        parent_id = int(row["parent_image_id"])

        if parent_id not in grouped:
            grouped[parent_id] = {
                "parent_image_id": parent_id,
                "parent_path": ppath,
                "whole_score": 0.0,
                "region_score": 0.0,
                "region_adjustment": 0.0,
                "region_rank_score": None,
                "best_region_row": None,
            }

        entry = grouped[parent_id]

        if row["is_whole_image"]:
            if score > entry["whole_score"]:
                entry["whole_score"] = score
        else:
            adjustment = (
                category_adjustment(
                    row,
                    query_category,
                    known_labels,
                    config.CATEGORY_ALIASES,
                    min_confidence=config.REGION_CATEGORY_MIN_CONFIDENCE,
                    bonus=config.REGION_CATEGORY_BONUS,
                    penalty=config.REGION_CATEGORY_PENALTY,
                )
                if config.REGION_CATEGORY_AWARE_ENABLED
                else 0.0
            )
            adjusted_score = score + adjustment
            if (
                entry["region_rank_score"] is None
                or adjusted_score > entry["region_rank_score"]
            ):
                entry["region_score"] = score
                entry["region_adjustment"] = adjustment
                entry["region_rank_score"] = adjusted_score
                entry["best_region_row"] = row

    # ---------- Apply scope filter and compute final score ----------
    results: list[dict] = []
    region_weight = 1.0 - whole_weight

    for entry in grouped.values():
        ws = entry["whole_score"]
        rs = entry["region_score"]
        region_rank_score = rs + entry["region_adjustment"]
        best_row = entry["best_region_row"]

        if search_scope == "whole":
            if ws == 0.0:
                continue
            final_score = ws
            matched_bbox = None
            class_label = "whole"
            match_type = "whole"
            confidence = 0.0
        elif search_scope == "regions":
            if rs == 0.0 or best_row is None:
                continue
            final_score = region_rank_score
            matched_bbox = _bbox_from_row(best_row)
            class_label = best_row["class_label"]
            match_type = best_row["crop_type"]
            confidence = best_row["confidence"] or 0.0
        else:  # "both"
            final_score = whole_weight * ws + region_weight * region_rank_score
            if best_row is not None and region_rank_score >= ws:
                matched_bbox = _bbox_from_row(best_row)
                class_label = best_row["class_label"]
                match_type = best_row["crop_type"]
                confidence = best_row["confidence"] or 0.0
            else:
                matched_bbox = None
                class_label = "whole"
                match_type = "whole"
                confidence = 0.0

        results.append({
            "parent_path": entry["parent_path"],
            "score": round(float(final_score), 6),
            "matched_bbox": matched_bbox,
            "class_label": class_label,
            "match_type": match_type,
            "confidence": round(float(confidence), 4),
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    results = results[:top_k]

    for rank, result in enumerate(results, start=1):
        result["rank"] = rank

    return results


def _search_separate_indexes(
    query_embedding: np.ndarray,
    *,
    search_scope: str,
    top_k: int,
    whole_weight: float,
    region_crop_type: str | None,
    request_id: str | None,
    query_region_id: int | None,
    query_label: str | None,
    query_dimensions: tuple[int, int] | None,
    region_faiss_path: Path,
    whole_faiss_path: Path,
    db_path: Path,
) -> list[dict]:
    """Search split indexes and group hits by parent image."""
    query_vec = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
    if query_vec.shape != (512,) or not np.isfinite(query_vec).all():
        raise ValueError("query_embedding must be a finite 512-dimensional vector")
    norm = np.linalg.norm(query_vec)
    if norm <= 1e-12:
        raise ValueError("query_embedding must have a non-zero norm")
    if not np.isclose(norm, 1.0, rtol=0.0, atol=1e-6):
        query_vec = query_vec / norm
    faiss_query_fingerprint = hashlib.sha256(
        np.ascontiguousarray(query_vec).tobytes()
    ).hexdigest()[:12]
    logger.info(
        "Region FAISS query request_id=%s query_fingerprint=%s query_norm=%.8f "
        "embedding_dimension=%d whole_index=%s region_index=%s",
        request_id,
        faiss_query_fingerprint,
        float(np.linalg.norm(query_vec)),
        query_vec.size,
        str(whole_faiss_path) if search_scope in ("whole", "both") else None,
        str(region_faiss_path) if search_scope in ("regions", "both") else None,
    )

    whole_hits: list[tuple[int, float]] = []
    region_hits: list[tuple[int, float]] = []
    if search_scope in ("whole", "both"):
        whole_hits = _raw_faiss_search(
            query_vec,
            load_whole_image_index(whole_faiss_path),
            top_k * 20,
        )
    if search_scope in ("regions", "both"):
        try:
            region_index = load_region_index(region_faiss_path)
            if region_crop_type is None:
                region_hits = _raw_faiss_search(query_vec, region_index, top_k * 20)
            else:
                target_parents = top_k if search_scope == "regions" else top_k * 20
                if config.REGION_CATEGORY_AWARE_ENABLED:
                    target_parents = max(target_parents, top_k * 20)
                region_hits = _filtered_region_hits(
                    query_vec,
                    region_index,
                    db_path=db_path,
                    crop_type=region_crop_type,
                    target_parents=target_parents,
                    deduplicate_parents=not config.REGION_CATEGORY_AWARE_ENABLED,
                )
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"Region FAISS index is missing at {region_faiss_path}. "
                "Build the region indexes with "
                "`python -m visionsearch_web_ui.region.index_manager`."
            ) from error

    with RegionMetadataDB(db_path) as db:
        whole_rows = [
            (db.get_vector_by_faiss_id("whole", vector_id), score)
            for vector_id, score in whole_hits
        ]
        region_rows = [
            (db.get_vector_by_faiss_id("region", vector_id), score)
            for vector_id, score in region_hits
        ]
        known_labels = (
            db.get_yolo_labels()
            if config.REGION_CATEGORY_AWARE_ENABLED
            and search_scope in ("regions", "both")
            else set()
        )

    query_category = (
        resolve_query_category(query_label, known_labels, config.CATEGORY_ALIASES)
        if config.REGION_CATEGORY_AWARE_ENABLED
        and search_scope in ("regions", "both")
        else None
    )

    candidate_hits = [
        (vector_id, score, row)
        for (vector_id, score), (row, _) in zip(region_hits, region_rows)
        if row is not None
    ] + [
        (vector_id, score, row)
        for (vector_id, score), (row, _) in zip(whole_hits, whole_rows)
        if row is not None
    ]
    _log_candidate_hits(
        query_region_id,
        query_label,
        query_dimensions,
        search_scope,
        region_crop_type,
        request_id,
        candidate_hits,
    )
    logger.info(
        "Region FAISS candidate counts request_id=%s whole=%d region=%d "
        "region_crop_type=%s region_candidate_crop_types=%s",
        request_id,
        len(whole_hits),
        len(region_hits),
        region_crop_type,
        sorted({row["crop_type"] for row, _ in region_rows if row is not None}),
    )
    if not whole_hits and not region_hits:
        return []

    grouped: dict[int, dict] = {}
    for (vector_id, _), (row, score) in zip(whole_hits, whole_rows):
        if row is None:
            logger.error("Whole-image FAISS row %d has no SQLite metadata.", vector_id)
            continue
        parent_id = int(row["parent_image_id"])
        entry = grouped.setdefault(
            parent_id,
            {
                "parent_path": row["parent_path"],
                "whole_score": None,
                "region_score": None,
                "region_adjustment": 0.0,
                "region_rank_score": None,
                "best_region": None,
            },
        )
        entry["whole_score"] = float(score)

    for (vector_id, _), (row, score) in zip(region_hits, region_rows):
        if row is None:
            logger.error("Region FAISS row %d has no SQLite metadata.", vector_id)
            continue
        parent_id = int(row["parent_image_id"])
        entry = grouped.setdefault(
            parent_id,
            {
                "parent_path": row["parent_path"],
                "whole_score": None,
                "region_score": None,
                "region_adjustment": 0.0,
                "region_rank_score": None,
                "best_region": None,
            },
        )
        adjustment = (
            category_adjustment(
                row,
                query_category,
                known_labels,
                config.CATEGORY_ALIASES,
                min_confidence=config.REGION_CATEGORY_MIN_CONFIDENCE,
                bonus=config.REGION_CATEGORY_BONUS,
                penalty=config.REGION_CATEGORY_PENALTY,
            )
            if config.REGION_CATEGORY_AWARE_ENABLED
            else 0.0
        )
        adjusted_score = float(score) + adjustment
        if (
            entry["region_rank_score"] is None
            or adjusted_score > entry["region_rank_score"]
        ):
            entry["region_score"] = float(score)
            entry["region_adjustment"] = adjustment
            entry["region_rank_score"] = adjusted_score
            entry["best_region"] = row

    results: list[dict] = []
    for entry in grouped.values():
        whole_score = entry["whole_score"]
        region_score = entry["region_score"]
        region_rank_score = (
            None
            if region_score is None
            else region_score + entry["region_adjustment"]
        )
        best_region = entry["best_region"]
        if search_scope == "whole":
            if whole_score is None:
                continue
            final_score = whole_score
            matched_row = None
        elif search_scope == "regions":
            if region_score is None or best_region is None:
                continue
            final_score = region_rank_score
            matched_row = best_region
        else:
            whole_contribution = whole_weight * (whole_score or 0.0)
            region_contribution = (1.0 - whole_weight) * (
                region_rank_score or 0.0
            )
            final_score = whole_contribution + region_contribution
            matched_row = (
                best_region
                if best_region is not None
                and (whole_score is None or region_contribution >= whole_contribution)
                else None
            )

        results.append(
            {
                "parent_path": entry["parent_path"],
                "score": round(float(final_score), 6),
                "matched_bbox": _bbox_from_row(matched_row) if matched_row else None,
                "class_label": (
                    matched_row["class_label"] if matched_row else "whole"
                ),
                "match_type": matched_row["crop_type"] if matched_row else "whole",
                "confidence": (
                    round(float(matched_row["confidence"] or 0.0), 4)
                    if matched_row
                    else 0.0
                ),
            }
        )

    results.sort(key=lambda result: result["score"], reverse=True)
    results = results[:top_k]
    for rank, result in enumerate(results, start=1):
        result["rank"] = rank
    return results


def _log_candidate_hits(
    query_region_id: int | None,
    query_label: str | None,
    query_dimensions: tuple[int, int] | None,
    search_scope: str,
    region_crop_type: str | None,
    request_id: str | None,
    hits: list[tuple[int | None, float, dict]],
) -> None:
    """Log the top FAISS candidates with their resolved image and region metadata."""
    candidates = [
        {
            "parent_image_id": int(row["parent_image_id"]),
            "faiss_id": faiss_id,
            "index_type": row["index_type"],
            "similarity": round(float(score), 6),
            "class_label": row["class_label"],
            "crop_type": row["crop_type"],
            "bbox": _bbox_from_row(row),
        }
        for faiss_id, score, row in hits
    ]
    candidates.sort(key=lambda candidate: candidate["similarity"], reverse=True)
    logger.info(
        "Region search candidates request_id=%s query_region_id=%s object_label=%s crop_dimensions=%s "
        "scope=%s region_crop_type=%s top_candidates=%s",
        request_id,
        query_region_id,
        query_label,
        query_dimensions,
        search_scope,
        region_crop_type,
        candidates[:10],
    )


def _filtered_region_hits(
    query_vec: np.ndarray,
    faiss_index: faiss.IndexFlatIP,
    *,
    db_path: Path,
    crop_type: str,
    target_parents: int,
    deduplicate_parents: bool = True,
) -> list[tuple[int, float]]:
    """Return top hits of one crop type, expanding until enough candidates match."""
    if faiss_index.ntotal == 0:
        return []

    requested = min(faiss_index.ntotal, max(target_parents, 20))
    matching_hits: list[tuple[int, float]] = []
    while True:
        raw_hits = _raw_faiss_search(query_vec, faiss_index, requested)
        matching_hits = []
        seen_parent_ids: set[int] = set()
        with RegionMetadataDB(db_path) as db:
            for faiss_id, score in raw_hits:
                row = db.get_vector_by_faiss_id("region", faiss_id)
                if row is None:
                    logger.error("Region FAISS row %d has no SQLite metadata.", faiss_id)
                    continue
                if row["crop_type"] != crop_type:
                    continue
                parent_id = int(row["parent_image_id"])
                if deduplicate_parents and parent_id in seen_parent_ids:
                    continue
                seen_parent_ids.add(parent_id)
                matching_hits.append((faiss_id, score))

        if len(matching_hits) >= target_parents or requested == faiss_index.ntotal:
            break
        requested = min(faiss_index.ntotal, requested * 2)

    return matching_hits[:target_parents]


def _filtered_legacy_hits(
    query_vec: np.ndarray,
    faiss_index: faiss.IndexFlatIP,
    *,
    db_path: Path,
    crop_type: str,
    target_parents: int,
    search_scope: str,
) -> list[tuple[int, float]]:
    """Filter candidates from legacy combined indexes while retaining whole-image hits."""
    if faiss_index.ntotal == 0:
        return []

    requested = min(faiss_index.ntotal, max(target_parents, 20))
    matching_hits: list[tuple[int, float]] = []
    while True:
        raw_hits = _raw_faiss_search(query_vec, faiss_index, requested)
        matching_hits = []
        seen_parent_ids: set[int] = set()
        with RegionMetadataDB(db_path) as db:
            for vector_id, score in raw_hits:
                row = db.get_vector(vector_id)
                if row is None:
                    logger.error("FAISS row %d has no SQLite metadata.", vector_id)
                    continue
                is_match = (
                    bool(row["is_whole_image"])
                    if search_scope == "whole"
                    else (
                        bool(row["is_whole_image"]) or row["crop_type"] == crop_type
                        if search_scope == "both"
                        else not bool(row["is_whole_image"])
                        and row["crop_type"] == crop_type
                    )
                )
                if not is_match:
                    continue
                parent_id = int(row["parent_image_id"])
                if parent_id in seen_parent_ids:
                    continue
                seen_parent_ids.add(parent_id)
                matching_hits.append((vector_id, score))

        if len(matching_hits) >= target_parents or requested == faiss_index.ntotal:
            break
        requested = min(faiss_index.ntotal, requested * 2)

    return matching_hits[:target_parents]


def _bbox_from_row(row: dict) -> list[float] | None:
    """Convert a SQLite row to a [x1, y1, x2, y2] list, or None."""
    if row.get("x1") is None:
        return None
    return [
        float(row["x1"]),
        float(row["y1"]),
        float(row["x2"]),
        float(row["y2"]),
    ]


# ---------------------------------------------------------------------------
# Query-image region detection (for the /search/image/detect endpoint)
# ---------------------------------------------------------------------------


def detect_query_regions(
    image,
    *,
    split_mode: str = "auto",
) -> dict:
    """Detect regions in an uploaded query image.

    Returns thumbnail metadata plus the actual detector mode and any
    automatic-mode fallback explanation.
    """
    import base64
    import io

    from visionsearch_web_ui.region.detector import detect_regions_with_status
    from visionsearch_web_ui.region.region_splitter import crop_region

    image = image.convert("RGB")
    outcome = detect_regions_with_status(image, split_mode=split_mode)

    region_list: list[dict] = []
    for idx, box in enumerate(outcome.regions):
        crop = crop_region(image, box)
        buf = io.BytesIO()
        crop.thumbnail((128, 128))
        crop.save(buf, format="JPEG", quality=75)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        region_list.append({
            "region_id": idx,
            "bbox": [box.x1, box.y1, box.x2, box.y2],
            "class_label": box.class_label,
            "confidence": round(box.confidence, 4),
            "crop_type": box.crop_type,
            "thumbnail_b64": b64,
        })

    return {
        "regions": region_list,
        "requested_mode": outcome.requested_mode,
        "used_mode": outcome.used_mode,
        "fallback": outcome.fallback_reason is not None
        and outcome.used_mode != "yolo",
        "message": outcome.fallback_reason,
    }
