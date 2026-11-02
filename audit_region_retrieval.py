"""Audit person/dog retrieval.

Run from the repository root with ``python audit_region_retrieval.py``.
For an uploaded query, pass its image and exact crop bbox, e.g.
``--person-image C:\\data\\query.jpg --person-bbox 12,18,545,360``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import sqlite3
from collections import Counter
from pathlib import Path

import faiss
import numpy as np
from PIL import Image

from models.clip_model import MODEL_NAME, load_clip_model
from visionsearch_web_ui.region import config
from visionsearch_web_ui.region.region_splitter import RegionBox, crop_region
from visionsearch_web_ui.region.search_service import search_regions


FIELDS = [
    "query_id",
    "query_label",
    "query_image",
    "query_region_id",
    "query_sqlite_vector_id",
    "query_faiss_id",
    "query_bbox",
    "query_crop_width",
    "query_crop_height",
    "query_embedding_dimension",
    "query_embedding_norm",
    "query_embedding_fingerprint",
    "query_embedding_index_cosine",
    "search_scope",
    "candidate_set",
    "candidate_rank",
    "faiss_id",
    "sqlite_vector_id",
    "parent_image_id",
    "image_filename",
    "parent_path",
    "matched_region_label",
    "crop_type",
    "raw_cosine_similarity",
    "result_score",
    "candidate_bbox",
]


def _parse_bbox(value: str | None) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    try:
        coordinates = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("bbox must be four comma-separated numbers") from error
    if len(coordinates) != 4:
        raise argparse.ArgumentTypeError("bbox must be four comma-separated numbers")
    return coordinates  # type: ignore[return-value]


def _indexed_query(database: sqlite3.Connection, label: str) -> dict:
    row = database.execute(
        """
        SELECT rv.vector_id, rv.faiss_id, rv.parent_image_id, rv.parent_path,
               rv.x1, rv.y1, rv.x2, rv.y2, rv.confidence
        FROM region_vectors rv
        WHERE rv.index_type = 'region'
          AND rv.crop_type = 'yolo'
          AND lower(rv.class_label) = ?
        ORDER BY rv.confidence DESC, rv.vector_id
        LIMIT 1
        """,
        (label,),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"No indexed YOLO region with class label {label!r}")
    return dict(row)


def _resolve_query(
    database: sqlite3.Connection,
    label: str,
    image_path: Path | None,
    bbox: tuple[float, float, float, float] | None,
    region_id: int | None,
) -> dict:
    if image_path is None:
        query = _indexed_query(database, label)
        query["bbox"] = tuple(query[key] for key in ("x1", "y1", "x2", "y2"))
        query["image_path"] = Path(query["parent_path"])
    else:
        if bbox is None:
            raise ValueError(f"--{label}-image requires --{label}-bbox")
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        query = {
            "vector_id": None,
            "faiss_id": None,
            "parent_image_id": None,
            "parent_path": str(image_path.resolve()),
            "bbox": bbox,
            "confidence": None,
            "image_path": image_path.resolve(),
        }

    if region_id is None:
        from visionsearch_web_ui.region.detector import detect_regions

        with Image.open(query["image_path"]) as source:
            detections = detect_regions(source.convert("RGB"), split_mode="yolo")
        matches = [
            index
            for index, detection in enumerate(detections)
            if detection.class_label.lower() == label
            and np.allclose(
                (detection.x1, detection.y1, detection.x2, detection.y2),
                query["bbox"],
                rtol=0.0,
                atol=1.0,
            )
        ]
        query["region_id"] = matches[0] if matches else None
    else:
        query["region_id"] = region_id
    return query


def _metadata_counts(database: sqlite3.Connection) -> dict[str, int]:
    rows = database.execute(
        """
        SELECT index_type, crop_type, COUNT(*)
        FROM region_vectors
        GROUP BY index_type, crop_type
        """
    ).fetchall()
    return {f"{index_type}/{crop_type}": count for index_type, crop_type, count in rows}


def _query_record(
    *,
    query_id: str,
    label: str,
    query: dict,
    crop: Image.Image,
    embedding: np.ndarray,
    cosine_to_index: float | None,
    candidate_set: str,
    rank: int,
    faiss_id: int | None,
    metadata: dict | None,
    similarity: float | None,
    search_scope: str,
    result_score: float | None = None,
) -> dict:
    bbox = query["bbox"]
    candidate_bbox = None
    if metadata is not None and metadata["crop_type"] != "whole":
        candidate_bbox = [
            metadata["x1"],
            metadata["y1"],
            metadata["x2"],
            metadata["y2"],
        ]
    return {
        "query_id": query_id,
        "query_label": label,
        "query_image": str(query["image_path"]),
        "query_region_id": query["region_id"],
        "query_sqlite_vector_id": query["vector_id"],
        "query_faiss_id": query["faiss_id"],
        "query_bbox": list(bbox),
        "query_crop_width": crop.width,
        "query_crop_height": crop.height,
        "query_embedding_dimension": int(embedding.size),
        "query_embedding_norm": float(np.linalg.norm(embedding)),
        "query_embedding_fingerprint": hashlib.sha256(
            np.ascontiguousarray(embedding, dtype=np.float32).tobytes()
        ).hexdigest()[:12],
        "query_embedding_index_cosine": cosine_to_index,
        "search_scope": search_scope,
        "candidate_set": candidate_set,
        "candidate_rank": rank,
        "faiss_id": faiss_id,
        "sqlite_vector_id": metadata["vector_id"] if metadata else None,
        "parent_image_id": metadata["parent_image_id"] if metadata else None,
        "image_filename": Path(metadata["parent_path"]).name if metadata else None,
        "parent_path": metadata["parent_path"] if metadata else None,
        "matched_region_label": metadata["class_label"] if metadata else None,
        "crop_type": metadata["crop_type"] if metadata else None,
        "raw_cosine_similarity": similarity,
        "result_score": result_score,
        "candidate_bbox": candidate_bbox,
    }


def audit(args: argparse.Namespace) -> list[dict]:
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    region_index = faiss.read_index(str(args.region_index))
    if region_index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise RuntimeError(f"Expected inner-product FAISS, got metric {region_index.metric_type}")
    if region_index.d != 512:
        raise RuntimeError(f"Expected 512-dimensional CLIP vectors, got {region_index.d}")

    metadata_connection = sqlite3.connect(args.database)
    metadata_connection.row_factory = sqlite3.Row
    try:
        counts = _metadata_counts(metadata_connection)
        region_count = sum(
            count
            for key, count in counts.items()
            if key.startswith("region/")
        )
        if region_count != region_index.ntotal:
            raise RuntimeError(
                f"Region FAISS/SQLite count mismatch: {region_index.ntotal} vs {region_count}"
            )
        print(f"Region index: {args.region_index} ({region_index.ntotal} vectors)")
        print(f"Metadata DB: {args.database}")
        print(f"Metadata counts: {counts}")
        indexed_vectors = region_index.reconstruct_n(0, region_index.ntotal)
        indexed_norms = np.linalg.norm(indexed_vectors, axis=1)
        if not np.isfinite(indexed_vectors).all() or not np.allclose(
            indexed_norms, 1.0, rtol=0.0, atol=1e-4
        ):
            raise RuntimeError("Region index contains invalid or non-normalized CLIP vectors")
        print(
            f"FAISS metric: inner product; region vector norm range: "
            f"{indexed_norms.min():.7f} .. {indexed_norms.max():.7f}"
        )
        model = load_clip_model()
        print(f"Query/index model wrapper: {MODEL_NAME} ({type(model).__module__}.{type(model).__name__})")
        print("Index preprocessing: ClipModel.encode_images -> CLIPProcessor(images=..., return_tensors='pt')")
        print("Query preprocessing: ClipModel.encode_image -> CLIPProcessor(images=..., return_tensors='pt')")

        queries = [
            _resolve_query(
                metadata_connection,
                "person",
                args.person_image,
                args.person_bbox,
                args.person_region_id,
            ),
            _resolve_query(
                metadata_connection,
                "dog",
                args.dog_image,
                args.dog_bbox,
                args.dog_region_id,
            ),
        ]
        rows: list[dict] = []
        whole_index: faiss.Index | None = None
        if args.whole_index.is_file():
            whole_index = faiss.read_index(str(args.whole_index))

        for label, query in zip(("person", "dog"), queries):
            with Image.open(query["image_path"]) as source:
                image = source.convert("RGB")
            x1, y1, x2, y2 = query["bbox"]
            query_box = RegionBox(
                x1, y1, x2, y2, label, query["confidence"] or 0.0, "yolo"
            )
            crop = crop_region(image, query_box)
            crop_path = output.parent / f"{output.stem}.{label}.query.jpg"
            crop.save(crop_path, quality=95)

            query_embedding = np.asarray(model.encode_image(crop), dtype=np.float32).reshape(-1)
            batched_embedding = np.asarray(model.encode_images([crop]), dtype=np.float32).reshape(-1)
            if query_embedding.shape != (region_index.d,):
                raise RuntimeError(f"Unexpected {label} query embedding shape {query_embedding.shape}")
            if not np.allclose(query_embedding, batched_embedding, rtol=0.0, atol=1e-6):
                raise RuntimeError(f"Single and batched CLIP preprocessing differ for {label}")
            query_norm = float(np.linalg.norm(query_embedding))
            if not np.isclose(query_norm, 1.0, atol=1e-5):
                raise RuntimeError(f"{label} query embedding is not normalized: {query_norm}")

            indexed_cosine = None
            if query["faiss_id"] is not None:
                with metadata_connection:
                    mapped = metadata_connection.execute(
                        """
                        SELECT vector_id, parent_image_id, parent_path, class_label,
                               crop_type, x1, y1, x2, y2
                        FROM region_vectors
                        WHERE index_type = 'region' AND faiss_id = ?
                        """,
                        (query["faiss_id"],),
                    ).fetchone()
                if mapped is None:
                    raise RuntimeError(f"Selected FAISS ID {query['faiss_id']} has no SQLite row")
                if int(mapped["vector_id"]) != int(query["vector_id"]):
                    raise RuntimeError("Selected FAISS-to-SQLite vector ID mapping is inconsistent")
                if int(mapped["parent_image_id"]) != int(query["parent_image_id"]):
                    raise RuntimeError("Selected FAISS-to-SQLite parent image mapping is inconsistent")
                if Path(mapped["parent_path"]).resolve() != query["image_path"].resolve():
                    raise RuntimeError("Selected FAISS-to-SQLite path mapping is inconsistent")
                indexed_vector = region_index.reconstruct(int(query["faiss_id"]))
                indexed_cosine = float(np.dot(query_embedding, indexed_vector))
                print(
                    f"{label} selected region: region_id={query.get('region_id')} "
                    f"vector_id={query['vector_id']} "
                    f"faiss_id={query['faiss_id']} parent_image_id={query['parent_image_id']} "
                    f"label={mapped['class_label']} type={mapped['crop_type']} "
                    f"bbox={[mapped['x1'], mapped['y1'], mapped['x2'], mapped['y2']]} "
                    f"confidence={query['confidence']}"
                )
                if not np.allclose(
                    [mapped[key] for key in ("x1", "y1", "x2", "y2")],
                    query["bbox"],
                    rtol=0.0,
                    atol=1e-5,
                ):
                    raise RuntimeError("Selected bbox differs from its SQLite record")
                if indexed_cosine < 0.999:
                    raise RuntimeError(
                        f"Re-embedded selected crop differs from indexed vector: cosine={indexed_cosine}"
                    )
            else:
                print(
                    f"{label} supplied crop: path={query['image_path']} "
                    f"bbox={list(query['bbox'])}"
                )
            print(
                f"{label} crop saved={crop_path} dimensions={crop.width}x{crop.height} "
                f"embedding_dimension={query_embedding.size} norm={query_norm:.8f} "
                f"fingerprint={hashlib.sha256(query_embedding.tobytes()).hexdigest()[:12]} "
                f"cosine_to_own_index_vector={indexed_cosine}"
            )

            all_scores, all_ids = region_index.search(
                query_embedding.reshape(1, -1), region_index.ntotal
            )
            candidates = []
            with metadata_connection:
                for score, faiss_id in zip(all_scores[0], all_ids[0]):
                    if faiss_id < 0:
                        continue
                    metadata = metadata_connection.execute(
                        """
                        SELECT rv.vector_id, rv.faiss_id, rv.parent_image_id,
                               rv.parent_path, rv.class_label, rv.crop_type,
                               rv.x1, rv.y1, rv.x2, rv.y2, rv.index_type,
                               rv.is_whole_image
                        FROM region_vectors rv
                        WHERE rv.index_type = 'region' AND rv.faiss_id = ?
                        """,
                        (int(faiss_id),),
                    ).fetchone()
                    if metadata is None:
                        raise RuntimeError(f"FAISS region ID {faiss_id} has no SQLite metadata")
                    candidates.append((int(faiss_id), float(score), dict(metadata)))

            candidate_types = Counter(metadata["crop_type"] for _, _, metadata in candidates)
            if sum(candidate_types.values()) != region_index.ntotal:
                raise RuntimeError("Not every region FAISS vector mapped to SQLite metadata")
            print(f"{label} region-vector types: {dict(candidate_types)}")

            for candidate_set, filtered in (
                ("raw_all_regions", candidates),
                ("raw_yolo_only", [hit for hit in candidates if hit[2]["crop_type"] == "yolo"]),
                ("raw_grid_only", [hit for hit in candidates if hit[2]["crop_type"] == "grid"]),
            ):
                for rank, (faiss_id, score, metadata) in enumerate(filtered[:20], start=1):
                    rows.append(
                        _query_record(
                            query_id=f"region-audit-{label}",
                            label=label,
                            query=query,
                            crop=crop,
                            embedding=query_embedding,
                            cosine_to_index=indexed_cosine,
                            candidate_set=candidate_set,
                            rank=rank,
                            faiss_id=faiss_id,
                            metadata=metadata,
                            similarity=score,
                            search_scope="regions",
                        )
                    )

            yolo_hits = [hit for hit in candidates if hit[2]["crop_type"] == "yolo"]
            best_by_parent: dict[int, tuple[int, float, dict]] = {}
            for hit in yolo_hits:
                parent_id = int(hit[2]["parent_image_id"])
                if parent_id not in best_by_parent or hit[1] > best_by_parent[parent_id][1]:
                    best_by_parent[parent_id] = hit
            grouped = sorted(best_by_parent.values(), key=lambda hit: hit[1], reverse=True)
            print(
                f"{label} raw top 20 parent IDs: "
                f"{[metadata['parent_image_id'] for _, _, metadata in candidates[:20]]}"
            )
            print(
                f"{label} grouped YOLO pool: {len(grouped)} unique parents "
                "(best region retained per parent; CSV stores the top 20)"
            )
            for rank, (faiss_id, score, metadata) in enumerate(grouped[:20], start=1):
                rows.append(
                    _query_record(
                        query_id=f"region-audit-{label}",
                        label=label,
                        query=query,
                        crop=crop,
                        embedding=query_embedding,
                        cosine_to_index=indexed_cosine,
                        candidate_set="grouped_yolo_best_per_parent",
                        rank=rank,
                        faiss_id=faiss_id,
                        metadata=metadata,
                        similarity=score,
                        search_scope="regions",
                    )
                )

            service_results_by_scope = {
                scope: search_regions(
                    query_embedding,
                    search_scope=scope,
                    top_k=20,
                    whole_weight=config.WHOLE_IMAGE_WEIGHT,
                    region_crop_type="yolo",
                    query_region_id=query["region_id"],
                    query_label=label,
                    query_dimensions=crop.size,
                    request_id=f"region-audit-{label}-{scope}",
                )
                for scope in ("regions", "both")
            }
            service_results = service_results_by_scope["regions"]
            if any(result["match_type"] != "yolo" for result in service_results):
                raise RuntimeError("Region-only YOLO search returned a non-YOLO result")
            if len({result["parent_path"] for result in service_results}) != len(service_results):
                raise RuntimeError("Search service returned duplicate parent images")
            print(f"{label} region-only service results: {len(service_results)} unique YOLO parents")
            for scope, scope_results in service_results_by_scope.items():
                if len({result["parent_path"] for result in scope_results}) != len(scope_results):
                    raise RuntimeError(f"{scope} search returned duplicate parent images")
                for rank, result in enumerate(scope_results, start=1):
                    print(
                        f"  {scope} rank={rank:02d} file={Path(result['parent_path']).name} "
                        f"label={result['class_label']} type={result['match_type']} "
                        f"score={result['score']:.6f} bbox={result['matched_bbox']}"
                    )
                    if scope == "both":
                        parent = metadata_connection.execute(
                            "SELECT id FROM parent_images WHERE path = ?",
                            (str(Path(result["parent_path"]).resolve()),),
                        ).fetchone()
                        if parent is None:
                            raise RuntimeError(
                                f"Search result path has no parent metadata: {result['parent_path']}"
                            )
                        matched_metadata = None
                        if result["match_type"] != "whole":
                            matched_metadata = metadata_connection.execute(
                                """
                                SELECT vector_id, faiss_id, parent_image_id, parent_path,
                                       class_label, crop_type, x1, y1, x2, y2
                                FROM region_vectors
                                WHERE index_type = 'region'
                                  AND parent_image_id = ?
                                  AND class_label = ?
                                  AND crop_type = ?
                                  AND abs(x1 - ?) < 0.00001
                                  AND abs(y1 - ?) < 0.00001
                                  AND abs(x2 - ?) < 0.00001
                                  AND abs(y2 - ?) < 0.00001
                                ORDER BY faiss_id
                                LIMIT 1
                                """,
                                (
                                    int(parent["id"]),
                                    result["class_label"],
                                    result["match_type"],
                                    *(result["matched_bbox"] or []),
                                ),
                            ).fetchone()
                            if matched_metadata is None:
                                raise RuntimeError(
                                    "Could not resolve grouped result's matched region metadata"
                                )
                            matched_metadata = dict(matched_metadata)
                        else:
                            matched_metadata = metadata_connection.execute(
                                """
                                SELECT vector_id, faiss_id, parent_image_id, parent_path,
                                       class_label, crop_type, x1, y1, x2, y2
                                FROM region_vectors
                                WHERE index_type = 'whole' AND parent_image_id = ?
                                LIMIT 1
                                """,
                                (int(parent["id"]),),
                            ).fetchone()
                            if matched_metadata is None:
                                raise RuntimeError(
                                    "Could not resolve grouped result's whole-image metadata"
                                )
                            matched_metadata = dict(matched_metadata)
                        rows.append(
                            _query_record(
                                query_id=f"region-audit-{label}",
                                label=label,
                                query=query,
                                crop=crop,
                                embedding=query_embedding,
                                cosine_to_index=indexed_cosine,
                                candidate_set="service_both_grouped",
                                rank=rank,
                                faiss_id=(
                                    int(matched_metadata["faiss_id"])
                                    if matched_metadata
                                    else None
                                ),
                                metadata=matched_metadata,
                                similarity=None,
                                search_scope=scope,
                                result_score=result["score"],
                            )
                        )

            if whole_index is not None:
                whole_vectors = whole_index.reconstruct_n(0, whole_index.ntotal)
                whole_norms = np.linalg.norm(whole_vectors, axis=1)
                if (
                    whole_index.metric_type != faiss.METRIC_INNER_PRODUCT
                    or whole_index.d != 512
                    or not np.isfinite(whole_vectors).all()
                    or not np.allclose(whole_norms, 1.0, rtol=0.0, atol=1e-4)
                ):
                    raise RuntimeError(
                        "Whole-image index metric, dimension, or vector norms are invalid"
                    )
                print(
                    f"{label} whole index exists ({whole_index.ntotal} vectors, "
                    f"norm range {whole_norms.min():.7f} .. {whole_norms.max():.7f}); "
                    "the regions-only call above uses the region index and returns only YOLO rows."
                )

            top20 = [hit for hit in candidates if hit[2]["crop_type"] == "yolo"][:20]
            print(f"{label} top YOLO FAISS candidates:")
            for rank, (faiss_id, score, metadata) in enumerate(top20, start=1):
                print(
                    f"  {rank:02d}. file={Path(metadata['parent_path']).name} "
                    f"parent_image_id={metadata['parent_image_id']} "
                    f"vector_id={metadata['vector_id']} faiss_id={faiss_id} "
                    f"label={metadata['class_label']} crop_type={metadata['crop_type']} "
                    f"cosine={score:.6f} "
                    f"bbox={[metadata['x1'], metadata['y1'], metadata['x2'], metadata['y2']]}"
                )
        return rows
    finally:
        metadata_connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region-index", type=Path, default=config.REGION_FAISS_PATH)
    parser.add_argument("--whole-index", type=Path, default=config.REGION_WHOLE_FAISS_PATH)
    parser.add_argument("--database", type=Path, default=config.REGION_DB_PATH)
    parser.add_argument("--output", type=Path, default=Path("region_retrieval_audit.csv"))
    parser.add_argument("--person-image", type=Path)
    parser.add_argument("--person-bbox", type=_parse_bbox)
    parser.add_argument("--person-region-id", type=int)
    parser.add_argument("--dog-image", type=Path)
    parser.add_argument("--dog-bbox", type=_parse_bbox)
    parser.add_argument("--dog-region-id", type=int)
    args = parser.parse_args()

    rows = audit(args)
    with Path(args.output).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV written: {Path(args.output).resolve()} ({len(rows)} candidate rows)")


if __name__ == "__main__":
    main()
