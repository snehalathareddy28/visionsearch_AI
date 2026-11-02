"""Run a 5-person/5-dog/5-object region retrieval test from the existing index.

The labeled groups come from the selected Caltech-101 test folders:
Faces_easy, dog, and umbrella. Search results are not category-filtered;
the labels are used only to report per-group retrieval.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path

import faiss
import numpy as np
from PIL import Image

from models.clip_model import MODEL_NAME, load_clip_model
from visionsearch_web_ui.region import config
from visionsearch_web_ui.region.metadata_db import RegionMetadataDB
from visionsearch_web_ui.region.region_splitter import RegionBox, crop_region
from visionsearch_web_ui.region.search_service import search_regions

GROUPS = (
    ("person", "Faces_easy", "person"),
    ("dog", "dog", "dog"),
    ("other", "umbrella", "umbrella"),
)
OUTPUT_FIELDS = (
    "query_id",
    "query_class",
    "query_image",
    "query_region_id",
    "query_bbox",
    "query_confidence",
    "query_embedding_dimension",
    "query_embedding_norm",
    "query_embedding_fingerprint",
    "search_scope",
    "region_crop_type",
    "rank",
    "parent_image_id",
    "parent_category",
    "image_filename",
    "matched_region_label",
    "matched_confidence",
    "crop_type",
    "raw_cosine_similarity",
    "matched_bbox",
    "faiss_id",
    "sqlite_vector_id",
)


def _choose_group_images(
    connection: sqlite3.Connection,
    folder: str,
    expected_label: str,
) -> list[dict]:
    rows = connection.execute(
        """
        SELECT p.id AS parent_image_id, p.path, p.file_hash,
               rv.vector_id, rv.faiss_id, rv.class_label, rv.confidence,
               rv.x1, rv.y1, rv.x2, rv.y2
        FROM parent_images p
        JOIN region_vectors rv ON rv.parent_image_id = p.id
        WHERE rv.index_type = 'region'
          AND rv.crop_type = 'yolo'
          AND lower(rv.class_label) = ?
        ORDER BY p.path, rv.confidence DESC
        """,
        (expected_label,),
    ).fetchall()
    by_parent: dict[str, dict] = {}
    for row in rows:
        image_path = Path(row["path"])
        if image_path.parent.name.lower() != folder.lower():
            continue
        key = str(image_path.resolve())
        by_parent.setdefault(key, dict(row))
    selected = list(by_parent.values())[:5]
    if len(selected) != 5:
        raise RuntimeError(
            f"Need 5 YOLO {expected_label!r} image regions in {folder!r}; "
            f"found {len(selected)}."
        )
    for row in selected:
        row["image_path"] = Path(row["path"]).resolve()
        row["bbox"] = tuple(row[key] for key in ("x1", "y1", "x2", "y2"))
        row["test_group"] = folder
        if not row["image_path"].is_file():
            raise FileNotFoundError(row["image_path"])
    return selected


def _load_parent_regions(
    connection: sqlite3.Connection,
    parent_path: str,
) -> list[sqlite3.Row]:
    return connection.execute(
        """
        SELECT vector_id, faiss_id, parent_image_id, parent_path,
               file_hash, class_label, confidence, x1, y1, x2, y2
        FROM region_vectors
        WHERE index_type = 'region'
          AND crop_type = 'yolo'
          AND parent_path = ?
        ORDER BY faiss_id
        """,
        (parent_path,),
    ).fetchall()


def run_test(output_path: Path) -> list[dict]:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)

    source_index = faiss.read_index(str(config.REGION_FAISS_PATH))
    if (
        source_index.metric_type != faiss.METRIC_INNER_PRODUCT
        or source_index.d != 512
    ):
        raise RuntimeError("The configured region index must be 512-D inner-product FAISS")

    source_connection = sqlite3.connect(config.REGION_DB_PATH)
    source_connection.row_factory = sqlite3.Row
    rows: list[dict] = []
    try:
        selected: dict[str, list[dict]] = {}
        for group, folder, label in GROUPS:
            selected[group] = _choose_group_images(
                source_connection,
                folder,
                label,
            )
        selected_paths = {
            str(image["image_path"])
            for images in selected.values()
            for image in images
        }
        if len(selected_paths) != 15:
            raise RuntimeError("The controlled groups must contain 15 distinct images")
        print("Controlled image groups (category labels used only for evaluation):")
        for group, images in selected.items():
            print(f"  {group}:")
            for image in images:
                print(
                    f"    {image['image_path'].parent.name}\\{image['image_path'].name} "
                    f"region_id={image['vector_id']} faiss_id={image['faiss_id']} "
                    f"label={image['class_label']} conf={image['confidence']:.4f}"
                )

        model = load_clip_model()
        print(f"CLIP model: {MODEL_NAME}; query and index wrappers: {type(model).__name__}")
        print(
            "Preprocessing: indexed regions use encode_images/CLIPProcessor; "
            "queries use encode_image/CLIPProcessor."
        )
        subset_index = faiss.IndexFlatIP(source_index.d)

        with tempfile.TemporaryDirectory(prefix="visionsearch-controlled-region-") as temp:
            temp_root = Path(temp)
            subset_index_path = temp_root / "controlled_region.index"
            subset_db_path = temp_root / "controlled_region.sqlite"
            candidate_categories: dict[int, str] = {}
            source_to_subset: dict[int, int] = {}

            with RegionMetadataDB(subset_db_path) as subset_db:
                next_vector_id = 0
                for category, images in selected.items():
                    for image in images:
                        parent_id = subset_db.add_parent_image(
                            image["image_path"],
                            image["file_hash"],
                        )
                        source_regions = _load_parent_regions(
                            source_connection,
                            str(image["image_path"]),
                        )
                        if not source_regions:
                            raise RuntimeError(
                                f"Selected controlled image has no YOLO regions: "
                                f"{image['image_path']}"
                            )
                        for source_row in source_regions:
                            source_faiss_id = int(source_row["faiss_id"])
                            vector = source_index.reconstruct(source_faiss_id)
                            if not np.isfinite(vector).all() or not np.isclose(
                                np.linalg.norm(vector), 1.0, atol=1e-4
                            ):
                                raise RuntimeError(
                                    f"Invalid source vector at FAISS ID {source_faiss_id}"
                                )
                            subset_id = int(subset_index.ntotal)
                            subset_index.add(vector.reshape(1, -1))
                            source_to_subset[source_faiss_id] = subset_id
                            candidate_categories[subset_id] = category
                            box = RegionBox(
                                source_row["x1"],
                                source_row["y1"],
                                source_row["x2"],
                                source_row["y2"],
                                source_row["class_label"],
                                source_row["confidence"],
                                "yolo",
                            )
                            subset_db.add_region_vector(
                                next_vector_id,
                                parent_id,
                                image["image_path"],
                                box,
                                index_type="region",
                                faiss_id=subset_id,
                                file_hash=image["file_hash"],
                            )
                            next_vector_id += 1

                if subset_index.ntotal == 0:
                    raise RuntimeError("Controlled dataset produced no YOLO vectors")
                faiss.write_index(subset_index, str(subset_index_path))
                subset_counts = subset_db.get_vector_counts()
                if subset_counts["region"] != subset_index.ntotal:
                    raise RuntimeError("Controlled FAISS/SQLite vector counts do not match")

            for query_class in ("person", "dog"):
                query_rows = []
                query = selected[query_class][0]
                with Image.open(query["image_path"]) as source_image:
                    image = source_image.convert("RGB")
                bbox = query["bbox"]
                crop = crop_region(
                    image,
                    RegionBox(*bbox, query["class_label"], query["confidence"], "yolo"),
                )
                query_path = output_path.parent / f"{output_path.stem}.{query_class}.jpg"
                crop.save(query_path, quality=95)

                query_embedding = np.asarray(
                    model.encode_image(crop), dtype=np.float32
                ).reshape(-1)
                query_norm = float(np.linalg.norm(query_embedding))
                if query_embedding.shape != (source_index.d,) or not np.isclose(
                    query_norm, 1.0, atol=1e-5
                ):
                    raise RuntimeError(
                        f"Invalid {query_class} query vector: "
                        f"shape={query_embedding.shape}, norm={query_norm}"
                    )
                query_fingerprint = hashlib.sha256(
                    np.ascontiguousarray(query_embedding).tobytes()
                ).hexdigest()[:12]

                own_source_vector = source_index.reconstruct(int(query["faiss_id"]))
                own_cosine = float(np.dot(query_embedding, own_source_vector))
                if own_cosine < 0.999:
                    raise RuntimeError(
                        f"{query_class} crop does not reproduce indexed vector "
                        f"{query['faiss_id']}: cosine={own_cosine}"
                    )
                print(
                    f"{query_class} query crop={query_path} region_id=0 "
                    f"source_vector_id={query['vector_id']} source_faiss_id={query['faiss_id']} "
                    f"bbox={list(bbox)} size={crop.width}x{crop.height} "
                    f"dimension={query_embedding.size} norm={query_norm:.8f} "
                    f"fingerprint={query_fingerprint} self_cosine={own_cosine:.8f}"
                )

                # Direct, unmodified cosine comparisons against a known person,
                # dog, and object region from the controlled set.
                for reference_class in ("person", "dog", "other"):
                    reference = selected[reference_class][0]
                    source_vector = source_index.reconstruct(
                        int(reference["faiss_id"])
                    )
                    print(
                        f"  cosine_to_{reference_class}="
                        f"{float(np.dot(query_embedding, source_vector)):.6f} "
                        f"({reference['image_path'].parent.name}\\"
                        f"{reference['image_path'].name}, "
                        f"label={reference['class_label']})"
                    )

                result_list = search_regions(
                    query_embedding,
                    search_scope="regions",
                    top_k=10,
                    region_crop_type="yolo",
                    request_id=f"controlled-{query_class}-regions-only",
                    query_region_id=0,
                    query_label=query_class,
                    query_dimensions=crop.size,
                    faiss_path=subset_index_path,
                    db_path=subset_db_path,
                    whole_faiss_path=None,
                )
                if len(result_list) != 10:
                    raise RuntimeError(
                        f"Expected 10 controlled results for {query_class}; "
                        f"received {len(result_list)}"
                    )
                scores = [float(result["score"]) for result in result_list]
                if scores != sorted(scores, reverse=True):
                    raise RuntimeError("Region-only results are not score-sorted")
                if any(result["match_type"] != "yolo" for result in result_list):
                    raise RuntimeError("Controlled YOLO search returned a non-YOLO result")
                if len({result["parent_path"] for result in result_list}) != len(result_list):
                    raise RuntimeError("Controlled search returned duplicate parent images")

                with RegionMetadataDB(subset_db_path) as subset_db:
                    for rank, result in enumerate(result_list, start=1):
                        parent_path = str(Path(result["parent_path"]).resolve())
                        parent_row = subset_db.conn.execute(
                            "SELECT id FROM parent_images WHERE path = ?",
                            (parent_path,),
                        ).fetchone()
                        if parent_row is None:
                            raise RuntimeError("A result parent is missing from controlled SQLite")
                        parent_id = int(parent_row["id"])
                        match = result["matched_bbox"]
                        metadata = subset_db.conn.execute(
                            """
                            SELECT vector_id, faiss_id
                            FROM region_vectors
                            WHERE index_type = 'region'
                              AND parent_image_id = ?
                              AND class_label = ?
                              AND crop_type = ?
                              AND abs(x1 - ?) < 0.00001
                              AND abs(y1 - ?) < 0.00001
                              AND abs(x2 - ?) < 0.00001
                              AND abs(y2 - ?) < 0.00001
                            LIMIT 1
                            """,
                            (parent_id, result["class_label"], result["match_type"], *match),
                        ).fetchone()
                        if metadata is None:
                            raise RuntimeError("Search result cannot be mapped to its FAISS/SQLite vector")
                        candidate_set_id = int(metadata["faiss_id"])
                        source_faiss_id = next(
                            original
                            for original, subset_id in source_to_subset.items()
                            if subset_id == candidate_set_id
                        )
                        source_metadata = source_connection.execute(
                            """
                            SELECT vector_id, parent_image_id, parent_path, class_label,
                                   crop_type, x1, y1, x2, y2
                            FROM region_vectors
                            WHERE index_type = 'region' AND faiss_id = ?
                            """,
                            (source_faiss_id,),
                        ).fetchone()
                        if source_metadata is None:
                            raise RuntimeError("Controlled candidate has no source metadata")
                        if (
                            Path(source_metadata["parent_path"]).resolve()
                            != Path(result["parent_path"]).resolve()
                            or source_metadata["class_label"] != result["class_label"]
                            or source_metadata["crop_type"] != result["match_type"]
                            or not np.allclose(
                                [
                                    source_metadata["x1"],
                                    source_metadata["y1"],
                                    source_metadata["x2"],
                                    source_metadata["y2"],
                                ],
                                match,
                                atol=1e-5,
                            )
                        ):
                            raise RuntimeError(
                                "FAISS result and source SQLite metadata do not match"
                            )
                        category = candidate_categories[candidate_set_id]
                        exact_score = float(
                            np.dot(
                                query_embedding,
                                source_index.reconstruct(source_faiss_id),
                            )
                        )
                        query_rows.append(
                            {
                                "query_id": f"controlled-{query_class}-regions-only",
                                "query_class": query_class,
                                "query_image": str(query["image_path"]),
                                "query_region_id": 0,
                                "query_bbox": ",".join(str(value) for value in bbox),
                                "query_confidence": query["confidence"],
                                "query_embedding_dimension": query_embedding.size,
                                "query_embedding_norm": query_norm,
                                "query_embedding_fingerprint": query_fingerprint,
                                "search_scope": "regions",
                                "region_crop_type": "yolo",
                                "rank": rank,
                                "parent_image_id": source_metadata["parent_image_id"],
                                "parent_category": category,
                                "image_filename": Path(source_metadata["parent_path"]).name,
                                "matched_region_label": result["class_label"],
                                "matched_confidence": result["confidence"],
                                "crop_type": result["match_type"],
                                "raw_cosine_similarity": exact_score,
                                "matched_bbox": ",".join(str(value) for value in match),
                                "faiss_id": source_faiss_id,
                                "sqlite_vector_id": source_metadata["vector_id"],
                            }
                        )
                        print(
                            f"  {rank:02d}. {category}\\{Path(result['parent_path']).name} "
                            f"parent_id={source_metadata['parent_image_id']} "
                            f"label={result['class_label']} type={result['match_type']} "
                            f"raw_cosine={exact_score:.8f} "
                            f"service_score={result['score']:.6f} bbox={match}"
                        )

                query_rows.sort(key=lambda row: row["rank"])
                hits_by_category = Counter(row["parent_category"] for row in query_rows)
                print(f"{query_class} top-10 group counts: {dict(hits_by_category)}")
                rows.extend(query_rows)

                for scope in ("regions", "both"):
                    production_results = search_regions(
                        query_embedding,
                        search_scope=scope,
                        top_k=10,
                        region_crop_type="yolo",
                        request_id=f"production-{query_class}-{scope}",
                        query_region_id=0,
                        query_label=query_class,
                        query_dimensions=crop.size,
                        faiss_path=config.REGION_FAISS_PATH,
                        db_path=config.REGION_DB_PATH,
                        whole_faiss_path=None,
                    )
                    if scope == "regions" and any(
                        result["match_type"] != "yolo"
                        for result in production_results
                    ):
                        raise RuntimeError(
                            "Production region-only search returned a non-YOLO result"
                        )
                    print(
                        f"{query_class} production {scope} top-10 "
                        f"(all region-index candidates filtered to YOLO):"
                    )
                    for rank, result in enumerate(production_results, start=1):
                        print(
                            f"  {rank:02d}. {Path(result['parent_path']).parent.name}\\"
                            f"{Path(result['parent_path']).name} "
                            f"label={result['class_label']} "
                            f"type={result['match_type']} "
                            f"score={result['score']:.6f} "
                            f"bbox={result['matched_bbox']}"
                        )
        return rows
    finally:
        source_connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("controlled_region_results.csv"),
    )
    args = parser.parse_args()
    results = run_test(args.output)
    with args.output.resolve().open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(results)
    print(f"CSV written: {args.output.resolve()} ({len(results)} rows)")


if __name__ == "__main__":
    main()
