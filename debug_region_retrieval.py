"""Audit person and dog crop embeddings against the configured region index.

Run from the repository root:
    python debug_region_retrieval.py

To supply the screenshot's exact query image and bounding box:
    python debug_region_retrieval.py --person-image C:\\data\\query.jpg \\
        --person-bbox 12,18,545,360
"""

from __future__ import annotations

import argparse
import csv
import logging
import sqlite3
from pathlib import Path

import faiss
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from models.clip_model import MODEL_NAME, load_clip_model
from visionsearch_web_ui.region import config
from visionsearch_web_ui.region.region_splitter import (
    RegionBox,
    clamp_bbox,
    crop_region,
)
from visionsearch_web_ui.region.search_service import (
    invalidate_region_index_cache,
    search_regions,
)

CSV_FIELDS = (
    "query_label",
    "query_image",
    "query_crop",
    "query_bbox",
    "query_crop_width",
    "query_crop_height",
    "model_name",
    "embedding_dimension",
    "embedding_norm",
    "result_source",
    "rank",
    "filename",
    "parent_path",
    "matched_label",
    "crop_type",
    "bbox",
    "raw_cosine_similarity",
    "parent_image_id",
    "sqlite_vector_id",
    "faiss_id",
)


def _parse_bbox(value: str | None) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    try:
        coordinates = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "bbox must contain four comma-separated numbers"
        ) from error
    if len(coordinates) != 4 or not np.isfinite(coordinates).all():
        raise argparse.ArgumentTypeError(
            "bbox must contain four finite comma-separated numbers"
        )
    return coordinates  # type: ignore[return-value]


def _query_region(
    connection: sqlite3.Connection,
    label: str,
    image_path: Path | None,
    bbox: tuple[float, float, float, float] | None,
) -> dict:
    if image_path is not None:
        if bbox is None:
            raise ValueError(f"--{label}-image requires --{label}-bbox")
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        return {
            "label": label,
            "image_path": image_path.resolve(),
            "bbox": bbox,
            "vector_id": None,
            "faiss_id": None,
        }
    if bbox is not None:
        raise ValueError(f"--{label}-bbox requires --{label}-image")

    row = connection.execute(
        """
        SELECT rv.vector_id, rv.faiss_id, rv.parent_image_id, rv.parent_path,
               rv.class_label, rv.x1, rv.y1, rv.x2, rv.y2, rv.confidence
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
        raise RuntimeError(f"No indexed YOLO crop labeled {label!r}")
    return {
        "label": label,
        "image_path": Path(row["parent_path"]).resolve(),
        "bbox": tuple(row[key] for key in ("x1", "y1", "x2", "y2")),
        "vector_id": int(row["vector_id"]),
        "faiss_id": int(row["faiss_id"]),
        "parent_image_id": int(row["parent_image_id"]),
        "confidence": float(row["confidence"]),
    }


def _known_reference(
    connection: sqlite3.Connection,
    label: str,
) -> dict:
    row = connection.execute(
        """
        SELECT rv.vector_id, rv.faiss_id, rv.parent_image_id, rv.parent_path,
               rv.class_label, rv.crop_type, rv.x1, rv.y1, rv.x2, rv.y2
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
        raise RuntimeError(f"No indexed YOLO reference crop labeled {label!r}")
    return dict(row)


def _crop_from_metadata(metadata: dict) -> Image.Image:
    path = Path(metadata["parent_path"])
    with Image.open(path) as source:
        image = source.convert("RGB")
    bbox = clamp_bbox(
        metadata["x1"],
        metadata["y1"],
        metadata["x2"],
        metadata["y2"],
        image.width,
        image.height,
    )
    box = RegionBox(*bbox, metadata["class_label"], 0.0, metadata["crop_type"])
    return crop_region(image, box)


def _assert_faiss_sqlite_mapping(
    connection: sqlite3.Connection,
    index: faiss.Index,
) -> None:
    row = connection.execute(
        """
        SELECT COUNT(*), COUNT(DISTINCT faiss_id),
               MIN(faiss_id), MAX(faiss_id)
        FROM region_vectors
        WHERE index_type = 'region'
        """
    ).fetchone()
    expected = (index.ntotal, index.ntotal, 0, index.ntotal - 1)
    if tuple(row) != expected:
        raise RuntimeError(
            f"Region FAISS/SQLite ID mismatch: metadata={tuple(row)}, "
            f"FAISS count={index.ntotal}"
        )
    missing_parent = connection.execute(
        """
        SELECT COUNT(*)
        FROM region_vectors rv
        LEFT JOIN parent_images p ON p.id = rv.parent_image_id
        WHERE rv.index_type = 'region' AND p.id IS NULL
        """
    ).fetchone()[0]
    if missing_parent:
        raise RuntimeError(f"{missing_parent} region metadata rows have no parent image")


def _embed_and_check(
    model,
    image: Image.Image,
    index: faiss.Index,
) -> np.ndarray:
    embedding = np.asarray(model.encode_image(image), dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(embedding))
    if (
        embedding.shape != (index.d,)
        or not np.isfinite(embedding).all()
        or not np.isclose(norm, 1.0, rtol=0.0, atol=1e-5)
    ):
        raise RuntimeError(
            f"Invalid embedding: shape={embedding.shape}, L2 norm={norm}"
        )
    return embedding


def _report_query(
    connection: sqlite3.Connection,
    index: faiss.Index,
    model,
    query: dict,
    crop_output: Path,
) -> tuple[list[dict], list[dict]]:
    with Image.open(query["image_path"]) as source:
        image = source.convert("RGB")
    bbox = clamp_bbox(*query["bbox"], image.width, image.height)
    query["bbox"] = bbox
    query_box = RegionBox(*bbox, query["label"], 1.0, "yolo")
    crop = crop_region(image, query_box)
    crop_output.parent.mkdir(parents=True, exist_ok=True)
    crop.save(crop_output, quality=95)
    query_embedding = _embed_and_check(model, crop, index)
    print(
        f"\n{query['label']} query crop: {crop_output.resolve()} "
        f"dimensions={crop.width}x{crop.height} bbox={list(bbox)}"
    )
    print(
        f"  model={MODEL_NAME} dimension={query_embedding.size} "
        f"L2_norm={np.linalg.norm(query_embedding):.8f}"
    )

    if query["faiss_id"] is not None:
        mapped = connection.execute(
            """
            SELECT vector_id, parent_image_id, parent_path, class_label,
                   crop_type, x1, y1, x2, y2
            FROM region_vectors
            WHERE index_type = 'region' AND faiss_id = ?
            """,
            (query["faiss_id"],),
        ).fetchone()
        if mapped is None:
            raise RuntimeError(f"Query FAISS ID {query['faiss_id']} has no SQLite record")
        if (
            int(mapped["vector_id"]) != query["vector_id"]
            or int(mapped["parent_image_id"]) != query["parent_image_id"]
            or Path(mapped["parent_path"]).resolve() != query["image_path"]
            or not np.allclose(
                [mapped[key] for key in ("x1", "y1", "x2", "y2")],
                bbox,
                rtol=0.0,
                atol=1e-5,
            )
        ):
            raise RuntimeError("Query FAISS ID does not map to the selected SQLite crop")
        indexed_vector = index.reconstruct(query["faiss_id"])
        self_similarity = float(np.dot(query_embedding, indexed_vector))
        print(
            f"  selected mapping: sqlite_vector_id={mapped['vector_id']} "
            f"faiss_id={query['faiss_id']} parent_image_id={mapped['parent_image_id']} "
            f"label={mapped['class_label']} crop_type={mapped['crop_type']} "
            f"reembedded_index_cosine={self_similarity:.8f}"
        )
        if self_similarity < 0.999:
            raise RuntimeError(
                f"Current CLIP preprocessing/model does not reproduce the indexed "
                f"{query['label']} crop: cosine={self_similarity:.8f}"
            )

    for reference_label in ("person", "dog", "umbrella"):
        reference = _known_reference(connection, reference_label)
        reference_crop = _crop_from_metadata(reference)
        reference_embedding = _embed_and_check(model, reference_crop, index)
        stored_embedding = index.reconstruct(int(reference["faiss_id"]))
        direct_similarity = float(np.dot(query_embedding, reference_embedding))
        stored_similarity = float(np.dot(query_embedding, stored_embedding))
        self_similarity = float(np.dot(reference_embedding, stored_embedding))
        print(
            f"  direct cosine to {reference_label}: {direct_similarity:.8f}; "
            f"FAISS-vector dot={stored_similarity:.8f}; "
            f"reference reembed-vs-index={self_similarity:.8f}; "
            f"file={Path(reference['parent_path']).name} "
            f"faiss_id={reference['faiss_id']}"
        )
        if self_similarity < 0.999:
            raise RuntimeError(
                f"Fresh {reference_label} crop embedding does not match FAISS "
                f"vector {reference['faiss_id']}: cosine={self_similarity:.8f}"
            )

    scores, ids = index.search(query_embedding.reshape(1, -1), min(20, index.ntotal))
    raw_rows: list[dict] = []
    print(
        f"  top {len(ids[0])} raw region-index FAISS matches "
        "(raw inner-product/cosine; YOLO and grid may both appear):"
    )
    for rank, (score, faiss_id) in enumerate(zip(scores[0], ids[0]), start=1):
        metadata = connection.execute(
            """
            SELECT rv.vector_id, rv.parent_image_id, rv.parent_path,
                   rv.class_label, rv.crop_type, rv.x1, rv.y1, rv.x2, rv.y2
            FROM region_vectors rv
            WHERE rv.index_type = 'region' AND rv.faiss_id = ?
            """,
            (int(faiss_id),),
        ).fetchone()
        if metadata is None:
            raise RuntimeError(f"FAISS region ID {faiss_id} has no SQLite metadata")
        direct_score = float(np.dot(query_embedding, index.reconstruct(int(faiss_id))))
        if not np.isclose(float(score), direct_score, rtol=0.0, atol=1e-6):
            raise RuntimeError(
                f"FAISS and direct cosine disagree for ID {faiss_id}: "
                f"{score} vs {direct_score}"
            )
        print(
            f"    {rank:02d}. file={Path(metadata['parent_path']).name} "
            f"label={metadata['class_label']} crop_type={metadata['crop_type']} "
            f"cosine={float(score):.8f} vector_id={metadata['vector_id']} "
            f"faiss_id={faiss_id} parent_image_id={metadata['parent_image_id']} "
            f"bbox={[metadata['x1'], metadata['y1'], metadata['x2'], metadata['y2']]}"
        )
        raw_rows.append(
            {
                "query_label": query["label"],
                "query_image": str(query["image_path"]),
                "query_crop": str(crop_output.resolve()),
                "query_bbox": ",".join(str(value) for value in bbox),
                "query_crop_width": crop.width,
                "query_crop_height": crop.height,
                "model_name": MODEL_NAME,
                "embedding_dimension": query_embedding.size,
                "embedding_norm": float(np.linalg.norm(query_embedding)),
                "result_source": "raw_region_faiss",
                "rank": rank,
                "filename": Path(metadata["parent_path"]).name,
                "parent_path": metadata["parent_path"],
                "matched_label": metadata["class_label"],
                "crop_type": metadata["crop_type"],
                "bbox": ",".join(
                    str(metadata[key]) for key in ("x1", "y1", "x2", "y2")
                ),
                "raw_cosine_similarity": direct_score,
                "parent_image_id": metadata["parent_image_id"],
                "sqlite_vector_id": metadata["vector_id"],
                "faiss_id": int(faiss_id),
            }
        )

    results = search_regions(
        query_embedding,
        search_scope="regions",
        top_k=20,
        region_crop_type="yolo",
        query_region_id=0,
        query_label=query["label"],
        query_dimensions=crop.size,
        request_id=f"debug-{query['label']}",
        faiss_path=config.REGION_FAISS_PATH,
        db_path=config.REGION_DB_PATH,
    )
    if any(result["match_type"] != "yolo" for result in results):
        raise RuntimeError("Region-only API search unexpectedly returned non-YOLO crops")
    if len({result["parent_path"] for result in results}) != len(results):
        raise RuntimeError("Region-only API results contain duplicate parent images")
    service_rows: list[dict] = []
    print("  region-only YOLO service results (whole-image index excluded):")
    for rank, result in enumerate(results, start=1):
        result_row = connection.execute(
            """
            SELECT vector_id, faiss_id
            FROM region_vectors
            WHERE index_type = 'region'
              AND parent_path = ?
              AND class_label = ?
              AND crop_type = 'yolo'
              AND abs(x1 - ?) < 0.00001
              AND abs(y1 - ?) < 0.00001
              AND abs(x2 - ?) < 0.00001
              AND abs(y2 - ?) < 0.00001
            ORDER BY faiss_id
            LIMIT 1
            """,
            (str(Path(result["parent_path"]).resolve()), result["class_label"], *result["matched_bbox"]),
        ).fetchone()
        if result_row is None:
            raise RuntimeError("A service result has no matching SQLite region row")
        direct_score = float(
            np.dot(query_embedding, index.reconstruct(int(result_row["faiss_id"])))
        )
        if not np.isclose(result["score"], direct_score, rtol=0.0, atol=1e-6):
            raise RuntimeError(
                f"Service and direct similarity disagree for {result['parent_path']}"
            )
        print(
            f"    {rank:02d}. file={Path(result['parent_path']).name} "
            f"label={result['class_label']} crop_type={result['match_type']} "
            f"cosine={direct_score:.8f} vector_id={result_row['vector_id']} "
            f"faiss_id={result_row['faiss_id']}"
        )
        service_rows.append(
            {
                "query_label": query["label"],
                "query_image": str(query["image_path"]),
                "query_crop": str(crop_output.resolve()),
                "query_bbox": ",".join(str(value) for value in bbox),
                "query_crop_width": crop.width,
                "query_crop_height": crop.height,
                "model_name": MODEL_NAME,
                "embedding_dimension": query_embedding.size,
                "embedding_norm": float(np.linalg.norm(query_embedding)),
                "result_source": "region_only_yolo_service",
                "rank": rank,
                "filename": Path(result["parent_path"]).name,
                "parent_path": result["parent_path"],
                "matched_label": result["class_label"],
                "crop_type": result["match_type"],
                "bbox": ",".join(str(value) for value in result["matched_bbox"]),
                "raw_cosine_similarity": direct_score,
                "parent_image_id": metadata["parent_image_id"],
                "sqlite_vector_id": result_row["vector_id"],
                "faiss_id": result_row["faiss_id"],
            }
        )
    return raw_rows, service_rows


def _write_contact_sheet(results: list[dict], path: Path) -> None:
    columns = 4
    cell_width = 260
    image_height = 200
    caption_height = 48
    rows = (len(results) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * cell_width, rows * (image_height + caption_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for index, result in enumerate(results):
        source_path = Path(result["parent_path"])
        x = (index % columns) * cell_width
        y = (index // columns) * (image_height + caption_height)
        try:
            with Image.open(source_path) as source:
                thumbnail = source.convert("RGB")
                thumbnail.thumbnail((cell_width - 8, image_height - 8))
                sheet.paste(thumbnail, (x + (cell_width - thumbnail.width) // 2, y + 4))
        except OSError as error:
            raise RuntimeError(f"Unable to inspect returned image {source_path}: {error}") from error
        draw.text(
            (x + 4, y + image_height + 2),
            f"{result['rank']}. {source_path.name}\n"
            f"{result['matched_label']} / {result['crop_type']} / "
            f"{result['raw_cosine_similarity']:.6f}",
            fill="black",
            font=ImageFont.load_default(),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def run(args: argparse.Namespace) -> None:
    for path in (args.region_index, args.database, args.whole_index):
        if not path.is_file():
            raise FileNotFoundError(path)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    region_index = faiss.read_index(str(args.region_index))
    if region_index.metric_type != faiss.METRIC_INNER_PRODUCT or region_index.d != 512:
        raise RuntimeError(
            f"Expected a 512-D inner-product region index; got "
            f"dimension={region_index.d}, metric={region_index.metric_type}"
        )
    whole_index = faiss.read_index(str(args.whole_index))
    if whole_index.metric_type != faiss.METRIC_INNER_PRODUCT or whole_index.d != 512:
        raise RuntimeError("Whole-image index metric or dimension is invalid")

    connection = sqlite3.connect(args.database)
    connection.row_factory = sqlite3.Row
    try:
        _assert_faiss_sqlite_mapping(connection, region_index)
        whole_count = connection.execute(
            "SELECT COUNT(*) FROM region_vectors WHERE index_type = 'whole'"
        ).fetchone()[0]
        if whole_count != whole_index.ntotal:
            raise RuntimeError(
                f"Whole-image FAISS/SQLite count mismatch: {whole_index.ntotal} vs {whole_count}"
            )
        vectors = region_index.reconstruct_n(0, region_index.ntotal)
        norms = np.linalg.norm(vectors, axis=1)
        if not np.isfinite(vectors).all() or not np.allclose(
            norms, 1.0, rtol=0.0, atol=1e-4
        ):
            raise RuntimeError("Region index contains invalid or unnormalized vectors")

        print(f"Configured region FAISS index: {args.region_index.resolve()}")
        print(f"Configured whole-image FAISS index: {args.whole_index.resolve()}")
        print(f"Configured SQLite metadata DB: {args.database.resolve()}")
        print(
            f"Region index: dim={region_index.d} metric=inner_product "
            f"vectors={region_index.ntotal} norm_range="
            f"{norms.min():.8f}..{norms.max():.8f}"
        )
        print(f"Whole-image index vectors={whole_index.ntotal}")
        print(
            "Index/metadata times (ns): "
            f"region={args.region_index.stat().st_mtime_ns} "
            f"whole={args.whole_index.stat().st_mtime_ns} "
            f"sqlite={args.database.stat().st_mtime_ns}"
        )
        print(
            "Model provenance: index metadata does not record a CLIP model/revision. "
            "The script verifies compatibility by re-embedding stored crop samples."
        )

        model = load_clip_model()
        loaded_name = getattr(getattr(model.model, "config", None), "_name_or_path", None)
        processor_name = getattr(model.processor, "name_or_path", None)
        print(
            f"Current query model={MODEL_NAME}; loaded model={loaded_name}; "
            f"processor={processor_name}; image preprocessing="
            "CLIPProcessor(images=..., return_tensors='pt'), RGB."
        )

        queries = [
            _query_region(
                connection,
                "person",
                args.person_image,
                args.person_bbox,
            )
        ]
        if not args.skip_dog:
            queries.append(
                _query_region(
                    connection,
                    "dog",
                    args.dog_image,
                    args.dog_bbox,
                )
            )
        invalidate_region_index_cache()
        csv_rows: list[dict] = []
        person_service_results: list[dict] = []
        for query in queries:
            destination = (
                args.person_crop_output
                if query["label"] == "person" and args.person_crop_output
                else args.output_dir / f"debug_{query['label']}_crop.jpg"
            )
            raw_rows, service_rows = _report_query(
                connection, region_index, model, query, destination
            )
            csv_rows.extend(raw_rows)
            csv_rows.extend(service_rows)
            if query["label"] == "person":
                person_service_results = service_rows
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"CSV written: {args.output_csv.resolve()} ({len(csv_rows)} rows)")
        if person_service_results:
            _write_contact_sheet(
                person_service_results,
                args.person_contact_sheet,
            )
            print(f"Person-result contact sheet: {args.person_contact_sheet.resolve()}")
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region-index", type=Path, default=config.REGION_FAISS_PATH)
    parser.add_argument("--whole-index", type=Path, default=config.REGION_WHOLE_FAISS_PATH)
    parser.add_argument("--database", type=Path, default=config.REGION_DB_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-csv", type=Path, default=Path("debug_region_retrieval.csv")
    )
    parser.add_argument("--person-crop-output", type=Path)
    parser.add_argument(
        "--person-contact-sheet",
        type=Path,
        default=Path("debug_region_person_results.jpg"),
    )
    parser.add_argument("--person-image", type=Path)
    parser.add_argument("--person-bbox", type=_parse_bbox)
    parser.add_argument("--dog-image", type=Path)
    parser.add_argument("--dog-bbox", type=_parse_bbox)
    parser.add_argument(
        "--skip-dog",
        action="store_true",
        help="Audit only the person query instead of adding the indexed dog control.",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
