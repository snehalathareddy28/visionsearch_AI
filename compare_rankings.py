
"""Compare baseline CLIP ranking with attribute-aware ranking.

This experiment does not modify the existing application or FAISS index.
It writes ranking results and timing observations to experiments/results/.
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.clip_model import load_clip_model
from services.attribute_service import extract_attributes
from services.search_service import (
    ATTRIBUTE_WEIGHT,
    CATEGORY_BONUS,
    CATEGORY_ALIASES,
    DEFAULT_SIMILARITY_THRESHOLD,
    load_index,
)

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "results"

TEST_QUERIES = [
    "car",
    "dog",
    "flower",
    "a car on the road",
    "a dog running",
]


def display_image_path(image_path):
    """Show the class folder as well as the filename."""
    path = Path(image_path)
    return f"{path.parent.name}/{path.name}"


def build_attribute_prompts(query):
    """Build prompts using the application's existing attribute parser."""
    attributes = extract_attributes(query)

    objects = attributes.get("objects", [])
    colors = attributes.get("colors", [])
    actions = attributes.get("actions", [])
    scenes = attributes.get("scenes", [])
    scene_elements = attributes.get("scene_elements", [])

    prompts = []

    if objects:
        prompts.append(f"a photo of {' '.join(objects)}")

    if objects and colors:
        prompts.append(
            f"a photo of {' '.join(colors)} {' '.join(objects)}"
        )

    if objects and actions:
        prompts.append(
            f"a photo of {' '.join(objects)} {' '.join(actions)}"
        )

    surroundings = scenes + scene_elements
    if objects and surroundings:
        prompts.append(
            f"a photo of {' '.join(objects)} "
            f"in {' '.join(surroundings)}"
        )

    return list(dict.fromkeys(prompts))


def get_category_match(image_path, query):
    """Match query objects to the dataset folder name."""
    attributes = extract_attributes(query)

    objects = {
        str(term).lower().replace("_", " ").strip()
        for term in attributes.get("objects", [])
        if str(term).strip()
    }

    expanded_objects = set(objects)

    for obj in objects:
        expanded_objects.update(CATEGORY_ALIASES.get(obj, set()))

    folder = Path(image_path).parent.name.lower().replace("_", " ")
    folder_words = set(folder.split())

    if not expanded_objects:
        return False

    normalized_objects = {
        word
        for obj in expanded_objects
        for word in obj.replace("_", " ").split()
    }

    if normalized_objects & folder_words:
        return True

    return any(
        alias.replace("_", " ") in folder
        for alias in expanded_objects
    )


def retrieve_candidates(model, index, paths, query):
    """Encode a query and retrieve threshold-qualified candidates."""
    query_embedding = model.encode_text(query)

    query_vector = np.asarray(
        query_embedding, dtype=np.float32
    ).reshape(1, -1)

    # CLIP embeddings are expected to be normalized by the model.
    scores, ids = index.search(query_vector, index.ntotal)

    candidates = []

    for score, image_id in zip(scores[0], ids[0]):
        if image_id < 0 or score < DEFAULT_SIMILARITY_THRESHOLD:
            continue

        candidates.append({
            "image_path": paths[int(image_id)],
            "faiss_id": int(image_id),
            "clip_score": float(score),
        })

    return candidates


def rank_baseline(candidates):
    """Rank candidates using the original full-query CLIP score."""
    return sorted(
        candidates,
        key=lambda item: item["clip_score"],
        reverse=True,
    )[:5]


def rank_attribute_aware(model, index, candidates, query):
    """Rerank candidates using attribute prompts and category bonus."""
    prompts = build_attribute_prompts(query)

    attribute_score_maps = []

    for prompt in prompts:
        embedding = model.encode_text(prompt)

        vector = np.asarray(
            embedding, dtype=np.float32
        ).reshape(1, -1)

        norm = np.linalg.norm(vector)

        if norm == 0:
            continue

        vector = vector / norm

        scores, ids = index.search(vector, index.ntotal)

        score_map = {
            int(image_id): float(score)
            for score, image_id in zip(scores[0], ids[0])
            if image_id >= 0
        }

        attribute_score_maps.append(score_map)

    proposed = []

    for item in candidates:
        image_id = item["faiss_id"]

        matching_scores = [
            score_map[image_id]
            for score_map in attribute_score_maps
            if image_id in score_map
        ]

        ranking_score = item["clip_score"]

        if matching_scores:
            ranking_score = (
                (1 - ATTRIBUTE_WEIGHT) * item["clip_score"]
                + ATTRIBUTE_WEIGHT * max(matching_scores)
            )

        category_match = get_category_match(
            item["image_path"], query
        )

        if category_match:
            ranking_score += CATEGORY_BONUS

        proposed.append({
            **item,
            "ranking_score": ranking_score,
            "category_match": category_match,
        })

    proposed.sort(
        key=lambda item: item["ranking_score"],
        reverse=True,
    )

    return proposed[:5]


def write_result_rows(writer, query, method, results, latency_ms):
    """Write ranked results to the CSV file."""
    rows = []

    for rank, item in enumerate(results, start=1):
        row = {
            "query": query,
            "method": method,
            "rank": rank,
            "image_path": item["image_path"],
            "clip_score": round(item["clip_score"], 6),
            "ranking_score": round(
                item.get("ranking_score", item["clip_score"]), 6
            ),
            "category_match": get_category_match(
                item["image_path"], query
            ),
            "latency_ms": round(latency_ms, 3),
        }

        writer.writerow(row)
        rows.append(row)

    return rows


def run_experiment():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading CLIP model...")
    model = load_clip_model()

    print("Loading FAISS index...")
    index, paths = load_index()

    if index.ntotal == 0:
        raise RuntimeError("The FAISS index contains no images.")

    if index.ntotal != len(paths):
        raise RuntimeError(
            f"Index/path mismatch: {index.ntotal} vectors, "
            f"{len(paths)} paths."
        )

    csv_path = OUTPUT_DIR / "ranking_comparison.csv"
    summary_path = OUTPUT_DIR / "experiment_summary.txt"

    csv_fields = [
        "query",
        "method",
        "rank",
        "image_path",
        "clip_score",
        "ranking_score",
        "category_match",
        "latency_ms",
    ]

    baseline_times = []
    proposed_times = []

    with csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=csv_fields)
        writer.writeheader()

        for query in TEST_QUERIES:
            print(f"\nQuery: {query}")

            # Baseline timing includes query encoding, retrieval,
            # threshold filtering, and baseline ranking.
            baseline_start = time.perf_counter()

            baseline_candidates = retrieve_candidates(
                model, index, paths, query
            )
            baseline = rank_baseline(baseline_candidates)

            baseline_latency = (
                time.perf_counter() - baseline_start
            ) * 1000

            baseline_times.append(baseline_latency)

            # Proposed timing includes the same query encoding and
            # retrieval, plus attribute prompt encoding and reranking.
            proposed_start = time.perf_counter()

            proposed_candidates = retrieve_candidates(
                model, index, paths, query
            )
            proposed = rank_attribute_aware(
                model, index, proposed_candidates, query
            )

            proposed_latency = (
                time.perf_counter() - proposed_start
            ) * 1000

            proposed_times.append(proposed_latency)

            baseline_rows = write_result_rows(
                writer,
                query,
                "baseline_clip",
                baseline,
                baseline_latency,
            )

            proposed_rows = write_result_rows(
                writer,
                query,
                "attribute_aware",
                proposed,
                proposed_latency,
            )

            print("  Baseline top results:")
            for item in baseline_rows:
                print(
                    f"    {item['rank']}. "
                    f"{display_image_path(item['image_path'])} "
                    f"(CLIP={item['clip_score']:.4f})"
                )

            print("  Attribute-aware top results:")
            for item in proposed_rows:
                print(
                    f"    {item['rank']}. "
                    f"{display_image_path(item['image_path'])} "
                    f"(ranking={item['ranking_score']:.4f}, "
                    f"CLIP={item['clip_score']:.4f})"
                )

            print(
                f"  Timing observation: "
                f"baseline={baseline_latency:.2f} ms; "
                f"proposed={proposed_latency:.2f} ms"
            )

    with summary_path.open("w", encoding="utf-8") as file:
        file.write("VisionSearch AI - Ranking Experiment\n")
        file.write("====================================\n\n")
        file.write(f"Indexed images: {index.ntotal}\n")
        file.write(
            f"Similarity threshold: "
            f"{DEFAULT_SIMILARITY_THRESHOLD}\n"
        )
        file.write(f"Attribute weight: {ATTRIBUTE_WEIGHT}\n")
        file.write(f"Category bonus: {CATEGORY_BONUS}\n")
        file.write(f"Queries tested: {len(TEST_QUERIES)}\n\n")

        file.write(
            "Mean baseline latency (ms): "
            f"{np.mean(baseline_times):.3f}\n"
        )
        file.write(
            "Mean attribute-aware latency (ms): "
            f"{np.mean(proposed_times):.3f}\n\n"
        )

        file.write(
            "Timing measurements are observations from this run, "
            "not definitive performance benchmarks. Model warm-up, "
            "CPU load, and execution order can affect timings. "
            "For stronger timing claims, repeat each query several "
            "times and report median latency after warm-up.\n\n"
        )

        file.write(
            "Precision@5, Recall@5, and mAP are not calculated "
            "because relevance labels have not yet been provided. "
            "Ranking scores alone do not establish retrieval quality.\n\n"
        )

        file.write(
            "CSV output: experiments/results/ranking_comparison.csv\n"
        )

    print("\nExperiment finished.")
    print(f"Results CSV: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    run_experiment()
