"""
evaluate_regions.py — Compare whole-image vs region-based retrieval.

Usage
-----
    python evaluate_regions.py \\
        --dataset-dir dataset/caltech-101-selected \\
        --queries queries.json \\
        --top-k 10 \\
        --output-dir results

Metrics computed (per mode: whole, regions, both)
-------------------------------------------------
- Recall@1, Recall@5, Recall@10  (fraction of queries where ground-truth
  image appears in top-k results)
- MRR (Mean Reciprocal Rank)
- Mean query latency (seconds)
- Index size: number of FAISS vectors and DB file size in MB

Query file format (JSON)
------------------------
A list of dicts:
[
  {
    "query": "a dog running on grass",
    "relevant_images": ["dataset/caltech-101-selected/dog/image_0001.jpg"]
  },
  ...
]

If --queries is omitted, a built-in set of 20 demo queries for Caltech-101
categories is used (no COCO download required).

Output
------
results/region_eval_metrics.csv   — one row per mode with all metrics
results/region_eval_chart.png     — grouped bar chart
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in demo queries (Caltech-101 categories, no COCO needed)
# ---------------------------------------------------------------------------

DEMO_QUERIES = [
    {"query": "a dog", "category": "dog"},
    {"query": "a cat face", "category": "Faces_easy"},
    {"query": "a stop sign", "category": "stop_sign"},
    {"query": "a wild horse", "category": "wild_horse"},
    {"query": "a motorbike", "category": "motorbikes"},
    {"query": "an elephant", "category": "elephant"},
    {"query": "an airplane", "category": "airplanes"},
    {"query": "a camera", "category": "camera"},
    {"query": "a chair", "category": "chair"},
    {"query": "scissors", "category": "scissors"},
    # Small-object queries (labelled for sub-analysis)
    {"query": "a small ant", "category": "ant", "small_object": True},
    {"query": "a tiny butterfly", "category": "butterfly", "small_object": True},
    {"query": "a bee on a flower", "category": "bonsai", "small_object": True},
    {"query": "a small stapler", "category": "stapler", "small_object": True},
    {"query": "a key", "category": "bonsai", "small_object": True},  # proxy
]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def recall_at_k(ranked_paths: list[str], relevant_paths: set[str], k: int) -> float:
    """1 if any relevant image is in the top-k results, else 0."""
    for path in ranked_paths[:k]:
        if path in relevant_paths:
            return 1.0
    return 0.0


def reciprocal_rank(ranked_paths: list[str], relevant_paths: set[str]) -> float:
    """1/rank of first relevant result, or 0 if not found."""
    for rank, path in enumerate(ranked_paths, start=1):
        if path in relevant_paths:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Result path extraction
# ---------------------------------------------------------------------------


def extract_paths(results: list[dict]) -> list[str]:
    """Extract parent_path strings from region search results."""
    return [r["parent_path"] for r in results]


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def build_relevant_set(query_dict: dict, dataset_dir: Path) -> set[str]:
    """Find all images whose parent folder name matches the query category."""
    if "relevant_images" in query_dict:
        return set(query_dict["relevant_images"])
    category = query_dict.get("category", "")
    relevant = set()
    for img_path in dataset_dir.rglob("*"):
        if img_path.is_file() and img_path.parent.name.lower() == category.lower():
            relevant.add(str(img_path.resolve()))
    return relevant


def run_evaluation(
    queries: list[dict],
    dataset_dir: Path,
    top_k: int,
    output_dir: Path,
) -> None:
    from models.clip_model import load_clip_model
    from visionsearch_web_ui.region.search_service import (
        invalidate_region_index_cache,
        search_regions,
    )
    from visionsearch_web_ui.region import config as region_config

    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_clip_model()
    invalidate_region_index_cache()

    scopes = ["whole", "regions", "both"]
    row_metrics: dict[str, dict] = {s: {"r1": [], "r5": [], "r10": [], "rr": [], "latency": []} for s in scopes}
    small_obj_metrics: dict[str, dict] = {s: {"r1": [], "r5": [], "r10": [], "rr": []} for s in scopes}

    for q in queries:
        query_text = q["query"]
        relevant_paths = build_relevant_set(q, dataset_dir)
        is_small = q.get("small_object", False)

        if not relevant_paths:
            logger.warning("No relevant images found for query '%s' (category=%s). Skipping.", query_text, q.get("category"))
            continue

        print(f"  Query: {query_text!r}  ({len(relevant_paths)} relevant images)")

        # Embed query once
        embedding = model.encode_text(query_text)

        for scope in scopes:
            t0 = time.perf_counter()
            results = search_regions(
                embedding,
                search_scope=scope,
                top_k=top_k,
                faiss_path=region_config.REGION_FAISS_PATH,
                db_path=region_config.REGION_DB_PATH,
            )
            elapsed = time.perf_counter() - t0

            ranked = extract_paths(results)

            r1 = recall_at_k(ranked, relevant_paths, 1)
            r5 = recall_at_k(ranked, relevant_paths, 5)
            r10 = recall_at_k(ranked, relevant_paths, top_k)
            rr = reciprocal_rank(ranked, relevant_paths)

            row_metrics[scope]["r1"].append(r1)
            row_metrics[scope]["r5"].append(r5)
            row_metrics[scope]["r10"].append(r10)
            row_metrics[scope]["rr"].append(rr)
            row_metrics[scope]["latency"].append(elapsed)

            if is_small:
                small_obj_metrics[scope]["r1"].append(r1)
                small_obj_metrics[scope]["r5"].append(r5)
                small_obj_metrics[scope]["r10"].append(r10)
                small_obj_metrics[scope]["rr"].append(rr)

    # ---- Index size ----
    from visionsearch_web_ui.region.index_manager import get_index_stats
    stats = get_index_stats()

    # ---- Aggregate ----
    def mean(lst): return sum(lst) / len(lst) if lst else 0.0

    import csv

    csv_path = output_dir / "region_eval_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "scope", "recall@1", "recall@5", f"recall@{top_k}", "MRR",
            "mean_latency_s",
            "small_obj_recall@1", "small_obj_recall@5", f"small_obj_recall@{top_k}",
            "faiss_vectors", "faiss_size_mb", "db_size_mb",
        ])
        for scope in scopes:
            m = row_metrics[scope]
            sm = small_obj_metrics[scope]
            writer.writerow([
                scope,
                round(mean(m["r1"]), 4),
                round(mean(m["r5"]), 4),
                round(mean(m["r10"]), 4),
                round(mean(m["rr"]), 4),
                round(mean(m["latency"]), 4),
                round(mean(sm["r1"]), 4) if sm["r1"] else "n/a",
                round(mean(sm["r5"]), 4) if sm["r5"] else "n/a",
                round(mean(sm["r10"]), 4) if sm["r10"] else "n/a",
                stats.get("faiss_vectors", "n/a"),
                stats.get("faiss_size_mb", "n/a"),
                stats.get("db_size_mb", "n/a"),
            ])

    print(f"\n✓ CSV saved to {csv_path}")

    # ---- Print table ----
    print("\n" + "=" * 70)
    print(f"{'Scope':10} {'R@1':>7} {'R@5':>7} {f'R@{top_k}':>8} {'MRR':>7} {'Latency':>10}")
    print("-" * 70)
    for scope in scopes:
        m = row_metrics[scope]
        print(
            f"{scope:10} {mean(m['r1']):7.4f} {mean(m['r5']):7.4f} "
            f"{mean(m['r10']):8.4f} {mean(m['rr']):7.4f} "
            f"{mean(m['latency'])*1000:8.1f} ms"
        )
    print("=" * 70)

    # ---- Matplotlib bar chart ----
    try:
        import matplotlib.pyplot as plt

        metric_names = ["Recall@1", "Recall@5", f"Recall@{top_k}", "MRR"]
        scope_means = {
            s: [
                mean(row_metrics[s]["r1"]),
                mean(row_metrics[s]["r5"]),
                mean(row_metrics[s]["r10"]),
                mean(row_metrics[s]["rr"]),
            ]
            for s in scopes
        }

        x = np.arange(len(metric_names))
        width = 0.25
        fig, ax = plt.subplots(figsize=(9, 5))
        colours = ["#4c9be8", "#7eebd9", "#ffd97d"]
        for i, (scope, vals) in enumerate(scope_means.items()):
            ax.bar(x + i * width, vals, width, label=scope.capitalize(), color=colours[i])

        ax.set_xticks(x + width)
        ax.set_xticklabels(metric_names)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title("VisionSearch AI — Whole vs Region Retrieval")
        ax.legend()
        plt.tight_layout()

        chart_path = output_dir / "region_eval_chart.png"
        fig.savefig(str(chart_path), dpi=150)
        print(f"✓ Chart saved to {chart_path}")
        plt.close(fig)
    except Exception as exc:
        logger.warning("Could not generate chart: %s", exc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate region-based retrieval.")
    parser.add_argument(
        "--dataset-dir",
        default="dataset/caltech-101-selected",
        help="Root directory of the dataset (default: dataset/caltech-101-selected)",
    )
    parser.add_argument(
        "--queries",
        default=None,
        help="Path to a JSON query file. If omitted, built-in demo queries are used.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Recall cutoff (default: 10)")
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for CSV and chart output (default: results)",
    )
    args = parser.parse_args()

    dataset_dir = (PROJECT_ROOT / args.dataset_dir).resolve()
    if not dataset_dir.is_dir():
        print(f"ERROR: Dataset directory not found: {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    if args.queries:
        with open(args.queries, encoding="utf-8") as fh:
            queries = json.load(fh)
    else:
        print("No --queries file provided. Using built-in Caltech-101 demo queries.")
        queries = DEMO_QUERIES

    output_dir = (PROJECT_ROOT / args.output_dir).resolve()

    print(f"\nVisionSearch AI — Region Retrieval Evaluation")
    print(f"Dataset   : {dataset_dir}")
    print(f"Queries   : {len(queries)}")
    print(f"Top-k     : {args.top_k}")
    print(f"Output dir: {output_dir}\n")

    run_evaluation(queries, dataset_dir, args.top_k, output_dir)
