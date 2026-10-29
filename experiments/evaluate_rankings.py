
"""Evaluate ranking results using manually assigned relevance labels."""

import csv
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_FILE = PROJECT_ROOT / "experiments" / "results" / "ranking_comparison.csv"
LABELS_FILE = PROJECT_ROOT / "experiments" / "results" / "relevance_labels.csv"


def precision_at_k(relevance, k=5):
    """Calculate Precision@K from binary relevance labels."""
    return sum(relevance[:k]) / k


def average_precision(relevance):
    """Calculate average precision for a ranked list."""
    relevant_total = sum(relevance)

    if relevant_total == 0:
        return 0.0

    hits = 0
    precision_sum = 0.0

    for rank, is_relevant in enumerate(relevance, start=1):
        if is_relevant:
            hits += 1
            precision_sum += hits / rank

    return precision_sum / relevant_total


def main():
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"Ranking results not found: {RESULTS_FILE}"
        )

    if not LABELS_FILE.exists():
        raise FileNotFoundError(
            f"Create the relevance-label file first: {LABELS_FILE}"
        )

    with RESULTS_FILE.open(
        newline="", encoding="utf-8"
    ) as file:
        results = list(csv.DictReader(file))

    with LABELS_FILE.open(
    newline="", encoding="utf-8-sig"
) as file:
        labels = {
            (row["query"], row["image_path"]): int(row["relevant"])
            for row in csv.DictReader(file)
        }

    grouped = defaultdict(list)

    for row in results:
        key = (row["query"], row["method"])
        image_key = (row["query"], row["image_path"])

        if image_key not in labels:
            raise ValueError(
                "Missing relevance label for: "
                f"{row['query']} | {row['image_path']}"
            )

        grouped[key].append((
            int(row["rank"]),
            labels[image_key],
        ))

    print("\nRanking Quality Evaluation")
    print("=" * 65)
    print(f"{'Query':25} {'Method':20} {'P@5':>8} {'AP@5':>8}")
    print("-" * 65)

    metrics = defaultdict(list)

    for (query, method), items in sorted(grouped.items()):
        items.sort(key=lambda item: item[0])
        relevance = [value for _, value in items]

        p_at_5 = precision_at_k(relevance, 5)
        ap_at_5 = average_precision(relevance)

        metrics[method].append((p_at_5, ap_at_5))

        print(
            f"{query:25} {method:20} "
            f"{p_at_5:8.3f} {ap_at_5:8.3f}"
        )

    print("-" * 65)

    for method, values in sorted(metrics.items()):
        mean_p = sum(item[0] for item in values) / len(values)
        mean_ap = sum(item[1] for item in values) / len(values)

        print(
            f"MEAN — {method}: "
            f"Precision@5={mean_p:.3f}, AP@5={mean_ap:.3f}"
        )

    print(
        "\nNote: these metrics are meaningful only when the relevance "
        "labels have been manually checked against the actual images."
    )


if __name__ == "__main__":
    main()
