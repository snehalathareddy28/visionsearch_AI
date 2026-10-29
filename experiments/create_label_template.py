
import csv
from pathlib import Path

root = Path(__file__).resolve().parent.parent
ranking_file = root / "experiments" / "results" / "ranking_comparison.csv"
label_file = root / "experiments" / "results" / "relevance_labels.csv"

pairs = set()

with ranking_file.open("r", newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        pairs.add((row["query"].strip(), row["image_path"].strip()))

with label_file.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["query", "image_path", "relevant"])
    for query, image_path in sorted(pairs):
        writer.writerow([query, image_path, ""])

print(f"Recreated {label_file}")
print(f"Rows requiring labels: {len(pairs)}")
