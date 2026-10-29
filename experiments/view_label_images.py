
import csv
from pathlib import Path
from PIL import Image, ImageDraw
import math

ROOT = Path(__file__).resolve().parent.parent
LABEL_FILE = ROOT / "experiments" / "results" / "relevance_labels.csv"
OUTPUT_FILE = ROOT / "experiments" / "results" / "label_review.jpg"

with LABEL_FILE.open("r", newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

if not rows:
    raise ValueError("No images found in relevance_labels.csv")

thumb_w, thumb_h = 220, 180
label_h = 65
columns = 4
cards = []

for row in rows:
    image_path = Path(row["image_path"])
    query = row["query"]

    try:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            image.thumbnail((thumb_w, thumb_h))
    except Exception as exc:
        print(f"Could not open {image_path}: {exc}")
        continue

    card = Image.new("RGB", (thumb_w, thumb_h + label_h), "white")
    x = (thumb_w - image.width) // 2
    y = (thumb_h - image.height) // 2
    card.paste(image, (x, y))

    draw = ImageDraw.Draw(card)
    draw.text((5, thumb_h + 4), f"Query: {query[:28]}", fill="black")
    draw.text((5, thumb_h + 23), f"Image: {image_path.parent.name}/{image_path.name}", fill="black")
    draw.text((5, thumb_h + 42), f"Current label: {row['relevant'] or 'BLANK'}", fill="blue")
    cards.append(card)

if not cards:
    raise ValueError("No images could be opened.")

rows_count = math.ceil(len(cards) / columns)
sheet = Image.new(
    "RGB",
    (columns * thumb_w, rows_count * (thumb_h + label_h)),
    "#dddddd"
)

for i, card in enumerate(cards):
    x = (i % columns) * thumb_w
    y = (i // columns) * (thumb_h + label_h)
    sheet.paste(card, (x, y))

sheet.save(OUTPUT_FILE, quality=90)
print(f"Contact sheet saved to: {OUTPUT_FILE}")
print(f"Images displayed: {len(cards)}")

