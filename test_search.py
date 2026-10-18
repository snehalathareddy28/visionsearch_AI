"""Run a text query against the persisted CLIP/FAISS index."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.clip_model import load_clip_model
from services.search_service import search


if __name__ == "__main__":
    query = input("Enter search query: ").strip()
    if not query:
        raise SystemExit("Search query cannot be empty.")

    model = load_clip_model()
    try:
        results = search(model.encode_text(query), top_k=5)
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error
    print("\nSearch Results:\n")
    if not results:
        print("No results found.")
    for result in results:
        print(f"{result['rank']}. {result['image_path']}")
        print(f"   Similarity: {result['score']:.4f}")
        print(f"   FAISS ID: {result['faiss_id']}\n")
