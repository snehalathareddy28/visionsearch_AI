import numpy as np
from pathlib import Path
from models.clip_model import load_clip_model
from services.search_service import load_index

model = load_clip_model()
index, paths = load_index()

queries = ["dog", "a dog", "a dog running on grass"]

print("\nTotal indexed images:", index.ntotal)
print("Total image paths:", len(paths))

for query in queries:
    print("\n" + "=" * 65)
    print("QUERY:", query)

    embedding = np.asarray(
        model.encode_text(query),
        dtype=np.float32
    )

    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)

    scores, ids = index.search(embedding, index.ntotal)

    ranked = [
        (float(score), int(idx))
        for score, idx in zip(scores[0], ids[0])
        if idx >= 0
    ]

    print("\nTOP 10 RESULTS:")
    for score, idx in ranked[:10]:
        print(f"{score:.4f} | {paths[idx]}")

    print("\nYOUR DOG IMAGES:")
    dog_results = [
        (score, idx)
        for score, idx in ranked
        if "\\dog\\" in paths[idx].lower()
    ]

    for score, idx in dog_results:
        print(f"{score:.4f} | {paths[idx]}")

    if not dog_results:
        print("No paths containing '\\dog\\' found.")
