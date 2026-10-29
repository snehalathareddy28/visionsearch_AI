from services.search_service import search_by_text

queries = [
    "dog",
    "a photo of a dog",
    "a dog running",
    "a dog on grass",
    "a dog running on grass",
    "a photo of a dog running on green grass",
]

for query in queries:
    print("\nQUERY:", query)
    results = search_by_text(query, limit=5)

    for result in results:
        name = result["image_path"].split("\\")[-1]
        category = result["image_path"].split("\\")[-2]

        print(
            f"{result['score']:.4f} | "
            f"{category}/{name}"
        )
