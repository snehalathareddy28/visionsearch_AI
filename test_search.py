from services.search_service import search_by_text

queries = ["dog", "a dog", "a dog running on grass"]

for query in queries:
    print("\nQUERY:", query)
    results = search_by_text(query, limit=5)

    for result in results:
        print(
            f"{result['score']:.4f} | "
            f"{result['image_path']}"
        )
