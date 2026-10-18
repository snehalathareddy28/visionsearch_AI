"""Combine vector similarity with metadata and attribute scores."""

from services.attribute_service import extract_attributes
from services.search_service import search_by_text


def hybrid_search(query: str, limit: int = 20):
    """Merge keyword/attribute heuristics with the existing FAISS text search."""
    if not isinstance(query, str):
        raise TypeError("Query must be a string")

    query = query.strip()
    if not query:
        return []

    attrs = extract_attributes(query)
    base_results = search_by_text(query, limit=limit)
    if not base_results:
        return []

    for result in base_results:
        result["attributes"] = attrs
        result["hybrid_score"] = float(result.get("score", 0.0))
        if attrs.get("colors"):
            result["hybrid_score"] += 0.01 * min(len(attrs["colors"]), 3)
        if attrs.get("scenes"):
            result["hybrid_score"] += 0.01 * min(len(attrs["scenes"]), 3)

    base_results.sort(key=lambda item: item["hybrid_score"], reverse=True)
    return base_results[:limit]
