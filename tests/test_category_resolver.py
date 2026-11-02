from visionsearch_web_ui.region.category_resolver import (
    category_adjustment,
    resolve_query_category,
)
from visionsearch_web_ui.region.config import CATEGORY_ALIASES


def test_resolves_object_mentions_from_natural_language():
    labels = {"person", "dog", "car"}

    assert resolve_query_category("a person standing", labels, CATEGORY_ALIASES) == "person"
    assert resolve_query_category(
        "a dog running on grass", labels, CATEGORY_ALIASES
    ) == "dog"
    assert resolve_query_category("a red car", labels, CATEGORY_ALIASES) == "car"


def test_unknown_or_ambiguous_categories_remain_unresolved():
    labels = {"person", "dog", "car"}

    assert resolve_query_category("a butterfly", labels, CATEGORY_ALIASES) is None
    assert resolve_query_category("a guitar", labels, CATEGORY_ALIASES) is None
    assert resolve_query_category("a person with a dog", labels, CATEGORY_ALIASES) is None
    assert resolve_query_category("a person standing", set(), CATEGORY_ALIASES) is None


def test_observed_custom_detector_label_can_be_resolved():
    assert (
        resolve_query_category("a butterfly", {"butterfly"}, CATEGORY_ALIASES)
        == "butterfly"
    )


def test_adjustment_is_soft_and_skips_uncertain_or_unknown_labels():
    known = {"person", "dog"}
    options = {
        "known_labels": known,
        "aliases": CATEGORY_ALIASES,
        "min_confidence": 0.6,
        "bonus": 0.08,
        "penalty": 0.03,
    }

    assert category_adjustment(
        {"class_label": "person", "crop_type": "yolo", "confidence": 0.9},
        "person",
        **options,
    ) == 0.08
    assert category_adjustment(
        {"class_label": "dog", "crop_type": "yolo", "confidence": 0.9},
        "person",
        **options,
    ) == -0.03

    for row in (
        {"class_label": "person", "crop_type": "yolo", "confidence": 0.2},
        {"class_label": None, "crop_type": "yolo", "confidence": 0.9},
        {"class_label": "unknown", "crop_type": "yolo", "confidence": 0.9},
        {"class_label": "grid_0_0", "crop_type": "grid", "confidence": 0.0},
    ):
        assert category_adjustment(row, "person", **options) == 0.0
