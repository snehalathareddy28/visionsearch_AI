"""
test_region_search.py — pytest tests for region search_service.py

Tests cover:
- search_regions groups results by parent image
- search_scope="whole" returns only whole-image vectors
- search_scope="regions" returns only crop vectors
- search_scope="both" combines scores correctly
- Results are sorted by descending score
- top_k limits number of results
- Invalid scope raises ValueError
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from visionsearch_web_ui.region.metadata_db import RegionMetadataDB
from visionsearch_web_ui.region.region_splitter import RegionBox

DIM = 512

SAMPLE_BOX = RegionBox(
    x1=10.0, y1=10.0, x2=100.0, y2=100.0,
    class_label="cat", confidence=0.75, crop_type="yolo",
)


@pytest.fixture(scope="module")
def index_paths():
    """Create temp dir and return (faiss_path, db_path)."""
    tmpdir = tempfile.mkdtemp()
    return Path(tmpdir) / "test_region.index", Path(tmpdir) / "test_region.db"


@pytest.fixture(scope="module")
def built_index(index_paths):
    """Build a minimal FAISS index + SQLite DB for testing.

    vector 0 → parent A, whole image  (strong match, component 0 = 1.0)
    vector 1 → parent A, yolo region  (slightly weaker, component 0 = 0.9)
    vector 2 → parent B, whole image  (weaker, component 0 = 0.5)
    """
    import faiss

    faiss_path, db_path = index_paths

    vecs = np.zeros((3, DIM), dtype=np.float32)
    vecs[0, 0] = 1.0   # parent A whole
    vecs[1, 0] = 0.9   # parent A region
    vecs[2, 0] = 0.5   # parent B whole

    idx = faiss.IndexFlatIP(DIM)
    idx.add(vecs)
    faiss_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(idx, str(faiss_path))

    with RegionMetadataDB(db_path) as db:
        pid_a = db.add_parent_image("/images/a.jpg", "hash_a")
        pid_b = db.add_parent_image("/images/b.jpg", "hash_b")
        db.add_region_vector(0, pid_a, "/images/a.jpg", None, is_whole_image=True)
        db.add_region_vector(1, pid_a, "/images/a.jpg", SAMPLE_BOX)
        db.add_region_vector(2, pid_b, "/images/b.jpg", None, is_whole_image=True)

    # Invalidate any cached index from prior runs
    from visionsearch_web_ui.region.search_service import invalidate_region_index_cache
    invalidate_region_index_cache()

    return faiss_path, db_path


QUERY_VEC = np.zeros((1, DIM), dtype=np.float32)
QUERY_VEC[0, 0] = 1.0


def _search(built_index, **kwargs):
    from visionsearch_web_ui.region.search_service import search_regions
    faiss_path, db_path = built_index
    return search_regions(
        QUERY_VEC,
        faiss_path=faiss_path,
        db_path=db_path,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# scope = "whole"
# ---------------------------------------------------------------------------


class TestScopeWhole:
    def test_returns_only_whole_image_results(self, built_index):
        results = _search(built_index, search_scope="whole", top_k=10)
        assert all(r["match_type"] == "whole" for r in results)

    def test_returns_two_parents(self, built_index):
        results = _search(built_index, search_scope="whole", top_k=10)
        assert len(results) == 2

    def test_sorted_descending(self, built_index):
        results = _search(built_index, search_scope="whole", top_k=10)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# scope = "regions"
# ---------------------------------------------------------------------------


class TestScopeRegions:
    def test_returns_only_region_results(self, built_index):
        results = _search(built_index, search_scope="regions", top_k=10)
        # Only parent A has a crop vector; parent B has no crop.
        assert len(results) == 1
        assert results[0]["class_label"] == "cat"

    def test_bbox_is_present(self, built_index):
        results = _search(built_index, search_scope="regions", top_k=10)
        assert results[0]["matched_bbox"] is not None
        assert len(results[0]["matched_bbox"]) == 4


# ---------------------------------------------------------------------------
# scope = "both"
# ---------------------------------------------------------------------------


class TestScopeBoth:
    def test_returns_both_parents(self, built_index):
        results = _search(built_index, search_scope="both", top_k=10)
        # metadata_db resolves paths, so match on basename
        basenames = {Path(r["parent_path"]).name for r in results}
        assert "a.jpg" in basenames
        assert "b.jpg" in basenames

    def test_parent_a_scores_higher_than_b(self, built_index):
        results = _search(built_index, search_scope="both", top_k=10)
        score_a = next(r["score"] for r in results if Path(r["parent_path"]).name == "a.jpg")
        score_b = next(r["score"] for r in results if Path(r["parent_path"]).name == "b.jpg")
        assert score_a > score_b


# ---------------------------------------------------------------------------
# top_k limiting
# ---------------------------------------------------------------------------


class TestTopK:
    def test_top_k_1(self, built_index):
        results = _search(built_index, search_scope="both", top_k=1)
        assert len(results) == 1

    def test_ranks_start_at_1(self, built_index):
        results = _search(built_index, search_scope="both", top_k=10)
        assert results[0]["rank"] == 1
        assert results[-1]["rank"] == len(results)


# ---------------------------------------------------------------------------
# Invalid scope
# ---------------------------------------------------------------------------


class TestInvalidScope:
    def test_invalid_scope_raises(self, built_index):
        with pytest.raises(ValueError, match="search_scope"):
            _search(built_index, search_scope="invalid")


def test_search_uses_split_indexes_and_sqlite_faiss_mappings(tmp_path):
    import faiss

    from visionsearch_web_ui.region.search_service import (
        invalidate_region_index_cache,
        search_regions,
    )

    region_path = tmp_path / "regions.faiss"
    whole_path = tmp_path / "whole.faiss"
    db_path = tmp_path / "metadata.sqlite"
    whole_index = faiss.IndexFlatIP(DIM)
    whole_index.add(np.asarray([[0.7] + [0.0] * (DIM - 1)], dtype=np.float32))
    region_index = faiss.IndexFlatIP(DIM)
    region_index.add(np.asarray([[0.95] + [0.0] * (DIM - 1)], dtype=np.float32))
    faiss.write_index(whole_index, str(whole_path))
    faiss.write_index(region_index, str(region_path))

    with RegionMetadataDB(db_path) as db:
        parent_id = db.add_parent_image("/images/dog.jpg", "dog-hash", 400, 300)
        db.add_region_vector(
            0, parent_id, "/images/dog.jpg", None,
            is_whole_image=True, index_type="whole", faiss_id=0,
            file_hash="dog-hash", image_width=400, image_height=300,
        )
        db.add_region_vector(
            1, parent_id, "/images/dog.jpg", SAMPLE_BOX,
            index_type="region", faiss_id=0, file_hash="dog-hash",
            image_width=400, image_height=300,
        )
    invalidate_region_index_cache()

    results = search_regions(
        QUERY_VEC,
        search_scope="regions",
        top_k=10,
        faiss_path=region_path,
        whole_faiss_path=whole_path,
        db_path=db_path,
    )
    assert len(results) == 1
    assert results[0]["class_label"] == "cat"
    assert results[0]["matched_bbox"] == [10.0, 10.0, 100.0, 100.0]
    assert results[0]["score"] == pytest.approx(0.95)


def test_search_filters_region_candidates_by_selected_crop_type(tmp_path, caplog):
    import logging
    import faiss

    from visionsearch_web_ui.region.search_service import (
        invalidate_region_index_cache,
        search_regions,
    )

    region_path = tmp_path / "regions-by-type.faiss"
    whole_path = tmp_path / "whole-by-type.faiss"
    db_path = tmp_path / "metadata-by-type.sqlite"
    whole_index = faiss.IndexFlatIP(DIM)
    whole_index.add(np.asarray([[0.5] + [0.0] * (DIM - 1)], dtype=np.float32))
    region_index = faiss.IndexFlatIP(DIM)
    region_index.add(
        np.asarray(
            [
                [0.99] + [0.0] * (DIM - 1),
                [0.95] + [0.0] * (DIM - 1),
                [0.90] + [0.0] * (DIM - 1),
                [0.97] + [0.0] * (DIM - 1),
            ],
            dtype=np.float32,
        )
    )
    faiss.write_index(whole_index, str(whole_path))
    faiss.write_index(region_index, str(region_path))

    grid_box = RegionBox(0, 0, 100, 100, "grid_0_0", 0.0, "grid")
    first_dog_box = RegionBox(10, 10, 90, 90, "dog", 0.91, "yolo")
    second_dog_box = RegionBox(20, 20, 80, 80, "dog", 0.82, "yolo")
    better_dog_box = RegionBox(15, 15, 85, 85, "dog", 0.94, "yolo")
    with RegionMetadataDB(db_path) as db:
        grid_parent = db.add_parent_image("/images/grid.jpg", "grid_hash")
        dog_parent = db.add_parent_image("/images/dog.jpg", "dog_hash")
        second_dog_parent = db.add_parent_image("/images/dog2.jpg", "dog2_hash")
        db.add_region_vector(
            0, grid_parent, "/images/grid.jpg", grid_box,
            index_type="region", faiss_id=0,
        )
        db.add_region_vector(
            1, dog_parent, "/images/dog.jpg", first_dog_box,
            index_type="region", faiss_id=1,
        )
        db.add_region_vector(
            2, second_dog_parent, "/images/dog2.jpg", second_dog_box,
            index_type="region", faiss_id=2,
        )
        db.add_region_vector(
            3, dog_parent, "/images/dog.jpg", better_dog_box,
            index_type="region", faiss_id=3,
        )
    invalidate_region_index_cache()
    caplog.set_level(logging.INFO, logger="visionsearch_web_ui.region.search_service")

    results = search_regions(
        QUERY_VEC,
        search_scope="regions",
        top_k=2,
        region_crop_type="yolo",
        query_region_id=2,
        query_label="dog",
        query_dimensions=(100, 100),
        request_id="dog-request-1",
        faiss_path=region_path,
        whole_faiss_path=whole_path,
        db_path=db_path,
    )

    assert [Path(result["parent_path"]).name for result in results] == [
        "dog.jpg",
        "dog2.jpg",
    ]
    assert [result["class_label"] for result in results] == ["dog", "dog"]
    assert all(result["match_type"] == "yolo" for result in results)
    assert results[0]["score"] == pytest.approx(0.97)
    assert results[0]["matched_bbox"] == [15, 15, 85, 85]
    assert "query_region_id=2" in caplog.text
    assert "request_id=dog-request-1" in caplog.text
    assert "object_label=dog" in caplog.text
    assert "embedding_dimension=512" in caplog.text
    assert "region_crop_type=yolo" in caplog.text
    assert "region_candidate_crop_types=['yolo']" in caplog.text
    assert "'index_type': 'region'" in caplog.text
    assert "parent_image_id" in caplog.text
    assert "bbox" in caplog.text

    grid_results = search_regions(
        QUERY_VEC,
        search_scope="regions",
        top_k=1,
        region_crop_type="grid",
        faiss_path=region_path,
        whole_faiss_path=whole_path,
        db_path=db_path,
    )
    assert len(grid_results) == 1
    assert grid_results[0]["class_label"] == "grid_0_0"
    assert grid_results[0]["match_type"] == "grid"


def test_both_scope_applies_modality_weights_when_only_one_index_matches(tmp_path):
    import faiss

    from visionsearch_web_ui.region.search_service import (
        invalidate_region_index_cache,
        search_regions,
    )

    region_path = tmp_path / "regions-weighted.faiss"
    whole_path = tmp_path / "whole-weighted.faiss"
    db_path = tmp_path / "metadata-weighted.sqlite"

    # The query is a person-region embedding. A dog whole-image match is
    # stronger than the person region's raw score, but should not beat it
    # after applying the configured 0.5 whole/region weights consistently.
    person_region = np.zeros((1, DIM), dtype=np.float32)
    person_region[0, 0] = 0.9
    person_region[0, 1] = np.sqrt(1.0 - 0.9**2)
    whole_vectors = np.zeros((2, DIM), dtype=np.float32)
    whole_vectors[0, 0] = 0.1
    whole_vectors[0, 1] = np.sqrt(1.0 - 0.1**2)
    whole_vectors[1, 0] = 0.8
    whole_vectors[1, 1] = 0.6
    region_index = faiss.IndexFlatIP(DIM)
    region_index.add(person_region)
    whole_index = faiss.IndexFlatIP(DIM)
    whole_index.add(whole_vectors)
    faiss.write_index(region_index, str(region_path))
    faiss.write_index(whole_index, str(whole_path))

    person_box = RegionBox(10, 10, 90, 90, "person", 0.86, "yolo")
    with RegionMetadataDB(db_path) as db:
        person_id = db.add_parent_image("/images/person.jpg", "person-hash")
        dog_id = db.add_parent_image("/images/dog.jpg", "dog-hash")
        db.add_region_vector(
            0, person_id, "/images/person.jpg", person_box,
            index_type="region", faiss_id=0,
        )
        db.add_region_vector(
            1, person_id, "/images/person.jpg", None,
            is_whole_image=True, index_type="whole", faiss_id=0,
        )
        db.add_region_vector(
            2, dog_id, "/images/dog.jpg", None,
            is_whole_image=True, index_type="whole", faiss_id=1,
        )
    invalidate_region_index_cache()

    results = search_regions(
        QUERY_VEC,
        search_scope="both",
        top_k=2,
        whole_weight=0.5,
        region_crop_type="yolo",
        faiss_path=region_path,
        whole_faiss_path=whole_path,
        db_path=db_path,
    )

    assert [Path(result["parent_path"]).name for result in results] == [
        "person.jpg",
        "dog.jpg",
    ]
    assert results[0]["score"] == pytest.approx(0.5)
    assert results[0]["class_label"] == "person"
    assert results[1]["score"] == pytest.approx(0.4)
    assert results[1]["match_type"] == "whole"


def test_category_aware_both_ranking_is_opt_in_and_does_not_filter(tmp_path, monkeypatch):
    import faiss

    from visionsearch_web_ui.region import config
    from visionsearch_web_ui.region.search_service import (
        invalidate_region_index_cache,
        search_regions,
    )

    region_path = tmp_path / "regions-category-aware.faiss"
    whole_path = tmp_path / "whole-category-aware.faiss"
    db_path = tmp_path / "metadata-category-aware.sqlite"

    query = np.zeros((1, DIM), dtype=np.float32)
    query[0, 0] = 1.0

    region_vectors = np.zeros((2, DIM), dtype=np.float32)
    region_vectors[0, :2] = [0.80, np.sqrt(1.0 - 0.80**2)]
    region_vectors[1, :2] = [0.84, np.sqrt(1.0 - 0.84**2)]
    whole_vectors = np.zeros((2, DIM), dtype=np.float32)
    whole_vectors[:, 0] = 0.5
    whole_vectors[:, 1] = np.sqrt(1.0 - 0.5**2)

    region_index = faiss.IndexFlatIP(DIM)
    region_index.add(region_vectors)
    whole_index = faiss.IndexFlatIP(DIM)
    whole_index.add(whole_vectors)
    faiss.write_index(region_index, str(region_path))
    faiss.write_index(whole_index, str(whole_path))

    with RegionMetadataDB(db_path) as db:
        person_id = db.add_parent_image("/images/person.jpg", "person-hash")
        dog_id = db.add_parent_image("/images/dog.jpg", "dog-hash")
        db.add_region_vector(
            0,
            person_id,
            "/images/person.jpg",
            RegionBox(0, 0, 80, 80, "person", 0.9, "yolo"),
            index_type="region",
            faiss_id=0,
        )
        db.add_region_vector(
            1,
            dog_id,
            "/images/dog.jpg",
            RegionBox(0, 0, 80, 80, "dog", 0.9, "yolo"),
            index_type="region",
            faiss_id=1,
        )
        db.add_region_vector(
            2,
            person_id,
            "/images/person.jpg",
            None,
            is_whole_image=True,
            index_type="whole",
            faiss_id=0,
        )
        db.add_region_vector(
            3,
            dog_id,
            "/images/dog.jpg",
            None,
            is_whole_image=True,
            index_type="whole",
            faiss_id=1,
        )

    monkeypatch.setattr(config, "REGION_CATEGORY_AWARE_ENABLED", False)
    invalidate_region_index_cache()
    baseline = search_regions(
        query,
        search_scope="both",
        top_k=2,
        whole_weight=0.5,
        query_label="a person standing",
        faiss_path=region_path,
        whole_faiss_path=whole_path,
        db_path=db_path,
    )
    assert [Path(result["parent_path"]).name for result in baseline] == [
        "dog.jpg",
        "person.jpg",
    ]

    monkeypatch.setattr(config, "REGION_CATEGORY_AWARE_ENABLED", True)
    invalidate_region_index_cache()
    category_aware = search_regions(
        query,
        search_scope="both",
        top_k=2,
        whole_weight=0.5,
        query_label="a person standing",
        faiss_path=region_path,
        whole_faiss_path=whole_path,
        db_path=db_path,
    )
    assert [Path(result["parent_path"]).name for result in category_aware] == [
        "person.jpg",
        "dog.jpg",
    ]
    assert len(category_aware) == 2
    assert category_aware[0]["class_label"] == "person"


def test_default_search_reports_missing_index_with_build_command(tmp_path):
    from visionsearch_web_ui.region.search_service import search_regions

    with pytest.raises(FileNotFoundError, match="missing_whole.faiss"):
        search_regions(
            QUERY_VEC,
            search_scope="whole",
            whole_faiss_path=tmp_path / "missing_whole.faiss",
        )


def test_index_loader_reloads_when_file_is_replaced_at_same_path(tmp_path):
    import faiss

    from visionsearch_web_ui.region.search_service import (
        invalidate_region_index_cache,
        load_region_index,
        load_whole_image_index,
    )

    region_path = tmp_path / "replace-region.faiss"
    whole_path = tmp_path / "replace-whole.faiss"

    for path in (region_path, whole_path):
        index = faiss.IndexFlatIP(DIM)
        index.add(np.ones((1, DIM), dtype=np.float32))
        faiss.write_index(index, str(path))
    invalidate_region_index_cache()

    assert load_region_index(region_path).ntotal == 1
    assert load_whole_image_index(whole_path).ntotal == 1

    for path in (region_path, whole_path):
        replacement = faiss.IndexFlatIP(DIM)
        replacement.add(np.ones((2, DIM), dtype=np.float32))
        faiss.write_index(replacement, str(path))

    assert load_region_index(region_path).ntotal == 2
    assert load_whole_image_index(whole_path).ntotal == 2
    invalidate_region_index_cache()
