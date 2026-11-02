"""
test_metadata_db.py — pytest tests for metadata_db.py

Tests cover:
- Parent image insert / dedup
- Region vector insert and retrieval
- Whole-image vector insert
- get_indexed_hashes dedup key
- stats() output shape
"""

import tempfile
from pathlib import Path

import pytest

from visionsearch_web_ui.region.metadata_db import RegionMetadataDB
from visionsearch_web_ui.region.region_splitter import RegionBox


def _make_db(tmp_path: Path) -> RegionMetadataDB:
    return RegionMetadataDB(tmp_path / "test_meta.db")


SAMPLE_BOX = RegionBox(
    x1=10.0, y1=20.0, x2=110.0, y2=120.0,
    class_label="dog", confidence=0.87, crop_type="yolo",
)


# ---------------------------------------------------------------------------
# Parent images
# ---------------------------------------------------------------------------


class TestParentImages:
    def test_add_parent_image_returns_id(self, tmp_path):
        with _make_db(tmp_path) as db:
            img_id = db.add_parent_image("/images/cat.jpg", "abc123")
            assert isinstance(img_id, int)
            assert img_id >= 1

    def test_duplicate_path_returns_same_id(self, tmp_path):
        with _make_db(tmp_path) as db:
            id1 = db.add_parent_image("/images/cat.jpg", "abc123")
            id2 = db.add_parent_image("/images/cat.jpg", "abc123")
            assert id1 == id2

    def test_different_paths_get_different_ids(self, tmp_path):
        with _make_db(tmp_path) as db:
            id1 = db.add_parent_image("/images/cat.jpg", "aaa")
            id2 = db.add_parent_image("/images/dog.jpg", "bbb")
            assert id1 != id2

    def test_get_indexed_hashes_returns_set(self, tmp_path):
        with _make_db(tmp_path) as db:
            db.add_parent_image("/images/a.jpg", "hash_a")
            db.add_parent_image("/images/b.jpg", "hash_b")
            hashes = db.get_indexed_hashes()
            assert "hash_a" in hashes
            assert "hash_b" in hashes

    def test_empty_db_has_no_hashes(self, tmp_path):
        with _make_db(tmp_path) as db:
            assert db.get_indexed_hashes() == set()

    def test_get_yolo_labels_excludes_grid_and_whole_rows(self, tmp_path):
        with _make_db(tmp_path) as db:
            parent_id = db.add_parent_image("/images/cat.jpg", "abc123")
            db.add_region_vector(
                0,
                parent_id,
                "/images/cat.jpg",
                None,
                is_whole_image=True,
                index_type="whole",
                faiss_id=0,
            )
            db.add_region_vector(
                1,
                parent_id,
                "/images/cat.jpg",
                SAMPLE_BOX,
                index_type="region",
                faiss_id=0,
            )
            grid_box = RegionBox(0, 0, 100, 100, "grid_0_0", 0.0, "grid")
            db.add_region_vector(
                2,
                parent_id,
                "/images/cat.jpg",
                grid_box,
                index_type="region",
                faiss_id=1,
            )

            assert db.get_yolo_labels() == {"dog"}

    def test_get_parent_by_path_not_found(self, tmp_path):
        with _make_db(tmp_path) as db:
            result = db.get_parent_by_path("/nonexistent.jpg")
            assert result is None


# ---------------------------------------------------------------------------
# Region vectors
# ---------------------------------------------------------------------------


class TestRegionVectors:
    def _setup(self, tmp_path) -> tuple[RegionMetadataDB, int]:
        db = _make_db(tmp_path)
        db.open()
        parent_id = db.add_parent_image("/images/cat.jpg", "abc123")
        return db, parent_id

    def test_add_and_retrieve_region_vector(self, tmp_path):
        with _make_db(tmp_path) as db:
            pid = db.add_parent_image("/images/cat.jpg", "abc123")
            db.add_region_vector(0, pid, "/images/cat.jpg", SAMPLE_BOX)
            row = db.get_vector(0)
            assert row is not None
            assert row["class_label"] == "dog"
            assert row["crop_type"] == "yolo"
            assert abs(row["confidence"] - 0.87) < 1e-5

    def test_bbox_stored_correctly(self, tmp_path):
        with _make_db(tmp_path) as db:
            pid = db.add_parent_image("/images/cat.jpg", "abc123")
            db.add_region_vector(0, pid, "/images/cat.jpg", SAMPLE_BOX)
            row = db.get_vector(0)
            assert abs(row["x1"] - 10.0) < 1e-5
            assert abs(row["y2"] - 120.0) < 1e-5

    def test_whole_image_vector_stored(self, tmp_path):
        with _make_db(tmp_path) as db:
            pid = db.add_parent_image("/images/cat.jpg", "abc123", 640, 480)
            db.add_region_vector(
                0,
                pid,
                "/images/cat.jpg",
                None,
                is_whole_image=True,
                faiss_id=0,
                file_hash="abc123",
                image_width=640,
                image_height=480,
            )
            row = db.get_vector(0)
            assert row["is_whole_image"] == 1
            assert row["class_label"] == "whole"
            assert row["index_type"] == "whole"
            assert row["faiss_id"] == 0
            assert row["file_hash"] == "abc123"
            assert row["image_width"] == 640
            assert row["image_height"] == 480
            assert row["created_at"]
            assert row["x1"] is None

    def test_separate_indexes_have_stable_faiss_id_mapping(self, tmp_path):
        with _make_db(tmp_path) as db:
            pid = db.add_parent_image("/images/dog.jpg", "dog-hash", 800, 600)
            whole_id = db.get_next_vector_id()
            db.add_region_vector(
                whole_id, pid, "/images/dog.jpg", None, is_whole_image=True,
                index_type="whole", faiss_id=0, file_hash="dog-hash",
                image_width=800, image_height=600,
            )
            region_id = db.get_next_vector_id()
            db.add_region_vector(
                region_id, pid, "/images/dog.jpg", SAMPLE_BOX,
                index_type="region", faiss_id=0, file_hash="dog-hash",
                image_width=800, image_height=600,
            )
            assert whole_id != region_id
            assert db.get_vector_by_faiss_id("whole", 0)["vector_id"] == whole_id
            assert db.get_vector_by_faiss_id("region", 0)["vector_id"] == region_id
            assert db.get_vector_counts() == {"whole": 1, "region": 1}

    def test_get_vector_nonexistent_returns_none(self, tmp_path):
        with _make_db(tmp_path) as db:
            assert db.get_vector(9999) is None

    def test_get_vectors_for_parent(self, tmp_path):
        with _make_db(tmp_path) as db:
            pid = db.add_parent_image("/images/cat.jpg", "abc123")
            db.add_region_vector(0, pid, "/images/cat.jpg", None, is_whole_image=True)
            db.add_region_vector(1, pid, "/images/cat.jpg", SAMPLE_BOX)
            rows = db.get_vectors_for_parent(pid)
            assert len(rows) == 2

    def test_multiple_images_isolated(self, tmp_path):
        with _make_db(tmp_path) as db:
            pid1 = db.add_parent_image("/images/a.jpg", "aaa")
            pid2 = db.add_parent_image("/images/b.jpg", "bbb")
            db.add_region_vector(0, pid1, "/images/a.jpg", SAMPLE_BOX)
            db.add_region_vector(1, pid2, "/images/b.jpg", SAMPLE_BOX)
            rows1 = db.get_vectors_for_parent(pid1)
            rows2 = db.get_vectors_for_parent(pid2)
            assert len(rows1) == 1
            assert len(rows2) == 1


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_empty_stats(self, tmp_path):
        with _make_db(tmp_path) as db:
            s = db.stats()
            assert s["indexed_images"] == 0
            assert s["total_vectors"] == 0

    def test_stats_after_insert(self, tmp_path):
        with _make_db(tmp_path) as db:
            pid = db.add_parent_image("/images/cat.jpg", "abc")
            db.add_region_vector(0, pid, "/images/cat.jpg", None, is_whole_image=True)
            db.add_region_vector(1, pid, "/images/cat.jpg", SAMPLE_BOX)
            s = db.stats()
            assert s["indexed_images"] == 1
            assert s["total_vectors"] == 2
            assert s["whole_image_vectors"] == 1
            assert s["yolo_region_vectors"] == 1
            assert s["grid_region_vectors"] == 0


# ---------------------------------------------------------------------------
# Context manager behaviour
# ---------------------------------------------------------------------------


class TestContextManager:
    def test_open_close_without_with(self, tmp_path):
        db = _make_db(tmp_path)
        db.open()
        pid = db.add_parent_image("/img.jpg", "hashx")
        db.close()
        # Re-open and verify persistence
        db.open()
        row = db.get_parent_by_path("/img.jpg")
        db.close()
        assert row is not None

    def test_conn_raises_without_open(self, tmp_path):
        db = _make_db(tmp_path)
        with pytest.raises(RuntimeError, match="not open"):
            _ = db.conn
