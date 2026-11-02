import shutil
from pathlib import Path

import faiss
import numpy as np
from PIL import Image

from visionsearch_web_ui.region.index_manager import (
    build_region_index,
    get_index_stats,
)
from visionsearch_web_ui.region.metadata_db import RegionMetadataDB
from visionsearch_web_ui.region.region_splitter import RegionBox


class FakeClipModel:
    def encode_image(self, image):
        vector = np.zeros((1, 512), dtype=np.float32)
        vector[0, 0] = float(image.size[0])
        vector[0, 1] = float(image.size[1])
        return vector


def test_index_builder_writes_two_indexes_and_metadata(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    image_path = source / "dog.jpg"
    Image.new("RGB", (320, 320), color=(100, 150, 200)).save(image_path)

    boxes = [
        RegionBox(0, 0, 160, 160, "dog", 0.9, "yolo"),
        RegionBox(160, 160, 320, 320, "person", 0.8, "yolo"),
    ]
    monkeypatch.setattr(
        "visionsearch_web_ui.region.index_manager.detect_regions",
        lambda image, **kwargs: boxes,
    )
    region_path = tmp_path / "region.faiss"
    whole_path = tmp_path / "whole.faiss"
    db_path = tmp_path / "metadata.sqlite"

    summary = build_region_index(
        source,
        FakeClipModel(),
        faiss_path=region_path,
        whole_faiss_path=whole_path,
        db_path=db_path,
    )

    assert summary["indexed"] == 1
    assert summary["whole_image_vectors"] == 1
    assert summary["region_vectors"] == 2
    assert faiss.read_index(str(whole_path)).ntotal == 1
    assert faiss.read_index(str(region_path)).ntotal == 2
    with RegionMetadataDB(db_path) as db:
        rows = [db.get_vector_by_faiss_id("whole", 0)]
        rows.extend(db.get_vector_by_faiss_id("region", i) for i in range(2))
        assert all(row is not None for row in rows)
        assert [row["class_label"] for row in rows] == ["whole", "dog", "person"]
        assert {row["parent_path"] for row in rows} == {str(image_path.resolve())}
        assert all(row["file_hash"] and row["image_width"] == 320 for row in rows)
        assert rows[1]["x1"] == 0
        assert rows[1]["confidence"] == 0.9
        assert rows[1]["crop_type"] == "yolo"

    stats = get_index_stats(region_path, whole_path, db_path)
    assert stats["status"] == "ready"
    assert stats["database_status"] == "ok"
    assert stats["faiss_vectors_match_metadata"] is True
    assert stats["whole_image_vectors"] == stats["whole_faiss_vectors"] == 1
    assert stats["region_faiss_vectors"] == stats["region_vectors"] == 2


def test_incremental_builder_skips_duplicate_content(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    image_path = source / "dog.jpg"
    duplicate_path = source / "dog-copy.jpg"
    Image.new("RGB", (320, 320), color=(20, 30, 40)).save(image_path)
    shutil.copyfile(image_path, duplicate_path)
    monkeypatch.setattr(
        "visionsearch_web_ui.region.index_manager.detect_regions",
        lambda image, **kwargs: [],
    )

    paths = {
        "faiss_path": tmp_path / "region.faiss",
        "whole_faiss_path": tmp_path / "whole.faiss",
        "db_path": tmp_path / "metadata.sqlite",
    }
    initial = build_region_index(
        source, FakeClipModel(), split_mode="yolo", **paths
    )
    repeated = build_region_index(
        source, FakeClipModel(), split_mode="yolo", **paths
    )

    assert initial["indexed"] == 1
    assert initial["skipped"] == 1
    assert repeated["indexed"] == 0
    assert repeated["skipped"] == 2
    assert faiss.read_index(str(paths["whole_faiss_path"])).ntotal == 1
    assert faiss.read_index(str(paths["faiss_path"])).ntotal == 0

