import io

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from visionsearch_web_ui.region import api_regions, config


def test_region_api_root_matches_region_index_root():
    assert api_regions._ROOT == config.INDEX_DIR.parent


def test_region_result_image_url_is_relative_to_project_root():
    image_path = config.INDEX_DIR.parent / "dataset" / "example.jpg"

    result = api_regions._to_region_results(
        [
            {
                "parent_path": str(image_path),
                "score": 1.0,
                "matched_bbox": None,
                "class_label": "whole",
                "match_type": "whole",
                "confidence": 0.0,
                "rank": 1,
            }
        ]
    )

    assert result[0].image_url == "/image/dataset/example.jpg"
    assert result[0].name == "example.jpg"
    assert not hasattr(result[0], "parent_path")


def _encoded_test_image():
    buffer = io.BytesIO()
    Image.new("RGB", (640, 480), color="red").save(buffer, format="JPEG")
    return buffer.getvalue()


def test_image_search_embeds_exact_selected_thumbnail_bbox(monkeypatch, caplog):
    import logging

    class FakeClipModel:
        image_size = None

        def encode_image(self, image):
            self.image_size = image.size
            return np.ones((1, 512), dtype=np.float32)

    model = FakeClipModel()
    search_args = {}

    def fake_search(embedding, **kwargs):
        search_args.update(kwargs)
        assert embedding.shape == (1, 512)
        return []

    monkeypatch.setattr(api_regions, "_get_model", lambda: model)
    monkeypatch.setattr(api_regions, "search_regions", fake_search)
    caplog.set_level(logging.INFO, logger="visionsearch_web_ui.region.api_regions")

    app = FastAPI()
    app.include_router(api_regions.region_router)
    with TestClient(app) as client:
        response = client.post(
            "/api/region/search/image",
            files={"file": ("sample.jpg", _encoded_test_image(), "image/jpeg")},
            data={
                "region_id": "2",
                "bbox_x1": "20",
                "bbox_y1": "30",
                "bbox_x2": "120",
                "bbox_y2": "230",
                "query_label": "dog",
                "query_crop_type": "yolo",
                "split_mode": "yolo",
                "search_scope": "regions",
                "request_id": "test-dog-search",
            },
        )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "test-dog-search"
    assert model.image_size == (100, 200)
    assert search_args["query_region_id"] == 2
    assert search_args["query_label"] == "dog"
    assert search_args["query_dimensions"] == (100, 200)
    assert search_args["region_crop_type"] == "yolo"
    assert search_args["request_id"] == "test-dog-search"
    assert "region_id=2 object_label=dog crop_dimensions=100x200" in caplog.text
    assert "bbox=[20.0, 30.0, 120.0, 230.0]" in caplog.text
    assert "embedding_dimension=512 embedding_norm=22.627416" in caplog.text


def test_image_search_uses_fresh_selected_crop_for_each_request(monkeypatch):
    class FakeClipModel:
        image_sizes = []

        def encode_image(self, image):
            self.image_sizes.append(image.size)
            embedding = np.zeros((1, 512), dtype=np.float32)
            embedding[0, len(self.image_sizes) - 1] = 1.0
            return embedding

    model = FakeClipModel()
    searched_embeddings = []
    search_metadata = []

    def fake_search(embedding, **kwargs):
        searched_embeddings.append(np.asarray(embedding).copy())
        search_metadata.append(kwargs)
        return []

    monkeypatch.setattr(api_regions, "_get_model", lambda: model)
    monkeypatch.setattr(api_regions, "search_regions", fake_search)

    app = FastAPI()
    app.include_router(api_regions.region_router)
    with TestClient(app) as client:
        dog_response = client.post(
            "/api/region/search/image",
            files={"file": ("sample.jpg", _encoded_test_image(), "image/jpeg")},
            data={
                "region_id": "2",
                "bbox_x1": "20",
                "bbox_y1": "30",
                "bbox_x2": "120",
                "bbox_y2": "230",
                "query_label": "dog",
                "query_crop_type": "yolo",
                "split_mode": "yolo",
                "search_scope": "regions",
                "request_id": "dog-request-1",
            },
        )
        person_response = client.post(
            "/api/region/search/image",
            files={"file": ("sample.jpg", _encoded_test_image(), "image/jpeg")},
            data={
                "region_id": "0",
                "bbox_x1": "100",
                "bbox_y1": "100",
                "bbox_x2": "300",
                "bbox_y2": "200",
                "query_label": "person",
                "query_crop_type": "yolo",
                "split_mode": "yolo",
                "search_scope": "regions",
                "request_id": "person-request-2",
            },
        )

    assert dog_response.status_code == person_response.status_code == 200
    assert dog_response.headers["X-Request-ID"] == "dog-request-1"
    assert person_response.headers["X-Request-ID"] == "person-request-2"
    assert model.image_sizes == [(100, 200), (200, 100)]
    assert len(searched_embeddings) == 2
    assert not np.array_equal(searched_embeddings[0], searched_embeddings[1])
    assert [item["query_label"] for item in search_metadata] == ["dog", "person"]
    assert [item["request_id"] for item in search_metadata] == [
        "dog-request-1",
        "person-request-2",
    ]


def test_text_search_passes_selected_split_mode_to_retrieval(monkeypatch):
    class FakeClipModel:
        def encode_text(self, text):
            assert text == "a dog sitting on grass"
            return np.ones((1, 512), dtype=np.float32)

    search_args = {}

    def fake_search(embedding, **kwargs):
        search_args.update(kwargs)
        return []

    monkeypatch.setattr(api_regions, "_get_model", lambda: FakeClipModel())
    monkeypatch.setattr(api_regions, "search_regions", fake_search)

    app = FastAPI()
    app.include_router(api_regions.region_router)
    with TestClient(app) as client:
        response = client.post(
            "/api/region/search/text",
            json={
                "query": "a dog sitting on grass",
                "search_scope": "regions",
                "split_mode": "yolo",
                "request_id": "text-request-1",
            },
        )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "text-request-1"
    assert search_args["region_crop_type"] == "yolo"
    assert search_args["query_label"] == "a dog sitting on grass"
    assert search_args["request_id"] == "text-request-1"
