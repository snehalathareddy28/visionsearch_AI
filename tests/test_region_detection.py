import io
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from visionsearch_web_ui.region import api_regions
from visionsearch_web_ui.region.detector import (
    detect_regions,
    detect_regions_with_status,
)


class FakeTensor(list):
    def cpu(self):
        return self

    def tolist(self):
        return list(self)


class FakeYolo:
    def __init__(self):
        self.calls = []

    def __call__(self, image, **kwargs):
        self.calls.append(kwargs)
        boxes = SimpleNamespace(
            xyxy=FakeTensor([[20, 20, 220, 300], [300, 30, 550, 400]]),
            conf=FakeTensor([0.94, 0.88]),
            cls=FakeTensor([0, 16]),
        )
        return [SimpleNamespace(boxes=boxes, names={0: "person", 16: "dog"})]


def test_yolo_mode_calls_model_and_returns_named_detections(caplog, monkeypatch):
    model = FakeYolo()
    from visionsearch_web_ui.region import detector

    monkeypatch.setattr(detector, "get_yolo", lambda: model)
    with caplog.at_level("INFO", logger="visionsearch_web_ui.region.detector"):
        regions = detect_regions(Image.new("RGB", (640, 480)), split_mode="yolo")

    assert model.calls == [{"conf": 0.35, "verbose": False}]
    assert [(region.class_label, region.confidence) for region in regions] == [
        ("person", 0.94),
        ("dog", 0.88),
    ]
    assert all(region.crop_type == "yolo" for region in regions)
    assert "class=person confidence=0.9400 bbox_xyxy=" in caplog.text
    assert "class=dog confidence=0.8800 bbox_xyxy=" in caplog.text


def test_strict_yolo_mode_propagates_model_failure_without_grid(monkeypatch):
    from visionsearch_web_ui.region import detector

    def fail(*args, **kwargs):
        raise RuntimeError("weights could not load")

    monkeypatch.setattr(detector, "_run_yolo", fail)
    try:
        detect_regions_with_status(
            Image.new("RGB", (300, 300)),
            split_mode="yolo",
        )
    except RuntimeError as error:
        assert "weights could not load" in str(error)
    else:
        raise AssertionError("strict YOLO mode must not replace failures with grid tiles")


def test_auto_mode_reports_grid_fallback_reason(monkeypatch):
    from visionsearch_web_ui.region import detector

    def fail(*args, **kwargs):
        raise RuntimeError("ultralytics is missing")

    monkeypatch.setattr(detector, "_run_yolo", fail)
    outcome = detect_regions_with_status(
        Image.new("RGB", (300, 300)),
        split_mode="auto",
    )
    assert outcome.used_mode == "grid"
    assert outcome.fallback_reason
    assert all(region.crop_type == "grid" for region in outcome.regions)


def test_yolo_zero_detections_does_not_substitute_grid(monkeypatch):
    from visionsearch_web_ui.region import detector

    monkeypatch.setattr(detector, "_run_yolo", lambda *args, **kwargs: [])
    outcome = detect_regions_with_status(
        Image.new("RGB", (300, 300)),
        split_mode="yolo",
    )
    assert outcome.used_mode == "yolo"
    assert outcome.regions == []
    assert "no valid objects" in outcome.fallback_reason


def _image_bytes():
    image = Image.new("RGB", (640, 480), color=(80, 120, 160))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_detect_endpoint_forwards_yolo_mode_and_returns_detection_type(monkeypatch):
    received_modes = []

    def fake_detect(image, *, split_mode):
        received_modes.append(split_mode)
        return {
            "regions": [
                {
                    "region_id": 0,
                    "bbox": [1, 2, 80, 90],
                    "class_label": "dog",
                    "confidence": 0.91,
                    "crop_type": "yolo",
                    "thumbnail_b64": "encoded",
                }
            ],
            "requested_mode": split_mode,
            "used_mode": "yolo",
            "fallback": False,
            "message": None,
        }

    monkeypatch.setattr(api_regions, "detect_query_regions", fake_detect)
    app = FastAPI()
    app.include_router(api_regions.region_router)
    with TestClient(app) as client:
        response = client.post(
            "/api/region/search/image/detect",
            files={"file": ("sample.jpg", _image_bytes(), "image/jpeg")},
            data={"split_mode": "yolo"},
        )

    assert response.status_code == 200
    assert received_modes == ["yolo"]
    assert response.json()["regions"][0]["crop_type"] == "yolo"
    assert response.json()["used_mode"] == "yolo"
    assert response.json()["fallback"] is False


def test_detect_endpoint_runs_detector_for_uploaded_image(monkeypatch):
    from visionsearch_web_ui.region import detector

    model = FakeYolo()
    monkeypatch.setattr(detector, "get_yolo", lambda: model)
    app = FastAPI()
    app.include_router(api_regions.region_router)
    with TestClient(app) as client:
        response = client.post(
            "/api/region/search/image/detect",
            files={"file": ("sample.jpg", _image_bytes(), "image/jpeg")},
            data={"split_mode": "yolo"},
        )

    assert response.status_code == 200
    result = response.json()
    assert result["requested_mode"] == "yolo"
    assert result["used_mode"] == "yolo"
    assert result["fallback"] is False
    assert [region["class_label"] for region in result["regions"]] == ["person", "dog"]
    assert model.calls == [{"conf": 0.35, "verbose": False}]


def test_detect_endpoint_returns_clear_error_when_yolo_fails(monkeypatch):
    def fail_detect(image, *, split_mode):
        raise RuntimeError("ultralytics is not installed")

    monkeypatch.setattr(api_regions, "detect_query_regions", fail_detect)
    app = FastAPI()
    app.include_router(api_regions.region_router)
    with TestClient(app) as client:
        response = client.post(
            "/api/region/search/image/detect",
            files={"file": ("sample.jpg", _image_bytes(), "image/jpeg")},
            data={"split_mode": "yolo"},
        )

    assert response.status_code == 503
    assert "YOLO detection is unavailable" in response.json()["detail"]
    assert "ultralytics is not installed" in response.json()["detail"]
