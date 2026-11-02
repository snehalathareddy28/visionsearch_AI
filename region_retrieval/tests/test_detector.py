from types import SimpleNamespace

from PIL import Image

from region_retrieval.config import RegionSettings
from region_retrieval.detector import detect_regions


class FakeTensor(list):
    def cpu(self):
        return self

    def tolist(self):
        return list(self)


class FakeModel:
    def __init__(self, boxes):
        self.boxes = boxes

    def __call__(self, image, **kwargs):
        assert kwargs["verbose"] is False
        return [SimpleNamespace(boxes=self.boxes, names={0: "cat", 1: "dog"})]


def test_yolo_mode_filters_small_regions_and_keeps_confidence_order():
    boxes = SimpleNamespace(
        xyxy=FakeTensor(
            [
                [10, 10, 90, 90],
                [20, 20, 30, 30],
                [100, 100, 180, 180],
            ]
        ),
        conf=FakeTensor([0.75, 0.99, 0.91]),
        cls=FakeTensor([0, 1, 1]),
    )
    settings = RegionSettings(max_detections=1)
    regions = detect_regions(
        Image.new("RGB", (200, 200)),
        split_mode="yolo",
        model=FakeModel(boxes),
        settings=settings,
    )
    assert len(regions) == 1
    assert regions[0].class_label == "dog"
    assert regions[0].confidence == 0.91
    assert regions[0].crop_type == "yolo"


def test_auto_mode_uses_grid_if_yolo_returns_no_regions():
    empty_boxes = SimpleNamespace(
        xyxy=FakeTensor([]),
        conf=FakeTensor([]),
        cls=FakeTensor([]),
    )
    regions = detect_regions(
        Image.new("RGB", (300, 300)),
        split_mode="auto",
        model=FakeModel(empty_boxes),
    )
    assert len(regions) == 9
    assert all(region.crop_type == "grid" for region in regions)


def test_grid_mode_does_not_load_or_call_yolo():
    regions = detect_regions(
        Image.new("RGB", (300, 300)),
        split_mode="grid",
    )
    assert len(regions) == 9


def test_invalid_split_mode_is_rejected():
    import pytest

    with pytest.raises(ValueError, match="split_mode"):
        detect_regions(Image.new("RGB", (300, 300)), split_mode="unknown")
