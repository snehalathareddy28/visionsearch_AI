import pytest
from PIL import Image

from region_retrieval.region_splitter import (
    RegionBox,
    clamp_bbox,
    crop_region,
    grid_regions,
    is_valid_bbox,
    pad_bbox,
)


def test_clamp_bbox_limits_coordinates_to_image():
    assert clamp_bbox(-4, 10, 120, 80, 100, 60) == (0.0, 10.0, 100.0, 60.0)


def test_clamp_bbox_rejects_invalid_image_dimensions():
    with pytest.raises(ValueError, match="positive integers"):
        clamp_bbox(0, 0, 10, 10, 0, 10)


def test_grid_has_default_tile_count_and_unique_regions():
    regions = grid_regions(900, 600)
    assert len(regions) == 9
    assert len({(r.x1, r.y1, r.x2, r.y2) for r in regions}) == 9


def test_grid_overlap_is_applied_between_adjacent_columns():
    regions = grid_regions(900, 600, rows=1, cols=3, overlap=0.2)
    assert len(regions) == 3
    assert regions[0].x2 > regions[1].x1
    assert regions[1].x2 > regions[2].x1
    assert regions[-1].x2 == pytest.approx(900)


def test_grid_rejects_non_positive_image_dimensions():
    with pytest.raises(ValueError, match="positive integers"):
        grid_regions(0, 300)


def test_grid_skips_tiles_smaller_than_minimum():
    assert grid_regions(90, 90, min_crop_px=64) == []


def test_padding_and_validity():
    padded = pad_bbox(20, 20, 80, 80, 100, 100, padding_fraction=0.05)
    assert padded == pytest.approx((17, 17, 83, 83))
    assert is_valid_bbox(*padded, min_crop_px=64)
    assert not is_valid_bbox(10, 10, 5, 5, min_crop_px=1)


def test_crop_region_clamps_box_and_returns_rgb():
    image = Image.new("RGBA", (100, 100), color=(1, 2, 3, 255))
    region = RegionBox(-5, 10, 20, 40, "test", 0.9, "yolo")
    crop = crop_region(image, region)
    assert crop.mode == "RGB"
    assert crop.size == (20, 30)


def test_crop_region_rejects_empty_box():
    image = Image.new("RGB", (100, 100))
    region = RegionBox(20, 20, 20, 40, "test", 0.9, "yolo")
    with pytest.raises(ValueError, match="empty"):
        crop_region(image, region)
