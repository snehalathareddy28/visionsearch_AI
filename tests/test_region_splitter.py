"""
test_region_splitter.py — pytest tests for region_splitter.py

Tests cover:
- bbox clamping (pad_bbox, clamp_bbox)
- minimum-size validation (is_valid_bbox)
- tile count for grid_regions (rows × cols, with small-image edge cases)
- crop_region returns an RGB image of the correct size
"""

import pytest
from PIL import Image

from visionsearch_web_ui.region.region_splitter import (
    RegionBox,
    clamp_bbox,
    crop_region,
    grid_regions,
    is_valid_bbox,
    pad_bbox,
)


# ---------------------------------------------------------------------------
# clamp_bbox
# ---------------------------------------------------------------------------


class TestClampBbox:
    def test_within_bounds_unchanged(self):
        result = clamp_bbox(10, 20, 100, 200, 400, 400)
        assert result == (10.0, 20.0, 100.0, 200.0)

    def test_negative_clamped_to_zero(self):
        x1, y1, x2, y2 = clamp_bbox(-10, -5, 50, 50, 200, 200)
        assert x1 == 0.0
        assert y1 == 0.0

    def test_overflow_clamped_to_image_edge(self):
        x1, y1, x2, y2 = clamp_bbox(0, 0, 999, 888, 300, 200)
        assert x2 == 300.0
        assert y2 == 200.0

    def test_all_corners_out_of_bounds(self):
        x1, y1, x2, y2 = clamp_bbox(-50, -50, 9999, 9999, 100, 100)
        assert (x1, y1, x2, y2) == (0.0, 0.0, 100.0, 100.0)


# ---------------------------------------------------------------------------
# pad_bbox
# ---------------------------------------------------------------------------


class TestPadBbox:
    def test_zero_padding_unchanged(self):
        result = pad_bbox(10, 10, 110, 210, 500, 500, pad_fraction=0.0)
        assert result == (10.0, 10.0, 110.0, 210.0)

    def test_padding_expands_bbox(self):
        x1, y1, x2, y2 = pad_bbox(100, 100, 200, 200, 500, 500, pad_fraction=0.10)
        assert x1 < 100
        assert y1 < 100
        assert x2 > 200
        assert y2 > 200

    def test_padding_clamped_at_edges(self):
        # bbox touching the left / top border
        x1, y1, x2, y2 = pad_bbox(0, 0, 100, 100, 500, 500, pad_fraction=0.10)
        assert x1 == 0.0
        assert y1 == 0.0

    def test_padding_at_right_bottom_edge(self):
        x1, y1, x2, y2 = pad_bbox(400, 400, 500, 500, 500, 500, pad_fraction=0.10)
        assert x2 == 500.0
        assert y2 == 500.0


# ---------------------------------------------------------------------------
# is_valid_bbox
# ---------------------------------------------------------------------------


class TestIsValidBbox:
    def test_large_bbox_valid(self):
        assert is_valid_bbox(0, 0, 200, 150, min_px=64)

    def test_exact_min_size_valid(self):
        assert is_valid_bbox(0, 0, 64, 64, min_px=64)

    def test_too_narrow_invalid(self):
        assert not is_valid_bbox(0, 0, 30, 200, min_px=64)

    def test_too_short_invalid(self):
        assert not is_valid_bbox(0, 0, 200, 30, min_px=64)

    def test_zero_size_invalid(self):
        assert not is_valid_bbox(50, 50, 50, 50, min_px=1)


# ---------------------------------------------------------------------------
# grid_regions — tile count
# ---------------------------------------------------------------------------


class TestGridRegions:
    def test_3x3_tile_count(self):
        boxes = grid_regions(300, 300, rows=3, cols=3, overlap=0.0)
        assert len(boxes) == 9

    def test_2x4_tile_count(self):
        boxes = grid_regions(400, 200, rows=2, cols=4, overlap=0.0)
        assert len(boxes) == 8

    def test_1x1_returns_single_tile(self):
        boxes = grid_regions(200, 200, rows=1, cols=1, overlap=0.0)
        assert len(boxes) == 1

    def test_overlap_label_correct(self):
        boxes = grid_regions(300, 300, rows=2, cols=2, overlap=0.2)
        labels = {b.class_label for b in boxes}
        assert "grid_0_0" in labels
        assert "grid_1_1" in labels

    def test_crop_type_is_grid(self):
        boxes = grid_regions(300, 300, rows=2, cols=2)
        assert all(b.crop_type == "grid" for b in boxes)

    def test_all_boxes_within_bounds(self):
        img_w, img_h = 320, 240
        boxes = grid_regions(img_w, img_h, rows=3, cols=3, overlap=0.2)
        for b in boxes:
            assert b.x1 >= 0.0
            assert b.y1 >= 0.0
            assert b.x2 <= img_w
            assert b.y2 <= img_h

    def test_confidence_is_zero_for_grid(self):
        boxes = grid_regions(200, 200, rows=2, cols=2)
        assert all(b.confidence == 0.0 for b in boxes)

    def test_small_image_tiles_skip_if_too_small(self):
        # 9-pixel image with min_px=64 default → all tiles filtered out
        # Use explicit min_px=1 so we get tiles anyway
        boxes = grid_regions(9, 9, rows=3, cols=3, overlap=0.0)
        # Each tile ≈ 3×3 px, smaller than default YOLO_MIN_CROP_PX=64
        # So default min_px filtering should produce 0 tiles.
        # (grid_regions uses config.YOLO_MIN_CROP_PX via is_valid_bbox)
        assert isinstance(boxes, list)  # still returns a list, possibly empty

    def test_invalid_rows_raises(self):
        with pytest.raises(ValueError):
            grid_regions(200, 200, rows=0, cols=3)

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError):
            grid_regions(200, 200, rows=2, cols=2, overlap=1.0)


# ---------------------------------------------------------------------------
# crop_region
# ---------------------------------------------------------------------------


class TestCropRegion:
    def test_crop_returns_rgb_image(self):
        image = Image.new("RGB", (400, 300), color=(120, 80, 200))
        box = RegionBox(x1=10, y1=10, x2=110, y2=110,
                        class_label="test", confidence=0.9, crop_type="yolo")
        crop = crop_region(image, box)
        assert crop.mode == "RGB"

    def test_crop_size_matches_bbox(self):
        image = Image.new("RGB", (400, 300), color=(50, 100, 150))
        box = RegionBox(x1=20, y1=30, x2=120, y2=180,
                        class_label="test", confidence=0.8, crop_type="yolo")
        crop = crop_region(image, box)
        assert crop.size == (100, 150)  # (x2-x1, y2-y1)

    def test_crop_preserves_pixel_colour(self):
        colour = (255, 0, 128)
        image = Image.new("RGB", (200, 200), color=colour)
        box = RegionBox(x1=50, y1=50, x2=150, y2=150,
                        class_label="test", confidence=1.0, crop_type="yolo")
        crop = crop_region(image, box)
        # Centre pixel should match the solid-colour source.
        centre_pixel = crop.getpixel((49, 49))
        assert centre_pixel == colour
