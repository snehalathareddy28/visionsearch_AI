from pathlib import Path

import pytest

from region_retrieval.config import RegionSettings


def test_settings_load_defaults_and_custom_paths():
    settings = RegionSettings.from_env(
        {
            "REGION_RETRIEVAL_YOLO_MODEL": "custom.pt",
            "REGION_RETRIEVAL_GRID_ROWS": "2",
            "REGION_RETRIEVAL_INDEX_DIR": "my-index",
        }
    )
    assert settings.yolo_model == "custom.pt"
    assert settings.grid_rows == 2
    assert settings.index_dir == (Path(__file__).parents[1] / "my-index").resolve()


def test_settings_reject_invalid_overlap():
    with pytest.raises(ValueError, match="grid_overlap"):
        RegionSettings(grid_overlap=1.0)
