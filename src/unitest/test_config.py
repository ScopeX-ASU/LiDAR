from pathlib import Path

import pytest
from pydantic import ValidationError

from picroute.config.config import RouterConfig


def test_config_loads_absolute_path_and_validated_overrides(tmp_path: Path) -> None:
    path = tmp_path / "router.yml"
    path.write_text("dr:\n  maxIteration: 3\n", encoding="utf-8")

    config = RouterConfig.load(
        path.resolve(),
        overrides=["--dr.maxIteration=7", "--run.output_layout_gds_path", "out.gds"],
    )

    assert config.dr.max_iteration == 7
    assert config.run.output_layout_gds_path == Path("out.gds")


def test_config_rejects_unknown_override(tmp_path: Path) -> None:
    path = tmp_path / "router.yml"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValidationError, match="grid_resoluton"):
        RouterConfig.load(path, overrides=["--dr.grid_resoluton=2"])


def test_config_rejects_invalid_values(tmp_path: Path) -> None:
    path = tmp_path / "router.yml"
    path.write_text("dr:\n  grid_resolution: 0\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="grid_resolution"):
        RouterConfig.load(path)
