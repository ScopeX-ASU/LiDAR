"""Validated router configuration loading and command-line overrides."""

from __future__ import annotations

from ast import literal_eval
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field

__all__ = ["DetailedRoutingConfig", "RouterConfig", "RunConfig"]


class _ConfigModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
    )


class RunConfig(_ConfigModel):
    random_state: int = 42
    output_layout_gds_path: Path = Path("result/layout.gds")


class DetailedRoutingConfig(_ConfigModel):
    router: Literal["GridRoute"] = "GridRoute"
    max_iteration: int = Field(10, alias="maxIteration", ge=1)
    net_order: Literal["topo", "naive"] = "topo"
    group: bool = True
    enable_45_neighbor: bool = True
    history_cost: int = Field(1000, alias="historyCost", ge=0)
    ripup_times: int = Field(10, ge=0)
    grid_resolution: int = Field(2, gt=0)
    bend_radius: int = Field(5, gt=0)
    cross_section: str = "strip"
    net_bound_scale_factor: float = Field(1.5, gt=0)
    net_default_bound: float = Field(100, gt=0)
    loss_propagation: float = Field(1.5, ge=0)
    loss_bending: float = Field(50, ge=0)
    loss_crossing: float = Field(0, ge=0)
    loss_congestion: float = Field(500, ge=0)


class RouterConfig(_ConfigModel):
    run: RunConfig = Field(default_factory=RunConfig)
    dr: DetailedRoutingConfig = Field(default_factory=DetailedRoutingConfig)
    show_temp: bool = False
    eval: Literal["comp", "noc"] = "comp"
    il_cross: float = Field(0.5, ge=0)

    @classmethod
    def load(cls, path: str | Path, overrides: list[str] | None = None) -> Self:
        """Load one YAML file and apply validated dotted command-line overrides."""
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise TypeError(
                f"Configuration root must be a mapping, got {type(data).__name__}"
            )
        _apply_overrides(data, overrides or [])
        return cls.model_validate(data)

    def __str__(self) -> str:
        return yaml.safe_dump(
            self.model_dump(mode="json", by_alias=True),
            sort_keys=False,
        ).rstrip()


def _apply_overrides(config: dict[str, Any], options: list[str]) -> None:
    index = 0
    while index < len(options):
        option = options[index]
        if not option.startswith("--"):
            raise ValueError(f"Unexpected configuration argument {option!r}")
        option = option[2:]
        if "=" in option:
            key, raw_value = option.split("=", 1)
            index += 1
        else:
            if index + 1 >= len(options):
                raise ValueError(f"Missing value for configuration override --{option}")
            key, raw_value = option, options[index + 1]
            index += 2

        try:
            value = literal_eval(raw_value)
        except (SyntaxError, ValueError):
            value = raw_value

        current = config
        parts = key.split(".")
        for part in parts[:-1]:
            child = current.setdefault(part, {})
            if not isinstance(child, dict):
                raise ValueError(f"Cannot assign nested option --{key}")
            current = child
        current[parts[-1]] = value
