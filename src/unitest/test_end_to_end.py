from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import gdsfactory as gf
import pytest
import yaml

from picroute.config.config import RouterConfig
from picroute.database.schematic import CustomSchematic
from picroute.main.picroute import PICRoute


def test_run_creates_output_directory(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "layout.gds"
    layout = SimpleNamespace(
        write_gds=Mock(side_effect=lambda path: Path(path).touch())
    )
    router = PICRoute.__new__(PICRoute)
    router.config = SimpleNamespace(run=SimpleNamespace(output_layout_gds_path=output))
    router.dr = SimpleNamespace(solve=Mock(return_value=True))
    router.cirdb = SimpleNamespace(layout=layout)

    router.run()

    assert output.is_file()
    layout.write_gds.assert_called_once_with(output)


def test_short_facing_ports_use_prebuilt_route(tmp_path: Path) -> None:
    benchmark = tmp_path / "short.yml"
    source = CustomSchematic(benchmark)
    source.update_settings(design="short", die_area=[[0, 0], [100, 100]])
    straight = gf.components.straight(length=10)
    source.add_instance("left", straight)
    source.add_instance("right", straight)
    source.update_placement("left", ["FIXED", [20, 20], "N", [0, 0, 0, 0]])
    source.update_placement("right", ["FIXED", [35, 20], "N", [0, 0, 0, 0]])
    source.add_net("left", "o2", "right", "o1")
    source.commit()

    config = RouterConfig()
    config.dr.group = False
    config.run.output_layout_gds_path = tmp_path / "short.gds"
    router = PICRoute(benchmark, config)

    router.run()

    net = router.cirdb.dbNets["n_0"]
    assert net.routed is True
    assert net.prebuilt_route is True
    assert net.wirelength == pytest.approx(5)
    assert router.cirdb.connectivity_report is not None
    assert config.run.output_layout_gds_path.is_file()


@pytest.mark.parametrize("shift", [(0, 0), (100, 200)])
def test_toy_route_end_to_end(tmp_path: Path, shift: tuple[int, int]) -> None:
    source = (
        Path(__file__).parents[1]
        / "picroute"
        / "benchmarks"
        / "toy_example"
        / "toy_example.gp.yml"
    )
    data = yaml.safe_load(source.read_text())
    dx, dy = shift
    data["settings"]["die_area"] = [[dx, dy], [dx + 200, dy + 200]]
    for instance in data["instances"].values():
        location = instance["settings"]["placement"][1]
        location[0] += dx
        location[1] += dy

    benchmark = tmp_path / "toy.gp.yml"
    benchmark.write_text(yaml.safe_dump(data, sort_keys=False))
    config = RouterConfig.load(
        Path(__file__).parents[1] / "picroute" / "config" / "comp_LiDAR.yml"
    )
    output = tmp_path / "toy.gds"
    config.run.output_layout_gds_path = str(output)

    router = PICRoute(benchmark, config)
    router.run()

    assert output.is_file()
    assert router.cirdb.connectivity_report is not None
    lengths = {
        name: round(net.wirelength, 3) for name, net in router.cirdb.dbNets.items()
    }
    assert lengths == {"n_0": 129.12, "n_1": 46.667}
    for net in router.cirdb.dbNets.values():
        for path_record in net.routed_path:
            assert all(dx <= point[0] <= dx + 200 for point in path_record[0])
            assert all(dy <= point[1] <= dy + 200 for point in path_record[0])
