from pathlib import Path

import pytest
import yaml

from picroute.config.config import Config
from picroute.main.picroute import PICRoute


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
    config = Config()
    config.load(Path(__file__).parents[1] / "picroute" / "config" / "comp_LiDAR.yml")
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
