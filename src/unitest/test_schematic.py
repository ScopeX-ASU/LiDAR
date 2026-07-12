from pathlib import Path
from shutil import copyfile

from picroute.database.schematic import CustomSchematic


def test_se_load_netlist(tmp_path: Path) -> None:
    benchmark = (
        Path(__file__).parents[1]
        / "picroute"
        / "benchmarks"
        / "toy_example"
        / "toy_example.gp.yml"
    )
    benchmark_copy = tmp_path / benchmark.name
    copyfile(benchmark, benchmark_copy)
    schematic = CustomSchematic(benchmark_copy)
    schematic.load_gp()

    assert len(schematic.initBlk) == 6
    assert len(schematic.dbNets) == 2
    assert len(schematic.layout.insts) == 6
    assert len(schematic.native_schematic.instances) == 6
    assert len(schematic.native_schematic.nets) == 2
