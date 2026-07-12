from pathlib import Path

import gdsfactory as gf
import numpy as np

from picroute.database.schematic import CustomSchematic


ORIENTATIONS = ("N", "S", "E", "W", "FN", "FS", "FE", "FW")


def test_database_uses_transformed_instance_geometry(tmp_path: Path) -> None:
    path = tmp_path / "orientations.yml"
    source = CustomSchematic(path)
    source.update_settings(design="orientations", die_area=[[0, 0], [500, 500]])
    component = gf.components.straight(length=10)
    for index, orientation in enumerate(ORIENTATIONS):
        name = f"straight_{orientation}"
        source.add_instance(name, component)
        source.update_placement(
            name,
            ["FIXED", [40 + index * 40, 100], orientation, [0, 0, 0, 0]],
        )
    source.commit()

    loaded = CustomSchematic(path)
    loaded.load_gp()

    for db_port in loaded._dbPorts.values():
        np.testing.assert_allclose(db_port.center, db_port.port.dcenter)
        assert db_port.width == db_port.port.dwidth
        assert db_port.orientation == round(db_port.port.orientation) % 360

    placed_layout = loaded.layout.insts[0].cell
    for name, blockage in loaded.initBlk.items():
        bbox = placed_layout.insts[name].dbbox()
        np.testing.assert_allclose(
            [
                blockage.bbox.left,
                blockage.bbox.bottom,
                blockage.bbox.right,
                blockage.bbox.top,
            ],
            [bbox.left, bbox.bottom, bbox.right, bbox.top],
        )


def test_split_instance_port_ids_are_contiguous(tmp_path: Path) -> None:
    path = tmp_path / "split.yml"
    source = CustomSchematic(path)
    source.update_settings(design="split", die_area=[[0, 0], [100, 100]])
    source.add_instance("mmi", gf.components.mmi1x2())
    source.schematic.instances["mmi"].settings["split"] = True
    source.update_placement("mmi", ["FIXED", [20, 20], "N", [0, 0, 0, 0]])
    source.commit()

    loaded = CustomSchematic(path)
    loaded.load_gp()

    assert [loaded._dbPorts[f"mmi,o{i}"].idBlk for i in (1, 2, 3)] == [0, 1, 2]
