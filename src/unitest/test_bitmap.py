from types import SimpleNamespace

import gdsfactory as gf
import numpy as np
import pytest

from picroute.drc.bitmap import Bitmap


def test_bitmap_uses_compact_directional_state() -> None:
    bitmap = Bitmap([[0, 0], [10, 10]], resolution=1)
    node = bitmap.bitMap[2, 3]

    node.update("waveguide", id="n1", wgtype=90)
    assert node.type == 90
    assert node.netID == ["n1"]

    node.update("waveguide", id="n2", wgtype=0)
    assert node.type == "compound"
    assert set(node.netID) == {"n1", "n2"}

    node.delete("n1")
    assert node.type == 0
    assert node.netID == ["n2"]
    assert bitmap._types.dtype == np.int16


def test_bitmap_query_returns_cell_state_without_a_node_view() -> None:
    bitmap = Bitmap([[0, 0], [10, 10]], resolution=1)

    assert bitmap.query(2, 3) == (False, "empty", None)
    bitmap.node(2, 3).update("waveguide", id="route", wgtype=90)
    assert bitmap.query(2, 3) == (True, 90, "route")
    bitmap.node(4, 5).update("port", id="net")
    assert bitmap.query(4, 5) == (True, "port", "net")

    with pytest.raises(IndexError, match="outside"):
        bitmap.query(-1, 0)


def test_bitmap_coordinates_are_relative_to_die_origin() -> None:
    bitmap = Bitmap([[-10, -20], [10, 20]], resolution=2)

    np.testing.assert_array_equal(bitmap.to_grid([-10, -20]), [0, 0])
    np.testing.assert_allclose(bitmap.to_world([0, 0]), [-9, -19])

    blockage = SimpleNamespace(
        bbox=gf.kdb.DBox(-12, -22, -8, -18), instanceName="partly-outside"
    )
    bitmap.initMap([blockage])
    assert bitmap.bitMap[0, 0].type == "blk"
    assert bitmap.bitMap[0, 0].blkID == "partly-outside"


@pytest.mark.parametrize(
    ("die_size", "resolution", "match"),
    [
        ([[0, 0], [10, 10]], 0, "resolution"),
        ([[0, 0], [0, 10]], 1, "upper die coordinate"),
        ([[0, 0, 0], [10, 10, 10]], 1, "two 2D coordinates"),
    ],
)
def test_bitmap_rejects_invalid_geometry(die_size, resolution, match) -> None:
    with pytest.raises(ValueError, match=match):
        Bitmap(die_size, resolution=resolution)
