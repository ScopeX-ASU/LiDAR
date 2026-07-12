from types import SimpleNamespace

import pytest

from picroute.drc.bitmap import Bitmap
from picroute.drc.drcmanager import DrcManager


def _manager(size: int = 7) -> DrcManager:
    manager = DrcManager.__new__(DrcManager)
    manager._bitmap = Bitmap([[0, 0], [size, size]], resolution=1)
    manager.width = manager._bitmap.width
    manager.height = manager._bitmap.height
    return manager


@pytest.mark.parametrize(
    ("orientation", "occupied"),
    [
        (0, (3, 2)),
        (45, (2, 4)),
        (90, (2, 3)),
        (135, (2, 2)),
        (180, (3, 2)),
        (225, (2, 4)),
        (270, (2, 3)),
        (315, (2, 2)),
    ],
)
def test_oriented_bitmap_scan_uses_perpendicular_cells(
    orientation: int, occupied: tuple[int, int]
) -> None:
    manager = _manager()
    manager.bitMap.node(*occupied).update("waveguide", id="other", wgtype=90)

    assert manager.checkBitmap_ori((3, 3, orientation)) == (True, 90, "other")


@pytest.mark.parametrize("orientation", [0, 45, 90, 135, 180, 225, 270, 315])
def test_oriented_bitmap_scan_treats_outside_die_as_blocked(orientation: int) -> None:
    manager = _manager(3)

    assert manager.checkBitmap_ori((0, 0, orientation)) == (True, "blk", None)


def test_spacing_ignores_group_routes_and_counts_outside_die() -> None:
    manager = _manager(5)
    manager.bitMap.node(2, 1).update("waveguide", id="group", wgtype=0)
    manager.bitMap.node(2, 3).update("waveguide", id="other", wgtype=0)

    node = SimpleNamespace(pos=(2, 2, 0))
    assert manager.checkSpacing(node, 1, groups={"group"}) == 1

    boundary_node = SimpleNamespace(pos=(0, 0, 0))
    assert manager.checkSpacing(boundary_node, 1) == 1
