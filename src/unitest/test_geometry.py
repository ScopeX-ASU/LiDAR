import gdsfactory as gf

from picroute.config.config import Config
from picroute.drc.bitmap import Bitmap
from picroute.routing.geometry import RouteGeometry


def test_occupied_grid_cells_use_filled_klayout_region() -> None:
    config = Config()
    config.dr = Config()
    config.dr.cross_section = "strip"
    config.dr.bend_radius = 5
    geometry = RouteGeometry(config)
    component = gf.path.extrude(
        gf.path.straight(length=4), width=4, layer=geometry.layer
    )
    bitmap = Bitmap([[-1, -3], [6, 3]], resolution=1)

    cells = geometry.occupied_grid_cells(component, bitmap)

    # The center of the wide guide is not on its polygon boundary, but it is
    # part of the filled KLayout region and must be occupied.
    center_cell = tuple(bitmap.to_grid([2, 0]))
    assert center_cell in cells
