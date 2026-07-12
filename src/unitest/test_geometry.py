import gdsfactory as gf
import pytest

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


@pytest.mark.parametrize(
    "points",
    [
        [(0, 0), (20, 0), (20, 20)],
        [(0, 0), (15, 0), (25, 10), (40, 10)],
    ],
)
def test_all_angle_placement_matches_smoothed_route_region(points) -> None:
    config = Config()
    config.dr = Config()
    config.dr.cross_section = "strip"
    config.dr.bend_radius = 5
    geometry = RouteGeometry(config)

    path = geometry.smooth(points)
    extruded = geometry.extrude(path)
    instance_route = geometry.place_all_angle(points)

    xor = extruded.get_region(geometry.layer, merge=True) ^ instance_route.get_region(
        geometry.layer, merge=True
    )
    assert xor.is_empty()


def test_all_angle_access_uses_native_route_result() -> None:
    config = Config()
    config.dr = Config()
    config.dr.cross_section = "strip"
    config.dr.bend_radius = 5
    geometry = RouteGeometry(config)
    layer = gf.get_layer(geometry.layer)
    port1 = gf.Port(
        name="o1", center=(0, 0), width=geometry.width, orientation=0, layer=layer
    )
    port2 = gf.Port(
        name="o2",
        center=(30, 15),
        width=geometry.width,
        orientation=210,
        layer=layer,
    )

    component, route = geometry.connect_ports_all_angle(port1, port2)

    assert component.routes["access"] is route
    assert route.length > 0
    assert not component.get_region(geometry.layer).is_empty()
