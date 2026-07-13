import gdsfactory as gf
import pytest

from picroute.config.config import RouterConfig
from picroute.drc.bitmap import Bitmap
from picroute.routing.geometry import RouteGeometry


def test_occupied_grid_cells_use_filled_klayout_region() -> None:
    config = RouterConfig()
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
    config = RouterConfig()
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
    config = RouterConfig()
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


@pytest.mark.parametrize(
    ("center2", "orientation2"),
    [((20, 0), 180), ((15, 15), 270)],
)
def test_simple_access_avoids_all_angle_optimizer(
    monkeypatch, center2, orientation2
) -> None:
    config = RouterConfig()
    config.dr.cross_section = "strip"
    config.dr.bend_radius = 5
    geometry = RouteGeometry(config)
    layer = gf.get_layer(geometry.layer)
    port1 = gf.Port(
        name="o1", center=(0, 0), width=geometry.width, orientation=0, layer=layer
    )
    port2 = gf.Port(
        name="o2",
        center=center2,
        width=geometry.width,
        orientation=orientation2,
        layer=layer,
    )

    def fail_all_angle(**kwargs):
        raise AssertionError("simple access should not run all-angle optimization")

    monkeypatch.setattr(gf.routing, "route_bundle_all_angle", fail_all_angle)
    component, route = geometry.connect_ports_all_angle(port1, port2)

    assert route.length > 0
    assert not component.get_region(geometry.layer).is_empty()


def test_coincident_access_does_not_create_redundant_geometry(monkeypatch) -> None:
    config = RouterConfig()
    config.dr.cross_section = "strip"
    geometry = RouteGeometry(config)
    layer = gf.get_layer(geometry.layer)
    route = gf.Component()
    route.add_port(
        name="o1", center=(2, 3), width=geometry.width, orientation=180, layer=layer
    )
    endpoint = gf.Port(
        name="endpoint",
        center=(2, 3),
        width=geometry.width,
        orientation=0,
        layer=layer,
    )

    def fail_connection(*args, **kwargs):
        raise AssertionError("coincident ports should not create an access route")

    monkeypatch.setattr(geometry, "connect_ports_all_angle", fail_connection)

    access, length = geometry._access_waveguide(route, endpoint, index=1)
    assert access is None
    assert length == 0
