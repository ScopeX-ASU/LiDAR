import gdsfactory as gf
import numpy as np
import pytest

from picroute.utils.smooth import alignment, prepare_waypoints


@pytest.mark.parametrize(
    ("points", "expected_points", "expected_bend"),
    [
        (
            [[0, 0], [20, 0], [20, 20]],
            [[0, 0], [20, 0], [20, 20]],
            90.0,
        ),
        (
            [[-50, 50], [-100, 100], [-100, 200]],
            [[-50, 50], [-100, 100], [-100, 200]],
            45.0,
        ),
        (
            [[0, 0], [20, 0], [40, 0], [40, 20]],
            [[0, 0], [40, 0], [40, 20]],
            90.0,
        ),
        (
            [[0, 0], [20, 0], [30, 10], [30, 30], [20, 40]],
            [[0, 0], [20, 0], [30, 10], [30, 30], [20, 40]],
            135.0,
        ),
    ],
)
def test_prepare_waypoints(points, expected_points, expected_bend) -> None:
    waypoints, accumulated_bend = prepare_waypoints(points)

    np.testing.assert_allclose(waypoints, expected_points)
    assert accumulated_bend == pytest.approx(expected_bend)


@pytest.mark.parametrize(
    "points",
    [
        [[0, 0], [20, 0], [20, 20]],
        [[-50, 50], [-100, 100], [-100, 200]],
        [[3, 4], [23, 24], [43, 24], [63, 44]],
        [[0, 0], [20, 0], [40, 0], [40, 20]],
        [[0, 0], [20, 0], [30, 10], [30, 30], [20, 40]],
    ],
)
@pytest.mark.parametrize(
    ("bend", "bend_kwargs"),
    [(gf.path.arc, {}), (gf.path.euler, {"use_eff": True})],
)
def test_prepared_waypoints_preserve_extruded_geometry_xor(
    points, bend, bend_kwargs
) -> None:
    """Explicit waypoint bookkeeping must not alter the routed polygon."""
    original = gf.path.smooth(points, radius=3, bend=bend, **bend_kwargs).extrude(
        width=0.5, layer=(1, 0)
    )
    waypoints, _ = prepare_waypoints(points)
    refactored = gf.path.smooth(waypoints, radius=3, bend=bend, **bend_kwargs).extrude(
        width=0.5, layer=(1, 0)
    )

    xor = original.get_region((1, 0), merge=True) ^ refactored.get_region(
        (1, 0), merge=True
    )
    assert xor.is_empty()


def test_alignment_uses_x_coordinate_for_vertical_destination() -> None:
    layer = gf.get_layer("WG")
    port1 = gf.Port(
        name="start", center=(0, 0), width=0.5, orientation=0, layer=layer
    )
    port2 = gf.Port(
        name="end", center=(25, 20), width=0.5, orientation=90, layer=layer
    )
    points = [[0, 0], [10, 0], [10, 10], [20, 20]]

    result = alignment(points, port1, port2, radius=1, bend=gf.path.euler)

    assert result is not None
    assert result[-1][0] == 25
    assert result[-2][0] == 25
