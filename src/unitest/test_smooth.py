import gdsfactory as gf
import numpy as np
import pytest

from picroute.utils.smooth import prepare_waypoints


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
