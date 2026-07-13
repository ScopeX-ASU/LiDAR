from types import SimpleNamespace

import pytest

from picroute.routing.astarsearch import AstarSearch
from picroute.routing.drgridroute import DrGridRoute
from picroute.routing.utils import vector_intersection


@pytest.mark.parametrize(
    ("center1", "ori1", "center2", "ori2", "max_distance", "expected"),
    [
        ((0, 0), 0, (5, -5), 90, 10, True),
        ((0, 0), 180, (5, -5), 90, 10, False),
        ((0, 0), 0, (20, -5), 90, 10, False),
        ((0, 0), 0, (0, 1), 0, 10, False),
        ((0, 0), 0, (10, 0), 0, 10, True),
        ((0, 0), 0, (5, 0), 0, 10, False),
    ],
)
def test_vector_intersection_matches_linestring_semantics(
    center1, ori1, center2, ori2, max_distance, expected
) -> None:
    assert (
        vector_intersection(
            center1,
            ori1,
            center2,
            ori2,
            max_distance=max_distance,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("center2", "orientation2", "expected"),
    [
        ((10, -10), 90, (True, "all_angle", 90)),
        ((-10, -10), 90, (False, "None", 0)),
    ],
)
def test_connection_check_uses_forward_segments(
    center2, orientation2, expected
) -> None:
    search = AstarSearch.__new__(AstarSearch)
    search.radius = 5
    port1 = SimpleNamespace(center=(0, 0), orientation=0)
    port2 = SimpleNamespace(center=center2, orientation=orientation2)

    assert search.connection_check(port1, port2) == expected


def test_topology_order_counts_crossing_edges() -> None:
    def net(start, end):
        return SimpleNamespace(
            port1=SimpleNamespace(center=start),
            port2=SimpleNamespace(center=end),
            comp_dist=None,
            topology_crossing=0,
        )

    nets = {
        "horizontal": net((0, 0), (10, 0)),
        "vertical": net((5, -5), (5, 5)),
        "separate": net((0, 10), (10, 10)),
    }
    route = DrGridRoute.__new__(DrGridRoute)
    route._config = SimpleNamespace(dr=SimpleNamespace(net_order="topo"))
    route._cirdb = SimpleNamespace(
        dbNets=nets,
        topology_orders=[["horizontal", "vertical", "separate"]],
    )

    route.process_net_order()

    assert nets["horizontal"].topology_crossing == 1
    assert nets["vertical"].topology_crossing == 1
    assert nets["separate"].topology_crossing == 0
