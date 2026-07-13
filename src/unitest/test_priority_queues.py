from heapq import heappop
from itertools import count
from types import SimpleNamespace
from unittest.mock import Mock

from picroute.routing.astarsearch import AstarSearch, GridAstarNode
from picroute.routing.drgridroute import DrGridRoute


def _empty_search() -> AstarSearch:
    search = AstarSearch.__new__(AstarSearch)
    search.nodepq = []
    search._nodepq_counter = count()
    return search


def test_astar_frontier_preserves_insertion_order_for_equal_costs() -> None:
    search = _empty_search()
    nodes = [GridAstarNode(pos=(index, 0, 0), cost_f=10) for index in range(3)]

    for node in nodes:
        search._push_node(node)

    assert [search._pop_node() for _ in nodes] == nodes
    assert search._pop_node() is None


def test_astar_frontier_discards_stale_priority_entries() -> None:
    search = _empty_search()
    updated = GridAstarNode(pos=(0, 0, 0), cost_f=10)
    other = GridAstarNode(pos=(1, 0, 0), cost_f=5)

    search._push_node(updated)
    search._push_node(other)
    updated.cost_f = 1
    search._push_node(updated)

    assert search._pop_node() is updated
    assert search._pop_node() is other
    assert search._pop_node() is None
    assert search.nodepq == []


def test_astar_node_equality_and_hash_use_position() -> None:
    first = GridAstarNode(pos=(1, 2, 90), cost_f=1)
    same_position = GridAstarNode(pos=(1, 2, 90), cost_f=100)
    same_cost = GridAstarNode(pos=(2, 2, 90), cost_f=1)

    assert first == same_position
    assert hash(first) == hash(same_position)
    assert first != same_cost


def test_astar_nodes_share_immutable_empty_violation_state() -> None:
    first = GridAstarNode(pos=(1, 2, 90))
    second = GridAstarNode(pos=(2, 2, 90))

    assert first.violated_nets is second.violated_nets
    assert first.violated_nets == frozenset()
    assert not hasattr(first, "neighbors")


def test_net_frontier_is_stable_and_rejects_duplicate_objects() -> None:
    route = DrGridRoute.__new__(DrGridRoute)
    route.net_PQ = []
    route._net_pq_ids = set()
    route._net_pq_counter = count()

    first = SimpleNamespace(netName="first", routed=False, comp_dist=1, routing_order=2)
    second = SimpleNamespace(
        netName="second", routed=False, comp_dist=1, routing_order=2
    )
    earlier = SimpleNamespace(
        netName="earlier", routed=False, comp_dist=0, routing_order=9
    )
    route._cirdb = SimpleNamespace(
        _dbNets={"first": first, "second": second, "earlier": earlier}
    )

    route.addUnroutedNetsToPQ()
    route.addUnroutedNetsToPQ()

    assert len(route.net_PQ) == 3
    assert [heappop(route.net_PQ)[-1].netName for _ in range(3)] == [
        "earlier",
        "first",
        "second",
    ]


def test_failed_crossing_retry_is_not_registered_as_success() -> None:
    route = DrGridRoute.__new__(DrGridRoute)
    route._config = SimpleNamespace(dr=SimpleNamespace(net_order="naive", group=True))
    crossing_net = SimpleNamespace(distance={"Euler": 2_000})
    current_net = SimpleNamespace(
        netName="current",
        crossing_budget=10,
        current_budget=10,
        crossing_num=1,
        crossing_nets={"long"},
        failed_count=0,
        distance={"Euler": 10},
        backup_net=Mock(return_value=(0, {"long"}, 1)),
        clearNet=Mock(),
    )
    route._cirdb = SimpleNamespace(dbNets={"long": crossing_net})
    route._run_astar = Mock(side_effect=[True, False])
    route.ripuplocalnets = Mock()
    route.registar_net = Mock()

    succeeded = route.routeSingleNet(current_net, strictDRC=True)

    assert succeeded is False
    route.registar_net.assert_not_called()
