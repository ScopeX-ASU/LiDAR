from heapq import heappop
from itertools import count
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

import picroute.routing.astarsearch as astarsearch_module
from picroute.routing.astarsearch import AstarSearch, GridAstarNode
from picroute.drc.drcmanager import DrcResult
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


def test_neighbor_generation_computes_current_distance_once(monkeypatch) -> None:
    search = AstarSearch.__new__(AstarSearch)
    search.nextSteps = {
        0: [(1, 0, 0, "straight_0"), (0, 1, 90, "bend_90")]
    }
    search.endNode = GridAstarNode(pos=(8, 8, 180))
    search.connect_th = 2
    search.routing_bound = ((-1, -1), (10, 10))
    search._drcmgr = SimpleNamespace(
        bViolateDRC=Mock(return_value=DrcResult(False))
    )
    search._bStrictDRC = True
    search._net = SimpleNamespace(netName="net")
    search.vAllNodesMap = {}
    current = GridAstarNode(pos=(1, 1, 0), crossing_budget=3)
    distance = Mock(return_value=14)
    monkeypatch.setattr(astarsearch_module, "manhattanDH", distance)

    neighbors = list(search.findNeighbors(current))

    assert len(neighbors) == 2
    distance.assert_called_once_with(current.pos, search.endNode.pos)


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


def test_history_map_promotes_before_overflow_and_clears_in_place() -> None:
    route = DrGridRoute.__new__(DrGridRoute)
    route.historyMap = np.zeros((2, 2, 8), dtype=np.uint16)
    route.historyMap[1, 1, 0] = np.iinfo(np.uint16).max - 5
    original_map = route.historyMap
    route.orientations = {0: 0}
    route._config = SimpleNamespace(dr=SimpleNamespace(history_cost=10))
    net = SimpleNamespace(
        NetPort1=SimpleNamespace(port_grids=[]),
        NetPort2=SimpleNamespace(port_grids=[]),
    )

    route.updateHistoryMap([(1, 1, 0)], net)

    assert route.historyMap.dtype == np.uint32
    assert route.historyMap[1, 1, 0] == np.iinfo(np.uint16).max + 5
    promoted_map = route.historyMap
    route.clearHistoryMap()
    assert route.historyMap is promoted_map
    assert np.count_nonzero(route.historyMap) == 0
    assert route.historyMap is not original_map
