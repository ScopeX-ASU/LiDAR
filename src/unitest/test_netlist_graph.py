import pytest

from picroute.database.netlist_graph import NetlistGraph


def test_topology_orders_and_degrees() -> None:
    graph = NetlistGraph()
    graph.add_edge(10, 20, weight=1.0, label="a")
    graph.add_edge(10, 30, weight=2.0, label="b")
    graph.add_edge(20, 40, weight=3.0, label="c")
    graph.add_edge(30, 40, weight=4.0, label="d")

    assert [set(level) for level in graph.topology_orders()] == [
        {"a", "b"},
        {"c", "d"},
    ]
    assert graph.out_degree(10) == 2
    assert graph.out_degree(40) == 0
    assert graph.out_degree(999) == 0


def test_weighted_longest_path_and_edge_update() -> None:
    graph = NetlistGraph()
    graph.add_edge(10, 20, weight=1.5, label="a")
    graph.add_edge(10, 30, weight=2.0, label="b")
    graph.add_edge(20, 40, weight=4.0, label="c")
    graph.add_edge(30, 40, weight=1.0, label="d")

    path, length = graph.weighted_longest_path()
    assert path == [10, 20, 40]
    assert length == pytest.approx(5.5)

    graph.add_edge(10, 20, weight=7.25, label="replacement")
    assert graph.number_of_edges == 4
    assert graph.edge_data(10, 20) == {
        "weight": 7.25,
        "label": "replacement",
    }


def test_topology_orders_terminates_for_a_reachable_cycle() -> None:
    graph = NetlistGraph()
    graph.add_edge(1, 2, weight=1, label="entry")
    graph.add_edge(2, 3, weight=1, label="forward")
    graph.add_edge(3, 2, weight=1, label="feedback")

    assert graph.topology_orders() == [["entry"], ["forward"], ["feedback"]]
