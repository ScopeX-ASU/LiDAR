"""Rustworkx-backed graph operations for circuit netlists."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import rustworkx as rx


class NetlistGraph:
    """Directed netlist graph keyed by the circuit database's block IDs."""

    def __init__(self) -> None:
        # NetworkX DiGraph.add_edge() replaces data for an existing edge. Disabling
        # parallel edges preserves that behavior in rustworkx.
        self._graph = rx.PyDiGraph(multigraph=False)
        self._node_indices: dict[int, int] = {}

    def __len__(self) -> int:
        return self._graph.num_nodes()

    @property
    def number_of_edges(self) -> int:
        return self._graph.num_edges()

    def _node_index(self, node: int) -> int:
        try:
            return self._node_indices[node]
        except KeyError:
            index = self._graph.add_node(node)
            self._node_indices[node] = index
            return index

    def add_edge(self, source: int, target: int, *, weight: float, label: str) -> None:
        """Add an edge, updating its payload if the endpoints already exist."""
        self._graph.add_edge(
            self._node_index(source),
            self._node_index(target),
            {"weight": float(weight), "label": label},
        )

    def add_edges(self, edges: Iterable[tuple[int, int, float, str]]) -> None:
        """Add multiple edges with one call into rustworkx."""
        indexed_edges = [
            (
                self._node_index(source),
                self._node_index(target),
                {"weight": float(weight), "label": label},
            )
            for source, target, weight, label in edges
        ]
        self._graph.add_edges_from(indexed_edges)

    def out_degree(self, node: int) -> int:
        """Return the number of outgoing edges for a block ID."""
        index = self._node_indices.get(node)
        return 0 if index is None else self._graph.out_degree(index)

    def edge_data(self, source: int, target: int) -> dict[str, Any]:
        """Return the payload for an edge identified by block IDs."""
        return self._graph.get_edge_data(
            self._node_indices[source], self._node_indices[target]
        )

    def topology_orders(self) -> list[list[str]]:
        """Group edge labels by breadth-first topology level."""
        graph = self._graph
        current_level = {
            index for index in graph.node_indices() if graph.in_degree(index) == 0
        }
        expanded: set[int] = set()
        visited: set[str] = set()
        orders: list[list[str]] = []

        while current_level:
            labels: list[str] = []
            next_level: set[int] = set()
            for node in current_level - expanded:
                expanded.add(node)
                for _source, neighbor, data in graph.out_edges(node):
                    if neighbor not in expanded:
                        next_level.add(neighbor)
                    label = data["label"]
                    if label not in visited:
                        labels.append(label)
                        visited.add(label)
            if labels:
                orders.append(labels)
            current_level = next_level

        return orders

    def weighted_longest_path(self) -> tuple[list[int], float]:
        """Return block IDs and total floating-point weight for the longest DAG path."""
        if not self._node_indices:
            return [], 0.0

        def edge_weight(_source: int, _target: int, data: dict[str, Any]) -> float:
            return data["weight"]

        indices = rx.dag_weighted_longest_path(self._graph, edge_weight)
        length = sum(
            self._graph.get_edge_data(source, target)["weight"]
            for source, target in zip(indices, indices[1:])
        )
        return [self._graph[index] for index in indices], length
