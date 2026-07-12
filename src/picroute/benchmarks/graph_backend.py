"""Benchmark the legacy NetworkX and rustworkx netlist graph workloads.

Run with::

    python -m picroute.benchmarks.graph_backend
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from collections.abc import Callable
from typing import Any

from picroute.database.netlist_graph import NetlistGraph

Edge = tuple[int, int, dict[str, Any]]
Timings = tuple[float, float, float]


def make_layered_dag(layers: int, width: int) -> list[Edge]:
    """Create a deterministic circuit-like DAG with three-way fan-out."""
    edges: list[Edge] = []
    for layer in range(layers - 1):
        for offset in range(width):
            source = layer * width + offset
            for shift in (0, 1, 7):
                target = (layer + 1) * width + (offset + shift) % width
                edges.append(
                    (
                        source,
                        target,
                        {
                            "weight": float((source % 17) + shift + 1),
                            "label": f"n{source}_{shift}",
                        },
                    )
                )
    return edges


def run_networkx(edges: list[Edge]) -> tuple[Timings, tuple[Any, ...]]:
    """Run the pre-migration workload for comparison."""
    import networkx as nx

    start = time.perf_counter()
    graph = nx.DiGraph()
    for source, target, data in edges:
        graph.add_edge(source, target, **data)
    build_seconds = time.perf_counter() - start

    start = time.perf_counter()
    current_level = {node for node in graph.nodes() if graph.in_degree(node) == 0}
    visited: set[str] = set()
    orders: list[list[str]] = []
    while current_level:
        labels: list[str] = []
        next_level: set[int] = set()
        for node in current_level:
            for neighbor in graph.successors(node):
                next_level.add(neighbor)
                label = graph[node][neighbor]["label"]
                if label not in visited:
                    labels.append(label)
                    visited.add(label)
        if labels:
            orders.append(labels)
        current_level = next_level
    topology_seconds = time.perf_counter() - start

    start = time.perf_counter()
    path = nx.dag_longest_path(graph, weight="weight")
    length = nx.dag_longest_path_length(graph, weight="weight")
    critical_path_seconds = time.perf_counter() - start
    result = (len(graph), graph.number_of_edges(), len(orders), len(path), length)
    return (build_seconds, topology_seconds, critical_path_seconds), result


def run_rustworkx(edges: list[Edge]) -> tuple[Timings, tuple[Any, ...]]:
    """Run the migrated workload."""
    start = time.perf_counter()
    graph = NetlistGraph()
    graph.add_edges(
        (source, target, data["weight"], data["label"])
        for source, target, data in edges
    )
    build_seconds = time.perf_counter() - start

    start = time.perf_counter()
    orders = graph.topology_orders()
    topology_seconds = time.perf_counter() - start

    start = time.perf_counter()
    path, length = graph.weighted_longest_path()
    critical_path_seconds = time.perf_counter() - start
    result = (len(graph), graph.number_of_edges, len(orders), len(path), length)
    return (build_seconds, topology_seconds, critical_path_seconds), result


def benchmark(
    name: str,
    runner: Callable[[list[Edge]], tuple[Timings, tuple[Any, ...]]],
    edges: list[Edge],
    repeats: int,
) -> dict[str, Any]:
    """Return median timings after one warm-up run."""
    runner(edges)
    samples: list[Timings] = []
    result: tuple[Any, ...] = ()
    for _ in range(repeats):
        gc.collect()
        timings, result = runner(edges)
        samples.append(timings)

    medians = [statistics.median(values) for values in zip(*samples, strict=True)]
    return {
        "backend": name,
        "result": result,
        "median_seconds": {
            "build": medians[0],
            "topology": medians[1],
            "critical_path": medians[2],
            "total": sum(medians),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=100)
    parser.add_argument("--width", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--backend", choices=("both", "networkx", "rustworkx"), default="both"
    )
    args = parser.parse_args()

    edges = make_layered_dag(args.layers, args.width)
    runners = {
        "networkx": run_networkx,
        "rustworkx": run_rustworkx,
    }
    selected = (
        runners if args.backend == "both" else {args.backend: runners[args.backend]}
    )
    output = {
        "nodes": args.layers * args.width,
        "edges": len(edges),
        "repeats": args.repeats,
        "benchmarks": [
            benchmark(name, runner, edges, args.repeats)
            for name, runner in selected.items()
        ],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
