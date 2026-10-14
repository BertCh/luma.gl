# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Reference multi-source shortest-path costs for GPUNetworkReachability from networkx.

Writes fixtures/network-shortest-paths.json next to this script (consumed by
test/gpu-network/network-reachability/network-reachability-reference.node.spec.ts, which checks the
TypeScript Dijkstra oracle against it, and by the 'networkx references' case of
gpu-network-reachability.spec.ts). Run from anywhere with
    <venv>/bin/python -I modules/experimental/test/oracles/network-shortest-paths.py
in a venv with numpy and networkx (versions are recorded in the fixture).

Reference: networkx.single_source_dijkstra_path_length on a MultiDiGraph with a virtual super
source joined to every source node by an edge of that source's initial cost, with
`cutoff=costLimit` (networkx keeps costs <= cutoff, as the GPU keeps candidates <= costLimit).
Edges with a negative weight are impassable on the GPU and are left out of the networkx graph.
Weights are float32 values (exact in the fixture); networkx sums them in float64.

Scenes:
- 'hand': 9 nodes, a cycle, zero-weight edges (a plateau), a negative (impassable) edge, an
  unreachable node, two sources with initial costs 0 and 2.5, and a cost limit of 9.
- 'random-<seed>': randomized property scenes. 400 nodes, 1600 directed edges drawn uniformly
  (parallel edges and self-loops allowed), 5% of weights 0, the rest uniform in [0.5, 10), the last
  5% of nodes without in-edges, three sources with initial costs in [0, 3). No cost limit.
- 'grid': a 24 x 24 grid with both directions of every street and independent weights in [1, 2),
  one source in a corner: about 46 hops deep, so it needs several frontier rounds.

Tolerance (stored per scene): the GPU relaxes in float32, so a cost is a float32 sum of k edge
weights plus the source cost, each addition rounding by at most 2^-24 relative. A float32 path sum
with k terms is within k 2^-24 C of its float64 value (C the path cost), and the float32 minimum is
within the larger of the two paths' bounds of the float64 minimum. The generator takes k as twice
the largest hop count of any networkx shortest path in the scene (the float32 optimum may take a
longer path of nearly equal cost) plus one, so costRelative = (2 h + 1) 2^-24. The reached set is
exact: the generator asserts that no cost lies within that tolerance of the cost limit.
"""

import json
import os
import sys

import networkx as nx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'fixtures', 'network-shortest-paths.json')
UNIT_ROUNDOFF = 2.0**-24
SUPER_SOURCE = 'super-source'


def to_float32(values):
    return [float(np.float32(value)) for value in values]


def reference_costs(node_count, edges, sources, source_costs, cost_limit):
    """networkx costs (None when unreached) and the largest shortest-path hop count."""
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(range(node_count))
    for source, target, weight in edges:
        if weight >= 0:
            graph.add_edge(source, target, weight=weight)
    for source, cost in zip(sources, source_costs):
        graph.add_edge(SUPER_SOURCE, source, weight=cost)
    cutoff = None if cost_limit is None else cost_limit
    lengths, paths = nx.single_source_dijkstra(graph, SUPER_SOURCE, cutoff=cutoff, weight='weight')
    costs = [lengths.get(node) for node in range(node_count)]
    hops = max(len(paths[node]) - 2 for node in range(node_count) if node in paths)
    return costs, hops


def make_scene(node_count, edges, sources, source_costs, cost_limit):
    edges = [(int(source), int(target), float(np.float32(weight))) for source, target, weight in edges]
    source_costs = to_float32(source_costs)
    costs, hops = reference_costs(node_count, edges, sources, source_costs, cost_limit)
    cost_relative = (2 * hops + 1) * UNIT_ROUNDOFF
    if cost_limit is not None:
        # Costs without the limit: no node may sit within the tolerance of the limit, on either
        # side, or float32 rounding could move it across.
        uncut, _ = reference_costs(node_count, edges, sources, source_costs, None)
        for cost in uncut:
            if cost is not None and abs(cost - cost_limit) <= cost_relative * cost_limit:
                raise RuntimeError('a cost lies within the tolerance of the cost limit')
    return {
        'nodeCount': node_count,
        'edges': [list(edge) for edge in edges],
        'sources': [int(source) for source in sources],
        'sourceCosts': source_costs,
        'costLimit': cost_limit,
        'costs': costs,
        'maximumHops': hops,
        'tolerance': {'costRelative': cost_relative, 'reachedSet': 'exact'}
    }


def hand_scene():
    edges = [
        (0, 1, 4.0),
        (0, 2, 1.0),
        (2, 1, 2.0),
        (1, 3, 1.0),
        (3, 4, 3.0),
        (4, 0, 1.0),
        (3, 5, 0.0),
        (5, 6, 0.0),
        (6, 3, 0.0),
        (2, 7, -1.0),
        (6, 7, 9.5)
    ]
    return make_scene(9, edges, [0, 5], [0.0, 2.5], 9.0)


def random_scene(seed):
    generator = np.random.default_rng(seed)
    node_count = 400
    edge_count = 1600
    reachable_count = int(node_count * 0.95)
    sources = generator.integers(0, node_count, edge_count)
    targets = generator.integers(0, reachable_count, edge_count)
    weights = generator.uniform(0.5, 10.0, edge_count)
    weights[generator.random(edge_count) < 0.05] = 0.0
    edges = list(zip(sources.tolist(), targets.tolist(), weights.tolist()))
    seeds = generator.choice(reachable_count, 3, replace=False).tolist()
    seed_costs = generator.uniform(0.0, 3.0, 3).tolist()
    return make_scene(node_count, edges, seeds, seed_costs, None)


def grid_scene():
    generator = np.random.default_rng(24)
    width = 24
    edges = []
    for y in range(width):
        for x in range(width):
            node = y * width + x
            for neighbor in ([node + 1] if x + 1 < width else []) + (
                [node + width] if y + 1 < width else []
            ):
                edges.append((node, neighbor, generator.uniform(1.0, 2.0)))
                edges.append((neighbor, node, generator.uniform(1.0, 2.0)))
    return make_scene(width * width, edges, [0], [0.0], None)


def main():
    fixture = {
        'generator': 'test/oracles/network-shortest-paths.py',
        'versions': {
            'python': sys.version.split()[0],
            'networkx': nx.__version__,
            'numpy': np.__version__
        },
        'scenes': {
            'hand': hand_scene(),
            'random-1': random_scene(1),
            'random-2': random_scene(2),
            'random-3': random_scene(3),
            'grid': grid_scene()
        }
    }
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, 'w', encoding='utf-8') as file:
        json.dump(fixture, file, separators=(',', ':'))
        file.write('\n')
    print(f'wrote {OUTPUT}')


if __name__ == '__main__':
    main()
