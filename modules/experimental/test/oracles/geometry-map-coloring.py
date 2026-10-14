#!/usr/bin/env python3
"""Quality reference for degree-first map coloring (algorithms roadmap P1).

Builds planar adjacency graphs (Delaunay triangulations of seeded random points, scipy) and records
the color counts of networkx `greedy_color` with the `largest_first`, `smallest_last` and
`random_sequential` strategies. GPUMapColoring's largest-degree-first order breaks degree ties by
a seeded hash instead of networkx's stable sort, so the GPU spec compares color COUNTS only (exact
colors are checked against the TS sequential oracle in the same spec).

Fixture written (relative to modules/experimental/test/oracles):
  fixtures/geometry-map-coloring.json

Usage, from the repository root:
  python3 -I modules/experimental/test/oracles/geometry-map-coloring.py

Needs scipy, numpy and networkx.
"""
import json
import os
from importlib.metadata import version

import networkx
import numpy as np
from scipy.spatial import Delaunay

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'fixtures', 'geometry-map-coloring.json')
GENERATOR = 'modules/experimental/test/oracles/geometry-map-coloring.py'


def build_graph(point_count, seed):
    rng = np.random.default_rng(seed)
    triangulation = Delaunay(rng.random((point_count, 2)))
    edges = set()
    for a, b, c in triangulation.simplices:
        for u, v in ((a, b), (b, c), (a, c)):
            edges.add((int(min(u, v)), int(max(u, v))))
    graph = networkx.Graph()
    graph.add_nodes_from(range(point_count))
    graph.add_edges_from(edges)
    counts = {}
    for strategy in ('largest_first', 'smallest_last', 'random_sequential'):
        colors = networkx.greedy_color(graph, strategy=strategy)
        counts[strategy] = max(colors.values()) + 1
    offsets, neighbors = [0], []
    for node in range(point_count):
        neighbors.extend(sorted(graph.neighbors(node)))
        offsets.append(len(neighbors))
    return {
        'name': f'delaunay-{point_count}-seed-{seed}',
        'offsets': offsets,
        'neighbors': neighbors,
        'networkxColorCounts': counts,
    }


def main():
    fixture = {
        'graphs': [build_graph(1500, 1), build_graph(3000, 2), build_graph(3000, 3)],
        'provenance': {
            'generator': GENERATOR,
            'networkx': version('networkx'),
            'scipy': version('scipy'),
            'numpy': version('numpy'),
            'tolerance': {
                'derivation': 'color counts of greedy heuristics differ by tie-breaking only; the '
                'spec allows one color above networkx largest_first and requires at most the '
                'same-run random-priority count.',
            },
        },
    }
    with open(OUTPUT, 'w') as handle:
        json.dump(fixture, handle)
        handle.write('\n')
    print('wrote', os.path.relpath(OUTPUT))


if __name__ == '__main__':
    main()
