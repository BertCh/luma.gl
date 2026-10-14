# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Reference degree, component and modularity statistics for GPUNetworkStatistics from networkx.

Writes fixtures/network-statistics.json next to this script (consumed by
test/gpu-network/network-statistics/network-statistics-reference.node.spec.ts, which checks the
TypeScript statistics oracle against it, and by the 'networkx references' case of
gpu-network-statistics.spec.ts). Run from anywhere with
    <venv>/bin/python -I modules/experimental/test/oracles/network-statistics.py
in a venv with numpy and networkx (versions are recorded in the fixture).

Reference: on a simple networkx Graph (undirected) or DiGraph (directed), with self-loops:
- per-node degree: `Graph.degree` (a self-loop adds 2, review B-15), or `DiGraph.out_degree` and
  `in_degree` (a self-loop adds 1 to each);
- weak components: `number_connected_components` / `number_weakly_connected_components` and the
  size of the largest one;
- isolated vertices: `len(list(nx.isolates(graph)))` (a vertex with only a self-loop is not
  isolated);
- modularity: `nx.community.modularity(graph, communities, resolution=gamma)` with the
  communities given by the per-node labels.

Scenes ('undirected' and 'directed'): 300 nodes, 700 distinct edges (a planted partition: 75%
inside one of 9 blocks of contiguous nodes, the rest uniform), 12 self-loops, the last 20 nodes
left isolated, community labels = blocks with 15% reassigned at random, resolution 1.25. Edges are listed once; the GPU CSR of an undirected
scene lists both directions of every non-loop edge and a self-loop once.

Tolerance (stored per scene): counts, degrees and component statistics are exact. Modularity is a
float32 computation on the GPU: the exact integer intra count and community volumes are divided by
the slot count and combined in a fixed-order f32 sum of at most nodeCount terms, each below 1, so
the error is below (nodeCount + 4) 2^-24 (absolute, Q is in [-1, 1]).
"""

import json
import os
import sys

import networkx as nx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'fixtures', 'network-statistics.json')
UNIT_ROUNDOFF = 2.0**-24
NODE_COUNT = 300
EDGE_COUNT = 700
SELF_LOOP_COUNT = 12
ISOLATED_TAIL = 20
COMMUNITY_COUNT = 9
RESOLUTION = 1.25


def draw_scene(seed, directed):
    generator = np.random.default_rng(seed)
    connected = NODE_COUNT - ISOLATED_TAIL
    block = connected // COMMUNITY_COUNT
    edges = set()
    while len(edges) < EDGE_COUNT:
        # Planted partition: 75% of edges inside one block of contiguous nodes, so the labels
        # carry structure and the modularity is far from 0.
        if generator.random() < 0.75:
            base = int(generator.integers(0, COMMUNITY_COUNT)) * block
            source, target = (base + int(value) for value in generator.integers(0, block, 2))
        else:
            source, target = (int(value) for value in generator.integers(0, connected, 2))
        if source == target:
            continue
        key = (source, target) if directed else (min(source, target), max(source, target))
        edges.add(key)
    loops = generator.choice(connected, SELF_LOOP_COUNT, replace=False).tolist()
    edge_list = sorted(edges) + [(int(node), int(node)) for node in loops]
    labels = [min(node // block, COMMUNITY_COUNT - 1) for node in range(NODE_COUNT)]
    for node in np.flatnonzero(generator.random(NODE_COUNT) < 0.15).tolist():
        labels[node] = int(generator.integers(0, COMMUNITY_COUNT))
    return edge_list, labels


def make_scene(seed, directed):
    edges, labels = draw_scene(seed, directed)
    graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_nodes_from(range(NODE_COUNT))
    graph.add_edges_from(edges)
    if directed:
        out_degrees = [graph.out_degree(node) for node in range(NODE_COUNT)]
        in_degrees = [graph.in_degree(node) for node in range(NODE_COUNT)]
        components = list(nx.weakly_connected_components(graph))
    else:
        out_degrees = [graph.degree(node) for node in range(NODE_COUNT)]
        in_degrees = out_degrees
        components = list(nx.connected_components(graph))
    communities = [
        {node for node in range(NODE_COUNT) if labels[node] == label}
        for label in sorted(set(labels))
    ]
    modularity = nx.community.modularity(graph, communities, resolution=RESOLUTION)
    return {
        'directed': directed,
        'nodeCount': NODE_COUNT,
        'edges': [list(edge) for edge in edges],
        'communities': labels,
        'resolution': RESOLUTION,
        'outDegrees': out_degrees,
        'inDegrees': in_degrees,
        'componentCount': len(components),
        'largestComponentSize': max(len(component) for component in components),
        'isolatedVertexCount': len(list(nx.isolates(graph))),
        'edgeCount': graph.number_of_edges(),
        'selfLoopCount': nx.number_of_selfloops(graph),
        'modularity': modularity,
        'tolerance': {'modularityAbsolute': (NODE_COUNT + 4) * UNIT_ROUNDOFF, 'counts': 'exact'}
    }


def main():
    fixture = {
        'generator': 'test/oracles/network-statistics.py',
        'versions': {
            'python': sys.version.split()[0],
            'networkx': nx.__version__,
            'numpy': np.__version__
        },
        'scenes': {
            'undirected': make_scene(7, False),
            'directed': make_scene(8, True)
        }
    }
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, 'w', encoding='utf-8') as file:
        json.dump(fixture, file, separators=(',', ':'))
        file.write('\n')
    print(f'wrote {OUTPUT}')


if __name__ == '__main__':
    main()
