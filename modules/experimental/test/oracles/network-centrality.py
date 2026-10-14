# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Reference betweenness and closeness centrality for GPUNetworkCentrality from networkx.

Writes fixtures/network-centrality.json next to this script (consumed by
test/gpu-network/network-centrality/network-centrality.node.spec.ts, which runs the TypeScript mirror
of the GPU schedule against it, and by gpu-network-centrality.spec.ts). Run from anywhere with
    <venv>/bin/python -I modules/experimental/test/oracles/network-centrality.py
in a venv with numpy and networkx (versions are recorded in the fixture).

Conventions (GPUNetworkCentrality measures outward from its pivots):
- Betweenness, all pivots: nx.betweenness_centrality(G, weight='weight', normalized=False) on a
  DiGraph (no 1/2 factor, endpoints excluded).
- Betweenness, sampled pivots: nx.betweenness_centrality_subset(G, sources=pivots, targets=list(G),
  weight='weight', normalized=False). The fixture stores the unscaled value; the GPU spec passes
  betweennessScale = nodeCount / pivotCount and compares against scale * value.
- Betweenness, radius R: networkx has no radius option, so the script runs a short explicit Brandes
  (2001) over nx.single_source_dijkstra(G, s, cutoff=R) and asserts it equals networkx when R is
  infinite (relative 1e-9). Paths longer than R are not counted and nodes beyond R are not part of any
  path (the GPU searches with costLimit = R, inclusive).
- Closeness of pivot s: (r - 1) / S with r the reached count (pivot included) and S the sum of
  costs, 0 when S == 0. networkx.closeness_centrality on a DiGraph uses incoming distances, so the
  script compares against nx.closeness_centrality(G.reverse(), u=s, distance='weight',
  wf_improved=False), which measures outward distances. Harmonic closeness is the sum of 1/cost
  over reached nodes with cost > 0, compared against nx.harmonic_centrality(G.reverse(), [s],
  distance='weight') for the same reason.
- Without a radius the closeness values equal networkx; with a radius they come from the same
  formula over the nodes within R (asserted against a networkx cutoff search).

Scenes (all integer weights in [1, 9], so f32 costs, tight-edge tests and path counts are exact; no
zero weights, which the GPU excludes from path counts):
- 'hand': 9 nodes with several equal-cost shortest paths (sigma up to 3), a one-way edge, a cycle
  and a node with no in-edges.
- 'grid': an 8 x 8 grid with both directions of every street and street weights in [1, 3], so many
  ties.
- 'random': 40 nodes, 160 distinct directed edges, weights in [1, 9].

Tolerance (stored per scene). Costs, tightness and sigma are exact: the generator asserts that every
sigma is below 2^24. Only delta (and sums of delta) round. For a node v, delta(v) is a sum over its
k tight out-edges of sigma(v) / sigma(w) * (1 + delta(w)): WGSL guarantees a division within
2.5 ULP (5 u), the addition and the multiplication round once each (2 u), and adding k positive
terms rounds at most k times (k u), with u = 2^-24 and every term positive, so no cancellation: the
relative error of delta(v) is at most the largest relative error of a successor plus (7 + k) u.
Along the longest tight chain of h hops with largest tight out-degree K this gives
deltaRelative = h (7 + K) u (an FMA contraction does not exceed it). The betweenness of a node adds one delta per pivot and then scales: adding P positive
values of relative error e gives e + P u, and one multiplication by the scale adds u, so
betweennessRelative = deltaRelative + (P + 2) u. The generator takes h and K as the maxima over every
pivot, with and without radius. Closeness is one division of an exact integer sum (S below 2^24 is
asserted): closenessRelative = 2 u. Harmonic closeness adds r values 1/c, each rounded once, in any
order (a tree reduction on the GPU): harmonicRelative = (r + 2) u with r the largest reached count.
"""

import json
import math
import os
import sys

import networkx as nx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'fixtures', 'network-centrality.json')
UNIT_ROUNDOFF = 2.0**-24
INFINITY = float('inf')


def hand_edges():
    return [
        (0, 1, 1),
        (0, 2, 1),
        (1, 3, 1),
        (2, 3, 1),
        (3, 4, 2),
        (3, 5, 2),
        (4, 6, 1),
        (5, 6, 1),
        (1, 4, 3),
        (6, 7, 1),
        (7, 0, 2),
        (1, 0, 1),
        (8, 0, 1)
    ]


def grid_edges():
    generator = np.random.default_rng(8)
    width = 8
    edges = []
    for y in range(width):
        for x in range(width):
            node = y * width + x
            for neighbor in ([node + 1] if x + 1 < width else []) + (
                [node + width] if y + 1 < width else []
            ):
                weight = int(generator.integers(1, 4))
                edges.append((node, neighbor, weight))
                edges.append((neighbor, node, weight))
    return width * width, edges


def random_edges():
    generator = np.random.default_rng(40)
    node_count = 40
    pairs = set()
    while len(pairs) < 160:
        source, target = (int(value) for value in generator.integers(0, node_count, 2))
        if source != target:
            pairs.add((source, target))
    edges = [(source, target, int(generator.integers(1, 10))) for source, target in sorted(pairs)]
    return node_count, edges


def make_graph(node_count, edges):
    graph = nx.DiGraph()
    graph.add_nodes_from(range(node_count))
    for source, target, weight in edges:
        graph.add_edge(source, target, weight=weight)
    return graph


def brandes(graph, pivots, radius):
    """Explicit Brandes (2001) with Dijkstra and an inclusive cutoff.

    Returns (betweenness, statistics). Statistics: largest sigma, longest tight chain (hops), and
    largest tight out-degree over the pivots.
    """
    betweenness = {node: 0.0 for node in graph}
    maximum_sigma = 0
    maximum_depth = 0
    maximum_out_degree = 0
    cutoff = None if radius is None else radius
    for pivot in pivots:
        distances = nx.single_source_dijkstra_path_length(graph, pivot, cutoff=cutoff, weight='weight')
        order = sorted(distances, key=lambda node: (distances[node], node))
        sigma = {node: 0 for node in order}
        sigma[pivot] = 1
        predecessors = {node: [] for node in order}
        successors_count = {node: 0 for node in order}
        for source in order:
            for target in graph.successors(source):
                weight = graph[source][target]['weight']
                if target in distances and distances[source] + weight == distances[target]:
                    sigma[target] += sigma[source]
                    predecessors[target].append(source)
                    successors_count[source] += 1
        depth = {node: 0 for node in order}
        for node in order:
            for source in predecessors[node]:
                depth[node] = max(depth[node], depth[source] + 1)
        delta = {node: 0.0 for node in order}
        for node in reversed(order):
            for source in predecessors[node]:
                delta[source] += sigma[source] / sigma[node] * (1.0 + delta[node])
            if node != pivot:
                betweenness[node] += delta[node]
        maximum_sigma = max(maximum_sigma, max(sigma.values()))
        maximum_depth = max(maximum_depth, max(depth.values()))
        maximum_out_degree = max(maximum_out_degree, max(successors_count.values()))
    return betweenness, {
        'maximumSigma': maximum_sigma,
        'maximumDepth': maximum_depth,
        'maximumOutDegree': maximum_out_degree
    }


def close_to(left, right, relative=1e-9):
    return abs(left - right) <= relative * max(abs(left), abs(right), 1e-300)


def closeness_rows(graph, pivots, radius):
    """Reached count, cost sum, closeness and harmonic closeness per pivot (outward distances)."""
    cutoff = None if radius is None else radius
    reached_counts = []
    cost_sums = []
    closeness = []
    harmonic = []
    for pivot in pivots:
        distances = nx.single_source_dijkstra_path_length(graph, pivot, cutoff=cutoff, weight='weight')
        total = sum(distances.values())
        reached = len(distances)
        reached_counts.append(reached)
        cost_sums.append(total)
        closeness.append((reached - 1.0) / total if total > 0 else 0.0)
        harmonic.append(sum(1.0 / cost for cost in distances.values() if cost > 0))
    return reached_counts, cost_sums, closeness, harmonic


def make_scene(node_count, edges, radius, pivot_count):
    graph = make_graph(node_count, edges)
    reverse = graph.reverse()
    nodes = list(range(node_count))
    # Sampled pivots: every k-th node, so the scale nodeCount / pivotCount is exactly representable.
    stride = node_count // pivot_count
    sampled = nodes[::stride][:pivot_count]
    assert node_count % pivot_count == 0 and len(sampled) == pivot_count

    exact_reference = nx.betweenness_centrality(graph, weight='weight', normalized=False)
    exact, exact_statistics = brandes(graph, nodes, None)
    for node in nodes:
        assert close_to(exact[node], exact_reference[node]), (node, exact[node], exact_reference[node])

    sampled_reference = nx.betweenness_centrality_subset(
        graph, sources=sampled, targets=nodes, weight='weight', normalized=False
    )
    sampled_values, sampled_statistics = brandes(graph, sampled, None)
    for node in nodes:
        assert close_to(sampled_values[node], sampled_reference[node]), node

    bounded, bounded_statistics = brandes(graph, nodes, radius)
    bounded_sampled, bounded_sampled_statistics = brandes(graph, sampled, radius)

    closeness_networkx = [
        nx.closeness_centrality(reverse, u=pivot, distance='weight', wf_improved=False)
        for pivot in nodes
    ]
    harmonic_networkx = [
        nx.harmonic_centrality(reverse, nbunch=[pivot], distance='weight')[pivot] for pivot in nodes
    ]
    reached, sums, closeness, harmonic = closeness_rows(graph, nodes, None)
    for index in nodes:
        assert close_to(closeness[index], closeness_networkx[index]), index
        assert close_to(harmonic[index], harmonic_networkx[index]), index
    reached_bounded, sums_bounded, closeness_bounded, harmonic_bounded = closeness_rows(
        graph, nodes, radius
    )

    statistics = [
        exact_statistics,
        sampled_statistics,
        bounded_statistics,
        bounded_sampled_statistics
    ]
    maximum_sigma = max(entry['maximumSigma'] for entry in statistics)
    assert maximum_sigma < 2**24
    assert max(sums + sums_bounded) < 2**24
    maximum_depth = max(entry['maximumDepth'] for entry in statistics)
    maximum_out_degree = max(entry['maximumOutDegree'] for entry in statistics)
    delta_relative = maximum_depth * (7 + maximum_out_degree) * UNIT_ROUNDOFF
    # Borderline guard: with integer costs the radius test is exact; nothing to check beyond that.
    assert float(radius).is_integer()
    return {
        'nodeCount': node_count,
        'edges': [[source, target, weight] for source, target, weight in edges],
        'radius': radius,
        'sampledPivots': sampled,
        'sampledScale': node_count / pivot_count,
        'betweenness': {
            'exact': [exact_reference[node] for node in nodes],
            'sampled': [sampled_reference[node] for node in nodes],
            'radius': [bounded[node] for node in nodes],
            'radiusSampled': [bounded_sampled[node] for node in nodes]
        },
        'closeness': {
            'unbounded': {
                'reachedCounts': reached,
                'closeness': closeness_networkx,
                'harmonicCloseness': harmonic_networkx
            },
            'radius': {
                'reachedCounts': reached_bounded,
                'closeness': closeness_bounded,
                'harmonicCloseness': harmonic_bounded
            }
        },
        'maximumSigma': maximum_sigma,
        'maximumDepth': maximum_depth,
        'maximumOutDegree': maximum_out_degree,
        'tolerance': {
            'deltaRelative': delta_relative,
            'betweennessRelative': delta_relative + (node_count + 2) * UNIT_ROUNDOFF,
            'closenessRelative': 2 * UNIT_ROUNDOFF,
            'harmonicRelative': (max(reached) + 2) * UNIT_ROUNDOFF
        }
    }


def main():
    grid_node_count, grid = grid_edges()
    random_node_count, random_graph = random_edges()
    fixture = {
        'generator': 'test/oracles/network-centrality.py',
        'versions': {
            'python': sys.version.split()[0],
            'networkx': nx.__version__,
            'numpy': np.__version__
        },
        'scenes': {
            'hand': make_scene(9, hand_edges(), 4, 3),
            'grid': make_scene(grid_node_count, grid, 6, 16),
            'random': make_scene(random_node_count, random_graph, 12, 8)
        }
    }
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, 'w', encoding='utf-8') as file:
        json.dump(fixture, file, separators=(',', ':'))
        file.write('\n')
    print(f'wrote {OUTPUT}')
    for name, scene in fixture['scenes'].items():
        print(name, scene['maximumSigma'], scene['maximumDepth'], scene['maximumOutDegree'])


if __name__ == '__main__':
    main()
