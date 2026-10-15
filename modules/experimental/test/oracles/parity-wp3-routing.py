# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""NetworkX Dijkstra, tight-edge DAG/plateau BFS, and complete extracted paths.

Full directed Chicago CSR reused from the round-3 inputs, with their explicit
1/8-second quantization and hashes. This is NOT raw travel-time parity. Positive
path sums remain below 2**21, hence all sums of these dyadic weights are exact f32.
Synthetic parallel arcs, exact alternative routes, zero-cost cycle, islands,
duplicate/invalid starts, positive initial costs, inclusive limits, stable IDs.
NetworkX supplies distances and tight-edge predecessor sets. The documented tie
adapter chooses the minimum strict predecessor, or the minimum previous-level
predecessor in NetworkX's equal-cost multi-source BFS. This is a convention
adapter, not a port of the GPU frontier algorithm. NetworkX shortest_path on the
resulting independently certified forest supplies complete node paths. Edge ties
select the smallest CSR edge of the shortest-weight parallel edges. E for IDs,
all dyadic costs and infinite masks (stronger than L); no statistical summaries.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path

import networkx

HERE = Path(__file__).resolve().parent
NONE = 0xffffffff
VERSIONS = {'networkx': importlib.metadata.version('networkx'), 'python': platform.python_version()}


def encode(value):
    if isinstance(value, dict):
        return {key: encode(entry) for key, entry in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(entry) for entry in value]
    return 'Infinity' if isinstance(value, float) and math.isinf(value) else value


def write(output, name, fixture):
    fixture.update(versions=VERSIONS, generator=Path(__file__).name, conventions=__doc__,
                   reference='networkx.dijkstra_predecessor_and_distance; multi_source_dijkstra_path_length; shortest_path')
    path = output/f'parity-wp3-routing-{name}.json'
    path.write_text(json.dumps(encode(fixture), separators=(',', ':'), allow_nan=False)+'\n')
    assert path.stat().st_size <= 1_500_000, (path, path.stat().st_size)
    print(path.name, path.stat().st_size, flush=True)


def generate(output, name, source, queries):
    count = source['nodeCount']
    offsets, neighbors, weights = (source[key] for key in ['offsets','neighbors','weights'])
    graph = networkx.MultiDiGraph()
    graph.add_nodes_from(range(count))
    for node in range(count):
        for edge in range(offsets[node], offsets[node+1]):
            graph.add_edge(node, neighbors[edge], key=edge, weight=weights[edge])
    write(output, name+'-input', dict(case=name, nodeCount=count, offsets=offsets,
          neighbors=neighbors, weights=weights, source=source['source'],
          nodeIdMapping='3*(nodeCount-1-row)+7', edgeIdMapping='5*(edgeCount-1-row)+11'))
    for queryName, starts, initial, limit in queries:
        augmented = graph.copy()
        for node, cost in zip(starts,initial):
            if node < count:
                augmented.add_edge(count, node, weight=cost)
        augmented.add_node(count)
        candidates, distances = networkx.dijkstra_predecessor_and_distance(augmented, count, cutoff=limit)
        costs = [distances.get(node, math.inf) for node in range(count)]
        assert max((value for value in costs if math.isfinite(value)), default=0) < 2**21
        roots = {node for node, cost in zip(starts,initial) if node < count and costs[node] == cost}
        strict = {node: sorted(parent for parent in candidates.get(node, [])
                               if parent != count and costs[parent] < costs[node]) for node in range(count)}
        equal = networkx.DiGraph()
        equal.add_nodes_from(range(count))
        for node in range(count):
            equal.add_edges_from((parent,node) for parent in candidates.get(node, [])
                                if parent != count and costs[parent] == costs[node])
        levelRoots = roots | {node for node in range(count) if strict[node]}
        levels = (networkx.multi_source_dijkstra_path_length(equal, sorted(levelRoots), weight=None)
                  if levelRoots else {})
        predecessors = []
        for node in range(count):
            eligible = [] if node in roots else strict[node]
            if not eligible and node not in roots and node in levels:
                eligible = [parent for parent in equal.predecessors(node)
                            if levels.get(parent, count)+1 == levels[node]]
            predecessors.append(min(eligible, default=NONE))
        forest = networkx.DiGraph()
        forest.add_nodes_from(range(count+1))
        forest.add_edges_from((parent,node) for node,parent in enumerate(predecessors) if parent != NONE)
        forest.add_edges_from((count,root) for root in roots)
        assert networkx.is_directed_acyclic_graph(forest)
        forestPaths = networkx.single_source_shortest_path(forest, count)
        assert {node for node in forestPaths if node != count} == {node for node,cost in enumerate(costs) if math.isfinite(cost)}
        targets = list(range(count)) if count < 100 else list(range(0,count,max(1,count//64)))
        targets += starts[:2] + [count-1, count, NONE]
        paths = [forestPaths.get(node, [count])[1:] for node in targets]
        edgePaths = []
        for path in paths:
            entries = [min(edge for edge, attributes in graph[first][second].items()
                           if costs[first]+attributes['weight'] == costs[second])
                       for first,second in zip(path,path[1:])]
            if path:
                assert sum(weights[edge] for edge in entries) + costs[path[0]] == costs[path[-1]]
            edgePaths.append(entries)
        breaks = [0,3,30,300,900,3600]
        bands = [next((index for index,bound in enumerate(breaks) if cost <= bound),NONE) for cost in costs]
        write(output, name+'-'+queryName, dict(case=name+'-'+queryName, startNodes=starts,
              startCosts=initial, limit=limit, costs=costs, predecessors=predecessors,
              breaks=breaks, bands=bands, bandCounts=[bands.count(index) for index in range(len(breaks))],
              targets=targets, paths=paths, edgePaths=edgePaths,
              pathCosts=[costs[node] if node < count else math.inf for node in targets],
              maximumPathLength=max(map(len, paths),default=1)+1,
              tightAlternativeCount=sum(len(value)>1 for value in candidates.values())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=HERE/'fixtures')
    output = parser.parse_args().output_dir
    output.mkdir(parents=True, exist_ok=True)
    for name in ['synthetic','chicago-forward','chicago-reverse']:
        path = HERE/f'fixtures/parity-wp3-cost-matrix-{name}-input.json'
        source = json.loads(path.read_text())
        source['source']['inputFixtureSha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        starts = source['startNodes'][:2]
        if name == 'synthetic':
            # Add a strict-cost diamond and an exactly tied parallel edge, while
            # retaining node 2's zero-cost plateau. This tests both tie policies.
            edges = [(node, source['neighbors'][edge], source['weights'][edge])
                     for node in range(source['nodeCount'])
                     for edge in range(source['offsets'][node], source['offsets'][node+1])]
            edges += [(0, 3, 3), (1, 3, 2), (0, 1, 1)]
            source['offsets'], source['neighbors'], source['weights'] = [0], [], []
            for node in range(source['nodeCount']):
                for first, second, weight in edges:
                    if first == node:
                        source['neighbors'].append(second)
                        source['weights'].append(weight)
                source['offsets'].append(len(source['neighbors']))
            source['source']['addedEdges'] = [[0, 3, 3], [1, 3, 2], [0, 1, 1]]
            queries = [('single',[0],[0],math.inf),('multi',[0,0,4,99],[0,3,1,0],3),
                       ('empty',[99,NONE],[0,0],math.inf)]
        else:
            queries = [('single',[starts[0]],[0],math.inf),('multi',starts,[0,15],300)]
        generate(output,name,source,queries)


if __name__ == '__main__':
    main()
