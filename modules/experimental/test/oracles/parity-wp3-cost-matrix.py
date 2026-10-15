# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""NetworkX multi-source Dijkstra matrices on the full directed Chicago network.

Facility -> all-node costs and reverse-graph all-node -> facility costs; positive
initial costs represented by one external super-source per row. Inclusive cutoff,
invalid starts ignored, duplicate starts minimized, empty rows infinite. All dense
entries and a compact target list with duplicates/invalid IDs are retained. Chicago
uses round-2 explicit 1/8-second travel-time quantization (not raw-time parity),
allowing exact f32 positive path sums below 2**21. Synthetic zero-cost cycle,
parallel/asymmetric arcs, isolated vertex, duplicate starts and threshold ties.
Three Chicago rows / five synthetic rows exercise a final short two-lane batch.
GPUNetworkCostMatrix publishes costs and convergence, not predecessors or paths;
this fixture does not claim GPU predecessor or path-extraction coverage. Profile L,
with exact costs possible for these dyadic inputs; exact finite/infinite masks.
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
VERSIONS = {'networkx': importlib.metadata.version('networkx'), 'python': platform.python_version()}


def encode(value):
    if isinstance(value, dict):
        return {key: encode(entry) for key, entry in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(entry) for entry in value]
    return 'Infinity' if isinstance(value, float) and math.isinf(value) else value


def write(output, name, fixture):
    fixture.update(versions=VERSIONS, generator=Path(__file__).name, conventions=__doc__,
                   reference='networkx.multi_source_dijkstra_path_length')
    path = output/f'parity-wp3-cost-matrix-{name}.json'
    path.write_text(json.dumps(encode(fixture), separators=(',', ':'), allow_nan=False)+'\n')
    assert path.stat().st_size <= 1_500_000
    print(path.name, path.stat().st_size, flush=True)


def generate(output, name, count, edges, starts, initial, rowCount, budgets, source):
    graph = networkx.MultiDiGraph()
    graph.add_nodes_from(range(count))
    graph.add_weighted_edges_from(edges)
    offsets, neighbors, weights = [0], [], []
    for node in range(count):
        for _, target, attributes in graph.out_edges(node, data=True):
            neighbors.append(target)
            weights.append(attributes['weight'])
        offsets.append(len(neighbors))
    targets = [0, count - 1, starts[0], starts[0], count, 0xffffffff] + list(range(0, count, max(1, count//128)))
    write(output, name+'-input', dict(case=name, nodeCount=count, offsets=offsets, neighbors=neighbors,
          weights=weights, startNodes=starts, startCosts=initial, startRows=[index//2 for index in range(len(starts))],
          startsPerRow=2, rowCount=rowCount, targets=targets, source=source))
    rows = []
    for row in range(rowCount):
        augmented = graph.copy()
        augmented.add_node(count)
        for index, node in enumerate(starts):
            if index//2 == row and 0 <= node < count:
                augmented.add_edge(count, node, weight=initial[index])
        distances = networkx.multi_source_dijkstra_path_length(augmented, [count], weight='weight')
        values = [distances.get(node, math.inf) for node in range(count)]
        assert max((value for value in values if math.isfinite(value)), default=0) < 2**21
        rows.append(values)
    for budgetName, budget in budgets:
        dense = [[value if value <= budget else math.inf for value in row] for row in rows]
        compact = [[row[target] if target < count else math.inf for target in targets] for row in dense]
        write(output, name+'-'+budgetName, dict(case=name+'-'+budgetName, budget=budget,
              costs=[value for row in dense for value in row], compact=[value for row in compact for value in row]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=HERE/'fixtures')
    output = parser.parse_args().output_dir
    output.mkdir(parents=True, exist_ok=True)
    generate(output, 'synthetic', 9, [(0,1,1),(0,1,4),(1,2,0),(2,1,0),(2,3,2),
             (4,3,1),(3,5,0.5),(5,4,7),(6,7,1),(7,6,3)],
             [0,0,4,2,99,99,7,7], [0,3,2,1,0,0,0,0], 5,
             [('zero',0),('boundary',3),('unlimited',math.inf)],
             dict(description='parallel arcs, zero-cost cycle, asymmetric component and island'))
    path = HERE/'fixtures/parity-wp3-service-areas-chicago-input.json'
    source = json.loads(path.read_text())
    edges = [(node, source['neighbors'][edge], source['weights'][edge])
             for node in range(source['nodeCount'])
             for edge in range(source['offsets'][node], source['offsets'][node+1])]
    facilities = source['facilities']
    for direction in ['forward', 'reverse']:
        generate(output, 'chicago-'+direction, source['nodeCount'],
                 edges if direction == 'forward' else [(second,first,weight) for first,second,weight in edges],
                 [facilities[0],facilities[1],facilities[2],facilities[2]], [0,15,30,45], 3,
                 [('five-minutes',300),('unlimited',math.inf)],
                 {**source['source'], 'direction': direction,
                  'inputFixtureSha256': hashlib.sha256(path.read_bytes()).hexdigest()})


if __name__ == '__main__':
    main()
