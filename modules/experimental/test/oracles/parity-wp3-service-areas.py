# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""NetworkX directed single-source Dijkstra reference for GPUNetworkServiceAreas.

Facility -> node direction, inclusive cutoff, smallest facility ROW on exact ties.
Full Chicago graph, all 29,557 nodes and 77,140 source arcs, stable original IDs.
Travel times quantized explicitly to nearest 1/8 second before f32 upload. This
keeps path accumulation exact (all tested costs < 2**21 seconds), isolating routing
and facility ties from order-dependent f32 path rounding. This is a quantized
travel-time study, not unrounded-time parity. Synthetic graph has zero-cost
plateaus, ties, duplicate facilities, asymmetric edges and an unreachable island.

Independent distances and predecessor sets come from NetworkX, not GPU logic.
Allowed predecessors are independently enumerated tight incoming edges in the
winning facility's NetworkX distance map; GPU predecessor chains must also be
acyclic and terminate at that facility. All outputs are compared, including
infinite masks, labels, per-facility counts and cost sums. Profile S for sums,
exact distances/labels/counts on these dyadic inputs. This class does NOT output
service-edge fractions or opportunity totals; those belong to other contributors.
"""
import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path

import networkx
import numpy

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[3] / 'examples/deck/spatial-analysis-showcase/public/data/chicago-roads'
VERSIONS = {name: importlib.metadata.version(name) for name in ['numpy', 'networkx']}
VERSIONS['python'] = platform.python_version()


def encode(value):
    if isinstance(value, dict):
        return {key: encode(entry) for key, entry in value.items()}
    if isinstance(value, (list, tuple, numpy.ndarray)):
        return [encode(entry) for entry in value]
    if isinstance(value, numpy.integer):
        return int(value)
    if isinstance(value, (float, numpy.floating)):
        return float(value) if math.isfinite(value) else 'Infinity'
    return value


def write(name, value):
    value.update(generator=Path(__file__).name, versions=VERSIONS,
                 reference='networkx.dijkstra_predecessor_and_distance', conventions=__doc__)
    path = HERE / 'fixtures' / f'parity-wp3-service-areas-{name}.json'
    path.write_text(json.dumps(encode(value), separators=(',', ':'), allow_nan=False) + '\n')
    assert path.stat().st_size <= 1_500_000, path.stat().st_size
    print(path.name, path.stat().st_size, flush=True)


def generate(name, nodeCount, edges, facilities, initialCosts, limits, source):
    rows = [[] for _ in range(nodeCount)]
    graph = networkx.DiGraph()
    graph.add_nodes_from(range(nodeCount))
    for first, second, weight in edges:
        rows[first].append((second, weight))
        prior = graph.get_edge_data(first, second, default={'weight': math.inf})['weight']
        graph.add_edge(first, second, weight=min(prior, weight))
    offsets, neighbors, weights = [0], [], []
    for row in rows:
        for neighbor, weight in sorted(row):
            neighbors.append(neighbor)
            weights.append(weight)
        offsets.append(len(neighbors))
    write(f'{name}-input', dict(case=name, nodeCount=nodeCount, offsets=offsets, neighbors=neighbors,
          weights=weights, facilities=facilities, initialCosts=initialCosts, source=source))
    distances, predecessors = [], []
    for facility, initial in zip(facilities, initialCosts):
        previous, costs = networkx.dijkstra_predecessor_and_distance(graph, facility)
        distances.append({node: cost + initial for node, cost in costs.items()})
        predecessors.append(previous)
    for budgetName, budget in limits:
        costs, assignments, allowedPredecessors = [], [], []
        for node in range(nodeCount):
            distance, assignment = min((costsByFacility.get(node, math.inf), row) for row, costsByFacility in enumerate(distances))
            if distance > budget or not math.isfinite(distance):
                distance, assignment = math.inf, 0xffffffff
            costs.append(distance)
            assignments.append(assignment)
            allowedPredecessors.append(predecessors[assignment].get(node, []) if assignment != 0xffffffff else [])
        counts = [assignments.count(row) for row in range(len(facilities))]
        sums = [math.fsum(cost for cost, assignment in zip(costs, assignments) if assignment == row) for row in range(len(facilities))]
        assert max((cost for cost in costs if math.isfinite(cost)), default=0) < 2**21
        write(f'{name}-{budgetName}', dict(case=f'{name}-{budgetName}', budget=budget,
              costs=costs, assignments=assignments, allowedPredecessors=allowedPredecessors,
              counts=counts, sums=sums, tolerance=dict(absolute=1e-6 * max(1, max(sums)), relative=1e-4)))


def main():
    generate('synthetic', 12,
        [(5,4,2),(0,4,2),(4,3,0),(3,4,0),(4,6,1),(8,7,1),(7,6,2),
         (1,2,1),(0,2,7),(9,10,1),(10,9,1)],
        [5,0,8,1,5], [0,0,0,5,0], [('zero',0),('boundary',3),('unlimited',math.inf)],
        dict(description='synthetic directed ties, plateau, duplicate facility, disconnected nodes'))
    nodes = numpy.fromfile(DATA/'nodes.bin', dtype='<f4').reshape(-1,2)
    sources = numpy.fromfile(DATA/'edgeSource.bin', dtype='<u4')
    targets = numpy.fromfile(DATA/'edgeTarget.bin', dtype='<u4')
    times = numpy.fromfile(DATA/'edgeTravelTime.bin', dtype='<f4')
    weights = numpy.rint(times.astype(numpy.float64)*8)/8
    facilities = [int(numpy.argmin(numpy.linalg.norm(nodes.astype(numpy.float64)-location,axis=1)))
                  for location in [(-87.635,41.885),(-87.65,41.93),(-87.63,41.80)]]
    generate('chicago', len(nodes), [(int(first),int(second),float(weight)) for first,second,weight in zip(sources,targets,weights)],
        facilities, [0, 15, 30], [('five-minutes',300),('fifteen-minutes',900),('unlimited',math.inf)],
        dict(hashes={name:hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in
             ['nodes.bin','edgeSource.bin','edgeTarget.bin','edgeTravelTime.bin']},
             attribution=json.loads((DATA/'manifest.json').read_text()).get('properties'),
             quantizationSeconds=0.125, maximumQuantizationError=float(max(abs(weights-times))),
             nodeIds='all original vertex IDs, unchanged', edgeCount=len(sources)))


if __name__ == '__main__':
    main()
