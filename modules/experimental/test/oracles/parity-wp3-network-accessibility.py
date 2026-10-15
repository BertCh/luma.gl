# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""NetworkX + PySAL access reference for GPUNetworkAccessibility (python -I).

All 29,557 Chicago origins and two deterministic facilities, on the full directed
77,140-edge showcase graph. Costs are seconds from edgeTravelTime.bin, NOT chord
lengths. NetworkX single_source_dijkstra_path_length runs from each facility on
the reverse graph to obtain origin->facility costs. Shortest costs are rounded
once to f32 before passing to access, exactly the matrix GPUNetworkAccessibility
consumes. This tests the scorer, not GPU cost-matrix construction. Supply/demand
are deterministic synthetic weights, not claims about actual jobs/populations.

References: access.Access.weighted_catchment and Access.two_stage_fca, with
normalize=False. Facility ratios use access.fca.weighted_catchment on demand.
No custom implementation of reductions or 2SFCA. Threshold <=, unreachable costs
excluded, unrepresented origins filled with zero, no extra normalization. Decay
functions: none, exp(-beta*seconds), max(seconds,minimumCost)**(-beta). Parameters
and matrix inputs are f32; access accumulates in f64. Both matrix orientations
are checked against the same output. Synthetic case adds zero travel cost,
asymmetric reverse costs, a disconnected component, an island and exact cutoffs.

Profile S: atol=1e-6*max(1,output scale), rtol=1e-4. Cumulative totals are exact
integer-valued sums below 2**24. A facility denominator reduces at most 29,557
positive terms; no enlarged tolerance is assumed. Record measured GPU errors.
Inputs and each budget's full output arrays are separate JSON files, each below
1.5 MB. Input f32 values use their shortest round-trip decimal representation;
the reference remains full f64, and no output arrays are summarized or rounded.
"""

import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path

import networkx
import numpy
import pandas
from access import Access, fca

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DATA = ROOT / 'examples/deck/spatial-analysis-showcase/public/data/chicago-roads'
VERSIONS = {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'networkx', 'access']}
VERSIONS['python'] = platform.python_version()
REFERENCE = 'networkx.single_source_dijkstra_path_length; access.Access.weighted_catchment/two_stage_fca; access.fca.weighted_catchment'


def encode(value):
    if isinstance(value, dict):
        return {key: encode(entry) for key, entry in value.items()}
    if isinstance(value, (list, tuple, numpy.ndarray)):
        return [encode(entry) for entry in value]
    if isinstance(value, numpy.float32):
        return float(str(value)) if numpy.isfinite(value) else 'Infinity'
    if isinstance(value, (float, numpy.floating)):
        return float(value) if math.isfinite(value) else 'Infinity'
    if isinstance(value, numpy.integer):
        return int(value)
    return value


def writeFixture(name, fixture):
    fixture.update(generator=Path(__file__).name, versions=VERSIONS, reference=REFERENCE)
    output = HERE / 'fixtures' / f'parity-wp3-network-accessibility-{name}.json'
    output.write_text(json.dumps(encode(fixture), separators=(',', ':'), allow_nan=False) + '\n')
    assert output.stat().st_size <= 1_500_000, (output, output.stat().st_size)
    print(output.name, output.stat().st_size)


def generateCase(name, graph, facilities, supply, demand, source, configurations):
    originCount = len(demand)
    costs = numpy.full((len(facilities), originCount), numpy.inf, dtype=numpy.float32)
    for row, facility in enumerate(facilities):
        distances = networkx.single_source_dijkstra_path_length(graph.reverse(copy=False), facility, weight='weight')
        for origin, distance in distances.items():
            costs[row, origin] = distance
    writeFixture(f'{name}-input', dict(case=name, originCount=originCount, facilityNodes=facilities,
                 costs=costs.flatten(), supply=supply, demand=demand, source=source,
                 conventions=dict(direction='origin to facility; reverse-graph Dijkstra', units='seconds',
                                  costRounding='float32 after float64 Dijkstra', vertexMapping='original node ID equals output index',
                                  weights='deterministic synthetic opportunity supply and origin demand', seed='none')))
    originIndices = list(range(originCount))
    facilityIndices = list(range(len(facilities)))
    demandFrame = pandas.DataFrame({'demand': numpy.asarray(demand, dtype=numpy.float64)}, index=originIndices)
    supplyFrame = pandas.DataFrame({'supply': numpy.asarray(supply, dtype=numpy.float64)}, index=facilityIndices)
    demandFrame.index.name = 'geoid'
    supplyFrame.index.name = 'geoid'
    costFrame = pandas.DataFrame([dict(origin=origin, dest=facility, cost=float(costs[facility, origin]))
                                 for facility in facilityIndices for origin in originIndices
                                 if numpy.isfinite(costs[facility, origin])])
    for configuration in configurations:
        settings = dict(configuration)
        caseId = settings.pop('name')
        for key, value in settings.items():
            if key != 'decay':
                settings[key] = float(numpy.float32(value))
        threshold = settings['threshold']
        decay = settings.get('decay', 'none')
        beta = settings.get('beta', 0)
        minimumCost = settings.get('minimumCost', 1)
        if decay == 'none':
            weightFunction = lambda cost: 1.0
        elif decay == 'exponential':
            weightFunction = lambda cost: math.exp(-beta * cost)
        else:
            weightFunction = lambda cost: max(cost, minimumCost) ** (-beta)
        analysis = Access(demandFrame.copy(), 'demand', supplyFrame.copy(), 'supply',
                          cost_df=costFrame, cost_origin='origin', cost_dest='dest', cost_name='cost')
        cumulative = analysis.weighted_catchment(name='cumulative', max_cost=threshold, normalize=False).iloc[:, 0].reindex(originIndices).fillna(0)
        gravity = analysis.weighted_catchment(name='gravity', max_cost=threshold, weight_fn=weightFunction, normalize=False).iloc[:, 0].reindex(originIndices).fillna(0)
        catchment = analysis.two_stage_fca(name='catchment', max_cost=threshold, weight_fn=weightFunction, normalize=False).iloc[:, 0].reindex(originIndices).fillna(0)
        totals = fca.weighted_catchment(demandFrame.copy(), costFrame, max_cost=threshold,
                                       loc_index=True, loc_value='demand', weight_fn=weightFunction)
        ratios = supplyFrame['supply'].divide(totals).reindex(facilityIndices).fillna(0)
        assert numpy.isfinite(ratios).all(), 'Fixture avoids the access/GPU zero-demand convention mismatch'
        expected = dict(cumulative=cumulative.tolist(), gravity=gravity.tolist(), catchment=catchment.tolist(), ratios=ratios.tolist())
        tolerances = {metric: dict(absolute=0 if metric == 'cumulative' else 1e-6 * max(1, max(abs(value) for value in values)),
                                   relative=0 if metric == 'cumulative' else 1e-4)
                      for metric, values in expected.items()}
        writeFixture(f'{name}-{caseId}', dict(case=f'{name}-{caseId}', input=f'{name}-input', settings=settings,
                     expected=expected, tolerances=tolerances, profile='S',
                     metadata=dict(originCount=originCount, facilityCount=len(facilities),
                                   finiteCosts=int(numpy.isfinite(costs).sum()), invalidMasks='unreachable costs excluded; unrepresented origins output zero')))


def main():
    graph = networkx.DiGraph()
    graph.add_nodes_from(range(8))
    edges = [(0, 1, 0), (1, 0, 3), (1, 2, 2), (2, 3, 4), (3, 1, 1), (4, 3, 2), (3, 4, 7), (5, 6, 3), (6, 5, 1)]
    graph.add_weighted_edges_from(edges)
    generateCase('synthetic', graph, [1, 4], [10, 9], [1, 3, 4, 8, 5, 2, 7, 6],
                 dict(edges=edges, nodeCount=8, description='asymmetric directed network with zero cost, disconnected component and island'),
                 [dict(name='zero', threshold=0), dict(name='boundary', threshold=3),
                  dict(name='exponential', threshold=4, decay='exponential', beta=0.125),
                  dict(name='power', threshold=8, decay='power', beta=1.5, minimumCost=1.5),
                  dict(name='unlimited', threshold=math.inf)])
    positions = numpy.fromfile(DATA / 'nodes.bin', dtype='<f4').reshape(-1, 2)
    sources = numpy.fromfile(DATA / 'edgeSource.bin', dtype='<u4')
    targets = numpy.fromfile(DATA / 'edgeTarget.bin', dtype='<u4')
    travelTimes = numpy.fromfile(DATA / 'edgeTravelTime.bin', dtype='<f4')
    graph = networkx.DiGraph()
    graph.add_nodes_from(range(len(positions)))
    for source, target, cost in zip(sources, targets, travelTimes):
        # Parallel arcs are retained in the source hash; shortest paths choose the cheapest.
        previous = graph.get_edge_data(int(source), int(target), default={'weight': math.inf})['weight']
        graph.add_edge(int(source), int(target), weight=min(previous, float(cost)))
    facilities = [int(numpy.argmin(numpy.linalg.norm(positions.astype(numpy.float64) - location, axis=1)))
                  for location in [(-87.63, 41.88), (-87.60, 41.80)]]
    demand = (1 + numpy.arange(len(positions)) % 31).tolist()
    source = dict(dataset=str(DATA.relative_to(ROOT)), fullNodeCount=len(positions), fullEdgeCount=len(sources),
                  sha256={name: hashlib.sha256((DATA / name).read_bytes()).hexdigest()
                          for name in ['manifest.json', 'nodes.bin', 'edgeSource.bin', 'edgeTarget.bin', 'edgeTravelTime.bin']},
                  attribution='OpenStreetMap contributors; ODbL 1.0', facilitySelection='nearest stored lon/lat node to (-87.63,41.88), (-87.60,41.80)')
    generateCase('chicago', graph, facilities, [1800, 850], demand, source,
                 [dict(name='zero', threshold=0), dict(name='five-minutes', threshold=300),
                  dict(name='exponential', threshold=900, decay='exponential', beta=0.005),
                  dict(name='power', threshold=math.inf, decay='power', beta=1.25, minimumCost=5),
                  dict(name='unlimited', threshold=math.inf)])


if __name__ == '__main__':
    main()
