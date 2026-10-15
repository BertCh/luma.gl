# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Spaghetti GlobalAutoK reference, run with the designated Python -I.

Native planar meters, float32 AEQD coordinates centered on downtown Chicago.
Undirected unique endpoint chords (not OSM polyline/travel-time costs), both CSR
orientations. Spaghetti snaps original f32 event positions independently. Its
Euclidean arc lengths are rounded to f32 before routing and simulation; endpoint
offsets remain Spaghetti's f64 geometric measurements. Radii span zero to the
maximum in equally spaced bands (including fractional metres in showcase cases),
strict <, unordered pairs, no self pairs, n**2 normalization, no edge correction.
Network length is the f32-rounded sum of unique f32 arc lengths; the reference
retains its f64 sum and records both plus event intensity.

Real input: deterministic spatially stratified Chicago places in a 2km square,
roads in a surrounding 3km square, largest connected component only. This is a
window study, not full-Chicago routing. All selected observations and radii are
retained, never selected by observed GPU output. Synthetic exact-boundary and
coincident events exercise strict <. Monte Carlo: 1024 Spaghetti uniform-length
patterns (NumPy seed 20261015), compared to 3 independent GPU seeds with a
family-wise two-sample DKW bound, not same-seed equality. Raw simulation arrays
are retained; Spaghetti significance-scaled envelopes are NOT GPU min/mean/max.
"""
import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path

import geopandas
import networkx
import numpy
import pyproj
import shapely
import spaghetti

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DATA = ROOT / 'examples/deck/spatial-analysis-showcase/public/data'
VERSIONS = {name: importlib.metadata.version(name) for name in ['numpy', 'spaghetti', 'libpysal', 'geopandas', 'networkx', 'pyproj', 'shapely']}
VERSIONS['python'] = platform.python_version()


def encode(value):
    if isinstance(value, dict):
        return {key: encode(entry) for key, entry in value.items()}
    if isinstance(value, (list, tuple, numpy.ndarray)):
        return [encode(entry) for entry in value]
    if isinstance(value, numpy.integer):
        return int(value)
    if isinstance(value, numpy.floating):
        return float(value)
    return value


def generate(name, nodes, edges, points, maximumDistance, source, simulations, bandCount=9):
    lines = [shapely.LineString([nodes[first], nodes[second]]) for first, second in edges]
    network = spaghetti.Network(geopandas.GeoDataFrame(geometry=lines), vertex_sig=15, extractgraph=False)
    for edge in network.arc_lengths:
        network.arc_lengths[edge] = float(numpy.float32(network.arc_lengths[edge]))
    network.snapobservations(geopandas.GeoDataFrame(geometry=[shapely.Point(point) for point in points]), 'events')
    numpy.random.seed(20261015)
    result = network.GlobalAutoK(network.pointpatterns['events'], nsteps=bandCount, permutations=max(1, simulations), upperbound=maximumDistance)
    networkLength = sum(network.arc_lengths.values())
    # Export Spaghetti's actual topology and rounded weights, not a second graph builder.
    positions = [network.vertex_coords[index] for index in range(len(network.vertex_coords))]
    if 'originalNodeIds' in source:
        # Spaghetti assigns its own vertex order; align provenance to the EXPORTED
        # positions, rather than assuming its IDs match the sorted source subset.
        inputPositions = numpy.asarray(nodes, dtype=numpy.float64)
        inputIndices = [int(numpy.argmin(numpy.linalg.norm(inputPositions-position, axis=1))) for position in positions]
        assert len(set(inputIndices)) == len(positions) == len(nodes)
        assert max(numpy.linalg.norm(inputPositions[index]-position) for index,position in zip(inputIndices,positions)) < 1e-8
        source = dict(source, originalNodeIds=[source['originalNodeIds'][index] for index in inputIndices])
    rows = [[] for _ in positions]
    for (first, second), length in network.arc_lengths.items():
        rows[first].append((second, length))
        rows[second].append((first, length))
    offsets, neighbors, weights = [0], [], []
    for row in rows:
        for neighbor, weight in sorted(row):
            neighbors.append(neighbor)
            weights.append(weight)
        offsets.append(len(neighbors))
    counts = numpy.rint(result.observed * len(points) ** 2 / (2 * networkLength)).astype(numpy.uint32)
    distances = network.allneighbordistances(network.pointpatterns['events'])
    upper = distances[numpy.triu_indices(len(points), 1)]
    margin = min(float(numpy.min(numpy.abs(upper - radius))) for radius in result.xaxis[1:])
    referenceGraph = networkx.Graph()
    referenceGraph.add_weighted_edges_from((first, second, weight) for (first, second), weight in network.arc_lengths.items())
    maximumHops = max(len(path) - 1 for node in referenceGraph for path in
                      networkx.single_source_dijkstra(referenceGraph, node, cutoff=maximumDistance)[1].values())
    # 16 coordinate ulps cover projection/subtraction/interpolation in the snap;
    # gamma_(h+4) bounds positive f32 sums on paths within the largest radius.
    unitRoundoff = 2**-24
    coordinateUlp = float(numpy.spacing(numpy.float32(max(1, numpy.max(numpy.abs(positions)), numpy.max(numpy.abs(points))))))
    # Bound alternative candidate paths too, not only the selected f64 shortest path.
    # Positive full edges consume at least min(weight); allow two fractional end edges.
    maximumCandidateHops = min(len(positions) + 1, math.ceil(maximumDistance / min(weights)) + 2)
    operationCount = maximumCandidateHops + 4
    distanceErrorBound = 16 * coordinateUlp + operationCount * unitRoundoff / (1-operationCount*unitRoundoff) * maximumDistance
    ambiguousPairs = []
    pairFirst, pairSecond = numpy.triu_indices(len(points), 1)
    lowerCounts, upperCounts = [], []
    for radius in result.xaxis:
        # Synthetic dyadic geometry is exact, including strict boundary ties.
        bound = 0 if name == 'boundary' or radius == 0 else distanceErrorBound
        lowerCounts.append(int(numpy.count_nonzero(upper < radius-bound)))
        upperCounts.append(int(numpy.count_nonzero(upper < radius+bound)))
        selectedPairs = numpy.flatnonzero(numpy.abs(upper-radius) < bound)
        ambiguousPairs.append([[int(pairFirst[index]), int(pairSecond[index]), float(upper[index])] for index in selectedPairs])
    fixture = dict(case=name, reference='spaghetti.Network.GlobalAutoK', generator=Path(__file__).name,
        versions=VERSIONS, source=source, positions=positions, points=points, offsets=offsets,
        neighbors=neighbors, weights=weights, networkLength=float(numpy.float32(networkLength)),
        exactNetworkLength=networkLength, intensity=result.lam, maximumDistance=maximumDistance,
        radii=result.xaxis, lowerCounts=lowerCounts, upperCounts=upperCounts, ambiguousPairs=ambiguousPairs,
        distanceErrorBound=distanceErrorBound, maximumHops=maximumHops, maximumCandidateHops=maximumCandidateHops, counts=counts, observed=result.observed, minimumThresholdMargin=margin,
        simulations=result.sim if simulations else [], gpuSimulationCount=63 if simulations else 0,
        gpuSeeds=[17, 71, 107] if simulations else [17], referenceSeed=20261015,
        tolerance=dict(absolute=1e-6 * max(1, float(max(result.observed))), relative=1e-4),
        distributionAlpha=0.001, conventions=__doc__)
    path = HERE / 'fixtures' / f'parity-wp3-network-k-function-{name}.json'
    path.write_text(json.dumps(encode(fixture), separators=(',', ':'), allow_nan=False) + '\n')
    assert path.stat().st_size <= 1_500_000
    print(path.name, path.stat().st_size, 'nodes', len(positions), 'events', len(points), 'margin', margin, flush=True)


def main():
    generate('boundary', [[0, 0], [16, 0], [16, 16], [32, 16]], [(0, 1), (1, 2), (2, 3)],
             [[0, 0], [0, 0], [4, 0], [8, 0], [16, 0], [16, 8], [24, 16], [32, 16]], 32,
             dict(description='strict threshold ties, duplicate events, endpoint snaps; meters'), 1024)
    roads = DATA / 'chicago-roads'
    places = DATA / 'chicago-places'
    nodeCoordinates = numpy.fromfile(roads / 'nodes.bin', dtype='<f4').reshape(-1, 2)
    placeCoordinates = numpy.fromfile(places / 'position.bin', dtype='<f4').reshape(-1, 2)
    projection = pyproj.Transformer.from_crs('EPSG:4326', '+proj=aeqd +lat_0=41.885 +lon_0=-87.635 +datum=WGS84 +units=m', always_xy=True)
    nodes = numpy.column_stack(projection.transform(nodeCoordinates[:, 0], nodeCoordinates[:, 1])).astype(numpy.float32)
    points = numpy.column_stack(projection.transform(placeCoordinates[:, 0], placeCoordinates[:, 1])).astype(numpy.float32)
    sources = numpy.fromfile(roads / 'edgeSource.bin', dtype='<u4')
    targets = numpy.fromfile(roads / 'edgeTarget.bin', dtype='<u4')
    inside = numpy.max(numpy.abs(nodes), axis=1) <= 1500
    graph = networkx.Graph()
    graph.add_edges_from((int(first), int(second)) for first, second in zip(sources, targets) if first != second and inside[first] and inside[second] and not numpy.array_equal(nodes[first], nodes[second]))
    graph = graph.subgraph(max(networkx.connected_components(graph), key=len)).copy()
    nodeIds = sorted(graph.nodes)
    remap = {node: index for index, node in enumerate(nodeIds)}
    edges = [(remap[first], remap[second]) for first, second in sorted(graph.edges)]
    pointIds = numpy.flatnonzero(numpy.max(numpy.abs(points), axis=1) <= 1000)
    # Sort into 100-meter strata before systematic subsampling, preserving original IDs.
    pointIds = sorted(pointIds, key=lambda index: (int(points[index, 0] // 100), int(points[index, 1] // 100), int(index)))
    selected = [pointIds[index] for index in numpy.linspace(0, len(pointIds) - 1, min(2048, len(pointIds)), dtype=int)]
    hashes = {str(path.relative_to(DATA)): hashlib.sha256(path.read_bytes()).hexdigest() for path in
              [roads / 'nodes.bin', roads / 'edgeSource.bin', roads / 'edgeTarget.bin', places / 'position.bin']}
    generate('chicago', nodes[nodeIds], edges, points[selected], 400,
        dict(originalNodeIds=nodeIds, originalPlaceIds=selected, hashes=hashes,
             availableCorePlaces=len(pointIds), fullNodeCount=len(nodes), fullPlaceCount=len(points),
             roadAttribution=json.loads((roads / 'manifest.json').read_text()).get('properties'),
             placeAttribution=json.loads((places / 'manifest.json').read_text()).get('properties'),
             coordinateSystem='WGS84 AEQD 41.885,-87.635; float32 local meters; induced road window'), 0)


if __name__ == '__main__':
    main()
