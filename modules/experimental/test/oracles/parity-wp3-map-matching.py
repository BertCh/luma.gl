# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Composite reference for GPUMapMatching; run with the reference venv's python -I.

Reference: Newson & Krumm (2009), equations 1--3, DOI 10.1145/1653771.1653818,
https://www.microsoft.com/en-us/research/publication/hidden-markov-map-matching-noise-sparseness/
Shapely STRtree/line_locate_point supply candidates; NetworkX single_source_dijkstra
supplies directed road distances; scipy.stats.norm/expon supply normalized log
probabilities; NetworkX all_shortest_paths solves the layered HMM. No implementation
of the contributor's candidate grid, bounded A*, or Viterbi recurrence is copied.

Why composite: Leuven DistanceMatcher 1.1.4 uses a *squared* route discrepancy,
omits density normalization, adds backtracking/disconnected-edge penalties and
restricts emitting transitions to neighboring edges. Its unmodified likelihood
and sparse-sample path are NOT this model. The installed version is recorded,
but is not claimed as an equal-answer oracle.

Conventions: native planar meters, float64 reference on float32-rounded nodes and
observations; directed edges; endpoint chords (not OSM polyline/travel-time costs);
at most eight candidates sorted by distance then CSR edge ID; radius inclusive;
Gaussian sigma and exponential beta in meters; complete route bounded by
routeFactor*observation_displacement+routeSlack. Same-edge forward travel uses
the partial edge, backward travel must follow a directed cycle. Exact optimal
path ties are retained, including their direction-dependent offsets. There are
no ambiguous radius boundaries or candidate-capacity ties. No random permutations.

Chicago: four deterministic west-to-east paths at spatially stratified northings,
three proposed observations per traversed edge with seeded Gaussian noise (seed
431); discard observations within 1 cm of the radius or eighth-candidate boundary.
All 77,140 edges are spatially indexed. All candidate-to-candidate route searches
run on the full graph first. The saved induced subgraph contains *every* vertex
reachable within the largest requested cutoff at each candidate target. Repeating
every candidate and route query on the subset must give exactly the full-graph
answer. This is selected-track/full-network reference coverage, not full-track or
full-dataset GPU coverage. Hashes and original vertex/edge IDs preserve provenance.

Synthetic: misleading closer reverse-only parallel road, multi-edge sparse sample,
exact tie, disconnected road, unmatched singleton, empty track. Break/restart
semantics within a track and geographic coordinates are deliberately not covered.
"""

import hashlib
import importlib.metadata
import itertools
import json
import math
import platform
from pathlib import Path

import networkx as networkx
import numpy as numpy
import pyproj
from scipy.stats import expon, norm
from shapely import LineString, Point, STRtree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DATA = ROOT / 'examples/deck/spatial-analysis-showcase/public/data/chicago-roads'
NONE = 4294967295
SETTINGS = dict(sigma=7, beta=20, searchRadius=18, routeFactor=2, routeSlack=20)


def roundInputs(values):
    return numpy.asarray(values, dtype=numpy.float32).astype(numpy.float64)


def makeNetwork(positions, edges):
    graph = networkx.DiGraph()
    graph.add_nodes_from(range(len(positions)))
    lines = [LineString(positions[[source, target]]) for source, target in edges]
    # Canonical orientation gives both directions precisely the same snap distance.
    canonicalLines = [LineString(positions[sorted([source, target])]) for source, target in edges]
    for (source, target), line in zip(edges, lines):
        # Preserve self-loops/parallel candidate IDs. Endpoint chords give parallel
        # arcs identical cost, so the route graph may represent that cost once.
        graph.add_edge(int(source), int(target), weight=line.length)
    return graph, lines, canonicalLines, STRtree(canonicalLines)


def findCandidates(positions, edges, lines, canonicalLines, tree, observation, settings):
    point = Point(observation)
    candidates = []
    for edge in tree.query(point, predicate='dwithin', distance=settings['searchRadius']):
        distance = canonicalLines[edge].distance(point)
        offset = lines[edge].project(point)
        candidates.append(dict(edge=int(edge), distance=distance, offset=offset,
                               fraction=offset / lines[edge].length if lines[edge].length else 0,
                               position=list(lines[edge].interpolate(offset).coords[0])))
    candidates.sort(key=lambda candidate: (candidate['distance'], candidate['edge']))
    assert all(abs(candidate['distance'] - settings['searchRadius']) > 0.01 for candidate in candidates)
    if len(candidates) > 8:
        assert candidates[8]['distance'] - candidates[7]['distance'] > 0.01
    return candidates[:8]


def prepareQueries(positions, edges, tracks, settings):
    graph, lines, canonicalLines, tree = makeNetwork(positions, edges)
    candidates = [[findCandidates(positions, edges, lines, canonicalLines, tree, point, settings)
                   for point in track] for track in tracks]
    cutoffs = {}
    for track, trackCandidates in zip(tracks, candidates):
        for index in range(1, len(track)):
            limit = settings['routeFactor'] * numpy.linalg.norm(track[index] - track[index - 1]) + settings['routeSlack']
            for candidate in trackCandidates[index - 1]:
                target = int(edges[candidate['edge'], 1])
                cutoffs[target] = max(cutoffs.get(target, 0), float(limit))
    searches = {source: networkx.single_source_dijkstra(graph, source, cutoff=limit)
                for source, limit in cutoffs.items()}
    return graph, lines, candidates, cutoffs, searches


def getRoute(previous, current, edges, lines, searches):
    if previous['edge'] == current['edge'] and current['fraction'] >= previous['fraction']:
        return current['offset'] - previous['offset'], []
    start = int(edges[previous['edge'], 1])
    end = int(edges[current['edge'], 0])
    distances, paths = searches[start]
    distance = distances.get(end, math.inf)
    return lines[previous['edge']].length - previous['offset'] + distance + current['offset'], paths.get(end, [])


def solveTrack(track, candidates, edges, lines, searches, settings):
    if not len(track) or (len(track) == 1 and not candidates[0]):
        return dict(logLikelihood=0, scoreGap='Infinity', alternatives=[dict(
            edges=[NONE] * len(track), fractions=[-1] * len(track), offsets=[-1] * len(track),
            distances=[-1] * len(track), positions=numpy.asarray(track).flatten().tolist(), routes=[])])
    lattice = networkx.DiGraph()
    for index, choices in enumerate(candidates):
        for slot, candidate in enumerate(choices):
            emission = -float(norm.logpdf(candidate['distance'], scale=settings['sigma']))
            lattice.add_node((index, slot))
            if index == 0:
                lattice.add_edge('start', (index, slot), weight=emission)
            else:
                straight = float(numpy.linalg.norm(track[index] - track[index - 1]))
                limit = settings['routeFactor'] * straight + settings['routeSlack']
                for previousSlot, previous in enumerate(candidates[index - 1]):
                    route, nodePath = getRoute(previous, candidate, edges, lines, searches)
                    if route <= limit:
                        transition = -float(expon.logpdf(abs(route - straight), scale=settings['beta']))
                        lattice.add_edge((index - 1, previousSlot), (index, slot),
                                         weight=transition + emission, route=route, nodePath=[int(node) for node in nodePath])
            if index == len(track) - 1:
                lattice.add_edge((index, slot), 'end', weight=0)
    bestScore = networkx.shortest_path_length(lattice, 'start', 'end', weight='weight')
    bestPaths = list(itertools.islice(networkx.all_shortest_paths(lattice, 'start', 'end', weight='weight'), 257))
    assert len(bestPaths) <= 256
    scoreGap = math.inf
    for path in itertools.islice(networkx.shortest_simple_paths(lattice, 'start', 'end', weight='weight'), len(bestPaths) + 1):
        score = networkx.path_weight(lattice, path, 'weight')
        if score > bestScore + 1e-10:
            scoreGap = score - bestScore
    alternatives = []
    for path in bestPaths:
        states = path[1:-1]
        chosen = [candidates[index][slot] for index, slot in states]
        alternatives.append(dict(
            edges=[candidate['edge'] for candidate in chosen],
            fractions=[candidate['fraction'] for candidate in chosen],
            offsets=[candidate['offset'] for candidate in chosen],
            distances=[candidate['distance'] for candidate in chosen],
            positions=[coordinate for candidate in chosen for coordinate in candidate['position']],
            routes=[dict(length=lattice.edges[left, right]['route'], nodePath=lattice.edges[left, right]['nodePath'])
                    for left, right in zip(states, states[1:])]))
    return dict(logLikelihood=-bestScore, scoreGap=scoreGap if math.isfinite(scoreGap) else 'Infinity', alternatives=alternatives)


def writeFixture(name, positions, edges, tracks, source, settings=SETTINGS):
    graph, lines, candidates, cutoffs, searches = prepareQueries(positions, edges, tracks, settings)
    expected = [solveTrack(track, choices, edges, lines, searches, settings)
                for track, choices in zip(tracks, candidates)]
    coordinateScale = max(1, float(numpy.abs(positions).max()), *(float(numpy.abs(track).max()) for track in tracks if len(track)))
    # Projection subtracts f32 coordinates, divides a dot product by squared length,
    # then interpolates. Eight coordinate ulps bound these short operations away from
    # zero-length segments. Fraction uncertainty is that bound / minimum edge length.
    coordinateError = 8 * float(numpy.spacing(numpy.float32(coordinateScale)))
    minimumLength = min(line.length for line in lines if line.length > 0)
    distanceAbsolute = max(1e-6 * coordinateScale, coordinateError)
    fractionAbsolute = max(1e-6, coordinateError / minimumLength)
    cellSize = max(128, math.ceil(float(numpy.abs(positions[edges[:, 1]] - positions[edges[:, 0]]).max()) / 14))
    allPositions = numpy.concatenate([positions] + [track for track in tracks if len(track)])
    minimum = numpy.floor(allPositions.min(axis=0) / cellSize) * cellSize - cellSize
    maximum = numpy.ceil(allPositions.max(axis=0) / cellSize) * cellSize + cellSize
    entryCapacity = sum(int(numpy.prod(numpy.abs(numpy.floor((positions[target] - minimum) / cellSize)
                                                    - numpy.floor((positions[start] - minimum) / cellSize)) + 1))
                         for start, target in edges) + 256
    offsets = numpy.concatenate([[0], numpy.cumsum(numpy.bincount(edges[:, 0], minlength=len(positions)))])
    fixture = dict(
        generator=Path(__file__).name,
        reference='composite reference: Newson-Krumm (2009) equations 1-3; scipy.stats.norm/expon; shapely.STRtree/line_locate_point; networkx.single_source_dijkstra/all_shortest_paths',
        versions={name: importlib.metadata.version(name) for name in ['numpy', 'scipy', 'shapely', 'networkx', 'pyproj', 'leuvenmapmatching']},
        python=platform.python_version(), case=name, source=source,
        conventions=dict(coordinateSpace='local planar AEQD meters' if name == 'chicago' else 'native planar meters',
                         inputs='float32 rounded, float64 reference', edgeCost='endpoint chord length',
                         direction='directed CSR', ties='all exact optimal paths', seed=431,
                         candidateCount=8, routeNodeBudget=128, leuven='not equivalent: squared transition and extra penalties'),
        settings=settings, cellSize=cellSize, bounds=dict(minimum=minimum.tolist(), maximum=maximum.tolist()), entryCapacity=entryCapacity,
        positions=positions.flatten().tolist(), offsets=offsets.tolist(), neighbors=edges[:, 1].tolist(),
        points=[coordinate for track in tracks for point in track for coordinate in point],
        trackOffsets=numpy.concatenate([[0], numpy.cumsum([len(track) for track in tracks])]).tolist(),
        candidates=candidates, expected=expected,
        diagnostics=dict(maximumDijkstraBall=max((len(search[0]) for search in searches.values()), default=0),
                         nonNearestMatches=sum(candidate['edge'] != choices[0]['edge']
                                               for result, trackChoices in zip(expected, candidates)
                                               for candidate, choices in zip([dict(edge=edge) for edge in result['alternatives'][0]['edges']], trackChoices) if choices)),
        tolerances=dict(profile='L', distance=dict(absolute=distanceAbsolute, relative=1e-4),
                        fraction=dict(absolute=fractionAbsolute, relative=1e-4),
                        likelihood=dict(absolute=1e-6 * max(1, *(abs(result['logLikelihood']) for result in expected)), relative=1e-4),
                        derivation='Distance atol=max(1e-6*coordinateScale,8*ulp32(coordinateScale)); fraction atol=max(1e-6,8*ulp32(coordinateScale)/minimumEdgeLength). Inputs rounded before Shapely/NetworkX. IDs, tie membership, breaks and overflow exact.',
                        coordinateScale=coordinateScale, minimumEdgeLength=minimumLength))
    output = HERE / 'fixtures' / f'parity-wp3-map-matching-{name}.json'
    output.write_text(json.dumps(fixture, separators=(',', ':'), allow_nan=False) + '\n')
    assert output.stat().st_size <= 1_500_000
    print(name, 'nodes', len(positions), 'edges', len(edges), 'points', len(fixture['points']) // 2,
          'bytes', output.stat().st_size, 'diagnostics', fixture['diagnostics'])


def makeSynthetic():
    positions = roundInputs([[0, 0], [100, 0], [200, 0], [300, 0], [0, 12], [100, 12], [200, 12], [300, 12], [500, 0], [600, 0]])
    edges = numpy.array(sorted([(0, 1), (1, 2), (2, 3), (5, 4), (6, 5), (7, 6), (8, 9)]))
    tracks = [roundInputs([[distance, 7] for distance in [20, 80, 120, 180, 220, 280]]),
              roundInputs([[30, 1], [90, 2], [230, 1], [290, 2]]),
              roundInputs([[50, 6]]), roundInputs([[540, 2], [580, 3]]),
              roundInputs([[400, 200]]), numpy.empty((0, 2))]
    source = dict(description='Synthetic directed parallel roads with an island',
                  sha256=hashlib.sha256(positions.tobytes() + edges.astype('<u4').tobytes()).hexdigest())
    writeFixture('synthetic', positions, edges, tracks, source)


def makeChicago():
    files = ['manifest.json', 'nodes.bin', 'edgeSource.bin', 'edgeTarget.bin']
    hashes = {name: hashlib.sha256((DATA / name).read_bytes()).hexdigest() for name in files}
    longitudeLatitude = numpy.fromfile(DATA / 'nodes.bin', dtype='<f4').reshape(-1, 2)
    projection = '+proj=aeqd +lat_0=41.88 +lon_0=-87.63 +datum=WGS84 +units=m'
    transformer = pyproj.Transformer.from_crs('EPSG:4326', projection, always_xy=True)
    positions = roundInputs(numpy.column_stack(transformer.transform(*longitudeLatitude.T)))
    edges = numpy.column_stack([numpy.fromfile(DATA / name, dtype='<u4') for name in ['edgeSource.bin', 'edgeTarget.bin']])
    graph, lines, canonicalLines, tree = makeNetwork(positions, edges)
    edgeLookup = {tuple(edge): index for index, edge in enumerate(edges)}
    random = numpy.random.default_rng(431)
    tracks = []
    rejectedObservations = 0
    for northing in [-1500, -500, 500, 1500]:
        start = int(numpy.argmin(numpy.linalg.norm(positions - [-1200, northing], axis=1)))
        end = int(numpy.argmin(numpy.linalg.norm(positions - [1200, northing + 500], axis=1)))
        path = networkx.shortest_path(graph, start, end, weight='weight')
        points = []
        for source, target in zip(path, path[1:]):
            line = lines[edgeLookup[source, target]]
            for fraction in [0.2, 0.5, 0.8]:
                point = roundInputs(numpy.asarray(line.interpolate(fraction, normalized=True).coords[0]) + random.normal(0, 3, 2))
                try:
                    findCandidates(positions, edges, lines, canonicalLines, tree, point, SETTINGS)
                except AssertionError:
                    rejectedObservations += 1
                    continue
                points.append(point)
        tracks.append(roundInputs(points))
    _, _, fullCandidates, cutoffs, searches = prepareQueries(positions, edges, tracks, SETTINGS)
    keptNodes = set()
    for distances, _ in searches.values():
        keptNodes.update(distances)
    for choices in fullCandidates:
        for candidates in choices:
            for candidate in candidates:
                keptNodes.update(int(value) for value in edges[candidate['edge']])
    nodeIds = sorted(keptNodes)
    nodeLookup = {node: index for index, node in enumerate(nodeIds)}
    edgeIds = [index for index, (source, target) in enumerate(edges) if source in keptNodes and target in keptNodes]
    subsetEdges = numpy.array([[nodeLookup[source], nodeLookup[target]] for source, target in edges[edgeIds]])
    subsetPositions = positions[nodeIds]
    subsetGraph, _, subsetCandidates, _, subsetSearches = prepareQueries(subsetPositions, subsetEdges, tracks, SETTINGS)
    for fullTrack, subsetTrack in zip(fullCandidates, subsetCandidates):
        for fullChoices, subsetChoices in zip(fullTrack, subsetTrack):
            assert [candidate['edge'] for candidate in fullChoices] == [edgeIds[candidate['edge']] for candidate in subsetChoices]
    comparedDistances = 0
    for source, (distances, _) in searches.items():
        subsetDistances = networkx.single_source_dijkstra_path_length(subsetGraph, nodeLookup[source], cutoff=cutoffs[source])
        assert {nodeIds[node]: distance for node, distance in subsetDistances.items()} == distances
        comparedDistances += len(distances)
    source = dict(dataset=str(DATA.relative_to(ROOT)), sha256=hashes, projection=projection,
                  fullNodeCount=len(positions), fullEdgeCount=len(edges), nodeIds=nodeIds, edgeIds=edgeIds,
                  selection='four northing-stratified shortest paths, complete full-graph candidate-radius and directed-route neighborhoods',
                  fullGraphCandidateQueries=sum(len(track) for track in tracks), fullGraphRouteQueries=len(searches),
                  fullGraphDistanceComparisons=comparedDistances, maximumSubsetCostError=0,
                  rejectedBoundaryObservations=rejectedObservations,
                  attribution='OpenStreetMap contributors; ODbL 1.0')
    writeFixture('chicago', subsetPositions, subsetEdges, tracks, source)


if __name__ == '__main__':
    makeSynthetic()
    makeChicago()
