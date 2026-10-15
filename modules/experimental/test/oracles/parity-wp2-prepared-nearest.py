# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent prepared/candidate/nearest parity using GEOS STRtree.

https://shapely.readthedocs.io/en/stable/strtree.html
STRtree.query(expanded query boxes) is the conservative candidate set (NOT exact
dwithin); query(predicate='dwithin') independently checks candidate completeness;
query_nearest(all_matches=True, return_distance=True, max_distance=radius) is the
exact nearest tie set. shortest_line supplies every nearest feature foot point.
The adapter sorts exact ties by feature ROW, then maps rows to stable source IDs.

All 77,140 Chicago road polylines and 109,711 crash points are projected with
always_xy to WGS84 AEQD centered -87.63,41.88; snap to a declared 1/16 meter grid,
then f32. Select every road whose box intersects [-1500,1500]^2 (full paths kept),
and at most 256 evenly source-ordered crashes in [-1000,1000]^2. Check this window's
exact dwithin pairs against a full 77,140-road STRtree for EVERY selected query.
Canonicalize line direction lexicographically so opposite directed edges use
identical arithmetic; IDs remain separate (ties retained). No segment thinning.

Three frames: initial, queries translated (17,-11)m with index reused, then roads
translated (64,32)m with explicit index invalidation. Synthetic cases have exact
radius/box boundary contacts, coincident lines, equal-distance ties, no matches,
and a false-positive box candidate. Every input/output is retained. Tolerances G:
16 ulps at max coordinate scale for positions/distances, exact IDs/counts/status;
no relaxed near-tie membership. No weights or polygon branches are claimed.
"""
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pyproj
import shapely
from shapely import LineString, Point, STRtree, box

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DATA = ROOT / 'examples/deck/spatial-analysis-showcase/public/data'
CRS = '+proj=aeqd +lat_0=41.88 +lon_0=-87.63 +datum=WGS84 +units=m'


def normalize_line(coordinates):
    values = [tuple(point) for point in coordinates]
    return np.asarray(min(values, list(reversed(values))), dtype=np.float32).astype(float)


def reference_frame(name, lines, queries, radius, source_ids):
    features = [LineString(line) for line in lines]
    tree = STRtree(features)
    candidates, exact = [], []
    neighbors = []
    nearest_gap = float('inf')
    for query_index, coordinate in enumerate(queries):
        point = Point(coordinate)
        query_box = box(coordinate[0]-radius, coordinate[1]-radius,
                        coordinate[0]+radius, coordinate[1]+radius)
        candidate_rows = sorted(map(int, tree.query(query_box)))
        exact_rows = sorted(map(int, tree.query(point, predicate='dwithin', distance=radius)))
        assert set(exact_rows).issubset(candidate_rows)
        candidates.extend([[query_index, row] for row in candidate_rows])
        exact.extend([[query_index, row] for row in exact_rows])
        rows, distances = tree.query_nearest(point, max_distance=radius, return_distance=True, all_matches=True)
        entries = []
        for row, distance in sorted(zip(rows, distances), key=lambda pair: (pair[1], pair[0])):
            foot = list(shapely.shortest_line(point, features[row]).coords)[-1]
            entries.append(dict(row=int(row), id=int(source_ids[row]), distance=float(distance), foot=list(foot)))
        neighbors.append(entries)
        all_distances = sorted(set(float(shapely.distance(point, features[row])) for row in exact_rows))
        if len(all_distances) > 1:
            nearest_gap = min(nearest_gap, all_distances[1] - all_distances[0])
    return dict(name=name, positions=np.concatenate(lines).reshape(-1).tolist(), queries=queries.reshape(-1).tolist(),
                candidates=candidates, dwithin=exact, neighbors=neighbors,
                nearestNonTieGap=nearest_gap if np.isfinite(nearest_gap) else None)


def make_case(name, lines, queries, radius, source_ids):
    lines = [normalize_line(line) for line in lines]
    queries = np.asarray(queries, dtype=np.float32).astype(float)
    offsets = np.concatenate([[0], np.cumsum([len(line) for line in lines])]).tolist()
    frames = [reference_frame('initial', lines, queries, radius, source_ids),
              reference_frame('query-moved-index-reused', lines, queries + [17, -11], radius, source_ids),
              reference_frame('features-moved-index-invalidated', [line + [64, 32] for line in lines],
                              queries + [17, -11], radius, source_ids)]
    scale = max(1., max(abs(value) for frame in frames for value in frame['positions'] + frame['queries']))
    return dict(name=name, radius=radius, lineOffsets=offsets, sourceIds=list(map(int, source_ids)), frames=frames,
                absolute=16 * float(np.spacing(np.float32(scale))))


roads = DATA / 'chicago-roads'
crashes = DATA / 'chicago-crashes'
offsets = np.fromfile(roads/'edgePathOffsets.bin', '<u4')
vertices = np.fromfile(roads/'edgeVertices.bin', '<f4').reshape(-1, 2)
crash_positions = np.fromfile(crashes/'position.bin', '<f4').reshape(-1, 2)
transformer = pyproj.Transformer.from_crs(4326, CRS, always_xy=True)
project = lambda values: (np.round(np.asarray(transformer.transform(values[:, 0], values[:, 1])).T * 16) / 16).astype(np.float32).astype(float)
projected = project(vertices)
all_lines = [normalize_line(projected[start:stop]) for start, stop in zip(offsets[:-1], offsets[1:])]
full_tree = STRtree([LineString(line) for line in all_lines])
selected_features = sorted(map(int, full_tree.query(box(-1500, -1500, 1500, 1500))))
projected_queries = project(crash_positions)
window_queries = np.flatnonzero((np.abs(projected_queries) <= 1000).all(axis=1))
selected_queries = window_queries[np.linspace(0, len(window_queries)-1, min(256, len(window_queries)), dtype=int)]
real = make_case('chicago-complete-road-window', [all_lines[index] for index in selected_features],
                 projected_queries[selected_queries], 128, selected_features)
full_exact = sorted((int(query), int(feature)) for query, feature in full_tree.query(
    [Point(point) for point in projected_queries[selected_queries]], predicate='dwithin', distance=128).T)
local_exact = [(query, selected_features[feature]) for query, feature in real['frames'][0]['dwithin']]
assert full_exact == local_exact
assert len(full_exact) > 100
edge = make_case('ties-radius-boundaries-and-box-false-positive',
                 [[[0, -4], [0, 4]], [[0, -4], [0, 4]], [[6, -4], [6, 4]],
                  [[-4, -4], [4, 4]], [[30, 0], [35, 0]]],
                 [[3, 0], [0, 0], [-3, 2], [3, -3], [100, 100], [30, 3], [0, 7]],
                 3, [80, 7, 33, 12, 5])
fixture = dict(generator=Path(__file__).name,
               reference='shapely.STRtree.query/query_nearest + shapely.shortest_line',
               versions=dict(python=platform.python_version(), numpy=np.__version__, shapely=shapely.__version__,
                             geos=shapely.geos_version_string, pyproj=pyproj.__version__),
               conventions=dict(crs=CRS, units='meters', snapGrid=1/16, inputPrecision='float32',
                                ties='all exact distance ties sorted by feature row, then source-ID mapping',
                                candidates='bounding-box overlap; exact dwithin is a subset',
                                invalidation='third frame rebuilds; second frame changes queries only'),
               source=dict(featureIds=selected_features, queryIds=selected_queries.tolist(),
                           fullFeatureCount=len(all_lines), fullQueryCount=len(crash_positions),
                           fullWindowQueryCount=len(window_queries), fullDataDwithinCount=len(full_exact),
                           fullDataDwithinHash=hashlib.sha256(json.dumps(full_exact).encode()).hexdigest(),
                           hashes={str(path.relative_to(DATA)): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in [roads/'manifest.json', roads/'edgePathOffsets.bin', roads/'edgeVertices.bin',
                                                crashes/'manifest.json', crashes/'position.bin']}),
               tolerance=dict(profile='G', absolute='16 ulp32(max coordinate magnitude)', relative=0,
                              idsCountsFlags='exact'), cases=[real, edge])
output = HERE / 'fixtures/parity-wp2-prepared-nearest.json'
output.write_text(json.dumps(fixture, separators=(',', ':'), allow_nan=False) + '\n')
assert output.stat().st_size <= 1500000
print(output.name, output.stat().st_size, len(selected_features), len(selected_queries),
      [(frame['name'], len(frame['candidates']), max(map(len, frame['neighbors'])), frame['nearestNonTieGap']) for frame in real['frames']])
