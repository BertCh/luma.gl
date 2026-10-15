# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent DISCRETE encounter parity, not MobilityDB continuous tDwithin.

Reference: scipy.spatial.cKDTree.query_pairs(r, p=2, eps=0), per common-clock
bucket, followed by grouping the returned pairs. Euclidean distances use SciPy
spatial.distance.euclidean. https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query_pairs.html

All 2,496 Celestrak source tracks, 13 evenly spaced clock instants (900 seconds).
Interpolate source lon/lat linearly with numpy.interp only within each track's
observed window, project EPSG:4326 -> EPSG:3857 using always_xy, then round inputs
to f32 BEFORE querying. Units are projected meters, NOT ground/geodesic meters.
Absent samples are NaN; bounds are inclusive; disabled tracks are absent. Pair
keys have lower ID first, sort lexicographically, and threshold ties are included.
The runtime clamps the query radius to cellSize. Every pair/column is preserved.
Synthetic cases cover exact distance ties, duplicates, invalid/disabled/outside
samples, bounds, clamping, and an encounter occurring ONLY between buckets (which
this public API intentionally misses). Continuous closest times/intervals cannot
be validated through GPUTrajectoryEncounters: it exposes no such outputs.
"""
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pyproj
import scipy
from scipy.spatial import cKDTree
from scipy.spatial.distance import euclidean

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DATA = ROOT / 'examples/deck/spatial-analysis-showcase/public/data/celestrak-ground-tracks'


def encode(value):
    if isinstance(value, dict):
        return {key: encode(entry) for key, entry in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [encode(entry) for entry in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def make_case(name, samples, times, bounds, distance, cell_size, valid=None):
    samples = np.asarray(samples, dtype=np.float32).astype(np.float64)
    times = np.asarray(times, dtype=np.float32).astype(np.float64)
    track_count, bucket_count, _ = samples.shape
    valid = np.ones(track_count, dtype=int) if valid is None else np.asarray(valid)
    pairs = {}
    hit_count = 0
    for bucket in range(bucket_count):
        points = samples[:, bucket]
        active = np.flatnonzero((valid != 0) & np.isfinite(points).all(axis=1)
                               & (points[:, 0] >= bounds[0]) & (points[:, 0] <= bounds[2])
                               & (points[:, 1] >= bounds[1]) & (points[:, 1] <= bounds[3]))
        for first, second in cKDTree(points[active]).query_pairs(min(distance, cell_size)):
            first, second = sorted((int(active[first]), int(active[second])))
            measured = euclidean(points[first], points[second])
            pair = pairs.setdefault((first, second), dict(ids=first, partners=second,
                                    first=bucket, distances=measured, counts=0, times=times[bucket]))
            pair['distances'] = min(pair['distances'], measured)
            pair['counts'] += 1
            hit_count += 1
    ordered = [pairs[key] for key in sorted(pairs)]
    expected = {key: [pair[key] for pair in ordered]
                for key in ('ids', 'partners', 'first', 'distances', 'counts', 'times')}
    coordinate_scale = max(1, float(np.nanmax(np.abs(samples[np.isfinite(samples)]))))
    return dict(name=name, samples=samples.reshape(-1), bucketTimes=times,
                trackCount=track_count, bucketCount=bucket_count, trackValid=valid,
                bounds=bounds, distance=distance, cellSize=cell_size, hitCount=hit_count,
                distanceAbsolute=max(1e-3, 8 * float(np.spacing(np.float32(coordinate_scale)))),
                expected=expected)


offsets = np.fromfile(DATA / 'pathOffsets.bin', '<u4')
vertices = np.fromfile(DATA / 'vertices.bin', '<f4').reshape(-1, 2)
timestamps = np.fromfile(DATA / 'timestamp.bin', '<u4')
clock = np.arange(0, 10801, 900, dtype=float)
samples = np.full((len(offsets) - 1, len(clock), 2), np.nan)
transformer = pyproj.Transformer.from_crs(4326, 3857, always_xy=True)
for track, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
    if stop <= start:
        continue
    active = (clock >= timestamps[start]) & (clock <= timestamps[stop - 1])
    longitude = np.interp(clock[active], timestamps[start:stop], vertices[start:stop, 0])
    latitude = np.interp(clock[active], timestamps[start:stop], vertices[start:stop, 1])
    samples[track, active] = np.array(transformer.transform(longitude, latitude)).T

edge_samples = [
    [[0, 0], [0, 0], [0, 0]],
    [[3, 4], [3, 4], [3, 4]],  # exact radius 5
    [[3, 4], [3, 4], [3, 4]],  # coincident tie
    [[20, 20], [24, 20], [20, 20]],
    [[24, 20], [20, 20], [24, 20]],  # between-bucket crossing, sampled distance 4
    [[0, 0], [np.nan, np.nan], [1, 0]],
    [[0, 0], [0, 0], [0, 0]],  # disabled
    [[-11, 0], [-11, 0], [-11, 0]],  # outside
    [[-10, -10], [-10, -10], [-10, -10]],  # inclusive bound
    [[-7, -6], [-7, -6], [-7, -6]],
]
cases = [make_case('celestrak-all-tracks-13-clock-buckets', samples, clock,
                   [-21000000, -21000000, 21000000, 21000000], 150000, 500000),
         make_case('ties-absent-bounds', edge_samples, [0, 10, 20], [-10, -10, 30, 30], 5, 5,
                   [1, 1, 1, 1, 1, 1, 0, 1, 1, 1]),
         make_case('radius-clamped-discrete-crossing', edge_samples, [0, 10, 20],
                   [-10, -10, 30, 30], 5, 2, [1, 1, 1, 1, 1, 1, 0, 1, 1, 1])]
assert len(cases[0]['expected']['ids']) > 100
assert (3, 4) not in set(zip(cases[2]['expected']['ids'], cases[2]['expected']['partners']))
fixture = dict(generator=Path(__file__).name,
               reference='scipy.spatial.cKDTree.query_pairs + scipy.spatial.distance.euclidean',
               versions=dict(python=platform.python_version(), numpy=np.__version__,
                             scipy=scipy.__version__, pyproj=pyproj.__version__),
               source=dict(dataset=str(DATA.relative_to(ROOT)), tracks=len(offsets)-1,
                           hashes={name: hashlib.sha256((DATA/name).read_bytes()).hexdigest()
                                   for name in ['manifest.json', 'vertices.bin', 'pathOffsets.bin', 'timestamp.bin']},
                           selection='all tracks; clock 0..10800 inclusive step 900 seconds'),
               conventions=dict(crs='EPSG:3857', units='projected meters', inputPrecision='float32',
                                temporal='discrete buckets; no continuous encounter assertion',
                                threshold='inclusive; radius=min(distance,cellSize)'),
               tolerance=dict(profile='D', relative=2e-6,
                              absolute='max(1e-3 m, 8*ulp32(max absolute input coordinate))',
                              idsCountsFlags='exact'), cases=cases)
output = HERE / 'fixtures/parity-wp2-trajectory-encounters.json'
output.write_text(json.dumps(encode(fixture), separators=(',', ':'), allow_nan=False) + '\n')
assert output.stat().st_size <= 1500000
print(output.name, output.stat().st_size, [(case['name'], len(case['expected']['ids']), case['hitCount']) for case in cases])
