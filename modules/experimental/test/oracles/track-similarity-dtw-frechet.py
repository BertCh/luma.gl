# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Reference discrete Frechet distances, Frechet decisions and (banded) dynamic time warping costs
for GPUTrackSimilarity.

Writes fixtures/track-similarity-dtw-frechet.json next to this script, consumed by
test/gpu-spatial-analysis/track-similarity/gpu-track-similarity-dtw-frechet.spec.ts (GPU) and
track-similarity-dtw-frechet.node.spec.ts (fixture sanity). Run from anywhere with
    <venv>/bin/python -I modules/experimental/test/oracles/track-similarity-dtw-frechet.py
in a venv with numpy, shapely, similaritymeasures, tslearn and dtaidistance (versions recorded).

Inputs are rounded to float32 first (the GPU reads float32 positions); every reference is then
evaluated in binary64.
- Discrete Frechet: `similaritymeasures.frechet_dist` (planar), cross-checked against
  `shapely.frechet_distance` (GEOS discrete Frechet) and against this script's own dynamic program,
  which is then reused with the haversine metric for the great-circle cases.
- Frechet decision: `within = frechet <= epsilon` for epsilons at 0.5, 0.999, 1.001 and 2 times the
  distance (away from the float32 boundary), plus one exact case on an integer grid where the
  distance is exactly 5 and epsilon is 5 (squared values are exact in float32), which is within.
- Dynamic time warping, the `similaritymeasures.dtw` convention: the smallest sum of point
  distances over monotone warping paths. Unbanded values are `similaritymeasures.dtw(a, b)[0]`.
  Banded values come from this script's dynamic program masked with
  `tslearn.metrics.sakoe_chiba_mask(n, m, radius)`; the same program with squared costs, square
  rooted, is checked against `tslearn.metrics.dtw(a, b, global_constraint='sakoe_chiba',
  sakoe_chiba_radius=radius)`, which pins the band and the recursion. `dtaidistance` uses a
  different window rule for unequal lengths, so it is only checked unbanded (its
  `dtw_ndim.distance` is the square root of summed squares, like tslearn).
- Great-circle cases: haversine distance in meters on the 6371008.8 m sphere between
  [longitude, latitude] degree vertices, the GPU's `geodesicHaversineDistance`.

Tolerances (stored in `tolerances`): Frechet distances are one float32 point distance, about 1e-6
relative plus float32 coordinate rounding (relative to the coordinate magnitude); dynamic time
warping sums `n + m - 1` such distances in float32, so its bound grows with the path length.
"""

import json
import math
import os
import platform

import numpy as np
import shapely
import similaritymeasures
import tslearn
import tslearn.metrics
import dtaidistance
from dtaidistance import dtw_ndim

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'fixtures', 'track-similarity-dtw-frechet.json')
EARTH_RADIUS = 6371008.8


def to_f32(points):
    return np.asarray(points, dtype=np.float32).astype(np.float64)


def euclidean(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])


def haversine(p, q):
    lon1, lat1, lon2, lat2 = map(math.radians, (p[0], p[1], q[0], q[1]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS * math.asin(min(1.0, math.sqrt(h)))


def frechet_program(a, b, metric):
    n, m = len(a), len(b)
    table = np.full((n, m), np.inf)
    for i in range(n):
        for j in range(m):
            cost = metric(a[i], b[j])
            if i == 0 and j == 0:
                table[i, j] = cost
            else:
                best = min(
                    table[i - 1, j] if i > 0 else np.inf,
                    table[i, j - 1] if j > 0 else np.inf,
                    table[i - 1, j - 1] if i > 0 and j > 0 else np.inf,
                )
                table[i, j] = max(best, cost)
    return float(table[-1, -1])


def warping_program(a, b, metric, mask=None, squared=False):
    n, m = len(a), len(b)
    table = np.full((n, m), np.inf)
    for i in range(n):
        for j in range(m):
            if mask is not None and not mask[i, j]:
                continue
            cost = metric(a[i], b[j]) ** (2 if squared else 1)
            if i == 0 and j == 0:
                table[i, j] = cost
                continue
            best = min(
                table[i - 1, j] if i > 0 else np.inf,
                table[i, j - 1] if j > 0 else np.inf,
                table[i - 1, j - 1] if i > 0 and j > 0 else np.inf,
            )
            table[i, j] = best + cost
    return float(table[-1, -1])


def random_track(rng, count, origin, step):
    position = np.array(origin, dtype=np.float64)
    points = []
    for _ in range(count):
        points.append(position.copy())
        position = position + rng.normal(0, step, 2) + np.array([step * 0.6, 0])
    return points


def planar_case(name, a, b, radii):
    a = to_f32(a)
    b = to_f32(b)
    frechet = frechet_program(a, b, euclidean)
    reference = float(similaritymeasures.frechet_dist(a, b))
    geos = float(shapely.frechet_distance(shapely.LineString(a), shapely.LineString(b))) if len(a) > 1 and len(b) > 1 else reference
    assert abs(frechet - reference) <= 1e-9 * max(1.0, frechet), (name, frechet, reference)
    assert abs(frechet - geos) <= 1e-9 * max(1.0, frechet), (name, frechet, geos)
    dtw = warping_program(a, b, euclidean)
    reference_dtw = float(similaritymeasures.dtw(a, b)[0])
    assert abs(dtw - reference_dtw) <= 1e-9 * max(1.0, dtw), (name, dtw, reference_dtw)
    # dtaidistance and tslearn (unbanded) are the square root of summed squares.
    squared = math.sqrt(warping_program(a, b, euclidean, squared=True))
    assert abs(squared - float(tslearn.metrics.dtw(a, b))) <= 1e-9 * max(1.0, squared), name
    assert abs(squared - float(dtw_ndim.distance(a, b))) <= 1e-6 * max(1.0, squared), name
    banded = []
    for radius in radii:
        mask = tslearn.metrics.sakoe_chiba_mask(len(a), len(b), radius)
        value = warping_program(a, b, euclidean, mask)
        squared_banded = math.sqrt(warping_program(a, b, euclidean, mask, squared=True))
        reference_banded = float(
            tslearn.metrics.dtw(a, b, global_constraint='sakoe_chiba', sakoe_chiba_radius=radius)
        )
        assert abs(squared_banded - reference_banded) <= 1e-9 * max(1.0, reference_banded), (name, radius)
        banded.append({'radius': radius, 'dynamicTimeWarping': value})
    return {
        'name': name,
        'a': a.tolist(),
        'b': b.tolist(),
        'frechet': frechet,
        'dynamicTimeWarping': dtw,
        'banded': banded,
        'decisions': decisions(frechet),
    }


def decisions(frechet):
    result = []
    for factor in (0.5, 0.999, 1.001, 2.0):
        epsilon = frechet * factor
        result.append({'epsilon': epsilon, 'within': 1 if frechet <= epsilon else 0})
    return result


def great_circle_case(name, a, b, radii):
    a = to_f32(a)
    b = to_f32(b)
    frechet = frechet_program(a, b, haversine)
    dtw = warping_program(a, b, haversine)
    banded = []
    for radius in radii:
        mask = tslearn.metrics.sakoe_chiba_mask(len(a), len(b), radius)
        banded.append({'radius': radius, 'dynamicTimeWarping': warping_program(a, b, haversine, mask)})
    return {
        'name': name,
        'a': a.tolist(),
        'b': b.tolist(),
        'frechet': frechet,
        'dynamicTimeWarping': dtw,
        'banded': banded,
        'decisions': decisions(frechet),
    }


def main():
    rng = np.random.default_rng(20261014)
    planar = []
    planar.append(planar_case('equal-length', [[0, 0], [1, 0.5], [2, 0], [3, 0.5], [4, 0]], [[0, 1], [1, 1.5], [2, 1.2], [3, 1.4], [4, 1]], [0, 1, 2]))
    planar.append(planar_case('unequal-length', [[0, 0], [2, 0], [4, 0]], [[0, 0.5], [0.5, 0.4], [1, 0.6], [2, 0.5], [3, 0.7], [3.5, 0.3], [4, 0.4]], [0, 1, 3, 7]))
    # The band forces the warping path to stay near the scaled diagonal although the cheapest
    # unbanded path would hold the first vertex of b for a long time.
    planar.append(planar_case('band-forces-path', [[0, 0]] * 6 + [[k, 0] for k in range(1, 7)], [[k * 0.5, 0.1] for k in range(12)], [0, 1, 2, 12]))
    planar.append(planar_case('single-vertex', [[3, 3]], [[0, 0], [1, 1], [2, 2]], [0, 3]))
    for index in range(12):
        n = int(rng.integers(5, 60))
        m = int(rng.integers(5, 60))
        a = random_track(rng, n, (rng.uniform(-100, 100), rng.uniform(-100, 100)), 1.5)
        b = random_track(rng, m, (a[0][0] + rng.normal(0, 2), a[0][1] + rng.normal(0, 2)), 1.5)
        planar.append(planar_case(f'random-{index}', a, b, [0, 2, 5]))
    # Exact case: |(3, 4)| = 5, so Frechet = 5 and epsilon 5 is within (25 is exact in float32).
    exact = planar_case('exact-integer', [[0, 0], [10, 0]], [[3, 4], [13, 4]], [0])
    exact['decisions'].append({'epsilon': 5.0, 'within': 1})
    planar.append(exact)

    great_circle = []
    great_circle.append(great_circle_case('meters', [[-73.99, 40.73], [-73.985, 40.735], [-73.98, 40.74], [-73.975, 40.742]], [[-73.9902, 40.7305], [-73.9851, 40.7348], [-73.9797, 40.7402]], [0, 1]))
    great_circle.append(great_circle_case('antimeridian', [[179.9, -16.0], [-179.95, -16.02], [-179.8, -16.05]], [[179.92, -16.01], [179.99, -16.015], [-179.9, -16.03], [-179.82, -16.06]], [0, 2]))
    for index in range(4):
        n = int(rng.integers(5, 30))
        m = int(rng.integers(5, 30))
        origin = (rng.uniform(-170, 170), rng.uniform(-60, 60))
        a = random_track(rng, n, origin, 0.01)
        b = random_track(rng, m, (origin[0] + 0.003, origin[1] - 0.002), 0.01)
        great_circle.append(great_circle_case(f'random-{index}', a, b, [0, 3]))

    fixture = {
        'generator': 'test/oracles/track-similarity-dtw-frechet.py',
        'versions': {
            'python': platform.python_version(),
            'numpy': np.__version__,
            'shapely': shapely.__version__,
            'similaritymeasures': getattr(similaritymeasures, '__version__', 'unknown'),
            'tslearn': tslearn.__version__,
            'dtaidistance': dtaidistance.__version__,
        },
        'earthRadius': EARTH_RADIUS,
        'tolerances': {
            'frechetRelative': 2e-6,
            'frechetAbsolutePlanar': 1e-4,
            'frechetAbsoluteMeters': 0.5,
            'dynamicTimeWarpingRelativePerStep': 2 ** -22,
            'dynamicTimeWarpingAbsolutePlanar': 1e-3,
            'dynamicTimeWarpingAbsoluteMetersPerStep': 0.5,
        },
        'planar': planar,
        'greatCircle': great_circle,
    }
    with open(OUTPUT, 'w') as file:
        json.dump(fixture, file, indent=1)
        file.write('\n')
    print(f'wrote {OUTPUT}: {len(planar)} planar and {len(great_circle)} great-circle cases')


if __name__ == '__main__':
    main()
