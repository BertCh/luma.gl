# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Reference great-circle kNN / radius neighbors (exact float64 haversine) and libpysal adaptive kernel
weights for `GPUNeighborSearch`.

Writes fixtures/statistics-neighbor-search.json next to this script, consumed (through
test/gpu-spatial-analysis/neighbor-search/neighbor-search-u5-oracle.ts) by gpu-neighbor-search-u5.spec.ts.
Run from anywhere with
    <venv>/bin/python -I modules/experimental/test/oracles/statistics-neighbor-search.py
in a venv with numpy and libpysal (versions are recorded in the fixture).

Scenes (seed 20261013, positions rounded to float32 and read back, so the GPU sees the same values):
- `global`: 400 points uniform on the sphere, k = 5;
- `antimeridian`: 80 points in lon 170..190 (wrapped), lat +-15, plus four points at +-179.99 / 180, k = 4;
- `pole`: 80 points at lat 82..90, 40 at lat -90..-85 and four exact-pole points, k = 4;
- `planar`: 40 points in [0, 100]^2, k = 4, libpysal `Kernel(points, k, function, fixed=False)` for the
  five kernels (libpysal 4.15 lists k non-self neighbors plus the self slot, bandwidth = k-th other
  distance * 1.0000001; the self slot is dropped here).
Distances are haversine with R = 6371008.8 m. Per row, `knn` lists [neighbor, distance] of the k nearest
and `radiusRows` those within `radius`, both ascending by id. The generator asserts a margin of more than
50 m between the k-th and (k+1)-th distance of every row and between every distance and the radius (the
first radius candidate that satisfies it is used), and more than 1e-3 between the planar k-th and
(k+1)-th distances, so cuts are tie-free.

Tolerances (stored in `tolerances`, the budgets of gpu-neighbor-search-u5.spec.ts): neighbor ids and counts
are exact because the margins above (50 m, 1e-3) exceed the float32 trigonometry error. Great-circle
distances agree to 1 m + 2e-5 relative: float32 inputs are exact, but the GPU evaluates haversine in float32
(sin/cos/asin of single precision, relative error ~ 1e-6 per call, accumulated over about ten operations and
amplified near the antipode by the asin slope), so 2e-5 relative bounds the observed error and 1 m covers
points a few meters apart. Planar kernel weights agree to 1e-6 absolute + 2e-5 relative (float32
distances, a float32 bandwidth and one exp / polynomial evaluation).
"""

import json
import math
import os
import platform
import warnings

import libpysal
import numpy as np
from libpysal.weights import Kernel

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'fixtures', 'statistics-neighbor-search.json')
EARTH_RADIUS = 6371008.8
generator = np.random.default_rng(20261013)


def encode(value):
    """JSON-safe copy: NaN -> null, infinities and -0 -> strings (decoded by test/oracles/oracle-fixture.ts)."""
    if isinstance(value, dict):
        return {key: encode(entry) for key, entry in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [encode(entry) for entry in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if math.isnan(value):
            return None
        if math.isinf(value):
            return 'Infinity' if value > 0 else '-Infinity'
        if value == 0 and math.copysign(1.0, value) < 0:
            return '-0'
        return value
    return value


def haversine(first, second):
    longitude_first, latitude_first = np.radians(first[..., 0]), np.radians(first[..., 1])
    longitude_second, latitude_second = np.radians(second[..., 0]), np.radians(second[..., 1])
    half = (
        np.sin((latitude_second - latitude_first) / 2) ** 2
        + np.cos(latitude_first) * np.cos(latitude_second) * np.sin((longitude_second - longitude_first) / 2) ** 2
    )
    return 2 * EARTH_RADIUS * np.arcsin(np.sqrt(np.clip(half, 0, 1)))


def to_float32(values):
    return np.asarray(values, dtype=np.float32).astype(np.float64)


def make_fixture(name, points, k, radius_candidates):
    positions = to_float32(points)
    count = len(positions)
    distance = haversine(positions[:, None, :], positions[None, :, :])
    np.fill_diagonal(distance, np.inf)
    order = np.argsort(distance, axis=1, kind='stable')
    knn = []
    for row in range(count):
        sorted_distance = distance[row][order[row]]
        assert sorted_distance[k] - sorted_distance[k - 1] > 50.0, (name, row, sorted_distance[k - 1], sorted_distance[k])
        ids = sorted(int(other) for other in order[row][:k])
        knn.append([[other, float(distance[row, other])] for other in ids])
    radius = None
    for candidate in radius_candidates:
        if np.abs(distance[np.isfinite(distance)] - candidate).min() > 50.0:
            radius = candidate
            break
    assert radius is not None, name
    radius_rows = []
    for row in range(count):
        ids = [int(other) for other in np.nonzero(distance[row] <= radius)[0]]
        radius_rows.append([[other, float(distance[row, other])] for other in ids])
    return {
        'name': name,
        'positions': [float(value) for value in positions.ravel()],
        'k': k,
        'radius': float(radius),
        'knn': knn,
        'radiusRows': radius_rows,
    }


def sample_sphere(count):
    z = generator.uniform(-1, 1, count)
    longitude = generator.uniform(-180, 180, count)
    return np.stack([longitude, np.degrees(np.arcsin(z))], axis=1)


global_points = sample_sphere(400)
antimeridian = np.stack(
    [((generator.uniform(170, 190, 80) + 180) % 360) - 180, generator.uniform(-15, 15, 80)], axis=1
)
antimeridian = np.vstack([antimeridian, [[179.99, 0.0], [-179.99, 0.0], [180.0, 5.0], [-180.0, 5.0]]])
pole = np.vstack(
    [
        np.stack([generator.uniform(-180, 180, 80), generator.uniform(82, 90, 80)], axis=1),
        np.stack([generator.uniform(-180, 180, 40), generator.uniform(-90, -85, 40)], axis=1),
        [[0.0, 90.0], [45.0, 89.9], [0.0, -90.0], [-120.0, -89.95]],
    ]
)
fixtures = [
    make_fixture('global', global_points, 5, [1500e3, 1480e3, 1520e3, 1550e3, 1450e3]),
    make_fixture('antimeridian', antimeridian, 4, [300e3, 290e3, 310e3, 320e3, 280e3, 330e3]),
    make_fixture('pole', pole, 4, [250e3, 240e3, 260e3, 270e3, 230e3, 280e3]),
]

planar_points = to_float32(generator.uniform(0, 100, (40, 2)))
K = 4
KERNELS = {
    'triangular': 'triangular',
    'epanechnikov': 'quadratic',
    'bisquare': 'quartic',
    'uniform': 'uniform',
    'gaussian': 'gaussian',
}
planar_distance = np.linalg.norm(planar_points[:, None] - planar_points[None], axis=2)
np.fill_diagonal(planar_distance, np.inf)
sorted_planar = np.sort(planar_distance, axis=1)
assert (sorted_planar[:, K] - sorted_planar[:, K - 1] > 1e-3).all()
planar = {'positions': [float(value) for value in planar_points.ravel()], 'k': K, 'kernels': {}}
for name, function in KERNELS.items():
    weights = Kernel(planar_points, k=K, function=function, fixed=False)
    rows = []
    for row in range(len(planar_points)):
        entries = sorted(
            [int(other), float(np.linalg.norm(planar_points[row] - planar_points[other])), float(weight)]
            for other, weight in zip(weights.neighbors[row], weights.weights[row])
            if other != row
        )
        rows.append(entries)
    planar['kernels'][name] = {'rows': rows, 'bandwidth': [float(value) for value in weights.bandwidth.ravel()]}

fixture = {
    'generator': 'test/oracles/statistics-neighbor-search.py',
    'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'libpysal': libpysal.__version__},
    'tolerances': {
        'greatCircleDistance': {'absolute': 1.0, 'relative': 2e-5},
        'planarKernelWeight': {'absolute': 1e-6, 'relative': 2e-5},
        'derivation': 'ids exact (cut margins of 50 m and 1e-3 exceed float32 error); great-circle distances '
        'are float32 haversine against float64 (1 m + 2e-5 relative); planar kernel weights are float32 '
        'distances and bandwidth through one exp or polynomial (1e-6 + 2e-5 relative)',
    },
    'reference': {'sphereRadius': EARTH_RADIUS, 'fixtures': fixtures, 'planar': planar},
}
with open(OUTPUT, 'w') as file:
    json.dump(encode(fixture), file, separators=(',', ':'))
    file.write('\n')
print('wrote', OUTPUT)
