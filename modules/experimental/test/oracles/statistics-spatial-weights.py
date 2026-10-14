# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Reference libpysal spatial weights: adaptive kernels, islands and s0 / s1 / s2 (U4), and the
construction of contiguity, lattice, kNN and distance-band weights with row standardization.

Writes fixtures/statistics-spatial-weights.json next to this script, consumed (through
test/gpu-spatial-analysis/spatial-weights/spatial-weights-u4-oracle.ts) by
gpu-spatial-weights-first-class.spec.ts, gpu-spatial-weights-first-class.node.spec.ts,
gpu-spatial-weights.spec.ts and spatial-weights-libpysal.node.spec.ts. Run from anywhere with
    <venv>/bin/python -I modules/experimental/test/oracles/statistics-spatial-weights.py
in a venv with numpy, shapely and libpysal (versions are recorded in the fixture).

`reference` (U4):
- Nine fixed float32-exact points, k = 3, `libpysal.weights.Kernel(points, k=3, function, fixed=False)`
  for the triangular, quadratic (epanechnikov), quartic (bisquare), uniform and gaussian kernels. Rows
  hold [neighbor, distance, weight] without the self slot (the GPU weights never store it), ascending by
  id; `bandwidth` is h_i = k-th distance * 1.0000001 and `selfWeight` is K(0).
- Islands: a `W` built without zero-weight entries (row 3 empty, row 4 with its only, zero-weight, slot
  omitted, row 5 empty): `W.islands`, `W.cardinalities`, `W.s0`, `W.s1`, `W.s2`.

`construction` (neighbors ascending by id, binary weights 1 unless row standardized; ids are the row
index):
- `contiguity.squareGrid`: the 7 x 5 square grid of gpu-spatial-weights.spec.ts (seed 3; ring rotations
  and reversals reproduced with the same mulberry32 generator, they do not change contiguity);
  `contiguity.randomRectangles`: 80 integer rectangles, seed 1 (overlaps, T-junctions, partial shared
  edges); `contiguity.holes`: a square with a hole filled by a second polygon, a corner contact and an
  isolated square. Each lists `polygons` (rings of [x, y]) and, for `queen` and `rook`,
  `libpysal.weights.Queen.from_iterable` / `Rook.from_iterable` over `shapely.Polygon`s.
- `lattice`: `libpysal.weights.lat2W(6, 9, rook, id_type='int')` (6 rows by 9 columns, id = row * 9 + column).
- `knn`: `libpysal.weights.KNN.from_array(points, k=4)` on 40 float32-exact random points in [0, 100]^2
  (the first seed from 20261014 upward, recorded as `knn.seed`, with a gap above 0.005 between the k-th and (k+1)-th distance in every row, so the
  documented ascending-id tie rule never matters); `distanceBand`:
  `libpysal.weights.DistanceBand(points, threshold, binary=True)` on the same points (first threshold with
  every |distance - threshold| above 0.005). `distances` are the float64 Euclidean distances of the listed
  slots.
- `rowStandardized`: `w.transform = 'r'` of the squareGrid queen weights and of the kNN weights.

Tolerances (stored in `tolerances`): neighbor sets and binary weights are compared exactly. Row-standardized
weights are one float64 division by an exactly summed integer on both sides, so they agree to
`rowStandardizedWeight` = 1e-15 relative (about 4.5 ulp, covering a different order of the exact sum).
`distance`: the TypeScript oracle computes float32 differences, squares, a sum and a square root (about
five roundings of 2^-24 = 6e-8), so 1e-6 relative bounds it with a margin of three. The U4 budgets are those
of the existing specs: kernel weights agree to 1e-12 in float64 (`kernelWeightCpu`, a few ulp of one
formula) and to 1e-5 absolute on the GPU (`kernelWeightGpu`: float32 distances, bandwidth and one exp or
polynomial, weights at most 0.4).
"""

import json
import math
import os
import platform
import warnings

import libpysal
import numpy as np
import shapely
from libpysal.weights import DistanceBand, KNN, Kernel, Queen, Rook, W, lat2W
from shapely.geometry import Polygon

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'fixtures', 'statistics-spatial-weights.json')
MASK = 0xFFFFFFFF


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


# --- U4: kernels, islands ---------------------------------------------------------------------------

POINTS = [[0.0, 0.0], [1.0, 0.5], [2.5, 0.0], [0.5, 2.0], [2.0, 2.5], [3.5, 2.0], [1.0, 4.0], [3.0, 4.0], [4.5, 0.5]]
K = 3
KERNELS = {
    'triangular': 'triangular',
    'epanechnikov': 'quadratic',
    'bisquare': 'quartic',
    'uniform': 'uniform',
    'gaussian': 'gaussian',
}

points = np.array(POINTS)
reference = {'points': POINTS, 'k': K, 'kernels': {}}
for name, function in KERNELS.items():
    kernel_weights = Kernel(points, k=K, function=function, fixed=False)  # k + 1 includes self internally
    rows = []
    for row in range(len(points)):
        entries = []
        for other, weight in zip(kernel_weights.neighbors[row], kernel_weights.weights[row]):
            if other == row:
                continue  # GPU weights never hold the self slot; K(0) is getGPUSpatialWeightsKernelValue
            distance = float(np.linalg.norm(points[row] - points[other]))
            entries.append([int(other), distance, float(weight)])
        entries.sort()
        rows.append(entries)
    self_weight = float(kernel_weights.weights[0][kernel_weights.neighbors[0].index(0)])
    reference['kernels'][name] = {
        'rows': rows,
        'bandwidth': [float(value) for value in kernel_weights.bandwidth.ravel()],
        'selfWeight': self_weight,
    }

# Islands: W without zero-weight entries.
island_neighbors = {0: [1, 2], 1: [0, 2], 2: [0, 1], 3: [], 4: [], 5: []}
island_weights = {0: [1.0, 2.0], 1: [1.0, 0.5], 2: [2.0, 0.5], 3: [], 4: [], 5: []}
island_w = W(island_neighbors, island_weights, silence_warnings=True)
cardinalities = [int(island_w.cardinalities[row]) for row in range(6)]
reference['islands'] = {
    'islands': [int(row) for row in island_w.islands],
    'cardinalities': cardinalities,
    'minCardinality': min(cardinalities),
    'maxCardinality': max(cardinalities),
    's0': float(island_w.s0),
    's1': float(island_w.s1),
    's2': float(island_w.s2),
}


# --- construction -----------------------------------------------------------------------------------

def create_seeded_random(seed):
    """mulberry32, as createSeededRandom in neighbor-search-oracle.ts (values in [0, 1))."""
    state = [seed & MASK]

    def next_value():
        state[0] = (state[0] + 0x6D2B79F5) & MASK
        t = state[0]
        t = ((t ^ (t >> 15)) * (t | 1)) & MASK
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & MASK)) & MASK
        return ((t ^ (t >> 14)) & MASK) / 4294967296

    return next_value


def create_square_grid(columns, rows, seed):
    random = create_seeded_random(seed)
    polygons = []
    for y in range(rows):
        for x in range(columns):
            ring = [[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1]]
            rotation = math.floor(random() * 4)
            ring = ring[rotation:] + ring[:rotation]
            if random() < 0.5:
                ring.reverse()
            polygons.append([ring])
    return polygons


def create_random_rectangles(count, seed):
    random = create_seeded_random(seed)
    polygons = []
    for _ in range(count):
        x = math.floor(random() * 16)
        y = math.floor(random() * 16)
        width = 1 + math.floor(random() * 3)
        height = 1 + math.floor(random() * 3)
        polygons.append([[[x, y], [x + width, y], [x + width, y + height], [x, y + height]]])
    return polygons


HOLES = [
    [[[0, 0], [6, 0], [6, 6], [0, 6]], [[2, 2], [4, 2], [4, 4], [2, 4]]],
    [[[2, 2], [4, 2], [4, 4], [2, 4]]],
    [[[6, 6], [8, 6], [8, 8], [6, 8]]],
    [[[20, 20], [21, 20], [21, 21], [20, 21]]],
]


def to_shapely(rings):
    return Polygon(rings[0], rings[1:])


def pack(weights):
    """Rows of ascending neighbor ids with their weights."""
    neighbors, values = [], []
    for row in range(weights.n):
        entries = sorted(zip(weights.neighbors[row], weights.weights[row]))
        neighbors.append([int(other) for other, _ in entries])
        values.append([float(weight) for _, weight in entries])
    return {'neighbors': neighbors, 'weights': values}


def standardize(weights):
    copy = W(dict(weights.neighbors), dict(weights.weights), silence_warnings=True)
    copy.transform = 'r'
    return copy


def contiguity(polygons):
    shapes = [to_shapely(rings) for rings in polygons]
    return {
        'polygons': polygons,
        'queen': pack(Queen.from_iterable(shapes)),
        'rook': pack(Rook.from_iterable(shapes)),
    }


square_grid = contiguity(create_square_grid(7, 5, 3))
square_grid['columns'] = 7
square_grid['rows'] = 5
random_rectangles = contiguity(create_random_rectangles(80, 1))
holes = contiguity(HOLES)

lattice = {'width': 9, 'height': 6}
for criterion in ('rook', 'queen'):
    lattice[criterion] = pack(lat2W(6, 9, rook=criterion == 'rook', id_type='int'))

# kNN and distance band on 40 random float32-exact points.
NEAREST = 4
SEED = 20261014
while True:
    # First seed whose every row has a gap above 0.005 between the k-th and (k+1)-th distance.
    generator = np.random.default_rng(SEED)
    planar = np.asarray(generator.uniform(0, 100, (40, 2)), dtype=np.float32).astype(np.float64)
    planar_distance = np.linalg.norm(planar[:, None] - planar[None], axis=2)
    np.fill_diagonal(planar_distance, np.inf)
    sorted_distance = np.sort(planar_distance, axis=1)
    if (sorted_distance[:, NEAREST] - sorted_distance[:, NEAREST - 1] > 0.005).all():
        break
    SEED += 1
threshold = None
for candidate in (14.0, 13.0, 15.0, 12.0, 16.0, 11.0):
    if np.abs(planar_distance[np.isfinite(planar_distance)] - candidate).min() > 0.005:
        threshold = candidate
        break
assert threshold is not None


def with_distances(weights):
    packed = pack(weights)
    packed['distances'] = [
        [float(planar_distance[row, other]) for other in neighbors] for row, neighbors in enumerate(packed['neighbors'])
    ]
    return packed


knn_weights = KNN.from_array(planar, k=NEAREST)
band_weights = DistanceBand(planar, threshold=threshold, binary=True, silence_warnings=True)
knn = {'seed': SEED, 'positions': [float(value) for value in planar.ravel()], 'k': NEAREST, 'bounds': [0, 0, 100, 100]}
knn.update(with_distances(knn_weights))
distance_band = {'threshold': float(threshold)}
distance_band.update(with_distances(band_weights))

construction = {
    'contiguity': {'squareGrid': square_grid, 'randomRectangles': random_rectangles, 'holes': holes},
    'lattice': lattice,
    'knn': knn,
    'distanceBand': distance_band,
    'rowStandardized': {
        'squareGridQueen': pack(standardize(Queen.from_iterable([to_shapely(rings) for rings in square_grid['polygons']]))),
        'knn': pack(standardize(knn_weights)),
    },
}

fixture = {
    'generator': 'test/oracles/statistics-spatial-weights.py',
    'versions': {
        'python': platform.python_version(),
        'numpy': np.__version__,
        'shapely': shapely.__version__,
        'libpysal': libpysal.__version__,
    },
    'tolerances': {
        'rowStandardizedWeight': {'relative': 1e-15},
        'distance': {'relative': 1e-6},
        'kernelWeightCpu': {'absolute': 1e-12},
        'kernelWeightGpu': {'absolute': 1e-5},
        'derivation': 'neighbors and binary weights exact; row standardization is one float64 division by an '
        'exact integer sum (4.5 ulp); float32 distances through five roundings of 6e-8 (x3 margin = 1e-6); '
        'U4 budgets are the existing specs (float64 formula 1e-12, float32 GPU kernel 1e-5)',
    },
    'reference': reference,
    'construction': construction,
}
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, 'w') as file:
    json.dump(encode(fixture), file, separators=(',', ':'))
    file.write('\n')
print('wrote', OUTPUT, 'threshold', threshold, 'islands in band', sum(1 for row in band_weights.neighbors.values() if not row))
