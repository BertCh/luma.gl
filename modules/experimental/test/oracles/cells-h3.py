# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Reference H3 cell indexing (point to cell) from h3-py, with the distance of every point to the
boundary of its cell.

Writes fixtures/cell-indexing-h3.json next to this script, consumed by
test/gpu-spatial-analysis/cell-indexing/gpu-point-to-cell.spec.ts ('GPUPointToCell H3 equals h3-py
away from cell edges'). Cell cover and rollup fixtures come from geometry-properties.py
(cell-cover-h3.json, cell-rollup.json). Run with
    <venv>/bin/python -I modules/experimental/test/oracles/cells-h3.py
in a venv with numpy and h3 >= 4 (versions are recorded in the fixture).

Points (float32 longitude / latitude, exactly what the GPU reads): equal-area random points (seed
20261014) plus hand-picked rows: poles, the antimeridian, signed zero, icosahedron face centers and
the 12 pentagon centers at every tested resolution, points on face seams and longitudes outside
[-180, 180] (h3 wraps them). Non-finite rows are listed separately with the policy result (cell 0,
invalid).

Reference: `h3.latlng_to_cell(lat, lng, resolution)` on the float64 value of each float32 input.
`margins[r][i]` is the great-circle distance (radians) from point i to the boundary of its cell: H3
cell edges are straight lines in the gnomonic projection of an icosahedron face, so each edge between
consecutive `cell_to_boundary` vertices (including distortion vertices) is a great-circle arc.
`neighbors[r][i]` lists `grid_disk(cell, 1)` for points within `marginBound` of an edge.

Tolerance (stored): the GPU evaluates the forward transform in float32 on float32 inputs. Measured
against this fixture, its mismatches (1 of 23 near-edge points at resolution 9, 56 of 425 at 12) lie
at most 9.9e-8 rad (63 cm) from an edge, about 6 float32 ulps of an angle near 1. A point farther
than `marginBound` = 2e-7 rad (twice that) from every edge of its cell must get exactly the h3 cell;
a nearer point may get a grid_disk(cell, 1) neighbor instead.
"""

import json
import math
import os
import platform

import h3
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'fixtures', 'cell-indexing-h3.json')
SEED = 20261014
RANDOM_COUNT = 3000
RESOLUTIONS = [0, 3, 4, 6, 9, 12]
MARGIN_BOUND = 2e-7


def to_float32(value):
    return float(np.float32(value))


def to_unit_vector(latitude, longitude):
    lat = math.radians(latitude)
    lng = math.radians(longitude)
    return np.array([math.cos(lat) * math.cos(lng), math.cos(lat) * math.sin(lng), math.sin(lat)])


def get_arc_distance(point, start, end):
    """Great-circle distance from unit vector `point` to the minor arc `start` -> `end`."""
    normal = np.cross(start, end)
    length = np.linalg.norm(normal)
    endpoint_distance = min(
        math.atan2(np.linalg.norm(np.cross(point, start)), np.dot(point, start)),
        math.atan2(np.linalg.norm(np.cross(point, end)), np.dot(point, end)),
    )
    if length < 1e-300:
        return endpoint_distance
    normal = normal / length
    sine = float(np.dot(point, normal))
    projection = point - sine * normal
    # The foot of the perpendicular lies on the arc when it is between start and end.
    if np.dot(np.cross(start, projection), normal) >= 0 and np.dot(np.cross(projection, end), normal) >= 0:
        return abs(math.asin(max(-1.0, min(1.0, sine))))
    return endpoint_distance


def get_margin(latitude, longitude, cell):
    point = to_unit_vector(latitude, longitude)
    vertices = [to_unit_vector(lat, lng) for lat, lng in h3.cell_to_boundary(cell)]
    return min(get_arc_distance(point, vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(len(vertices)))


def create_points():
    generator = np.random.default_rng(SEED)
    points = []
    for _ in range(RANDOM_COUNT):
        longitude = generator.random() * 360 - 180
        latitude = math.degrees(math.asin(generator.random() * 2 - 1))
        points.append((longitude, latitude, 'random'))
    special = [
        (0, 90), (0, -90), (180, 90), (-180, -90), (180, 0), (-180, 0), (0, 0), (-0.0, -0.0),
        (45, 0), (-45, 0), (135, 0), (-135, 0), (0, 35.264389), (90, -35.264389),
        (179.99999, 85.05113), (-179.99999, -85.05113), (10.40744, 57.64911), (-3.7038, 40.4168),
        (-111.709298, 40.256312), (1e-30, 1e-30), (360, 10), (-540, -10), (179.9999, -0.0001),
    ]
    points += [(longitude, latitude, 'special') for longitude, latitude in special]
    for resolution in RESOLUTIONS:
        for pentagon in h3.get_pentagons(resolution):
            latitude, longitude = h3.cell_to_latlng(pentagon)
            points.append((longitude, latitude, 'pentagon center'))
    for base in h3.get_res0_cells():
        latitude, longitude = h3.cell_to_latlng(base)
        points.append((longitude, latitude, 'base cell center'))
    return [(to_float32(longitude), to_float32(latitude), kind) for longitude, latitude, kind in points]


points = create_points()
cells = {}
margins = {}
neighbors = {}
for resolution in RESOLUTIONS:
    keys = []
    distances = []
    near = {}
    for index, (longitude, latitude, _) in enumerate(points):
        cell = h3.latlng_to_cell(latitude, longitude, resolution)
        margin = get_margin(latitude, longitude, cell)
        keys.append(cell)
        # Three significant digits: the margins only gate which points may differ.
        distances.append(float('%.3g' % margin))
        if margin < MARGIN_BOUND:
            near[str(index)] = sorted(h3.grid_disk(cell, 1))
    cells[str(resolution)] = keys
    margins[str(resolution)] = distances
    neighbors[str(resolution)] = near
    print('resolution', resolution, 'points within the margin bound', len(near), 'of', len(points))

fixture = {
    'generator': 'test/oracles/cells-h3.py',
    'seed': SEED,
    'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'h3': h3.__version__},
    'oracle': 'h3.latlng_to_cell on the float64 value of the float32 input; margins are great-circle '
    'distances (radians) to the cell boundary arcs of h3.cell_to_boundary',
    'tolerances': {
        'marginBound': MARGIN_BOUND,
        'derivation': 'GPU float32 mismatches measured at most 9.9e-8 rad from an edge (about 6 ulps of '
        'an angle near 1); twice that: points farther than marginBound from an edge must match exactly, '
        'nearer points may get a grid_disk(cell, 1) neighbor',
    },
    'resolutions': RESOLUTIONS,
    'points': [[longitude, latitude] for longitude, latitude, _ in points],
    'kinds': [kind for _, _, kind in points],
    'cells': cells,
    'margins': margins,
    'neighbors': neighbors,
    'edgePolicy': {
        'policy': 'a non-finite longitude or latitude gives cell 0 and validity 0',
        'points': [[None, 10], [10, None], [None, None], ['Infinity', 10], [10, '-Infinity']],
    },
}
with open(OUTPUT, 'w') as file:
    json.dump(fixture, file, separators=(',', ':'))
    file.write('\n')
print('wrote', OUTPUT, len(points), 'points')
