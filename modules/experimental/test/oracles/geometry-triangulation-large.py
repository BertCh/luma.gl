#!/usr/bin/env python3
"""Oracle generator for the monotone path of luma.gl GPUPolygonTriangulation (algorithms roadmap P2).

References: shapely 2.x for the polygon area and validity (GEOS), and
`shapely.constrained_delaunay_triangles` (GEOS constrained Delaunay) for an independent
triangulation of the same polygon: its triangle count must be `vertices + 2 holes - 2` and its
triangle areas must sum to the polygon area. The GPU result is checked against the Shapely area
and against the area sum of the constrained Delaunay triangles; the two triangulations are not
compared triangle by triangle (earcut-like and monotone triangulations differ from Delaunay).

The polygons are built from integers only, so the GPU spec can rebuild every vertex bit for bit
without a fixture of coordinates (they would be megabytes): `createSkylinePolygon`,
`createSpiral` and `createDoubleComb` in `polygon-triangulation-monotone-cases.ts` are twins of
`skyline`, `spiral` and `double_comb` below, with a splitmix32 hash in place of a random generator.
Each case records a checksum of its coordinates that the specs recompute, so a drift between the
two generators fails loudly. All coordinates are integers below 2^22, exact in float32.

Fixture: `cases`, each with `family`, `params`, `vertexCount`, `holeCount`, `triangleCount`
(`vertexCount + 2 holeCount - 2`), `checksum`, `area` (Shapely), `valid`, and the constrained
Delaunay results `cdtArea` and `cdtTriangleCount`.

Usage (writes fixtures/geometry-triangulation-large.json):  python3 -I geometry-triangulation-large.py
Needs shapely 2.1 or later (versions recorded in the fixture).
"""
import json
import os
import time
from importlib.metadata import version

import shapely
from shapely.geometry import Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
GENERATOR = 'modules/experimental/test/oracles/geometry-triangulation-large.py'
MASK = 0xFFFFFFFF


def splitmix32(seed):
    """Deterministic 32-bit hash stream; the TypeScript twin uses Math.imul."""
    state = seed & MASK
    while True:
        state = (state + 0x9E3779B9) & MASK
        z = state
        z = ((z ^ (z >> 16)) * 0x85EBCA6B) & MASK
        z = ((z ^ (z >> 13)) * 0xC2B2AE35) & MASK
        yield (z ^ (z >> 16)) & MASK


def skyline(steps, seed, holes):
    """Bar graph outline of `steps` columns: heights differ between neighbors but repeat often
    (16 to 55), columns are 1 to 3 wide, plus `holes` 2x2 square holes under the lowest column."""
    stream = splitmix32(seed)
    heights = []
    widths = []
    previous = 0
    for _ in range(steps):
        height = 16 + next(stream) % 40
        if height == previous:
            height = 16 + (height - 16 + 1) % 40
        previous = height
        heights.append(height)
        widths.append(1 + next(stream) % 3)
    columns = [0]
    for width in widths:
        columns.append(columns[-1] + width)
    total = columns[-1]
    shell = [(0, 0), (total, 0)]
    for index in range(steps - 1, -1, -1):
        shell.append((columns[index + 1], heights[index]))
        shell.append((columns[index], heights[index]))
    rings = [shell]
    for hole in range(holes):
        x = 3 * hole + 2
        if x + 2 >= total:
            raise ValueError('too many holes for the skyline width')
        rings.append([(x, 4), (x, 6), (x + 2, 6), (x + 2, 4)])
    return rings


def spiral(loops):
    """Rectilinear spiral corridor of half width 1 (4 segments per loop, loops 4 apart)."""
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    path = [(0, 0)]
    length = 8 * loops + 8
    x = y = 0
    for segment in range(4 * loops):
        dx, dy = directions[segment % 4]
        if segment >= 2 and segment % 2 == 0:
            length -= 4
        x += dx * length
        y += dy * length
        path.append((x, y))
    count = len(path) - 1

    def normal(index):
        dx, dy = directions[index % 4]
        return (-dy, dx)

    def offset(point, a, b, sign):
        nax, nay = normal(a)
        nbx, nby = normal(b) if b >= 0 else (0, 0)
        return (point[0] + sign * (nax + nbx), point[1] + sign * (nay + nby))

    left = [offset(path[0], 0, -1, 1)]
    right = [offset(path[0], 0, -1, -1)]
    for index in range(1, count):
        left.append(offset(path[index], index - 1, index, 1))
        right.append(offset(path[index], index - 1, index, -1))
    left.append(offset(path[count], count - 1, -1, 1))
    right.append(offset(path[count], count - 1, -1, -1))
    return [left + right[::-1]]


def double_comb(teeth):
    """Thick bar with `teeth` teeth down and `teeth` up: a split and a merge per tooth."""
    ring = [(0, 7)]
    for k in range(teeth):
        ring += [(4 * k + 1, 7), (4 * k + 1, 0), (4 * k + 3, 0), (4 * k + 3, 7)]
    ring += [(4 * teeth, 7), (4 * teeth, 13)]
    for k in range(teeth - 1, -1, -1):
        ring += [(4 * k + 3, 13), (4 * k + 3, 20), (4 * k + 1, 20), (4 * k + 1, 13)]
    ring.append((0, 13))
    return [ring]


def checksum(rings):
    """Sum of x + 3 y + row over all rows, modulo 2^32 (the specs recompute it)."""
    total = 0
    row = 0
    for ring in rings:
        for x, y in ring:
            total += x + 3 * y + row
            row += 1
    return total & MASK


CASES = [
    # name, family, params
    ('skyline-5k', 'skyline', {'steps': 2499, 'seed': 11, 'holes': 0}),
    ('skyline-50k', 'skyline', {'steps': 24999, 'seed': 12, 'holes': 0}),
    ('skyline-200k', 'skyline', {'steps': 99999, 'seed': 13, 'holes': 0}),
    ('skyline-holes-20k', 'skyline', {'steps': 4000, 'seed': 14, 'holes': 2000}),
    # The constrained Delaunay of a spiral is superlinear in GEOS (about 12 s at 5k vertices).
    ('spiral-5k', 'spiral', {'loops': 625}),
    ('double-comb-5k', 'doubleComb', {'teeth': 625}),
    ('double-comb-200k', 'doubleComb', {'teeth': 25000}),
]


def build(family, params):
    if family == 'skyline':
        return skyline(params['steps'], params['seed'], params['holes'])
    if family == 'spiral':
        return spiral(params['loops'])
    if family == 'doubleComb':
        return double_comb(params['teeth'])
    raise ValueError(family)


def main():
    cases = []
    for name, family, params in CASES:
        started = time.time()
        rings = build(family, params)
        polygon = Polygon(rings[0], rings[1:])
        vertex_count = sum(len(ring) for ring in rings)
        hole_count = len(rings) - 1
        triangles = shapely.constrained_delaunay_triangles(polygon)
        cdt_area = float(sum(triangle.area for triangle in triangles.geoms))
        case = {
            'name': name,
            'family': family,
            'params': params,
            'vertexCount': vertex_count,
            'holeCount': hole_count,
            'triangleCount': vertex_count + 2 * hole_count - 2,
            'checksum': checksum(rings),
            'area': float(polygon.area),
            'valid': bool(polygon.is_valid),
            'cdtArea': cdt_area,
            'cdtTriangleCount': len(triangles.geoms),
        }
        cases.append(case)
        print(name, vertex_count, 'valid', case['valid'], 'area', case['area'], 'cdt', cdt_area,
              len(triangles.geoms), '%.1fs' % (time.time() - started))
    fixture = {
        'generator': GENERATOR,
        'versions': {'shapely': version('shapely'), 'geos': shapely.geos_version_string},
        'tolerances': {
            'area': 'relative 1e-9: float64 sum of the GPU triangle areas against Shapely and the '
            'constrained Delaunay area sum (integer coordinates, so the polygon area is exact)',
            'triangleCount': 'exact: vertices + 2 holes - 2',
        },
        'cases': cases,
    }
    path = os.path.join(HERE, 'fixtures', 'geometry-triangulation-large.json')
    with open(path, 'w') as handle:
        json.dump(fixture, handle)
    print('wrote', path, os.path.getsize(path), 'bytes')


if __name__ == '__main__':
    main()
