#!/usr/bin/env python3
"""Oracle generator for GPUDotDensity / GPURandomPointsInPolygon placement.

Reference implementation: shapely (GEOS). The fixture polygons are rectilinear, so shapely's
`contains` is turned into an exact lookup table the JavaScript spec can use without shapely:
the polygon's vertex coordinates split the plane into grid cells, and `inside[row][column]` is
`polygon.contains(cell center)` per feature. A dot is inside the feature iff its grid cell is
flagged (a dot within a tolerance of a grid line is checked against the neighbouring cells too).
Also recorded: shapely area, bounding-box fill (area / bbox area) and validity per feature, plus
the area of every cell so the spec can check that dots are uniform in area.

Features:
  0  C shape, 100 x 100 bounding box, arms 1.7 thick (about 5% bbox fill) - the case where
     bounding-box rejection with 32 attempts lost 19% of the dots.
  1  square [0, 10]^2 with a hole [2, 8]^2, plus a disjoint square [20, 30] x [0, 10] (multipolygon).
  2  square with a hole whose winding equals the shell's (orientation independence).

Usage (writes fixtures/geometry-dot-density.json):  python3 -I geometry-dot-density.py
Needs shapely (version recorded in the fixture).
"""
import json
import os
from importlib.metadata import version

from shapely.geometry import MultiPolygon, Point, Polygon

HERE = os.path.dirname(os.path.abspath(__file__))

C_THICKNESS = 1.7
C_SHAPE = [
    [(0, 0), (100, 0), (100, C_THICKNESS), (C_THICKNESS, C_THICKNESS), (C_THICKNESS, 100 - C_THICKNESS),
     (100, 100 - C_THICKNESS), (100, 100), (0, 100)],
]

# Each feature: list of polygons, each a list of rings (shell first).
FEATURES = [
    [C_SHAPE],
    [
        [[(0, 0), (10, 0), (10, 10), (0, 10)], [(2, 2), (8, 2), (8, 8), (2, 8)]],
        [[(20, 0), (30, 0), (30, 10), (20, 10)]],
    ],
    # Shell and hole both counter-clockwise.
    [[[(40, 0), (50, 0), (50, 10), (40, 10)], [(42, 2), (48, 2), (48, 8), (42, 8)]]],
]


def build_geometry(feature):
    polygons = [Polygon(rings[0], rings[1:]) for rings in feature]
    return polygons[0] if len(polygons) == 1 else MultiPolygon(polygons)


def build_feature(feature):
    geometry = build_geometry(feature)
    xs = sorted({x for rings in feature for ring in rings for x, _ in ring})
    ys = sorted({y for rings in feature for ring in rings for _, y in ring})
    inside = [
        [
            geometry.contains(Point((xs[column] + xs[column + 1]) / 2, (ys[row] + ys[row + 1]) / 2))
            for column in range(len(xs) - 1)
        ]
        for row in range(len(ys) - 1)
    ]
    minx, miny, maxx, maxy = geometry.bounds
    return {
        'xs': xs,
        'ys': ys,
        'inside': inside,
        'area': geometry.area,
        'boundingBoxFill': geometry.area / ((maxx - minx) * (maxy - miny)),
        'valid': geometry.is_valid,
        'rings': [[[list(vertex) for vertex in ring] for ring in rings] for rings in feature],
    }


def main():
    features = [build_feature(feature) for feature in FEATURES]
    for index, feature in enumerate(features):
        assert feature['valid'], f'feature {index} is not valid'
    fixture = {
        'generator': 'geometry-dot-density.py',
        'shapely': version('shapely'),
        'note': 'inside[row][column] = shapely contains() of the grid cell center; cells span '
        'xs[column]..xs[column + 1] and ys[row]..ys[row + 1].',
        'features': features,
    }
    path = os.path.join(HERE, 'fixtures', 'geometry-dot-density.json')
    with open(path, 'w') as file:
        json.dump(fixture, file, indent=1)
    print(
        'wrote', path, 'fills', [round(feature['boundingBoxFill'], 4) for feature in features],
        'shapely', fixture['shapely']
    )


if __name__ == '__main__':
    main()
