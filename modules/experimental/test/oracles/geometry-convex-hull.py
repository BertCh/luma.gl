#!/usr/bin/env python3
"""Oracle generator for luma.gl GPUGroupConvexHull (algorithms roadmap P1, Akl-Toussaint filter).

References: shapely 2.x `MultiPoint.convex_hull` (GEOS) and scipy.spatial.ConvexHull (Qhull). Both drop
collinear boundary points, like the contributor's monotone chain, so the hull vertex set of a group
is the same set of points. Coordinates are multiples of 0.25 below 256, exactly representable in
float32, and the contributor's power-of-two lattice maps them to integers with no rounding, so the
lattice hull equals the exact float64 hull and the GPU result can be compared to these vertex sets
exactly. Degenerate groups (one point, all collinear) have a Point or LineString hull in shapely and
no scipy hull (Qhull raises); their vertex set is the distinct extreme points.

Fixture: `groups`, each with `points` (flat x, y), `hull` (sorted vertex list `[x, y]`) and `area`.
The contributor's hull order (counter-clockwise from the smallest vertex) is checked on the GPU side.

Usage (writes fixtures/geometry-convex-hull.json):  python3 -I geometry-convex-hull.py
Needs numpy, scipy and shapely (versions recorded in the fixture).
"""
import json
import os
from importlib.metadata import version

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from shapely.geometry import MultiPoint

HERE = os.path.dirname(os.path.abspath(__file__))
GENERATOR = 'modules/experimental/test/oracles/geometry-convex-hull.py'


def snap(points):
    return np.round(np.asarray(points, dtype=np.float64) * 4.0) / 4.0


def make_groups(rng):
    angle = rng.uniform(0, 2 * np.pi, 2000)
    radius = np.sqrt(rng.uniform(0, 1, 2000)) * 100
    ring_angle = np.linspace(0, 2 * np.pi, 300, endpoint=False)
    triangle = np.array([[10, 10], [200, 20], [90, 190]], dtype=np.float64)
    interior = triangle[0] + rng.uniform(0, 1, (400, 2)) @ np.array([[0.5, 0], [0, 0.0]])
    weights = rng.dirichlet([1, 1, 1], 400)
    inside = weights @ triangle
    return [
        ('gaussian blob', rng.normal([120, 130], [14, 9], size=(3000, 2))),
        ('uniform square', rng.uniform(0, 250, (2000, 2))),
        ('disk', np.stack([120 + radius * np.cos(angle), 120 + radius * np.sin(angle)], axis=1)),
        ('circle', np.stack([120 + 100 * np.cos(ring_angle), 120 + 100 * np.sin(ring_angle)], axis=1)),
        ('integer grid', np.array([[x, y] for x in range(12) for y in range(9)], dtype=np.float64)),
        ('diagonal with duplicates', np.array([[i * 3, i * 3] for i in range(30)] + [[6, 6], [9, 9]], dtype=np.float64)),
        ('two points', np.array([[5, 5], [17, 9]], dtype=np.float64)),
        ('one point repeated', np.array([[42, 24]] * 5, dtype=np.float64)),
        ('triangle with interior', np.vstack([triangle, inside])),
        ('thin sliver', np.stack([rng.uniform(0, 250, 500), 100 + rng.uniform(0, 0.5, 500)], axis=1)),
    ]


def main():
    rng = np.random.default_rng(20261014)
    groups = []
    for name, raw in make_groups(rng):
        points = np.clip(snap(raw), 0.0, 255.75)
        assert np.all(points.astype(np.float32).astype(np.float64) == points)
        shape = MultiPoint(points.tolist()).convex_hull
        if shape.geom_type == 'Polygon':
            coordinates = list(shape.exterior.coords)[:-1]
        elif shape.geom_type == 'LineString':
            coordinates = list(shape.coords)
        else:
            coordinates = [tuple(shape.coords[0])]
        hull = sorted([float(x), float(y)] for x, y in coordinates)
        if shape.geom_type == 'Polygon':
            try:
                scipy_hull = ConvexHull(points)
            except QhullError:
                scipy_hull = None
            if scipy_hull is not None:
                scipy_vertices = sorted(points[scipy_hull.vertices].tolist())
                assert scipy_vertices == hull, name
                assert abs(scipy_hull.volume - shape.area) <= 1e-9 * max(1.0, shape.area), name
        groups.append({
            'name': name,
            'points': points.ravel().tolist(),
            'hull': hull,
            'area': float(shape.area),
        })
    fixture = {
        'generator': GENERATOR,
        'versions': {
            'numpy': version('numpy'),
            'scipy': version('scipy'),
            'shapely': version('shapely'),
        },
        'tolerances': {
            'vertexSets': 'exact: coordinates are float32-representable and the lattice is exact',
            'area': 'relative 1e-9 (float64 shoelace of the GPU hull)',
        },
        'groups': groups,
    }
    path = os.path.join(HERE, 'fixtures', 'geometry-convex-hull.json')
    with open(path, 'w') as handle:
        json.dump(fixture, handle)
    print('wrote', path, os.path.getsize(path), 'bytes')


if __name__ == '__main__':
    main()
