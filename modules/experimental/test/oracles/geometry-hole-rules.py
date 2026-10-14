#!/usr/bin/env python3
"""Shapely oracle for ring roles by position (GeoArrow polygonOffsets), review D-13.

GPUGeometryMeasures and GPUShapeDescriptors used to decide shells and holes from winding (default)
or from the first ring of a feature, so a multi-polygon whose parts wind differently, or a holed
polygon whose hole winds like its shell, gave wrong areas and centroids. In the four-level layout
(`featureOffsets`, `polygonOffsets`, `ringOffsets`) the first ring of each polygon is the shell and
later rings are holes. This script builds such cases,
with coordinates exact in float32, and records shapely's area, perimeter and centroid.

Fixture written (relative to modules/experimental/test/gpu-spatial-analysis):
  geometry-measures/fixtures/hole-rules.json

Usage, from the repository root:
  python3 -I modules/experimental/test/oracles/geometry-hole-rules.py

Needs shapely >= 2.0 and numpy. Versions and tolerances are recorded in the fixture.
"""
import json
import math
import os
from importlib.metadata import version

import numpy as np
import shapely

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(
    HERE, '..', 'gpu-spatial-analysis', 'geometry-measures', 'fixtures', 'hole-rules.json'
)
GENERATOR = 'modules/experimental/test/oracles/geometry-hole-rules.py'


def square(x, y, size, clockwise):
    ring = [(x, y), (x + size, y), (x + size, y + size), (x, y + size)]
    return ring[::-1] if clockwise else ring


# Each feature is a list of polygons; a polygon is a list of rings (shell first). Winding is chosen
# per ring on purpose: the cases are the ones the winding and first-ring rules get wrong.
CASES = [
    ('mixed_winding_multipolygon', [
        [square(0, 0, 10, False)],
        [square(20, 0, 4, True)],
    ]),
    ('hole_wound_like_shell', [
        [square(0, 0, 10, False), square(2, 2, 4, False)],
    ]),
    ('holed_polygon_then_clockwise_part', [
        [square(0, 0, 10, False), square(2, 2, 4, True)],
        [square(30, 0, 6, True), square(32, 2, 2, True)],
    ]),
    ('clockwise_shell_ccw_hole', [
        [square(0, 0, 8, True), square(2, 2, 2, False)],
    ]),
    ('three_parts_mixed', [
        [square(0, 0, 3, True)],
        [square(10, 0, 5, False), square(11, 1, 2, False)],
        [square(20, 0, 2, True)],
    ]),
]


def build():
    positions, ring_offsets, polygon_offsets, feature_offsets = [], [0], [0], [0]
    expected = []
    for name, polygons in CASES:
        shapes = []
        for polygon in polygons:
            for ring in polygon:
                positions.extend(ring)
                ring_offsets.append(len(positions))
            polygon_offsets.append(len(ring_offsets) - 1)
            shapes.append(shapely.Polygon(polygon[0], polygon[1:]))
        feature_offsets.append(len(polygon_offsets) - 1)
        geometry = shapely.MultiPolygon(shapes) if len(shapes) > 1 else shapes[0]
        area = geometry.area
        perimeter = geometry.length
        centroid = geometry.centroid
        expected.append({
            'name': name,
            'area': area,
            'perimeter': perimeter,
            'centroid': [centroid.x, centroid.y],
            'polsbyPopper': 4 * math.pi * area / (perimeter * perimeter),
        })
    assert np.array_equal(np.asarray(positions, dtype=np.float32), np.asarray(positions))
    return {
        'positions': [list(map(float, p)) for p in positions],
        'ringOffsets': ring_offsets,
        'polygonOffsets': polygon_offsets,
        'featureOffsets': feature_offsets,
        'expected': expected,
    }


def main():
    fixture = build()
    fixture['provenance'] = {
        'generator': GENERATOR,
        'shapely': version('shapely'),
        'numpy': version('numpy'),
        'tolerance': {
            'relative': 1e-4,
            'derivation': 'coordinates are small integers, exact in float32; area, perimeter and '
            'centroid sums are Neumaier-compensated float32, so the error is a few ulp (1e-6) and '
            '1e-4 relative leaves two orders of margin.',
        },
    }
    with open(OUTPUT, 'w') as handle:
        json.dump(fixture, handle, indent=1)
        handle.write('\n')
    print('wrote', os.path.relpath(OUTPUT))


if __name__ == '__main__':
    main()
