#!/usr/bin/env python3
"""Shapely oracle for tolerance-0 repeated-point removal on adversarial paths and rings (P1).

GPUGeometryCleanup removes exact repeats with vertex-parallel passes when `tolerance = 0` (and with
the sequential per-ring walk otherwise). This script records `shapely.remove_repeated_points(tolerance=0)`
(optionally after `set_precision(grid, mode='pointwise')`) on cases where the two formulations could
differ: all-identical paths, repeats across the closing vertex, runs reaching the first vertex,
rings that collapse (GEOS raises, recorded as "collapsed"), negative zero, and 1- and 2-vertex paths.
NaN and infinite coordinates are not recorded: GEOS does not define them, and the spec compares the
parallel and forced-sequential GPU passes bitwise on them instead.

Fixture written (relative to modules/experimental/test/oracles):
  fixtures/geometry-cleanup.json

Usage, from the repository root:
  python3 -I modules/experimental/test/oracles/geometry-cleanup.py

Needs shapely >= 2.0 and numpy. Coordinates are exact in float32. Negative zero is written as "-0".
"""
import json
import os
from importlib.metadata import version

import numpy as np
import shapely

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'fixtures', 'geometry-cleanup.json')
GENERATOR = 'modules/experimental/test/oracles/geometry-cleanup.py'

PATHS = [
    ('all_identical', [(1, 1)] * 5),
    ('repeat_across_closing_vertex', [(0, 0), (4, 0), (4, 4), (4, 4)]),
    ('run_before_closing_vertex', [(0, 0), (4, 0), (4, 4), (4, 4), (4, 4), (4, 4)]),
    ('run_reaches_first_vertex', [(2, 2), (2, 2), (2, 2), (2, 2)]),
    ('negative_zero', [(0, 0), (-0.0, -0.0), (0, -0.0), (5, 5), (-0.0, 0)]),
    ('single_vertex', [(7, 7)]),
    ('two_equal_vertices', [(7, 7), (7, 7)]),
    ('two_different_vertices', [(7, 7), (8, 8)]),
    ('middle_repeats_first', [(1, 1), (1, 1), (2, 2)]),
    ('middle_repeats_last', [(1, 1), (2, 2), (2, 2)]),
    ('alternating_repeats', [(0, 0), (0, 0), (1, 0), (1, 0), (2, 0), (2, 0), (3, 0)]),
    ('snaps_to_repeats', [(0.01, 0.01), (0.02, -0.02), (0.4, 0.4), (0.45, 0.35), (0.01, 0.01)]),
    ('empty_path', []),
]
RINGS = [
    ('square_with_repeats', [(0, 0), (4, 0), (4, 0), (4, 4), (0, 4), (0, 4), (0, 0)]),
    ('repeat_across_closing_vertex', [(0, 0), (4, 0), (4, 4), (0, 4), (0, 4), (0, 0)]),
    ('triangle_two_repeats', [(0, 0), (3, 0), (3, 0), (3, 3), (3, 3), (0, 0)]),
    ('collapses_to_segment', [(0, 0), (0, 0), (3, 3), (3, 3), (0, 0)]),
    ('collapses_to_point', [(1, 1), (1, 1), (1, 1), (1, 1)]),
    ('already_clean', [(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)]),
]
SETTINGS = [(0.0, 'no snapping'), (0.5, 'grid 0.5'), (1.0, 'grid 1')]


def encode(value):
    if value == 0 and np.signbit(value):
        return '-0'
    return float(value)


def encode_path(coords):
    return [[encode(x), encode(y)] for x, y in coords]


def expected_path(coords, grid, ring):
    if len(coords) <= 1:
        # Shapely cannot represent a one-vertex path; GPUGeometryCleanup leaves it untouched.
        return list(coords)
    try:
        shape = shapely.LinearRing(coords) if ring else shapely.LineString(coords)
        if grid > 0:
            shape = shapely.set_precision(shape, grid, mode='pointwise')
        shape = shapely.remove_repeated_points(shape, 0)
    except Exception:
        return 'collapsed'
    if shape.is_empty:
        return 'collapsed' if ring else []
    return [tuple(c) for c in shape.coords]


def build():
    cases = []
    for grid, label in SETTINGS:
        cases.append({
            'grid': grid,
            'label': label,
            'lines': [
                {'name': name, 'coords': encode_path(coords),
                 'expected': encode_path(expected_path(coords, grid, False))}
                for name, coords in PATHS
            ],
            'rings': [
                {'name': name, 'coords': encode_path(coords),
                 'expected': (lambda e: e if e == 'collapsed' else encode_path(e))(
                     expected_path(coords, grid, True))}
                for name, coords in RINGS
            ],
        })
    return cases


def main():
    fixture = {
        'cases': build(),
        'provenance': {
            'generator': GENERATOR,
            'shapely': version('shapely'),
            'numpy': version('numpy'),
            'tolerance': {
                'absolute': 0,
                'derivation': 'coordinates and grid sizes are exact in float32 and snapping to a '
                'power-of-two grid is exact, so the comparison is exact (negative zero folded).',
            },
        },
    }
    with open(OUTPUT, 'w') as handle:
        json.dump(fixture, handle, indent=1)
        handle.write('\n')
    print('wrote', os.path.relpath(OUTPUT))


if __name__ == '__main__':
    main()
