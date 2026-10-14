#!/usr/bin/env python3
"""Oracle generator for the luma.gl GPUTerrainFlow depression fill (spill graph and relaxation).

Reference implementation: pysheds `Grid.fill_depressions`, a Priority-Flood (Barnes, Lehman and Mulla
2014) with grid-edge outlets and 8-connectivity, run on float64 copies of float32 DEMs. richdem has no
wheel for Python 3.14, so pysheds is used (it needs the same numpy.in1d alias as terrain-hydrology.py).

Convention differences (the fixture only holds DEMs where they cannot matter):
  outlets   pysheds seeds the priority queue with the grid edge only; GPUTerrainFlow also treats a
            valid cell next to an invalid (NaN) cell as an outlet. The DEMs here have no NaN.
  epsilon   pysheds fills to the spill height without an increment, as fillEpsilon 0 does.
The fill is a fixpoint of F(c) = max(z(c), min over neighbors F(n)) with F = z on the edge, so the
result is unique and must match bit for bit.

Usage (writes fixtures/terrain-fill-depressions.json):
  python3 -I terrain-fill-depressions.py
Needs numpy, pysheds and affine (versions are recorded in the fixture).
"""
import json
import os
from collections import deque
from importlib.metadata import version

import numpy as np

np.in1d = lambda a, b, **kwargs: np.isin(a, b, **kwargs)  # numpy 2.x removed in1d; pysheds 0.5 calls it

from affine import Affine  # noqa: E402
from pysheds.grid import Grid  # noqa: E402
from pysheds.view import Raster, ViewFinder  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, 'fixtures', 'terrain-fill-depressions.json')
GENERATOR = 'modules/experimental/test/oracles/terrain-fill-depressions.py'


def f32(array):
    return np.asarray(array, dtype='float32').astype('float64')


def noise(width, height, seed, levels):
    rng = np.random.default_rng(seed)
    return f32(rng.integers(0, levels, size=(height, width)))


def bowl(width, height, notch):
    rows, columns = np.mgrid[0:height, 0:width]
    radius = np.hypot(columns - (width - 1) / 2, rows - (height - 1) / 2)
    dem = np.minimum(60, np.floor(radius)) + 10
    dem[notch[1], notch[0]] = 7
    return f32(dem)


def spiral(size, corridor_width):
    """Concentric corridor rings with one gap per wall, floors descending inward by BFS distance."""
    period = corridor_width + 1

    def is_wall(column, row):
        depth = min(column, row, size - 1 - column, size - 1 - row)
        if depth % period != period - 1:
            return False
        ring = depth // period
        gap_column = size - 1 - depth - 1 if ring % 2 == 0 else depth + 1
        on_gap_side = row == depth if ring % 2 == 0 else row == size - 1 - depth
        return not (on_gap_side and column == gap_column)

    distance = -np.ones((size, size), dtype='int64')
    queue = deque()
    for row in range(size):
        for column in range(size):
            if min(column, row, size - 1 - column, size - 1 - row) == 0 and not is_wall(column, row):
                distance[row, column] = 0
                queue.append((row, column))
    while queue:
        row, column = queue.popleft()
        for drow, dcolumn in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            r, c = row + drow, column + dcolumn
            if 0 <= r < size and 0 <= c < size and distance[r, c] < 0 and not is_wall(c, r):
                distance[r, c] = distance[row, column] + 1
                queue.append((r, c))
    dem = np.empty((size, size))
    for row in range(size):
        for column in range(size):
            wall = is_wall(column, row) or distance[row, column] < 0
            dem[row, column] = 100000 + (column * 7 + row * 13) % 5 if wall else 50000 - distance[row, column]
    return f32(dem)


def run_case(name, dem):
    height, width = dem.shape
    view = ViewFinder(affine=Affine(1.0, 0, 0, 0, -1.0, 0), shape=dem.shape, nodata=np.nan)
    grid = Grid(viewfinder=view)
    filled = np.asarray(grid.fill_depressions(Raster(dem.copy(), viewfinder=view))).astype('float64')
    return {
        'name': name,
        'width': width,
        'height': height,
        'elevation': [float(value) for value in dem.ravel()],
        'filled': [float(value) for value in filled.ravel()],
        'raisedCells': int(np.count_nonzero(filled != dem))
    }


def main():
    cases = [
        run_case('noise-41x33', noise(41, 33, 3, 20)),
        run_case('plateau-97x65', noise(97, 65, 5, 3)),
        run_case('noise-100x45', noise(100, 45, 7, 2000) / 4),
        run_case('bowl-100x70', bowl(100, 70, (99, 30))),
        run_case('spiral-23-w1', spiral(23, 1)),
        run_case('spiral-37-w3', spiral(37, 3)),
        run_case('spiral-61-w1', spiral(61, 1)),
        run_case('strip-1x37', noise(1, 37, 11, 9)),
        run_case('strip-41x1', noise(41, 1, 12, 9)),
        run_case('strip-31x65', noise(31, 65, 13, 9)),
        run_case('negative-34x34', f32(np.fromfunction(lambda r, c: (c * r) % 11 - 5.5, (34, 34))))
    ]
    fixture = {
        'generator': GENERATOR,
        'reference': 'pysheds Grid.fill_depressions (Priority-Flood, grid-edge outlets)',
        'versions': {
            'python': os.sys.version.split()[0],
            'numpy': version('numpy'),
            'pysheds': version('pysheds'),
            'affine': version('affine')
        },
        'shims': ['numpy.in1d aliased to numpy.isin (removed in numpy 2.x, called by pysheds 0.5)'],
        'conventions': {
            'outlets': 'grid edge only; the DEMs have no NaN cells',
            'tolerance': 'exact (float32 values; the fill only takes maxima of input values)'
        },
        'cases': cases
    }
    with open(FIXTURE, 'w') as handle:
        json.dump(fixture, handle, separators=(',', ':'))
        handle.write('\n')


if __name__ == '__main__':
    main()
