#!/usr/bin/env python3
"""Oracle generator for the luma.gl GPUTerrainFlow D8 directions and accumulation.

Reference implementation: pysheds (https://github.com/mdbartos/pysheds), `Grid.flowdir` (D8,
`routing='d8'`) and `Grid.accumulation`. richdem has no wheel for Python 3.14, so pysheds is used.
pysheds 0.5 still calls `numpy.in1d`, which numpy 2.x removed; the generator aliases it to
`numpy.isin` (same semantics) before calling pysheds. Nothing else is shimmed or reimplemented.

Conventions (pysheds -> luma.gl):
  directions   pysheds' default dirmap (N, NE, E, SE, S, SW, W, NW) = (64, 128, 1, 2, 4, 8, 16, 32)
               is the ESRI code set GPUTerrainFlow uses, with row 0 north (south-advancing rows).
               pysheds writes -1 for flats and -2 for pits (cells with no lower neighbour, including
               grid-border outlets); luma.gl writes 0 for every terminal cell (flat, pit, outlet).
  slope        both divide the drop by the ground length of the move (cardinal dx or dy, diagonal
               sqrt(dx^2 + dy^2)) and take the steepest; the DEMs have no ties.
  border       neither routes flow off the grid.
  accumulation pysheds counts the cell itself (weight 1), as GPUTerrainFlow 'cells' units do.
The DEMs are depression- and flat-free (every interior cell has a strictly lower neighbour by at
least 0.7 elevation units), so no fill or flat convention enters the comparison.

Usage (writes fixtures/terrain-hydrology.json):
  python3 -I terrain-hydrology.py
Needs numpy, pysheds and affine (versions are recorded in the fixture).
"""
import json
import os
from importlib.metadata import version

import numpy as np

np.in1d = lambda a, b, **kwargs: np.isin(a, b, **kwargs)  # numpy 2.x removed in1d; pysheds 0.5 calls it

from affine import Affine  # noqa: E402
from pysheds.grid import Grid  # noqa: E402
from pysheds.view import Raster, ViewFinder  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, 'fixtures', 'terrain-hydrology.json')
GENERATOR = 'modules/experimental/test/oracles/terrain-hydrology.py'


def make_dem(width, height, outlet, seed):
    """Elevation = Chebyshev distance to the outlet cell plus noise in [0, 0.3): no pits or flats."""
    rng = np.random.default_rng(seed)
    rows, columns = np.mgrid[0:height, 0:width]
    distance = np.maximum(np.abs(rows - outlet[0]), np.abs(columns - outlet[1])).astype('float64')
    dem = distance + rng.random((height, width)) * 0.3
    # float32 values, so the GPU, the TS oracle and pysheds see the same inputs.
    return dem.astype('float32').astype('float64')


def run_case(name, width, height, cell_size, outlet, seed):
    dem = make_dem(width, height, outlet, seed)
    view = ViewFinder(
        affine=Affine(cell_size[0], 0, 0, 0, -cell_size[1], 0), shape=dem.shape, nodata=np.nan
    )
    grid = Grid(viewfinder=view)
    directions = np.asarray(grid.flowdir(Raster(dem, viewfinder=view), routing='d8')).astype('int64')
    accumulation = np.asarray(grid.accumulation(grid.flowdir(Raster(dem, viewfinder=view))))
    terminal = directions < 0
    return {
        'name': name,
        'width': width,
        'height': height,
        'cellSize': list(cell_size),
        'elevation': [float(value) for value in dem.ravel()],
        'pyshedsDirections': [int(value) for value in directions.ravel()],
        'directions': [0 if t else int(value) for t, value in zip(terminal.ravel(), directions.ravel())],
        'accumulation': [float(value) for value in accumulation.ravel()],
        'tolerance': {
            'directions': 'exact (no ties: the steepest drop beats the next by more than f32 rounding)',
            'accumulation': 'exact (integer cell counts below 2^24 are exact in float32)'
        }
    }


def main():
    cases = [
        run_case('square-17x13', 17, 13, (10.0, 10.0), (12, 8), 3),
        run_case('anisotropic-15x11', 15, 11, (10.0, 15.0), (0, 4), 7)
    ]
    fixture = {
        'generator': GENERATOR,
        'reference': 'pysheds Grid.flowdir (d8) and Grid.accumulation',
        'versions': {
            'python': os.sys.version.split()[0],
            'numpy': version('numpy'),
            'pysheds': version('pysheds'),
            'affine': version('affine')
        },
        'shims': ['numpy.in1d aliased to numpy.isin (removed in numpy 2.x, called by pysheds 0.5)'],
        'conventions': {
            'directionCodes': 'ESRI (1 E, 2 SE, 4 S, 8 SW, 16 W, 32 NW, 64 N, 128 NE), row 0 north',
            'terminal': 'pysheds -1 (flat) and -2 (pit or border outlet) map to luma.gl 0',
            'accumulationUnits': 'cells, including the cell itself'
        },
        'cases': cases
    }
    with open(FIXTURE, 'w') as handle:
        json.dump(fixture, handle, indent=1)
        handle.write('\n')


if __name__ == '__main__':
    main()
