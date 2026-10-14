#!/usr/bin/env python3
"""Oracle generator for GPUKernelDensity `method: 'binned'` and `'auto'` at large bandwidths.

Reference: the float64 gaussian kernel sum, which is what the exact estimator computes,
`lambda(u) = sum_i w_i exp(-|u - x_i|^2 / 2 h^2) / (2 pi h^2)`, truncated at 5 h like the exact
gather. The kernel normalisation is cross-checked against `scipy.stats.multivariate_normal`
(isotropic covariance h^2 I, untruncated) for every case. `scipy.stats.gaussian_kde` cannot be used
directly: its covariance is the data covariance scaled by a factor, never an isotropic h^2 I, and the
bandwidth here is in ground units. Edge-corrected fields divide by the window mass
`e(u) = [Phi((x1 - ux) / h) - Phi((x0 - ux) / h)] [Phi((y1 - uy) / h) - Phi((y0 - uy) / h)]`
with `scipy.stats.norm.cdf`.

Fixture: 128 x 96 cells of 0.5 units on a window starting at the origin (rows advance north), 400
points: two tight clusters, a broad cluster, points exactly on the edges and corners of the window,
zero weights, and rows that every kernel must exclude (NaN position, NaN weight, negative weight).
Points are rounded to float32 first, the GPU input precision. Bandwidths are 2 and 6 cells (the exact
path under 'auto') and 40, 80 and 200 cells (the binned path), for 'none' and 'uniform' edge
correction. The tolerance is not stored: the GPU spec derives it per case from
`getGPUKernelDensityBinnedErrorBound(cell / h)` (kernel-density-math.ts).

Usage (writes fixtures/geometry-kernel-density-binned.json):  python3 -I geometry-kernel-density-binned.py
Needs numpy and scipy (versions recorded in the fixture).
"""
import json
import os
from importlib.metadata import version

import numpy as np
from scipy.stats import multivariate_normal, norm

HERE = os.path.dirname(os.path.abspath(__file__))
GENERATOR = 'modules/experimental/test/oracles/geometry-kernel-density-binned.py'
WIDTH, HEIGHT, CELL = 128, 96, 0.5
WINDOW = [0.0, 0.0, WIDTH * CELL, HEIGHT * CELL]
BANDWIDTH_CELLS = [2, 6, 40, 80, 200]


def make_points():
    rng = np.random.default_rng(20261014)
    extent = np.array([WINDOW[2], WINDOW[3]])
    clusters = [
        rng.normal([14.0, 12.0], [1.5, 1.0], size=(120, 2)),
        rng.normal([50.0, 36.0], [0.8, 2.5], size=(100, 2)),
        rng.normal([32.0, 24.0], [10.0, 8.0], size=(120, 2)),
    ]
    points = np.clip(np.vstack(clusters), 0.0, extent)
    # Points exactly on the window edges and corners.
    edge = np.array([
        [0.0, 0.0], [extent[0], 0.0], [0.0, extent[1]], [extent[0], extent[1]],
        [0.0, 24.0], [extent[0], 24.0], [32.0, 0.0], [32.0, extent[1]],
        [0.1, 0.1], [extent[0] - 0.1, extent[1] - 0.1],
    ])
    points = np.vstack([points, edge, edge + [0.0, 0.0]])
    weights = rng.uniform(0.5, 3.0, size=len(points))
    weights[rng.choice(len(points), 20, replace=False)] = 0.0
    points32 = points.astype(np.float32).astype(np.float64)
    weights32 = weights.astype(np.float32).astype(np.float64)
    return points32, weights32


def pick_cells():
    cells = [[0, 0], [WIDTH - 1, 0], [0, HEIGHT - 1], [WIDTH - 1, HEIGHT - 1],
             [WIDTH // 2, HEIGHT // 2], [0, HEIGHT // 2], [WIDTH - 1, HEIGHT // 2],
             [WIDTH // 2, 0], [WIDTH // 2, HEIGHT - 1], [28, 24], [100, 72], [64, 20]]
    rng = np.random.default_rng(7)
    for _ in range(36):
        cells.append([int(rng.integers(0, WIDTH)), int(rng.integers(0, HEIGHT))])
    return cells


def density(points, weights, centre, h):
    d2 = np.sum((points - centre) ** 2, axis=1)
    k = np.where(d2 < (5 * h) ** 2, np.exp(-0.5 * d2 / h**2) / (2 * np.pi * h**2), 0.0)
    return float(np.sum(weights * k))


def window_mass(centre, h):
    x0, y0, x1, y1 = WINDOW
    mass_x = norm.cdf((x1 - centre[0]) / h) - norm.cdf((x0 - centre[0]) / h)
    mass_y = norm.cdf((y1 - centre[1]) / h) - norm.cdf((y0 - centre[1]) / h)
    return float(mass_x * mass_y)


def main():
    points, weights = make_points()
    cells = pick_cells()
    cases = []
    for cells_per_bandwidth in BANDWIDTH_CELLS:
        h = cells_per_bandwidth * CELL
        for edge_correction in ('none', 'uniform'):
            expected = []
            for column, row in cells:
                centre = np.array([(column + 0.5) * CELL, (row + 0.5) * CELL])
                value = density(points, weights, centre, h)
                # Independent normalisation check of the untruncated kernel (first three cells).
                if len(expected) < 3:
                    reference = sum(
                        w * multivariate_normal(mean=p, cov=h * h * np.eye(2)).pdf(centre)
                        for p, w in zip(points, weights)
                    )
                    assert abs(reference - value) <= 1e-5 * max(abs(reference), 1e-30) + 1e-12, (
                        reference, value)
                if edge_correction == 'uniform':
                    value /= window_mass(centre, h)
                expected.append(value)
            cases.append({
                'bandwidthCells': cells_per_bandwidth,
                'bandwidth': h,
                'edgeCorrection': edge_correction,
                'expected': expected,
            })
    # Rows every kernel excludes, appended after the oracle points (the oracle ignores them).
    excluded_positions = [float('nan'), 10.0, 20.0, 20.0, 30.0, 30.0]
    excluded_weights = [1.0, float('nan'), -1.0]
    def finite(values):
        return [None if isinstance(v, float) and v != v else v for v in values]
    fixture = {
        'generator': GENERATOR,
        'versions': {'scipy': version('scipy'), 'numpy': version('numpy')},
        'tolerances': 'binned: getGPUKernelDensityBinnedErrorBound(cell / h) * W / (2 pi h^2); '
                      'exact (h <= 6 cells): 2e-4 * max|expected|',
        'width': WIDTH,
        'height': HEIGHT,
        'cellSize': CELL,
        'window': WINDOW,
        'points': points.ravel().tolist(),
        'weights': weights.tolist(),
        'weightSum': float(weights.sum()),
        'excludedPositions': finite(excluded_positions),
        'excludedWeights': finite(excluded_weights),
        'cells': cells,
        'cases': cases,
    }
    path = os.path.join(HERE, 'fixtures', 'geometry-kernel-density-binned.json')
    with open(path, 'w') as handle:
        json.dump(fixture, handle)
    print('wrote', path, os.path.getsize(path), 'bytes')


if __name__ == '__main__':
    main()
