#!/usr/bin/env python3
"""Oracle generator for the GPU bandwidth rule of luma.gl GPUKernelDensity (E-17).

Reference: scipy.stats.gaussian_kde. For `bw_method='scott'` and `'silverman'` scipy sets
`factor = n_eff^(-1/(d+4))` and `(n_eff (d+2)/4)^(-1/(d+4))`; `kde.covariance = factor^2 * data_covariance`
with the unbiased weighted data covariance (`numpy.cov(aweights=w)`) and `n_eff = 1 / sum(w_n^2)` of
the normalised weights. For d = 2 both factors are `n_eff^(-1/6)`. The isotropic bandwidth of the
radial kernels is `sqrt(trace(kde.covariance) / 2)`. The IQR-based Silverman variant (statsmodels
`bw_silverman`, `0.9 min(sd, IQR / 1.349) n^(-1/5)`) is a different, 1D rule and is not used.

Density oracle: float64 sum of weighted gaussian kernels `w K_h(d)`, `K_h = exp(-d^2 / 2h^2) / (2 pi h^2)`
truncated at 5 h, i.e. GPUKernelDensity with edgeCorrection 'none' and normalization 'intensity'.

Fixtures: the same 240 points at the origin and shifted by (1e6, 4e6). Points are rounded to float32
first (the GPU input precision), so the oracle sees exactly the values the GPU sees.

Usage (writes fixtures/geometry-kernel-density.json):  python3 -I geometry-kernel-density.py
Needs numpy and scipy (versions recorded in the fixture).
"""
import json
import os
from importlib.metadata import version

import numpy as np
from scipy.stats import gaussian_kde

HERE = os.path.dirname(os.path.abspath(__file__))
GENERATOR = 'modules/experimental/test/oracles/geometry-kernel-density.py'
WIDTH = HEIGHT = 20
SIZE = 100.0
CELLS = [[3, 4], [10, 10], [15, 6], [19, 19], [0, 0]]


def make_case(offset, weighted, points32, weights):
    points = points32.astype(np.float64)
    weights_used = weights if weighted else None
    result = {'offset': offset, 'weighted': weighted, 'rules': {}}
    for rule in ('scott', 'silverman'):
        kde = gaussian_kde(points.T, bw_method=rule, weights=weights_used)
        result['rules'][rule] = {
            'factor': kde.factor,
            'effectiveSampleSize': float(kde.neff),
            'isotropic': float(np.sqrt(np.trace(kde.covariance) / 2.0)),
        }
    bandwidth = result['rules']['scott']['isotropic']
    cell = SIZE / WIDTH
    densities = []
    for column, row in CELLS:
        centre = np.array([offset[0] + (column + 0.5) * cell, offset[1] + (row + 0.5) * cell])
        d2 = np.sum((points - centre) ** 2, axis=1)
        t = d2 / bandwidth**2
        k = np.where(t < 25.0, np.exp(-0.5 * t) / (2 * np.pi * bandwidth**2), 0.0)
        w = weights_used if weighted else np.ones(len(points))
        densities.append(float(np.sum(w * k)))
    result['cells'] = CELLS
    result['intensity'] = densities
    return result


def main():
    rng = np.random.default_rng(20261014)
    base = np.vstack([
        rng.normal([30, 35], [9, 5], size=(120, 2)),
        rng.normal([70, 60], [6, 12], size=(120, 2)),
    ])
    base = np.clip(base, 1.0, SIZE - 1.0)
    weights = rng.uniform(0.5, 3.0, size=len(base)).astype(np.float32).astype(np.float64)
    cases = []
    for offset in ([0.0, 0.0], [1.0e6, 4.0e6]):
        # float32 rounding of the shifted coordinates: 0.0625 spacing at 1e6, 0.5 at 4e6.
        shifted = (base + np.array(offset)).astype(np.float32)
        for weighted in (False, True):
            case = make_case(offset, weighted, shifted, weights)
            case['points'] = shifted.astype(np.float64).ravel().tolist()
            cases.append(case)
    fixture = {
        'generator': GENERATOR,
        'scipy': version('scipy'),
        'numpy': version('numpy'),
        'width': WIDTH,
        'height': HEIGHT,
        'size': SIZE,
        'weights': weights.tolist(),
        'bandwidthTolerance': 'relative 1e-4 (see gpu-kernel-density-bandwidth.spec.ts)',
        'cases': cases,
    }
    path = os.path.join(HERE, 'fixtures', 'geometry-kernel-density.json')
    with open(path, 'w') as handle:
        json.dump(fixture, handle)
    print('wrote', path, os.path.getsize(path), 'bytes')


if __name__ == '__main__':
    main()
