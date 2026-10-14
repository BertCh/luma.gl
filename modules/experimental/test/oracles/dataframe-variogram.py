# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Reference empirical semivariograms for GPUVariogram from scikit-gstat.

Writes fixtures/dataframe-variogram.json next to this script (consumed by
test/gpu-dataframe/pair-statistics/variogram-reference.node.spec.ts and the
'scikit-gstat references' case of gpu-variogram.spec.ts). Run from anywhere with
    <venv>/bin/python -I modules/experimental/test/oracles/dataframe-variogram.py
in a venv with numpy and scikit-gstat (versions are recorded in the fixture).

Reference: skgstat.Variogram(coordinates, values, estimator=..., bin_func='even', n_lags=L,
maxlag=D, normalize=False, fit_method=None).experimental, for the 'matheron' and 'cressie'
estimators, plus the pair count and mean distance of each lag group (Variogram.lag_groups()).
scikit-gstat bins are half-open [i w, (i + 1) w) with w = D / L and drops d >= D; GPUVariogram uses
floor(d / D * L) and keeps d == D in the last lag. The scenes sit on an integer lattice (distances
are square roots of integers) and the generator asserts that no pair distance lies within
MARGIN of a lag edge or of D, so float32 GPU distances cannot change a pair's lag.

Scenes (float32 values, exact in the fixture):
- 'field': 500 distinct lattice points in [0, 60]^2, z = sin(x / 7) + cos(y / 11) + 0.3 N(0, 1).
- 'outlier': the same points and values with one value set to 2e4 (a sentinel embedded among its
  neighbors; finding B-01 lost up to 83% of the semivariance there before the fixed-point unit fix).

Tolerances (stored per scene, derived from the GPUVariogram fixed-point contract):
- semivariance: u 2^-25 + 2^-20 |gamma|, with u the robust variance unit (at most the population
  variance) and 2^-20 covering the f32 pair terms and the large-term channel (relative 5e-7).
- robust (Cressie-Hawkins): the sqrt terms have unit sqrt(max - min) 2^-24, so the mean of
  |dz|^(1/2) is off by at most half of it, plus about four f32 roundings (|dz|, sqrt, quantize) of
  relative 2^-24 each; the fourth power multiplies the relative error by 4, and the final f32
  power and division add a few more roundings: 4 (sqrt(range) 2^-25 / m) + 2^-19, relative, with
  m the smallest bin mean of |dz|^(1/2).
- mean distance: D 2^-25 + 2^-20 D (unit plus f32 distance rounding).
- pair counts: exact.
"""

import json
import os
import sys

import numpy as np
import skgstat

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'fixtures', 'dataframe-variogram.json')
LAG_COUNT = 8
MARGIN = 1e-3


def choose_maximum_distance(distances, lag_count):
    """First candidate D whose lag edges are all at least MARGIN from every pair distance."""
    for candidate in np.arange(30.0, 40.0, 0.0625):
        edges = candidate / lag_count * np.arange(1, lag_count + 1)
        nearest = np.min(np.abs(distances[:, None] - edges[None, :]))
        if nearest > MARGIN:
            return float(candidate), float(nearest)
    raise RuntimeError('no maximum distance keeps every pair away from the lag edges')


def reference_bins(coordinates, values, maximum_distance):
    result = {}
    for estimator in ('matheron', 'cressie'):
        variogram = skgstat.Variogram(
            coordinates,
            values,
            estimator=estimator,
            bin_func='even',
            n_lags=LAG_COUNT,
            maxlag=maximum_distance,
            normalize=False,
            fit_method=None,
        )
        result[estimator] = [float(value) for value in variogram.experimental]
        groups = variogram.lag_groups()
        distances = variogram.distance
    counts = [int(np.sum(groups == lag)) for lag in range(LAG_COUNT)]
    means = [float(np.mean(distances[groups == lag])) for lag in range(LAG_COUNT)]
    return result, counts, means


def make_scenes():
    rng = np.random.default_rng(20261014)
    lattice = np.array([(x, y) for x in range(61) for y in range(61)], dtype=np.float64)
    points = lattice[rng.choice(len(lattice), size=500, replace=False)]
    noise = rng.normal(0.0, 1.0, len(points))
    field = (np.sin(points[:, 0] / 7) + np.cos(points[:, 1] / 11) + 0.3 * noise).astype(np.float32)
    outlier = field.copy()
    # A central point, so the sentinel joins many pairs.
    center = int(np.argmin(np.sum((points - 30.0) ** 2, axis=1)))
    outlier[center] = np.float32(2e4)
    return points, {'field': field, 'outlier': outlier}, center


def robust_variance(values):
    """GPUVariogram's fixed-point unit: (IQR / 1.349)^2 clamped to [variance 2^-32, variance]."""
    values = values.astype(np.float64)
    variance = float(np.var(values))
    q1, q3 = np.quantile(values, [0.25, 0.75])
    unit = ((q3 - q1) / 1.349) ** 2
    if unit <= 0:
        unit = variance * 2.0**-12
    return min(max(unit, variance * 2.0**-32), variance)


def main():
    points, scenes, center = make_scenes()
    distances = np.sqrt(
        np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=2)[np.triu_indices(len(points), 1)]
    )
    maximum_distance, nearest = choose_maximum_distance(distances, LAG_COUNT)
    fixture = {
        'generator': 'test/oracles/dataframe-variogram.py',
        'versions': {
            'python': sys.version.split()[0],
            'numpy': np.__version__,
            'scikit-gstat': skgstat.__version__,
        },
        'positions': [float(value) for value in points.reshape(-1)],
        'bounds': [0, 0, 60, 60],
        'maximumDistance': maximum_distance,
        'lagCount': LAG_COUNT,
        'nearestEdgeDistance': nearest,
        'outlierRow': center,
        'scenes': {},
    }
    for name, values in scenes.items():
        estimators, counts, means = reference_bins(points, values.astype(np.float64), maximum_distance)
        # Direct numpy cross-check of the Matheron bins (scikit-gstat must agree to 1e-12).
        differences = (values[:, None].astype(np.float64) - values[None, :])[np.triu_indices(len(points), 1)]
        lags = np.floor(distances / maximum_distance * LAG_COUNT).astype(int)
        for lag in range(LAG_COUNT):
            direct = 0.5 * np.mean(differences[lags == lag] ** 2)
            assert abs(direct - estimators['matheron'][lag]) <= 1e-12 * abs(direct), (name, lag)
            assert int(np.sum(lags == lag)) == counts[lag], (name, lag)
        unit = robust_variance(values)
        value_range = float(values.max() - values.min())
        root_means = [
            float(np.mean(np.sqrt(np.abs(differences[lags == lag])))) for lag in range(LAG_COUNT)
        ]
        robust_relative = 4 * (np.sqrt(value_range) * 2.0**-25 / min(root_means)) + 2.0**-19
        fixture['scenes'][name] = {
            'values': [float(value) for value in values],
            'semivariances': estimators['matheron'],
            'robustSemivariances': estimators['cressie'],
            'pairCounts': counts,
            'meanDistances': means,
            'tolerance': {
                'semivarianceAbsolute': unit * 2.0**-25,
                'semivarianceRelative': 2.0**-20,
                'robustSemivarianceRelative': float(robust_relative),
                'meanDistanceAbsolute': maximum_distance * (2.0**-25 + 2.0**-20),
            },
        }
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, 'w') as file:
        json.dump(fixture, file, indent=1)
        file.write('\n')
    print(f'wrote {OUTPUT} (maximumDistance {maximum_distance}, nearest edge {nearest:.4g})')


if __name__ == '__main__':
    main()
