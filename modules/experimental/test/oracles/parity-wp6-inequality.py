# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent GPUInequality references; run with the reference venv's python -I.

All 791 Chicago tract per-capita incomes, grouped by communityArea, with and
without population weights. These are inequality among tract means, NOT a
claim to recover individual income inequality. Source bytes and versions are
recorded. No coordinates/CRS enter this calculation. Input and parameter
floats are rounded to f32 before float64 reference evaluation. No RNG is used.

Weights are frequency weights: integer population on real data, quarter-unit
weights on synthetic data. Exact replication (multiply fractional weights by
4) allows independent unweighted inequality.Gini/Theil/TheilD/Atkinson and
Schutz calls; scale invariance preserves every index. Counts remain included
source rows, not replicated population. Exclude masked rows, invalid zones,
non-finite/negative incomes and non-finite/nonpositive weights before calling
libraries. Zeros remain in the population (Theil 0 log 0 = 0).

Theil T is cross-checked against scipy.stats.entropy(income, population).
Theil L uses reversed relative entropy; map infinity to the API's documented
NaN for zero income. Empty/zero-total zones have NaN indices and Lorenz knots;
mean is NaN/0 respectively. Atkinson for strictly positive incomes uses the
library; for zero incomes and epsilon < 1, composite reference from Atkinson
(1970), Journal of Economic Theory 2, 244-263, DOI 10.1016/0022-0531(70)90039-6,
using scipy.stats.pmean. At epsilon >= 1 map zero-income cases to API NaN.
Lorenz knots and fractional Palma cuts interpolate inequality.Schutz's
published Lorenz table (Schutz 1951, doi:10.2307/1908078); Hoover is its
maximum vertical distance. All decomposition fields come from TheilD.
There is no standalone Shannon entropy output in GPUInequality.

Tolerance S: per-element atol=1e-6*max(1,abs(reference)), rtol=1e-4.
Included row counts (including the global count) are exact. NaN/Inf masks
are exact. No tolerance is enlarged for these <=791 source-row reductions.
"""

import hashlib
import json
import math
import os
import tempfile
import platform
from pathlib import Path

# Avoid user-home cache writes from the reference library's plotting imports.
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'parity-wp6-matplotlib'))

import inequality
import numpy as np
import pandas as pd
import scipy
from inequality.atkinson import Atkinson
from inequality.gini import Gini
from inequality.schutz import Schutz
from inequality.theil import Theil, TheilD
from scipy.stats import entropy, pmean

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DATA = ROOT / 'examples/deck/spatial-analysis-showcase/public/data/chicago-tracts'
KNOT_COUNT = 11


def encode(value):
    if isinstance(value, dict):
        return {key: encode(entry) for key, entry in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [encode(entry) for entry in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return 'Infinity' if value > 0 else '-Infinity'
        return float(value)
    return value


def computeZone(values, weights, settings, replicationScale):
    result = {name: math.nan for name in ['gini', 'theilT', 'theilL', 'atkinson', 'hoover', 'palma', 'mean']}
    result['counts'] = len(values)
    result['lorenzKnots'] = np.full(KNOT_COUNT, math.nan)
    if not len(values):
        return result
    result['mean'] = np.average(values, weights=weights)
    if not np.dot(values, weights) > 0:
        return result
    repetitions = (weights * replicationScale).astype(np.int64)
    assert np.array_equal(repetitions, weights * replicationScale)
    population = np.repeat(values, repetitions)
    result['gini'] = Gini(population).g
    result['theilT'] = Theil(population).T
    np.testing.assert_allclose(result['theilT'], entropy(values * weights, weights), atol=1e-12)
    result['theilL'] = entropy(weights, values * weights) if np.all(values > 0) else math.nan
    epsilon = settings['epsilon']
    if np.all(values > 0):
        result['atkinson'] = Atkinson(population, epsilon).A
    elif epsilon < 1:
        result['atkinson'] = 1 - pmean(population, 1 - epsilon) / population.mean()
    lorenz = Schutz(pd.DataFrame({'income': population}), 'income')
    result['hoover'] = lorenz.distance
    populationShares = np.r_[0, lorenz.df_processed['ucpct'].to_numpy()]
    incomeShares = np.r_[0, lorenz.df_processed['ycpct'].to_numpy()]
    result['lorenzKnots'] = np.interp(np.linspace(0, 1, KNOT_COUNT), populationShares, incomeShares)
    bottom = np.interp(settings['palmaBottomShare'], populationShares, incomeShares)
    top = 1 - np.interp(1 - settings['palmaTopShare'], populationShares, incomeShares)
    result['palma'] = top / bottom if bottom > 0 else math.nan
    return result


def makeCase(name, values, zoneIds, zoneCount, weights=None, mask=None, replicationScale=1):
    values = np.asarray(values, dtype=np.float32).astype(np.float64)
    zoneIds = np.asarray(zoneIds, dtype=np.uint32)
    populationWeights = np.ones(len(values)) if weights is None else np.asarray(weights, dtype=np.float32).astype(np.float64)
    mask = np.ones(len(values), dtype=np.uint32) if mask is None else np.asarray(mask, dtype=np.uint32)
    valid = (mask != 0) & (zoneIds < zoneCount) & np.isfinite(values) & (values >= 0) & np.isfinite(populationWeights) & (populationWeights > 0)
    includedValues = values[valid]
    includedWeights = populationWeights[valid]
    summary = np.full(8, math.nan)
    summary[4] = valid.sum()
    summary[6] = includedWeights.sum()
    summary[7] = np.dot(includedValues, includedWeights)
    if valid.any():
        summary[5] = np.average(includedValues, weights=includedWeights)
    if summary[7] > 0:
        repetitions = (includedWeights * replicationScale).astype(np.int64)
        assert np.array_equal(repetitions, includedWeights * replicationScale)
        population = np.repeat(includedValues, repetitions)
        partition = np.repeat(zoneIds[valid], repetitions)
        decomposition = TheilD(population, partition)
        summary[:3] = [decomposition.T, np.asarray(decomposition.bg).item(), np.asarray(decomposition.wg).item()]
        summary[3] = Gini(population).g
        np.testing.assert_allclose(summary[0], entropy(includedValues * includedWeights, includedWeights), atol=1e-12)
        np.testing.assert_allclose(summary[0], summary[1] + summary[2], atol=1e-12)
    runs = []
    for epsilon, top, bottom in [(0, .05, .3), (.5, .1, .4), (1, .1, .4), (2, .2, .5)]:
        settings = dict(zip(['epsilon', 'palmaTopShare', 'palmaBottomShare'], np.array([epsilon, top, bottom], dtype=np.float32).astype(float)))
        zones = [computeZone(values[valid & (zoneIds == zone)], populationWeights[valid & (zoneIds == zone)], settings, replicationScale) for zone in range(zoneCount)]
        expected = {key: [zone[key] for zone in zones] for key in zones[0] if key != 'lorenzKnots'}
        expected['lorenzKnots'] = np.concatenate([zone['lorenzKnots'] for zone in zones])
        expected['globalSummary'] = summary
        runs.append({'settings': settings, 'expected': expected})
    case = {'name': name, 'values': values, 'zoneIds': zoneIds, 'zoneCount': zoneCount, 'mask': mask, 'runs': runs}
    if weights is not None:
        case['weights'] = populationWeights
    return case


sourceFiles = ['perCapitaIncome.bin', 'communityArea.bin', 'population.bin', 'manifest.json']
manifest = json.loads((DATA / 'manifest.json').read_text())
incomes = np.fromfile(DATA / 'perCapitaIncome.bin', dtype='<f4')
communities = np.fromfile(DATA / 'communityArea.bin', dtype='u1').astype(np.uint32)
assert manifest['columns']['communityArea']['dtype'] == 'uint8'
populations = np.fromfile(DATA / 'population.bin', dtype='<f4')
assert len(incomes) == manifest['count'] == 791
zoneCount = int(communities[communities != 0xffffffff].max()) + 2
cases = [
    makeCase('chicago-tracts-all-unweighted', incomes, communities, zoneCount),
    makeCase('chicago-tracts-all-population-weighted', incomes, communities, zoneCount, populations),
    makeCase('zero-ties-empty-mask-invalid-weights',
             [0, 0, 12, 4, 4, 4, 1, 2, 7, 0, 0, 5, -2, math.nan, math.inf, 9, 6, 8, 8, 8, 8, 8],
             [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 2, 2, 2, 99, 0xffffffff, 2, 2, 2, 2, 2], 6,
             [1, .5, 2, .25, 1.25, 2, 1, 2, .5, 1, 2, 3, 1, 1, 1, 1, 1, 1, 0, -1, math.nan, math.inf],
             [1] * 17 + [0, 1, 1, 1, 1], replicationScale=4),
    makeCase('all-zero-income', [0, 0, 0], [0, 0, 1], 3, [1, 2, 4]),
    makeCase('no-included-rows', [-1, math.nan, 10], [0, 0, 0], 2, mask=[1, 1, 0]),
]
fixture = {
    'generator': Path(__file__).name,
    'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'pandas': pd.__version__, 'inequality': inequality.__version__},
    'sources': {str((DATA / name).relative_to(ROOT)): hashlib.sha256((DATA / name).read_bytes()).hexdigest() for name in sourceFiles},
    'conventions': {'input': 'f32 then f64', 'weights': 'frequency; exact replication', 'counts': 'source rows, not weight sums', 'zeroIncome': 'retained; Theil L and Atkinson epsilon>=1 become NaN', 'selection': 'all 791 source rows; API validity filter', 'crs': 'not applicable', 'seed': 'none'},
    'reference': 'inequality.Gini/Theil/TheilD/Atkinson/Schutz; scipy.stats.entropy/pmean; numpy.interp',
    'tolerance': {'profile': 'S', 'absoluteScale': 1e-6, 'relative': 1e-4},
    'lorenzKnotCount': KNOT_COUNT,
    'cases': cases,
}
output = HERE / 'fixtures/parity-wp6-inequality.json'
output.write_text(json.dumps(encode(fixture), separators=(',', ':'), allow_nan=False) + '\n')
assert output.stat().st_size <= 1_500_000
print('wrote', output, output.stat().st_size, 'bytes')
