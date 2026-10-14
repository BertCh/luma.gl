# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Reference local Getis-Ord, local Moran and local Geary values from esda, plus statsmodels
multiple-testing adjustments of the Gi* p-values.

Writes fixtures/statistics-local-autocorrelation.json next to this script, consumed (through
test/gpu-spatial-analysis/spatial-autocorrelation/spatial-autocorrelation-reference.ts) by
gpu-hot-spot-analysis.spec.ts, gpu-local-moran.spec.ts, additional-spatial-statistics.spec.ts and
spatial-autocorrelation.node.spec.ts. Run from anywhere with
    <venv>/bin/python -I modules/experimental/test/oracles/statistics-local-autocorrelation.py
in a venv with numpy, scipy, esda, libpysal and statsmodels (versions are recorded in the fixture).

Scene: 60 rows on a line, values = a bump at row 15, a dip at row 42 and N(0, 0.6) noise (seed 7),
rounded to 3 decimals and read as float32. Binary neighbors i +- 1, i +- 2 among rows 0..58; row 59
is an island. Configurations:
- `keep`: all 60 rows (esda convention: islands stay in n);
- `exclude`: rows 0..58 only (W rebuilt on the subset);
- `masked`: rows 0..58 without `maskedRows` (W rebuilt on the subset, re-standardized).
Arrays are indexed by the full row id; null (NaN) where the row is outside the configuration.

References:
- `giStar`, `gi`: `esda.G_Local(..., transform='B', star=True/False).Zs`;
- `localIBinary`, `localIRow`: `esda.Moran_Local(..., transformation='o'/'r').Is`;
- `lagRow`, `lagBinary`: `libpysal.weights.spatial_lag.lag_spatial` of the centered values;
- `gearyRow`: `esda.Geary_Local(..., drop_islands=False).localG` (row-standardized W);
- `multipleTesting`: `statsmodels.stats.multitest.multipletests` over the 59 non-island Gi* p-values,
  binned at the levels 0.10 / 0.05 / 0.01 with the sign of z.

Tolerances (stored in `tolerances`): the GPU reads the same float32 values and computes in float32
with a second-pass mean, so local statistics agree to 1e-4 absolute + 1e-3 relative, z-scores to
2e-3 + 1e-3 |z| (z is a ratio of two float32 sums of at most 60 terms), p-values to 1e-5 + 2e-3
relative, and adjusted p-values to 1e-6 + 2e-3 relative. Bins are exact: `minimumMargin` (the
smallest relative gap between an adjusted p-value and a level) and `zMargin` (the smallest gap
between |z| and a critical value) are far above those tolerances.
"""

import json
import math
import os
import platform
import warnings

import esda
import libpysal
import numpy as np
import scipy
import statsmodels
from libpysal.weights import W
from libpysal.weights.spatial_lag import lag_spatial
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'fixtures', 'statistics-local-autocorrelation.json')
ROW_COUNT = 60
ISLAND = 59
MASKED_OUT = [3, 11, 30, 31, ISLAND]
LEVELS = [0.10, 0.05, 0.01]
CRITICAL_Z = [1.6448536269514722, 1.959963984540054, 2.5758293035489004]


def encode(value):
    """JSON-safe copy: NaN -> null, infinities -> strings (decoded by test/oracles/oracle-fixture.ts)."""
    if isinstance(value, dict):
        return {key: encode(entry) for key, entry in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [encode(entry) for entry in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if math.isnan(value):
            return None
        if math.isinf(value):
            return 'Infinity' if value > 0 else '-Infinity'
        return value
    return value


generator = np.random.default_rng(7)
index = np.arange(ROW_COUNT)
rounded = np.round(
    10
    + 4 * np.exp(-(((index - 15) / 5) ** 2))
    - 3.5 * np.exp(-(((index - 42) / 4) ** 2))
    + generator.normal(0, 0.6, ROW_COUNT),
    3,
)
# Exactly what the GPU reads.
values = rounded.astype(np.float32).astype(np.float64)
neighbors = {
    row: [other for other in (row - 2, row - 1, row + 1, row + 2) if 0 <= other <= 58]
    for row in range(ISLAND)
}
neighbors[ISLAND] = []


def make_weights(keep):
    position = {row: slot for slot, row in enumerate(keep)}
    subset = {position[row]: [position[other] for other in neighbors[row] if other in position] for row in keep}
    return W(subset, {row: [1.0] * len(entries) for row, entries in subset.items()}, silence_warnings=True)


def scatter(keep, subset_values):
    full = np.full(ROW_COUNT, np.nan)
    full[list(keep)] = subset_values
    return full


def run(keep):
    # esda mutates the transform of the W it is given, so every statistic gets a fresh W.
    keep = list(keep)
    subset = values[keep]
    result = {
        'giStar': scatter(keep, esda.G_Local(subset, make_weights(keep), transform='B', permutations=0, star=True).Zs),
        'gi': scatter(keep, esda.G_Local(subset, make_weights(keep), transform='B', permutations=0, star=False).Zs),
        'localIBinary': scatter(keep, esda.Moran_Local(subset, make_weights(keep), transformation='o', permutations=0).Is),
        'localIRow': scatter(keep, esda.Moran_Local(subset, make_weights(keep), transformation='r', permutations=0).Is),
    }
    centered = subset - subset.mean()
    row_weights = make_weights(keep)
    row_weights.transform = 'r'
    binary_weights = make_weights(keep)
    binary_weights.transform = 'o'
    result['lagRow'] = scatter(keep, lag_spatial(row_weights, centered))
    result['lagBinary'] = scatter(keep, lag_spatial(binary_weights, centered))
    geary = esda.Geary_Local(connectivity=make_weights(keep), permutations=0, drop_islands=False).fit(subset)
    result['gearyRow'] = scatter(keep, geary.localG)
    result['count'] = len(keep)
    result['mean'] = float(subset.mean())
    result['variance'] = float(subset.var())
    return result


all_rows = list(range(ROW_COUNT))
non_island = list(range(ISLAND))
masked = [row for row in non_island if row not in MASKED_OUT]
reference = {
    'values': rounded,
    'island': ISLAND,
    'keep': run(all_rows),
    'exclude': run(non_island),
    'masked': run(masked),
    'maskedRows': sorted(MASKED_OUT),
}

# esda scores an island from its own value (Gi*); the GPU never does, so the island is untested.
for key in ('giStar', 'gi'):
    reference['keep'][key][ISLAND] = np.nan

# Multiple testing over the 59 tested Gi* rows (island untested), esda keep-islands z.
z = np.array(reference['keep']['giStar'])
tested = np.isfinite(z)
p_values = np.full(ROW_COUNT, np.nan)
p_values[tested] = 2 * norm.sf(np.abs(z[tested]))
multiple_testing = {'none': {'adjusted': p_values.copy()}}
for name, method in [('bonferroni', 'bonferroni'), ('holm', 'holm'), ('benjamini-hochberg', 'fdr_bh'), ('benjamini-yekutieli', 'fdr_by')]:
    adjusted = np.full(ROW_COUNT, np.nan)
    adjusted[tested] = multipletests(p_values[tested], method=method)[1]
    multiple_testing[name] = {'adjusted': adjusted}
minimum_margin = math.inf
for entry in multiple_testing.values():
    adjusted = entry['adjusted']
    bins = np.zeros(ROW_COUNT, dtype=int)
    for row in range(ROW_COUNT):
        if np.isfinite(adjusted[row]):
            level = sum(adjusted[row] <= alpha for alpha in LEVELS)
            bins[row] = int(level) * (1 if z[row] > 0 else -1)
            for alpha in LEVELS:
                if adjusted[row] > 0:
                    minimum_margin = min(minimum_margin, abs(adjusted[row] - alpha) / alpha)
    entry['bins'] = bins
multiple_testing['pValues'] = p_values
reference['multipleTesting'] = multiple_testing
reference['minimumMargin'] = float(minimum_margin)
# Uncorrected bins from critical z values.
reference['noneBinsFromZ'] = [
    0 if not np.isfinite(score) else int(sum(abs(score) >= critical for critical in CRITICAL_Z)) * (1 if score > 0 else -1)
    for score in z
]
reference['zMargin'] = float(min(abs(abs(score) - critical) for score in z[tested] for critical in CRITICAL_Z))

fixture = {
    'generator': 'test/oracles/statistics-local-autocorrelation.py',
    'versions': {
        'python': platform.python_version(),
        'numpy': np.__version__,
        'scipy': scipy.__version__,
        'esda': esda.__version__,
        'libpysal': libpysal.__version__,
        'statsmodels': statsmodels.__version__,
    },
    'tolerances': {
        'local': {'absolute': 1e-4, 'relative': 1e-3},
        'zScore': {'absolute': 2e-3, 'relative': 1e-3},
        'pValue': {'absolute': 1e-5, 'relative': 2e-3},
        'adjustedPValue': {'absolute': 1e-6, 'relative': 2e-3},
        'derivation': 'float32 inputs identical to the oracle; float32 sums of at most 60 terms with a '
        'second-pass mean; bins exact because minimumMargin and zMargin exceed the p and z bounds',
    },
    'reference': reference,
}
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, 'w') as file:
    json.dump(encode(fixture), file, separators=(',', ':'))
    file.write('\n')
print('wrote', OUTPUT, 'minimum multiple-testing margin', reference['minimumMargin'], 'z margin', reference['zMargin'])
