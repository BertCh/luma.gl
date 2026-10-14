# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Reference permutation p-values for each alternative from esda, plus the esda randomization moments
of Moran's I used to check the shuffle.

Writes fixtures/statistics-permutation-alternative.json next to this script, consumed (through
test/gpu-spatial-analysis/permutation-inference/permutation-alternative-reference.ts) by
permutation-alternative.node.spec.ts, permutation-shuffle.node.spec.ts and
gpu-permutation-alternative.spec.ts. Run from anywhere with
    <venv>/bin/python -I modules/experimental/test/oracles/statistics-permutation-alternative.py
in a venv with numpy, scipy, esda and libpysal (versions are recorded in the fixture).

`alternatives`: `esda.significance.calculate_significance(observed, simulated, alternative=...)` for
20 fixed simulated values (`reference`, `partialTie`, or twenty copies of 0.5) and one observed value
per case. `greater` / `lesser` are #{sim >= observed} / #{sim <= observed}; `directed`, `greaterP`,
`lesserP` are the esda p-values (R = 20, so every value is k / 21); `foldedP` is the `folded`
alternative. `minRuleDiffers` marks the cases where esda `directed` differs from
(min(greater, lesser) + 1) / (R + 1), the rule the contributor uses; this happens exactly when
simulations tie the observed value, because esda counts the ties only on the greater side.

`moran`: `Moran(x, lat2W(10, 10, rook=True), transformation='O', permutations=0)` for the skewed
deterministic x_i = (((i * 37) % 100) + 1) ** 2 / 100: `expectedI` is EI and `varianceRandomization`
is VI_rand. `chiSquareCritical` is `scipy.stats.chi2.isf(1e-3, df)` for df 9 and 99.

Tolerances (stored in `tolerances`): every p-value is a ratio of small integers computed in float64
on both sides, so the specs compare to 12 decimal places (`pValueDecimals`, 5e-13 absolute); the
Moran moments are float64 sums over 100 terms, compared to 14 decimals for the mean
(`expectedIDecimals`) and 12 decimals for the variance (`varianceDecimals`).
"""

import json
import math
import os
import platform

import esda
import esda.significance as significance
import libpysal
import numpy as np
import scipy
from esda.moran import Moran
from libpysal.weights import lat2W
from scipy.stats import chi2

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'fixtures', 'statistics-permutation-alternative.json')

REFERENCE = [0.5, -0.2, 0.1, 0.9, -0.4, 0.3, 0.0, 0.7, -0.6, 0.2, 0.4, -0.1, 0.15, 0.35, -0.3, 0.8, -0.05, 0.25, 0.6, -0.7]
ALL_TIED = [0.5] * 20
PARTIAL_TIE = [0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, 0.0, 0.0, 0.8, 0.9]
CASES = {
    'distinct_high': (REFERENCE, 0.9),
    'distinct_mid': (REFERENCE, 0.05),
    'distinct_extreme': (REFERENCE, 1.5),
    'partial_tie_none': (PARTIAL_TIE, 0.45),
    'distinct_low': (REFERENCE, -0.4),
    'partial_tie_zero': (PARTIAL_TIE, 0.0),
    'all_tied': (ALL_TIED, 0.5),
}


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


def p_value(observed, simulated, alternative):
    return float(significance.calculate_significance(observed, np.array(simulated), alternative=alternative))


alternatives = {}
for name, (simulated, observed) in CASES.items():
    greater = int(sum(value >= observed for value in simulated))
    lesser = int(sum(value <= observed for value in simulated))
    directed = p_value(observed, simulated, 'directed')
    contributor_rule = (min(greater, lesser) + 1) / (len(simulated) + 1)
    alternatives[name] = {
        'observed': observed,
        'greater': greater,
        'lesser': lesser,
        'directed': directed,
        'greaterP': p_value(observed, simulated, 'greater'),
        'lesserP': p_value(observed, simulated, 'lesser'),
        'foldedP': p_value(observed, simulated, 'folded'),
        'minRuleDiffers': abs(directed - contributor_rule) > 1e-12,
    }

side = 10
x = np.array([(((i * 37) % 100) + 1) ** 2 / 100 for i in range(side * side)])
moran = Moran(x, lat2W(side, side, rook=True), transformation='O', permutations=0)

fixture = {
    'generator': 'modules/experimental/test/oracles/statistics-permutation-alternative.py',
    'versions': {
        'python': platform.python_version(),
        'numpy': np.__version__,
        'scipy': scipy.__version__,
        'esda': esda.__version__,
        'libpysal': libpysal.__version__,
    },
    'tolerances': {'pValueDecimals': 12, 'expectedIDecimals': 14, 'varianceDecimals': 12},
    'reference': {
        'simulatedCount': 20,
        'simulated': REFERENCE,
        'alternatives': alternatives,
        'moran': {
            'expectedI': float(moran.EI),
            'varianceRandomization': float(moran.VI_rand),
            'chiSquareCritical': {'df9': float(chi2.isf(1e-3, 9)), 'df99': float(chi2.isf(1e-3, 99))},
        },
    },
}

with open(OUTPUT, 'w') as handle:
    json.dump(encode(fixture), handle, separators=(',', ':'))
    handle.write('\n')
print('wrote', OUTPUT)
