# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Reference multiple-testing adjustments from statsmodels.

Writes fixtures/statistics-multiple-testing.json next to this script, consumed (through
test/gpu-spatial-analysis/multiple-testing/multiple-testing-fixtures.ts) by
gpu-multiple-testing-adjustment.spec.ts and gpu-multiple-testing-adjustment.node.spec.ts. Run from
anywhere with
    <venv>/bin/python -I modules/experimental/test/oracles/statistics-multiple-testing.py
in a venv with numpy and statsmodels (versions are recorded in the fixture).

Scenes (p-value families; NaN = untested row, which stays NaN and is excluded from the family):
- `distinct`, `ties`: 12 sorted p-values without and with ties (and exact 0 and 1);
- `missing`: 13 entries of which 4 are NaN;
- `minimum-family`, `minimum-family-missing`: the same p-values padded with p = 1 up to a family of
  30 / 20 tests, as the GPU does for `minimumFamilySize` (the padding is dropped from the output);
- `discrete`: 26 p-values from the discrete counts (c + 1) / 100 with two NaN.

References: `statsmodels.stats.multitest.multipletests(p, alpha=0.05, method=...)[1]` for
bonferroni, holm, fdr_bh (benjamini-hochberg) and fdr_by (benjamini-yekutieli).

Tolerances (stored in `tolerances`, applied by the specs): the GPU computes in float32, whose unit
roundoff is 2^-24 ~ 6e-8, and the adjustments are a product of a p-value with a ratio of integers
followed by running minima/maxima, so a relative 1e-6 (about 17 roundoffs) plus an absolute 1e-7
(for adjusted values at or near 0) bounds the error. NaN must match NaN exactly.
"""

import json
import math
import os
import platform

import numpy as np
import statsmodels
from statsmodels.stats.multitest import multipletests

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'fixtures', 'statistics-multiple-testing.json')
NAN = float('nan')
METHODS = {
    'bonferroni': 'bonferroni',
    'holm': 'holm',
    'benjamini-hochberg': 'fdr_bh',
    'benjamini-yekutieli': 'fdr_by',
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


discrete_counts = [0, 0, 1, 1, 1, 2, 3, 3, 5, 8, 10, 13, 20, 25, 30, 40, 50, 60, 70, 80, 99, 99, 99, 12, 3, 3]
discrete = [(count + 1) / 100 for count in discrete_counts]
discrete[4] = NAN
discrete[17] = NAN

DISTINCT = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216, 0.222, 0.251]
MISSING = [NAN, 0.003, 0.2, NAN, 0.0, 0.03, 0.03, 0.9, NAN, 1.0, 0.012, 0.049, NAN]
scenes = [
    ('distinct', DISTINCT, 0),
    ('ties', [0.0, 0.01, 0.01, 0.02, 0.04, 0.04, 0.04, 0.2, 0.5, 0.5, 1.0, 1.0], 0),
    ('missing', MISSING, 0),
    ('minimum-family', DISTINCT, 30),
    ('minimum-family-missing', MISSING, 20),
    ('discrete', discrete, 0),
]


def adjust(p_values, method, minimum_family_size):
    p_values = np.array(p_values, dtype=float)
    finite = np.isfinite(p_values)
    adjusted = np.full(p_values.shape, np.nan)
    tested = p_values[finite]
    padding = max(minimum_family_size - tested.size, 0)
    padded = np.concatenate([tested, np.ones(padding)])
    adjusted[finite] = multipletests(padded, alpha=0.05, method=METHODS[method])[1][: tested.size]
    return adjusted


fixtures = [
    {
        'name': name,
        'pValues': p_values,
        'minimumFamilySize': minimum_family_size,
        'expected': {method: adjust(p_values, method, minimum_family_size) for method in METHODS},
    }
    for name, p_values, minimum_family_size in scenes
]

fixture = {
    'generator': 'test/oracles/statistics-multiple-testing.py',
    'versions': {
        'python': platform.python_version(),
        'numpy': np.__version__,
        'statsmodels': statsmodels.__version__,
    },
    'tolerances': {
        'adjustedPValue': {'absolute': 1e-7, 'relative': 1e-6},
        'derivation': 'float32 unit roundoff 2^-24 (~6e-8); products and running extrema of at most 30 '
        'terms stay within 17 roundoffs relative, absolute floor for adjusted values near 0',
    },
    'fixtures': fixtures,
}
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, 'w') as file:
    json.dump(encode(fixture), file, separators=(',', ':'))
    file.write('\n')
print('wrote', OUTPUT)
