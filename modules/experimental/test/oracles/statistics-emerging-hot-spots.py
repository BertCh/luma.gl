# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Reference multiple-testing adjustments, Mann-Kendall trend statistics and calendar time-step
slice indices for the emerging hot spots specs.

Writes fixtures/statistics-emerging-hot-spots.json next to this script, consumed (through
test/gpu-spatial-analysis/emerging-hot-spots/emerging-hot-spots-reference.ts) by
emerging-hot-spots.node.spec.ts. Run from anywhere with
    <venv>/bin/python -I modules/experimental/test/oracles/statistics-emerging-hot-spots.py
in a venv with numpy, scipy and statsmodels (versions are recorded in the fixture).

- `multipleTesting`: 100 bins at radius 0 and window 0, so the Gi* z of a bin is its population
  standard score. The values are N(0, 1) noise (seed 7, 92 values rounded to 1 decimal) followed by
  the outliers 7.0, 6.0, -6.5 (strong, bins 92-94), 3.4, -3.3, 3.1 (weak, bins 95-97) and two zeros.
  `p` is `2 * scipy.stats.norm.sf(|z|)`; `adjusted` is `statsmodels.stats.multitest.multipletests`
  (bonferroni, holm, fdr_bh, fdr_by) of all 100 p-values, reported for `topBins`; `significantAt05`
  lists the bins whose adjusted p is at most 0.05.
- `mannKendall`: S, tie-corrected variance, continuity-corrected z and two-sided p of a numpy
  implementation; S is cross-checked against `scipy.stats.kendalltau` for the tie-free series.
- `slices`: slice index of `events` (local calendar, no offset) for each step, interval, origin and
  count, from python `datetime` arithmetic, independent of the days-from-civil formula under test.

Tolerances (stored in `tolerances`): the oracle and the shader evaluate the normal tail with the
erfcc fit, whose relative error is below 1.2e-7, so p-values are compared to 1e-6 relative (+ 1e-12
absolute) and the adjusted p-values, which rescale and re-rank those p-values, to 1e-5 relative
(+ 1e-9 absolute). The Mann-Kendall p uses the same fit: 2e-7 relative to max(p, 1e-3), + 1e-9
absolute. S, counts, bins and slice indices are exact; the Mann-Kendall variance and z are float64
arithmetic compared to 10 decimals.
"""

import datetime as dt
import json
import math
import os
import platform

import numpy as np
import scipy
import statsmodels
from scipy.stats import kendalltau, norm
from statsmodels.stats.multitest import multipletests

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'fixtures', 'statistics-emerging-hot-spots.json')


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


# 1. Multiple testing.
generator = np.random.default_rng(7)
background = np.round(generator.normal(0, 1, 92), 1)
values = np.concatenate([background, [7.0, 6.0, -6.5, 3.4, -3.3, 3.1, 0.0, 0.0]])[:100]
z = (values - values.mean()) / values.std()
p = 2 * norm.sf(np.abs(z))
adjusted = {
    'none': p,
    'bonferroni': multipletests(p, method='bonferroni')[1],
    'holm': multipletests(p, method='holm')[1],
    'benjamini-hochberg': multipletests(p, method='fdr_bh')[1],
    'benjamini-yekutieli': multipletests(p, method='fdr_by')[1],
}
top_bins = list(range(92, 98))
multiple_testing = {
    'values': values,
    'topBins': top_bins,
    'z': [z[bin] for bin in top_bins],
    'p': [p[bin] for bin in top_bins],
    'adjusted': {method: [adjustment[bin] for bin in top_bins] for method, adjustment in adjusted.items() if method != 'none'},
    'significantAt05': {method: np.where(adjustment <= 0.05)[0] for method, adjustment in adjusted.items()},
}


# 2. Mann-Kendall (S, tie-corrected variance, continuity-corrected z, two-sided p).
def get_mann_kendall(series):
    series = np.asarray(series, float)
    series = series[np.isfinite(series)]
    count = len(series)
    statistic = 0
    for first in range(count):
        for second in range(first + 1, count):
            statistic += int(np.sign(series[second] - series[first]))
    _, group_sizes = np.unique(series, return_counts=True)
    tie_term = sum(size * (size - 1) * (2 * size + 5) for size in group_sizes if size > 1)
    variance = (count * (count - 1) * (2 * count + 5) - tie_term) / 18
    z_score = 0.0 if statistic == 0 or variance <= 0 else (statistic - np.sign(statistic)) / np.sqrt(variance)
    p_value = 1.0 if variance <= 0 else 2 * norm.sf(abs(z_score))
    return {'count': count, 'statistic': statistic, 'variance': variance, 'z': float(z_score), 'p': float(p_value)}


SERIES = {
    'twelveTied': [1, 3, 2, 3, 5, 4, 6, 6, 8, 7, 9, 10],
    'sixMonotone': [1, 2, 3, 4, 5, 6],
    'tenDown': [9, 8, 8, 6, 5, 5, 3, 2, 1, 0],
}
mann_kendall = {name: {'series': series, **get_mann_kendall(series)} for name, series in SERIES.items()}
tau = kendalltau(np.arange(6), np.array(SERIES['sixMonotone'], float)).statistic
assert round(tau * 15) == mann_kendall['sixMonotone']['statistic']


# 3. Slice indices for the time-step slicing.
def get_month_index(year, month):
    return year * 12 + month - 1


EVENTS = [
    (2022, 10, 31, 23), (2022, 11, 1, 0), (2022, 12, 25, 23), (2022, 12, 26, 0),
    (2022, 12, 31, 23), (2023, 1, 1, 0), (2023, 1, 1, 23), (2023, 1, 2, 0),
    (2023, 2, 19, 12), (2023, 2, 20, 0), (2023, 4, 30, 12), (2023, 5, 1, 0),
    (2023, 12, 29, 12), (2023, 12, 30, 0), (2023, 12, 31, 23), (2024, 1, 1, 0),
    (2024, 2, 28, 23), (2024, 2, 29, 5), (2024, 3, 1, 5),
]  # fmt: skip


def get_slice(step, interval, origin, count, event):
    year, month, day, hour = event
    if step == 'month':
        difference = get_month_index(year, month) - get_month_index(origin[0], origin[1])
        divisor = interval
    elif step == 'quarter':
        difference = (year * 4 + (month - 1) // 3) - (origin[0] * 4 + (origin[1] - 1) // 3)
        divisor = interval
    elif step == 'year':
        difference = year - origin[0]
        divisor = interval
    elif step == 'week':
        difference = (dt.date(year, month, day) - dt.date(*origin)).days
        divisor = 7 * interval
    elif step == 'day':
        difference = (dt.date(year, month, day) - dt.date(*origin)).days
        divisor = interval
    elif step == 'hour':
        difference = int((dt.datetime(year, month, day, hour) - dt.datetime(*origin)).total_seconds() / 3600)
        divisor = interval
    else:
        raise ValueError(step)
    if difference < 0:
        return -1
    index = difference // divisor
    return int(index) if index < count else -1


CASES = [
    ('month', 1, (2022, 11), 6),
    ('month', 2, (2022, 11), 3),
    ('week', 1, (2022, 12, 26), 8),
    ('week', 2, (2022, 12, 26), 4),
    ('day', 1, (2023, 12, 30), 100),
    ('quarter', 1, (2022, 10), 4),
    ('year', 1, (2022,), 3),
    ('hour', 6, (2024, 2, 29, 0), 20),
]
slices = [
    {
        'step': step,
        'interval': interval,
        'origin': list(origin),
        'count': count,
        'expected': [get_slice(step, interval, origin, count, event) for event in EVENTS],
    }
    for step, interval, origin, count in CASES
]

fixture = {
    'generator': 'modules/experimental/test/oracles/statistics-emerging-hot-spots.py',
    'versions': {
        'python': platform.python_version(),
        'numpy': np.__version__,
        'scipy': scipy.__version__,
        'statsmodels': statsmodels.__version__,
    },
    'tolerances': {
        'pValue': {'absolute': 1e-12, 'relative': 1e-6},
        'adjustedPValue': {'absolute': 1e-9, 'relative': 1e-5},
        'mannKendallPValue': {'absolute': 1e-9, 'relative': 2e-7, 'floor': 1e-3},
        'mannKendallDecimals': 10,
    },
    'reference': {
        'multipleTesting': multiple_testing,
        'mannKendall': mann_kendall,
        'events': EVENTS,
        'slices': slices,
    },
}
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, 'w') as handle:
    json.dump(encode(fixture), handle, separators=(',', ':'))
    handle.write('\n')
print('wrote', OUTPUT)
