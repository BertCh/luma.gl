# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Edge-policy tables (quality roadmap rule 22) of the core spatial statistics: global Moran's I,
Geary's C, Getis-Ord General G and join counts (GPUGlobalSpatialStatistics), local Getis-Ord Gi*
(GPUHotSpotAnalysis) and local Moran's I (GPULocalMoran).

Writes fixtures/statistics-edge-policy.json, consumed by
test/gpu-spatial-analysis/statistics-edge-policy/statistics-edge-policy.spec.ts (GPU) and
statistics-edge-policy.node.spec.ts (CPU oracles and parameter packers). Run with
    <venv>/bin/python -I modules/experimental/test/oracles/statistics-edge-policy.py
in a venv with numpy, scipy, esda and libpysal (versions are recorded in the fixture).

The documented policy (session 32 statistics decision 4 and the contributor TSDoc): an analysis row
has a nonzero mask and a finite value. NaN, +Infinity and -Infinity are missing: the row leaves n, the
moments and every neighborhood, and its local outputs are NaN. -0 equals +0 (white for join counts).
With no analysis row every statistic (and E[I], E[C]) is NaN, n is 0 and the join counts are 0 (a
count of nothing, not an undefined value). The global variances divide by n - 3, so they
(and z, p) are NaN for n < 4; a statistic whose denominator is zero (constant values) is NaN, never 0.
Join counts treat a nonzero finite value as black (counted on the adjacency like esda.Join_Counts,
which raises on a one-colour input).

Expected values are esda on W.subset over the analysis rows (permutations=0, islands stay in n, as
`islands: 'keep'`), with the policy applied where esda divides by zero (esda then warns and returns
inf or nan; the policy value is NaN). Values are written as JSON with null = NaN and the strings
"Infinity", "-Infinity" and "-0" (decoded by test/oracles/oracle-fixture.ts).

Scene: 12 rows on a ring, binary symmetric neighbors i +- 1 and i +- 2 (mod 12), so no exclusion
below leaves an island. Small integer values (with zeros, so join counts have white rows).

Tolerances (stored): statistics 1e-5 absolute + 1e-4 relative, z 1e-4 + 1e-3 |z|, p 1e-5 + 1e-3
relative (float32 sums of at most 60 terms of small integers); counts, n and join counts exact;
NaN must be NaN.
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
from libpysal.weights import W
from scipy.stats import norm

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'fixtures', 'statistics-edge-policy.json')
ROW_COUNT = 12
BASE_VALUES = [0.0, 3.0, 2.0, 0.0, 5.0, 4.0, 0.0, 1.0, 6.0, 2.0, 0.0, 3.0]
NEIGHBORS = {row: sorted({(row + offset) % ROW_COUNT for offset in (-2, -1, 1, 2)}) for row in range(ROW_COUNT)}
NAN = float('nan')


def encode(value):
    """JSON-safe copy: NaN -> null, infinities and -0 -> strings."""
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
        if value == 0 and math.copysign(1, value) < 0:
            return '-0'
        return value
    return value


def finite_or_nan(value):
    value = float(value)
    return value if math.isfinite(value) else NAN


def make_weights(keep):
    position = {row: slot for slot, row in enumerate(keep)}
    subset = {position[row]: [position[other] for other in NEIGHBORS[row] if other in position] for row in keep}
    return W(subset, {row: [1.0] * len(entries) for row, entries in subset.items()}, silence_warnings=True)


def get_analysis_rows(values, mask):
    return [row for row in range(ROW_COUNT) if (mask is None or mask[row]) and math.isfinite(values[row])]


def get_global_expected(values, mask):
    keep = get_analysis_rows(values, mask)
    n = len(keep)
    nan_block = {
        'statistic': NAN, 'expected': NAN, 'varianceRandomization': NAN, 'zRandomization': NAN, 'pRandomization': NAN,
    }
    if n == 0:
        return {'count': 0, 'blackCount': 0, 'moran': dict(nan_block), 'geary': dict(nan_block), 'getisOrdG': dict(nan_block),
                'joinCount': {'blackBlack': 0.0, 'blackWhite': 0.0, 'whiteWhite': 0.0, 'joins': 0.0}}
    subset = np.array([values[row] for row in keep], dtype=np.float64)
    constant = bool(np.all(subset == subset[0]))
    result = {'count': n}

    def block(statistic, expected, variance):
        statistic = finite_or_nan(statistic)
        expected = finite_or_nan(expected)
        variance = finite_or_nan(variance)
        if n < 4 or constant:
            # Variances divide by n - 3; constant values have a zero variance (policy: NaN).
            variance = NAN
        z = (statistic - expected) / math.sqrt(variance) if variance > 0 else NAN
        return {
            'statistic': statistic,
            'expected': expected,
            'varianceRandomization': variance,
            'zRandomization': z,
            'pRandomization': float(2 * norm.sf(abs(z))) if math.isfinite(z) else NAN,
        }

    moran = esda.Moran(subset, make_weights(keep), transformation='O', permutations=0)
    geary = esda.Geary(subset, make_weights(keep), transformation='O', permutations=0)
    getis_ord = esda.G(subset, make_weights(keep), permutations=0)
    result['moran'] = block(moran.I if not constant else NAN, moran.EI, moran.VI_rand)
    result['geary'] = block(geary.C if not constant else NAN, geary.EC, geary.VC_rand)
    result['getisOrdG'] = block(getis_ord.G, getis_ord.EG, getis_ord.VG)
    black = (subset != 0).astype(np.float64)
    result['blackCount'] = int(black.sum())
    # Halved ordered-link counts, esda's definition; esda itself raises (chi2 table) on one colour.
    links = [(slot, other) for slot, row in enumerate(keep) for other in make_weights(keep).neighbors[slot]]
    result['joinCount'] = {
        'blackBlack': sum(black[a] * black[b] for a, b in links) / 2,
        'blackWhite': sum(black[a] != black[b] for a, b in links) / 2,
        'whiteWhite': sum((1 - black[a]) * (1 - black[b]) for a, b in links) / 2,
        'joins': len(links) / 2,
    }
    if 0 < result['blackCount'] < n:
        join_counts = esda.Join_Counts(black, make_weights(keep), permutations=0)
        assert (join_counts.bb, join_counts.bw, join_counts.ww, join_counts.J) == tuple(result['joinCount'].values())
    return result


def get_local_expected(values, mask):
    keep = get_analysis_rows(values, mask)
    gi_star = [NAN] * ROW_COUNT
    local_i = [NAN] * ROW_COUNT
    if keep:
        subset = np.array([values[row] for row in keep], dtype=np.float64)
        constant = bool(np.all(subset == subset[0]))
        if not constant:
            scores = esda.G_Local(subset, make_weights(keep), transform='B', permutations=0, star=True).Zs
            moran = esda.Moran_Local(subset, make_weights(keep), transformation='o', permutations=0).Is
            for slot, row in enumerate(keep):
                gi_star[row] = finite_or_nan(scores[slot])
                local_i[row] = finite_or_nan(moran[slot])
    return {'count': len(keep), 'giStar': gi_star, 'localI': local_i}


def with_rows(replacements):
    values = list(BASE_VALUES)
    for row, value in replacements.items():
        values[row] = value
    return values


CASES = [
    ('baseline', 'no edge values', BASE_VALUES, None),
    ('NaN is missing', 'rows 2 and 7 leave n, the moments and every neighborhood', with_rows({2: NAN, 7: NAN}), None),
    ('+Infinity is missing', 'row 4 is missing, like NaN', with_rows({4: math.inf}), None),
    ('-Infinity is missing', 'row 9 is missing, like NaN', with_rows({9: -math.inf}), None),
    ('-0 equals +0', 'rows 3 and 6 hold -0: identical to the baseline (white for join counts)', with_rows({3: -0.0, 6: -0.0}), None),
    ('mask and NaN', 'masked row 5 and NaN row 8 are both outside the analysis', with_rows({8: NAN}), [0 if row == 5 else 1 for row in range(ROW_COUNT)]),
    ('mask values other than 1', 'any nonzero mask word includes the row', BASE_VALUES, [7 if row % 2 else 1 for row in range(ROW_COUNT)]),
    ('empty input', 'no finite value: n = 0 and every statistic NaN', [NAN] * ROW_COUNT, None),
    ('three analysis rows', 'n = 3 < 4: the randomization variances, z and p are NaN', [3.0, 1.0, 2.0] + [NAN] * 9, None),
    ('constant values', 'zero variance: Moran and Geary are NaN, z and p NaN (G is defined)', [2.0] * ROW_COUNT, None),
]

global_cases = []
local_cases = []
for name, policy, values, mask in CASES:
    global_cases.append({'name': name, 'policy': policy, 'values': values, 'mask': mask, 'expected': get_global_expected(values, mask)})
    if name not in ('three analysis rows',):
        local_cases.append({'name': name, 'policy': policy, 'values': values, 'mask': mask, 'expected': get_local_expected(values, mask)})

fixture = {
    'generator': 'test/oracles/statistics-edge-policy.py',
    'versions': {
        'python': platform.python_version(),
        'numpy': np.__version__,
        'scipy': scipy.__version__,
        'esda': esda.__version__,
        'libpysal': libpysal.__version__,
    },
    'tolerances': {
        'statistic': {'absolute': 1e-5, 'relative': 1e-4},
        'zScore': {'absolute': 1e-4, 'relative': 1e-3},
        'pValue': {'absolute': 1e-5, 'relative': 1e-3},
        'counts': 'exact',
    },
    'rows': [NEIGHBORS[row] for row in range(ROW_COUNT)],
    'global': global_cases,
    'local': local_cases,
    'parameters': {
        'policy': 'a per-frame significanceLevel outside (0, 1) or NaN is refused by the packer; written raw '
        'past the packer, a NaN level makes every local Moran quadrant 0 (p <= NaN is false)',
        'rejectedSignificanceLevels': [0.0, 1.0, 1.5, -0.25, NAN, math.inf],
    },
}
with open(OUTPUT, 'w') as file:
    json.dump(encode(fixture), file, separators=(',', ':'))
    file.write('\n')
for case in global_cases:
    expected = case['expected']
    print(case['name'], 'n', expected['count'], 'I', expected['moran']['statistic'], 'z', expected['moran']['zRandomization'], 'G', expected['getisOrdG']['statistic'], 'zG', expected['getisOrdG']['zRandomization'], expected['joinCount'])
print('wrote', OUTPUT)
