# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Reference EB-standardized local Moran (esda) and Benjamini-Hochberg adjustments (statsmodels)
for the rate-cluster map recipe.

Writes fixtures/statistics-rate-cluster-recipe.json next to this script, consumed (through
test/gpu-spatial-analysis/recipes/rate-cluster-esda-reference.ts) by rate-cluster-map-recipe.spec.ts.
Run from anywhere with
    <venv>/bin/python -I modules/experimental/test/oracles/statistics-rate-cluster-recipe.py
in a venv with numpy, esda, libpysal and statsmodels (versions are recorded in the fixture).

Scenes:
- `parity`: a 5 x 4 lattice (row-major ids, 5 columns), integer events and populations; rook
  contiguity `lat2W(4, 5, rook=True)`, row-standardized. References: `assuncaoRate` =
  `esda.smoothing.assuncao_rate(e, b)`, `localI` and `quadrants` =
  `esda.Moran_Local_Rate(e, b, w, permutations=0, transformation='r')` `.Is` / `.q`.
- `benjaminiHochberg`: 25 fixed p-values; `adjusted` =
  `statsmodels.stats.multitest.multipletests(p, method='fdr_bh')[1]`. The spec interleaves NaN
  (untested) rows, which must not enter the family.

Tolerances (stored in `tolerances`, the bounds the spec asserts, absolute + relative): the GPU reads
float32 events and populations and computes the Assuncao-Reis rate in float32 (a ratio of sums over
20 rows, a few ulp of the z range 2: 2e-4 + 2e-4 relative); local I multiplies two such standardized
values and a lag over at most 4 neighbors, amplifying the rate error about tenfold (2e-3 + 2e-3).
The adjusted GPU p-values are compared to the double-precision CPU adjustment of the GPU p-values
(1e-5); the CPU adjustment itself is compared to statsmodels to 12 decimals (both are the same
double-precision step-up formula).
"""

import json
import math
import os
import platform
import warnings

import esda
import libpysal
import numpy as np
import statsmodels
from esda.smoothing import assuncao_rate
from libpysal.weights import lat2W
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'fixtures', 'statistics-rate-cluster-recipe.json')
COLUMNS = 5
ROWS = 4


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


events = np.array(
    [4, 9, 12, 3, 14, 6, 11, 20, 15, 8, 3, 7, 16, 18, 5, 2, 5, 9, 10, 4], dtype=float
)
populations = np.array(
    [800, 1200, 900, 1000, 1500, 700, 1100, 1600, 1300, 900,
     900, 1000, 1400, 1500, 1100, 1200, 900, 1000, 1300, 800],
    dtype=float,
)
weights = lat2W(ROWS, COLUMNS, rook=True)
weights.transform = 'r'
local = esda.Moran_Local_Rate(events, populations, weights, permutations=0, transformation='r')

family = np.array([
    0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216, 0.222, 0.251, 0.269,
    0.275, 0.34, 0.341, 0.384, 0.569, 0.594, 0.696, 0.762, 0.94, 0.942, 0.975, 0.986,
])

fixture = {
    'generator': 'test/oracles/statistics-rate-cluster-recipe.py',
    'versions': {
        'python': platform.python_version(),
        'numpy': np.__version__,
        'esda': esda.__version__,
        'libpysal': libpysal.__version__,
        'statsmodels': statsmodels.__version__,
    },
    'tolerances': {
        'assuncaoRate': {'absolute': 2e-4, 'relative': 2e-4},
        'localI': {'absolute': 2e-3, 'relative': 2e-3},
        'adjustedGPU': {'absolute': 1e-5, 'relative': 1e-5},
        'adjustedCPU': {'decimals': 12},
    },
    'parity': {
        'columns': COLUMNS,
        'rows': ROWS,
        'events': events,
        'populations': populations,
        'assuncaoRate': assuncao_rate(events, populations),
        'localI': local.Is,
        'quadrants': [int(value) for value in local.q],
    },
    'benjaminiHochberg': {
        'pValues': family,
        'adjusted': multipletests(family, alpha=0.05, method='fdr_bh')[1],
    },
}

with open(OUTPUT, 'w') as file:
    json.dump(encode(fixture), file, indent=1)
    file.write('\n')
print(f'wrote {OUTPUT}')
