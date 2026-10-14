# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Reference Getis-Ord Gi* z-scores from esda on a lattice with a hot spot and a cold spot.

Writes fixtures/statistics-hot-spot-recipe.json next to this script, consumed (through
test/gpu-spatial-analysis/recipes/hot-spot-esda-fixture.ts) by hot-spot-analysis-recipe.node.spec.ts.
Run from anywhere with
    <venv>/bin/python -I modules/experimental/test/oracles/statistics-hot-spot-recipe.py
in a venv with numpy, esda and libpysal (versions are recorded in the fixture).

Scene: 8x8 lattice of Poisson(3) counts (seed 3), a hot block (+12) on rows/columns 0..2 and a block
of zeros on rows/columns 5..7 (a cold spot that must come out negative). Queen contiguity
(`lat2W(8, 8, rook=False)`); `offsets`/`neighbors` are the CSR of the sorted neighbors WITHOUT the
cell itself.

Reference: `zScores` = `esda.getisord.G_Local(values, w, transform='B', star=True, permutations=0).Zs`.

Tolerances (stored in `tolerances`): the node spec compares a double-precision TypeScript oracle to
esda with `toBeCloseTo(expected, 6)`, i.e. 5e-7 absolute; the two implementations sum the same at
most 9 terms in double precision (roundoff 1.1e-16), so the true difference is orders below that.
"""

import json
import math
import os
import platform
import warnings

import esda
import libpysal
import numpy as np
from esda.getisord import G_Local
from libpysal.weights import lat2W

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'fixtures', 'statistics-hot-spot-recipe.json')
SIZE = 8


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


generator = np.random.default_rng(3)
counts = generator.poisson(3, size=(SIZE, SIZE)).astype(float)
counts[0:3, 0:3] += 12  # hot block
counts[5:8, 5:8] = 0.0  # zero block: must be a cold spot
values = counts.ravel()
weights = lat2W(SIZE, SIZE, rook=False)
neighbor_lists = [sorted(weights.neighbors[cell]) for cell in range(SIZE * SIZE)]
local = G_Local(values, weights, transform='B', star=True, permutations=0)

reference = {
    'width': SIZE,
    'height': SIZE,
    'values': values,
    'zScores': local.Zs,
    'offsets': np.concatenate([[0], np.cumsum([len(entries) for entries in neighbor_lists])]),
    'neighbors': [other for entries in neighbor_lists for other in entries],
}

fixture = {
    'generator': 'test/oracles/statistics-hot-spot-recipe.py',
    'versions': {
        'python': platform.python_version(),
        'numpy': np.__version__,
        'esda': esda.__version__,
        'libpysal': libpysal.__version__,
    },
    'tolerances': {
        'zScore': {'decimals': 6},
        'derivation': 'double-precision TypeScript oracle against esda over at most 9 summed terms '
        '(roundoff 1.1e-16); the spec asserts 6 decimals (5e-7), far above that',
    },
    'reference': reference,
}
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, 'w') as file:
    json.dump(encode(fixture), file, separators=(',', ':'))
    file.write('\n')
print('wrote', OUTPUT)
