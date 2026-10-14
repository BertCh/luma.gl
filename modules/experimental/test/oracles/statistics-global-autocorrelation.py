# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Reference global Moran's I, Geary's C, Getis-Ord General G and join counts from esda.

Writes fixtures/statistics-global-autocorrelation.json next to this script, consumed (through
test/gpu-spatial-analysis/global-spatial-statistics/global-spatial-statistics-reference.ts) by
gpu-global-spatial-statistics.spec.ts and gpu-global-spatial-statistics.node.spec.ts. Run with
    <venv>/bin/python -I modules/experimental/test/oracles/statistics-global-autocorrelation.py
in a venv with numpy, scipy, esda and libpysal (versions are recorded in the fixture).

Moran / Geary / General G scene (`reference`): 40 rows, two bumps plus U(0, 2.6) noise (seed 11),
rounded to 3 decimals and read as float32. Asymmetric positive weights: out-links to the +-1, +-2
neighbors with float32-exact weights 0.5 + ((7 i + 3 j) mod 5) 0.375; rows 7 and 19 have no
out-links (islands) but stay targets of other rows. Configurations:
- `keep`: esda on the whole W (islands stay in n);
- `excludeSubset`: esda on W.subset without the island rows;
- `standardizedMasked`: esda on the subset W without the `masked` rows, row-standardized again;
- `binaryKeep` / `binaryExclude`: the same on binary W, where General G is defined (esda.G forces
  binary weights).
Every statistic comes from `esda.Moran` / `esda.Geary` / `esda.G` with `permutations=0`; p-values are
`scipy.stats.norm` tails of esda's z.

Join counts (`joinCounts`): `esda.Join_Counts` gives BB, BW, WW and J on the adjacency list (each
directed link is half a join, as on the GPU). esda has no analytic moments, so the randomization
(non-free sampling) mean and variance of BB and BW are computed here by exhaustive enumeration of
every placement of the n1 black rows (C(16, 7) = 11440 and C(12, 5) = 792 placements). This checks
the Cliff-Ord formulas on the GPU independently of their algebra. Scenes: a symmetric 4 x 4 rook
lattice, and 12 rows with asymmetric positive (non-binary, binarized by the contributor) out-links.

Tolerances (stored in `tolerances`): statistics and moments 1e-6 absolute + 1e-3 relative, z 2e-3 +
2e-3 |z|, p 2e-5 + 5e-3 relative; join counts BB / BW / WW / J are exact (integer halves). These are
the bounds the GPU spec uses: float32 sums of at most 160 weighted terms with a second-pass mean.
"""

import itertools
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
OUTPUT = os.path.join(HERE, 'fixtures', 'statistics-global-autocorrelation.json')
ROW_COUNT = 40
ISLANDS = [7, 19]
MASKED = [3, 11, 25, 30]


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


generator = np.random.default_rng(11)
index = np.arange(ROW_COUNT)
raw = (
    6
    + 2.4 * np.exp(-(((index - 10) / 4) ** 2))
    + 2.2 * np.exp(-(((index - 30) / 3) ** 2))
    + generator.uniform(0, 2.6, ROW_COUNT)
)
# Exactly what the GPU reads.
values = np.round(raw, 3).astype(np.float32).astype(np.float64)
neighbors = {}
weights = {}
for row in range(ROW_COUNT):
    if row in ISLANDS:
        neighbors[row], weights[row] = [], []
        continue
    candidates = [other for other in (row - 2, row - 1, row + 1, row + 2) if 0 <= other < ROW_COUNT]
    neighbors[row] = candidates
    weights[row] = [float(np.float32(0.5 + ((row * 7 + other * 3) % 5) * 0.375)) for other in candidates]


def make_weights(keep, standardize=False, binary=False):
    keep = list(keep)
    position = {row: slot for slot, row in enumerate(keep)}
    subset_neighbors, subset_weights = {}, {}
    for row in keep:
        pairs = [(position[other], 1.0 if binary else weight) for other, weight in zip(neighbors[row], weights[row]) if other in position]
        subset_neighbors[position[row]] = [slot for slot, _ in pairs]
        subset_weights[position[row]] = [weight for _, weight in pairs]
    result = W(subset_neighbors, subset_weights, silence_warnings=True)
    if standardize:
        result.transform = 'r'
    return result


def add_p_values(block):
    for z_name, p_name in (('zNormality', 'pNormality'), ('zRandomization', 'pRandomization')):
        if z_name in block:
            z = block[z_name]
            block[p_name] = float(2 * norm.sf(abs(z)))
            block[p_name + 'Greater'] = float(norm.sf(z))
            block[p_name + 'Lesser'] = float(norm.cdf(z))


def compute_statistics(keep, standardize=False, binary=False):
    # esda mutates the transform of the W it is given, so every statistic gets a fresh W.
    keep = list(keep)
    subset = values[keep]
    transform = 'r' if standardize else 'O'
    base = make_weights(keep, standardize, binary)
    result = {
        'count': len(keep),
        's0': float(base.s0),
        's1': float(base.s1),
        's2': float(base.s2),
        'mean': float(subset.mean()),
        'variance': float(subset.var()),
    }
    moran = esda.Moran(subset, make_weights(keep, standardize, binary), transformation=transform, permutations=0)
    geary = esda.Geary(subset, make_weights(keep, standardize, binary), transformation=transform, permutations=0)
    result['moran'] = {
        'statistic': moran.I,
        'expected': moran.EI,
        'varianceNormality': moran.VI_norm,
        'zNormality': moran.z_norm,
        'varianceRandomization': moran.VI_rand,
        'zRandomization': moran.z_rand,
    }
    result['geary'] = {
        'statistic': geary.C,
        'expected': geary.EC,
        'varianceNormality': geary.VC_norm,
        'zNormality': geary.z_norm,
        'varianceRandomization': geary.VC_rand,
        'zRandomization': geary.z_rand,
    }
    if binary:
        getis_ord = esda.G(subset, make_weights(keep, standardize, True), permutations=0)
        result['getisOrdG'] = {
            'statistic': getis_ord.G,
            'expected': getis_ord.EG,
            'varianceRandomization': getis_ord.VG,
            'zRandomization': getis_ord.z_norm,
        }
    # Direct formula check of Moran's I on the weights as given.
    sparse = base.sparse
    centered = subset - subset.mean()
    assert abs(moran.I - len(subset) / sparse.sum() * (centered @ (sparse @ centered)) / (centered @ centered)) < 1e-9
    for key in ('moran', 'geary', 'getisOrdG'):
        if key in result:
            add_p_values(result[key])
    return result


def count_joins(links, black):
    """Ordered-link join counts halved: (BB, BW, WW)."""
    black_black = sum(1 for source, target in links if black[source] and black[target]) / 2
    black_white = sum(1 for source, target in links if black[source] != black[target]) / 2
    white_white = sum(1 for source, target in links if not black[source] and not black[target]) / 2
    return black_black, black_white, white_white


def make_join_count_scene(name, row_count, lists, black_rows):
    """`lists[row]` = [(neighbor, weight)]; every weight is positive (binarized to 1)."""
    black = [1.0 if row in black_rows else 0.0 for row in range(row_count)]
    links = [(row, neighbor) for row in range(row_count) for neighbor, _ in lists[row]]
    binary = W({row: [neighbor for neighbor, _ in lists[row]] for row in range(row_count)}, silence_warnings=True)
    join_counts = esda.Join_Counts(np.array(black), binary, permutations=0)
    observed = count_joins(links, black)
    assert abs(join_counts.bb - observed[0]) < 1e-12 and abs(join_counts.bw - observed[1]) < 1e-12
    # Randomization moments: every placement of the black rows is equally likely.
    black_black_draws, black_white_draws = [], []
    for placement in itertools.combinations(range(row_count), len(black_rows)):
        chosen = set(placement)
        draw = count_joins(links, [row in chosen for row in range(row_count)])
        black_black_draws.append(draw[0])
        black_white_draws.append(draw[1])
    black_black_draws = np.array(black_black_draws)
    black_white_draws = np.array(black_white_draws)
    expected_black_black = float(black_black_draws.mean())
    variance_black_black = float(black_black_draws.var())
    expected_black_white = float(black_white_draws.mean())
    variance_black_white = float(black_white_draws.var())
    z_black_black = (join_counts.bb - expected_black_black) / math.sqrt(variance_black_black)
    z_black_white = (join_counts.bw - expected_black_white) / math.sqrt(variance_black_white)
    return {
        'name': name,
        'rows': [{'neighbors': [neighbor for neighbor, _ in lists[row]], 'weights': [weight for _, weight in lists[row]]} for row in range(row_count)],
        'values': black,
        'blackCount': len(black_rows),
        'placementCount': len(black_black_draws),
        'blackBlack': float(join_counts.bb),
        'blackWhite': float(join_counts.bw),
        'whiteWhite': float(join_counts.ww),
        'joins': float(join_counts.J),
        'expectedBlackBlack': expected_black_black,
        'varianceBlackBlack': variance_black_black,
        'zBlackBlack': float(z_black_black),
        'pBlackBlack': float(2 * norm.sf(abs(z_black_black))),
        'expectedBlackWhite': expected_black_white,
        'varianceBlackWhite': variance_black_white,
        'zBlackWhite': float(z_black_white),
        'pBlackWhite': float(2 * norm.sf(abs(z_black_white))),
    }


def make_rook_lists(side):
    lists = []
    for row in range(side * side):
        x, y = row % side, row // side
        entries = []
        for dx, dy in ((0, -1), (-1, 0), (1, 0), (0, 1)):
            if 0 <= x + dx < side and 0 <= y + dy < side:
                entries.append(((y + dy) * side + x + dx, 1.0))
        lists.append(entries)
    return lists


def make_asymmetric_lists(row_count):
    lists = []
    for row in range(row_count):
        entries = []
        for other in (row - 3, row - 1, row + 1, row + 2):
            # Drop about a third of the links deterministically, so W is asymmetric.
            if 0 <= other < row_count and (row * 5 + other) % 3 != 0:
                entries.append((other, float(np.float32(0.25 + ((row + 2 * other) % 4) * 0.5))))
        lists.append(entries)
    return lists


all_rows = list(range(ROW_COUNT))
non_island = [row for row in all_rows if row not in ISLANDS]
masked_keep = [row for row in all_rows if row not in MASKED]

reference = {
    'values': [float(value) for value in values],
    'islands': ISLANDS,
    'masked': MASKED,
    'rows': [{'neighbors': neighbors[row], 'weights': weights[row]} for row in all_rows],
    'keep': compute_statistics(all_rows),
    'excludeSubset': compute_statistics(non_island),
    'standardizedMasked': compute_statistics(masked_keep, standardize=True),
    'binaryKeep': compute_statistics(all_rows, binary=True),
    'binaryExclude': compute_statistics(non_island, binary=True),
}
join_count_scenes = [
    make_join_count_scene('rook-4x4', 16, make_rook_lists(4), {0, 1, 4, 5, 10, 14, 15}),
    make_join_count_scene('asymmetric-12', 12, make_asymmetric_lists(12), {1, 2, 3, 8, 11}),
]
for scene in join_count_scenes:
    assert all(scene['rows'][row]['neighbors'] for row in range(len(scene['values']))), 'no islands in join count scenes'

fixture = {
    'generator': 'test/oracles/statistics-global-autocorrelation.py',
    'versions': {
        'python': platform.python_version(),
        'numpy': np.__version__,
        'scipy': scipy.__version__,
        'esda': esda.__version__,
        'libpysal': libpysal.__version__,
    },
    'tolerances': {
        'statistic': {'absolute': 1e-6, 'relative': 1e-3},
        'zScore': {'absolute': 2e-3, 'relative': 2e-3},
        'pValue': {'absolute': 2e-5, 'relative': 5e-3},
        'joinCounts': 'exact',
        'derivation': 'float32 inputs identical to the oracle; float32 sums of at most 160 weighted terms '
        'with a second-pass mean; join counts are integer halves',
    },
    'reference': reference,
    'joinCounts': join_count_scenes,
}
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, 'w') as file:
    json.dump(encode(fixture), file, separators=(',', ':'))
    file.write('\n')
for scene in join_count_scenes:
    print(scene['name'], {key: scene[key] for key in ('blackBlack', 'blackWhite', 'whiteWhite', 'joins', 'expectedBlackBlack', 'varianceBlackBlack', 'zBlackBlack', 'expectedBlackWhite', 'varianceBlackWhite', 'zBlackWhite')})
print('wrote', OUTPUT)
