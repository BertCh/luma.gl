# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Reference local join counts (univariate, BJC, CLC), local bivariate Moran, local Geary labels and
Lee's L from esda.

Writes fixtures/statistics-local-join-counts-and-bivariate.json next to this script, consumed
(through test/gpu-spatial-analysis/permutation-inference/new-statistics-reference.ts) by
new-statistics.node.spec.ts and new-statistics.spec.ts. Run with
    <venv>/bin/python -I modules/experimental/test/oracles/statistics-local-join-counts-and-bivariate.py
in a venv with numpy, scipy, esda and libpysal (versions are recorded in the fixture). It takes about
a minute (the consistent BJC null below draws 99999 permutations per row in python).

Scene: 60 uniform points in [0, 100]^2 (RandomState 7), 5 nearest neighbors (ties by index),
row-standardized; x and y are float32 smooth fields plus noise; bx, bz and bv are binary variables
(bv is the noisy complement of bx, so BJC pairs occur). Permutation p-values use esda's own
conditional permutation with seed 12345 and 9999 permutations, so the GPU (a different generator)
is compared against them statistically, not bit for bit (the spec states each bound).

References: `Geary_Local` (statistic, p_sim, labels from an injected p vector), `Moran_Local_BV`
(I, quadrants, p_sim), `Join_Counts_Local` / `Join_Counts_Local_BV` (BJC, CLC) with the 'greater'
tail (esda's percentile two-sided p differs from 2 min(g, l) under ties), a BJC null of the same
statistic as observed (`joinCountBJCConsistentPSim`, numpy RandomState 99, 99999 draws), and
`Spatial_Pearson` / `Spatial_Pearson_Local` (Lee's L).

Tolerances (as the specs use them): deterministic statistics (local Geary, bivariate I, Lee's L)
are float32 sums of 5 row-standardized terms, 1e-4 absolute + 1e-4 relative; join counts are exact
integers. Permutation p-values are Monte Carlo estimates of the same p from different generators:
the spec bounds each difference by 5 binomial standard errors, 5 sqrt(2 * 0.25 / permutations).
"""

import json
import warnings

import numpy as np
from libpysal.weights import W
import esda
import esda.geary_local as geary_local_module
from esda import Geary_Local, Join_Counts_Local, Join_Counts_Local_BV, Moran_Local_BV, Spatial_Pearson, Spatial_Pearson_Local

warnings.filterwarnings('ignore')
import math
import os
import platform

import libpysal
import scipy

HERE = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(HERE, 'fixtures', 'statistics-local-join-counts-and-bivariate.json')
rng = np.random.RandomState(7)
n = 60
k = 5
points = rng.rand(n, 2) * 100.0
field = np.sin(points[:, 0] / 20.0) + np.cos(points[:, 1] / 25.0)
x = (field + 0.35 * rng.randn(n)).astype(np.float32).astype(np.float64)
y = (0.7 * field + 0.5 * np.cos(points[:, 0] / 30.0) + 0.45 * rng.randn(n)).astype(np.float32).astype(np.float64)
# Binary variables: the upper ~45 percent of two smooth fields (spatially clustered ones).
bx = (field + 0.25 * rng.randn(n) > 0.15).astype(np.float64)
bz = (0.6 * field + 0.5 * np.cos(points[:, 0] / 30.0) + 0.25 * rng.randn(n) > 0.1).astype(np.float64)

# BJC pair: z is the (noisy) complement of x, so x_i = 1, z_i = 0 with x_j = 0, z_j = 1 neighbors occur.
flip = rng.rand(n) < 0.15
bv = np.where(flip, bx, 1.0 - bx)

neighbors = {}
weights = {}
for i in range(n):
    distance = np.hypot(*(points - points[i]).T)
    order = [j for j in np.lexsort((np.arange(n), distance)) if j != i][:k]
    order = sorted(int(j) for j in order)
    neighbors[i] = order
    weights[i] = [1.0 / k] * k

def make_w():
    return W({i: list(v) for i, v in neighbors.items()}, {i: list(v) for i, v in weights.items()}, id_order=list(range(n)))

SEED = 12345
PERMUTATIONS = 9999
result = {'n': n, 'k': k, 'x': x.tolist(), 'y': y.tolist(), 'bx': bx.tolist(), 'bz': bz.tolist(), 'bv': bv.tolist(),
          'neighbors': [neighbors[i] for i in range(n)], 'permutations': PERMUTATIONS}

# --- Local Geary (statistic and labels) ---
lg = Geary_Local(connectivity=make_w(), permutations=PERMUTATIONS, seed=SEED, alternative='two-sided').fit(x)
result['gearyLocal'] = lg.localG.tolist()
result['gearyPSim'] = lg.p_sim.astype(np.float64).tolist()
# Labels with injected p-values (esda's label rule, fed a fixed p vector with one NaN).
injected = np.array([(((i * 37) % 100) + 1) / 1000.0 for i in range(n)], dtype=np.float32)
injected[3] = np.nan
original = geary_local_module._crand_plus
geary_local_module._crand_plus = lambda **kwargs: (injected.copy(), None)
labelled = Geary_Local(connectivity=make_w(), labels=True, sig=0.05, permutations=99, seed=SEED, alternative='two-sided').fit(x)
geary_local_module._crand_plus = original
result['gearyInjectedP'] = [None if np.isnan(v) else float(v) for v in injected.astype(np.float64)]
result['gearyLabels'] = [None if np.isnan(v) else int(v) for v in labelled.labs]

# --- Local bivariate Moran ---
bvm = Moran_Local_BV(x, y, make_w(), transformation='r', permutations=PERMUTATIONS, seed=SEED, alternative='two-sided')
result['bivariateMoranI'] = bvm.Is.tolist()
result['bivariateMoranQuadrants'] = [int(v) for v in bvm.q]
result['bivariateMoranPSim'] = [float(v) for v in bvm.p_sim]

# --- Local join counts ---
# Join counts are discrete: esda's percentile-based 'two-sided' p differs from 2 min(g, l) under ties, so
# the pins use the one-sided 'greater' tail (clustering), whose counts are identical in both.
ljc = Join_Counts_Local(connectivity=make_w(), seed=SEED, alternative='greater').fit(bx, permutations=PERMUTATIONS)
result['joinCount'] = ljc.LJC.tolist()
result['joinCountPSim'] = [None if np.isnan(v) else float(v) for v in ljc.p_sim]
for case in ('BJC', 'CLC'):
    second = bv if case == 'BJC' else bz
    bivariate_join = Join_Counts_Local_BV(connectivity=make_w(), seed=SEED, alternative='greater').fit(bx, second, case=case, permutations=PERMUTATIONS)
    result['joinCount' + case] = bivariate_join.LJC.tolist()
    result['joinCount' + case + 'PSim'] = [None if np.isnan(v) else float(v) for v in bivariate_join.p_sim]

# BJC with the permutation null of the SAME statistic as observed (esda simulates x_i * sum(z_j) instead:
# it ignores z_i = 0 and x_j = 0 in the simulated term). 99999 draws per row from numpy, conditional permutation.
rng2 = np.random.RandomState(99)
bjc_p = []
for i in range(n):
    observed = result['joinCountBJC'][i]
    if observed == 0:
        bjc_p.append(None)
        continue
    nbr = len(neighbors[i])
    others = np.array([j for j in range(n) if j != i])
    greater = 0
    lesser = 0
    draws = 99999
    for _ in range(draws):
        pick = rng2.choice(others, nbr, replace=False)
        simulated = float(np.sum((1 - bx[pick]) * bv[pick])) * bx[i] * (1 - bv[i])
        greater += simulated >= observed
        lesser += simulated <= observed
    bjc_p.append((greater + 1) / (draws + 1))
result['joinCountBJCConsistentPSim'] = bjc_p

# --- Lee's L ---
sparse = make_w().sparse.tocsr()
np.random.seed(SEED)
sp = Spatial_Pearson(connectivity=sparse, permutations=PERMUTATIONS).fit(x.reshape(-1, 1), y.reshape(-1, 1))
result['leeGlobal'] = float(sp.association_[0, 1])
result['leeGlobalSmoothingX'] = float(sp.association_[0, 0])
result['leeGlobalSmoothingY'] = float(sp.association_[1, 1])
result['leeGlobalPSim'] = float(sp.significance_[0, 1])
np.random.seed(SEED)
spl = Spatial_Pearson_Local(connectivity=sparse, permutations=PERMUTATIONS).fit(x.reshape(-1, 1), y.reshape(-1, 1))
result['leeLocal'] = [float(v) for v in np.asarray(spl.associations_).ravel()]
result['leeLocalPSim'] = [float(v) for v in np.asarray(spl.significance_).ravel()]



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


fixture = {
    'generator': 'test/oracles/statistics-local-join-counts-and-bivariate.py',
    'versions': {
        'python': platform.python_version(),
        'numpy': np.__version__,
        'scipy': scipy.__version__,
        'esda': esda.__version__,
        'libpysal': libpysal.__version__,
    },
    'seed': SEED,
    'tolerances': {
        'statistic': {'absolute': 1e-4, 'relative': 1e-4},
        'joinCount': 'exact',
        'permutationPValue': '5 binomial standard errors: 5 sqrt(2 * 0.25 / permutations)',
    },
    'reference': result,
}
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w') as handle:
    json.dump(encode(fixture), handle, separators=(',', ':'))
    handle.write('\n')
print('wrote', out_path)
