# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Reference GWR local inference, bandwidth search and spherical fits from mgwr (float64).

Reads the `planar_12_77` (`createNonstationarityScene(12, 77)`) and `geographic_12_77`
(`createGeographicScene` of it) scenes of fixtures/statistics-regression-scenes.json (kept in sync
with the TypeScript generators by statistics-regression-scenes.node.spec.ts) and writes
fixtures/statistics-gwr.json, consumed through
test/gpu-spatial-analysis/spatial-regression/gwr-mgwr-inference-reference.ts by the GWR inference
specs. Run with
    <venv>/bin/python -I modules/experimental/test/oracles/statistics-gwr.py
in a venv with numpy and mgwr (versions are recorded in the fixture; it takes about a minute).

All fits are `constant = True` bisquare unless stated. Planar scene: 144 rows, 2 predictors (k = 3).
`significant` is `filter_tvals() != 0`, that is |t| >= `critical_tval()` with mgwr's adjusted alpha at
0.05 (`adj_alpha[1]`). `sigma2` is mgwr's `sigma2_v1` (RSS / (n - tr(S))). Bandwidth search: the
exhaustive adaptive AICc curve k = 6..144 and a fixed curve 1.0..6.0 against `Sel_BW` golden section.
Spherical fits use mgwr's haversine radius 6371.0 km, so bandwidths are kilometres there.
Golden section (`bandwidthSearch.goldenSection`): `golden_trace` is a line-by-line port of
`mgwr.search.golden_section` that also records every evaluated bandwidth in call order and counts
iterations (mgwr returns neither); each case asserts that the port returns the bandwidth of the real
`Sel_BW(...).search(search_method='golden_section')` call with the same bounds. `lower` / `upper` are
the bounds the search used: mgwr's `_init_section` defaults (adaptive `[40 + 2 k, n]`, fixed
`[min_dist / 2, 2 max_dist]`) or the stated bounds. `bandwidth` is what mgwr returns (rounded to two
decimals), `unroundedBandwidth` the value whose AICc `aicc` is. The adaptive default upper bound is capped at 128
(the GPU neighbour list), so adaptive cases pass `bw_max=128` where the default would be `n = 144`.
`studentTQuantiles`: `scipy.stats.t.ppf(1 - tail / 2, df)` as `[df, two-sided tail, quantile]` for the
GWR critical-value spec.
Tolerances are stated by each consuming spec (float32 local solves).
"""

import json
import math
import os
import platform
import warnings

import mgwr
import numpy as np
import scipy
from scipy.stats import t as student_t
from mgwr.diagnostics import get_AICc
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW

warnings.filterwarnings('ignore')


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


def f(x):
    return float(np.ravel(x)[0])


HERE = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(HERE, 'fixtures', 'statistics-regression-scenes.json')
OUTPUT = os.path.join(HERE, 'fixtures', 'statistics-gwr.json')
scenes = json.load(open(INPUT))['scenes']


def load(name):
    s = scenes[name]
    pos = np.array(s['positions'], dtype=float).reshape(-1, 2)
    k = s['k']
    X = np.array(s['predictors'], dtype=float).reshape(-1, k)
    y = np.array(s['response'], dtype=float).reshape(-1, 1)
    return pos, X, y


def fit_summary(pos, X, y, bw, fixed, kernel, spherical=False, full=True):
    model = GWR(pos, y, X, bw, fixed=fixed, kernel=kernel, spherical=spherical, constant=True, n_jobs=1)
    res = model.fit()
    crit = res.critical_tval()
    filt = res.filter_tvals()
    out = {
        'enp': f(res.ENP),
        'trS': f(res.tr_S),
        'rss': f(res.resid_ss),
        'sigma2': f(res.sigma2),
        'aicc': f(res.aicc),
        'aic': f(res.aic),
        'bic': f(res.bic),
        'llf': f(res.llf),
        'r2': f(res.R2),
        'adjAlpha05': f(res.adj_alpha[1]),
        'criticalT': f(crit),
        'n': int(res.n),
        'k': int(res.k),
        'significantCount': int((filt != 0).sum()),
    }
    if full:
        out['coefficients'] = res.params.reshape(-1).tolist()
        out['standardErrors'] = res.bse.reshape(-1).tolist()
        out['tValues'] = res.tvalues.reshape(-1).tolist()
        out['significant'] = (filt != 0).astype(int).reshape(-1).tolist()
    return out


def aicc_for(pos, X, y, bw, fixed, kernel, spherical=False):
    model = GWR(pos, y, X, bw, fixed=fixed, kernel=kernel, spherical=spherical, constant=True, n_jobs=1)
    return f(get_AICc(model.fit(lite=True)))


pins = {}
pos, X, y = load('planar_12_77')
pins['planar'] = {
    'fixedBisquare22': fit_summary(pos, X, y, 2.2, True, 'bisquare'),
    'adaptiveBisquare20': fit_summary(pos, X, y, 20, False, 'bisquare'),
    'fixedGaussian15': fit_summary(pos, X, y, 1.5, True, 'gaussian', full=False),
    # Exact (untruncated) Gaussian fits with per-row arrays: the reference of the gaussianCutoff specs.
    'fixedGaussian15Full': fit_summary(pos, X, y, 1.5, True, 'gaussian'),
    'adaptiveGaussian20Full': fit_summary(pos, X, y, 20, False, 'gaussian'),
    # Non-default level 0.1 is mgwr adj_alpha[0]; checks the level plumbing.
    'adjAlpha10': f(GWR(pos, y, X, 2.2, fixed=True, kernel='bisquare', constant=True, n_jobs=1).fit().adj_alpha[0]),
}

# Bandwidth search, adaptive bisquare: exhaustive integer AICc curve versus mgwr golden section.
ks = list(range(6, 145))
curve = [aicc_for(pos, X, y, k, False, 'bisquare') for k in ks]
best_k = ks[int(np.argmin(curve))]
sel = Sel_BW(pos, y, X, kernel='bisquare', fixed=False, constant=True, n_jobs=1)
golden_default = f(sel.search(criterion='AICc'))
sel2 = Sel_BW(pos, y, X, kernel='bisquare', fixed=False, constant=True, n_jobs=1)
golden_wide = f(sel2.search(criterion='AICc', bw_min=6, bw_max=144))
pins['adaptiveSearch'] = {
    'ks': ks,
    'aicc': curve,
    'argminK': best_k,
    'goldenDefaultBounds': golden_default,
    'goldenBounds6to144': golden_wide,
}

# Fixed bisquare: dense distance curve and golden section.
ds = [round(1.0 + 0.1 * i, 10) for i in range(0, 51)]
dcurve = [aicc_for(pos, X, y, d, True, 'bisquare') for d in ds]
sel3 = Sel_BW(pos, y, X, kernel='bisquare', fixed=True, constant=True, n_jobs=1)
fixed_golden = f(sel3.search(criterion='AICc', bw_min=1.0, bw_max=6.0))
pins['fixedSearch'] = {
    'distances': ds,
    'aicc': dcurve,
    'argminDistance': ds[int(np.argmin(dcurve))],
    'goldenBounds1to6': fixed_golden,
    'goldenAicc': aicc_for(pos, X, y, fixed_golden, True, 'bisquare'),
}

DELTA = 0.38197


def golden_trace(a, c, function, tol, max_iter, int_score):
    """Port of mgwr.search.golden_section recording the evaluations (call order) and the iteration count."""
    b = a + DELTA * np.abs(c - a)
    d = c - DELTA * np.abs(c - a)
    opt_val = None
    opt_score = np.inf
    diff = 1.0e9
    iters = 0
    cache = {}
    evaluated = []

    def score_of(value):
        if value in cache:
            return cache[value]
        score = function(value)
        cache[value] = score
        evaluated.append([float(value), float(score)])
        return score

    while np.abs(diff) > tol and iters < max_iter and a != np.inf:
        iters += 1
        if int_score:
            b = np.round(b)
            d = np.round(d)
        score_b = score_of(b)
        score_d = score_of(d)
        if score_b <= score_d:
            opt_val = b
            opt_score = score_b
            c = d
            d = b
            b = a + DELTA * np.abs(c - a)
        else:
            opt_val = d
            opt_score = score_d
            a = b
            b = d
            d = c - DELTA * np.abs(c - a)
        diff = score_b - score_d
    return float(opt_val), float(opt_score), iters, evaluated


def golden_case(label, pos, X, y, kernel, fixed, bw_min, bw_max):
    """Golden-section search of one configuration, verified against the real Sel_BW call."""
    n = pos.shape[0]
    if fixed:
        deltas = pos[:, None, :] - pos[None, :, :]
        distances = np.sqrt((deltas ** 2).sum(axis=2))
        np.fill_diagonal(distances, np.inf)
        lower = float(distances.min() / 2.0)
        np.fill_diagonal(distances, 0.0)
        upper = float(distances.max() * 2.0)
    else:
        lower = float(40 + 2 * (X.shape[1] + 1))
        upper = float(n)
    if bw_min is not None:
        lower = float(bw_min)
    if bw_max is not None:
        upper = float(bw_max)
    value, score, iterations, evaluated = golden_trace(
        lower, upper, lambda bw: aicc_for(pos, X, y, bw, fixed, kernel), 1.0e-6, 200, not fixed)
    real = Sel_BW(pos, y, X, kernel=kernel, fixed=fixed, constant=True, n_jobs=1)
    real_value = f(real.search(criterion='AICc', bw_min=bw_min, bw_max=bw_max))
    # mgwr returns opt_val rounded to two decimals (the AICc stays that of the unrounded value).
    assert real_value == round(value, 2), (label, real_value, value)
    return {
        'label': label,
        'kernel': kernel,
        'fixed': bool(fixed),
        'lower': lower,
        'upper': upper,
        'bandwidth': real_value,
        'unroundedBandwidth': value,
        'aicc': score,
        'iterationCount': int(iterations),
        'tolerance': 1.0e-6,
        'evaluated': evaluated,
    }


golden_cases = [
    golden_case('adaptive bisquare default lower, upper 128', pos, X, y, 'bisquare', False, None, 128),
    golden_case('adaptive bisquare 6 to 128', pos, X, y, 'bisquare', False, 6, 128),
    golden_case('adaptive gaussian default lower, upper 128', pos, X, y, 'gaussian', False, None, 128),
    golden_case('adaptive gaussian 6 to 128', pos, X, y, 'gaussian', False, 6, 128),
    golden_case('fixed bisquare default bounds', pos, X, y, 'bisquare', True, None, None),
    golden_case('fixed bisquare 1 to 6', pos, X, y, 'bisquare', True, 1.0, 6.0),
    golden_case('fixed gaussian default bounds', pos, X, y, 'gaussian', True, None, None),
    golden_case('fixed gaussian 0.5 to 6', pos, X, y, 'gaussian', True, 0.5, 6.0),
]
pins['goldenSection'] = golden_cases

# Spherical: mgwr haversine uses R = 6371.0 km, so distances and bandwidths are in kilometres.
gpos, gX, gy = load('geographic_12_77')
pins['spherical'] = {
    'radiusKilometers': 6371.0,
    'fixedBisquare25km': fit_summary(gpos, gX, gy, 25.0, True, 'bisquare', spherical=True),
    'adaptiveBisquare20': fit_summary(gpos, gX, gy, 20, False, 'bisquare', spherical=True),
}
fixture = {
    'generator': 'test/oracles/statistics-gwr.py',
    'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'mgwr': mgwr.__version__},
    'inference': {**pins['planar'], 'adjustedAlpha10': pins['planar'].pop('adjAlpha10')},
    'bandwidthSearch': {
        'adaptive': pins['adaptiveSearch'],
        'fixed': pins['fixedSearch'],
        'goldenSection': pins['goldenSection'],
    },
    'spherical': pins['spherical'],
    'studentTQuantiles': [
        [df, tail, float(student_t.ppf(1 - tail / 2, df))]
        for df in (1, 2, 5, 10, 30, 143, 2000, 20000)
        for tail in (0.5, 0.1, 0.05, 0.0025, 1e-4, 1e-6)
    ],
}
with open(OUTPUT, 'w') as file:
    json.dump(encode(fixture), file, separators=(',', ':'))
    file.write('\n')
print('wrote', OUTPUT)
