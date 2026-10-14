# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Reference OLS, spatial error (GM and ML) and spatial lag (ML) fits from PySAL spreg.

Reads fixtures/statistics-regression-scenes.json (the TypeScript scenes, kept in sync by
statistics-regression-scenes.node.spec.ts) and writes fixtures/statistics-spatial-regression.json,
consumed through test/gpu-spatial-analysis/spatial-regression/{spreg-ols-reference,
spatial-error-gm-reference,spatial-maximum-likelihood-reference}.ts by the OLS, GM error and ML specs.
Run with
    <venv>/bin/python -I modules/experimental/test/oracles/statistics-spatial-regression.py
in a venv with numpy, scipy, spreg and libpysal (versions are recorded in the fixture). Every fit is
float64 on the float32 inputs the GPU reads; weights are passed to spreg as `libpysal.weights.W` with
the exact CSR values (transform 'O'), so both sides see identical inputs.

Sections:
- `ordinaryLeastSquares`: `spreg.OLS(y, X)` with spreg's diagnostics. `breuschPagan` and
  `koenkerBassett` use `z = reg.x` (the `[1, X]` auxiliary design the GPU uses); spreg 1.9.1 tests
  `z == None`, which raises for an array, so the script runs spreg's own function source with
  `z is None`. `maximumVif` is the largest centered VIF; `correlationConditionNumber` the max/min
  eigenvalue of `np.corrcoef(X.T)`; `studentT` is `scipy.stats.t.sf` and `2 t.sf(|t|)`.
- `errorGM`: `spreg.GM_Error(y, X, w)` on the frozen `lattice` (10 x 10 rook, row-standardized) and
  `knn` (80 points, 4 nearest neighbors, row-standardized) scenes, with their inputs.
- `errorGMHom`: `spreg.GM_Error_Hom(y, X, w, A1='hom_sc', max_iter=1)` and `GM_Error` on `lattice`,
  `knn` and `binary` (`err_binary_324_9`), with the intermediate moments of the Hom path replayed from
  spreg's own functions and the float64 conditioning of the final filtered regression.
- `lagMaximumLikelihood` / `errorMaximumLikelihood`: `spreg.ML_Lag(..., method='full',
  spat_impacts='full')` and `spreg.ML_Error(..., method='full')`; z = estimate / se, p two-sided
  normal; impacts from `spmultiplier(w, rho, 'full')`; traces tr(W A^-1), tr((W A^-1)^2),
  tr((W A^-1)' W A^-1) in numpy. spreg minimizes with scipy `minimize_scalar` (xatol 1e-7), so its
  rho / lambda agree with the exact maximizer to about 1e-7.
- `diagnostics`: `spreg.OLS(y, X, w, spat_diag=True, moran=True)` LM tests (`[statistic, p]`), Moran
  of the residuals (`[I, z, p]`) and `sig2n`; `twoStage*`: `spreg.GM_Lag(y, X, w, w_lags=order,
  spat_diag=True)`; `maskedDiagnostics`: the OLS diagnostics on the fitted rows of a mask, with the
  restricted W built by hand from the surviving links (`original`, transform 'o'; libpysal 4.15
  `w_subset` drops the weight values) or `w_subset` row-standardized (`restandardized`);
  `twoStageDiagnostics`: numpy float64 first-stage F and partial R^2 of the excluded instruments
  (W y on [1, X, W X (, W^2 X)] against [1, X]), Belsley condition number of [1, P Z], correlation
  condition number and largest VIF of the centered P Z, condition number of the centered
  instruments, ||W||_inf and the GM_Lag rho with its admissibility flag (|rho| ||W||_inf >= 1);
  `twoStageIllConditioned`: `np.linalg.cond` of the unit-column [1, P Z] and of the correlation of
  the centered P Z.

Tolerances are stated by each consuming spec next to its assertions (float32 Gram accumulation and
solves; the specs derive them from the condition numbers recorded here).
"""

import inspect
import json
import math
import os
import platform
import warnings

import libpysal
import numpy as np
import scipy
import spreg
from libpysal.weights import W, w_subset
from scipy import stats
from scipy.stats import norm
from spreg import GM_Error, GM_Error_Hom, GM_Lag, ML_Error, ML_Lag
from spreg import diagnostics
from spreg import error_sp_hom
from spreg import ols as spreg_ols
from spreg import utils as spreg_utils
from spreg.sputils import spmultiplier

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(HERE, 'fixtures', 'statistics-regression-scenes.json')
OUTPUT = os.path.join(HERE, 'fixtures', 'statistics-spatial-regression.json')

# spreg 1.9.1 tests `z == None`, which raises for an array z; run spreg's own source with `z is None`.
for function_name in ('breusch_pagan', 'koenker_bassett'):
    source = inspect.getsource(getattr(diagnostics, function_name)).replace('z == None', 'z is None')
    exec(source, diagnostics.__dict__)


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


def flat(array):
    return [float(value) for value in np.asarray(array).flatten()]


def scalar(value):
    return float(np.asarray(value).flatten()[0])


def build(scene):
    k = scene['k']
    n = len(scene['response'])
    X = np.array(scene['predictors'], dtype=float).reshape(n, k)
    y = np.array(scene['response'], dtype=float).reshape(n, 1)
    if 'offsets' not in scene:
        return n, k, None, None, X, y
    offsets = scene['offsets']
    ids = list(range(n))
    neighbors = {row: [int(j) for j in scene['neighbors'][offsets[row]:offsets[row + 1]]] for row in ids}
    weights = {row: [float(v) for v in scene['weights'][offsets[row]:offsets[row + 1]]] for row in ids}
    w = W(neighbors, weights, id_order=ids, silence_warnings=True)
    w.transform = 'O'
    dense = np.zeros((n, n))
    for row in ids:
        for slot in range(offsets[row], offsets[row + 1]):
            dense[row, scene['neighbors'][slot]] = scene['weights'][slot]
    return n, k, w, dense, X, y


def unit(gram):
    scale = np.sqrt(np.diag(gram))
    return gram / np.outer(scale, scale)


def eigenvalue_ratio(matrix):
    values = np.linalg.eigvalsh(matrix)
    return float(values[-1] / values[0])


# --- OLS ---------------------------------------------------------------------------------------

def fit_ordinary_least_squares(scene):
    n, k, _, _, X, y = build(scene)
    reg = spreg.OLS(y, X)
    breusch_pagan = diagnostics.breusch_pagan(reg, z=reg.x)
    koenker_bassett = diagnostics.koenker_bassett(reg, z=reg.x)
    jarque_bera = reg.jarque_bera
    vifs = [value[0] for value in diagnostics.vif(reg)[1:]]
    correlation = np.corrcoef(X.T) if k > 1 else np.array([[1.0]])
    eigenvalues = np.linalg.eigvalsh(correlation)
    return {
        'n': int(reg.n),
        'betas': reg.betas[:, 0].tolist(),
        'stdErrors': [float(v) for v in reg.std_err],
        'tStats': [float(t[0]) for t in reg.t_stat],
        'pValues': [float(t[1]) for t in reg.t_stat],
        'sig2': float(reg.sig2),
        'sig2n': float(reg.sig2n),
        'utu': float(reg.utu),
        'r2': float(reg.r2),
        'ar2': float(reg.ar2),
        'logll': float(diagnostics.log_likelihood(reg)),
        'aic': float(diagnostics.akaike(reg)),
        'schwarz': float(diagnostics.schwarz(reg)),
        'mulColli': float(reg.mulColli),
        'jarqueBera': [float(jarque_bera['jb']), float(jarque_bera['pvalue'])],
        'breuschPagan': [float(breusch_pagan['bp']), float(breusch_pagan['pvalue']), int(breusch_pagan['df'])],
        'koenkerBassett': [float(koenker_bassett['kb']), float(koenker_bassett['pvalue']), int(koenker_bassett['df'])],
        'correlationConditionNumber': float(eigenvalues.max() / eigenvalues.min()),
        'maximumVif': float(max(vifs)),
        'vifs': [float(v) for v in vifs],
    }


# --- GM error ----------------------------------------------------------------------------------

def fit_error_gm(scene):
    n, k, w, _, X, y = build(scene)
    reg = GM_Error(y, X, w)
    betas = reg.betas.flatten()
    return {
        'offsets': scene['offsets'],
        'neighbors': scene['neighbors'],
        'weights': scene['weights'],
        'predictors': scene['predictors'],
        'response': scene['response'],
        'betas': [float(v) for v in betas[:-1]],
        'standardErrors': [float(v) for v in reg.std_err],
        'zStatistics': [float(z[0]) for z in reg.z_stat],
        'pValues': [float(z[1]) for z in reg.z_stat],
        'lambda': float(betas[-1]),
        'sigma2': float(reg.sig2),
        'pseudoRSquared': float(reg.pr2),
    }


def fit_error_gm_hom(scene):
    n, k, w, dense, X, y = build(scene)

    def conditioning(lam):
        filter_matrix = np.eye(n) - lam * dense
        filtered = filter_matrix @ np.hstack((np.ones((n, 1)), X))
        belsley = float(np.sqrt(eigenvalue_ratio(unit(filtered.T @ filtered))))
        factored = np.hstack((filter_matrix @ np.ones((n, 1)), filter_matrix @ (X - X.mean(axis=0))))
        correlation = unit(factored.T @ factored)
        return {
            'conditionNumber': belsley,
            'correlationConditionNumber': eigenvalue_ratio(correlation),
            'maximumVarianceInflationFactor': float(np.max(np.diag(np.linalg.inv(correlation))[1:])),
        }

    blocks = {}
    for tag, model, options in (('hom', GM_Error_Hom, {'A1': 'hom_sc', 'max_iter': 1}), ('kelejianPrucha', GM_Error, {})):
        reg = model(y, X, w, **options)
        betas = reg.betas.flatten()
        blocks[tag] = {
            'betas': [float(v) for v in betas[:-1]],
            'lambda': float(betas[-1]),
            'standardErrors': [float(v) for v in reg.std_err],
            'z': [float(z[0]) for z in reg.z_stat],
            'p': [float(z[1]) for z in reg.z_stat],
            'sigma2': float(reg.sig2),
            'pseudoRSquared': float(reg.pr2),
        }
        blocks[tag]['conditioning'] = conditioning(blocks[tag]['lambda'])
    # Intermediate quantities of the Hom path, replayed from spreg's own functions.
    sparse = w.sparse
    a1 = spreg_utils.get_A1_hom(sparse, scalarKP=True)
    a2 = spreg_utils.get_A2_hom(sparse)
    base = spreg_ols.BaseOLS(y=y, x=np.hstack((np.ones((n, 1)), X)))
    moments0 = error_sp_hom.moments_hom(sparse, a1, a2, base.u)
    lambda1 = spreg_utils.optim_moments(moments0)
    filtered_x = spreg_utils.get_spFilter(w, lambda1, base.x)
    filtered_y = spreg_utils.get_spFilter(w, lambda1, base.y)
    filtered = spreg_ols.BaseOLS(y=filtered_y, x=filtered_x)
    residuals = base.y - base.x @ filtered.betas
    moments1 = error_sp_hom.moments_hom(sparse, a1, a2, residuals)

    class Residuals:
        pass

    replay = Residuals()
    replay.u = residuals
    replay.x = base.x
    psi1 = error_sp_hom.get_vc_hom(sparse, a1, a2, replay, lambda1)[0]
    lambda2 = spreg_utils.optim_moments(moments1, psi1)
    psi2 = error_sp_hom.get_vc_hom(sparse, a1, a2, replay, lambda2)[0]
    return {
        'rowSumNorm': float(np.abs(dense).sum(axis=1).max()),
        'hom': blocks['hom'],
        'kelejianPrucha': blocks['kelejianPrucha'],
        'steps': {
            'lambda1': float(lambda1),
            'momentMatrix': flat(moments1[0]),
            'momentVector': flat(moments1[1]),
            'psiAtLambda1': flat(psi1),
            'psiAtLambda': flat(psi2),
        },
    }


# --- ML lag and error --------------------------------------------------------------------------

def get_conditioning(raw_columns, factored_columns):
    """Belsley condition number of the raw columns, correlation condition number of the factored ones."""
    belsley = float(np.sqrt(eigenvalue_ratio(unit(raw_columns.T @ raw_columns))))
    return {
        'conditionNumber': belsley,
        'correlationConditionNumber': eigenvalue_ratio(unit(factored_columns.T @ factored_columns)),
    }


def fit_error_maximum_likelihood(scene):
    n, k, w, dense, X, y = build(scene)
    reg = ML_Error(y, X, w, method='full')
    standard_errors = np.sqrt(np.diag(reg.vm))
    z = np.array([value[0] for value in reg.betas]) / standard_errors
    lam = scalar(reg.lam)
    constant = 1 - lam * dense.sum(axis=1, keepdims=True)
    filtered = X - lam * (dense @ X)
    filtered_centered = filtered - constant @ X.mean(axis=0, keepdims=True)
    conditioning = get_conditioning(np.hstack((constant, filtered)), np.hstack((constant, filtered_centered)))
    correlation = unit(np.hstack((constant, filtered_centered)).T @ np.hstack((constant, filtered_centered)))
    conditioning['maximumVarianceInflationFactor'] = float(np.max(np.diag(np.linalg.inv(correlation))[1:]))
    return {
        'rowSumNorm': float(np.abs(dense).sum(axis=1).max()),
        'betas': flat(reg.betas),
        'parameter': lam,
        'standardErrors': flat(standard_errors),
        'z': flat(z),
        'p': flat(2 * norm.sf(np.abs(z))),
        'sigma2': scalar(reg.sig2),
        'logLikelihood': scalar(reg.logll),
        'aic': float(reg.aic),
        'schwarz': float(reg.schwarz),
        'pseudoRSquared': float(reg.pr2),
        'specK': int(reg.k),
        'conditioning': conditioning,
    }


def fit_lag_maximum_likelihood(scene):
    n, k, w, dense, X, y = build(scene)
    reg = ML_Lag(y, X, w, method='full', spat_impacts='full', spat_diag=False)
    standard_errors = np.sqrt(np.diag(reg.vm))
    z = np.array([value[0] for value in reg.betas]) / standard_errors
    multipliers = spmultiplier(w, reg.rho, method='full')
    direct = float(multipliers['adi'])
    indirect = scalar(multipliers['aii'])
    total = scalar(multipliers['ati'])
    betas = flat(reg.betas)
    weighted_inverse = dense @ np.linalg.inv(np.eye(n) - reg.rho * dense)
    design = np.hstack((np.ones((n, 1)), X))
    centered = X - X.mean(axis=0)
    conditioning = get_conditioning(design, centered)
    conditioning['maximumVarianceInflationFactor'] = float(np.max(np.diag(np.linalg.inv(unit(centered.T @ centered)))))
    return {
        'rowSumNorm': float(np.abs(dense).sum(axis=1).max()),
        'betas': betas,
        'parameter': float(reg.rho),
        'standardErrors': flat(standard_errors),
        'z': flat(z),
        'p': flat(2 * norm.sf(np.abs(z))),
        'sigma2': scalar(reg.sig2),
        'logLikelihood': float(reg.logll),
        'aic': float(reg.aic),
        'schwarz': float(reg.schwarz),
        'pseudoRSquared': float(reg.pr2),
        'specK': int(reg.k),
        'conditioning': conditioning,
        'pseudoRSquaredSpatial': float(reg.pr2_e),
        'traces': [
            float(np.trace(weighted_inverse)),
            float(np.trace(weighted_inverse @ weighted_inverse)),
            float(np.trace(weighted_inverse.T @ weighted_inverse)),
        ],
        'impacts': {
            'multipliers': [direct, indirect, total],
            'values': [[direct * beta, indirect * beta, total * beta] for beta in betas[1:1 + k]],
        },
    }


# --- OLS spatial diagnostics and GM_Lag (2SLS) --------------------------------------------------

def get_diagnostics(reg):
    return {
        'lmLag': [float(v) for v in reg.lm_lag],
        'lmError': [float(v) for v in reg.lm_error],
        'robustLmLag': [float(v) for v in reg.rlm_lag],
        'robustLmError': [float(v) for v in reg.rlm_error],
        'lmSarma': [float(v) for v in reg.lm_sarma],
        'moran': [float(v) for v in reg.moran_res],
        'sigmaSquared': float(reg.sig2n),
    }


def fit_diagnostics(scene):
    n, k, w, _, X, y = build(scene)
    return get_diagnostics(spreg.OLS(y, X, w=w, spat_diag=True, moran=True))


def fit_two_stage(scene, order, with_p_values=False):
    n, k, w, _, X, y = build(scene)
    reg = GM_Lag(y, X, w=w, w_lags=order, spat_diag=True)
    result = {
        'betas': flat(reg.betas),
        'standardErrors': [float(v) for v in reg.std_err],
        'z': [float(v[0]) for v in reg.z_stat],
    }
    if with_p_values:
        result['pValues'] = [float(v[1]) for v in reg.z_stat]
    result.update({
        'sigma2': float(reg.sig2),
        'pseudoRSquared': float(reg.pr2),
        'anselinKelejian': [float(v) for v in reg.ak_test],
    })
    return result


def fit_masked_diagnostics(scene):
    n, k, _, _, X, y = build(scene)
    offsets = scene['offsets']
    neighbors = {row: [int(j) for j in scene['neighbors'][offsets[row]:offsets[row + 1]]] for row in range(n)}
    weights = {row: [float(v) for v in scene['weights'][offsets[row]:offsets[row + 1]]] for row in range(n)}
    w = W(neighbors, weights, silence_warnings=True)
    keep = [row for row in range(n) if scene['mask'][row] != 0]
    kept = set(keep)
    result = {'excluded': n - len(keep)}
    restricted = W(
        {row: [j for j in neighbors[row] if j in kept] for row in keep},
        {row: [value for j, value in zip(neighbors[row], weights[row]) if j in kept] for row in keep},
        id_order=keep,
        silence_warnings=True,
    )
    for mode in ('original', 'restandardized'):
        if mode == 'original':
            subset = restricted
            subset.transform = 'o'
        else:
            subset = w_subset(w, keep, silence_warnings=True)
            subset.transform = 'r'
            restricted.transform = 'r'
            # Each input row holds equal weights, so w_subset (binary) row-standardized equals the
            # hand-built restricted W row-standardized.
            assert np.allclose(subset.sparse.toarray(), restricted.sparse.toarray())
        reg = spreg.OLS(y[keep], X[keep], w=subset, spat_diag=True, moran=True)
        result[mode] = {**get_diagnostics(reg), 'weightsSum': float(subset.s0), 'islands': len(subset.islands)}
    return result


def fit_two_stage_diagnostics(scene):
    n, k, w, dense, X, _ = build(scene)
    y = np.array(scene['response'], dtype=float)
    result = {'weightsInfinityNorm': float(np.abs(dense).sum(axis=1).max())}
    for order in (1, 2):
        lagged = dense @ X
        columns = [X, lagged] + ([dense @ lagged] if order == 2 else [])
        instruments = np.hstack([np.ones((n, 1))] + columns)
        lag_y = dense @ y
        design = np.hstack([np.ones((n, 1)), X, lag_y[:, None]])

        def get_residual_sum(matrix):
            coefficients = np.linalg.lstsq(matrix, lag_y, rcond=None)[0]
            residual = lag_y - matrix @ coefficients
            return float(residual @ residual)

        residual_full = get_residual_sum(instruments)
        residual_restricted = get_residual_sum(np.hstack([np.ones((n, 1)), X]))
        total = float(((lag_y - lag_y.mean()) ** 2).sum())
        explained_full = total - residual_full
        explained_restricted = total - residual_restricted
        m = instruments.shape[1] - 1
        first_stage_f = ((explained_full - explained_restricted) / (m - k)) / (residual_full / (n - m - 1))
        partial = (explained_full - explained_restricted) / (total - explained_restricted)
        projected = instruments @ np.linalg.lstsq(instruments, design, rcond=None)[0]
        centered = projected[:, 1:] - projected[:, 1:].mean(axis=0)
        correlation = unit(centered.T @ centered)
        centered_instruments = instruments[:, 1:] - instruments[:, 1:].mean(axis=0)
        reg = GM_Lag(y.reshape(-1, 1), X, w=w, w_lags=order)
        rho = float(reg.betas[-1][0])
        result['order%d' % order] = {
            'rho': rho,
            'rhoOutsideAdmissibleRange': bool(abs(rho) * result['weightsInfinityNorm'] >= 1),
            'firstStageF': float(first_stage_f),
            'firstStagePartialRSquared': float(partial),
            'conditionNumber': float(np.sqrt(eigenvalue_ratio(unit(projected.T @ projected)))),
            'correlationConditionNumber': eigenvalue_ratio(correlation),
            'maximumVarianceInflationFactor': float(np.max(np.diag(np.linalg.inv(correlation)))),
            'instrumentCorrelationConditionNumber': eigenvalue_ratio(unit(centered_instruments.T @ centered_instruments)),
        }
    return result


def fit_two_stage_ill_conditioned(scene):
    n, k, _, dense, X, _ = build(scene)
    y = np.array(scene['response'], dtype=float)
    result = {}
    for order in (1, 2):
        lagged = dense @ X
        columns = [X, lagged] + ([dense @ lagged] if order == 2 else [])
        instruments = np.hstack([np.ones((n, 1))] + columns)
        design = np.hstack([np.ones((n, 1)), X, (dense @ y)[:, None]])
        projected = instruments @ np.linalg.lstsq(instruments, design, rcond=None)[0]
        centered = projected[:, 1:] - projected[:, 1:].mean(axis=0)
        result['order%d' % order] = {
            'conditionNumber': float(np.linalg.cond(projected / np.linalg.norm(projected, axis=0))),
            'correlationConditionNumber': float(np.linalg.cond(np.corrcoef(centered.T))),
        }
    return result


data = json.load(open(INPUT))
scenes = data['scenes']
frozen = data['frozen']
T_CASES = [(0.0, 5), (1.0, 5), (-1.0, 5), (2.5, 10), (-3.7, 296), (12.0, 296), (30.0, 40), (0.3, 1), (5.0, 2), (1.96, 1000), (40.0, 3)]
fixture = {
    'generator': 'test/oracles/statistics-spatial-regression.py',
    'versions': {
        'python': platform.python_version(),
        'numpy': np.__version__,
        'scipy': scipy.__version__,
        'spreg': spreg.__version__,
        'libpysal': libpysal.__version__,
    },
    'ordinaryLeastSquares': {
        'heteroskedastic': fit_ordinary_least_squares(scenes['hetero_41_300']),
        'collinearModerate': fit_ordinary_least_squares(scenes['collinear_moderate_17_200']),
        'collinearSevere': fit_ordinary_least_squares(scenes['collinear_severe_19_200']),
        'plain': fit_ordinary_least_squares(scenes['plain_2_400']),
        'studentT': [
            {'t': t, 'df': df, 'sf': float(stats.t.sf(t, df)), 'twoSided': float(2 * stats.t.sf(abs(t), df))}
            for t, df in T_CASES
        ],
    },
    'errorGM': {'lattice': fit_error_gm(frozen['lattice']), 'knn': fit_error_gm(frozen['knn'])},
    'errorGMHom': {
        'lattice': fit_error_gm_hom(frozen['lattice']),
        'knn': fit_error_gm_hom(frozen['knn']),
        'binary': fit_error_gm_hom(scenes['err_binary_324_9']),
    },
    'lagMaximumLikelihood': {
        name: fit_lag_maximum_likelihood(scenes[name])
        for name in ('lag_12_21', 'lag_binary_10_5', 'lag_6pred_10_7', 'knn_60_13', 'knn_90_29')
    },
    'errorMaximumLikelihood': {
        name: fit_error_maximum_likelihood(scenes[name])
        for name in ('err_lattice_400_31', 'err_knn_400_31', 'err_binary_324_9')
    },
    'diagnostics': {
        'lattice': fit_diagnostics(scenes['diagnostics_lattice_8_11']),
        'nearestNeighbor': fit_diagnostics(scenes['diagnostics_knn_80_7']),
    },
    'twoStage': fit_two_stage(scenes['lag_12_21'], 1, with_p_values=True),
    'asymmetricNearestNeighbor': {
        'scene_60_13': {'diagnostics': fit_diagnostics(scenes['knn_60_13']), 'twoStage': fit_two_stage(scenes['knn_60_13'], 1)},
        'scene_90_29': {'diagnostics': fit_diagnostics(scenes['knn_90_29']), 'twoStage': fit_two_stage(scenes['knn_90_29'], 1)},
    },
    'twoStageOrderTwo': {
        'lattice': fit_two_stage(scenes['lag_12_21'], 2),
        'scene_60_13': fit_two_stage(scenes['knn_60_13'], 2),
        'scene_90_29': fit_two_stage(scenes['knn_90_29'], 2),
    },
    'maskedDiagnostics': {
        'lattice': fit_masked_diagnostics(scenes['masked_lattice_8_11']),
        'knnDirected': fit_masked_diagnostics(scenes['masked_knn_60_13']),
    },
    'twoStageDiagnostics': {
        name: fit_two_stage_diagnostics(scenes[name])
        for name in ('lag_12_21', 'lag_binary_10_5', 'lag_binary_10_5_high', 'weak_80_7', 'knn_60_13', 'knn_90_29')
    },
    'twoStageIllConditioned': fit_two_stage_ill_conditioned(scenes['ill_conditioned_12_21']),
}
with open(OUTPUT, 'w') as file:
    json.dump(encode(fixture), file, separators=(',', ':'))
    file.write('\n')
print('wrote', OUTPUT)
