# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Reference weights traces, Gram traces, log-determinants, multiplier sums and spreg lag-impact
multipliers for `GPUSpatialWeightsTraces` and `GPUSpatialLagImpacts` (numpy / scipy / spreg).

Writes fixtures/statistics-spatial-weights-traces.json next to this script, consumed (through
test/gpu-spatial-analysis/spatial-regression/spatial-weights-traces-reference.ts) by
spatial-weights-traces.spec.ts, spatial-weights-traces.node.spec.ts and spatial-lag-impacts.spec.ts.
Run from anywhere with
    <venv>/bin/python -I modules/experimental/test/oracles/statistics-spatial-weights-traces.py
in a venv with numpy, scipy, libpysal and spreg (versions are recorded in the fixture).

Scenes: read from fixtures/statistics-spatial-weights-traces-scenes.json, which
test/gpu-spatial-analysis/spatial-regression/spatial-weights-traces-scenes.node.spec.ts checks against
the TypeScript generators (float32-rounded weights, predictors and response):
- `lattice`: `createLagScene(12, 21, 0.5, true)` (row-standardized rook lattice, n = 144; bipartite, so
  odd traces are exactly zero);
- `rawLattice`: `createLagScene(10, 8, 0.12, false)` (binary rook lattice, spectral radius 3.84);
- `knn60`: `createNearestNeighborScene(60, 3, 13, 0.6, false)` (directed row-standardized kNN);
- `knn90`: `createNearestNeighborScene(90, 3, 29, 0.6, false)`.

Per scene, with W the dense float64 matrix of the float32 weights:
- `powerTraces[q-1] = trace(W^q)` and `powerSums[q-1] = 1' W^q 1` for q = 1..12;
- `gramTraces = [tr(W'W), tr(W'WW), tr((W'W)^2), sum_i c_i^2]` (c_i the column sums of squares);
- `logDeterminant`, `multiplierTrace` and `multiplierTotal` at each rho in {-0.6, -0.3, 0.2, 0.5, 0.9}
  from `numpy.linalg.slogdet(I - rho W)` and `numpy.linalg.inv(I - rho W)` (trace and 1' inv 1);
- `spectralRadius`: max |eigenvalue|;
- `spreg` (the three row-standardized scenes): `spreg.GM_Lag(y, X, w, w_lags=1)` rho and betas, the
  multipliers `[direct, indirect, total]` of `spreg.sputils.spmultiplier(w, rho, method)` for `simple` and
  `full` (`full.impacts` are the multipliers times each beta), and `exact`, the general-W definitions
  `[tr(inv)/n, 1'inv1/n - tr(inv)/n, 1'inv1/n]`. spreg `'power'` is not pinned: it stops after the first
  term whose trace vanishes, which is wrong on the bipartite lattice (1.0679 vs 1.0802).

Tolerances (stored in `tolerances`): every quantity is a float64 dense linear-algebra result of at most
144 x 144 matrices. Matrix powers up to q = 12 and sums of at most 144 terms accumulate at most
12 * 144 * 2^-53 ~ 2e-13 relative error; slogdet and inv at rho = 0.9 amplify by the condition number
(at most 1 / (1 - 0.9) = 10 for the row-standardized scenes; the raw lattice at rho = 0.9 has
rho * 3.84 > 1 but is still not singular, and its cancellation is bounded by 1e-9). The stored
`regeneration` bound of 1e-9 relative (1e-12 absolute) is therefore the largest difference expected
when the fixture is regenerated on another BLAS or LAPACK; the specs compare the GPU (float32) results
with their own, much looser, bounds.
"""

import json
import math
import os
import platform
import warnings

import libpysal
import numpy as np
import scipy
import spreg
from libpysal.weights import W
from spreg import GM_Lag
from spreg.sputils import spmultiplier

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
SCENES = os.path.join(HERE, 'fixtures', 'statistics-spatial-weights-traces-scenes.json')
OUTPUT = os.path.join(HERE, 'fixtures', 'statistics-spatial-weights-traces.json')
MAXIMUM_POWER = 12
RHOS = [-0.6, -0.3, 0.2, 0.5, 0.9]
SPREG_SCENES = ['lattice', 'knn60', 'knn90']


def encode(value):
    """JSON-safe copy: NaN -> null, infinities and -0 -> strings (decoded by test/oracles/oracle-fixture.ts)."""
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
        if value == 0 and math.copysign(1.0, value) < 0:
            return '-0'
        return value
    return value


def dense(scene):
    rows = len(scene['offsets']) - 1
    matrix = np.zeros((rows, rows))
    for row in range(rows):
        for slot in range(scene['offsets'][row], scene['offsets'][row + 1]):
            matrix[row, scene['neighbors'][slot]] = scene['weights'][slot]
    return matrix


def describe(name, scene):
    matrix = dense(scene)
    rows = matrix.shape[0]
    ones = np.ones(rows)
    entry = {}
    power = np.eye(rows)
    traces, sums = [], []
    for _ in range(MAXIMUM_POWER):
        power = power @ matrix
        traces.append(float(np.trace(power)))
        sums.append(float(ones @ power @ ones))
    entry['powerTraces'] = traces
    entry['powerSums'] = sums
    gram = matrix.T @ matrix
    entry['gramTraces'] = [
        float(np.trace(gram)),
        float(np.trace(gram @ matrix)),
        float(np.trace(gram @ gram)),
        float(np.sum((matrix**2).sum(axis=0) ** 2)),
    ]
    identity = np.eye(rows)
    entry['logDeterminant'] = {str(rho): float(np.linalg.slogdet(identity - rho * matrix)[1]) for rho in RHOS}
    entry['multiplierTrace'] = {str(rho): float(np.trace(np.linalg.inv(identity - rho * matrix))) for rho in RHOS}
    entry['multiplierTotal'] = {str(rho): float(ones @ np.linalg.inv(identity - rho * matrix) @ ones) for rho in RHOS}
    entry['spectralRadius'] = float(np.max(np.abs(np.linalg.eigvals(matrix))))
    if name in SPREG_SCENES:
        ids = list(range(rows))
        neighbors = {row: [int(j) for j in scene['neighbors'][scene['offsets'][row] : scene['offsets'][row + 1]]] for row in ids}
        weights = {row: [float(v) for v in scene['weights'][scene['offsets'][row] : scene['offsets'][row + 1]]] for row in ids}
        spatial_weights = W(neighbors, weights, id_order=ids, silence_warnings=True)
        spatial_weights.transform = 'O'
        predictor_count = scene['k']
        predictors = np.array(scene['predictors'], dtype=float).reshape(rows, predictor_count)
        response = np.array(scene['response'], dtype=float).reshape(rows, 1)
        regression = GM_Lag(response, predictors, w=spatial_weights, w_lags=1)
        rho = float(regression.betas[-1][0])
        betas = [float(beta[0]) for beta in regression.betas]
        impacts = {'rho': rho, 'betas': betas}
        multipliers = spmultiplier(spatial_weights, rho, method='full')
        direct, total, indirect = float(multipliers['adi']), float(multipliers['ati']), float(multipliers['aii'])
        impacts['full'] = {
            'multipliers': [direct, indirect, total],
            'pow': multipliers['pow'],
            'impacts': [[direct * beta, indirect * beta, total * beta] for beta in betas[1 : 1 + predictor_count]],
        }
        # exact general-W total (does not assume row standardization)
        inverse = np.linalg.inv(identity - rho * matrix)
        impacts['exact'] = {
            'multipliers': [
                float(np.trace(inverse) / rows),
                float(ones @ inverse @ ones / rows - np.trace(inverse) / rows),
                float(ones @ inverse @ ones / rows),
            ]
        }
        simple = spmultiplier(spatial_weights, rho, method='simple')
        impacts['simple'] = [float(simple['adi']), float(simple['aii']), float(simple['ati'])]
        entry['spreg'] = impacts
    return entry


with open(SCENES) as file:
    scenes = json.load(file)['scenes']
reference = {name: describe(name, scene) for name, scene in scenes.items()}

fixture = {
    'generator': 'test/oracles/statistics-spatial-weights-traces.py',
    'versions': {
        'python': platform.python_version(),
        'numpy': np.__version__,
        'scipy': scipy.__version__,
        'libpysal': libpysal.__version__,
        'spreg': spreg.__version__,
    },
    'tolerances': {
        'regeneration': {'absolute': 1e-12, 'relative': 1e-9},
        'derivation': 'float64 dense algebra of at most 144 x 144; 12 matrix powers of 144-term sums give '
        '~2e-13 relative, slogdet and inv at rho = 0.9 amplify by a condition number of at most 10 (raw lattice '
        'cancellation bounded by 1e-9); the bound is what regeneration on another BLAS or LAPACK may change',
    },
    'maximumPower': MAXIMUM_POWER,
    'reference': reference,
}
with open(OUTPUT, 'w') as file:
    json.dump(encode(fixture), file, separators=(',', ':'))
    file.write('\n')
print('wrote', OUTPUT)
