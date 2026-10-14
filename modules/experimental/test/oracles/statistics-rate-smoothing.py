# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Reference empirical Bayes rate smoothing from esda: low-dispersion global and spatial EB, and
Spatial_Rate / Spatial_Empirical_Bayes on a 6x6 queen lattice.

Writes fixtures/statistics-rate-smoothing.json next to this script, consumed (through
test/gpu-spatial-analysis/rate-smoothing/eb-low-dispersion-fixture.ts and esda-spatial-eb-fixture.ts) by
gpu-empirical-bayes-rates.spec.ts / .node.spec.ts and gpu-spatial-empirical-bayes-rates.spec.ts. Run
from anywhere with
    <venv>/bin/python -I modules/experimental/test/oracles/statistics-rate-smoothing.py
in a venv with numpy, esda and libpysal (versions are recorded in the fixture).

Scenes:
- `lowDispersion`: 12 areas, populations uniform(800, 1200) rounded and counts at one common rate
  (seed 7), so the estimated prior variance `a` is negative.
  `unclamped` is `esda.smoothing.Empirical_Bayes(e, b).r` (does not clamp `a`, so area 0 comes out
  negative); `clamped` is Marshall (1991) / GeoDa, `a = max(a, 0)` in `w = a / (a + m / b)`, computed
  in numpy; `priorVariance` is `a` and `pooledRate` is `m`.
- `spatialLowDispersion`: the same construction on a 6x6 queen lattice (seed 11, rate 0.01 plus
  uniform(-0.5, 0.5) counts, at least 1): `smoothed` is `Spatial_Empirical_Bayes(e, b, w).r`, which
  clamps negative local prior variances to 0 itself.
- `lattice`: `lat2W(6, 6, rook=False)` with populations `default_rng(42).integers(200, 5000)` and
  fixed event counts (areas 3 and 20 have zero events; the event generator was not recoverable, so
  the literal counts are the scene input): `spatialRate` is `Spatial_Rate(e, b, w).r` and `smoothed`
  is `Spatial_Empirical_Bayes(e, b, w).r`. `offsets`/`neighbors` are the CSR of the sorted neighbors.

Tolerances (stored in `tolerances`): the specs compare the float32 GPU result to these double
references: `spatialRate` to 1e-5 relative (float32 sums of at most 9 window terms, roundoff 6e-8
amplified by one division), `smoothed` to 2e-4 relative + 1e-6 absolute (the shrinkage weight
a / (a + m / b) contains a difference of similar float32 sums, so roundoff is amplified by the
condition of that difference); the node spec compares the CPU oracle at 12 decimals.
"""

import json
import math
import os
import platform
import warnings

import esda
import libpysal
import numpy as np
from esda.smoothing import Empirical_Bayes, Spatial_Empirical_Bayes, Spatial_Rate
from libpysal.weights import lat2W

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'fixtures', 'statistics-rate-smoothing.json')

# Event counts of the lattice scene (population = default_rng(42).integers(200, 5000, 36)).
LATTICE_EVENTS = [
    21, 106, 105, 0, 88, 156, 19, 41, 24, 4, 28, 141, 115, 149, 59, 71, 57, 9,
    40, 51, 0, 56, 21, 158, 116, 52, 72, 137, 52, 35, 68, 12, 7, 15, 145, 14,
]  # fmt: skip


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


def make_lattice():
    weights = lat2W(6, 6, rook=False)
    weights.id_order = list(range(36))
    offsets = np.concatenate([[0], np.cumsum([len(weights.neighbors[row]) for row in range(36)])])
    neighbors = [other for row in range(36) for other in sorted(weights.neighbors[row])]
    return weights, offsets, neighbors


# Global EB on 12 areas with a common rate.
generator = np.random.default_rng(7)
count = 12
populations = np.round(generator.uniform(800, 1200, count))
events = np.clip(np.round(populations * 0.01 + generator.uniform(-0.6, 0.6, count)), 1, None)
rates = events / populations
pooled_rate = events.sum() / populations.sum()
prior_variance = (populations * (rates - pooled_rate) ** 2).sum() / populations.sum() - pooled_rate / populations.mean()
unclamped = Empirical_Bayes(events, populations).r.ravel()
clamped_variance = max(prior_variance, 0.0)
shrinkage = clamped_variance / (clamped_variance + pooled_rate / populations)
clamped = shrinkage * rates + (1 - shrinkage) * pooled_rate
low_dispersion = {
    'events': events,
    'populations': populations,
    'priorVariance': prior_variance,
    'pooledRate': pooled_rate,
    'unclamped': unclamped,
    'clamped': clamped,
}

# Spatial EB on a 6x6 lattice with common-rate counts, so local prior variances go negative in places.
generator = np.random.default_rng(11)
lattice_weights, lattice_offsets, lattice_neighbors = make_lattice()
lattice_populations = np.round(generator.uniform(800, 1200, 36))
lattice_events = np.clip(np.round(lattice_populations * 0.01 + generator.uniform(-0.5, 0.5, 36)), 1, None)
spatial_low_dispersion = {
    'events': lattice_events,
    'populations': lattice_populations,
    'offsets': lattice_offsets,
    'neighbors': lattice_neighbors,
    'smoothed': Spatial_Empirical_Bayes(lattice_events, lattice_populations, lattice_weights).r.ravel(),
}

# Spatial_Rate / Spatial_Empirical_Bayes on the heterogeneous lattice scene.
scene_weights, scene_offsets, scene_neighbors = make_lattice()
scene_populations = np.random.default_rng(42).integers(200, 5000, 36).astype(np.float64)
scene_events = np.array(LATTICE_EVENTS, dtype=np.float64)
lattice = {
    'events': scene_events,
    'populations': scene_populations,
    'offsets': scene_offsets,
    'neighbors': scene_neighbors,
    'spatialRate': Spatial_Rate(scene_events, scene_populations, scene_weights).r.ravel(),
    'smoothed': Spatial_Empirical_Bayes(scene_events, scene_populations, scene_weights).r.ravel(),
}

fixture = {
    'generator': 'test/oracles/statistics-rate-smoothing.py',
    'versions': {
        'python': platform.python_version(),
        'numpy': np.__version__,
        'esda': esda.__version__,
        'libpysal': libpysal.__version__,
    },
    'tolerances': {
        'spatialRate': {'absolute': 0, 'relative': 1e-5},
        'smoothed': {'absolute': 1e-6, 'relative': 2e-4},
        'node': {'decimals': 12},
        'derivation': 'float32 unit roundoff 6e-8; Spatial_Rate sums at most 9 window terms and divides once; '
        'the EB weight a / (a + m / b) contains a difference of similar sums (cancellation); '
        'the node oracle is double precision against esda',
    },
    'reference': {
        'lowDispersion': low_dispersion,
        'spatialLowDispersion': spatial_low_dispersion,
        'lattice': lattice,
    },
}
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, 'w') as file:
    json.dump(encode(fixture), file, separators=(',', ':'))
    file.write('\n')
print('wrote', OUTPUT)
