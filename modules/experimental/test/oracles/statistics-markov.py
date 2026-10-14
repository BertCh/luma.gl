# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Reference Markov-chain diagnostics from giddy (Spatial_Markov homogeneity, steady_state,
LISA_Markov) and the esda local Moran quadrants they are built on.

Writes fixtures/statistics-markov.json next to this script, consumed (through
test/gpu-spatial-analysis/distribution-dynamics/markov-giddy-fixture.ts) by
gpu-distribution-dynamics.spec.ts and transition-diagnostics.node.spec.ts. Run from anywhere with
    <venv>/bin/python -I modules/experimental/test/oracles/statistics-markov.py
in a venv with numpy, scipy, giddy, esda and libpysal (versions are recorded in the fixture).

Scenes:
- `a`: 60 rows x 5 periods of 3 classes (numpy `default_rng(seed)`, the first seed 0..99 whose
  conditional transition matrices have no empty from-state), with 3 conditions. Counts, giddy's
  `homogeneity` (Q, LR and their p-values, dof) and `steady_state` of the pooled chain and of each
  conditional chain. Arrays `classes` and `conditions` are period-major (period t, row i at
  t * rows + i); `steadyStates` holds the pooled chain first, then one chain per condition.
- `bGood`, `bAll`: hand-built transition tensors with empty rows, a reducible chain and a dead end.
  The steady state follows the documented rule: states with no transition in or out are dropped
  (value 0); the result is NaN unless every kept state has an outgoing transition and the kept chain is
  one communicating class; otherwise giddy `steady_state`. `homogeneity` uses the documented rule
  where a from-state without transitions adds 0 degrees of freedom (dof = giddy dof - number of
  empty pooled rows, p-values at that dof); `giddyHomogeneity` is giddy's own, which adds 1 per such row.
- `c1`, `c2`: a 6 x 6 rook grid plus one island row, 4 periods, values from sin/cos trends and
  N(0, 1.2) noise (seed 11), rounded to 3 decimals. `c1` has row 7 NaN in every period; `c2` has row
  20 NaN in period 2 only. States are esda `Moran_Local(..., transformation 'r', permutations=0).q`
  per period on that period's finite, non-island rows (-1 = no state); counts are the transition
  counts over states 1..4 (the 5 x 5 table has row/column 0 unused); `ignored` is the number of
  row-period pairs with a missing state. For `c1` the counts are asserted equal to giddy
  `LISA_Markov` over the complete rows, and `steadyState` / `giddyP` are its chain.

Tolerances (stored in `tolerances`): all counts and states are exact integers. Q, LR and the steady
states are float64 sums and eigen-solves over at most 75 cells and 5 states, compared to 1e-9
absolute + 1e-9 relative (the specs' bound; the TypeScript side computes the same float64 sums);
chi-square p-values are compared to 1e-6 relative because the TypeScript side evaluates the
chi-square survival function with its own incomplete-gamma series instead of scipy's.
"""

import json
import math
import os
import platform

import esda
import giddy
import libpysal
import numpy as np
import scipy
from esda.moran import Moran_Local
from giddy.ergodic import steady_state
from giddy.markov import LISA_Markov, Markov, homogeneity
from libpysal.weights import W
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'fixtures', 'statistics-markov.json')


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


# ---------- a: pooled and conditional chains from raw class arrays ----------
def make_a_scene(seed, rows=60, periods=5, class_count=3, condition_count=3):
    generator = np.random.default_rng(seed)
    classes = np.zeros((rows, periods), int)
    classes[:, 0] = generator.integers(0, class_count, rows)
    for period in range(1, periods):
        stay = generator.random(rows) < 0.55
        classes[:, period] = np.where(stay, classes[:, period - 1], generator.integers(0, class_count, rows))
    conditions = generator.integers(0, condition_count, (rows, periods))
    # Make the conditional matrices differ: condition 2 pushes upward.
    for period in range(1, periods):
        up = (conditions[:, period - 1] == 2) & (generator.random(rows) < 0.5)
        classes[:, period] = np.where(up, np.minimum(classes[:, period - 1] + 1, class_count - 1), classes[:, period])
    return classes, conditions


def count_transitions(classes, conditions, class_count, condition_count):
    rows, periods = classes.shape
    counts = np.zeros((condition_count, class_count, class_count), int)
    for period in range(periods - 1):
        for row in range(rows):
            counts[conditions[row, period], classes[row, period], classes[row, period + 1]] += 1
    return counts


def giddy_homogeneity(counts):
    test = homogeneity(counts.astype(float))
    return {
        'Q': float(test.Q),
        'Qp': float(test.Q_p_value),
        'LR': float(test.LR),
        'LRp': float(test.LR_p_value),
        'dof': int(test.dof),
    }


class_count = 3
condition_count = 3
for seed in range(100):
    classes, conditions = make_a_scene(seed)
    counts = count_transitions(classes, conditions, class_count, condition_count)
    if all((counts[condition].sum(axis=1) > 0).all() for condition in range(condition_count)):
        break
pooled = counts.sum(axis=0)
pooled_probabilities = pooled / pooled.sum(axis=1, keepdims=True)
steady = [steady_state(pooled_probabilities).tolist()]
for condition in range(condition_count):
    steady.append(steady_state(counts[condition] / counts[condition].sum(axis=1, keepdims=True)).tolist())
totals = counts.sum(axis=2).reshape(-1)
SPARSITY_THRESHOLD = 25
# Cross-check against giddy Markov on the same raw classes (pooled chain).
markov = Markov(classes, summary=False)
assert np.allclose(markov.p, pooled_probabilities)
assert np.allclose(markov.steady_state, steady[0])
scene_a = {
    'seed': seed,
    'classes': classes.T.reshape(-1),
    'conditions': conditions.T.reshape(-1),
    'counts': counts.reshape(-1),
    'homogeneity': giddy_homogeneity(counts),
    'steadyStates': [value for chain in steady for value in chain],
    'sparsityThreshold': SPARSITY_THRESHOLD,
    'sparsity': {
        'sparse': int(((totals > 0) & (totals < SPARSITY_THRESHOLD)).sum()),
        'empty': int((totals == 0).sum()),
    },
}


# ---------- b: hand-built tensors ----------
def get_steady_state(matrix):
    """Documented rule: NaN unless the kept chain is one communicating class; else giddy steady_state."""
    state_count = matrix.shape[0]
    outgoing = matrix.sum(axis=1)
    incoming = matrix.sum(axis=0)
    active = [state for state in range(state_count) if outgoing[state] > 0 or incoming[state] > 0]
    if not active:
        return [float('nan')] * state_count, 'empty'
    sub = matrix[np.ix_(active, active)]
    if any(outgoing[state] == 0 for state in active):
        # giddy raises ValueError (rows of 0s) on the unfilled matrix.
        divisor = np.where(sub.sum(axis=1, keepdims=True) == 0, 1, sub.sum(axis=1, keepdims=True))
        try:
            steady_state(sub / divisor)
            raise AssertionError('giddy did not raise')
        except ValueError:
            pass
        return [float('nan')] * state_count, 'dead-end'
    result = steady_state(sub / sub.sum(axis=1, keepdims=True))
    if np.ndim(result) == 2:
        return [float('nan')] * state_count, 'reducible(%d)' % len(result)
    full = [0.0] * state_count
    for state, share in zip(active, result):
        full[state] = float(share)
    return full, 'ok'


def make_b_scene(conditional_counts):
    tensor = np.array(conditional_counts, dtype=int)
    condition_total = tensor.shape[0]
    diagnostics = giddy_homogeneity(tensor)
    pooled_tensor = tensor.sum(axis=0)
    empty_pooled = int((pooled_tensor.sum(axis=1) == 0).sum())
    rows = list(get_steady_state(pooled_tensor)[0])
    for condition in range(condition_total):
        rows += get_steady_state(tensor[condition])[0]
    totals = tensor.sum(axis=2).reshape(-1)
    corrected = dict(diagnostics)
    corrected['dof'] = diagnostics['dof'] - empty_pooled
    # p-values at the corrected dof (giddy's use the dof including the empty-row +1).
    corrected['Qp'] = float(stats.chi2.sf(diagnostics['Q'], corrected['dof']))
    corrected['LRp'] = float(stats.chi2.sf(diagnostics['LR'], corrected['dof']))
    return {
        'counts': tensor.reshape(-1),
        'homogeneity': corrected,
        'giddyHomogeneity': diagnostics,
        'steadyStates': rows,
        'sparsity': {'sparse': int(((totals > 0) & (totals < 5)).sum()), 'empty': int((totals == 0).sum())},
    }


GOOD = [
    [[10, 3, 2, 0], [4, 12, 1, 0], [2, 2, 9, 0], [0, 0, 0, 0]],
    [[6, 5, 0, 0], [3, 8, 2, 0], [1, 4, 7, 0], [0, 0, 0, 0]],
    [[0, 4, 0, 0], [5, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
]
BAD = [
    [[5, 0, 0, 0], [0, 7, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],  # reducible: two absorbing states
    [[4, 3, 0, 0], [2, 5, 3, 0], [0, 0, 0, 0], [0, 0, 0, 0]],  # state 2 entered, never left
]
scene_b_good = make_b_scene(GOOD)
scene_b_all = make_b_scene(GOOD + BAD)

# ---------- c: LISA Markov with a NaN row and an island ----------
SIDE = 6
GRID_ROWS = SIDE * SIDE
ROWS = GRID_ROWS + 1  # the last row is an island
PERIODS = 4
NAN_EVERYWHERE_ROW = 7
NAN_ONCE_ROW, NAN_ONCE_PERIOD = 20, 2


def get_rook_neighbors(row):
    grid_row, grid_column = divmod(row, SIDE)
    found = []
    if grid_row > 0:
        found.append(row - SIDE)
    if grid_row < SIDE - 1:
        found.append(row + SIDE)
    if grid_column > 0:
        found.append(row - 1)
    if grid_column < SIDE - 1:
        found.append(row + 1)
    return sorted(found)


rook = {row: get_rook_neighbors(row) for row in range(GRID_ROWS)}
rook[GRID_ROWS] = []
generator = np.random.default_rng(11)
coordinates = np.array([divmod(row, SIDE) for row in range(GRID_ROWS)], float)
values = np.zeros((PERIODS, ROWS))
for period in range(PERIODS):
    trend = np.sin(coordinates[:, 0] * 0.9 + period * 0.7) + np.cos(coordinates[:, 1] * 0.8 - period * 0.5)
    values[period, :GRID_ROWS] = np.round(trend * 3 + generator.normal(0, 1.2, GRID_ROWS) + 10, 3)
    values[period, GRID_ROWS] = 40 + period  # island value: must not enter the mean


def make_weights(keep):
    position = {row: slot for slot, row in enumerate(keep)}
    neighbors = {position[row]: [position[other] for other in rook[row] if other in position] for row in keep}
    weights = W(neighbors, {slot: [1.0] * len(entries) for slot, entries in neighbors.items()})
    weights.transform = 'r'
    return weights, position, neighbors


def get_states(period_values):
    """Per period: Moran_Local (row-standardized, no permutations) on that period's finite, non-island rows."""
    states = np.full((PERIODS, ROWS), -1)
    for period in range(PERIODS):
        keep = [row for row in range(ROWS) if np.isfinite(period_values[period, row]) and row != GRID_ROWS]
        weights, position, neighbors = make_weights(keep)
        # A row left without neighbors would be an island in this period.
        assert all(len(entries) > 0 for entries in neighbors.values())
        local = Moran_Local(period_values[period][keep], weights, permutations=0, geoda_quads=False)
        for row in keep:
            states[period, row] = local.q[position[row]]
    return states


def count_state_transitions(states):
    counts = np.zeros((5, 5), int)
    ignored = 0
    for period in range(PERIODS - 1):
        for row in range(ROWS):
            before, after = states[period, row], states[period + 1, row]
            if before < 0 or after < 0:
                ignored += 1
            else:
                counts[before, after] += 1
    return counts, ignored


values_one = values.copy()
values_one[:, NAN_EVERYWHERE_ROW] = np.nan
states_one = get_states(values_one)
counts_one, ignored_one = count_state_transitions(states_one)
complete = [row for row in range(GRID_ROWS) if row != NAN_EVERYWHERE_ROW]
complete_weights, _, _ = make_weights(complete)
lisa = LISA_Markov(values_one[:, complete].T, complete_weights, permutations=0)
assert np.array_equal(lisa.transitions, counts_one[1:, 1:])
assert counts_one[0].sum() == 0 and counts_one[:, 0].sum() == 0
lisa_steady = steady_state(lisa.p)
assert np.ndim(lisa_steady) == 1
scene_c1 = {
    'values': values_one.reshape(-1),
    'counts': counts_one.reshape(-1),
    'ignored': ignored_one,
    'states': states_one.reshape(-1),
    'steadyState': [0.0] + [float(share) for share in lisa_steady],
    'giddyP': lisa.p.reshape(-1),
}

values_two = values.copy()
values_two[NAN_ONCE_PERIOD, NAN_ONCE_ROW] = np.nan
states_two = get_states(values_two)
counts_two, ignored_two = count_state_transitions(states_two)
scene_c2 = {
    'values': values_two.reshape(-1),
    'counts': counts_two.reshape(-1),
    'ignored': ignored_two,
    'states': states_two.reshape(-1),
}

fixture = {
    'generator': 'modules/experimental/test/oracles/statistics-markov.py',
    'versions': {
        'python': platform.python_version(),
        'numpy': np.__version__,
        'scipy': scipy.__version__,
        'giddy': giddy.__version__,
        'esda': esda.__version__,
        'libpysal': libpysal.__version__,
    },
    'tolerances': {
        'homogeneityStatistic': {'absolute': 1e-9, 'relative': 1e-9},
        'pValue': {'absolute': 0, 'relative': 1e-6},
        'steadyState': {'absolute': 1e-9, 'relative': 1e-9},
    },
    'reference': {
        'a': scene_a,
        'bGood': scene_b_good,
        'bAll': scene_b_all,
        'c1': scene_c1,
        'c2': scene_c2,
    },
}
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, 'w') as handle:
    json.dump(encode(fixture), handle, separators=(',', ':'))
    handle.write('\n')
print('wrote', OUTPUT, 'scene a seed', seed)
