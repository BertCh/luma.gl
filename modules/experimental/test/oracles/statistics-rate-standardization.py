# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Reference Assuncao-Reis empirical Bayes standardization, Moran_Rate and Moran_Local_Rate from esda.

Writes fixtures/statistics-rate-standardization.json next to this script, consumed (through
test/gpu-spatial-analysis/spatial-autocorrelation/gpu-rate-standardization-reference.ts) by
gpu-rate-standardization.spec.ts, gpu-rate-standardization.node.spec.ts and the fixture module
gpu-rate-standardization-fixture.ts. Run from anywhere with
    <venv>/bin/python -I modules/experimental/test/oracles/statistics-rate-standardization.py
in a venv with numpy, esda and libpysal (versions are recorded in the fixture).

Scene: 60 areas on a line. Populations are exp(N(7, 1)) rounded (seed 11), areas 3 and 40 shrunk to
12 and 30 (strong shrinkage); events are Poisson(population * rate) with a bump at area 15 and a dip
at area 42. Binary neighbors i +- 1, i +- 2 among areas 0..58; area 59 is an island. Datasets:
- `full`: every area;
- `missing`: areas 5 (NaN events), 20 (zero population) and 33 (negative events) invalid, the
  neighbor graph rebuilt without them (`invalidRows` names the cause);
- `under`: under-dispersed identical-rate data (events = round(0.05 * population)), so the prior
  variance `a` is negative and `assuncao_rate` falls back to the unshrunk variance.

References:
- `<dataset>Z`: `esda.smoothing.assuncao_rate(events, populations)`, scattered to full row ids;
- `<dataset>Statistics`: `[n, b, s2, a, totalPopulation]`, the global quantities of that standardization;
- `<dataset>NegativeVarianceRows`: rows whose raw variance a + b / population is negative;
- `<dataset>Moran`: `esda.Moran_Rate(..., transformation='r')` I, EI, z_norm, z_rand, p_norm, p_rand;
- `<dataset>LocalI`: `esda.Moran_Local_Rate(..., transformation='r').Is`.

Tolerances (stored in `tolerances`): `node` is the double-precision TypeScript oracle against
esda, which differ only by summation order, so 1e-12 relative (double roundoff 1.1e-16 times a few
dozen terms with cancellation in `a`). The GPU values are float32 reads of the same integers:
`statistic` (Moran I) 1e-3 absolute + 1e-3 relative; `zScore` 2e-2 + 5e-3 relative because z is a
ratio of float32 sums over 60 terms; `aFraction` bounds `a`, a difference of two similar numbers,
by 1e-3 of s2 + 1e-3 of |a|.
"""

import json
import math
import os
import platform
import warnings

import esda
import libpysal
import numpy as np
from esda.smoothing import assuncao_rate
from libpysal.weights import W

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'fixtures', 'statistics-rate-standardization.json')
ROW_COUNT = 60
ISLAND = 59
# Invalid rows of the 'missing' dataset: NaN event, zero population, negative event.
INVALID = {5: ('events', float('nan')), 20: ('population', 0.0), 33: ('events', -4.0)}


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
population = np.round(np.exp(generator.normal(7.0, 1.0, ROW_COUNT))).astype(np.float64)
population[[3, 40]] = [12, 30]
rate = 0.02 * (1 + 1.5 * np.exp(-(((index - 15) / 5) ** 2)) - 0.6 * np.exp(-(((index - 42) / 4) ** 2)))
events = generator.poisson(population * rate).astype(np.float64)

# Under-dispersed data: identical rates, so s2 ~ 0 and the prior variance a is negative.
population_under = np.round(np.exp(generator.normal(6.0, 1.2, ROW_COUNT))).astype(np.float64)
population_under[[7, 21]] = [5, 8]
events_under = np.round(population_under * 0.05)

neighbors = {row: [other for other in (row - 2, row - 1, row + 1, row + 2) if 0 <= other <= 58] for row in range(59)}
neighbors[ISLAND] = []


def make_weights(keep):
    position = {row: slot for slot, row in enumerate(keep)}
    subset = {position[row]: [position[other] for other in neighbors[row] if other in position] for row in keep}
    return W(subset, {row: [1.0] * len(entries) for row, entries in subset.items()}, silence_warnings=True)


def scatter(keep, subset_values):
    full = np.full(ROW_COUNT, np.nan)
    full[list(keep)] = subset_values
    return full


def global_quantities(dataset_events, dataset_population):
    """[n, b, s2, a, total population] of the Assuncao-Reis standardization."""
    total_population = dataset_population.sum()
    pooled_rate = dataset_events.sum() / total_population
    observed_rate = dataset_events / dataset_population
    s2 = (dataset_population * (observed_rate - pooled_rate) ** 2).sum() / total_population
    a = s2 - pooled_rate / (total_population / len(dataset_events))
    return [float(len(dataset_events)), float(pooled_rate), float(s2), float(a), float(total_population)]


all_rows = list(range(ROW_COUNT))
valid_rows = [row for row in all_rows if row not in INVALID]
events_missing = events.copy()
population_missing = population.copy()
for row, (field, value) in INVALID.items():
    (events_missing if field == 'events' else population_missing)[row] = value

reference = {
    'events': events,
    'population': population,
    'island': ISLAND,
    'invalidRows': {str(row): field for row, (field, _) in INVALID.items()},
    'eventsMissing': events_missing,
    'populationMissing': population_missing,
    'eventsUnder': events_under,
    'populationUnder': population_under,
}
for name, dataset_events, dataset_population, keep in (
    ('full', events, population, all_rows),
    ('missing', events, population, valid_rows),
    ('under', events_under, population_under, all_rows),
):
    reference[name + 'Z'] = scatter(keep, assuncao_rate(dataset_events[keep], dataset_population[keep]))
    reference[name + 'Statistics'] = global_quantities(dataset_events[keep], dataset_population[keep])
# Count of rows whose raw variance a + b / population is negative (the esda fallback), per dataset.
for name, dataset_events, dataset_population in (
    ('full', events, population),
    ('under', events_under, population_under),
):
    _, pooled_rate, _, a, _ = global_quantities(dataset_events, dataset_population)
    reference[name + 'NegativeVarianceRows'] = int(np.sum(a + pooled_rate / dataset_population < 0))

for name, keep in (('full', all_rows), ('missing', valid_rows)):
    moran = esda.Moran_Rate(events[keep], population[keep], make_weights(keep), transformation='r', permutations=0)
    local = esda.Moran_Local_Rate(events[keep], population[keep], make_weights(keep), transformation='r', permutations=0)
    reference[name + 'Moran'] = {
        'I': moran.I,
        'EI': moran.EI,
        'zNorm': moran.z_norm,
        'zRand': moran.z_rand,
        'pNorm': moran.p_norm,
        'pRand': moran.p_rand,
    }
    reference[name + 'LocalI'] = scatter(keep, local.Is)
assert reference['underNegativeVarianceRows'] > 0, 'under-dispersed dataset must hit the esda fallback'
assert reference['underStatistics'][3] < 0, 'a must be negative in the under-dispersed dataset'

fixture = {
    'generator': 'test/oracles/statistics-rate-standardization.py',
    'versions': {
        'python': platform.python_version(),
        'numpy': np.__version__,
        'esda': esda.__version__,
        'libpysal': libpysal.__version__,
    },
    'tolerances': {
        'node': {'relative': 1e-12},
        'statistic': {'absolute': 1e-3, 'relative': 1e-3},
        'zScore': {'absolute': 2e-2, 'relative': 5e-3},
        'aFraction': {'ofS2': 1e-3, 'ofA': 1e-3},
        'derivation': 'node: double roundoff 1.1e-16 times a few dozen summed terms with cancellation in a; '
        'GPU: float32 sums of at most 60 terms, z is a ratio of two of them, a is a difference of similar sums',
    },
    'reference': reference,
}
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, 'w') as file:
    json.dump(encode(fixture), file, separators=(',', ':'))
    file.write('\n')
print('wrote', OUTPUT)
