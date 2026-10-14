# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Reference classification-fit measures (ADCM, ADAM, GADF, TSS, class medians) from mapclassify.

Writes fixtures/distribution-mapclassify-fit.json next to this script, consumed (through
test/gpu-spatial-analysis/distribution-dynamics/mapclassify-fit-fixture.ts) by
gpu-classification-fit.spec.ts. Run from anywhere with
    <venv>/bin/python -I modules/experimental/test/oracles/distribution-mapclassify-fit.py
in a venv with numpy and mapclassify (versions are recorded in the fixture).

Scene: numpy `default_rng(42)`, 200 values rounded to float32 (the GPU reads the same numbers): 120
lognormal(2, 0.8), 60 normal(-3, 2) and 20 integers in 0..4 (ties). Classified by
Quantiles(k=5), FisherJenks(k=5) (the `naturalBreaks` case) and EqualInterval(k=4); `classes` are `Classifier.yb`, which the
GPU is fed, so the fit is compared independent of the classifier. `FisherJenks` is the exact optimum that
`NaturalBreaks` approximates with unseeded random k-means starts, so it keeps the fixture deterministic.

References (mapclassify 2.11): `adcm` is `Classifier.adcm` (sum of absolute deviations around the
class medians), `gadf` is `get_gadf()` (1 - adcm / adam), `tss` is `get_tss()` (sum of squared
deviations around the class means, per mapclassify's definition), `adam` is the sum of absolute
deviations around the overall median, `medians` are `np.median` of each class and `overallMedian`
is `np.median(y)`.

Tolerances (stored in `tolerances`): class medians are medians of float32 numbers (a value or the
midpoint of two), so the GPU float32 result is within one float32 rounding, a relative 2^-24 =
6e-8, of the float64 reference; the spec compares them to 1e-6 relative. The other measures keep the
default tolerance of the spec's `expectClose`.
"""

import json
import math
import os
import platform

import mapclassify
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'fixtures', 'distribution-mapclassify-fit.json')


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


generator = np.random.default_rng(42)
values = np.concatenate(
    [
        generator.lognormal(2, 0.8, 120).astype(np.float32),
        generator.normal(-3, 2, 60).astype(np.float32),
        generator.integers(0, 5, 20).astype(np.float32),
    ]
).astype(np.float64)

overall_median = float(np.median(values))
adam = float(np.abs(values - overall_median).sum())
cases = {}
for name, classifier in {
    'quantiles': mapclassify.Quantiles(values, k=5),
    'naturalBreaks': mapclassify.FisherJenks(values, k=5),
    'equalInterval': mapclassify.EqualInterval(values, k=4),
}.items():
    classes = np.asarray(classifier.yb)
    cases[name] = {
        'k': int(classifier.k),
        'classes': classes,
        'counts': [int((classes == index).sum()) for index in range(classifier.k)],
        'medians': [float(np.median(values[classes == index])) for index in range(classifier.k)],
        'adcm': float(classifier.adcm),
        'gadf': float(classifier.get_gadf()),
        'tss': float(classifier.get_tss()),
        'adam': adam,
        'overallMedian': overall_median,
    }

fixture = {
    'generator': 'modules/experimental/test/oracles/distribution-mapclassify-fit.py',
    'versions': {
        'python': platform.python_version(),
        'numpy': np.__version__,
        'mapclassify': mapclassify.__version__,
    },
    'tolerances': {'medians': {'absolute': 0, 'relative': 1e-6}},
    'reference': {'values': values, 'cases': cases},
}
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, 'w') as handle:
    json.dump(encode(fixture), handle, separators=(',', ':'))
    handle.write('\n')
print('wrote', OUTPUT)
