# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""SciPy reference for the complete scalar GPUChangeDetection output contract.

Run <reference-venv>/bin/python -I this-file. Decode the full 16x256x256 Sentinel
NDVI stack, select the fixed lattice rows 0::8 / columns 0::4 (2048 cells), and
compare every scalar for all 16 dates. Zero bytes are NaN; decoded inputs are
rounded to float32 before float64 references. Cell-major layout, a cell mask,
first/last or selected two-slice comparisons, Welch split at ordinal slice 8.
Time is the ORIGINAL ordinal index, not days.bin and not the compressed index
of valid observations; this matches the public API. No claim of slope/year.

Sen slope: scipy.stats.theilslopes. Welch: scipy.stats.ttest_ind(equal_var=False)
with after minus before and scipy's df; fewer than 2 samples per group or zero
standard error -> NaN. Difference/log-ratio/percent are NumPy ufuncs.
Mann-Kendall is a composite reference: scipy.stats.kendalltau(variant='b',
method='asymptotic') supplies tau and uncorrected two-sided p. Recover integer
S using the tau-b denominator; norm.isf(p/2) supplies the uncorrected Z, then
apply the published continuity correction Z *= (|S|-1)/|S|. This is not SciPy's
uncorrected Kendall p-value presented as Mann-Kendall. All ties -> S=Z=0,p=1;
fewer than two valid values -> S=0,Z=p=NaN. Significance p<alpha, 1 up / 2 down.
Definition: Hirsch, Slack & Smith (1982), Water Resources Research 18, 107-121,
https://doi.org/10.1029/WR018i001p00107. Also SciPy kendalltau documentation:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html
Profile S: atol=1e-6*output scale, rtol=1e-4, integer outputs and NaN masks exact.
"""
import hashlib
import json
import math
import platform
import warnings
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import kendalltau, norm, theilslopes, ttest_ind

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DATA = ROOT / 'examples/deck/spatial-analysis-showcase/public/data/ndvi-timeseries'


def encode(value):
    if isinstance(value, dict):
        return {key: encode(entry) for key, entry in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [encode(entry) for entry in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return 'Infinity' if value > 0 else '-Infinity'
        return float(value)
    return value


def make_case(name, slices, mask, before, after):
    slices = np.asarray(slices, dtype=np.float32).astype(float)
    cell_count, slice_count = slices.shape
    split = 8
    alpha, epsilon = float(np.float32(.05)), float(np.float32(1e-6))
    names = ['difference', 'logRatio', 'percentChange', 'tStatistic', 'tDegreesOfFreedom',
             'tPValue', 'senSlope', 'mannKendallZ', 'mannKendallP']
    expected = {name: np.full(cell_count, np.nan) for name in names}
    for output_name in ['mannKendallS', 'significance', 'welchSignificance']:
        expected[output_name] = np.zeros(cell_count, dtype=int)
    for cell, values in enumerate(slices):
        if not mask[cell]:
            continue
        first, last = values[before], values[after]
        expected['difference'][cell] = last - first
        if first > -epsilon and last > -epsilon:
            expected['logRatio'][cell] = np.log((last + epsilon) / (first + epsilon))
        if first != 0:
            expected['percentChange'][cell] = 100 * (last - first) / abs(first)
        valid = np.flatnonzero(np.isfinite(values))
        observed = values[valid]
        if len(valid) >= 2:
            expected['senSlope'][cell] = theilslopes(observed, valid).slope
            _, multiplicities = np.unique(observed, return_counts=True)
            pairs = len(valid) * (len(valid) - 1) / 2
            tied_pairs = np.sum(multiplicities * (multiplicities - 1) / 2)
            if tied_pairs == pairs:
                score, statistic, probability = 0, 0., 1.
            else:
                result = kendalltau(valid, observed, variant='b', method='exact' if len(valid) == 2 else 'asymptotic')
                score = int(round(result.statistic * np.sqrt(pairs * (pairs - tied_pairs))))
                statistic = 0. if score == 0 else np.sign(score) * norm.isf(result.pvalue / 2) * (abs(score) - 1) / abs(score)
                probability = 2 * norm.sf(abs(statistic))
            expected['mannKendallS'][cell] = score
            expected['mannKendallZ'][cell] = statistic
            expected['mannKendallP'][cell] = probability
            expected['significance'][cell] = (1 if score > 0 else 2) if probability < alpha else 0
        first_group = values[:split][np.isfinite(values[:split])]
        last_group = values[split:][np.isfinite(values[split:])]
        if len(first_group) >= 2 and len(last_group) >= 2 and (np.var(first_group) + np.var(last_group)) > 0:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                result = ttest_ind(last_group, first_group, equal_var=False)
            expected['tStatistic'][cell] = result.statistic
            expected['tDegreesOfFreedom'][cell] = result.df
            expected['tPValue'][cell] = result.pvalue
            expected['welchSignificance'][cell] = (1 if result.statistic > 0 else 2) if result.pvalue < alpha else 0
    return dict(name=name, slices=slices.ravel(), cellCount=cell_count, sliceCount=slice_count,
                mask=mask, settings=dict(beforeSlice=before, afterSlice=after,
                splitSlice=split, alpha=alpha, epsilon=epsilon), expected=expected)


source = DATA / 'ndvi.bin'
encoded = np.frombuffer(source.read_bytes(), dtype=np.uint8).reshape(16, 256, 256)
decoded = np.where(encoded == 0, np.nan, encoded.astype(float) / 127.5 - 1)
selected = decoded[:, ::8, ::4].transpose(1, 2, 0).reshape(-1, 16)
cases = [make_case('sentinel-dixie-fire-full-time-axis-lattice', selected, np.ones(len(selected), dtype=int), 7, 9)]
synthetic = np.array([
    np.arange(16), np.arange(16)[::-1], np.ones(16), np.zeros(16),
    [np.nan] * 16, [1] + [np.nan] * 15,
    [0] + [np.nan] * 14 + [3],
    [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
    [0, 2, 1, 3, 2, 4, 3, 5, 4, 2, 3, 1, 2, 0, 1, -1],
    [np.nan, 1, 3, np.nan, 5, 2, 8, np.nan, 7, 4, np.nan, 6, 9, 11, np.nan, 12],
    [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    np.arange(16),
], dtype=float)
mask = np.ones(len(synthetic), dtype=int)
mask[-1] = 0
cases.append(make_case('ties-missing-original-times-zero-negative-mask', synthetic, mask, 0, 15))
fixture = dict(versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
    source=dict(path=str(source.relative_to(ROOT)), sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    manifestSha256=hashlib.sha256((DATA/'manifest.json').read_bytes()).hexdigest(),
    daysSha256=hashlib.sha256((DATA/'days.bin').read_bytes()).hexdigest(),
    originalShape=[16, 256, 256], selection='all times, rows 0::8, columns 0::4', crs='EPSG:32610',
    timeUnits='original ordinal slice index', nodata='byte 0 -> NaN', inputPrecision='float32'),
    reference='scipy.stats.theilslopes / kendalltau + norm (continuity correction) / ttest_ind; numpy ufuncs',
    tolerance=dict(profile='S', absoluteFactor=1e-6, relative=1e-4), cases=cases)
output = HERE / 'fixtures/parity-wp4-change-detection.json'
output.write_text(json.dumps(encode(fixture), separators=(',', ':'), allow_nan=False) + '\n')
assert output.stat().st_size <= 1_500_000
print(f'{output.name}: {output.stat().st_size} bytes; {len(cases)} cases')
