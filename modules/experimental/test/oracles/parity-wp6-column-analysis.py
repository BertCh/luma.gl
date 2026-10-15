# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""WP6 Round 4 independent column references (run with the supplied Python -I).

All numeric inputs/parameters are f32-rounded first; statistics use f64. Full
US county (3109) and Chicago tract (791) data, no sampling or random seeds.
Nonspatial attributes need no CRS conversion. Manifest/source SHA256 and
versions are recorded. Every output array is retained, including masks.

Profile: pandas describe/sum/var/std, NaN-only nulls, infinities count and
participate in extrema but not moments/histograms. Population and sample ddof
are explicit. Categories include overflow in count/HLL/extrema, not top-K or
histogram. pandas value_counts with count-desc/code-asc ties supplies top-K.
No skewness/kurtosis slots exist in this API. HLL is a COMPOSITE REFERENCE:
Flajolet et al. (2007), Algorithm 2, https://algo.inria.fr/flajolet/Publications/FlFuGaMe07.pdf
with Appleby's published fmix32 finalizer (public domain),
https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp .
No Python library exposes the API's standalone fmix32(seed XOR f32 bits)
register convention. NumPy maximum.at implements the published register
maximum definition; the estimate is f64 harmonic/linear counting, NOT true
cardinality. Fixed seed=(column+1)*0x9e3779b9 mod 2^32. -0 canonicalized for
hashing, distinct signed zeros retained for extrema. Histogram cases use exact
dyadic domains/widths so numpy.histogram is independent of GPU bin arithmetic.

Quantiles: numpy.quantile(method=all five public methods). Finite unmasked
rows only; invalid p maps to NaN. Valid p are dyadic and n<2^24, so (n-1)*p
is exactly f32 representable in these cases. Filter rank selection is the
published deck.gl floor(n*lower), ceil(n*upper)-1 contract; NumPy partition
selects values and vector comparisons supply every mask. Ties are inclusive.

Composite: sklearn StandardScaler/MinMaxScaler, scipy rankdata(average),
numpy.average and scipy gmean. Listwise finite deletion before statistics,
mask!=0, population std, inverse indicators after scaling, constant columns
zero before direction flip, singleton rank .5. Negative sum weights normalize
by sum(abs(weights)); geometric weights are nonnegative, shifted by f32 epsilon.
Principal components use scipy.linalg.eigh of direction-adjusted standardized
population covariance, signed sum>=0; compare every loading, projection,
eigenvalue, explained ratio, residual. Well-separated eigenvalues only;
no local fixed-step power iteration is used as a reference. Empty/all-constant
PCA has no unique reference eigenvector and is explicitly not requested.
"""
import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import linalg, stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DATA = ROOT / 'examples/deck/spatial-analysis-showcase/public/data'
VERSIONS = {'python': platform.python_version(), 'numpy': np.__version__,
            'pandas': pd.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__}
SOURCES = {}


def load(dataset, column):
    manifest_path = DATA / dataset / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    description = manifest['columns'][column]
    path = DATA / dataset / description['file']
    for source in [manifest_path, path]:
        SOURCES[str(source.relative_to(ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
    dtype = {'float32': '<f4', 'uint32': '<u4', 'uint8': 'u1', 'uint16': '<u2'}[description['dtype']]
    return np.fromfile(path, dtype=dtype)


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
        if value == 0 and math.copysign(1, value) < 0:
            return '-0'
        return float(value)
    return value


def write(output, topic, cases, reference):
    path = output / f'parity-wp6-{topic}.json'
    content = {'generator': Path(__file__).name, 'versions': VERSIONS, 'sources': SOURCES.copy(),
               'reference': reference, 'conventions': __doc__, 'cases': cases}
    path.write_text(json.dumps(encode(content), allow_nan=False, separators=(',', ':')) + '\n')
    assert path.stat().st_size <= 1500000, (path.name, path.stat().st_size)
    print(path.name, path.stat().st_size)


def hll_reference(bits, column, precision):
    # Composite reference: the published MurmurHash3 fmix32 finalizer, using
    # uint64 intermediates and explicit modular arithmetic instead of GPU u32.
    hashed = np.asarray(bits, np.uint64) ^ (((column + 1) * 0x9e3779b9) & 0xffffffff)
    for shift, multiplier in [(16, 0x85ebca6b), (13, 0xc2b2ae35)]:
        hashed ^= hashed >> shift
        hashed = (hashed * multiplier) & 0xffffffff
    hashed ^= hashed >> 16
    register_count = 1 << precision
    indices = (hashed >> (32 - precision)).astype(int)
    suffix = hashed & ((1 << (32 - precision)) - 1)
    ranks = np.array([33 - precision - int(value).bit_length() for value in suffix])
    registers = np.zeros(register_count, dtype=int)
    np.maximum.at(registers, indices, ranks)
    alpha = {16: .673, 32: .697, 64: .709}.get(register_count, .7213 / (1 + 1.079 / register_count))
    estimate = alpha * register_count ** 2 / np.sum(np.exp2(-registers.astype(float)))
    zero_count = np.count_nonzero(registers == 0)
    if estimate <= 2.5 * register_count and zero_count:
        estimate = register_count * np.log(register_count / zero_count)
    return registers, estimate


def profile_case(name, columns, mask, bins=16, precision=8, top_count=8):
    output = {field: [] for field in ['statistics', 'counts', 'histograms', 'registers', 'topCategories', 'topCategoryCounts']}
    normalized = []
    for column_index, column in enumerate(columns):
        category = column.get('kind') == 'category'
        values = np.asarray(column['values'], np.uint32 if category else np.float32)
        selected = values[np.asarray(mask) != 0]
        nulls = selected == 0xffffffff if category else np.isnan(selected)
        counted = selected[~nulls]
        finite = counted[np.isfinite(counted)].astype(float)
        minimum, maximum = (float(np.min(counted)), float(np.max(counted))) if len(counted) else (np.nan, np.nan)
        if not category and len(counted):
            if minimum == 0 and np.any(np.signbit(counted[counted == 0])):
                minimum = -0.
            if maximum == 0 and np.any(~np.signbit(counted[counted == 0])):
                maximum = 0.
        bits = counted.copy()
        if not category:
            bits[bits == 0] = 0.
            bits = bits.view(np.uint32)
        registers, estimate = hll_reference(bits, column_index, precision)
        output['registers'].extend(registers)
        top_categories = np.full(top_count, 0xffffffff, dtype=np.uint32)
        top_counts = np.zeros(top_count, dtype=int)
        if category:
            category_count = column['categoryCount']
            frequencies = pd.Series(counted[counted < category_count]).value_counts()
            ordered = sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))[:top_count]
            for index, (code, count) in enumerate(ordered):
                top_categories[index], top_counts[index] = code, count
            histogram = frequencies.reindex(range(bins), fill_value=0).to_numpy()
            moments = [np.nan] * 5
            overflow_count = np.count_nonzero(counted >= category_count)
        else:
            series = pd.Series(finite)
            description = series.describe()
            moments = [series.sum() if len(series) else np.nan, description['mean'], series.var(ddof=0),
                       series.var(ddof=1), series.std(ddof=0)]
            overflow_count = 0
            domain = column.get('domain')
            if domain is None:
                domain = [float(np.min(finite)), float(np.max(finite))] if len(finite) else [np.nan, np.nan]
            if not np.all(np.isfinite(domain)) or domain[0] > domain[1]:
                histogram = np.zeros(bins, dtype=int)
            elif domain[0] == domain[1]:
                histogram = np.zeros(bins, dtype=int)
                histogram[0] = np.count_nonzero(finite == domain[0])
            else:
                histogram = np.histogram(finite, bins=bins, range=domain)[0]
        output['statistics'].extend([len(counted), int(np.sum(nulls)), minimum, maximum, *moments, estimate, overflow_count])
        output['counts'].extend([len(counted), int(np.sum(nulls))])
        output['histograms'].extend(histogram)
        output['topCategories'].extend(top_categories)
        output['topCategoryCounts'].extend(top_counts)
        normalized.append({**column, 'values': values})
    return {'name': name, 'columns': normalized, 'mask': list(mask), 'bins': bins,
            'precision': precision, 'topCount': top_count, 'expected': output}


def quantile_case(name, values, mask, filter_range):
    values = np.asarray(values, np.float32).astype(float)
    probabilities = np.array([0, .125, .25, .5, .75, .875, 1, -.25, 1.25, np.nan], np.float32)
    valid = np.isfinite(values) & (np.asarray(mask) != 0)
    sample = values[valid]
    count = len(sample)
    quantiles = {}
    for method in ['lower', 'higher', 'nearest', 'linear', 'midpoint']:
        quantiles[method] = [np.quantile(sample, probability, method=method)
                            if count and 0 <= probability <= 1 else np.nan for probability in probabilities]
    bounds = [np.nan, np.nan]
    output_mask = np.zeros(len(values), dtype=int)
    if count:
        lower = 0 if np.isnan(filter_range[0]) else filter_range[0]
        upper = 1 if np.isnan(filter_range[1]) else filter_range[1]
        lower_index = np.clip(int(np.floor(np.float32(count) * np.float32(lower))), 0, count - 1)
        upper_index = np.clip(int(np.ceil(np.float32(count) * np.float32(upper))) - 1, 0, count - 1)
        partitioned = np.partition(sample, [lower_index, upper_index])
        bounds = partitioned[[lower_index, upper_index]]
        output_mask = (valid & (values >= bounds[0]) & (values <= bounds[1])).astype(int)
    return {'name': name, 'values': values, 'mask': list(mask), 'probabilities': probabilities,
            'filterRange': filter_range, 'expected': {'quantiles': quantiles, 'validCount': [count],
            'filterBounds': bounds, 'outputMask': output_mask}}


def composite_case(name, indicators, mask, scaler, aggregation, weights, directions, principal=False):
    indicators = np.asarray(indicators, np.float32).astype(float)
    weights = np.asarray(weights, np.float32).astype(float)
    directions = np.asarray(directions, float)
    valid = np.all(np.isfinite(indicators), axis=1) & (np.asarray(mask) != 0)
    selected = indicators[valid]
    row_count, column_count = indicators.shape
    epsilon = float(np.float32(.001))
    scaled = np.full(indicators.shape, np.nan)
    scores = np.full(row_count, np.nan)
    column_statistics = np.full((column_count, 4), np.nan)
    output = {}
    eigengap = None
    if len(selected):
        standard = StandardScaler().fit(selected)
        deviations = np.sqrt(standard.var_)
        column_statistics = np.stack([np.min(selected, axis=0), np.max(selected, axis=0),
                                      standard.mean_, deviations], axis=1)
        if scaler == 'z-score' or principal:
            components = standard.transform(selected) * directions
        else:
            components = (MinMaxScaler().fit_transform(selected) if scaler == 'min-max' else
                          (stats.rankdata(selected, axis=0, method='average') - 1) / (len(selected) - 1)
                          if len(selected) > 1 else np.full(selected.shape, .5))
            components = np.where(directions < 0, 1 - components, components)
        scaled[valid] = components
        if principal:
            correlation = components.T @ components / len(selected)
            eigenvalues, eigenvectors = linalg.eigh(correlation)
            loadings = eigenvectors[:, -1]
            if np.sum(loadings) < 0 or (np.sum(loadings) == 0 and loadings[np.argmax(np.abs(loadings))] < 0):
                loadings = -loadings
            eigengap = float(eigenvalues[-1] - eigenvalues[-2]) if column_count > 1 else float(eigenvalues[-1])
            assert eigengap > .25, (name, eigenvalues)
            eigenvalue = eigenvalues[-1]
            scores[valid] = components @ loadings
            output['loadings'] = loadings
            output['principalComponentSummary'] = [eigenvalue, eigenvalue / np.count_nonzero(deviations),
                                                   np.linalg.norm(correlation @ loadings - eigenvalue * loadings)]
        elif aggregation == 'weighted-sum' and np.any(weights):
            scores[valid] = np.average(components * np.sign(weights), axis=1, weights=np.abs(weights))
        elif aggregation == 'weighted-geometric-mean' and np.any(weights > 0):
            positive = weights > 0
            shifted = components[:, positive] + epsilon
            good = np.all(shifted > 0, axis=1)
            selected_scores = np.full(len(selected), np.nan)
            selected_scores[good] = stats.gmean(shifted[good], axis=1, weights=weights[positive])
            scores[valid] = selected_scores
    output.update(score=scores, scaled=scaled.ravel(), columnStatistics=column_statistics.ravel())
    return {'name': name, 'indicators': indicators.ravel(), 'indicatorCount': column_count,
            'mask': list(mask), 'principal': principal, 'eigengap': eigengap,
            'settings': {'scaler': scaler, 'aggregation': aggregation, 'weights': weights,
                         'directions': directions, 'epsilon': epsilon}, 'expected': output}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=HERE / 'fixtures')
    output = parser.parse_args().output_dir
    output.mkdir(parents=True, exist_ok=True)
    income = load('us-counties', 'medianHouseholdIncome')
    population = load('us-counties', 'population')
    states = load('us-counties', 'stateFips')
    assert len(income) == len(population) == len(states) == 3109
    columns = [{'values': income, 'domain': [0, 262144]}, {'values': population, 'domain': [0, 16777216]},
               {'values': states, 'kind': 'category', 'categoryCount': 57}]
    profiles = [profile_case('counties-all-small-registers', columns, np.ones(3109, int), bins=64, precision=4),
                profile_case('counties-masked-large-registers', columns, np.where(np.arange(3109) % 7, 7, 0), bins=64, precision=12)]
    edge_columns = [{'values': [-0., 0, 1, 1, 2, 4, 8, np.nan, np.inf, -np.inf, 99, .5], 'domain': [0, 8]},
                    {'values': [2, 1, 2, 1, 3, 3, 4, 0xffffffff, 7, 8, 0, 0], 'kind': 'category', 'categoryCount': 5},
                    {'values': [1048576 + offset for offset in [-4, -2, -1, 0, 0, 1, 2, 3, 4, 6, 12, 7]], 'domain': [1048560, 1048592]}]
    profiles.append(profile_case('null-infinity-overflow-ties-offset', edge_columns, [1] * 10 + [0, 7]))
    for name, values in [('signed-zeros', [-0., 0., -0.]), ('singleton', [7, np.nan, np.nan]),
                         ('no-finite', [np.inf, -np.inf, np.nan]), ('constant', [3, 3, 3])]:
        profiles.append(profile_case(name, [{'values': values}], [1, 1, 1]))
    profiles.append(profile_case('all-masked', edge_columns, [0] * 12))
    write(output, 'column-profile', profiles, 'pandas.Series.describe/sum/var/std/value_counts; numpy.histogram; composite Flajolet HLL + Appleby fmix32')
    quantiles = [quantile_case('counties-income-all', income, np.ones(3109, int), [.125, .875]),
                 quantile_case('counties-population-masked', population, np.where(np.arange(3109) % 7, 7, 0), [np.nan, np.nan]),
                 quantile_case('ties-missing-inclusive', [1, 1, 1, 2, 2, 4, -0., 0, np.nan, np.inf, -np.inf, 99], [1] * 11 + [0], [.25, .75]),
                 quantile_case('nearest-half-even-upper', [0, 10, 20, 30], [1] * 4, [0, 1]),
                 quantile_case('nearest-half-even-lower', [0, 10, 20, 30, 40, 50], [1] * 6, [.75, .25]),
                 quantile_case('singleton', [np.nan, 7, np.inf], [1, 7, 1], [0, 1]),
                 quantile_case('all-masked', [1, 2, 3], [0, 0, 0], [0, 1]),
                 quantile_case('all-nonfinite', [np.nan, np.inf, -np.inf], [1, 1, 1], [0, 1])]
    write(output, 'column-quantiles', quantiles, 'numpy.quantile(lower/higher/nearest/linear/midpoint); numpy.partition + inclusive comparisons')
    SOURCES.clear()
    indicators = np.column_stack([load('chicago-tracts', name) for name in ['poverty150Pct', 'unemployedPct', 'noVehiclePct', 'perCapitaIncome']])
    assert indicators.shape == (791, 4)
    composite = []
    directions = [1, 1, 1, -1]
    for scaler in ['min-max', 'z-score', 'rank']:
        for aggregation in ['weighted-sum', 'weighted-geometric-mean']:
            composite.append(composite_case(f'chicago-all-{scaler}-{aggregation}', indicators, np.ones(791, int),
                scaler, aggregation, [1, 2, 3, 4], directions))
    composite.append(composite_case('chicago-all-principal-component', indicators, np.ones(791, int),
        'rank', 'principal-component', [0, 0, 0, 0], directions, principal=True))
    edge_indicators = [[0, 4, 2], [1, 3, 2], [1, 3, 2], [3, 1, 2], [4, 0, 2],
                       [np.nan, 9, 2], [8, np.inf, 2], [-np.inf, 1, 2], [99, 99, 99], [2, 2, 2]]
    edge_mask = [1] * 8 + [0, 7]
    for scaler in ['min-max', 'z-score', 'rank']:
        composite.append(composite_case(f'edges-{scaler}-signed-weights', edge_indicators, edge_mask,
            scaler, 'weighted-sum', [1, -2, 3], [1, -1, -1]))
    composite.append(composite_case('edges-principal-constant-column', edge_indicators, edge_mask,
        'min-max', 'principal-component', [1, 2, 3], [1, -1, 1], principal=True))
    composite.append(composite_case('zero-weights', edge_indicators, edge_mask,
        'min-max', 'weighted-sum', [0, 0, 0], [1, 1, 1]))
    composite.append(composite_case('geometric-zero-weight-negative-component', edge_indicators, edge_mask,
        'z-score', 'weighted-geometric-mean', [0, 0, 1], [1, 1, 1]))
    for scaler in ['min-max', 'z-score', 'rank']:
        composite.append(composite_case(f'singleton-{scaler}', [[1, 2, 3]], [7], scaler,
            'weighted-sum', [1, 2, 3], [1, -1, 1]))
    composite.append(composite_case('all-masked', edge_indicators, [0] * len(edge_mask),
        'min-max', 'weighted-sum', [1, 2, 3], [1, 1, 1]))
    write(output, 'composite-score', composite, 'sklearn.preprocessing.StandardScaler/MinMaxScaler; scipy.stats.rankdata/gmean; numpy.average; scipy.linalg.eigh')


if __name__ == '__main__':
    main()
