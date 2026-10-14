# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Reference class breaks for GPUClassBreaks from mapclassify and jenkspy.

Writes fixtures/dataframe-class-breaks.json next to this script (consumed by
test/gpu-dataframe/column-classification/class-breaks-reference.node.spec.ts and the
'mapclassify and jenkspy references' case of gpu-class-breaks.spec.ts). Run from anywhere with
    <venv>/bin/python -I modules/experimental/test/oracles/dataframe-class-breaks.py
in a venv with numpy, mapclassify and jenkspy (versions are recorded in the fixture).

Scenes are float32 columns from numpy's seeded PCG64 generator; the fixture stores the exact
float32 values, so the GPU and CPU specs classify the same numbers the references saw.

Method mapping (GPUClassBreaks inner edges e[1..k-1] against the reference bins):
- equal-interval k: mapclassify EqualInterval(y, k).bins[:-1].
- quantile k: mapclassify Quantiles(y, k).bins[:-1] (numpy linear percentiles, R-7).
- standard-deviation k, interval 1: mapclassify StdMean(y, multiples=[j - (k - 2) / 2 for
  j in 0..k-2]).bins (sample sd), clipped to [min, max] as GPUClassBreaks clamps inner edges.
- head-tail with headTailRatio 1: mapclassify HeadTailBreaks(y).bins[:-1]; its last bin is the
  mean of the final single-valued head, which is the maximum, i.e. GPUClassBreaks' e[k].
- box-plot hinge 1.5: mapclassify BoxPlot(y).bins[:5] (fences, quartiles) clipped to [min, max].
- maximum-breaks k: mapclassify MaximumBreaks(y, k).bins[:-1].
- natural-breaks k: the exact Fisher-Jenks partition, from an f64 dynamic program over the sorted
  values (classes are upper-inclusive (b[i-1], b[i]], so b[i] is the largest value of class i),
  cross-checked against jenkspy.jenks_breaks (exact; it prepends the minimum, and repeats it when
  the first class is a single value) and mapclassify FisherJenks(y, k).bins. GPUClassBreaks edges are LOWER edges: e[i] is the smallest value of class i, i.e.
  the smallest data value above b[i-1]. The fixture stores both (`exactBreaks`, `exactLowerEdges`),
  the class sizes and the goodness of variance fit (GVF). Every case is checked to have a unique
  optimum: an f64 dynamic program with lowest-start and highest-start tie-breaking must return the
  DP partition both ways, so a GPU tie-break cannot legitimately differ.
  Besides the four closed-form scenes, `NATURAL_SCENES` holds adversarial columns (heavy ties, a
  single extreme outlier, an outlier pair, a Pareto tail, plateaus). A class count above the number
  of distinct values is refused by mapclassify and jenkspy; GPUClassBreaks produces one class per
  distinct value, recorded as `distinctValueEdges` (the sorted distinct values after the first).

Tolerances are derived per case and stored with it:
- equal-interval, box-plot, maximum-breaks: the GPU evaluates the same formula in float32, a few
  roundings of values bounded by max |y|: 8 ulp32(max |y|).
- quantile: the GPU uses p = fround(i / k) and h = fround(fround(n - 1) * p), so h is off by at most
  (n - 1) 2^-25 + h 2^-24 rank units; the value moves by that times the largest gap between adjacent
  sorted values, plus 8 ulp32(max |y|) of interpolation rounding.
- standard-deviation and head-tail: float32 means and a float32 Chan-merged M2 are accurate to a
  relative 2^-20 here (n <= 2^11, fixed tree order): 2^-20 (|mean| + 3 sd) + 8 ulp32(max |y|).
  Head-tail membership (value > mean) must not depend on that rounding, so every value is asserted
  to be farther than 2^-16 relative from every cut.
"""

import json
import os
import sys

import jenkspy
import mapclassify
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'fixtures', 'dataframe-class-breaks.json')


def ulp32(value):
    value = np.float32(abs(value))
    return float(np.spacing(value))


def f32_list(values):
    return [float(np.float32(value)) for value in values]


def make_scenes():
    rng = np.random.default_rng(20261014)
    lognormal = rng.lognormal(mean=0.0, sigma=1.0, size=800).astype(np.float32)
    centers = np.array([5.0, 40.0, 90.0])
    clusters = (centers[np.arange(240) % 3] + rng.uniform(-2.0, 2.0, 240)).astype(np.float32)
    outliers = np.concatenate(
        [rng.normal(50.0, 5.0, 480), rng.normal(50.0, 40.0, 20)]
    ).astype(np.float32)
    uniform = rng.uniform(-100.0, 100.0, 500).astype(np.float32)
    return {
        'lognormal': lognormal,
        'clusters': clusters,
        'outliers': outliers,
        'uniform': uniform,
    }


def goodness_of_variance_fit(values, breaks):
    """GVF = 1 - SDCM / SDAM for upper-inclusive breaks [min, b1, ..., max] (jenkspy convention)."""
    values = np.sort(values.astype(np.float64))
    total = np.sum((values - values.mean()) ** 2)
    within = 0.0
    lower = -np.inf
    for upper in breaks[1:]:
        members = values[(values > lower) & (values <= upper)]
        if lower == -np.inf:
            members = values[values <= upper]
        if members.size:
            within += np.sum((members - members.mean()) ** 2)
        lower = upper
    return 1.0 - within / total


def class_sizes(values, breaks):
    values = values.astype(np.float64)
    sizes = []
    lower = -np.inf
    for upper in breaks[1:]:
        sizes.append(int(np.sum((values > lower) & (values <= upper))))
        lower = upper
    return sizes



NATURAL_CLASS_COUNTS = {
    'lognormal': (2, 3, 5, 7, 10),
    'clusters': (2, 3, 4, 5, 10),
    'outliers': (3, 5, 8),
    'uniform': (3, 5, 9),
    'ties-counts': (2, 3, 5, 7),
    'extreme-outlier': (2, 3, 4, 5),
    'outlier-pair': (3, 4, 5),
    'pareto': (3, 5, 8),
    'plateaus': (2, 4, 6),
    'two-values': (2,),
}


def make_natural_scenes():
    """Adversarial columns for the natural-breaks cases only (n <= 2000, own seeded generator, so
    the closed-form scenes above keep their values)."""
    rng = np.random.default_rng(20261016)
    extreme = np.concatenate([rng.normal(10.0, 1.0, 500), [1.0e6]])
    rng.shuffle(extreme)
    pair = np.concatenate([rng.normal(0.0, 1.0, 400), [-1.0e5, 1.0e5]])
    rng.shuffle(pair)
    return {
        'ties-counts': rng.poisson(3.0, 1000).astype(np.float32),
        'extreme-outlier': extreme.astype(np.float32),
        'outlier-pair': pair.astype(np.float32),
        'pareto': (rng.pareto(1.1, 1500) + 1.0).astype(np.float32),
        'plateaus': np.repeat(np.array([1, 2, 4, 8, 16, 32], dtype=np.float32), 100),
        'two-values': np.repeat(np.array([0, 1], dtype=np.float32), 50),
    }


def exact_partitions(values, k):
    """Exact Fisher-Jenks by an f64 dynamic program over the sorted values.

    Returns the class start indices of the partition found with lowest-start and with
    highest-start tie-breaking, and the optimum cost. The two agree when the optimum is unique.
    """
    x = np.sort(values.astype(np.float64))
    x = x - x.mean()
    n = x.size
    prefix1 = np.concatenate([[0.0], np.cumsum(x)])
    prefix2 = np.concatenate([[0.0], np.cumsum(x * x)])
    start = np.arange(n)[:, None]
    end = np.arange(n)[None, :]
    count = (end - start + 1).astype(np.float64)
    count[count <= 0] = np.nan
    sums = prefix1[end + 1] - prefix1[start]
    squares = prefix2[end + 1] - prefix2[start]
    cost = squares - sums * sums / count
    cost[np.isnan(count)] = np.inf
    totals = [cost[0].copy()]
    lowest = [np.zeros(n, dtype=int)]
    highest = [np.zeros(n, dtype=int)]
    for layer in range(1, k):
        previous = np.full(n, np.inf)
        previous[layer:] = totals[-1][layer - 1 : n - 1]
        # candidate[s, j] = totals[layer - 1][s - 1] + cost[s, j] for s in [layer, j]
        candidate = np.full((n, n), np.inf)
        candidate[layer:, :] = totals[-1][layer - 1 : n - 1, None] + cost[layer:, :]
        best = candidate.min(axis=0)
        tolerance = 1e-9 * np.maximum(np.abs(best), 1.0)
        near = candidate <= best[None, :] + tolerance[None, :]
        lowest.append(np.argmax(near, axis=0))
        highest.append(n - 1 - np.argmax(near[::-1, :], axis=0))
        totals.append(best)

    def backtrack(starts):
        end_index = n - 1
        result = []
        for layer in range(k - 1, 0, -1):
            begin = int(starts[layer][end_index])
            result.append(begin)
            end_index = begin - 1
        return result[::-1]

    return backtrack(lowest), backtrack(highest), float(totals[-1][n - 1])


def natural_reference(name, y, k):
    y64 = y.astype(np.float64)
    sorted_values = np.sort(y64)
    distinct = np.unique(y64)
    if k > distinct.size:
        raise ValueError(f'{name}: k={k} above {distinct.size} distinct values')
    low, high, _ = exact_partitions(y, k)
    unique = low == high
    if not unique:
        raise ValueError(f'{name}: k={k} has tied optima (lowest {low}, highest {high})')
    # A start index inside a run of equal values cannot be written as a value threshold.
    for start in low:
        if sorted_values[start] == sorted_values[start - 1]:
            raise ValueError(f'{name}: k={k} splits equal values at sorted index {start}')
    lower_edges = [float(sorted_values[start]) for start in low]
    upper_bounds = [float(sorted_values[start - 1]) for start in low] + [float(y64.max())]
    breaks = [float(y64.min())] + upper_bounds
    # Independent implementations as cross-checks. jenkspy 0.4.1 is exact (same partition as the
    # DP); mapclassify 2.11 FisherJenks matches on most cases but returns a marginally suboptimal
    # partition on some (lognormal k=7: SSD 102.0537 against the optimum 102.0536), so agreement
    # is recorded instead of asserted for it.
    cross = jenkspy.jenks_breaks(y64.tolist(), n_classes=k)
    jenkspy_agrees = len(cross) == k + 1 and bool(np.array_equal(np.asarray(cross[1:]), upper_bounds))
    jenkspy_values_agree = bool(
        np.array_equal(np.unique(np.asarray(cross[1:])), np.unique(np.asarray(upper_bounds)))
    )
    fisher = mapclassify.FisherJenks(y64, k=k).bins
    mapclassify_agrees = bool(np.array_equal(np.asarray(fisher), upper_bounds))
    mapclassify_lower_edges = [
        float(sorted_values[np.searchsorted(sorted_values, upper, side='right')])
        for upper in fisher[:-1]
    ]
    assert jenkspy_agrees or jenkspy_values_agree or mapclassify_agrees, (
        f'{name} k={k}: neither jenkspy {cross} nor mapclassify {fisher} matches the DP {upper_bounds}'
    )
    return {
        'classCount': k,
        'exactBreaks': breaks,
        'exactLowerEdges': lower_edges,
        'exactClassSizes': class_sizes(y, breaks),
        'exactGoodnessOfVarianceFit': float(goodness_of_variance_fit(y, breaks)),
        'jenkspyAgrees': jenkspy_agrees or jenkspy_values_agree,
        'mapclassifyAgrees': mapclassify_agrees,
        'mapclassifyLowerEdges': mapclassify_lower_edges,
    }


def edge_case(values, method, k, edges, tolerance, extra=None):
    case = {
        'method': method,
        'classCount': k,
        'innerEdges': [float(edge) for edge in edges],
        'absoluteTolerance': float(tolerance),
    }
    if extra:
        case.update(extra)
    return case


def build_cases(name, y):
    y64 = y.astype(np.float64)
    maximum_abs = float(np.max(np.abs(y64)))
    rounding = 8 * ulp32(maximum_abs)
    minimum, maximum = float(y64.min()), float(y64.max())
    sorted_values = np.sort(y64)
    n = y.size
    cases = []

    for k in (4, 5, 7):
        cases.append(
            edge_case(y, 'equal-interval', k, mapclassify.EqualInterval(y64, k).bins[:-1], rounding)
        )
        bins = mapclassify.Quantiles(y64, k=k).bins
        assert len(bins) == k, f'{name}: tied quantiles at k={k}'
        # Linear interpolation is continuous in h, so the value moves by the rank error times the
        # gap of the segment around h (one neighbor either side covers a floor(h) step).
        rank_error = (n - 1) * 2.0**-25 + (n - 1) * 2.0**-24
        gaps = np.diff(sorted_values)
        local_gap = 0.0
        for index in range(1, k):
            low = int(np.floor((n - 1) * index / k))
            local_gap = max(local_gap, float(np.max(gaps[max(low - 1, 0) : min(low + 2, n - 1)])))
        cases.append(edge_case(y, 'quantile', k, bins[:-1], rank_error * local_gap + rounding))

    mean = float(y64.mean())
    sample_sd = float(y64.std(ddof=1))
    moment_tolerance = 2.0**-20 * (abs(mean) + 3 * sample_sd) + rounding
    for k in (4, 6):
        multiples = [j - (k - 2) / 2 for j in range(k - 1)]
        bins = mapclassify.StdMean(y64, multiples=multiples).bins
        cuts = np.clip(bins[: k - 1], minimum, maximum)
        cases.append(
            edge_case(
                y,
                'standard-deviation',
                k,
                cuts,
                moment_tolerance,
                {'standardDeviationInterval': 1},
            )
        )

    head_tail = mapclassify.HeadTailBreaks(y64).bins
    assert head_tail[-1] == maximum, f'{name}: head/tail last cut is not the maximum'
    assert len(head_tail) <= 10, f'{name}: head/tail needs more than 10 classes'
    for cut in head_tail[:-1]:
        nearest = float(np.min(np.abs(y64 - cut)))
        assert nearest > 2.0**-16 * max(abs(cut), 1e-30), f'{name}: value too close to cut {cut}'
    cases.append(
        edge_case(
            y,
            'head-tail',
            len(head_tail),
            head_tail[:-1],
            moment_tolerance,
            {'headTailRatio': 1, 'requestedClassCount': 10},
        )
    )

    box = mapclassify.BoxPlot(y64, hinge=1.5).bins
    cases.append(
        edge_case(
            y, 'box-plot', 6, np.clip(box[:5], minimum, maximum), rounding, {'boxPlotHinge': 1.5}
        )
    )

    for k in (3, 5):
        diffs = np.diff(sorted_values)
        top = np.sort(diffs)[::-1]
        assert top[k - 2] > top[k - 1] * (1 + 1e-6), f'{name}: tied gaps at k={k}'
        cases.append(
            edge_case(y, 'maximum-breaks', k, mapclassify.MaximumBreaks(y64, k=k).bins[:-1], rounding)
        )

    natural = [natural_reference(name, y, k) for k in NATURAL_CLASS_COUNTS[name]]
    return cases, natural


def main():
    scenes = make_scenes()
    fixture = {
        'versions': {
            'python': sys.version.split()[0],
            'numpy': np.__version__,
            'mapclassify': mapclassify.__version__,
            'jenkspy': getattr(jenkspy, '__version__', '0.4.1'),
        },
        'scenes': {},
    }
    for name, y in scenes.items():
        cases, natural = build_cases(name, y)
        fixture['scenes'][name] = {
            'values': f32_list(y),
            'cases': cases,
            'naturalBreaks': natural,
        }
    for name, y in make_natural_scenes().items():
        fixture['scenes'][name] = {
            'values': f32_list(y),
            'cases': [],
            'naturalBreaks': [natural_reference(name, y, k) for k in NATURAL_CLASS_COUNTS[name]],
            'distinctValueEdges': [float(v) for v in np.unique(y.astype(np.float64))[1:]],
        }
    fixture['generator'] = 'test/oracles/dataframe-class-breaks.py'
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, 'w') as file:
        json.dump(fixture, file, indent=1)
        file.write('\n')
    print(f'wrote {OUTPUT}')


if __name__ == '__main__':
    main()
