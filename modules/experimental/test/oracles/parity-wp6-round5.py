# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent WP6 round 5 references; run with the supplied Python -I.

Pandas merge/groupby: unsigned key equality, all-ones sentinel removed BEFORE
merge (pandas otherwise matches nulls), mask != 0, smallest right row gather,
finite-only aggregates, and ascending compact left IDs. Exact fixed-point
words use the API's np.rint(f32(value * 65536)) representation. No fanout rows
are exposed by GPUKeyJoin; compare every first match, multiplicity, membership,
gather and aggregate instead. All 3109 counties are right-side inputs.

Pandas Series.between: closed instant windows, reversed window empty; f32
relative seconds or exact signed Int64 epoch milliseconds. Real inputs retain
all 77231 USGS events. Sorted and scan paths use identical fixtures. Full masks
are losslessly encoded, not summarized. Synthetic cases test negative epoch,
Int64 beyond 2^53, duplicates, NaN, infinities and bounded compact output.

NumPy searchsorted(side='right'): upper-owned interior class edges, clamped
outer edges, signed zeros equal, infinities classified, NaN/mask -> no class.
Joint id=y*classCountX+x. NumPy bincount supplies every class count; palette
lookup and optional linear alpha are direct public mapping, not local shader
code. Alpha uses dyadic factors and alpha=128 so every resulting byte is exact.

Pandas var(ddof=0/1) supplies large-offset regressions. 11, 257, 2051 and 4103
rows cross invocation/tile boundaries with leading invalid/masked values,
positive/negative offsets and constant inputs. No tolerance is increased.
No random seed or spatial CRS applies. Sources are hashed; versions recorded.
"""
import argparse
import base64
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('column_reference', HERE / 'parity-wp6-column-analysis.py')
column = importlib.util.module_from_spec(spec)
spec.loader.exec_module(column)
NONE = 0xffffffff


def pack(values, dtype):
    values = np.asarray(values, dtype=dtype)
    return {'dtype': dtype, 'length': len(values), 'base64': base64.b64encode(values.tobytes()).decode()}


def write(output, name, cases, reference):
    path = output / f'parity-wp6-{name}.json'
    body = {'generator': Path(__file__).name, 'versions': column.VERSIONS, 'sources': column.SOURCES.copy(),
            'reference': reference, 'conventions': __doc__, 'cases': cases}
    path.write_text(json.dumps(column.encode(body), allow_nan=False, separators=(',', ':')) + '\n')
    assert path.stat().st_size < 1500000, (path, path.stat().st_size)
    print(path.name, path.stat().st_size)


def join_case(name, left_keys, right_keys, values, left_mask, right_mask, bits=32, capacity=None):
    sentinel = (1 << bits) - 1
    left_keys, right_keys = list(map(int, left_keys)), list(map(int, right_keys))
    values = np.asarray(values, np.float32).astype(float)
    left = pd.DataFrame({'key': left_keys, 'leftRow': np.arange(len(left_keys))})
    right = pd.DataFrame({'key': right_keys, 'rightRow': np.arange(len(right_keys)), 'value': values})
    left = left[(np.asarray(left_mask) != 0) & (left.key != sentinel)]
    right = right[(np.asarray(right_mask) != 0) & (right.key != sentinel)]
    pairs = left.merge(right, on='key', how='inner', sort=False)
    all_rows = np.arange(len(left_keys))
    grouped = pairs.groupby('leftRow', sort=True)
    counts = grouped.size().reindex(all_rows, fill_value=0).to_numpy()
    first_rows = grouped.rightRow.min().reindex(all_rows, fill_value=NONE).to_numpy()
    gathered = np.full(len(left_keys), np.nan)
    gathered[counts > 0] = values[first_rows[counts > 0].astype(int)]
    finite = pairs[np.isfinite(pairs.value)].copy()
    finite['quantized'] = np.rint((finite.value.to_numpy().astype(np.float32) * np.float32(65536))).astype(np.int64)
    aggregates = finite.groupby('leftRow').agg(count=('value', 'count'), sum=('quantized', 'sum'),
                                               minimum=('value', 'min'), maximum=('value', 'max'))
    aggregates = aggregates.reindex(all_rows)
    finite_counts = aggregates['count'].fillna(0).to_numpy()
    integer_sums = aggregates['sum'].fillna(0).astype(np.int64).to_numpy()
    sums = integer_sums.astype(float) / 65536
    sums[counts == 0] = np.nan
    means = np.divide(integer_sums.astype(float) / 65536, finite_counts,
                      out=np.full(len(all_rows), np.nan), where=finite_counts > 0)
    selected = np.flatnonzero(counts)
    capacity = len(left_keys) if capacity is None else capacity
    right_matched = np.zeros(len(right_keys), int)
    right_matched[pairs.rightRow.unique()] = 1
    words = np.array([[int(value) & NONE, (int(value) >> 32) & NONE] for value in integer_sums]).ravel()
    expected = {'rightRows': first_rows, 'matchCounts': counts, 'matched': (counts > 0).astype(int),
                'rightMatched': right_matched, 'gathered': gathered, 'countAll': counts,
                'countFinite': finite_counts, 'sum': sums, 'mean': means,
                'minimum': aggregates.minimum.to_numpy(), 'maximum': aggregates.maximum.to_numpy(),
                'sumWords': words, 'meanWords': words, 'ids': selected[:capacity],
                'count': [min(capacity, len(selected))], 'requiredCount': [len(selected)],
                'overflow': [int(len(selected) > capacity)]}
    return {'name': name, 'bits': bits, 'leftKeys': list(map(str, left_keys)),
            'rightKeys': list(map(str, right_keys)), 'values': values,
            'leftMask': left_mask, 'rightMask': right_mask, 'capacity': capacity, 'expected': expected}


def time_case(name, values, start, end, mode='float', capacity=None, real=False):
    values = np.asarray(values, np.float32).astype(float) if mode == 'float' else np.asarray(values, np.int64)
    if mode == 'float':
        start, end = float(np.float32(start)), float(np.float32(end))
    mask = pd.Series(values).between(start, end, inclusive='both').to_numpy().astype(np.uint32)
    ids = np.flatnonzero(mask)
    capacity = len(values) if capacity is None else capacity
    return {'name': name, 'mode': mode, 'real': real, 'values': ([] if real else list(map(str, values)) if mode == 'word' else values),
            'start': str(start) if mode == 'word' else start, 'end': str(end) if mode == 'word' else end,
            'capacity': capacity, 'rowCount': len(values), 'sortedEligible': bool(np.all(np.diff(values) >= 0)),
            'expected': {'mask': pack(mask, '<u4'), 'ids': pack(ids[:capacity], '<u4'),
                         'count': [min(len(ids), capacity)], 'requiredCount': [len(ids)],
                         'overflow': [int(len(ids) > capacity)]}}


def bivariate_case(name, values_x, values_y, mask):
    values_x, values_y = np.asarray(values_x, np.float32), np.asarray(values_y, np.float32)
    alpha = np.resize(np.array([0, .25, .5, .75, 1, np.nan], np.float32), len(mask))
    palette = np.array([0x80000000 | (index * 9127 + 12345) for index in range(16)], np.uint32)
    frames = []
    for edges_x, edges_y, use_alpha in [([0, 32768, 65536, 131072], [0, 65536, 262144, 16777216], False),
                                       ([-1, 0, 0, 1, 2], [-2, 0, 2], True),
                                       ([0, 1], [0, 1], False)]:
        count_x, count_y = len(edges_x) - 1, len(edges_y) - 1
        class_x = np.searchsorted(edges_x[1:-1], values_x, side='right')
        class_y = np.searchsorted(edges_y[1:-1], values_y, side='right')
        valid = ~np.isnan(values_x) & ~np.isnan(values_y) & (np.asarray(mask) != 0)
        classes = np.where(valid, class_y * count_x + class_x, NONE).astype(np.uint32)
        colors = np.full(len(mask), 0x0a0b0c0d, np.uint32)
        colors[valid] = palette[classes[valid]]
        if use_alpha:
            factors = .25 + .75 * np.nan_to_num(alpha.astype(float), nan=0)
            alpha_bytes = np.rint(128 * factors).astype(np.uint32)
            colors[valid] = (colors[valid] & 0xffffff) | (alpha_bytes[valid] << 24)
        counts = np.bincount(classes[valid], minlength=16)
        frames.append({'breaksX': edges_x, 'breaksY': edges_y, 'alpha': use_alpha,
                       'expected': {'classIds': classes, 'colors': colors, 'classCounts': counts}})
    return {'name': name, 'valuesX': values_x, 'valuesY': values_y, 'mask': mask,
            'alphaValues': alpha, 'palette': palette, 'frames': frames}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=HERE / 'fixtures')
    output = parser.parse_args().output_dir
    output.mkdir(parents=True, exist_ok=True)
    profiles = []
    moments = []
    for length, origin in [(11, 2**20), (257, -2**20), (2051, 2**24), (4103, 2**20)]:
        values = (origin + (np.arange(length) * 7 % 23) * (2 if origin == 2**24 else .25)).astype(np.float32)
        mask = np.where(np.arange(length) % 17, 7, 0)
        values[:3] = [np.nan, np.inf, -np.inf]
        # An entirely invalid first tile also exercises empty left operands in the merge.
        if length == 4103:
            mask[:2048] = 0
        name = f'offset-{origin}-rows-{length}'
        profiles.append(column.profile_case(name, [{'values': values, 'domain': [origin - 64, origin + 64]}], mask))
        selected = values[np.isfinite(values) & (mask != 0)].astype(float)
        series = pd.Series(selected)
        moments.append({'name': name, 'values': values, 'mask': mask,
                        'expected': [len(series), series.mean(), series.var(ddof=0) * len(series), series.min(), series.max()]})
    profiles.append(column.profile_case('offset-constant-many-tiles', [{'values': [2**24] * 4103}], [1] * 4103))
    write(output, 'offset-profiles', profiles, 'pandas.Series.var/std/describe; existing independent HLL reference')
    write(output, 'offset-moments', moments, 'pandas.Series.var(ddof=0)/mean/min/max')

    income = column.load('us-counties', 'medianHouseholdIncome')
    population = column.load('us-counties', 'population')
    states = column.load('us-counties', 'stateFips')
    joins = [join_case('counties-by-state', range(61), states, income, [1] * 61, [1] * len(states)),
             join_case('counties-masked-capacity', range(61), states, income, [0 if i % 7 == 0 else 7 for i in range(61)],
                       [0 if i % 5 == 0 else 3 for i in range(len(states))], capacity=17)]
    for bits in [32, 64]:
        high = 0 if bits == 32 else 1 << 40
        no_key = (1 << bits) - 1
        joins.append(join_case(f'duplicates-nulls-{bits}', [high, high + 1, high + 1, high + 2, high + 3, no_key, 0, high + 4],
                    [high + 1, high, high + 1, high + 2, no_key, high + 3, high + 3, high + 4],
                    [1.25, -2, 2.75, np.nan, 999, np.inf, -np.inf, 3],
                    [1, 7, 1, 1, 1, 1, 0, 1], [1, 1, 1, 1, 1, 1, 1, 0], bits, capacity=3))
    joins.append(join_case('no-matches', [NONE, 2], [NONE, 3], [4, 5], [1, 1], [1, 1]))
    write(output, 'key-join', joins, 'pandas.DataFrame.merge/groupby.agg; numpy.rint')
    bivariate = [bivariate_case('counties-all', income, population, [1] * len(income)),
                 bivariate_case('ties-masks-infinities', [-np.inf, -1, -0., 0., 1, 2, np.inf, np.nan, 0, 1],
                                [np.inf, 0, -0., 0., 2, -np.inf, 1, 0, np.nan, 1], [1] * 9 + [0])]
    write(output, 'bivariate', bivariate, 'numpy.searchsorted/bincount; exact palette lookup and dyadic alpha')

    seconds = column.load('poopdeck-earthquakes', 'timestamp')
    manifest = json.loads((column.DATA / 'poopdeck-earthquakes/manifest.json').read_text())
    epoch = manifest['properties']['timeOriginMs']
    start, end = int(seconds[20000]), int(seconds[40000])
    real = [time_case('usgs-all-float', seconds, start, end, real=True),
            time_case('usgs-all-int64', seconds.astype(np.int64) * 1000 + epoch,
                      start * 1000 + epoch, end * 1000 + epoch, 'word', real=True)]
    write(output, 'time-window-real', real, 'pandas.Series.between(inclusive=both)')
    cases = [time_case('float-nonfinite', [np.nan, -np.inf, -1, -0., 0., 1, 2, np.inf], 0, 1, capacity=2),
             time_case('float-ties', [-2, -1, 0, 0, 0, 1, 2], 0, 0, capacity=2),
             time_case('reversed', [-2, 0, 1, 2], 2, -2), time_case('empty-selection', [1, 2], 3, 4)]
    for origin in [-(1 << 40), (1 << 54)]:
        cases.append(time_case(f'int64-{origin}', [origin - 1, origin, origin, origin + 1, origin + 2], origin, origin + 1, 'word', capacity=2))
    write(output, 'time-window-edges', cases, 'pandas.Series.between(inclusive=both)')


if __name__ == '__main__':
    main()
