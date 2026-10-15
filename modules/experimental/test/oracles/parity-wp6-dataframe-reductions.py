# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent pandas/NumPy/SciPy references for WP6 Round 3 (python -I).

Group statistics: full Chicago tract incomes/population by community area;
finite-only values, mask != 0, unsigned keys, all-ones sentinel, ascending
keys, smallest keys retained at capacity. pandas groupby supplies counts and
reductions; scipy.stats supplies biased Fisher skew/excess kurtosis, NumPy
linear quantiles. Sample/population ddof is explicit. Singleton sample
variance is NaN but z score is zero. Mode ties choose smallest finite value.
Fixed-point sums follow the public representation: np.rint(f32(value*scale)),
then pandas integer sum; means use that sum / scale / count. Moments use the
original f32-rounded values in f64. Empty tails and dense gaps are explicit.
No spatial CRS, RNG, or temporal grouping is involved in this class.

Temporal: full 77,231 USGS showcase events, original f32 magnitude, category
mag_band as cell, and exact stored seconds since manifest timeOriginMs.
Word mode uses UTC epoch milliseconds, 7-day half-open bins; float mode uses
f32 relative seconds and exact dyadic widths. pandas.cut with an IntervalIndex
assigns bins (left-closed/right-open); groupby min/max/idxmin/idxmax supplies
reductions, with original first row winning timestamp ties. Only NaN values
are missing; infinities remain. Float subtraction and edges are rounded to
f32 per the public interval contract BEFORE pandas classification. Sorted and
unsorted GPU paths consume the identical sorted real rows. Synthetic word
cases include negative epoch times, nanoseconds beyond 2^53, and ties; no
calendar/timezone inference, DST, resampling or rolling API is claimed.

Histogram: numpy.histogram of all real magnitudes and synthetic float/signed/
unsigned data. Finite-only, nonzero mask, half-open internal edges, inclusive
last edge. Auto extent includes masked rows, per API. Equal domains are the
API's documented degenerate bin-zero convention, adapted via count_nonzero.
Exact dyadic regular edges avoid f32 ratio ambiguity. All bins are compared
exactly. Chunking/mask seams do not change the independent reference.

Every fixture includes dependency versions and source hashes. Full-source
input arrays are losslessly base64 encoded little endian, not sampled.
S is unchanged for floats; E for keys, words, counts, masks, histograms and
all temporal extrema/first/last outputs. No reference imports local CPU/WGSL.
"""
import argparse
import base64
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DATA = ROOT / 'examples/deck/spatial-analysis-showcase/public/data'
VERSIONS = {'python': platform.python_version(), 'numpy': np.__version__,
            'pandas': pd.__version__, 'scipy': scipy.__version__}
SOURCES = {}


def load(dataset, name, dtype):
    path = DATA / dataset / name
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return np.fromfile(path, dtype=dtype)


def encode(value):
    if isinstance(value, dict):
        return {key: encode(entry) for key, entry in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [encode(entry) for entry in value]
    if isinstance(value, (np.integer,)):
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


def write(output, topic, content, reference):
    path = output / f'parity-wp6-{topic}.json'
    path.write_text(json.dumps(encode({'reference': reference, 'versions': VERSIONS,
                                      'sources': SOURCES.copy(), 'generator': Path(__file__).name,
                                      'conventions': {'inputFloats': 'float32 rounded before float64 reference',
                                      'selection': 'nonzero mask', 'randomness': 'none',
                                      'tolerance': 'S unchanged; E counts/keys/words/histogram/temporal'}, **content}), allow_nan=False,
                               separators=(',', ':')) + '\n')
    assert path.stat().st_size <= 1500000, (path, path.stat().st_size)
    print(path.name, path.stat().st_size)


def pack(values, dtype):
    raw = np.asarray(values, dtype=dtype).tobytes()
    return {'dtype': dtype, 'length': len(values), 'sha256': hashlib.sha256(raw).hexdigest(),
            'base64': base64.b64encode(raw).decode('ascii')}


def key_words(keys, bits):
    return [word for key in keys for word in ([int(key) & 0xffffffff, int(key) >> 32]
                                             if bits == 64 else [int(key)])]


def group_case(name, keys, columns, mask, capacity, dense=0, bits=32, scale=65536):
    keys = [int(key) for key in keys]
    columns = [np.asarray(values, dtype=np.float32).astype(float) for values in columns]
    fractions = np.asarray([0, .1, .25, .5, .75, .9, 1, -.25, 1.25, np.nan], np.float32).astype(float)
    frame = pd.DataFrame({'key': pd.Series(keys, dtype=object), 'mask': mask})
    valid = (frame['mask'] != 0) & (frame['key'] != (1 << bits) - 1)
    if dense:
        valid &= frame['key'] < dense
    grouped = frame[valid].groupby('key', sort=True)
    group_keys = list(range(dense)) if dense else list(grouped.size().index)
    retained = group_keys[:capacity]
    output_keys = retained + [(1 << bits) - 1] * (capacity - len(retained))
    result = {}
    for variance in ['sample', 'population']:
        output_columns = []
        for values in columns:
            finite_frame = frame[valid & np.isfinite(values)].copy()
            finite_frame['value'] = values[finite_frame.index]
            finite_frame['fixed'] = [int(value) for value in np.rint(
                (finite_frame['value'].to_numpy().astype(np.float32) * np.float32(scale)).astype(float))]
            value_groups = finite_frame.groupby('key', sort=True)
            aggregate = value_groups.agg(counts=('value', 'count'), sumInteger=('fixed', 'sum'),
                                         minimums=('value', 'min'), maximums=('value', 'max'),
                                         medians=('value', 'median'), uniqueCounts=('value', 'nunique'))
            output = {field: [] for field in ['counts', 'sums', 'sumValues', 'means', 'minimums',
                      'maximums', 'variances', 'standardDeviations', 'skewness', 'kurtosis',
                      'medians', 'percentiles', 'modes', 'uniqueCounts']}
            z_scores = np.full(len(keys), np.nan)
            for key in output_keys:
                exists = key in aggregate.index and key in retained
                series = value_groups.get_group(key)['value'] if exists else pd.Series([], dtype=float)
                count = len(series)
                sum_integer = int(aggregate.loc[key, 'sumInteger']) if exists else 0
                mean = sum_integer / scale / count if count else np.nan
                deviation = series.std(ddof=1 if variance == 'sample' else 0)
                output['counts'].append(count)
                output['sums'].extend([sum_integer & 0xffffffff, (sum_integer >> 32) & 0xffffffff])
                output['sumValues'].append(sum_integer / scale)
                output['means'].append(mean)
                for field in ['minimums', 'maximums', 'medians', 'uniqueCounts']:
                    output[field].append(aggregate.loc[key, field] if exists else (0 if field == 'uniqueCounts' else np.nan))
                output['variances'].append(series.var(ddof=1 if variance == 'sample' else 0))
                output['standardDeviations'].append(deviation)
                varying = count > 1 and series.nunique() > 1
                output['skewness'].append(stats.skew(series, bias=True) if varying else np.nan)
                output['kurtosis'].append(stats.kurtosis(series, fisher=True, bias=True) if varying else np.nan)
                output['modes'].append(series.mode().iloc[0] if count else np.nan)
                output['percentiles'].extend([np.quantile(series, fraction, method='linear')
                    if count and 0 <= fraction <= 1 else np.nan for fraction in fractions])
                if count:
                    z_scores[series.index] = ((series - mean) / deviation if deviation > 0 else 0)
            output['zScores'] = z_scores
            output_columns.append(output)
        result[variance] = output_columns
    return {'name': name, 'keys': key_words(keys, bits), 'keyBits': bits, 'columns': columns,
            'mask': list(mask), 'capacity': capacity, 'keyCount': dense, 'sumScale': scale,
            'fractions': fractions, 'expected': {'keys': key_words(output_keys, bits),
            'counts': [int(grouped.size().get(key, 0)) if key in retained else 0 for key in output_keys],
            'status': [len(retained), int(len(group_keys) > capacity), len(group_keys)], 'columns': result}}


def temporal_case(name, cells, timestamps, values, mask, cell_count, bucket_count,
                  origin, width, mode, capacity=None, source=False):
    values = np.asarray(values, np.float32).astype(float)
    cells = np.asarray(cells, np.uint32)
    timestamps = np.asarray(timestamps, np.int64 if mode == 'word' else np.float32)
    if mode == 'word':
        boundaries = np.arange(bucket_count + 1, dtype=np.int64) * width + origin
        interval_values = timestamps
    else:
        boundaries = (np.arange(bucket_count + 1) * np.float32(width)).astype(np.float32).astype(float)
        interval_values = (timestamps - np.float32(origin)).astype(np.float32).astype(float)
    intervals = pd.IntervalIndex.from_breaks(boundaries, closed='left')
    buckets = pd.cut(interval_values, intervals).codes
    valid = (np.asarray(mask) != 0) & (cells < cell_count) & ~np.isnan(values) & (buckets >= 0)
    frame = pd.DataFrame({'slot': cells.astype(np.int64) * bucket_count + buckets,
                          'time': timestamps, 'value': values})[valid]
    grouped = frame.groupby('slot', sort=True)
    slots = pd.RangeIndex(cell_count * bucket_count)
    aggregate = grouped['value'].agg(['count', 'min', 'max']).reindex(slots)
    # idxmin/idxmax return the first original index for tied times; pandas.last would not.
    first = frame.loc[grouped['time'].idxmin(), ['slot', 'value']].set_index('slot')['value'].reindex(slots)
    last = frame.loc[grouped['time'].idxmax(), ['slot', 'value']].set_index('slot')['value'].reindex(slots)
    occupied = grouped.size().index.to_numpy()
    capacity = len(slots) if capacity is None else capacity
    result = {'counts': aggregate['count'].fillna(0).astype(int).to_numpy(),
              'minimums': aggregate['min'].to_numpy(), 'maximums': aggregate['max'].to_numpy(),
              'first': first.to_numpy(), 'last': last.to_numpy(), 'ids': occupied[:capacity],
              'status': [min(capacity, len(occupied)), int(len(occupied) > capacity), len(occupied)]}
    return {'name': name, 'mode': mode, 'source': source, 'cellCount': cell_count,
            'bucketCount': bucket_count, 'origin': str(origin), 'width': str(width),
            'capacity': capacity, 'cells': [] if source else cells,
            'timestamps': [] if source else ([str(int(value)) for value in timestamps] if mode == 'word' else timestamps.tolist()),
            'values': [] if source else values, 'mask': [] if source else list(mask), 'expected': result}


def histogram_case(name, values, mask, bins, domain=None, edges=None, scalar_format='float32', source=False):
    dtype = {'float32': np.float32, 'uint32': np.uint32, 'sint32': np.int32}[scalar_format]
    values = np.asarray(values, dtype=dtype)
    selected = values[np.isfinite(values) & (np.asarray(mask) != 0)]
    if edges is not None:
        expected = np.histogram(selected, bins=np.asarray(edges))[0]
    else:
        actual_domain = ([float(np.min(values[np.isfinite(values)])), float(np.max(values[np.isfinite(values)]))]
                         if domain == 'auto' else domain)
        if actual_domain[0] == actual_domain[1]:
            expected = np.zeros(bins, dtype=int)
            expected[0] = np.count_nonzero(selected == actual_domain[0])
        else:
            expected = np.histogram(selected.astype(float), bins=bins, range=actual_domain)[0]
    return {'name': name, 'source': source, 'format': scalar_format, 'values': [] if source else values,
            'mask': [] if source else list(mask), 'bins': bins,
            **({'edges': edges} if edges is not None else {'domain': domain, 'resolvedDomain': actual_domain}), 'expected': expected}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=HERE / 'fixtures')
    output = parser.parse_args().output_dir
    output.mkdir(parents=True, exist_ok=True)
    tract_manifest = DATA / 'chicago-tracts/manifest.json'
    SOURCES[str(tract_manifest.relative_to(ROOT))] = hashlib.sha256(tract_manifest.read_bytes()).hexdigest()
    keys = load('chicago-tracts', 'communityArea.bin', 'u1')
    incomes = load('chicago-tracts', 'perCapitaIncome.bin', '<f4')
    population = load('chicago-tracts', 'population.bin', '<f4')
    assert len(keys) == len(incomes) == len(population) == 791
    synthetic_keys = [5, 2, 5, 2, 7, 7, 9, 0, 0, 0, 3, 3, 4, 4, 4, 0xffffffff, 12, 12, 18]
    synthetic_values = [1, -2, 3, -2, 4, 4, 99, -0., 0., 2, np.nan, np.inf, -np.inf, np.nan, np.inf, 10, .00000762939453125, .00002288818359375, -99]
    synthetic_second = [9, 2, 1, 4, 8, 8, np.nan, -3, -3, 1, 5, 7, 2, 3, 4, 99, .25, -.25, 0]
    mask = [1] * len(synthetic_keys)
    mask[6] = 0
    mask[10] = 7
    group_cases = [group_case('chicago-all-tracts-compact', keys, [incomes, population], np.ones(791, int), 82),
                   group_case('chicago-all-tracts-dense', keys, [incomes, population], np.ones(791, int), 80, dense=80)]
    for name, capacity, dense in [('edges-compact', 15, 0), ('edges-capacity', 3, 0), ('edges-dense', 15, 15)]:
        group_cases.append(group_case(name, synthetic_keys, [synthetic_values, synthetic_second], mask, capacity, dense))
    large_keys = [((1 << 53) + key * (1 << 32) + (index % 2) + ((1 << 63) if key % 2 else 0)) if key != 0xffffffff else (1 << 64) - 1
                  for index, key in enumerate(synthetic_keys)]
    group_cases.append(group_case('unsigned64-carries', large_keys, [synthetic_values, synthetic_second], mask, 20, bits=64))
    group_cases.append(group_case('all-masked', synthetic_keys, [synthetic_values], np.zeros(len(mask), int), 4))
    write(output, 'group-statistics', {'cases': group_cases}, 'pandas.DataFrame.groupby.agg; scipy.stats.skew/kurtosis; numpy.quantile(linear)/rint')

    SOURCES.clear()
    magnitude = load('poopdeck-earthquakes', 'magnitude.bin', '<f4')
    seconds = load('poopdeck-earthquakes', 'time.bin', '<u4')
    cells = load('poopdeck-earthquakes', 'mag_band.bin', 'u1').astype(np.uint32)
    manifest_path = DATA / 'poopdeck-earthquakes/manifest.json'
    SOURCES[str(manifest_path.relative_to(ROOT))] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest = json.loads(manifest_path.read_text())
    epoch = manifest['properties']['timeOriginMs']
    assert len(magnitude) == len(seconds) == len(cells) == 77231
    assert np.all(np.diff(seconds.astype(np.int64)) >= 0)
    write(output, 'earthquake-input', {'rowCount': len(magnitude), 'timeOriginMs': epoch,
          'magnitude': pack(magnitude, '<f4'), 'seconds': pack(seconds, '<u4'), 'cells': pack(cells, '<u4')},
          'Original full showcase USGS inputs; lossless little-endian bytes')
    temporal_cases = [temporal_case('usgs-all-word-weekly', cells, seconds.astype(np.int64) * 1000 + epoch,
        magnitude, np.ones(len(cells), int), 5, 262, epoch, 604800000, 'word', source=True),
        temporal_case('usgs-all-float-dyadic', cells, seconds, magnitude, np.ones(len(cells), int),
        5, 152, 0, 1048576, 'float', source=True)]
    edge_cells = [0, 0, 0, 0, 1, 1, 1, 1, 0xffffffff, 3, 2, 2, 2, 0, 0, 1, 1]
    edge_times = [0, 2, 2, 1, -1, 0, 3, 4, 0, 0, 7, 8, 7, 2, 2, 6, 6]
    edge_values = [10, 20, 99, -2, 5, 6, np.nan, np.inf, 9, 9, -np.inf, 40, 8, -99, 7, 3, -3]
    edge_mask = [1] * len(edge_times)
    edge_mask[13] = 0
    for mode, origin in [('float', 0), ('word', -1000), ('word', 1577836800000000000)]:
        times = np.asarray(edge_times, np.int64) + origin
        temporal_cases.append(temporal_case(f'edges-{mode}-{origin}', edge_cells, times, edge_values,
            edge_mask, 3, 4, origin, 2, mode, capacity=4))
    decimal_width = np.float32(.1)
    decimal_edges = (np.arange(5) * decimal_width).astype(np.float32)
    # Avoid subnormals at zero: WGSL implementations may flush them to zero.
    decimal_times = np.array([np.nextafter(edge, np.float32(-np.inf)) if edge else -.01 for edge in decimal_edges] +
                             list(decimal_edges) + [np.nextafter(edge, np.float32(np.inf)) if edge else .01 for edge in decimal_edges], np.float32)
    temporal_cases.append(temporal_case('float-decimal-boundaries', np.zeros(15, int), decimal_times,
        np.arange(15) / 4, np.ones(15, int), 2, 4, 0, float(decimal_width), 'float'))
    temporal_cases.append(temporal_case('float-nan-time-empty', [0, 0], [np.nan, 1], [2, 3], [1, 0], 2, 2, 0, 1, 'float'))
    write(output, 'temporal-reduction', {'cases': temporal_cases}, 'pandas.cut(IntervalIndex closed=left); DataFrame.groupby.agg/idxmin/idxmax')

    histogram_cases = []
    for bins in [16, 256, 1024]:
        histogram_cases.append(histogram_case(f'usgs-all-{bins}-bins', magnitude, np.ones(len(cells), int), bins, [0, 16], source=True))
    histogram_cases.append(histogram_case('usgs-all-irregular', magnitude, np.ones(len(cells), int), 7,
                                        edges=[0, 4, 4.5, 5, 6, 7, 8, 10], source=True))
    values = [-np.inf, -1, -0., 0, np.nextafter(np.float32(1), np.float32(0)), 1, 1.5, 2, 3, 4, np.nan, np.inf, 7, 8]
    masks = [1, 1, 1, 0, 1, 7, 1, 1, 1, 1, 1, 1, 0, 1]
    histogram_cases.extend([
        histogram_case('float-edges', values, masks, 4, [0, 4]),
        histogram_case('float-auto-unmasked-extent', values, masks, 9, 'auto'),
        histogram_case('float-irregular', values, masks, 4, edges=[-1, 0, 1, 3, 8]),
        histogram_case('float-degenerate', [7, 7, 8, np.nan], [1, 1, 1, 1], 5, [7, 7]),
        histogram_case('all-masked', values, [0] * len(values), 4, [0, 4]),
        histogram_case('empty', [], [], 4, [0, 4]),
        histogram_case('uint32-full-range', [0, 1, 1073741823, 1073741824, 2147483647, 2147483648, 4294967294, 4294967295], [1] * 8, 4, [0, 4294967295], scalar_format='uint32'),
        histogram_case('sint32-full-range', [-2147483648, -2147483647, -1073741825, -1073741824, -1, 0, 1073741823, 1073741824, 2147483647], [1] * 9, 4, [-2147483648, 2147483647], scalar_format='sint32')])
    seam_values = (np.arange(513) % 17) / 4
    seam_masks = np.where(np.arange(513) % 5 == 0, 0, 7)
    seam_masks[255:258] = [0, 9, 0]
    for bins in [16, 512]:
        histogram_cases.append(histogram_case(f'masked-chunk-seams-513-{bins}-bins',
            seam_values, seam_masks, bins, [0, 4]))
    # Independent interval regression for the Round-3 GPU failure. All edges and
    # widths here are exact dyadics. nextDown(1) belongs below 1 even though f32
    # subtraction from -1 rounds its offset to exactly 2. Keep E; do not waive
    # adjacent-bin errors. Test both local and global atomics, explicit/auto
    # domains, and all edge neighbors (excluding nonportable subnormals at 0).
    for bins, domain in [(9, [-1, 8]), (512, [-1, 511]), (16, [-1.5, 2.5])]:
        boundaries = np.linspace(*domain, bins + 1).astype(np.float32)
        neighbors = np.concatenate([np.nextafter(boundaries[boundaries != 0], np.float32(-np.inf)),
                                    boundaries,
                                    np.nextafter(boundaries[boundaries != 0], np.float32(np.inf))])
        histogram_cases.append(histogram_case(f'rounded-offset-edge-neighbors-{bins}', neighbors,
            np.ones(len(neighbors), int), bins, domain))
    write(output, 'histogram', {'cases': histogram_cases}, 'numpy.histogram; numpy.count_nonzero (documented degenerate domain)')


if __name__ == '__main__':
    main()
