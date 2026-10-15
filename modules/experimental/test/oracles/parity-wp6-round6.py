# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent pandas/NumPy references for WP6 round 6; execute with Python -I.

Full shipped OSM NYC history: all 400000 rows (the showcase itself is a seeded
sample of its upstream history; no further sampling here). Category=kind;
value=f32(contributor rank / 16), score=2*value+.5. Dyadic arithmetic is exact
for these ranks. Explicit synthetic nullable sidecars drop every 97th key and
89th value; nonzero sidecars include 7, exercising boolean validity semantics.
Pandas query selects category < the supplied bound with null keys excluded;
null values DO NOT remove rows. Dense groups include unused category slots.
Group/global metric contributions require finite score and valid value.
Count means selected rows; sums of empty groups=0 with validity=0; other empty
metrics=NaN. NumPy histogram uses left-closed/right-open bins, final upper
endpoint included. All parameters/values are f32-rounded before f64 reference.
Complete arrays are losslessly base64 encoded, split only for the 1.5MB cap.
Original record-batch boundaries, including empty batches, remain explicit.
No spatial transform/randomness applies; source hashes and versions recorded.

Moment triage uses pandas var on f32-rounded inputs, plus translation by an
exact f32 origin. At 2^24 quarter-unit increments intentionally round at input;
reference variance is always computed AFTER that rounding. Tile references
are independently pandas reductions, not a port of the Chan/Welford shader.
"""
import argparse
import base64
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
specification = importlib.util.spec_from_file_location('column_reference', HERE / 'parity-wp6-column-analysis.py')
column = importlib.util.module_from_spec(specification)
specification.loader.exec_module(column)


def pack(values, dtype):
    values = np.asarray(values, dtype=dtype)
    return {'dtype': dtype, 'length': len(values), 'base64': base64.b64encode(values.tobytes()).decode()}


def write(output, name, payload):
    body = {'generator': Path(__file__).name, 'versions': column.VERSIONS,
            'sources': column.SOURCES.copy(), 'conventions': __doc__, **payload}
    path = output / f'parity-wp6-{name}.json'
    path.write_text(json.dumps(column.encode(body), allow_nan=False, separators=(',', ':')) + '\n')
    assert path.stat().st_size < 1500000, (path, path.stat().st_size)
    print(path.name, path.stat().st_size)


def summarize(frame, group_count):
    grouped = frame.groupby('category').size().reindex(range(group_count), fill_value=0)
    finite = frame[frame.valueValid & np.isfinite(frame.score)]
    reduced = finite.groupby('category').score.agg(['sum', 'min', 'max', 'mean', 'count']).reindex(range(group_count))
    groups = {'category': list(range(group_count)), 'count': grouped.to_numpy(),
              'sum': reduced['sum'].fillna(0).to_numpy(), 'minimum': reduced['min'].to_numpy(),
              'maximum': reduced['max'].to_numpy(), 'mean': reduced['mean'].to_numpy(),
              'validity': (reduced['count'].fillna(0).to_numpy() > 0).astype(int)}
    series = finite.score
    global_values = {'count': [len(frame)], 'sum': [series.sum()], 'minimum': [series.min()],
                     'maximum': [series.max()], 'mean': [series.mean()], 'validity': [int(len(series) > 0)]}
    return groups, global_values, series.to_numpy()


def make_case(name, categories, values, category_validity, value_validity, batch_lengths, source_offsets, group_count=8):
    values = np.asarray(values, np.float32).astype(float)
    categories = np.asarray(categories, np.uint32)
    frame = pd.DataFrame({'category': categories, 'value': values,
                          'categoryValid': np.asarray(category_validity) != 0,
                          'valueValid': np.asarray(value_validity) != 0})
    frame['score'] = (frame.value * 2 + .5).astype(np.float32).astype(float)
    edges = [-4, 0, .5, 1, 2, 4, 16, 128, 512, 2048]
    equal_edges = np.linspace(-4, 2044, 17)
    frames = []
    for bound in [8, 3, 0]:
        selected = frame.query('categoryValid and category < @bound')
        groups, global_values, accepted = summarize(selected, group_count)
        selection = frame.index.isin(selected.index).astype(np.uint8)
        frames.append({'bound': bound, 'mask': selection, 'groups': groups, 'global': global_values,
                       'histogram': np.histogram(accepted, bins=edges)[0],
                       'equalHistogram': np.histogram(accepted, bins=equal_edges)[0]})
    batches = []
    start = 0
    for length, source_offset in zip(batch_lengths, source_offsets):
        stop = start + length
        batches.append({'length': length, 'sourceOffset': source_offset,
                        'category': pack(categories[start:stop], 'u1'),
                        'values': pack(values[start:stop], '<f4'),
                        'categoryValidity': pack(category_validity[start:stop], 'u1'),
                        'valueValidity': pack(value_validity[start:stop], 'u1'),
                        'score': pack(frame.score.to_numpy()[start:stop], '<f4'),
                        'frames': [{'mask': pack(item['mask'][start:stop], 'u1'),
                                    'ids': pack(np.flatnonzero(item['mask'][start:stop]) + source_offset, '<u4')}
                                   for item in frames]})
        start = stop
    assert start == len(values)
    for item in frames: del item['mask']
    return {'name': name, 'groupCount': group_count, 'edges': edges,
            'domain': [-4, 2044], 'binCount': 16, 'frames': frames}, batches


def moment_case(name, values, mask):
    values = np.asarray(values, np.float32)
    mask = np.asarray(mask, np.uint32)
    def summarize_values(start, stop):
        selected = values[start:stop][(mask[start:stop] != 0) & np.isfinite(values[start:stop])].astype(float)
        series = pd.Series(selected)
        return [len(series), series.mean(), series.var(ddof=0) * len(series), series.min(), series.max()] if len(series) else [0, 0, 0, 0, 0]
    return {'name': name, 'values': values, 'mask': mask, 'expected': summarize_values(0, len(values)),
            'tiles': [summarize_values(start, start+2048) for start in range(0, len(values), 2048)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=HERE / 'fixtures')
    output = parser.parse_args().output_dir
    output.mkdir(parents=True, exist_ok=True)
    categories = column.load('poopdeck-osm-nyc', 'kind')
    values = column.load('poopdeck-osm-nyc', 'contributor').astype(np.float32) / 16
    rows = np.arange(len(values))
    lengths = [1, 255, 0, 50000, 50000, 50000, 50000, 50000, 50000, 50000, 49744]
    offsets = np.cumsum([1000] + lengths[:-1]).tolist()
    scene, batches = make_case('osm-full-nullable-batches', categories, values,
                               np.where(rows % 97, 7, 0), np.where(rows % 89, 7, 0), lengths, offsets)
    scene['batchFiles'] = []
    for index, batch in enumerate(batches):
        name = f'dataframe-osm-batch-{index}'
        scene['batchFiles'].append(f'parity-wp6-{name}.json')
        write(output, name, {'batch': batch})
    synthetic, synthetic_batches = make_case('nulls-nonfinite-boundaries',
        [0, 1, 2, 3, 4, 5, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2],
        [-2.25, -.25, 0, .25, .75, 1.75, 7.75, 63.75, 255.75, 1023.75,
         np.nan, np.inf, -np.inf, 1, 1, 3, -0., 0.],
        [7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
        [3, 0, 7, 8], [40, 100, 200, 400])
    synthetic['batches'] = synthetic_batches
    write(output, 'dataframe-queries', {'reference': 'pandas.DataFrame.query/groupby.agg; pandas.Series.agg; numpy.histogram',
                                       'cases': [scene, synthetic]})
    # Reconstruct former fixtures from their exact serialized f32 inputs, including NaN.
    original = json.loads((HERE/'fixtures/parity-wp6-offset-moments.json').read_text())
    moments = []
    for entry in original['cases']:
        values = np.array([np.nan if item is None else float(item) for item in entry['values']], np.float32)
        moments.append(moment_case(entry['name'], values, entry['mask']))
        finite = values[np.isfinite(values)]
        moments.append(moment_case(entry['name']+'-translated', values - finite[0], entry['mask']))
    values = (2**24 + (np.arange(2051) * 7 % 23) * .25).astype(np.float32)
    moments.append(moment_case('quarter-deltas-rounded-at-2pow24', values, np.ones(len(values), int)))
    write(output, 'moment-diagnostics', {'reference': 'pandas.Series.var(ddof=0)/mean/min/max on f32-rounded inputs', 'cases': moments})


if __name__ == '__main__':
    main()
