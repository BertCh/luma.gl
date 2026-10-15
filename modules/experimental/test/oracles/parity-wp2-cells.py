# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""H3 + pandas reference for aggregation, hierarchical rollup and outer comparison.

All 43,557 Chicago nature rows, original f32 coordinates, mapped by h3.latlng_to_cell
at resolution 10, then h3.cell_to_parent at 9/7/5. This tests PRE-KEYED H3 APIs,
not GPU point indexing. Values are the actual introduced-species indicator; masks
select all / research-grade / non-research-grade / empty observations. No thinning.
Input JSON is chunked only for file size; the GPU receives the entire concatenation.

https://uber.github.io/h3-py/api_quick.html
https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html
https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.zscore.html

Independent H3 validity/parent mapping; pandas count/sum/min/max and outer merge.
Fixed-point sums adapt the documented scale 65536, f32 product, half-even rounding
(np.rint), two's-complement words. Invalid keys, coarser keys, masked rows and
nonfinite values excluded. Sorted unsigned keys; smallest-key prefix on overflow.
Bounded pyramid rolls up only retained children and propagates overflow. Standardized
scores use scipy.stats.zscore(ddof=0) on the retained union (capacity-bounded API);
constant delta gives zero. Poisson scores use the documented delta/sqrt(total)
composite definition. Missing measures zero; zero denominator and tail derived
columns NaN. All ID/count/sum words and extrema exact; float outputs profile S.
"""
import hashlib
import json
import platform
from pathlib import Path
import h3
import numpy as np
import pandas as pd
import scipy
from scipy.stats import zscore

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[3] / 'examples/deck/spatial-analysis-showcase/public/data/chicago-nature'
VERSIONS = dict(python=platform.python_version(), h3=h3.__version__, numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__)
SCALE = 65536
RESOLUTIONS = [9, 7, 5]
REFERENCE = 'h3.cell_to_parent/is_valid_cell + pandas.groupby.agg/merge + scipy.stats.zscore'
HASHES = {name: hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in ['manifest.json', 'position.bin', 'introduced.bin', 'researchGrade.bin']}


def clean(value):
    if isinstance(value, dict): return {key: clean(entry) for key, entry in value.items()}
    if isinstance(value, list): return [clean(entry) for entry in value]
    if isinstance(value, float) and not np.isfinite(value): return None if np.isnan(value) else ('Infinity' if value > 0 else '-Infinity')
    return value


def write(suffix, payload):
    path = HERE / f'fixtures/parity-wp2-cells-{suffix}.json'
    data = dict(generator=Path(__file__).name, reference=REFERENCE, versions=VERSIONS,
                source=(dict(dataset='synthetic H3 pentagon descendants and invalid rows', rows=102) if suffix.startswith('edges') else dict(dataset='chicago-nature / iNaturalist contributors', hashes=HASHES, fullRows=43557)),
                conventions=dict(resolutions=RESOLUTIONS, sumScale=SCALE, input='f32 coordinates, H3 resolution 10; pre-keyed GPU input', masks=['all', 'research-grade', 'non-research-grade', 'empty']),
                tolerance=dict(profile='S', absolute=1e-6, relative=1e-4, exact='keys/counts/fixed-point-sum words/extrema/status'), **payload)
    path.write_text(json.dumps(clean(data), separators=(',', ':'), allow_nan=False)+'\n')
    assert path.stat().st_size < 1500000, path
    print(path.name, path.stat().st_size)


def words(values):
    return [word for value in values for word in [int(value) & 0xffffffff, (int(value) >> 32) & 0xffffffff]]


def aggregate(keys, values, mask, resolution):
    selected = [(key, float(value)) for key, value, enabled in zip(keys, values, mask)
                if enabled and np.isfinite(value) and h3.is_valid_cell(key) and h3.get_resolution(key) >= resolution]
    parents = [int(h3.cell_to_parent(key, resolution), 16) for key, _ in selected]
    numbers = np.array([value for _, value in selected], dtype=np.float32)
    data = pd.DataFrame(dict(key=pd.Series(parents, dtype='int64'), value=numbers.astype(float),
                             fixed=np.rint(numbers * np.float32(SCALE)).astype(np.int64)))
    return data.groupby('key', sort=True).agg(count=('value', 'size'), fixed=('fixed', 'sum'), minimum=('value', 'min'), maximum=('value', 'max'))


def rollup(table, resolution):
    data = table.reset_index()
    data['key'] = pd.Series([int(h3.cell_to_parent(format(int(key), 'x'), resolution), 16) for key in data.key], dtype='int64')
    return data.groupby('key', sort=True).agg(count=('count', 'sum'), fixed=('fixed', 'sum'), minimum=('minimum', 'min'), maximum=('maximum', 'max'))


def encode(table, required=None, overflow=False):
    return dict(cells=words(table.index), counts=table['count'].astype(int).tolist(), sums=words(table.fixed),
                sumValues=(table.fixed.astype(float)/SCALE).tolist(), minimums=table.minimum.tolist(), maximums=table.maximum.tolist(),
                count=len(table), requiredCount=len(table) if required is None else required, overflow=int(overflow))


def make_case(name, keys, values, masks):
    raw = [aggregate(keys, values, mask, RESOLUTIONS[0]) for mask in masks]
    full_capacity = max(map(len, raw)) + 3
    frames = []
    bounded = []
    for frame_index, table in enumerate(raw):
        levels = [table]
        for resolution in RESOLUTIONS[1:]: levels.append(rollup(levels[-1], resolution))
        # Independent direct input grouping verifies the unbounded hierarchical reference.
        for resolution, level in zip(RESOLUTIONS, levels):
            eligible = [enabled and h3.is_valid_cell(key) and h3.get_resolution(key) >= RESOLUTIONS[0] for key, enabled in zip(keys, masks[frame_index])]
            pd.testing.assert_frame_equal(level, aggregate(keys, values, eligible, resolution), check_dtype=False)
        frames.append(dict(name=['all','research','nonresearch','empty'][frame_index], levels=[encode(level) for level in levels]))
        limited_levels = []
        inherited = False
        for resolution in RESOLUTIONS:
            if resolution != RESOLUTIONS[0]: table = rollup(table, resolution)
            required = len(table)
            inherited |= required > 3
            table = table.iloc[:3]
            limited_levels.append(encode(table, required, inherited))
        bounded.append(dict(name=frames[-1]['name'], levels=limited_levels))
    write(name+'-expected', dict(name=name, capacity=full_capacity, frames=frames, boundedFrames=bounded))
    for start in range(0, len(keys), 8192):
        stop=min(start+8192, len(keys))
        write(f'{name}-input-{start//8192}', dict(name=name, start=start, keys=words(int(key,16) for key in keys[start:stop]), values=values[start:stop], masks=[mask[start:stop] for mask in masks]))
    # Outer union, including swapped / empty updates; inputs are independent reference tables.
    comparisons=[]
    for before_index, after_index in [(1,2),(2,1),(0,3),(3,0),(3,3)]:
        before, after = raw[before_index], raw[after_index]
        joined=before[['count','fixed']].merge(after[['count','fixed']], how='outer', left_index=True, right_index=True, suffixes=('_before','_after'), indicator=True).sort_index()
        presence=joined['_merge'].map({'left_only':1,'right_only':2,'both':3}).astype(int)
        joined=joined.drop(columns='_merge').fillna(0)
        for capacity in [max(len(joined)+3, 3), 3]:
            retained=joined.iloc[:capacity]
            for measure in ['count','sum']:
                column='count' if measure=='count' else 'fixed'
                divisor=1 if measure=='count' else SCALE
                first=retained[column+'_before'].to_numpy(float)/divisor
                second=retained[column+'_after'].to_numpy(float)/divisor
                delta=second-first
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio=np.where(first!=0, second/first, np.nan)
                    percent=np.where(first!=0, 100*delta/first, np.nan)
                    score=np.where(first+second>0,delta/np.sqrt(first+second),0) if measure=='count' else (zscore(delta,ddof=0) if len(delta) and np.std(delta)>0 else np.zeros(len(delta)))
                comparisons.append(dict(before=before_index, after=after_index, capacity=capacity, measure=measure,
                    expected=dict(cells=words(retained.index), presence=presence.iloc[:capacity].tolist(), before=first.tolist(), after=second.tolist(), delta=delta.tolist(), ratio=ratio.tolist(), percentChange=percent.tolist(), zScore=score.tolist(), count=len(retained), requiredCount=len(joined), overflow=int(len(joined)>capacity))))
    write(name+'-compare', dict(name=name, comparisons=comparisons))

positions=np.fromfile(DATA/'position.bin', '<f4').reshape(-1,2)
introduced=np.fromfile(DATA/'introduced.bin','u1').astype(float).tolist()
research=np.fromfile(DATA/'researchGrade.bin','u1').astype(int).tolist()
keys=[h3.latlng_to_cell(float(latitude),float(longitude),10) for longitude,latitude in positions]
make_case('nature', keys, introduced, [[1]*len(keys),research,[1-value for value in research],[0]*len(keys)])

# All twelve pentagons; valid center/neighbor descendants, mixed finer resolutions,
# signed carry/borrow, ties-to-even quantization, invalid/coarse cells and nonfinite rows.
edge_keys=[]
for pentagon in h3.get_pentagons(0):
    center=h3.cell_to_center_child(pentagon,9)
    edge_keys.extend([center,h3.cell_to_center_child(center,15),*sorted(h3.grid_disk(center,1))])
edge_values=np.resize(np.array([100000.,-99999.75,2**-17,3*2**-17,-3*2**-17,0.,4.25],dtype=np.float32), len(edge_keys)).astype(float).tolist()
edge_keys += ['0','ffffffffffffffff',h3.get_pentagons(0)[0],edge_keys[0],edge_keys[0],edge_keys[0]]
edge_values += [1.,2.,3.,float('nan'),float('inf'),float('-inf')]
masks=[[1]*len(edge_keys),[int(index%2==0) for index in range(len(edge_keys))],[int(index%2==1) for index in range(len(edge_keys))],[0]*len(edge_keys)]
make_case('edges',edge_keys,edge_values,masks)
