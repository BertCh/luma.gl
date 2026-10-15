# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Final raster parity: independent library calls, not a TS/WGSL translation.

Sieve remove uses skimage.morphology.remove_small_objects on integer labels
(max_size=minimumPixels-1, preserving IDs). Full global Dixie labels come from
verified independent Round 6 fixtures. Merge uses rasterio.features.sieve with
background masked; synthetic patches have a unique eligible large neighbor,
no small chains or tied target sizes. GDAL's leave-isolated-small behavior and
its traversal tie rule differ from our API, so those modes are NOT claimed.
Targets are extracted from the reference output; counts use numpy.unique.
https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.remove_small_objects
https://rasterio.readthedocs.io/en/stable/api/rasterio.features.html#rasterio.features.sieve

Statistics uses NumPy float64 sum/mean/min/max on f32 input/calibration. Raw
NaN/Inf/sentinel and mask=0 are excluded; empty result is API-defined zero.
This class has NO histogram, variance or quantile API. 512x512 B08_after tile
from full 750x750 decode; calibrated values in DN/16-1 (dyadic test units, not
physical reflectance). Repeat with all-invalid then restored input buffers.
https://numpy.org/doc/stable/reference/generated/numpy.mean.html

Overlay: numpy.clip linear maps and numpy.digitize class maps; numpy.sum of
weighted values, divided by sum(abs(weights)) over contributing layers when
normalized. This differs from numpy.average's signed-weight denominator.
All calculations f64 after f32 input rounding. Exact dyadic linear endpoints
avoid introducing a separate CPU parameter-reciprocal rounding convention.
NaN/sentinel layers skip or propagate; restricted NaN classes always reject,
even at zero weight. Infinities are valid inputs and clamp to finite endpoints.
Zero total absolute weight is nodata only when normalizing. Tables use left
or right closure, repeated breaks, active prefixes and restricted classes.
Real inputs: four Dixie B08/B12 before/after bands, raw zero -> -9999, fixed
512x512 tile in eight pointwise partitions, no halo needed. These DN suitability
scores are a numerical test, not an ecological interpretation.
https://numpy.org/doc/stable/reference/generated/numpy.digitize.html
https://numpy.org/doc/stable/reference/generated/numpy.sum.html
"""
import argparse
import base64
import gzip
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import rasterio
from rasterio.features import sieve
import skimage
from skimage.morphology import remove_small_objects

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', type=Path, default=HERE/'fixtures')
OUTPUT = parser.parse_args().output_dir
OUTPUT.mkdir(parents=True, exist_ok=True)
VERSIONS = dict(python=platform.python_version(), numpy=np.__version__, rasterio=rasterio.__version__, skimage=skimage.__version__)


def pack(values):
    numbers = [None if np.isnan(value) else 'Infinity' if value == np.inf else '-Infinity' if value == -np.inf else float(value) for value in np.asarray(values).ravel()]
    return base64.b64encode(gzip.compress(json.dumps(numbers, separators=(',', ':')).encode(), mtime=0)).decode()


def unpack(value):
    return np.array(json.loads(gzip.decompress(base64.b64decode(value))), dtype=float)


def source(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(name, cases, sources):
    path = OUTPUT/f'parity-wp4-{name}.json'
    path.write_text(json.dumps(dict(versions=VERSIONS, sources=sources, cases=cases), separators=(',', ':'), allow_nan=False)+'\n')
    assert path.stat().st_size < 1_500_000, (name, path.stat().st_size)
    print(path.name, path.stat().st_size, [(case['name'], len(case['frames'])) for case in cases])


def sieve_case(name, labels, capacity, mode, connectivity, thresholds):
    labels = np.asarray(labels, dtype=np.int32)
    frames = []
    unique, first_indices, counts = np.unique(labels, return_index=True, return_counts=True)
    for threshold in thresholds:
        result = (remove_small_objects(labels, max_size=threshold-1) if mode == 'remove' else
                  sieve(labels, size=threshold, mask=labels != 0, connectivity=connectivity))
        targets = np.zeros(capacity, dtype=np.uint32)
        nonzero = unique != 0
        targets[unique[nonzero]-1] = result.ravel()[first_indices[nonzero]]
        assert np.array_equal(np.concatenate(([0], targets))[labels], result), 'Reference patches must not split'
        frames.append(dict(minimumPixels=threshold, expected=dict(labels=pack(result), patchTargets=pack(targets),
            sievedCount=pack([np.sum((unique != 0) & (counts < threshold))]))))
    return dict(name=name, reference='skimage.morphology.remove_small_objects; numpy.unique' if mode == 'remove' else 'rasterio.features.sieve; numpy.unique',
        width=labels.shape[1], height=labels.shape[0], labels=pack(labels), capacity=capacity,
        mode=mode, connectivity=connectivity, frames=frames)


for connectivity in [4, 8]:
    path = HERE/f'fixtures/parity-wp4-patches-real-{connectivity}.json'
    parent = json.loads(path.read_text())
    for entry in parent['sources']:
        assert source(ROOT/entry['path'])['sha256'] == entry['sha256']
    scene = parent['cases'][0]
    labels = unpack(scene['labels']).reshape(750, 750).astype(np.int32)
    write(f'sieve-real-{connectivity}', [sieve_case(f'dixie-global-{connectivity}', labels, scene['capacity'], 'remove', connectivity, [1, 2, 10, 100, 2])], [source(path), *parent['sources']])

labels = np.ones((17, 67), dtype=np.int32)
labels[:, 45:] = 2
labels[2:4, 15:17] = 3
labels[7:10, 31:34] = 4
labels[12, 55] = 5
labels[0, :5] = 0
merge_cases = [sieve_case(f'unique-large-target-{connectivity}', labels, 7, 'merge', connectivity, [1, 4, 5, 9, 10, 1]) for connectivity in [4, 8]]
# Diagonal-only adjacency for label 3: connectivity distinguishes its two large targets.
diagonal = np.array([[1,1,1,0,0], [1,1,1,0,0], [1,1,1,3,2], [0,0,0,2,2], [0,0,0,2,2]], dtype=np.int32)
merge_cases += [sieve_case(f'diagonal-target-{connectivity}', diagonal, 4, 'merge', connectivity, [2]) for connectivity in [4, 8]]
write('sieve-synthetic', merge_cases, [])

paths = [ROOT/f'examples/deck/spatial-analysis-showcase/public/data/dixie-fire/{band}_{epoch}.bin' for epoch in ['before', 'after'] for band in ['B08', 'B12']]
raw = np.stack([np.frombuffer(path.read_bytes(), dtype='<u2').reshape(750,750).astype(float) for path in paths])
raw[raw == 0] = -9999
sources = [source(path) for path in paths]


def statistics_case(name, values, validity, scale, offset):
    values = np.asarray(values, dtype=np.float32).astype(float)
    validity = np.asarray(validity, dtype=np.uint32)
    calibrated = values*scale+offset
    accepted = (validity != 0) & np.isfinite(values) & (values != -9999) & np.isfinite(calibrated)
    selected = calibrated[accepted]
    expected = dict(count=[len(selected)], sum=[float(np.sum(selected))], mean=[float(np.mean(selected)) if len(selected) else 0], extent=[float(np.min(selected)), float(np.max(selected))] if len(selected) else [0,0])
    empty = dict(count=[0], sum=[0], mean=[0], extent=[0,0])
    return dict(name=name, reference='numpy.sum/mean/min/max', width=values.shape[1], height=values.shape[0], values=pack(values), validity=pack(validity), scale=scale, offset=offset,
        frames=[dict(masked=False, expected=expected),dict(masked=True, expected=empty),dict(masked=False, expected=expected)])

values = raw[2,119:631,119:631]
validity = np.ones(values.shape, dtype=np.uint32)
validity[::17,::13] = 0
write('statistics-real', [statistics_case('dixie-512-calibrated',values,validity,1/16,-1)], [sources[2]])
values = ((np.arange(513)*37)%127-63).reshape(3,171).astype(float)/8
values.ravel()[:8] = [np.nan,np.inf,-np.inf,-9999,0,-0.,1024,-1024]
validity = np.where(np.arange(513).reshape(3,171)%11 == 0, 0, 7)
write('statistics-synthetic', [statistics_case('negative-calibration-tail-cancellation',values,validity,-2,.5), statistics_case('singleton',[[3]],[[1]],1,0), statistics_case('empty',[[np.nan,-9999,np.inf]],[[1,1,1]],1,0)], [])


def overlay_case(name, stack, settings):
    stack = np.asarray(stack,dtype=np.float32).astype(float)
    layer_count, cell_count = stack.shape
    breaks = np.tile([0.,.5,.5,1.],(layer_count,1))
    remap_values = np.tile([0.,.25,.5,.75,1.],(layer_count,1))
    remap_values[0,0] = np.nan
    frames=[]
    usable = ~np.isnan(stack)&(stack!=-9999)
    for setting in settings:
        mapped = np.empty_like(stack)
        weights = np.array([layer['weight'] for layer in setting['layers']],dtype=np.float32).astype(float)[:,None]
        for layer_index, layer in enumerate(setting['layers']):
            values=stack[layer_index]
            if layer.get('mode')=='table':
                count=layer.get('breakCount',0)
                indices=np.digitize(values,breaks[layer_index,:min(count,4)],right=layer.get('closed')=='right')
                mapped[layer_index]=remap_values[layer_index,indices]
            else:
                minimum=layer.get('inputMinimum',0); maximum=layer.get('inputMaximum',1)
                with np.errstate(invalid='ignore'):
                    mapped[layer_index] = (values>=minimum).astype(float) if minimum==maximum else np.clip((values-minimum)/(maximum-minimum),0,1)
                if layer.get('invert'): mapped[layer_index]=1-mapped[layer_index]
        restricted = np.any(usable & np.isnan(mapped),axis=0)
        denominator=np.sum(np.where(usable,np.abs(weights),0),axis=0)
        with np.errstate(invalid='ignore',divide='ignore'):
            score=np.sum(np.where(usable,mapped*weights,0),axis=0)
            if setting.get('normalizeWeights'): score=score/denominator
        defined=np.any(usable,axis=0)&~restricted
        if setting.get('noDataPolicy','propagate')=='propagate': defined &= np.all(usable,axis=0)
        if setting.get('normalizeWeights'): defined &= denominator>0
        score=np.where(defined,score,np.nan)
        valid=~np.isnan(score)
        limits=[np.min(score[valid]),np.max(score[valid])] if np.any(valid) else [np.nan,np.nan]
        frames.append(dict(settings=setting,expected=dict(score=pack(score),outputValidity=pack(valid),scoreRange=pack(limits))))
    return dict(name=name,reference='numpy.clip/digitize/sum/min/max (absolute-weight normalization)',cellCount=cell_count,layerCount=layer_count,stack=pack(stack),remapBreaks=pack(breaks),remapValues=pack(remap_values),frames=frames)


def overlay_settings(layer_count, maximum):
    linear=[dict(weight=(layer%5-2)/4,inputMinimum=0,inputMaximum=maximum,invert=layer%2==0) for layer in range(layer_count)]
    table=[dict(weight=(layer+1)/4,mode='table',breakCount=4,closed='left') for layer in range(layer_count)]
    return [dict(layers=linear),dict(layers=linear,normalizeWeights=True,noDataPolicy='ignore'),
        dict(layers=[dict(weight=1,inputMinimum=maximum,inputMaximum=0) if layer%2 else dict(weight=-.25,inputMinimum=maximum/2,inputMaximum=maximum/2) for layer in range(layer_count)],normalizeWeights=True,noDataPolicy='ignore'),
        dict(layers=table,noDataPolicy='ignore'),dict(layers=[dict(**{**layer,'closed':'right','breakCount':2}) for layer in table],normalizeWeights=True),
        dict(layers=[dict(weight=0) for _ in range(layer_count)],normalizeWeights=True,noDataPolicy='ignore'),
        dict(layers=[dict(weight=0) for _ in range(layer_count)],noDataPolicy='ignore'),dict(layers=linear)]

# Real remaps use only the linear modes; table boundaries are stressed with exact small synthetic inputs.
for part,(row,column) in enumerate((row,column) for row in [119,247,375,503] for column in [119,375]):
    stack=raw[:,row:row+128,column:column+256].reshape(4,-1)
    write(f'overlay-real-{part}',[overlay_case(f'dixie-512-part-{part}',stack,overlay_settings(4,8192)[:3])],sources)
stack=((np.arange(16)[:,None]*7+np.arange(257)[None,:]*11)%25-8)/8
stack[:,0]=np.nan; stack[:,1]=-9999; stack[:,2]=0; stack[:,3]=.5; stack[:,4]=1
stack[::3,5:]=np.nan; stack[::7,5:]=-9999
stack[:,6]=np.inf; stack[:,7]=-np.inf
settings=overlay_settings(16,2)
# Zero-weight restricted table class must still invalidate, and breakCount=0 uses first class.
settings += [dict(layers=[dict(weight=0,mode='table',breakCount=0) if layer==0 else dict(weight=1) for layer in range(16)],noDataPolicy='ignore')]
write('overlay-synthetic',[overlay_case('16-layer-policies-boundaries',stack,settings),overlay_case('single-layer',[[np.nan,-9999,0,.5,1,2,-1]],overlay_settings(1,2))],[])
