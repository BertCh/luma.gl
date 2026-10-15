# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""NumPy cell-stack references: all ten outputs, every cell.

https://numpy.org/doc/stable/reference/generated/numpy.nanstd.html
https://numpy.org/doc/stable/reference/generated/numpy.unique.html

Decode full Dixie B08/B12 before/after uint16 rasters (750x750); fixed 512x512
window rows/cols 119:631, eight lossless 128x256 partitions. This is pointwise
across layers, so no halo or independent-domain approximation. Raw zero DN is
sentinel -9999; no SCL masking, calibration, weighting or spectral-composite
claim. Values round to f32 BEFORE f64 NumPy reductions. Standard deviation is
population (ddof=0). NaN and sentinel invalid; infinities valid per public API.
Empty/under-minimum/propagate-missing cells have NaN floats and variety=0, but
counts always retain actual valid count. Frequency ties select smallest value;
+0/-0 are equal. API has no weights; weighted overlay is a separate class.

Real data: default ignore. Synthetic: 64 layers, all-missing,
part-missing, tied frequencies, single-valid, large offset with small variance,
negative/cancelling values, signed zero; five policies including minimum=0 and
minimum above layer count. Separate infinity cases preserve exact masks without
letting extreme scales weaken finite cases. Single-layer topology included.
R bound unchanged for moments; exact selected values, counts, variety and masks.
"""
import base64
import gzip
import hashlib
import json
import platform
import warnings
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
DATA=ROOT/'examples/deck/spatial-analysis-showcase/public/data/dixie-fire'
VERSIONS=dict(python=platform.python_version(),numpy=np.__version__)
FLOAT_NAMES=['minimum','maximum','range','sum','mean','standardDeviation','majority','minority']


def pack(values):
    numbers=[None if np.isnan(value) else 'Infinity' if value==np.inf else '-Infinity' if value==-np.inf else '-0' if value==0 and np.signbit(value) else float(value) for value in np.asarray(values).ravel()]
    return base64.b64encode(gzip.compress(json.dumps(numbers,separators=(',',':')).encode(),mtime=0)).decode()


def make_case(name,stack,settings):
    stack=np.asarray(stack,dtype=np.float32).astype(float)
    layer_count,cell_count=stack.shape
    usable=~np.isnan(stack)&(stack!=-9999)
    counts=usable.sum(axis=0)
    clean=np.where(usable,stack,np.nan)
    with warnings.catch_warnings(),np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        minimum=np.nanmin(clean,axis=0); maximum=np.nanmax(clean,axis=0)
        outputs=dict(minimum=minimum,maximum=maximum,range=maximum-minimum,
            sum=np.nansum(clean,axis=0),mean=np.nanmean(clean,axis=0),standardDeviation=np.nanstd(clean,axis=0,ddof=0))
    majority=np.full(cell_count,np.nan); minority=majority.copy(); variety=np.zeros(cell_count)
    for cell in range(cell_count):
        unique,frequency=np.unique(stack[usable[:,cell],cell],return_counts=True)
        if len(unique):
            majority[cell]=unique[np.argmax(frequency)]
            minority[cell]=unique[np.argmin(frequency)]
            variety[cell]=len(unique)
    outputs.update(majority=majority,minority=minority,variety=variety,counts=counts)
    frames=[]
    for setting in settings:
        defined=counts>=max(1,setting.get('minimumValidCount',1))
        if setting.get('noDataPolicy')=='propagate': defined &= counts==layer_count
        expected={key:pack(np.where(defined,values,np.nan)) for key,values in outputs.items() if key in FLOAT_NAMES}
        expected.update(variety=pack(np.where(defined,variety,0)),counts=pack(counts))
        frames.append(dict(settings=setting,expected=expected))
    return dict(name=name,reference='numpy.nanmin/nanmax/nansum/nanmean/nanstd/unique',
        cellCount=cell_count,layerCount=layer_count,stack=pack(stack),frames=frames)


def write(name,cases,sources):
    output=HERE/f'fixtures/parity-wp4-{name}.json'
    output.write_text(json.dumps(dict(versions=VERSIONS,sources=sources,cases=cases),separators=(',',':'))+'\n')
    assert output.stat().st_size<1_500_000,(output,output.stat().st_size)
    print(output.name,output.stat().st_size)


paths=[DATA/f'{band}_{epoch}.bin' for epoch in ['before','after'] for band in ['B08','B12']]
raw=np.stack([np.frombuffer(path.read_bytes(),dtype='<u2').reshape(750,750).astype(float) for path in paths])
raw[raw==0]=-9999
sources=[dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths]
for part,(row,column) in enumerate((row,column) for row in [119,247,375,503] for column in [119,375]):
    scene=make_case(f'dixie-512-tile-part-{part}',raw[:,row:row+128,column:column+256].reshape(4,-1),[{}])
    scene.update(width=256,height=128,sourceRow=row,sourceColumn=column)
    write(f'cell-statistics-real-{part}',[scene],sources)

stack=((np.arange(64)[:,None]*17+np.arange(257)[None,:]*11)%19-9).astype(float)/4
stack[:,0]=np.nan; stack[:,1]=-9999; stack[:,2]=0; stack[::2,2]=-0.
stack[:,3]=np.arange(64)%2; stack[:,4]=np.nan; stack[17,4]=3
stack[:,5]=4096+np.arange(64)/8
stack[:,6]=np.tile([100,-100,1,-1],16)
stack[::7,7:]=np.nan; stack[::11,7:]=-9999
settings=[{},dict(noDataPolicy='propagate'),dict(minimumValidCount=60),dict(minimumValidCount=0),dict(minimumValidCount=65)]
special=np.array([[np.inf,-np.inf,np.inf,1,np.nan,-9999],[np.inf,-np.inf,-np.inf,np.inf,2,3],[1,2,4,3,4,5]])
cases=[make_case('64-layer-ties-missingness-cancellation',stack,settings),
    make_case('infinities-are-values',special,[{},dict(noDataPolicy='propagate')]),
    make_case('single-layer',[[np.nan,-9999,-0.,0,1,2,-4]],settings[:2])]
write('cell-statistics-synthetic',cases,[])
