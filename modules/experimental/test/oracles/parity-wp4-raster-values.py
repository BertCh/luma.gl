# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""NumPy ufunc arithmetic and direct footprint extrema references.
https://numpy.org/doc/stable/reference/ufuncs.html
https://numpy.org/doc/stable/reference/generated/numpy.min.html

Arithmetic: f64 NumPy on f32 inputs, explicit public domain adaptation (divide
by zero -> NaN; negative-base power -> NaN even at integer exponents; log<=0,
sqrt<0 -> NaN); raw NaN/sentinel detection before calibration. Unary operations
ignore b. Round is ties-to-even. Clamp follows evaluation; finite overflow is
rounded to f32 before clamp, matching the declared f32 storage/evaluation type.
Full 512x512 Dixie NBR tile rows/cols 119:631 uses raw B08/B12 after bands,
decoded from the FULL 750x750 source. Four 256x256 fixture parts are lossless
transport partitions of this local operator, not independently solved global
problems. No halo required. The 16-operation synthetic matrix covers constants,
array operands, calibration, clamp, NaN, sentinel, domain failures and halves.
Separate overflow cases avoid letting extreme scale loosen ordinary budgets.

Pyramids: decode full 2048x2048 Terrarium DEM then fixed 512x512 tile 768:1280.
Every level is independently reduced DIRECTLY from original pixel footprints,
never from a mirror of GPU reduction passes. Cell footprint SxS; bilinear S+1
in each dimension, clipped to grid. Invalid/nonfinite/raw sentinel excluded
before scale=.5,offset=-100; empty min/max are +/-FLT_MAX. Both footprints,
base block 4 on real terrain; blocks 1/2/4 on ragged masked synthetic terrain.
All levels/partial cells retained, global maximum tested on the bounded tile.
R for arithmetic; E (stronger than R) for extrema, which only select exactly
representable calibrated inputs. No tolerance widening.
"""
import base64
import gzip
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import rasterio

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
DATA=ROOT/'examples/deck/spatial-analysis-showcase/public/data'
VERSIONS=dict(python=platform.python_version(),numpy=np.__version__,rasterio=rasterio.__version__)


def clean(value):
    if isinstance(value,dict): return {key:clean(entry) for key,entry in value.items()}
    if isinstance(value,(list,tuple)): return [clean(entry) for entry in value]
    if isinstance(value,float) and not np.isfinite(value): return None if np.isnan(value) else 'Infinity' if value>0 else '-Infinity'
    return value


def pack(value):
    return base64.b64encode(gzip.compress(json.dumps(clean(np.asarray(value).ravel().tolist()),separators=(',',':')).encode(),mtime=0)).decode()


def source(path):
    return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(name,cases,sources):
    output=HERE/f'fixtures/parity-wp4-{name}.json'
    output.write_text(json.dumps(clean(dict(versions=VERSIONS,sources=sources,cases=cases)),separators=(',',':'))+'\n')
    assert output.stat().st_size<=1_500_000,output.stat().st_size
    print(output.name,output.stat().st_size)


OPERATIONS=dict(add=np.add,subtract=np.subtract,multiply=np.multiply,divide=np.divide,
    minimum=np.minimum,maximum=np.maximum,power=np.power,absoluteDifference=lambda first,second:np.abs(first-second),
    normalizedDifference=lambda first,second:(first-second)/(first+second),absolute=np.abs,squareRoot=np.sqrt,
    logarithm=np.log,exponential=np.exp,floor=np.floor,ceil=np.ceil,round=np.rint)
BINARY=list(OPERATIONS)[:9]


def arithmetic(name,first,second,settings):
    first=np.asarray(first,dtype=np.float32).astype(float)
    second=np.asarray(second,dtype=np.float32).astype(float) if second is not None else None
    operation=settings['operation']
    raw_second=second if second is not None else settings.get('constantB',0)
    invalid=np.isnan(first)|(first==-9999)
    if operation in BINARY: invalid=invalid|np.isnan(raw_second)|(raw_second==-9999)
    first_calibrated=first*settings.get('scaleA',1)+settings.get('offsetA',0)
    second_calibrated=raw_second*settings.get('scaleB',1)+settings.get('offsetB',0)
    with np.errstate(all='ignore'):
        result=OPERATIONS[operation](first_calibrated,second_calibrated) if operation in BINARY else OPERATIONS[operation](first_calibrated)
        if operation=='divide': invalid=invalid|(second_calibrated==0)
        if operation=='normalizedDifference': invalid=invalid|((first_calibrated+second_calibrated)==0)
        if operation=='power': invalid=invalid|(first_calibrated<0)|((first_calibrated==0)&(second_calibrated<=0))
        if operation=='logarithm': invalid=invalid|(first_calibrated<=0)
        result=np.where(np.abs(result)>np.finfo(np.float32).max,np.copysign(np.inf,result),result)
        result=np.clip(result,settings.get('clampMinimum',-np.inf),settings.get('clampMaximum',np.inf))
    result=np.where(invalid,np.nan,result)
    return dict(name=name,reference='numpy arithmetic ufuncs with documented domain/nodata adapter',
        first=pack(first),second=pack(second) if second is not None else None,settings=settings,expected=pack(result),length=first.size)


first_path=DATA/'dixie-fire/B08_after.bin'; second_path=DATA/'dixie-fire/B12_after.bin'
first=np.frombuffer(first_path.read_bytes(),dtype='<u2').reshape(750,750).astype(float)
second=np.frombuffer(second_path.read_bytes(),dtype='<u2').reshape(750,750).astype(float)
first[first==0]=-9999; second[second==0]=-9999
for part,(row,column) in enumerate([(119,119),(119,375),(375,119),(375,375)]):
    case=arithmetic(f'dixie-512-tile-quarter-{part}',first[row:row+256,column:column+256],second[row:row+256,column:column+256],dict(operation='normalizedDifference'))
    case.update(width=256,height=256,sourceRow=row,sourceColumn=column)
    write(f'arithmetic-real-{part}',[case],[source(first_path),source(second_path)])
first=np.array([-9999,np.nan,-4,-2.5,-1.5,-.5,-0.,0,.125,.5,1,1.5,2.5,4,8,12]*9,dtype=float)
second=np.resize(np.array([1,2,np.nan,-9999,0,0,-1,.5,-.125,2,3,1.5,-2,4,.25,1.]),first.shape)
cases=[]
for operation in OPERATIONS:
    cases.append(arithmetic(operation+'-array',first,second,dict(operation=operation)))
    cases.append(arithmetic(operation+'-constant-calibrated-clamped',first,None,
        dict(operation=operation,scaleA=.5,offsetA=1,scaleB=2,offsetB=-1,constantB=2,clampMinimum=-3,clampMaximum=5)))
cases.extend([
    arithmetic('multiply-f32-overflow',[-3e30,3e30,2,0],[-3e20,3e20,4,1],dict(operation='multiply')),
    arithmetic('exponential-f32-overflow',[89,90,-20,0],None,dict(operation='exponential')),
    arithmetic('overflow-before-finite-clamp',[3e30,-3e30],[3e20,3e20],dict(operation='multiply',clampMinimum=-8,clampMaximum=8)),
    arithmetic('unary-ignores-invalid-constant',[-2,0,2,4],None,dict(operation='squareRoot',constantB=-9999))])
for operation in ['add','subtract','multiply','divide','minimum','maximum','normalizedDifference','absolute','squareRoot','logarithm']:
    cases.append(arithmetic(operation+'-infinite-operands',[np.inf,-np.inf,np.inf,np.inf,4],
        [2,2,np.inf,-np.inf,np.inf],dict(operation=operation)))
write('arithmetic-synthetic',cases,[])


def pyramid(name,values,validity,first_block,footprint):
    values=np.asarray(values,dtype=np.float32).astype(float)
    height,width=values.shape
    usable=np.isfinite(values)&(values!=-9999)&(validity!=0)
    calibrated=np.where(usable,values*.5-100,np.nan)
    maximum=[]; minimum=[]; levels=[]; block=first_block
    while True:
        level_width=(width+block-1)//block; level_height=(height+block-1)//block
        levels.append(dict(blockSize=block,width=level_width,height=level_height,offset=len(maximum)))
        for row in range(level_height):
            for column in range(level_width):
                extra=int(footprint=='bilinear')
                samples=calibrated[row*block:min(height,(row+1)*block+extra),column*block:min(width,(column+1)*block+extra)]
                finite=samples[np.isfinite(samples)]
                maximum.append(float(np.max(finite)) if finite.size else -float(np.finfo(np.float32).max))
                minimum.append(float(np.min(finite)) if finite.size else float(np.finfo(np.float32).max))
        if level_width==level_height==1: break
        block*=2
    return dict(name=name,reference='numpy.min/max on original pixel footprints independently at every level',
        width=width,height=height,values=pack(values),validity=pack(validity),firstBlockSize=first_block,footprint=footprint,
        levels=levels,maximum=pack(maximum),minimum=pack(minimum),length=len(maximum),scale=.5,offset=-100)


path=DATA/'grand-canyon-dem/dem.png'
with rasterio.open(path) as dataset: channels=dataset.read().astype(float)
elevation=channels[0]*256+channels[1]+channels[2]/256-32768
values=elevation[768:1280,768:1280]
for footprint in ['cell','bilinear']:
    write('pyramid-real-'+footprint,[pyramid('grand-canyon-512-square-'+footprint,values,np.ones_like(values,dtype=np.uint32),4,footprint)],[source(path)])
rows,columns=np.indices((37,65)); values=((rows*53+columns*17)%113-56).astype(float)/4
validity=np.ones_like(values,dtype=np.uint32); validity[:10,:10]=0; validity[16,16]=0
values[17,32]=np.nan; values[33,64]=np.inf; values[36,63]=-9999; values[0,64]=-np.inf
cases=[pyramid(f'ragged-{footprint}-block-{block}',values,validity,block,footprint) for footprint in ['cell','bilinear'] for block in [1,2,4]]
cases.append(pyramid('singleton-empty',[[np.nan]],np.ones((1,1),dtype=np.uint32),1,'cell'))
write('pyramid-synthetic',cases,[])
