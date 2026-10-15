# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent pointpats K/L/F/G/J references, rectangle and no edge correction.

Run supplied venv python -I. Chicago source f32 EPSG:4326 -> pyproj EPSG:26916,
subtract [400000,4600000] metres, then f32 inputs / f64 reference. K uses 2048
systematic spatial-order samples (lexsort x,y then evenly spaced ranks), F/G/J
also use ALL 105808 events with a fixed 128x128 cell-center F grid. No RNG.
Full input is reused losslessly from parity-wp6-point-patterns-positions.json.

Conventions explicitly adapted from pointpats 2.6 API:
- k/l use observed bbox and n^2: rescale by A_window/A_bbox * n/(n-1).
- k uses strict '<': nextafter(radius,+inf) gives API inclusive '<='.
- f/g histogram normalizes only within support: append infinity so ALL samples
  enter the denominator, then retain every requested radius. Start at zero to
  retain duplicate points; shift internal bin edges with nextafter for <=.
- j receives these identical precomputed nearest distances and truncate=False;
  F==1 maps undefined J to API NaN (including pointpats' +inf convention).
- pairCounts = differences of scipy.pdist ordered cumulative counts; unsmoothed
  annulus pairCorrelation = differences of pointpats K / annulus area, a
  composite reference (Baddeley/Rubak/Turner 2015, section 7.3).
Masks/window membership/nonfinite exclusions are explicit before library calls.
Empty and singleton API invalid masks are explicit. Radius-choice screening
rejects non-exact f32-ambiguous boundaries; synthetic exact ties are retained.
R/spatstat unavailable: no corrected-mode or polygon-window parity is claimed.
"""
import hashlib
import base64
import json
import math
import os
import platform
import tempfile
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'parity-wp6-matplotlib'))
import numpy as np
import pointpats
import pyproj
import scipy
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
DATA=ROOT/'examples/deck/spatial-analysis-showcase/public/data/chicago-places'
VERSIONS={'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'pointpats':pointpats.__version__,'pyproj':pyproj.__version__,'proj':pyproj.proj_version_str}
SOURCES={str((DATA/name).relative_to(ROOT)):hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in ['manifest.json','position.bin']}

def encode(value):
    if isinstance(value,dict):return {key:encode(entry) for key,entry in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):return [encode(entry) for entry in value]
    if isinstance(value,np.integer):return int(value)
    if isinstance(value,(float,np.floating)):
        if np.isnan(value):return None
        if np.isinf(value):return 'Infinity' if value>0 else '-Infinity'
        return float(value)
    return value

def write(name,cases):
    output=HERE/f'fixtures/parity-wp6-{name}.json'
    output.write_text(json.dumps(encode({'versions':VERSIONS,'sources':SOURCES,'conventions':__doc__,'cases':cases}),separators=(',',':'),allow_nan=False)+'\n')
    assert output.stat().st_size<=1500000
    print(output.name,output.stat().st_size)

def makeCase(name,coordinates,bounds,maximumDistance,radiusCount=16,mask=None,includeK=True,full=False):
    coordinates=np.asarray(coordinates,dtype=np.float32).astype(float)
    bounds=np.asarray(bounds,dtype=np.float32).astype(float)
    mask=np.ones(len(coordinates),dtype=np.uint32) if mask is None else np.asarray(mask,dtype=np.uint32)
    selected=(mask!=0)&np.isfinite(coordinates).all(axis=1)&(coordinates>=bounds[:2]).all(axis=1)&(coordinates<=bounds[2:]).all(axis=1)
    points=coordinates[selected];count=len(points)
    width,height=bounds[2:]-bounds[:2];area=width*height
    referenceGrid=[128,128] if full else [32,32]
    horizontal=bounds[0]+(np.arange(referenceGrid[0])+.5)*width/referenceGrid[0]
    vertical=bounds[1]+(np.arange(referenceGrid[1])+.5)*height/referenceGrid[1]
    # Dyadic grids and integer bounds keep all reference locations exactly representable.
    referencePoints=np.stack(np.meshgrid(horizontal,vertical),axis=-1).reshape(-1,2)
    np.testing.assert_array_equal(referencePoints,referencePoints.astype(np.float32).astype(float))
    nearest=cKDTree(points).query(points,k=2)[0][:,1] if count>1 else np.full(count,np.inf)
    emptyDistances=cKDTree(points).query(referencePoints)[0] if count else np.full(len(referencePoints),np.inf)
    pairDistances=pdist(points) if includeK and count>1 else np.array([])
    # Deterministic geometry-only screening, never inspecting the GPU result or tolerance.
    for attempt in range(256):
        maximum=float(np.float32(maximumDistance+attempt*radiusCount))
        radii=maximum*np.arange(1,radiusCount+1)/radiusCount
        candidates=np.concatenate([pairDistances,nearest,emptyDistances])
        finite=candidates[np.isfinite(candidates)]
        slots=np.clip(np.rint(finite/maximum*radiusCount),1,radiusCount)
        gaps=np.abs(finite-slots*maximum/radiusCount)
        ambiguous=(gaps>0)&(gaps<=8*np.finfo(np.float32).eps*np.maximum(1,finite))
        if not ambiguous.any():break
    else:raise AssertionError('no unambiguous radius grid')
    common={'name':name,'positions':None if full else coordinates.ravel(),'fullPositionsPayload':full,'mask':None if full else mask,'bounds':bounds,'maximumDistance':maximum,'radiusCount':radiusCount,'sourceCount':len(coordinates),'includedCount':count,'referenceGrid':referenceGrid,'radiusScreenAttempts':attempt,'radii':radii}
    resultK=None
    if includeK:
        if count>1:
            support=np.nextafter(radii,np.inf)
            bboxArea=np.prod(np.ptp(points,axis=0))
            factor=area/bboxArea*count/(count-1)
            curveK=pointpats.k(points,support=support,distances=pairDistances)[1]*factor
            curveL=pointpats.l(points,support=support,distances=pairDistances)[1]*math.sqrt(factor)
            cumulative=2*np.searchsorted(np.sort(pairDistances),radii,side='right')
            np.testing.assert_allclose(curveK,cumulative*area/(count*(count-1)),rtol=1e-13,atol=1e-13)
            pairCounts=np.diff(np.r_[0,cumulative])
        else:
            curveK=curveL=np.full(radiusCount,np.nan);pairCounts=np.zeros(radiusCount,dtype=int)
        resultK={**common,'expected':{'k':curveK,'l':curveL,'lMinusR':curveL-radii,'pairCorrelation':np.diff(np.r_[0,curveK])/(np.pi*np.diff(np.r_[0,radii*radii])),'pairCounts':pairCounts,'radii':radii}}
    support=np.r_[0,np.nextafter(radii,np.inf),np.inf]
    curveG=pointpats.g(points,support=support,distances=nearest)[1][1:-1] if count>1 else np.full(radiusCount,np.nan)
    curveF=pointpats.f(points,support=support,distances=emptyDistances)[1][1:-1] if count else np.full(radiusCount,np.nan)
    curveJ=pointpats.j(points,support=support,distances=(nearest,emptyDistances),truncate=False)[1][1:-1] if count>1 else np.full(radiusCount,np.nan)
    curveJ[curveF>=1]=np.nan
    if count>1:np.testing.assert_array_equal(curveG,np.array([(nearest<=radius).mean() for radius in radii]))
    if count:np.testing.assert_array_equal(curveF,np.array([(emptyDistances<=radius).mean() for radius in radii]))
    resultDistance={**common,'expected':{'g':curveG,'f':curveF,'j':curveJ,'radii':radii}}
    return resultK,resultDistance

source=np.fromfile(DATA/'position.bin',dtype='<f4').reshape(-1,2)
transformer=pyproj.Transformer.from_crs(4326,26916,always_xy=True)
projected=(np.column_stack(transformer.transform(source[:,0],source[:,1]))-[400000,4600000]).astype(np.float32).astype(float)
payload=json.loads((HERE/'fixtures/parity-wp6-point-patterns-positions.json').read_text())['positions']
payloadBytes=base64.b64decode(payload['base64'])
np.testing.assert_array_equal(np.frombuffer(payloadBytes,dtype='<f4').reshape(-1,2),projected)
bounds=[math.floor(projected[:,0].min())-16,math.floor(projected[:,1].min())-16,math.ceil(projected[:,0].max())+16,math.ceil(projected[:,1].max())+16]
order=np.lexsort((projected[:,1],projected[:,0]));rows=order[np.linspace(0,len(order)-1,2048,dtype=int)]
pairs=[makeCase('chicago-places-spatial-ranks-2048',projected[rows],bounds,1024)]
pairs[0][0]['sourceRows']=rows;pairs[0][1]['sourceRows']=rows
pairs.append(makeCase('duplicates-exact-radii-masks',[[0,0],[0,0],[2,0],[0,2],[2,2],[4,4],[8,8],[4,0],[0,4],[2,4],[1,1],[-1,2],[np.nan,2],[2,np.inf]],[0,0,8,8],8,8,[1]*10+[0,1,1,1]))
pairs.append(makeCase('singleton',[[1,1],[3,3]],[0,0,8,8],8,8,[1,0]))
pairs.append(makeCase('empty-selection',[[1,1],[3,3]],[0,0,8,8],8,8,[0,0]))
full=makeCase('chicago-places-all',projected,bounds,1024,includeK=False,full=True)[1]
write('ripley',[pair[0] for pair in pairs])
write('ripley-distance',[pair[1] for pair in pairs]+[full])
