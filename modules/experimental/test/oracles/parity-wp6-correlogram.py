# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""esda.Moran distance-band reference; full 3109 US counties, no permutations.

Run supplied venv python -I. Original f32 county centroids EPSG:4326 -> pyproj
EPSG:5070 metres -> f32; diabetes percentages use original f32. Euclidean planar
metric, NOT geodesic distances. All finite, in-window, nonzero-mask rows count,
including islands; binary symmetric weights, no diagonal, transformation='B'.
Both annulus and cumulative bands and esda normality/randomization analytic
variance/two-sided normal p-values are checked. P profile uses S for observed
values and analytic inference; this API has no permutation inference.

scipy.pdist supplies independent full pair distances. Bands are half-open at
internal boundaries, last upper endpoint inclusive (the public contributor's
explicit floor convention, distinct from Ripley's ceil convention). Real-data
radii are chosen by geometry-only screening to exclude f32-ambiguous boundaries;
synthetic axis-aligned exact boundary ties remain. Exact pair counts and all
summary/curve values are retained. scipy.stats.kurtosis(fisher=False,bias=True)
provides population kurtosis. Peaks are scipy.signal.find_peaks on finite z,
with strict maxima (plateaus removed) and first-index tie ownership.

Empty graph, constant values and n<3 have NaN Moran; expected I exists for n>=2.
Complete graphs have zero theoretical variance: NaN z/p/variance by API contract.
No near-degenerate positive-variance case is silently masked to fit the GPU's
cancellation guard. Polygon windows and overflow saturation are not covered.
"""
import hashlib
import json
import math
import os
import platform
import tempfile
import warnings
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'parity-wp6-matplotlib'))
import esda
import libpysal
import numpy as np
import pyproj
import scipy
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kurtosis
from scipy.signal import find_peaks
from libpysal.weights import W

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
DATA=ROOT/'examples/deck/spatial-analysis-showcase/public/data/us-counties'
VERSIONS={'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'esda':esda.__version__,'libpysal':libpysal.__version__,'pyproj':pyproj.__version__,'proj':pyproj.proj_version_str}
SOURCES={str((DATA/name).relative_to(ROOT)):hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in ['manifest.json','centroid.bin','places_diabetes_ageAdj.bin']}

def encode(value):
    if isinstance(value,dict):return {key:encode(entry) for key,entry in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):return [encode(entry) for entry in value]
    if isinstance(value,np.integer):return int(value)
    if isinstance(value,(float,np.floating)):
        if np.isnan(value):return None
        if np.isinf(value):return 'Infinity' if value>0 else '-Infinity'
        return float(value)
    return value

def makeCase(name,coordinates,values,bounds,maximumDistance,bandCount=8,mask=None,screen=False):
    coordinates=np.asarray(coordinates,dtype=np.float32).astype(float);values=np.asarray(values,dtype=np.float32).astype(float)
    bounds=np.asarray(bounds,dtype=np.float32).astype(float)
    mask=np.ones(len(values),dtype=np.uint32) if mask is None else np.asarray(mask,dtype=np.uint32)
    selected=(mask!=0)&np.isfinite(values)&np.isfinite(coordinates).all(axis=1)&(coordinates>=bounds[:2]).all(axis=1)&(coordinates<=bounds[2:]).all(axis=1)
    points=coordinates[selected];observations=values[selected];count=len(points)
    distances=pdist(points)
    if screen:
        distancesSorted=np.sort(distances)
        # Exclude every pair within an 8-epsilon distance-arithmetic bound of a radius.
        for attempt in range(10000):
            maximum=float(np.float32(maximumDistance+attempt*bandCount))
            radii=maximum*np.arange(1,bandCount+1)/bandCount
            insertion=np.searchsorted(distancesSorted,radii)
            closest=np.minimum(abs(distancesSorted[np.clip(insertion,0,len(distances)-1)]-radii),abs(distancesSorted[np.clip(insertion-1,0,len(distances)-1)]-radii))
            if np.all(closest>8*np.finfo(np.float32).eps*radii):break
        else:raise AssertionError('no unambiguous band grid')
    else:maximum=float(np.float32(maximumDistance));attempt=0
    radii=maximum*np.arange(1,bandCount+1)/bandCount
    matrix=squareform(distances) if count else np.zeros((0,0))
    np.fill_diagonal(matrix,np.inf)
    if count:
        centered=observations-observations.mean();sumSquares=float(centered@centered)
        statistics=[count,observations.mean(),observations.var(),sumSquares,kurtosis(observations,fisher=False,bias=True)]
    else:sumSquares=0;statistics=[0,math.nan,math.nan,math.nan,math.nan]
    runs=[]
    for mode in ['cumulative','annulus']:
        expected={assumption:{key:[] for key in ['moransI','expectedI','varianceI','zScores','pValues','pairCounts']} for assumption in ['normality','randomization']}
        for band,radius in enumerate(radii):
            adjacency=matrix<=radius if band==bandCount-1 else matrix<radius
            if mode=='annulus' and band:adjacency &= matrix>=radii[band-1]
            pairCount=int(adjacency.sum()//2)
            reference=None
            if count>=4 and pairCount and sumSquares>0:
                weights=W({index:np.flatnonzero(adjacency[index]).tolist() for index in range(count)},silence_warnings=True)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    reference=esda.Moran(observations,weights,transformation='B',permutations=0,two_tailed=True)
            for assumption,output in expected.items():
                output['pairCounts'].append(pairCount);output['expectedI'].append(-1/(count-1) if count>=2 else math.nan)
                if reference is None:
                    for key in ['moransI','varianceI','zScores','pValues']:output[key].append(math.nan)
                else:
                    suffix='norm' if assumption=='normality' else 'rand'
                    output['moransI'].append(reference.I)
                    complete=2*pairCount==count*(count-1)
                    variance=getattr(reference,'VI_'+suffix)
                    if not complete:assert variance>0, (name,mode,band,assumption,variance)
                    output['varianceI'].append(math.nan if complete else variance)
                    output['zScores'].append(math.nan if complete else getattr(reference,'z_'+suffix))
                    output['pValues'].append(math.nan if complete else getattr(reference,'p_'+suffix))
        for assumption,output in expected.items():
            scores=np.asarray(output['zScores']);finite=np.flatnonzero(np.isfinite(scores))
            peaks=find_peaks(np.where(np.isfinite(scores),scores,-np.inf))[0]
            peaks=[index for index in peaks if np.isfinite(scores[index-1:index+2]).all() and scores[index]>scores[index-1] and scores[index]>scores[index+1]]
            output['peakBands']=[peaks[0] if len(peaks) else 0xffffffff,finite[np.argmax(scores[finite])] if len(finite) else 0xffffffff]
            output['statistics']=statistics;output['overflow']=[0]
            runs.append({'bandMode':mode,'varianceAssumption':assumption,'expected':output})
    return {'name':name,'positions':coordinates.ravel(),'values':values,'mask':mask,'bounds':bounds,'maximumDistance':maximum,'bandCount':bandCount,'includedCount':count,'radiusScreenAttempts':attempt,'runs':runs}

source=np.fromfile(DATA/'centroid.bin',dtype='<f4').reshape(-1,2)
transformer=pyproj.Transformer.from_crs(4326,5070,always_xy=True)
projected=np.column_stack(transformer.transform(source[:,0],source[:,1])).astype(np.float32).astype(float)
values=np.fromfile(DATA/'places_diabetes_ageAdj.bin',dtype='<f4')
bounds=[math.floor(projected[:,0].min())-16,math.floor(projected[:,1].min())-16,math.ceil(projected[:,0].max())+16,math.ceil(projected[:,1].max())+16]
cases=[makeCase('us-counties-all-diabetes',projected,values,bounds,262144,screen=True)]
points=[[0,0],[1,0],[2,0],[3,0],[0,3],[4,4],[8,8],[8,8],[np.nan,2],[2,2],[9,0],[0,8]]
values=[1,4,2,7,8,2,9,11,2,np.nan,7,4]
cases.append(makeCase('exact-boundaries-duplicates-islands-exclusions',points,values,[0,0,8,8],4,4,[1]*11+[0]))
cases.append(makeCase('complete-graph',[[0,0],[1,0],[2,0],[3,0],[0,3]],[1,4,2,7,8],[0,0,4,4],8,1))
cases.append(makeCase('constant',[[0,0],[1,0],[2,0],[3,0],[0,3]],[4]*5,[0,0,4,4],4,4))
cases.append(makeCase('empty-selection',[[1,1],[3,3]],[2,3],[0,0,4,4],4,4,[0,0]))
output=HERE/'fixtures/parity-wp6-correlogram.json'
output.write_text(json.dumps(encode({'versions':VERSIONS,'sources':SOURCES,'conventions':__doc__,'cases':cases}),separators=(',',':'),allow_nan=False)+'\n')
assert output.stat().st_size<=1500000
print(output.name,output.stat().st_size, 'real radius',cases[0]['maximumDistance'],'screen attempts',cases[0]['radiusScreenAttempts'])
