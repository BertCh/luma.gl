# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent corrected showcase-mode references (Python -I plus isolated WebR).

K: spatstat.explore::Kest, explicit r, none/border/isotropic. Border is requested
separately to select Kborder.engine: generic polygon Kount bins censor distances
on the coarse r grid, whereas this API uses exact reduced-sample eligibility.
Each border curve is cross-checked against Shapely distances and SciPy cdist.
L and unsmoothed
annulus differences are algebraic transforms, NOT spatstat's smoothed pcf.
Guard Clark-Evans: spatstat::clarkevans(clipregion=erosion(W,4)); scipy cKDTree
for IDs, rayleigh moments and norm/chi2 for significance. Donnelly is adapted
from spatstat's 0.0412 to the API's published 0.041 by subtracting exactly
0.0002*perimeter/n**1.5 from the reference expectation; no tolerance change.
Polygon quadrat areas: Shapely membership at the documented 16x16 midpoints,
normalized to exact Shapely area; SciPy chisquare with unequal expected counts.
F locations: identical f32 cell centers; Shapely contains/distance determines
selection/boundary distances; exact boundary samples use the public downward/leftward
symbolic ownership (one f64 nextafter each direction suffices for these axis-aligned windows); scipy cKDTree supplies distances, scipy.stats.ecdf
supplies empirical and Kaplan-Meier CDFs (CensoredData). KM times are binned
on the public radius lattice, events precede censored observations in each bin.
Border estimates use scipy ECDF of independently selected interior samples.
Hanisch is a COMPOSITE REFERENCE (Hanisch 1984, Biom. J. 26:79-90;
Baddeley/Rubak/Turner 2015 ch. 8): weighted empirical CDF via numpy.average.
No available package implements this API's pointwise weights plus capped,
reference-lattice-interpolated polygon erosion and search-truncated denominator.
The geometric sampling convention is matched, with no port of TS/WGSL.
No fixed-point rounding is applied to reference weights; unchanged S applies.

Input: deterministic 256 spatial-rank sample of the previously projected full
Chicago fixture, affine rescaled into a 96x96 test window then f32. Synthetic
L-shaped window with hole; 'view' is its independently Shapely-clipped viewport.
These stand in for window topologies, not a full Chicago polygon certification.
All 3 window options x 3 radii x 3 K corrections, and x 3 F grids x 4 distance
corrections are retained. No seeds, CPU runtime fallback, or GPU-derived outputs.
"""
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import scipy
from scipy import stats
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist, pdist
import shapely
from shapely.geometry import Polygon, box
from shapely.geometry.polygon import orient
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
VENV=Path('/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-deck-gl-master/82c646e8-3e98-44d4-84b8-212745483f4a/scratchpad/eval/venv')

def encode(value):
    if isinstance(value,dict): return {key:encode(entry) for key,entry in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)): return [encode(entry) for entry in value]
    if isinstance(value,np.integer): return int(value)
    if isinstance(value,(float,np.floating)): return float(value) if np.isfinite(value) else None
    return value

def membership(window, coordinates):
    coordinates=np.asarray(coordinates,float)
    # These windows have axis-aligned edges. A single f64 predecessor in each
    # coordinate realizes downward/leftward boundary ownership, independently
    # through GEOS containment, without reimplementing the GPU ray predicate.
    on_boundary=shapely.distance(shapely.points(coordinates),window.boundary)==0
    probes=coordinates.copy()
    probes[on_boundary]=np.nextafter(probes[on_boundary],-np.inf)
    return shapely.contains(window,shapely.points(probes))

def write(topic,cases,versions):
    path=HERE/'fixtures'/f'parity-wp6-round8-{topic}.json'
    path.write_text(json.dumps(encode({'generator':Path(__file__).name,'conventions':__doc__,
        'versions':versions,'sources':{'parity-wp6-point-patterns-positions.json':hashlib.sha256(source_path.read_bytes()).hexdigest()},
        'reference':'spatstat.explore::Kest/clarkevans; scipy cKDTree/ecdf/rayleigh/norm/chi2; Shapely; Hanisch composite', 'cases':cases}),separators=(',',':'),allow_nan=False)+'\n')
    assert path.stat().st_size<=1500000,(path,path.stat().st_size)
    print(path.name,path.stat().st_size)

source_path=HERE/'fixtures/parity-wp6-point-patterns-positions.json'
payload=json.loads(source_path.read_text())['positions']
source=np.frombuffer(base64.b64decode(payload['base64']),'<f4').reshape(-1,2).astype(float)
order=np.lexsort((source[:,1],source[:,0])); source=source[order[np.linspace(0,len(source)-1,256,dtype=int)]]
positions=(2+92*(source-source.min(axis=0))/np.ptp(source,axis=0)).astype(np.float32).astype(float)
land=Polygon([(0,0),(96,0),(96,48),(72,48),(72,96),(0,96)],holes=[[(24,24),(24,40),(40,40),(40,24)]])
windows={'rectangle':box(0,0,96,96),'city':land,'view':land.intersection(box(12,0,84,72))}
inputs=[]
for name,window in windows.items():
    window=orient(window,sign=1)
    rings=[list(window.exterior.coords)[:-1]]+[list(ring.coords)[:-1] for ring in window.interiors]
    selected=membership(window,positions)
    points=positions[selected]
    inputs.append({'name':name,'bounds':list(window.bounds),'rings':rings,'points':points.tolist()})
# Geometry-only screening, before any external output/GPU run. A dyadic radius
# step avoids exact radius representation disagreements. Keep genuine exact ties.
candidates=[]
for geometry in inputs:
    window=windows[geometry['name']];bounds=np.array(window.bounds);points=np.array(geometry['points'])
    candidates.extend([pdist(points),shapely.distance(shapely.points(points),window.boundary)])
    for grid in [24,47,94]:
        horizontal=(bounds[0]+(np.arange(grid)+.5)*(bounds[2]-bounds[0])/grid).astype(np.float32).astype(float)
        vertical=(bounds[1]+(np.arange(grid)+.5)*(bounds[3]-bounds[1])/grid).astype(np.float32).astype(float)
        locations=np.stack(np.meshgrid(horizontal,vertical),axis=-1).reshape(-1,2)
        locations=locations[membership(window,locations)]
        candidates.extend([cKDTree(points).query(locations)[0],shapely.distance(shapely.points(locations),window.boundary)])
candidates=np.concatenate(candidates)
for attempt in range(4096):
    maximum_distance=15+attempt*15/4096
    nearest_radius=np.clip(np.rint(candidates/maximum_distance*60),1,60)*maximum_distance/60
    gaps=abs(candidates-nearest_radius)
    if not np.any((gaps>0)&(gaps<=8*np.finfo(np.float32).eps*np.maximum(1,candidates))):break
else:raise AssertionError('no unambiguous shared radius lattice')
print('Geometry-only radius screening:',maximum_distance,'attempt',attempt)
for geometry in inputs:geometry['maximumDistance']=maximum_distance
with tempfile.TemporaryDirectory(prefix='parity-wp6-round8-') as temporary:
    input_path=Path(temporary)/'input.json';output_path=Path(temporary)/'spatstat.json'
    input_path.write_text(json.dumps(inputs))
    subprocess.run(['node',str(HERE/'parity-wp6-round8-spatstat.mjs'),str(VENV/'parity-wp6-webr'),str(input_path),str(output_path)],check=True)
    external=json.loads(output_path.read_text())
versions={**external['versions'],'numpy':np.__version__,'scipy':scipy.__version__,'shapely':shapely.__version__}
index_cases=[];ripley_cases=[];distance_cases=[]
for geometry,reference in zip(inputs,external['results']):
    name=geometry['name'];window=windows[name];bounds=np.array(window.bounds);area=window.area
    selected=membership(window,positions); points=positions[selected];count=len(points)
    source_rows=np.flatnonzero(selected)
    nearest=cKDTree(points).query(points,k=2)[0][:,1]
    np.testing.assert_allclose(nearest,reference['nearest'],atol=1e-12)
    border=shapely.distance(shapely.points(points),window.boundary)
    np.testing.assert_allclose(border,reference['border'],atol=1e-12)
    ring_positions=[];ring_offsets=[0]
    for ring in geometry['rings']:
        ring_positions.extend(np.asarray(ring).ravel().tolist());ring_offsets.append(len(ring_positions)//2)
    packed_window=None if name=='rectangle' else {'positions':ring_positions,'ringOffsets':ring_offsets,'polygonOffsets':[0,len(ring_offsets)-1]}
    common={'name':name,'positions':positions.ravel(),'bounds':bounds,'window':packed_window,'maximumDistance':maximum_distance}
    neighbor_distances=np.full(len(positions),np.nan);neighbor_ids=np.full(len(positions),0xffffffff,dtype=np.uint32)
    all_distances=cdist(points,points);np.fill_diagonal(all_distances,np.inf)
    neighbor_distances[selected]=nearest;neighbor_ids[selected]=source_rows[np.argmin(all_distances,axis=1)]
    focus=border>=4
    expected_mean=stats.rayleigh.mean(scale=math.sqrt(area/(2*math.pi*count)))
    observed=nearest[focus].mean()
    np.testing.assert_allclose(observed/expected_mean,reference['guardRatio'],rtol=1e-12)
    for grid in [6,12,24]:
        horizontal=np.linspace(bounds[0],bounds[2],grid+1);vertical=np.linspace(bounds[1],bounds[3],grid+1)
        quadrat_counts=np.histogram2d(points[:,1],points[:,0],bins=[vertical,horizontal])[0].ravel().astype(int)
        quadrat_areas=[]
        for row in range(grid):
            for column in range(grid):
                x=horizontal[column]+(np.arange(16)+.5)*(horizontal[column+1]-horizontal[column])/16
                y=vertical[row]+(np.arange(16)+.5)*(vertical[row+1]-vertical[row])/16
                samples=np.stack(np.meshgrid(x,y),axis=-1).reshape(-1,2)
                quadrat_areas.append(np.count_nonzero(membership(window,samples)))
        quadrat_areas=np.array(quadrat_areas,dtype=float);quadrat_areas*=area/quadrat_areas.sum()
        positive=quadrat_areas>0;expected_counts=count*quadrat_areas[positive]/area
        statistic=stats.chisquare(quadrat_counts[positive],expected_counts).statistic
        degrees=int(positive.sum())-1
        variance=np.sum((quadrat_counts[positive]-expected_counts)**2)/degrees
        for correction in (['guard','none','donnelly'] if name=='rectangle' else ['guard','none']):
            for alternative in ['two-sided','greater','lesser']:
                observed_mean=observed if correction=='guard' else nearest.mean()
                expectation=expected_mean
                standard_error=stats.rayleigh.std(scale=math.sqrt(area/(2*math.pi*count)))/math.sqrt(focus.sum() if correction=='guard' else count)
                if correction=='donnelly':
                    expectation=nearest.mean()/reference['donnellyRatio']-.0002*window.length/count**1.5
                    standard_error=math.sqrt(.0703*area/count**2+.037*window.length*math.sqrt(area/count**5))
                zscore=(observed_mean-expectation)/standard_error
                normal_tails=[stats.norm.cdf(zscore),stats.norm.sf(zscore)]
                chi_tails=[stats.chi2.cdf(statistic,degrees),stats.chi2.sf(statistic,degrees)]
                def pvalue(tails):return min(1,2*min(tails)) if alternative=='two-sided' else tails[1 if alternative=='greater' else 0]
                index_cases.append({**common,'name':f'{name}/{grid}/{correction}/{alternative}','quadratGrid':[grid,grid],
                    'correction':correction,'alternative':alternative,'guardDistance':4,'expected':{'nearestNeighborDistances':neighbor_distances,
                    'nearestNeighborIds':neighbor_ids,'quadratCounts':quadrat_counts,'quadratAreas':quadrat_areas,
                    'clarkEvans':[count,observed_mean,expectation,observed_mean/expectation,standard_error,zscore,pvalue(normal_tails)],
                    'quadratStatistics':[int(positive.sum()),count/positive.sum(),variance,statistic/degrees,statistic,degrees,pvalue(chi_tails)]}})
    for curves in reference['curves']:
        radius_count=curves['radiusCount'];radii=maximum_distance*np.arange(1,radius_count+1)/radius_count
        pair_counts=np.diff(np.r_[0,2*np.searchsorted(np.sort(pdist(points)),radii,side='right')])
        for correction in ['none','border','isotropic']:
            curve=np.array([np.nan if value is None else value for value in curves[correction]])
            if correction=='border':
                reduced_sample=np.array([area*np.count_nonzero(all_distances[border>=radius]<=radius)/(count*np.count_nonzero(border>=radius))
                    if np.any(border>=radius) else np.nan for radius in radii])
                np.testing.assert_allclose(curve,reduced_sample,rtol=1e-12,atol=1e-12)
            curve_l=np.sqrt(curve/np.pi)
            ripley_cases.append({**common,'name':f'{name}/{radius_count}/{correction}','radiusCount':radius_count,'correction':correction,
                'expected':{'k':curve,'l':curve_l,'lMinusR':curve_l-radii,'radii':radii,'pairCounts':pair_counts,
                    'pairCorrelation':np.diff(np.r_[0,curve])/(np.pi*np.diff(np.r_[0,radii*radii]))}})
        for grid in [24,47,94]:
            horizontal=(bounds[0]+(np.arange(grid)+.5)*(bounds[2]-bounds[0])/grid).astype(np.float32).astype(float)
            vertical=(bounds[1]+(np.arange(grid)+.5)*(bounds[3]-bounds[1])/grid).astype(np.float32).astype(float)
            locations=np.stack(np.meshgrid(horizontal,vertical),axis=-1).reshape(-1,2)
            locations=locations[membership(window,locations)]
            reference_distances=cKDTree(points).query(locations)[0]
            reference_border=shapely.distance(shapely.points(locations),window.boundary)
            step=maximum_distance/radius_count
            def distance_curve(distances,boundaries,correction):
                if correction=='none':return stats.ecdf(distances).cdf.evaluate(radii)
                if correction=='border':return np.array([stats.ecdf(distances[boundaries>=radius]).cdf.evaluate(radius)
                    if np.any(boundaries>=radius) else np.nan for radius in radii])
                uncensored=(distances<=boundaries)&(distances<=maximum_distance)
                if correction=='kaplan-meier':
                    observations=np.where(uncensored,distances,boundaries)
                    # Integer radius times preserve same-bin event-before-censor ordering.
                    slots=np.maximum(1,np.ceil(observations/step))
                    return stats.ecdf(stats.CensoredData.right_censored(slots,~uncensored)).cdf.evaluate(np.arange(1,radius_count+1))
                retained=uncensored|((distances>maximum_distance)&(boundaries>=maximum_distance))
                weighting_distances=np.minimum(distances[retained],maximum_distance)
                if name=='rectangle':
                    eroded=(np.maximum(bounds[2]-bounds[0]-2*weighting_distances,0)*np.maximum(bounds[3]-bounds[1]-2*weighting_distances,0))
                    ratios=np.divide(area,eroded,out=np.full(len(eroded),1024.),where=eroded>0)
                else:
                    thresholds=np.linspace(0,maximum_distance,4*radius_count+1)
                    sizes=np.array([(reference_border>=threshold).sum() for threshold in thresholds])
                    remaining=np.interp(weighting_distances,thresholds,sizes)
                    ratios=np.divide(len(locations),remaining,out=np.full(len(remaining),1024.),where=remaining>0)
                weights=np.minimum(ratios,1024)
                return np.array([np.average(distances[retained]<=radius,weights=weights) for radius in radii]) if len(weights) else np.full(radius_count,np.nan)
            for correction in ['none','border','kaplan-meier','hanisch']:
                g=distance_curve(nearest,border,correction);f=distance_curve(reference_distances,reference_border,correction)
                with np.errstate(divide='ignore',invalid='ignore'):j=(1-g)/(1-f)
                j[(f>=1)|~np.isfinite(f)|~np.isfinite(g)]=np.nan
                distance_cases.append({**common,'name':f'{name}/{radius_count}/{grid}/{correction}',
                    'radiusCount':radius_count,'referenceGrid':[grid,grid],'correction':correction,'expected':{'g':g,'f':f,'j':j,'radii':radii}})
write('indices',index_cases,versions)
write('ripley',ripley_cases,versions)
# Split by window to retain every full curve without exceeding 1.5 MB per file.
for name in windows:write('distance-'+name,[scene for scene in distance_cases if scene['name'].startswith(name+'/')],versions)
