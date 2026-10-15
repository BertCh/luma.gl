# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""GEOS clipping in lon/lat followed by PROJ spherical endpoint distance.

Linear-degree edges are the public clipping model; each source segment is kept
separate (no union removes repeated traversals). Density uses the spherical
rectangle area, evaluated independently by scipy.integrate.quad of cos(latitude)
on the WGS84 authalic sphere. Meter and kilometer units and a custom radius are
separate cases. Real input: 200 complete roads from the earlier hashed Chicago
window, inverse-projected with PROJ and rounded to f32 before GEOS.
D/G bound: 16 coordinate ulps converted to arc length per clipped piece, plus
2e-6 times length and gamma(n) summation roundoff. Density propagates numerator
error plus error in sin(N)-sin(S): absolute sine error <=8 epsilon32, hence
relative area bound 16 epsilon32/abs(sin(N)-sin(S)) + 4 epsilon32.
Bounds are derived before GPU execution; empty outputs must remain zero.
"""
import ast,json,hashlib,platform
from pathlib import Path
import numpy as np
import pyproj,shapely,scipy
from scipy.integrate import quad
from shapely import LineString,Polygon,box,STRtree
HERE=Path(__file__).resolve().parent
source=HERE/'fixtures/parity-wp2-prepared-nearest.json'; road=json.loads(source.read_text())['cases'][0]
offsets=road['lineOffsets'][:201]; projected=np.array(road['frames'][0]['positions']).reshape(-1,2)[:offsets[-1]]
transform=pyproj.Transformer.from_crs('+proj=aeqd +lat_0=41.88 +lon_0=-87.63 +datum=WGS84 +units=m',4326,always_xy=True)
positions=np.array(transform.transform(projected[:,0],projected[:,1])).T.astype(np.float32).astype(float)
segments=[(line,LineString([first,second])) for line,(start,stop) in enumerate(zip(offsets[:-1],offsets[1:])) for first,second in zip(positions[start:stop-1],positions[start+1:stop]) if not np.array_equal(first,second)]
versions=dict(python=platform.python_version(),numpy=np.__version__,pyproj=pyproj.__version__,shapely=shapely.__version__,scipy=scipy.__version__)
source_info={source.name:hashlib.sha256(source.read_bytes()).hexdigest(),'selection':'first 200 complete roads; no vertex thinning'}
epsilon=float(np.finfo(np.float32).eps)

def emit(name,cases):
 path=HERE/f'fixtures/parity-wp2-round8-{name}.json'
 path.write_text(json.dumps(dict(generator=Path(__file__).name,reference='shapely.intersection/difference; pyproj.Geod.inv; scipy.integrate.quad spherical cell area',versions=versions,source=source_info,tolerance=dict(length='16 coordinate ulps converted to arc length per piece + 2e-6 distance + gamma(n) reduction'),cases=cases),separators=(',',':'),allow_nan=False)+'\n'); assert path.stat().st_size<1500000
 print(path.name,path.stat().st_size)

density_cases=[]; polygon_cases=[]
for units,custom in [('meters',False),('kilometers',False),('meters',True)]:
 scale=1000 if units=='kilometers' else 1
 radius=(6400000 if custom else 6371008.8)/scale
 area_radius=(6400000 if custom else 6371007.180918475)/scale
 context=dict(coordinateSpace='longitude-latitude',metric='great-circle',units=units)
 if custom:context['sphereRadius']=radius
 geod=pyproj.Geod(a=radius,b=radius)
 geometry_bound=16*float(np.spacing(np.float32(np.abs(positions).max())))*np.pi/180*radius
 def measure(piece):
  return sum(geod.line_length(*np.array(part.coords).T) for part in shapely.get_parts(piece) if part.geom_type=='LineString' and part.length>0)
 def budget(lengths,counts):
  operations=np.maximum(counts-1,0); gamma=operations*epsilon/(1-operations*epsilon)
  return counts*geometry_bound+(gamma+2e-6)*lengths
 frames=[]
 for grid in [[-87.66,41.86,.005,.005],[-87.65,41.87,.0025,.004]]:
  minimum_x,minimum_y,width,height=map(float,np.array(grid,dtype=np.float32))
  cells=[box(minimum_x+column*width,minimum_y+row*height,minimum_x+(column+1)*width,minimum_y+(row+1)*height) for row in range(8) for column in range(8)]
  lengths=np.zeros(64); counts=np.zeros(64,dtype=int)
  for _,segment in segments:
   for index in STRtree(cells).query(segment,predicate='intersects'):
    piece=segment.intersection(cells[index]); row,column=divmod(int(index),8)
    left,bottom,right,top=cells[index].bounds
    if column<7:piece=piece.difference(LineString([(right,bottom),(right,top)]))
    if row<7:piece=piece.difference(LineString([(left,top),(right,top)]))
    value=measure(piece)
    if value>0:lengths[index]+=value; counts[index]+=1
  areas=[]; area_errors=[]
  for row in range(8):
   south=np.radians(minimum_y+row*height);north=np.radians(minimum_y+(row+1)*height)
   integral=quad(np.cos,south,north,epsabs=1e-14)[0]
   areas.extend([area_radius**2*np.radians(width)*integral]*8)
   area_errors.extend([16*epsilon/abs(integral)+4*epsilon]*8)
  areas=np.array(areas); densities=lengths/areas; bounds=budget(lengths,counts)
  frames.append(dict(parameters=[minimum_x,minimum_y,width,height],lengths=lengths.tolist(),densities=densities.tolist(),lengthBounds=bounds.tolist(),densityBounds=(bounds/areas+np.array(area_errors)*densities).tolist(),areas=areas.tolist(),positivePieces=int(counts.sum()),contributors=counts.tolist()))
 density_cases.append(dict(name=f'200-complete-Chicago-roads-{units}-custom={custom}',spatialContext=context,positions=positions.flatten().tolist(),lineOffsets=offsets,columns=8,rows=8,frames=frames))
 features=[]
 for row in range(3):
  for column in range(3):
   left=-87.655+column*.015;bottom=41.865+row*.01
   shell=box(left,bottom,left+.02,bottom+.015)
   features.append(Polygon(shell.exterior.coords,[box(left+.004,bottom+.004,left+.008,bottom+.008).exterior.coords]))
 features=[shapely.transform(feature,lambda coordinates:coordinates.astype(np.float32).astype(float)) for feature in features]
 polygon_positions=[];ring_offsets=[0];polygon_offsets=[0];feature_offsets=[0]
 for feature in features:
  for ring in [feature.exterior,*feature.interiors]:polygon_positions.extend(np.asarray(ring.coords)[:-1].flatten().tolist());ring_offsets.append(len(polygon_positions)//2)
  polygon_offsets.append(len(ring_offsets)-1);feature_offsets.append(len(polygon_offsets)-1)
 for weight_rule in ['none','speed','indicator']:
  use_weights=weight_rule!='none'
  weights=np.array([1 if weight_rule=='none' else (line%3)*25 if weight_rule=='speed' else line%2 for line in range(200)])
  lengths=np.zeros(9);weighted=np.zeros(9);counts=np.zeros(9,dtype=int);pieces=np.zeros(9,dtype=int);weighted_pieces=np.zeros(9);candidate_count=0
  tree=STRtree(features)
  for line,segment in segments:
   candidate_count+=len(tree.query(segment))
   for index in tree.query(segment,predicate='intersects'):
    piece=segment.intersection(features[index]);value=measure(piece)
    if value>0:
     lengths[index]+=value;weighted[index]+=value*weights[line];counts[index]+=1
     count=sum(part.geom_type=='LineString' and part.length>0 for part in shapely.get_parts(piece));pieces[index]+=count;weighted_pieces[index]+=count*weights[line]
  polygon_cases.append(dict(name=f'200-roads-holes-{units}-custom={custom}-weights={weight_rule}',spatialContext=context,useWeights=use_weights,positions=positions.flatten().tolist(),lineOffsets=offsets,lineWeights=weights.tolist(),polygons=dict(positions=polygon_positions,ringOffsets=ring_offsets,polygonOffsets=polygon_offsets,featureOffsets=feature_offsets),expected=dict(lengths=lengths.tolist(),weightedLengths=weighted.tolist(),segmentCounts=counts.tolist()),lengthBounds=budget(lengths,pieces).tolist(),weightedBounds=(weighted_pieces*geometry_bound+(2e-6+(pieces+counts)*epsilon)*weighted).tolist(),candidateCount=candidate_count,stationarySegmentCount=0))
# Independently generated planar lengths also cover the absent-weight input.
planar_source=HERE/'fixtures/parity-wp2-line-length-per-polygon.json'
planar=json.loads(planar_source.read_text())['cases'][0]
polygon_cases.append({**planar,'name':'planar-default-unit-weights','spatialContext':{'coordinateSpace':'planar','metric':'native','units':'meters'},'useWeights':False,'expected':{**planar['expected'],'weightedLengths':planar['expected']['lengths']},'weightedBounds':planar['lengthBounds']})
source_info[planar_source.name]=hashlib.sha256(planar_source.read_bytes()).hexdigest()
emit('spherical-density',density_cases);emit('spherical-polygons',polygon_cases)
