# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Additional mode references, independent of the TS/WGSL implementation.

Inputs are f32 before reference calls. Encounter pairs use PROJ spherical inverse
on each common-clock sample (no continuous-trajectory claim). Full-longitude
bounds fold longitude; local bounds do not. Threshold clamping follows the public
analytic grid contract; test pairs stay away from that threshold. D tolerance is
8 coordinate ulps converted from degrees to arc length, plus 2e-6 relative.
Join geometry uses GEOS distance/shortest_line and STRtree box intersections;
IDs are input rows, ties ordered by row, polygon interiors have zero distance.
The test corpus has 16 features of each kind and 20 point queries, including ties,
islands and polygon holes. Exact integer coordinates make topology unambiguous.
"""
import json
import platform
from pathlib import Path
import numpy as np
import pyproj
import scipy
import shapely
from shapely import Point, LineString, Polygon, STRtree, box

HERE = Path(__file__).resolve().parent
VERSIONS = dict(python=platform.python_version(), numpy=np.__version__, pyproj=pyproj.__version__, scipy=scipy.__version__, shapely=shapely.__version__)

def write(name, reference, cases):
    path = HERE / 'fixtures' / f'parity-wp2-round8-{name}.json'
    path.write_text(json.dumps(dict(generator=Path(__file__).name, versions=VERSIONS, reference=reference, cases=cases), separators=(',', ':'), allow_nan=False)+'\n')
    assert path.stat().st_size <= 1500000
    print(path.name, path.stat().st_size, len(cases))

samples = np.asarray([[[179.8,20],[-179.9,20],[179.9,20]], [[-179.8,20],[179.9,20],[179.7,20]], [[179.8,20],[179.8,20],[179.8,20]], [[10,70],[10.2,70],[10.4,70]], [[10.3,70],[10.4,70],[10.5,70]], [[0,0],[0,0],[0,0]]], dtype=np.float32).astype(float)
encounters=[]
for wrap in [True,False]:
 for radius,units in [(6371008.8,'meters'),(6371.0088,'kilometers'),(6400000.,'meters')]:
    bounds=[-180,-75,180,75] if wrap else [-20,-75,30,75]
    cell_size=5
    maximum=radius*np.arcsin(np.sin(np.radians(cell_size))*np.cos(np.radians(75)))*(1-1e-6)
    for distance in [maximum/3,maximum*2]:
      geod=pyproj.Geod(a=radius,b=radius)
      pairs={}; hits=0
      for bucket in range(3):
       for first in range(len(samples)):
        for second in range(first+1,len(samples)):
         a,b=samples[first,bucket],samples[second,bucket]
         if not all(bounds[0]<=p[0]<=bounds[2] and bounds[1]<=p[1]<=bounds[3] for p in [a,b]): continue
         measured=geod.inv(*a,*b)[2]
         if measured<=min(float(np.float32(distance)),maximum):
          pair=pairs.setdefault((first,second),dict(ids=first,partners=second,first=bucket,distances=measured,counts=0,times=bucket*10))
          pair['distances']=min(pair['distances'],measured); pair['counts']+=1; hits+=1
      ordered=[pairs[key] for key in sorted(pairs)]
      encounters.append(dict(name=f'sphere-wrap={wrap}-{radius}-{units}-clamp={distance>maximum}',spatialContext=dict(coordinateSpace='longitude-latitude',metric='great-circle',units=units,sphereRadius=radius),samples=samples.flatten().tolist(),bucketTimes=[0,10,20],trackCount=len(samples),bucketCount=3,trackValid=[1]*len(samples),bounds=bounds,distance=distance,cellSize=cell_size,hitCount=hits,distanceAbsolute=max(.001 if units=='meters' else .000001,8*float(np.spacing(np.float32(180)))*np.pi/180*radius),expected={key:[p[key] for p in ordered] for key in ['ids','partners','first','distances','counts','times']}))
# The live vessel class receives only the planar distance control. Clock/speed
# options change its sample data upstream in addClockEncounters, not its mode.
from scipy.spatial import cKDTree
from scipy.spatial.distance import euclidean
import ast
planar_source=HERE/'parity-wp2-trajectory-encounters.py'
function=next(node for node in ast.parse(planar_source.read_text()).body if isinstance(node,ast.FunctionDef) and node.name=='make_case')
namespace=dict(np=np,cKDTree=cKDTree,euclidean=euclidean);exec(compile(ast.Module(body=[function],type_ignores=[]),str(planar_source),'exec'),namespace)
planar_samples=[[[0,0],[0,0],[0,0]],[[0,0],[0,0],[0,0]],[[25,0],[100,0],[400,0]],[[300,100],[350,100],[500,100]]]
for distance in range(10,401,5):
 entry=namespace['make_case'](f'vessel-distance-{distance}',planar_samples,[0,120,240],[-800,-800,800,800],distance,400)
 entry['spatialContext']=dict(coordinateSpace='planar',metric='native',units='meters')
 # Convert NumPy arrays for strict JSON without allowing nonfinite tokens.
 entry={key:value.tolist() if isinstance(value,np.ndarray) else value for key,value in entry.items()}
 encounters.append(entry)
write('encounters','pyproj.Geod.inv (sphere); scipy.cKDTree.query_pairs/euclidean (planar)',encounters)

queries=np.asarray([[x*300+75,y*300+100] for y in range(4) for x in range(4)]+[[0,0],[450,450],[600,600],[4000,4000]],dtype=np.float32).astype(float)
join_cases=[]
for kind in ['points','segments','lines','polygons']:
 features=[]; positions=[]; offsets=[0]; rings=[0]; polygon_offsets=[0]; feature_offsets=[0]
 for index in range(16):
  x,y=(index%4)*300,(index//4)*300
  if kind=='points': coordinates=[[x,y]]; geometry=Point(coordinates[0])
  elif kind in ['segments','lines']:
   coordinates=[[x-100,y-50],[x+100,y+50]]
   if kind=='lines': coordinates.append([x+120,y+180])
   geometry=LineString(coordinates)
  else:
   coordinates=[[x-100,y-100],[x+100,y-100],[x+100,y+100],[x-100,y+100]]
   hole=[[x-20,y-20],[x-20,y+20],[x+20,y+20],[x+20,y-20]]
   geometry=Polygon(coordinates,[hole]); coordinates+=hole
   rings += [len(positions)+4,len(positions)+8]; polygon_offsets.append(len(rings)-1); feature_offsets.append(index+1)
  positions+=coordinates; offsets.append(len(positions)); features.append(geometry)
 distances=shapely.distance(np.array([Point(q) for q in queries])[:,None],np.array(features)[None,:])
 feet=[[list(shapely.shortest_line(Point(query),feature).coords)[-1] for feature in features] for query in queries]
 frames=[]
 for distance in sorted(set(range(0,2001,25)) | set(range(100,8001,100))):
  pairs=STRtree([box(*g.bounds) for g in features]).query([box(q[0]-distance,q[1]-distance,q[0]+distance,q[1]+distance) for q in queries]).T.tolist()
  frames.append(dict(radius=distance,candidates=sorted(pairs) if distance in [0,250,500,1000,2000] else []))
 join_cases.append(dict(name=kind,kind=kind,queries=queries.flatten().tolist(),positions=np.array(positions).flatten().tolist(),lineOffsets=offsets,ringOffsets=rings,polygonOffsets=polygon_offsets,featureOffsets=feature_offsets,distances=distances.tolist(),feet=feet,frames=frames))
write('joins','shapely.distance/shortest_line; STRtree.query on expanded boxes',join_cases)

# All eight k controls and all six inverse-distance powers, including the
# scene's default floor. scipy CSR canonicalizes each independently chosen row.
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
points=np.asarray([[index*30,(index*index%13)*20] for index in range(12)],dtype=np.float32).astype(float)
matrix=cdist(points,points)
weight_cases=[]
for neighbor_count in range(1,9):
 rows=[]; columns=[]; distances=[]
 for row in range(len(points)):
  indices=sorted((index for index in range(len(points)) if index!=row),key=lambda index:(matrix[row,index],index))[:neighbor_count]
  for index in indices: rows.append(row); columns.append(index); distances.append(matrix[row,index])
 sparse=csr_matrix((distances,(rows,columns)),shape=(len(points),len(points))); sparse.sort_indices()
 rules=[dict(weightType='binary',power=1,floor=1e-6,weights=[1]*sparse.nnz)]
 for power in [.5,1,1.5,2,2.5,3]: rules.append(dict(weightType='inverse-distance',power=power,floor=1e-6,weights=(1/np.maximum(sparse.data,1e-6)**power).tolist()))
 weight_cases.append(dict(name=f'showcase-self-k{neighbor_count}',queries=points.flatten().tolist(),features=points.flatten().tolist(),selfJoin=True,k=neighbor_count,absolute=16*float(np.spacing(np.float32(points.max()))),expected=dict(offsets=sparse.indptr.tolist(),neighbors=sparse.indices.tolist(),distances=sparse.data.tolist()),rules=rules))
write('weights','scipy.spatial.distance.cdist + scipy.sparse.csr_matrix; explicit inverse-power definition',weight_cases)

# GEOS supplies all crossings; per-track prefix is the documented event cap.
import ast,math
from shapely.ops import transform
zone_source=HERE/'parity-wp2-zone-events.py'
functions=[node for node in ast.parse(zone_source.read_text()).body if isinstance(node,ast.FunctionDef)]
namespace=dict(np=np,math=math,shapely=shapely,LineString=LineString,Point=Point,Polygon=Polygon,STRtree=STRtree,transform=transform)
exec(compile(ast.Module(body=functions,type_ignores=[]),str(zone_source),'exec'),namespace)
tracks=[[[(-5 if index%2==0 else 35),5+track,index*10+track*1000] for index in range(42)] for track in range(2)]
entry=namespace['make_case']('every-showcase-event-cap',tracks,[box(0,0,10,10),box(20,0,30,10)])
entry['expected']['events'].sort(key=lambda event:(event[0],event[3]))
assert len(set((event[0],event[3]) for event in entry['expected']['events']))==len(entry['expected']['events'])
write('zone-caps','shapely.intersection/line_locate_point + explicit relative-time mapping', [entry])
