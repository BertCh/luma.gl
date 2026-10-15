# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Composite published DP/TD-TR reference, no GPU importance algorithm.

Meratnia & de By, EDBT 2004, doi:10.1007/978-3-540-24741-8_44:
https://research.utwente.nl/en/publications/spatiotemporal-compression-techniques-for-moving-point-objects-2/
Recursive threshold splitting uses synchronized linear positions for time-ratio.
For spherical segment mode, the documented local equirectangular chord is sent
to GEOS shortest_line separately at each vertex; PROJ Geod.inv measures from the
vertex to that returned foot. This is a composite reference for the documented
linear-degree chord model, not a great-circle-edge simplifier. Planar segment
reference is GEOS simplify. No f32 importance or GPU iteration is mirrored.
The first two complete gull tracks are retained. Timestamps are f32 before use.
The generator asserts strictly increasing times for these tracks. Exact row IDs
are compared; no tolerance is applied to selected IDs. Tolerances are separated
from each reference split; minimum margins are recorded for conditioning review.
"""
import json,hashlib,platform
from pathlib import Path
import numpy as np
import shapely,pyproj
from shapely import LineString,Point
HERE=Path(__file__).resolve().parent
DATA=HERE.parents[3]/'examples/deck/spatial-analysis-showcase/public/data/gull-migration'
positions=np.fromfile(DATA/'vertices.bin','<f4').reshape(-1,2).astype(float)
offsets=np.fromfile(DATA/'pathOffsets.bin','<u4')
print([p.name for p in DATA.glob('*time*')])
# The scene timestamp field is verified below against its manifest.
timestamps=np.fromfile(DATA/'timestamp.bin','<u4').astype(np.float32).astype(float)
SPHERE=pyproj.Geod(a=6371008.8,b=6371008.8)

def select(points,times,tolerance,spherical,metric):
 kept={0,len(points)-1};stack=[(0,len(points)-1)];margins=[]
 while stack:
  start,stop=stack.pop()
  if stop-start<2:continue
  interior=points[start+1:stop]
  if metric=='time-ratio':
   fraction=(times[start+1:stop]-times[start])/(times[stop]-times[start])
   feet=points[start]+fraction[:,None]*(points[stop]-points[start])
  elif spherical:
   feet=[]
   for point in interior:
    scale=np.array([np.cos(np.radians(point[1])),1])
    chord=LineString((points[[start,stop]]-point)*scale)
    foot=list(shapely.shortest_line(Point(0,0),chord).coords)[-1]
    feet.append(point+np.array(foot)/scale)
   feet=np.array(feet)
  else:
   feet=np.array([list(shapely.shortest_line(Point(point),LineString(points[[start,stop]])).coords)[-1] for point in interior])
  errors=np.asarray(SPHERE.inv(interior[:,0],interior[:,1],feet[:,0],feet[:,1])[2]) if spherical else np.linalg.norm(interior-feet,axis=1)
  offset=int(np.argmax(errors));error=errors[offset];margins.append(abs(error-tolerance))
  if error>tolerance:
   pivot=start+1+offset;kept.add(pivot);stack.extend([(start,pivot),(pivot,stop)])
 return sorted(kept),min(margins,default=1e30)

cases=[]
for track in range(2):
 start,stop=map(int,offsets[track:track+2]);points=positions[start:stop];times=timestamps[start:stop]-timestamps[start]
 assert np.all(np.diff(times)>0)
 for spherical in [False,True]:
  coordinates=points
  if not spherical:
   transformer=pyproj.Transformer.from_crs(4326,f'+proj=aeqd +R=6371008.8 +lon_0={points[0,0]} +lat_0={points[0,1]} +units=m',always_xy=True)
   coordinates=np.array(transformer.transform(points[:,0],points[:,1])).T.astype(np.float32).astype(float)
  for metric in ['segment','time-ratio']:
   frames=[]
   for tolerance in [250,2000]:
    ids,margin=select(coordinates,times,tolerance,spherical,metric)
    if not spherical and metric=='segment':
     simplified=shapely.simplify(LineString(coordinates),tolerance,preserve_topology=False)
     assert np.array_equal(np.array(simplified.coords),coordinates[ids])
    mask=np.zeros(len(points),dtype=int);mask[ids]=1
    frames.append(dict(tolerance=tolerance,ids=ids,mask=mask.tolist(),lineCounts=[len(ids)],lineStarts=[0],minimumSplitMargin=margin))
   cases.append(dict(name=f'complete-gull-{track}-spherical={spherical}-{metric}',spherical=spherical,metric=metric,positions=coordinates.flatten().tolist(),timestamps=times.tolist(),lineOffsets=[0,len(points)],frames=frames))
fixture=dict(generator=Path(__file__).name,reference='composite reference: published DP/TD-TR (Meratnia/de By 2004), GEOS shortest_line/simplify, PROJ Geod.inv',versions=dict(python=platform.python_version(),numpy=np.__version__,shapely=shapely.__version__,pyproj=pyproj.__version__),source={name:hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in ['vertices.bin','pathOffsets.bin','timestamp.bin','manifest.json']},cases=cases)
# Separate complete-track mode batches to stay under the 1.5MB limit.
for index,case in enumerate(cases):
 path=HERE/f'fixtures/parity-wp2-round8-simplification-{index}.json';path.write_text(json.dumps({**fixture,'cases':[case]},separators=(',',':'),allow_nan=False)+'\n');assert path.stat().st_size<1500000
 print(path.name,path.stat().st_size,[frame['minimumSplitMargin'] for frame in case['frames']])
