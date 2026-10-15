# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent rhumb and custom-model reference (f32 inputs before calls).

Rhumbs: PyGeodesy sphericalTrigonometry.LatLon rhumbDistanceTo/AzimuthTo/
Destination (exact=False spherical loxodromes, explicit radius).
The existing gpu-rhumb.spec.ts and rhumbMidpoint helper define the midpoint as
linear in unrolled longitude/latitude, not half-distance along a loxodrome.
Use GEOS line_interpolate_point for that representation convention, explicitly
separate from PyGeodesy's true rhumbMidpointTo (which differs on long edges).
PROJ Geod for sphere/ellipsoid. Midpoints in ellipsoidal mode are spherical per
public API. Longitudes unrolled relative to origin. No antipodal or pole
ambiguities in this added corpus. D: .001 meters +2e-6 relative for distance;
8 ulps of coordinate degrees for positions; 1e-4 degrees for bearings. Rhumb
near-parallel cases deliberately retain these bounds to expose cancellation.
Sphere direct arrival bearing is reconstructed from a rounded destination: its
atan2 derivative has norm 1/abs(sin(distance/radius)). Two endpoint coordinate
ulps therefore add 2*ulp32(coordinate)/abs(sin(distance/radius)) degrees, the
same pre-existing derived conditioning budget as parity-wp2-geodesics.py.
"""
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import pyproj
import pygeodesy
import shapely
from shapely import LineString
from pygeodesy.sphericalTrigonometry import LatLon
HERE=Path(__file__).resolve().parent
source=HERE/'fixtures/parity-wp2-geodesic-pairs-ecco-00.json'
data=json.loads(source.read_text())['cases'][0]
origins=np.asarray(data['origins'],dtype=np.float32).reshape(-1,2)[:128].astype(float)
targets=np.asarray(data['targets'],dtype=np.float32).reshape(-1,2)[:128].astype(float)
# Long and short edges, E/W limit, date line, southern hemisphere.
extra=np.array([[10,20,40,50],[179,25,-179,25],[-75,-40,-70,-35],[20,30,20.0001,30],[30,10,30,20]],dtype=np.float32).astype(float)
origins=np.vstack([origins,extra[:,:2]]); targets=np.vstack([targets,extra[:,2:]])

def continuous(longitude,origin): return origin+(longitude-origin+180)%360-180
cases=[]
for metric in ['rhumb','great-circle','ellipsoidal']:
 for units in ['meters','kilometers']:
  scale=1000 if units=='kilometers' else 1
  radius=(6371008.8 if metric=='rhumb' else 6400000)/scale
  context=dict(coordinateSpace='longitude-latitude',metric=metric,units=units)
  if metric=='ellipsoidal': context['ellipsoid']=dict(semiMajorAxis=6400000,flattening=1/300)
  else: context['sphereRadius']=radius
  geod=pyproj.Geod(a=6400000,f=1/300) if metric=='ellipsoidal' else pyproj.Geod(a=radius*scale,b=radius*scale)
  midpoint_geod=pyproj.Geod(a=6371008.8,b=6371008.8) if metric=='ellipsoidal' else geod
  distances=[]; bearings=[]; final=[]; midpoints=[]; destinations=[]; destination_bearings=[]; input_bearings=[]; input_distances=[]
  for origin,target in zip(origins,targets):
   if metric=='rhumb':
    first=LatLon(origin[1],origin[0]); second=LatLon(target[1],target[0])
    distance=first.rhumbDistanceTo(second,radius=radius); bearing=first.rhumbAzimuthTo(second); last=bearing
    middle=shapely.line_interpolate_point(LineString([origin,[continuous(target[0],origin[0]),target[1]]]),.5,normalized=True)
    midpoint=list(middle.coords[0])
   else:
    bearing,back,distance=geod.inv(*origin,*target); distance/=scale; last=(back+360)%360-180
    midpoint=list(midpoint_geod.npts(*origin,*target,1)[0]); midpoint[0]=continuous(midpoint[0],origin[0])
   distances.append(distance); bearings.append(bearing); final.append(last); midpoints.extend(midpoint)
   input_bearing=float(np.float32(bearing)); input_distance=float(np.float32(distance))
   input_bearings.append(input_bearing); input_distances.append(input_distance)
   if metric=='rhumb':
    dest=first.rhumbDestination(input_distance,input_bearing,radius=radius)
    destinations.extend([continuous(dest.lon,origin[0]),dest.lat]); destination_bearings.append(input_bearing)
   else:
    lon,lat,back=geod.fwd(*origin,input_bearing,input_distance*scale)
    destinations.extend([continuous(lon,origin[0]),lat]); destination_bearings.append((back+360)%360-180)
  destination_bounds=np.full(len(origins),1e-4)
  if metric=='great-circle':
   coordinate_scale=np.maximum(1,np.maximum(np.abs(origins),np.abs(np.array(destinations).reshape(-1,2))).max(axis=1))
   destination_bounds += 2*np.spacing(coordinate_scale.astype(np.float32)).astype(float)/np.maximum(np.abs(np.sin(np.array(input_distances)/radius)),1e-15)
  cases.append(dict(destinationBearingBounds=destination_bounds.tolist(),name=f'{metric}-{units}',spatialContext=context,origins=origins.flatten().tolist(),targets=targets.flatten().tolist(),distances=distances,initialBearings=bearings,finalBearings=final,midpoints=midpoints,bearings=input_bearings,inputDistances=input_distances,destinations=destinations,destinationBearings=destination_bearings,distanceAbsolute=.001/scale,positionAbsolute=8*float(np.spacing(np.float32(180)))))
fixture=dict(generator=Path(__file__).name,reference='PyGeodesy sphericalTrigonometry.LatLon.rhumb* (spherical); GEOS linear-degree midpoint adapter; pyproj.Geod.inv/fwd/npts',versions=dict(python=platform.python_version(),numpy=np.__version__,pyproj=pyproj.__version__,pygeodesy=pygeodesy.__version__,shapely=shapely.__version__),source={source.name:hashlib.sha256(source.read_bytes()).hexdigest()},cases=cases)
path=HERE/'fixtures/parity-wp2-round8-geodesics.json'; path.write_text(json.dumps(fixture,separators=(',',':'),allow_nan=False)+'\n'); assert path.stat().st_size<1500000
print(path.name,path.stat().st_size)
