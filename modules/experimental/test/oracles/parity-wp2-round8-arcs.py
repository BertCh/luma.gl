# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""PROJ vertices for every showcase subdivision control combination.

Reuses the earlier independent PROJ reference function (only its function body,
not its dataset writer). Three routes exercise seam crossing, short and long
arcs. 41 max-length values (0..2000 km by 50) x 64 minimum counts. Airline's
50..1000 km / minimum=2 are a subset. Unique subdivision vertex arrays are
stored once; frames refer to these independent arrays, not GPU output.
Also covers custom sphere radii and kilometer units. Same derived D bounds as
the original arc generator; excluded antipodal ambiguity remains excluded.
"""
import ast,json,platform
from pathlib import Path
import numpy as np
import pyproj
HERE=Path(__file__).resolve().parent
source=HERE/'parity-wp2-great-circle-arcs.py'
function=next(node for node in ast.parse(source.read_text()).body if isinstance(node,ast.FunctionDef) and node.name=='make_case')
namespace=dict(np=np);exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),namespace)
fixtures=[]
for radius,units in [(6371008.8,'meters'),(6400000.,'meters'),(6400.,'kilometers')]:
 namespace['RADIUS']=radius;namespace['SPHERE']=pyproj.Geod(a=radius,b=radius)
 entry=namespace['make_case'](f'showcase-controls-{units}-{radius}',[[179,20],[10,30],[0,0]],[[-179,25],[10.2,30.1],[110,60]],[(0,count,64) for count in range(1,65)])
 distances=[namespace['SPHERE'].inv(*a,*b)[2] for a,b in zip(np.array(entry['sources']).reshape(-1,2),np.array(entry['targets']).reshape(-1,2))]
 settings=[]
 for maximum in (range(0,2001,50) if radius==6371008.8 else [0,50,2000]):
  maximum_length=maximum*(1000 if units=='meters' else 1)
  for minimum in (range(1,65) if radius==6371008.8 else [1,2,24,64]):
   counts=[min(64,max(minimum,int(np.ceil(distance/maximum_length)) if maximum_length else 1)) for distance in distances]
   settings.append(dict(maximumSegmentLength=maximum_length,minimumSegments=minimum,counts=counts))
 fixtures.append(dict(name=entry['name'],sphereRadius=radius,units=units,sources=entry['sources'],targets=entry['targets'],catalog=entry['frames'],settings=settings))
for index,case in enumerate(fixtures):
 path=HERE/f'fixtures/parity-wp2-round8-arcs-{index}.json';path.write_text(json.dumps(dict(generator=Path(__file__).name,reference='pyproj.Geod.inv/npts (explicit sphere)',versions=dict(python=platform.python_version(),numpy=np.__version__,pyproj=pyproj.__version__),cases=[case]),separators=(',',':'))+'\n');assert path.stat().st_size<1500000
 print(path.name,path.stat().st_size,len(case['settings']))
