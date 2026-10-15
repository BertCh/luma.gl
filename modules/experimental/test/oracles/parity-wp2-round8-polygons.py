# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent PROJ great-circle densification then GEOS ring preparation.

No pole heuristic is mirrored. Four short great-circle edges have unique PROJ
solutions; Geod.npts inserts vertices at the public maximum central-angle step.
The previous independent GEOS clipping/orientation adapter is reused as a
function only. Input ring is f32; reference inserted vertices are f32 before
GEOS. G position budget remains 16 coordinate ulps, never fitted to a GPU run.
"""
import ast,json,platform
from pathlib import Path
import numpy as np
import shapely,pyproj
from shapely import Polygon,box
HERE=Path(__file__).resolve().parent
source=HERE/'parity-wp2-round5-geometry.py'
functions=[node for node in ast.parse(source.read_text()).body if isinstance(node,ast.FunctionDef) and node.name in ['bound','flatten','open_ring','prepare']]
namespace=dict(np=np,shapely=shapely,Polygon=Polygon,box=box);exec(compile(ast.Module(body=functions,type_ignores=[]),str(source),'exec'),namespace)
ring=np.asarray([[10,20],[12.25,20],[12.25,22.25],[10,22.25]],dtype=np.float32).astype(float)
geod=pyproj.Geod(a=6371008.8,b=6371008.8);dense=[];step=.75
for first,second in zip(ring,np.roll(ring,-1,axis=0)):
 angle=np.degrees(geod.inv(*first,*second)[2]/6371008.8);segments=int(np.ceil(angle/step))
 dense.extend([first,*geod.npts(*first,*second,segments-1)])
entry=namespace['prepare']('great-circle-dense-unique-short-edges',[[[dense]]])
entry.update(namespace['flatten']([[[ring]]]))
entry['maximumSegmentDegrees']=step
entry['absolute']=16*float(np.spacing(np.float32(32))) # coordinate scale including curved edges stays below 32 degrees.
path=HERE/'fixtures/parity-wp2-round8-polygons.json'
path.write_text(json.dumps(dict(generator=Path(__file__).name,reference='pyproj.Geod.inv/npts; shapely.intersection/orient_polygons',versions=dict(python=platform.python_version(),numpy=np.__version__,shapely=shapely.__version__,pyproj=pyproj.__version__),cases=[entry]),separators=(',',':'))+'\n')
print(path.name,path.stat().st_size)
