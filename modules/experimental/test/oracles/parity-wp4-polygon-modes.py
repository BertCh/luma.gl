# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Rasterio pixel-centre rasterization and sampling for signed rows and longitude copies.

Two overlapping rectangles have no centres on an edge. Rasterio's reverse burn
order implements lowest-feature ownership. Longitude input is stored wrapped;
Shapely explicitly translates the unwrapped geometry by +/-360 before rasterio.
Both sides of the antimeridian and both row directions. The GPU rasterizer feeds
the GPU join; expected full labels, point zones and counts come only from rasterio.
Planar variants test the scene's with-boundary and zones-only compilation paths.
"""
import json,platform
from pathlib import Path
import numpy as np
import rasterio,shapely
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.io import MemoryFile
from rasterio.sample import sample_gen
from shapely.geometry import box,mapping
from shapely.affinity import translate
HERE=Path(__file__).resolve().parent
NONE=2**32-1
cases=[]
for geographic in [False,True]:
 for west in ([178,-182] if geographic else [0]):
  for direction in ['north','south']:
   width,height=8,8; south=0; north=4
   original=[box(179 if geographic else 1,1,181 if geographic else 3,3),box(180 if geographic else 2,.5,181.5 if geographic else 3.5,2.5)]
   geometry=[translate(poly,xoff=-360 if west<0 else 0) for poly in original]
   transform=Affine(.5,0,west,0,.5 if direction=='north' else -.5,south if direction=='north' else north)
   labels=rasterize([(mapping(poly),index) for index,poly in reversed(list(enumerate(geometry)))],out_shape=(height,width),transform=transform,fill=NONE,dtype='uint32')
   positions=[]
   for poly in original:
    for x,y in list(poly.exterior.coords)[:-1]:positions.extend([((x+180)%360-180) if geographic else x,y])
   points=[transform*(column+.5,row+.5) for row,column in np.ndindex((height,width))]+[(west-.5,2),(west+4.5,2)]
   with MemoryFile() as memory:
    with memory.open(driver='GTiff',width=width,height=height,count=1,dtype='uint32',nodata=NONE,transform=transform) as dataset:
     dataset.write(labels,1);point_zones=np.array([value[0] for value in sample_gen(dataset,points)],dtype=np.uint32)
   counts=np.bincount(point_zones[point_zones!=NONE],minlength=2)
   cases.append(dict(name=f'{"longitude" if geographic else "planar"}-{west}-{direction}',geographic=geographic,width=width,height=height,
    rowDirection=direction,origin=[west,south if direction=='north' else north],cellSize=[.5,.5],positions=positions,points=np.asarray(points).ravel().tolist(),
    zones=labels.ravel().tolist(),pointZones=point_zones.tolist(),counts=counts.tolist()))
fixture=dict(reference='rasterio.features.rasterize + rasterio.sample.sample_gen; shapely.affinity.translate longitude copies',
 versions=dict(python=platform.python_version(),numpy=np.__version__,rasterio=rasterio.__version__,shapely=shapely.__version__),cases=cases)
output=HERE/'fixtures/parity-wp4-polygon-modes.json';output.write_text(json.dumps(fixture,separators=(',',':'))+'\n');print(output.name,output.stat().st_size)
