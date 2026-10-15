# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Shapely oriented-edge, area/perimeter and closed-cell boundary reference.

Only valid planar windows are in this API. make_valid, projection, antimeridian
splitting are INPUT adapters, not alleged operations of StudyWindowPreparation.
Use Shapely orient_polygons for shell/hole direction; index each original edge
into the oriented ring's directed edge set. Cells are kept if a center is inside
or any boundary touches the CLOSED cell; boundary-bit makes center ties irrelevant.
Projected coordinates are snapped to 1/4m then f32 before GEOS reference evaluation.
This tests preparation of pyproj-projected inputs, not GPU projection accuracy.
G profile: exact edge endpoints, offsets, sample IDs and masks; S summary scalars.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import shapely
from shapely.geometry import shape, Polygon, MultiPolygon, box
from shapely.ops import transform
import pyproj

ROOT=Path(__file__).resolve().parents[4]

def makeCase(name,geometry,grid=32,provenance=None):
    polygons=list(geometry.geoms) if geometry.geom_type=='MultiPolygon' else [geometry]
    positions=[]; rings=[0]; polygonsOffsets=[0]; edges=[]
    for polygon in polygons:
        oriented=shapely.orient_polygons(polygon)
        for originalRing, orientedRing in zip([polygon.exterior,*polygon.interiors],[oriented.exterior,*oriented.interiors]):
            points=list(originalRing.coords)[:-1]
            directed={(tuple(start),tuple(end)) for start,end in zip(list(orientedRing.coords)[:-1],list(orientedRing.coords)[1:])}
            positions.extend(points)
            for start,end in zip(points,points[1:]+points[:1]):
                edges.extend([*start,*end] if (tuple(start),tuple(end)) in directed else [*end,*start])
            rings.append(len(positions))
        polygonsOffsets.append(len(rings)-1)
    bounds=np.asarray(geometry.bounds,dtype=np.float32)
    cells=[]
    # Parameter division and cell coordinates mirror f32 input quantization ONLY;
    # all geometric predicates come from GEOS, not a port of rasterization.
    step=np.asarray((bounds[2:]-bounds[:2])/grid,dtype=np.float32)
    for row in range(grid):
        for column in range(grid):
            lower=np.asarray(bounds[:2]+np.asarray([column,row],dtype=np.float32)*step,dtype=np.float32)
            upper=np.asarray(lower+step,dtype=np.float32)
            center=np.asarray(lower+step/np.float32(2),dtype=np.float32)
            boundary=geometry.boundary.intersects(box(*lower,*upper))
            if boundary or geometry.contains(shapely.Point(center)):
                cells.append((row*grid+column) | (0x80000000 if boundary else 0))
    return dict(name=name,reference='shapely.orient_polygons/area/length/contains/intersects; pyproj.Transformer input adapter',positions=np.asarray(positions).ravel().tolist(),ringOffsets=rings,polygonOffsets=polygonsOffsets,featureOffsets=[0,len(polygons)],samplingGrid=[grid,grid],summary=[*geometry.bounds,geometry.area,geometry.length,len(positions),0],edges=edges,sampleCells=cells,provenance=provenance)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output-dir',type=Path,default=Path(__file__).parent/'fixtures'); args=parser.parse_args()
    shell=[(0,0),(0,12),(16,12),(16,0)]
    hole=[(3,3),(8,3),(8,9),(3,9)]
    base=MultiPolygon([Polygon(shell,[hole]),box(20,2,24,7)])
    cases=[makeCase('reversed-shell-hole-and-island',base),makeCase('rectangle',box(0,0,16,12)),makeCase('viewport-clip',shapely.intersection(base,box(6,-2,22,10)))]
    source=ROOT/'examples/deck/spatial-analysis-showcase/public/data/chicago-boundary/city.geojson'
    raw=json.loads(source.read_text()); geographic=shapely.union_all([shape(feature['geometry']) for feature in raw['features']])
    projection='+proj=laea +lat_0=41.85 +lon_0=-87.7 +R=6371007.180918474 +units=m +no_defs'
    transformer=pyproj.Transformer.from_crs('EPSG:4326',projection,always_xy=True)
    projected=transform(transformer.transform,geographic)
    projected=shapely.set_precision(projected,.25)
    projected=shapely.make_valid(projected)
    provenance=dict(sourceSha256=hashlib.sha256(source.read_bytes()).hexdigest(),projection=projection,snap=.25)
    cases.append(makeCase('chicago-city-projected',projected,64,provenance))
    minimumX,minimumY,maximumX,maximumY=projected.bounds
    clipped=shapely.set_precision(shapely.intersection(projected,box(minimumX,minimumY,(minimumX+maximumX)/2,(minimumY+maximumY)/2)),.25)
    cases.append(makeCase('chicago-view-projected',clipped,64,provenance))
    fixture=dict(generator=Path(__file__).name,command=f'/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-deck-gl-master/82c646e8-3e98-44d4-84b8-212745483f4a/scratchpad/eval/venv/bin/python -I {Path(__file__).relative_to(ROOT)}',versions={'numpy':np.__version__,'shapely':shapely.__version__,'GEOS':shapely.geos_version_string,'pyproj':pyproj.__version__,'PROJ':pyproj.proj_version_str},cases=cases)
    args.output_dir.mkdir(exist_ok=True,parents=True); path=args.output_dir/'parity-wp1-study-window.json'; path.write_text(json.dumps(fixture,separators=(',',':'))+'\n'); assert path.stat().st_size<1500000
    print(path,len(cases),path.stat().st_size)
if __name__=='__main__':main()
