# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent contour segments and assembled isoband rings.

https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.find_contours
https://shapely.readthedocs.io/en/stable/reference/shapely.union_all.html

Inputs and cell polygons come from parity-wp4-isobands.json (the reproducible
Round 4 scikit-image/Shapely reference), never a TS/WGSL implementation. Verify
its raw source hashes. Rings polygonize independent cell boundaries and union them by band, retaining
all boundary vertices, holes and their owning shell. The added annulus uses
independent polygonize/triangulate faces. Union uses a 2^-36 pixel
precision grid: <=2^-37 per axis, six orders smaller than the G budget, to join
f64 copies of shared intersections. This is reference robustness, not a GPU
snap allowance. Shapely orients shells CCW, holes CW in local x-right/y-up.

Isolines call find_contours on EACH valid 2x2 cell, choosing connectivity using
the documented arithmetic-centre convention (not the asymptotic decider).
Negating values adapts equality-high; positive_orientation='high' puts original
high values on the left after swapping (row,col) to (x,y). Zero-length corner
bridges omitted by skimage are recovered by the high-side limit: perturb ONLY
equal corner samples by < min(nonzero gap)/2^20 for library topology, then
re-evaluate every crossing on its original scalar edge, preserving exact
zero-length segments and global edge IDs. This is a composite equality adapter,
not a claim about unmodified skimage output. Real levels use +1/8 offsets from
quarter-unit samples, separating precision tests from equality tests. Synthetic masks
include both saddle signs and an exact saddle-centre tie at level 0. Their corners are off-level; separate exact-break and
point/plateau cases exercise equality. The nested annulus uses break 1 between values 0 and 2.
Every segment and global edge ID is retained; ordering within
a cell is canonicalized by edge IDs, not skimage's incidental output order.
Scale: 65x65 bounded terrain domain (not global/512x512 geometry); samples at
(col+.5,row+.5), row zero increasing y; no CRS claim. Whole invalid-corner cells
are removed. All inputs rounded to f32 before f64 references. G: 8 ULPs at
coordinate scale, exact counts/topology. No tolerance fitted to GPU results.
"""
import base64
import gzip
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import shapely
import skimage
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import polygonize, unary_union, triangulate
from shapely.geometry.polygon import orient
from skimage.measure import find_contours

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
SOURCE=HERE/'fixtures/parity-wp4-isobands.json'
original=json.loads(SOURCE.read_text())
for source in original['sources']:
    assert hashlib.sha256((ROOT/source['path']).read_bytes()).hexdigest()==source['sha256']
VERSIONS=dict(python=platform.python_version(),numpy=np.__version__,shapely=shapely.__version__,skimage=skimage.__version__)


def unpack(value):
    return np.asarray(json.loads(gzip.decompress(base64.b64decode(value))),dtype=float)


def pack(value):
    numbers=[None if np.isnan(item) else 'Infinity' if item==np.inf else '-Infinity' if item==-np.inf else float(item) for item in np.asarray(value).ravel()]
    return base64.b64encode(gzip.compress(json.dumps(numbers,separators=(',',':')).encode(),mtime=0)).decode()


def make_rings(scene):
    triangles=unpack(scene['triangles']).reshape(-1,8)
    cell_edges=unpack(scene['edges']).reshape(-1,6) if 'edges' in scene else None
    rings=[]
    for band in range(scene['firstBand'],scene['lastBand']+1):
        if cell_edges is None:
            pieces=[Polygon(triangle[2:].reshape(3,2)) for triangle in triangles if triangle[1]==band]
        else:
            pieces=[]
            selected=cell_edges[cell_edges[:,1]==band]
            for cell in np.unique(selected[:,0]):
                pieces.extend(polygonize([LineString(edge[2:].reshape(2,2)) for edge in selected[selected[:,0]==cell]]))
        if not pieces: continue
        region=shapely.union_all(pieces,grid_size=2**-36)
        polygons=[region] if region.geom_type=='Polygon' else list(region.geoms)
        for polygon in polygons:
            polygon=orient(polygon,sign=1)
            shell=len(rings)
            for is_hole,ring in [(0,polygon.exterior),*[(1,ring) for ring in polygon.interiors]]:
                coordinates=np.asarray(ring.coords)
                rings.append(dict(band=band,isHole=is_hole,shell=shell,
                    area=Polygon(ring).area*(-1 if is_hole else 1),
                    positions=coordinates.ravel().tolist()))
    return dict(**{key:scene[key] for key in ['name','width','height','values','validity','breaks','firstBand','lastBand']},
        reference='composite: skimage.measure.find_contours + shapely.union_all/orient',rings=rings)


def make_lines(scene):
    width,height=scene['width'],scene['height']
    values=unpack(scene['values']).reshape(height,width)
    validity=unpack(scene['validity']).reshape(height,width)
    # Real raster is on a quarter-unit grid; the +1/8 levels cannot hit corners.
    levels=[value+.125 for value in scene['breaks']] if scene['name'].startswith('grand-canyon') else [-.5,0.,.5]
    if scene['name'].startswith('exact-breaks'): levels=[1.,3.,5.,7.]
    if scene['name'].startswith(('nested-','corner-')): levels=[1.]
    usable=np.isfinite(values)&(values!=-9999)&(validity!=0)
    segments=[]
    for row in range(height-1):
        for column in range(width-1):
            if not usable[row:row+2,column:column+2].all(): continue
            cell_values=values[row:row+2,column:column+2]
            for level_index,level in enumerate(levels):
                # Equality belongs to high. skimage drops degenerate segments;
                # use the high-side limit only for topology, then evaluate every
                # crossing on the ORIGINAL scalar edge, restoring exact corners.
                # epsilon < min(nonzero gap)/2^20 cannot cross any other corner.
                nonzero_gaps=np.abs(cell_values[cell_values!=level]-level)
                epsilon=(nonzero_gaps.min() if nonzero_gaps.size else 1.)*2**-20
                topology_values=np.where(cell_values==level,level+epsilon,cell_values)
                contours=find_contours(-topology_values,-level,
                    fully_connected='low' if np.mean(cell_values)>=level else 'high',positive_orientation='high')
                for contour in contours:
                    assert len(contour)==2
                    coordinates=contour[:,::-1]
                    def edge_id(point):
                        horizontal_count=height*(width-1)
                        if point[1]==0: return row*(width-1)+column
                        if point[1]==1: return (row+1)*(width-1)+column
                        if point[0]==0: return horizontal_count+row*width+column
                        assert point[0]==1
                        return horizontal_count+row*width+column+1
                    edge_ids=[edge_id(point) for point in coordinates]
                    exact_coordinates=[]
                    for point in coordinates:
                        if point[1] in (0,1):
                            edge_row=int(point[1])
                            first,second=cell_values[edge_row,:]
                            exact_coordinates.append([(level-first)/(second-first),float(edge_row)])
                        else:
                            edge_column=int(point[0])
                            first,second=cell_values[:,edge_column]
                            exact_coordinates.append([float(edge_column),(level-first)/(second-first)])
                    coordinates=np.asarray(exact_coordinates)+[column+.5,row+.5]
                    segments.append([row*(width-1)+column,level_index,*edge_ids,*coordinates.ravel()])
    segments.sort(key=lambda segment:segment[:4])
    return dict(name=scene['name'],reference='composite: skimage.measure.find_contours with per-cell centre connectivity',
        width=width,height=height,values=scene['values'],validity=scene['validity'],levels=levels,segments=pack(segments),segmentCount=len(segments))

# A closed annulus plus an inner island ensures nested shells and holes, and
# a masked outer sample creates a domain notch. Reference faces use polygonize.
rows,columns=np.indices((17,17))
radius=np.maximum(np.abs(columns-8),np.abs(rows-8))
values=np.where(((radius>=3)&(radius<=6))|(radius<=1),2.,0.)
validity=np.ones_like(values,dtype=np.uint32); validity[1,8]=0
square=shapely.box(0,0,1,1)
triangles=[]
for row in range(16):
    for column in range(16):
        if not validity[row:row+2,column:column+2].all(): continue
        cell_values=values[row:row+2,column:column+2]
        contours=find_contours(-cell_values,-1.,fully_connected='low' if np.mean(cell_values)>=1 else 'high')
        faces=list(polygonize(unary_union([square.boundary,*[LineString(contour[:,::-1]) for contour in contours]])))
        high_points=[Point(column_index,row_index) for row_index,column_index in np.ndindex(2,2) if cell_values[row_index,column_index]>1]
        high=unary_union([face for face in faces if any(face.covers(point) for point in high_points)])
        for band,region in enumerate([square.difference(high),high]):
            for triangle in triangulate(region):
                if not region.covers(triangle): continue
                coordinates=np.asarray(orient(triangle,sign=1).exterior.coords[:-1])+[column+.5,row+.5]
                triangles.extend([row*16+column,band,*coordinates.ravel()])
original['cases'].append(dict(name='nested-annulus-island-and-domain-notch',width=17,height=17,values=pack(values),validity=pack(validity),breaks=[1.],firstBand=0,lastBand=1,triangles=pack(triangles)))

line_cases=[make_lines(scene) for scene in original['cases']]
corner_values=np.zeros((7,7)); corner_values[1,1]=1.; corner_values[4:6,4:6]=1.; corner_values[1:3,4:6]=2.
line_cases.append(make_lines(dict(name='corner-ties-point-plateau-and-high-region',width=7,height=7,
    values=pack(corner_values),validity=pack(np.ones_like(corner_values)))))

fixture=dict(versions=VERSIONS,sources=[*original['sources'],dict(path=str(SOURCE.relative_to(ROOT)),sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest())],
    ringCases=[make_rings(scene) for scene in original['cases']],lineCases=line_cases)
output=HERE/'fixtures/parity-wp4-contours.json'
output.write_text(json.dumps(fixture,separators=(',',':'))+'\n')
assert output.stat().st_size<1_500_000
print(output.name,output.stat().st_size)
for scene in fixture['ringCases']:print(scene['name'],'rings',len(scene['rings']),'vertices',sum(len(ring['positions'])//2 for ring in scene['rings']))
for scene in fixture['lineCases']:print(scene['name'],'segments',scene['segmentCount'])
