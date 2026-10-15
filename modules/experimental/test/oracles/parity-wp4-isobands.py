# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Composite reference: skimage.measure.find_contours + Shapely polygonization.
https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.find_contours
https://shapely.readthedocs.io/en/stable/manual.html#shapely.ops.polygonize

Marching squares uses linear edge interpolation. Select skimage's high/low
connectivity separately per cell according to the documented arithmetic-centre
saddle convention. Negated samples adapt skimage strict-> equality to the
API >= high-side ownership without perturbing any input/break; this is NOT GDAL/asymptotic-decider equivalence. Polygonize
those independent contours with the cell boundary, identify the above-level
faces by their high-valued corners, and difference nested above-level regions.
Shapely triangulates the resulting polygons, independently of GPU fan order.
No TS/WGSL is imported or translated. Full cell/band geometry, boundaries and
classes are retained, allowing symmetric difference and boundary comparisons.

Fully decode Grand Canyon Terrarium DEM; fixed 65x65 window rows/cols 960:1025,
no resampling. This bounded geometry test covers all 64x64 interpolation cells;
it is not a 512x512 geometry-scale claim. Values round to f32 before float64
reference. Row zero maps to positive local y, samples at (col+.5,row+.5).
Any nodata corner removes the whole interpolation cell. No projection/units
claim: geometry uses local pixel coordinates. Synthetic cases exhaust all 16
binary masks with high/low/tied saddle centres, plus holes, exact breaks and
band windows. G: coordinate bound 8 ulps at raster coordinate scale; per-region
area bound follows perimeter*coordinate error, derived in the TS comparator.
"""
import base64
import gzip
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import rasterio
import shapely
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import polygonize, unary_union, triangulate
from shapely.geometry.polygon import orient
import skimage
from skimage.measure import find_contours

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT/'examples/deck/spatial-analysis-showcase/public/data/grand-canyon-dem/dem.png'
VERSIONS = dict(python=platform.python_version(), numpy=np.__version__, rasterio=rasterio.__version__, shapely=shapely.__version__, skimage=skimage.__version__)


def pack(values):
    numbers = [None if np.isnan(value) else 'Infinity' if value == np.inf else '-Infinity' if value == -np.inf else float(value) for value in np.asarray(values).ravel()]
    return base64.b64encode(gzip.compress(json.dumps(numbers,separators=(',',':')).encode(),mtime=0)).decode()


def above_level(values, level, square):
    if np.all(values >= level): return square
    if np.all(values < level): return Polygon()
    # skimage marks strictly greater samples high; negate to make original
    # values equal to the break belong to the >= region, as the API declares.
    contours = find_contours(-values, -level, fully_connected='low' if np.mean(values) >= level else 'high')
    lines = [square.boundary, *[LineString(contour[:, ::-1]) for contour in contours if len(contour)>1]]
    faces = list(polygonize(unary_union(lines)))
    high = [Point(column,row) for row,column in np.ndindex(2,2) if values[row,column] > level]
    return unary_union([face for face in faces if any(face.covers(point) for point in high)])


def make_case(name, values, breaks, validity=None, first_band=0, last_band=None):
    values=np.asarray(values,dtype=np.float32).astype(float)
    height,width=values.shape
    validity=np.ones_like(values,dtype=np.uint32) if validity is None else validity
    usable=np.isfinite(values)&(values!=-9999)&(validity!=0)
    classes=np.where(usable,np.searchsorted(breaks,values,side='right'),4294967295)
    square=shapely.box(0,0,1,1)
    triangles=[]; edges=[]; areas=[]; components=[]
    last_band=len(breaks) if last_band is None else last_band
    for row in range(height-1):
        for column in range(width-1):
            cell=row*(width-1)+column
            if not usable[row:row+2,column:column+2].all(): continue
            cell_values=values[row:row+2,column:column+2]
            above=[square,*[above_level(cell_values,level,square) for level in breaks],Polygon()]
            for band in range(first_band,last_band+1):
                region=above[band].difference(above[band+1])
                if region.is_empty or region.area==0: continue
                pieces=[region] if region.geom_type=='Polygon' else list(region.geoms)
                area=0
                for piece in pieces:
                    piece=orient(piece,sign=1)
                    assert len(piece.interiors)==0
                    for triangle in triangulate(piece):
                        if not piece.covers(triangle.representative_point()): continue
                        # This guards unconstrained Delaunay edges crossing a concavity.
                        assert triangle.difference(piece).area < 1e-13
                        area += triangle.area
                        coordinates=np.array(orient(triangle,sign=1).exterior.coords[:-1])+[column+.5,row+.5]
                        triangles.extend([cell,band,*coordinates.ravel()])
                    coordinates=np.array(piece.exterior.coords)+[column+.5,row+.5]
                    for start,end in zip(coordinates[:-1],coordinates[1:]):
                        edges.extend([cell,band,*start,*end])
                assert abs(area-region.area)<1e-12
                areas.extend([cell,band,region.area])
                components.extend([cell,band,len(pieces)])
    return dict(name=name,reference='composite: skimage.measure.find_contours + shapely.polygonize/difference/triangulate',
        width=width,height=height,values=pack(values),validity=pack(validity),breaks=breaks,
        firstBand=first_band,lastBand=last_band,classes=pack(classes),triangles=pack(triangles),edges=pack(edges),areas=pack(areas),components=pack(components))

with rasterio.open(SOURCE) as dataset:
    channels=dataset.read().astype(float)
elevation=channels[0]*256+channels[1]+channels[2]/256-32768
real=elevation[960:1025,960:1025]
breaks=[float(value) for value in np.arange(0,4000,50) if real.min()<value<real.max()]
# Three fixed quantile-near ROUND elevation breaks, not GPU-conditioned selection.
breaks=[breaks[len(breaks)//4],breaks[len(breaks)//2],breaks[3*len(breaks)//4]]
cases=[make_case('grand-canyon-65-square',real,breaks)]
# Independent 2x2 cells separated by invalid sample rows/columns.
for lower,upper,label in [(-1,3,'high'),(-3,1,'low'),(-1,1,'tie')]:
    values=np.full((11,11),np.nan)
    for mask in range(16):
        row,column=divmod(mask,4)
        values[row*3:row*3+2,column*3:column*3+2]=np.array([upper if mask&(1<<bit) else lower for bit in range(4)]).reshape(2,2)
    cases.append(make_case('all-masks-saddle-'+label,values,[-.5,0.,.5]))
values=np.tile(np.arange(9,dtype=float),(7,1)); values[2,3]=np.nan; values[4,5]=-9999; values[5,1]=np.inf
validity=np.ones_like(values,dtype=np.uint32); validity[3,7]=0
cases.append(make_case('exact-breaks-nodata-and-band-window',values,[1.,3.,5.,7.],validity,1,3))
fixture=dict(versions=VERSIONS,sources=[dict(path=str(SOURCE.relative_to(ROOT)),sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest())],cases=cases)
output=HERE/'fixtures/parity-wp4-isobands.json'
output.write_text(json.dumps(fixture,separators=(',',':'))+'\n')
assert output.stat().st_size<1_500_000,output.stat().st_size
print(output.name,output.stat().st_size,'breaks',breaks)
