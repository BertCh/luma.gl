# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent Round 6 raster references; no TS/WGSL imports or CPU-model ports.

Join: rasterio.sample.sample_gen on the Round 3 rasterio tract raster, all 43,557
Chicago observations transformed by the SAME declared local affine then rounded
to f32. This is local pixel geometry, not a metric projection or exact PIP join.
The caller-supplied boundary raster is the independent exact edge-intersection
mask, not a claim about the GPU rasterizer's conservative padding. Reverse
north-up storage before sampling a positive-y Affine: join cells are half-open
from minimum y even on a north-up grid (GDAL's native north-up tie differs).
Invalid zone IDs join nothing. Nonfinite values contribute zero but still count.
Quarter-unit category values make sums exact integers scaled by 1/4 below 2^24.
https://rasterio.readthedocs.io/en/stable/api/rasterio.sample.html

Patch metrics: full global Round 2 scipy/skimage dense labels, checked hashes and
source provenance; skimage.regionprops_table supplies area and inclusive-adapted
bbox. rasterio.features.shapes(connectivity=4) polygonizes cell unions; Shapely
length/area after the full affine supplies perimeter/area. This measures exposed
cell faces, NOT skimage's Crofton/contour perimeter. Disconnected pieces sharing
a label sum; raster-border exclusion subtracts the exact boundary intersection.
Masks, capacity and status gates are public-contract adapters applied BEFORE
reference measurement. All eight columns, empty rows, and failed statuses retained.
https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops_table
https://rasterio.readthedocs.io/en/stable/api/rasterio.features.html#rasterio.features.shapes

Sampling: decode full 750x750 B08_after DN, use rows/columns 119:631 (512 square).
Every query is compared, including regular subpixel lattice and tile/workgroup
seams. No spatial crop is interpolated independently: whole tile support is used.
The declared tile domain clamps at its edges. Sentinel/NaN/mask invalid; nearest
uses scipy.ndimage.map_coordinates(order=0), bilinear uses order=1 on both valid
values and weights. Strict rejects any positive missing weight. Composite cubic:
SciPy CubicHermiteSpline with centered secant derivatives is Catmull-Rom a=-1/2
(Keys, 1981, doi:10.1109/TASSP.1981.1163711); tensor evaluation with replicated
edge support. It is NOT scipy's order=3 B-spline. Any participating cubic nodata
falls back to bilinear of the same policy; zero-weight neighbors do not count.
All numerical interpolation is f64 on f32-rounded inputs/positions. Dyadic cell
sizes and query coordinates isolate interpolation from affine roundoff. North-up,
point-pixel and large world origin with local-origin subtraction are explicit.
https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicHermiteSpline.html
"""
import base64
import gzip
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import scipy
from scipy.ndimage import map_coordinates
from scipy.interpolate import CubicHermiteSpline
import rasterio
from rasterio.io import MemoryFile
from rasterio.sample import sample_gen
from rasterio.features import shapes
from affine import Affine
import shapely
from shapely.geometry import shape, box
from shapely.affinity import affine_transform
import skimage
from skimage.measure import regionprops_table

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
NONE = 4294967295
VERSIONS = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                rasterio=rasterio.__version__, shapely=shapely.__version__, skimage=skimage.__version__)


def pack(values):
    numbers = [None if np.isnan(value) else 'Infinity' if value == np.inf else '-Infinity' if value == -np.inf else float(value)
               for value in np.asarray(values).ravel()]
    return base64.b64encode(gzip.compress(json.dumps(numbers, separators=(',', ':')).encode(), mtime=0)).decode()


def unpack(value):
    return np.array(json.loads(gzip.decompress(base64.b64decode(value))), dtype=float)


def source(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def read_parent(name):
    path = HERE / f'fixtures/parity-wp4-{name}.json'
    parent = json.loads(path.read_text())
    for entry in parent['sources']:
        assert hashlib.sha256((ROOT/entry['path']).read_bytes()).hexdigest() == entry['sha256']
    return parent, [source(path), *parent['sources']]


def write(name, cases, sources):
    path = HERE / f'fixtures/parity-wp4-{name}.json'
    path.write_text(json.dumps(dict(versions=VERSIONS, sources=sources, cases=cases), separators=(',', ':'), allow_nan=False)+'\n')
    assert path.stat().st_size < 1_500_000, (name, path.stat().st_size)
    print(path.name, path.stat().st_size, [(case['name'], len(case['frames'])) for case in cases])


def grid(width, height, affine, pixel='area'):
    return dict(width=width, height=height, affine=affine, pixelInterpretation=pixel)


def join_frame(zones, boundary, points, values, metadata, zoneCount):
    stepX, _, originX, _, stepY, originY = metadata['affine']
    if metadata['pixelInterpretation'] == 'point':
        originX -= stepX/2
        originY -= stepY/2
    height, width = zones.shape
    if stepY < 0:
        originY += height*stepY
        zones = zones[::-1]
        boundary = boundary[::-1]
    finite = np.all(np.isfinite(points), axis=1)
    sampled = np.zeros((len(points), 3), dtype=np.uint32)
    sampled[:, 0] = NONE
    # Third band distinguishes outside points from in-domain no-zone pixels.
    with MemoryFile() as memory:
        with memory.open(driver='GTiff', width=width, height=height, count=3, dtype='uint32',
                         transform=Affine(stepX, 0, originX, 0, abs(stepY), originY)) as dataset:
            dataset.write(np.stack([zones, boundary, np.ones_like(zones)]).astype(np.uint32))
            sampled[finite] = np.array(list(sample_gen(dataset, points[finite])))
    inside = sampled[:, 2] == 1
    pointZones = np.where(inside & (sampled[:, 0] < zoneCount), sampled[:, 0], NONE)
    pointBoundary = inside & (sampled[:, 1] != 0)
    joined = pointZones != NONE
    return dict(pointZones=pack(pointZones), pointBoundaryMask=pack(pointBoundary),
        counts=pack(np.bincount(pointZones[joined].astype(int), minlength=zoneCount)),
        sums=pack(np.bincount(pointZones[joined].astype(int), weights=np.where(np.isfinite(values), values, 0)[joined], minlength=zoneCount)),
        boundaryCounts=pack(np.bincount(pointZones[joined & pointBoundary].astype(int), minlength=zoneCount)),
        unassignedBoundaryCount=pack([np.count_nonzero(pointBoundary & ~joined)]),
        outsideCount=pack([np.count_nonzero(~inside)]))


def join_case(name, zones, boundary, points, values, metadata, zoneCount):
    points = np.asarray(points, dtype=np.float32).astype(float)
    values = np.asarray(values, dtype=np.float32).astype(float)
    shifted = {**metadata, 'affine': metadata['affine'].copy()}
    shifted['affine'][2] += 4096
    frames = [dict(grid=placement, expected=join_frame(zones, boundary, points, values, placement, zoneCount))
              for placement in [metadata, shifted, metadata]]
    return dict(name=name, reference='rasterio.sample.sample_gen + numpy.bincount', grid=metadata,
        zones=pack(zones), boundary=pack(boundary), points=pack(points), values=pack(values),
        pointCount=len(points), zoneCount=zoneCount, frames=frames)


parent, joinSources = read_parent('polygons')
city = parent['cases'][0]
zones = unpack(city['zones']).reshape(512, 512).astype(np.uint32)
boundary = unpack(city['boundaryMinimum']).reshape(512, 512).astype(np.uint32)
tractPath = ROOT/'examples/deck/spatial-analysis-showcase/public/data/chicago-tracts/tracts.geojson'
original = [shape(feature['geometry']) for feature in json.loads(tractPath.read_text())['features']]
bounds = shapely.total_bounds(original)
pointPath = ROOT/'examples/deck/spatial-analysis-showcase/public/data/chicago-nature/position.bin'
categoryPath = pointPath.with_name('category.bin')
points = np.frombuffer(pointPath.read_bytes(), dtype='<f4').reshape(-1, 2).astype(float)
points = ((points-bounds[:2])*(496/(bounds[2:]-bounds[:2]))+8).astype(np.float32)
values = np.frombuffer(categoryPath.read_bytes(), dtype='u1').astype(float)/4
write('join-real', [join_case('chicago-all-observations', zones, boundary, points, values,
    grid(512, 512, [1, 0, 0, 0, 1, 0]), len(original)+2)], [*joinSources, source(pointPath), source(categoryPath)])
zones = np.array([[0, 1, NONE, 2, 99], [2, 1, 0, NONE, 2], [0, 0, 1, 1, 2]], dtype=np.uint32)
boundary = np.array([[0, 7, 1, 0, 1], [1, 0, 1, 0, 0], [0, 1, 0, 1, 1]], dtype=np.uint32)
points = [[column/2-2, row/4+3] for row in range(-1, 8) for column in range(-1, 22)]
points += [[np.nan, 4], [2, np.inf], [-np.inf, 4]]
values = np.resize([1, -.25, 2, np.nan, np.inf, -np.inf, 0], len(points))
write('join-synthetic', [join_case('north-up-point-pixels-half-open-invalid-ids', zones, boundary,
    points, values, grid(5, 3, [2, 0, -1, 0, -.5, 4.25], 'point'), 4)], [])


INTEGER_METRICS = ['pixelCounts', 'perimeterFaceCounts', 'minColumns', 'minRows', 'maxColumns', 'maxRows']
FLOAT_METRICS = ['areas', 'perimeters']


def patch_frame(labels, validity, metadata, capacity, countBorder, componentCount, converged=1, overflow=0):
    settings = dict(componentCount=componentCount, converged=converged, overflow=overflow)
    if not converged or overflow:
        return dict(**settings, expected={name: pack(np.full(capacity, NONE if name in INTEGER_METRICS else np.nan))
            for name in INTEGER_METRICS+FLOAT_METRICS})
    accepted = np.where((validity != 0) & (labels <= min(capacity, componentCount)), labels, 0).astype(np.int32)
    result = {name: np.zeros(capacity) for name in INTEGER_METRICS+FLOAT_METRICS}
    properties = regionprops_table(accepted, properties=['label', 'area', 'bbox'])
    indices = properties['label'].astype(int)-1
    result['pixelCounts'][indices] = properties['area']
    for field, key, adjustment in [('minRows', 'bbox-0', 0), ('minColumns', 'bbox-1', 0),
                                    ('maxRows', 'bbox-2', -1), ('maxColumns', 'bbox-3', -1)]:
        result[field][indices] = properties[key]+adjustment
    stepX, shearX, originX, shearY, stepY, originY = metadata['affine']
    transform = [stepX, shearX, shearY, stepY, originX, originY]
    domainBoundary = box(0, 0, metadata['width'], metadata['height']).boundary
    for geometry, labelValue in shapes(accepted, mask=accepted != 0, connectivity=4):
        polygon = shape(geometry)
        edge = polygon.boundary if countBorder else polygon.boundary.difference(domainBoundary)
        index = int(labelValue)-1
        result['perimeterFaceCounts'][index] += edge.length
        result['perimeters'][index] += affine_transform(edge, transform).length
        result['areas'][index] += affine_transform(polygon, transform).area
    np.testing.assert_allclose(result['areas'], result['pixelCounts']*abs(stepX*stepY-shearX*shearY), rtol=1e-12)
    return dict(**settings, expected={name: pack(array) for name, array in result.items()})


def patch_case(name, labels, validity, metadata, capacity, countBorder, componentCount, allFrames=False):
    states = [(componentCount, 1, 0)]
    if allFrames:
        states += [(componentCount, 0, 0), (componentCount, 1, 1), (1, 1, 0), (0, 1, 0), (componentCount, 1, 0)]
    return dict(name=name, reference='skimage.measure.regionprops_table + rasterio.features.shapes + shapely.length/area',
        grid=metadata, labels=pack(labels), validity=pack(validity), capacity=capacity, countRasterBorder=countBorder,
        frames=[patch_frame(labels, validity, metadata, capacity, countBorder, *state) for state in states])


for connectivity in [4, 8]:
    parent, sources = read_parent(f'components-dixie-{connectivity}')
    scene = parent['cases'][0]
    def expandRuns(key):
        array = np.array(scene[key], dtype=np.uint32).reshape(-1, 2)
        return np.repeat(array[:, 1], array[:, 0]).reshape(750, 750)
    labels = expandRuns('dense')
    validity = expandRuns('expectedValidity')
    metadata = grid(750, 750, [20, 0, 0, 0, -20, 15000])
    write(f'patches-real-{connectivity}', [patch_case(f'dixie-full-{connectivity}-border-{border}', labels, validity,
        metadata, scene['componentCount']+2, border, scene['componentCount']) for border in [True, False]], sources)
labels = np.ones((17, 67), dtype=np.uint32)
labels[2:15, 2:65] = 0
labels[4:13, 4:63] = 2
labels[6:11, 6:61] = 0
labels[8, 8:59] = 3
labels[0, 30] = 99
labels[16, 33] = NONE
validity = np.ones_like(labels)
validity[4, 32] = 0
metadata = grid(67, 17, [2, .5, 10, -.25, -3, 20])
cases = [patch_case(f'holes-sheared-capacity-guards-border-{border}', labels, validity, metadata, 5, border, 99, True)
         for border in [True, False]]
for height, width in [(1, 70), (70, 1)]:
    labels = (np.arange(height*width).reshape(height, width)//17 % 3+1).astype(np.uint32)
    cases.append(patch_case(f'thin-{width}x{height}', labels, np.ones_like(labels),
        grid(width, height, [1, 0, 0, 0, 1, 0]), 4, False, 3))
write('patches-synthetic', cases, [])


def interpolate_cubic(support, fraction):
    # Centered secants define Catmull-Rom. SciPy constructs/evaluates the polynomial;
    # no four-weight formula from the production implementation is copied here.
    return CubicHermiteSpline([0., 1.], support[1:3], (support[2:4]-support[0:2])/2, axis=0)(fraction)


def sample_frame(values, validity, coordinates, method, policy, activeCount):
    height, width = values.shape
    count = len(coordinates)
    usable = (validity != 0) & ~np.isnan(values) & (values != -9999)
    clean = np.where(usable, values, 0)
    result = np.full(count, np.nan)
    inside = np.all(np.isfinite(coordinates), axis=1) & (coordinates[:, 0] >= 0) & (coordinates[:, 0] <= width) & (coordinates[:, 1] >= 0) & (coordinates[:, 1] <= height) & (np.arange(count) < activeCount)
    indices = np.flatnonzero(inside)
    positions = coordinates[inside]-0.5
    if method == 'nearest':
        nearest = np.floor(coordinates[inside]).T[::-1]
        result[indices] = map_coordinates(np.where(usable, values, np.nan), nearest, order=0, mode='nearest', prefilter=False)
    else:
        numerator = map_coordinates(clean, positions.T[::-1], order=1, mode='nearest', prefilter=False)
        denominator = map_coordinates(usable.astype(float), positions.T[::-1], order=1, mode='nearest', prefilter=False)
        defined = denominator > 0 if policy == 'renormalize' else denominator == 1
        result[indices[defined]] = numerator[defined]/denominator[defined]
        if method == 'bicubic':
            for index, (column, row) in zip(indices, positions):
                floorColumn, floorRow = int(np.floor(column)), int(np.floor(row))
                fractionColumn, fractionRow = column-floorColumn, row-floorRow
                columns = np.clip(np.arange(floorColumn-1, floorColumn+3), 0, width-1)
                rows = np.clip(np.arange(floorRow-1, floorRow+3), 0, height-1)
                supportValid = usable[np.ix_(rows, columns)]
                participating = np.ones((4, 4), dtype=bool)
                if fractionColumn == 0: participating[:, [0, 2, 3]] = False
                if fractionRow == 0: participating[[0, 2, 3], :] = False
                if np.all(supportValid[participating]):
                    support = clean[np.ix_(rows, columns)]
                    rowValues = interpolate_cubic(support.T, fractionColumn)
                    result[index] = interpolate_cubic(rowValues, fractionRow)
    return dict(values=pack(result), outputValidity=pack(~np.isnan(result)))


def sampling_case(name, values, validity, coordinates, metadata, localOrigin, extraFrames=False):
    values = values.astype(np.float32).astype(float)
    stepX, _, originX, _, stepY, originY = metadata['affine']
    if metadata['pixelInterpretation'] == 'point':
        originX -= stepX/2; originY -= stepY/2
    points = (coordinates*[stepX, stepY]+[originX-localOrigin[0], originY-localOrigin[1]]).astype(np.float32).astype(float)
    # Recompute reference coordinates from the actual uploaded bits, not pre-rounding positions.
    coordinates = (points-[originX-localOrigin[0], originY-localOrigin[1]])/[stepX, stepY]
    settings = [(method, policy, len(points)) for method in ['nearest', 'bilinear', 'bicubic'] for policy in ['strict', 'renormalize']]
    if extraFrames: settings += [('bicubic', 'renormalize', 7), ('nearest', 'strict', 0), ('bilinear', 'strict', len(points)+17), ('nearest', 'strict', len(points))]
    return dict(name=name, reference='scipy.ndimage.map_coordinates + composite scipy.interpolate.CubicHermiteSpline (Catmull-Rom)',
        grid=metadata, localOrigin=localOrigin, values=pack(values), validity=pack(validity), points=pack(points), pointCount=len(points),
        frames=[dict(method=method, noDataPolicy=policy, activeCount=activeCount,
            expected=sample_frame(values, validity, coordinates, method, policy, activeCount)) for method, policy, activeCount in settings])


path = ROOT/'examples/deck/spatial-analysis-showcase/public/data/dixie-fire/B08_after.bin'
raw = np.frombuffer(path.read_bytes(), dtype='<u2').reshape(750, 750).astype(float)
values = raw[119:631, 119:631].copy()
values[values == 0] = -9999
coordinates = [[column+.125, row+.375] for row in range(0, 512, 8) for column in range(0, 512, 8)]
# Query both sides of 16/256-cell seams plus centers, all outer edges, and outside.
coordinates += [[column, row] for column in [0, .5, 15.875, 16, 16.125, 255.875, 256, 256.125, 511.5, 512] for row in np.arange(.5, 512, 4)]
coordinates += [[-0.125, 256], [512.125, 256], [256, 0], [256, 512]]
write('sampling-real', [sampling_case('dixie-512-tile-local-origin-north-up', values, np.ones_like(values), np.array(coordinates),
    grid(512, 512, [2, 0, 600000, 0, -.5, 4600000]), [600000, 4600000])], [source(path)])
rows, columns = np.indices((7, 9))
values = (columns**3-2*rows**2+3*columns*rows).astype(float)/4
validity = np.ones_like(values)
values[2, 3] = np.nan; values[4, 5] = -9999; validity[1, 7] = 0
coordinates = [[column/8, row/8] for row in range(-1, 58, 3) for column in range(-1, 74, 3)]
coordinates += [[column+.5, row+.5] for row in range(7) for column in range(9)]
coordinates += [[0, 0], [9, 7], [np.nan, 1], [np.inf, 1], [1, -np.inf]]
cases = [sampling_case('missing-support-point-pixels-south-up', values, validity, np.array(coordinates),
    grid(9, 7, [2, 0, 101, 0, .5, -19.75], 'point'), [100, -20], True)]
cases.append(sampling_case('single-cell-all-invalid', np.array([[np.nan]]), np.ones((1, 1)),
    np.array([[0, 0], [.5, .5], [1, 1], [-.125, .5]]), grid(1, 1, [1, 0, 0, 0, 1, 0]), [0, 0], True))
write('sampling-synthetic', cases, [])
