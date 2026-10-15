# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Rasterio/Shapely polygon reference and SciPy zonal reduction reference.
https://rasterio.readthedocs.io/en/stable/topics/features.html
https://shapely.readthedocs.io/en/stable/strtree.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.sum_labels.html

All 791 Chicago tract features map by a declared affine from lon/lat into a
512x512 LOCAL planar raster (not a metric/geodesic projection). All vertices
round to f32 before any oracle. Row 0 south; center inclusion, all_touched=False;
reverse feature burn order implements smallest feature ID overlap precedence.
Shapely STRtree independently verifies every rasterio cell. Holes/multipolygons
retain GeoJSON topology. Synthetic half-integer boundary tests adapt GDAL's
edge ownership by shifting the reference probe +1e-7 in BOTH axes, implementing
left/bottom closed, right/top open; no blanket boundary mismatch allowance.

Boundary output is conservative, not GDAL all_touched polygon fill. Every cell
must include every actual edge intersection and may include only cells within
the documented padding (1e-3 + 1e-6*coordinate magnitude) of an edge. Shapely
STRtree intersects exact and padded closed cell boxes; uncertainty cells are
explicitly enumerated, all other flags compare exactly. No GPU code is ported.

Zonal statistics use the independent raster labels directly, isolating reducer
failures from rasterization. Values are a declared synthetic quarter-step
surface over real tracts (NOT a real Chicago measurement). Raw nodata -9999,
NaN, Infinity and a validity mask are excluded before scale=.5, offset=1.
Counts are unweighted; sample variance ddof=1. All zone outputs including empty
zones, singleton zones, ignored IDs and out-of-range IDs are retained.
Uniform sorted and atomic modes share this reference; geographic weights are
not covered. Gzip/base64 JSON is lossless transport for EVERY cell, no sampling.
"""
import base64
import gzip
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import scipy
from scipy import ndimage
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
import shapely
from shapely.geometry import shape, Polygon, MultiPolygon, LineString
from shapely.affinity import translate
from shapely import STRtree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT/'examples/deck/spatial-analysis-showcase/public/data/chicago-tracts/tracts.geojson'
NONE = 4294967295
VERSIONS = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                rasterio=rasterio.__version__, shapely=shapely.__version__)
SOURCES = [dict(path=str(SOURCE.relative_to(ROOT)), sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest())]


def clean(value):
    if isinstance(value, dict): return {key: clean(entry) for key, entry in value.items()}
    if isinstance(value, list): return [clean(entry) for entry in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None if np.isnan(value) else 'Infinity' if value > 0 else '-Infinity'
    return value


def pack(value):
    return base64.b64encode(gzip.compress(json.dumps(clean(np.asarray(value).ravel().tolist()),
                            separators=(',', ':'), allow_nan=False).encode(), mtime=0)).decode()


def make_case(name, geometries, width, height, boundary_shift=False):
    positions = []; feature_offsets = [0]; polygon_offsets = [0]; ring_offsets = [0]; edges = []
    for geometry in geometries:
        polygons = [geometry] if geometry.geom_type == 'Polygon' else list(geometry.geoms)
        for polygon in polygons:
            for ring in [polygon.exterior, *polygon.interiors]:
                coordinates = np.array(ring.coords[:-1], dtype=np.float32).astype(float)
                positions.extend(coordinates.ravel().tolist()); ring_offsets.append(len(positions)//2)
                edges.extend(LineString([start, end]) for start, end in zip(coordinates, np.roll(coordinates, -1, axis=0)))
            polygon_offsets.append(len(ring_offsets)-1)
        feature_offsets.append(len(polygon_offsets)-1)
    rows, columns = np.indices((height, width))
    shift = 1e-7 if boundary_shift else 0
    oracle_geometries = [translate(geometry, xoff=-shift, yoff=-shift) for geometry in geometries]
    zones = rasterize([(geometry, index) for index, geometry in reversed(list(enumerate(oracle_geometries)))],
                      out_shape=(height, width), transform=Affine.identity(), fill=NONE, dtype='uint32', all_touched=False)
    probes = shapely.points(columns.ravel()+.5+shift, rows.ravel()+.5+shift)
    pairs = STRtree(geometries).query(probes, predicate='within')
    other = np.full(width*height, NONE, dtype=np.uint32)
    np.minimum.at(other, pairs[0], pairs[1].astype(np.uint32))
    np.testing.assert_array_equal(zones.ravel(), other)
    tree = STRtree(edges)
    boundary = []
    # Conservative output may differ ONLY in this declared narrow geometric band.
    padding = .001 + .000001 * max(np.abs(positions))
    for expansion in [0, padding]:
        boxes = shapely.box(columns.ravel()-expansion, rows.ravel()-expansion,
                            columns.ravel()+1+expansion, rows.ravel()+1+expansion)
        touched = np.zeros(width*height, dtype=np.uint32)
        touched[np.unique(tree.query(boxes, predicate='intersects')[0])] = 1
        boundary.append(touched)
    # Crossing count: one event per edge intersecting each half-open horizontal center line.
    crossings = sum(np.count_nonzero((np.arange(height)+.5 >= min(edge.coords[0][1], edge.coords[1][1])) &
                                    (np.arange(height)+.5 < max(edge.coords[0][1], edge.coords[1][1]))) for edge in edges)
    return dict(name=name, reference='rasterio.features.rasterize + shapely.STRtree', width=width, height=height,
                positions=positions, featureOffsets=feature_offsets, polygonOffsets=polygon_offsets, ringOffsets=ring_offsets,
                zones=pack(zones), boundaryMinimum=pack(boundary[0]), boundaryMaximum=pack(boundary[1]),
                boundaryAmbiguousCells=np.flatnonzero(boundary[0] != boundary[1]).tolist(),
                crossingCount=int(crossings), boundaryShift=shift), zones


features = json.loads(SOURCE.read_text())['features']
original = [shape(feature['geometry']) for feature in features]
bounds = shapely.total_bounds(original)
# Leave an 8-cell margin around every edge; no implicit clipping of the source features.
scale = 496/(bounds[2:]-bounds[:2])

def local_geometry(geometry):
    return shapely.transform(geometry, lambda coordinates: np.asarray((coordinates-bounds[:2])*scale+8, dtype=np.float32).astype(float))

geometries = [local_geometry(geometry) for geometry in original]
city, city_zones = make_case('chicago-all-791-tracts-512-square', geometries, 512, 512)
edge_geometries = [Polygon([(1.25, 1.25), (12.25, 1.25), (12.25, 10.25), (1.25, 10.25)],
                          holes=[[(4.25, 3.25), (7.25, 3.25), (7.25, 7.25), (4.25, 7.25)]]),
                   Polygon([(8.25, 2.25), (17.25, 2.25), (15.25, 14.25), (8.25, 12.25)]),
                   MultiPolygon([Polygon([(2.25, 12.25), (4.25, 12.25), (4.25, 14.25), (2.25, 14.25)]),
                                 Polygon([(18.1, 1.1), (18.2, 1.1), (18.2, 1.2), (18.1, 1.2)])])]
# Round synthetic decimal vertices before reconstructing either oracle geometry.
edge_geometries = [shapely.transform(geometry, lambda coordinates: coordinates.astype(np.float32).astype(float)) for geometry in edge_geometries]
edge, edge_zones = make_case('holes-overlap-multipart-subpixel', edge_geometries, 23, 17)
boundary, boundary_zones = make_case('half-open-center-edge-ownership',
                    [shapely.box(.5, .5, 4.5, 4.5), shapely.box(4.5, .5, 8.5, 4.5)], 10, 6, True)


def zonal(name, zones, capacity, ignored=NONE):
    height, width = zones.shape
    rows, columns = np.indices(zones.shape)
    values = (((rows*17 + columns*13) % 129)-64).astype(float)/4
    validity = ((rows*3+columns) % 17 != 0).astype(np.uint32)
    values[(rows+columns) % 43 == 0] = -9999
    values[(rows*5+columns) % 71 == 0] = np.nan
    values[(rows+columns*3) % 89 == 0] = np.inf
    calibrated = values*.5+1
    usable = (validity != 0) & np.isfinite(values) & (values != -9999) & np.isfinite(calibrated)
    active = (zones < capacity) & (zones != ignored)
    labels = np.where(active, zones+1, 0)
    indices = np.arange(1, capacity+1)
    cell_counts = ndimage.sum_labels(np.ones(zones.shape), labels, indices)
    value_counts = ndimage.sum_labels(usable.astype(float), labels, indices)
    sums = ndimage.sum_labels(np.where(usable, calibrated, 0), labels, indices)
    means = np.divide(sums, value_counts, out=np.full(capacity, np.nan), where=value_counts > 0)
    minimums = ndimage.minimum(np.where(usable, calibrated, np.inf), labels, indices).astype(float)
    maximums = ndimage.maximum(np.where(usable, calibrated, -np.inf), labels, indices).astype(float)
    minimums[value_counts == 0] = np.nan; maximums[value_counts == 0] = np.nan
    variances = np.full(capacity, np.nan)
    for zone in range(capacity):
        selected = calibrated[active & usable & (zones == zone)]
        if len(selected) > 1: variances[zone] = np.var(selected, ddof=1)
    expected = dict(cellCounts=cell_counts.astype(int).tolist(), valueCounts=value_counts.astype(int).tolist(),
                    sums=sums.tolist(), means=means.tolist(), minimums=minimums.tolist(), maximums=maximums.tolist(),
                    weightSums=value_counts.tolist(), variances=variances.tolist(), standardDeviations=np.sqrt(variances).tolist())
    return dict(name=name, reference='scipy.ndimage.sum_labels/minimum/maximum + numpy.var(ddof=1)',
                width=width, height=height, zoneCapacity=capacity, ignoredZone=ignored, zones=pack(zones),
                values=pack(values), validity=pack(validity), scale=.5, offset=1, noDataValue=-9999,
                overflow=int(np.any((zones >= capacity) & (zones != ignored))), expected=expected)


small_zones = edge_zones.copy(); small_zones[0, 0] = 8; small_zones[1, 1] = 4
zonal_cases = [zonal('chicago-full-raster-synthetic-values', city_zones, 793),
               zonal('holes-missing-singleton-overflow-empty', small_zones, 6),
               zonal('ignored-zone-zero', small_zones, 6, ignored=0)]
for suffix, cases in [('polygons', [city, edge, boundary]), ('zonal', zonal_cases)]:
    output = HERE/f'fixtures/parity-wp4-{suffix}.json'
    output.write_text(json.dumps(clean(dict(versions=VERSIONS, sources=SOURCES,
        sourceBounds=bounds.tolist(), localScale=scale.tolist(), cases=cases)), separators=(',', ':'), allow_nan=False)+'\n')
    assert output.stat().st_size <= 1_500_000, output.stat().st_size
    print(output.name, output.stat().st_size)
print('crossings', city['crossingCount'], 'boundary ambiguity', len(city['boundaryAmbiguousCells']))
