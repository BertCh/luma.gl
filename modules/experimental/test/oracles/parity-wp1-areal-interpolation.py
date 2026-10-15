# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Tobler/Shapely area interpolation on identical cell-support polygons.

Run <reference-venv>/bin/python -I <this script> [--output-dir DIRECTORY].
The GPU API consumes zone RASTERS, not original polygons. We rasterize all 791
Chicago tracts (EPSG:26916, cell-center membership, rasterio all_touched=False)
on one full-extent 128x160 grid. All 105,808 Chicago places supply extensive
counts using the dataset's tract IDs; intensive input is count/source support
area. Geometry accuracy of the original polygons is outside this test's scope.
Rasterio.features.shapes polygonizes each uploaded zone raster in unit cell
coordinates. Shapely unions, STRtree queries, intersections and areas therefore
measure exactly the supports consumed by the GPU, including holes/disconnected
parts, absent sources, uncovered targets, and sources outside the target system.

Tobler.area_weighted.area_interpolate runs its own polygon intersection builder
(no supplied table). allocate_total=False matches extensive denominator='zone';
True matches 'overlap'. Tobler's intensive denominator is always overlap. The
intensive 'zone' variant multiplies its estimate by covered/whole target area,
a composite reference using the area-weighted definition in Tobler's docstring.
Shapely independently supplies all CSR pair areas and both normalization tables.
No overlapping ownership within either raster; one source and target ID per cell.
Source and target polygons may overlap across systems. No anchors, dasymetric
weights, spherical areas, masks, seeds or permutations. Empty rows sum to zero;
source transferred fraction is undefined (JSON null -> NaN) for zero-area sources.

Profile G: integer grid, exact pair topology/areas; no coordinate reconstruction
is returned by this API, so Hausdorff/symmetric difference is inapplicable.
Normalized weights/transferred values use profile S (1e-6*scale + 1e-4*abs(ref));
integer cell areas and masks are exact. Float32 inputs are promoted to float64.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform

import geopandas
import numpy as numpy
import pyproj
import rasterio
from rasterio.features import rasterize, shapes
from rasterio.transform import from_bounds
import scipy
from scipy.sparse import csr_matrix
import shapely
from shapely import STRtree, Polygon, intersection, area, union_all
from shapely.geometry import shape
import tobler
from tobler.area_weighted import area_interpolate

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / 'examples/deck/spatial-analysis-showcase/public/data'
NO_ZONE = 4294967295


def hashFile(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def polygonize(zones, count):
    pieces = [[] for _ in range(count)]
    for geometry, zone in shapes(zones.astype(numpy.int32), mask=zones < count):
        pieces[int(zone)].append(shape(geometry))
    return [union_all(group) if group else Polygon() for group in pieces]


def makeCase(name, sourceZones, targetZones, sourceCount, targetCount, values, provenance):
    sourcePolygons = polygonize(sourceZones, sourceCount)
    targetPolygons = polygonize(targetZones, targetCount)
    sourceArea = area(sourcePolygons)
    targetArea = area(targetPolygons)
    pairs = STRtree(sourcePolygons).query(targetPolygons, predicate='intersects')
    pairAreas = area(intersection(numpy.asarray(targetPolygons)[pairs[0]], numpy.asarray(sourcePolygons)[pairs[1]]))
    positive = pairAreas > 0
    matrix = csr_matrix((pairAreas[positive], (pairs[0][positive], pairs[1][positive])), shape=(targetCount, sourceCount))
    matrix.sort_indices()
    targetOverlap = numpy.asarray(matrix.sum(axis=1)).ravel()
    sourceOverlap = numpy.asarray(matrix.sum(axis=0)).ravel()
    values = numpy.asarray(values, dtype=numpy.float32).astype(numpy.float64)
    rates = numpy.divide(values, sourceArea, out=numpy.zeros(sourceCount), where=sourceArea > 0).astype(numpy.float32).astype(numpy.float64)
    sourceFrame = geopandas.GeoDataFrame({'mass': values, 'rate': rates}, geometry=sourcePolygons, crs='EPSG:26916')
    targetFrame = geopandas.GeoDataFrame(geometry=targetPolygons, crs='EPSG:26916')
    references = {}
    for extensiveDenominator in ['zone', 'overlap']:
        estimates = area_interpolate(sourceFrame, targetFrame, extensive_variables=['mass'], intensive_variables=['rate'], allocate_total=extensiveDenominator == 'overlap')
        sourceDenominator = sourceArea if extensiveDenominator == 'zone' else sourceOverlap
        extensiveWeights = matrix.multiply(numpy.divide(1, sourceDenominator, out=numpy.zeros(sourceCount), where=sourceDenominator > 0)).tocsr()
        fractions = numpy.divide(sourceOverlap, sourceDenominator, out=numpy.zeros(sourceCount), where=sourceDenominator > 0)
        fractions[sourceArea == 0] = numpy.nan
        for intensiveDenominator in ['zone', 'overlap']:
            targetDenominator = targetArea if intensiveDenominator == 'zone' else targetOverlap
            intensiveWeights = matrix.multiply(numpy.divide(1, targetDenominator, out=numpy.zeros(targetCount), where=targetDenominator > 0)[:, None]).tocsr()
            intensive = estimates['rate'].to_numpy().copy()
            if intensiveDenominator == 'zone':
                intensive *= numpy.divide(targetOverlap, targetArea, out=numpy.zeros(targetCount), where=targetArea > 0)
            extensive = estimates['mass'].to_numpy()
            # Independent Shapely and Tobler paths must agree before publishing a fixture.
            numpy.testing.assert_allclose(extensiveWeights @ values, extensive, rtol=1e-6, atol=1e-8)
            numpy.testing.assert_allclose(intensiveWeights @ rates, intensive, rtol=1e-6, atol=1e-8)
            references[f'{extensiveDenominator}-{intensiveDenominator}'] = {
                'extensiveWeights': extensiveWeights.data.tolist(),
                'intensiveWeights': intensiveWeights.data.tolist(),
                'extensive': extensive.tolist(), 'intensive': intensive.tolist(),
                'sourceTransferredFractions': [float(value) if numpy.isfinite(value) else None for value in fractions],
                'sourceLostFractions': [float(1-value) if numpy.isfinite(value) else None for value in fractions],
                'massTotal': float(extensive.sum()), 'massLost': float(values.sum() - extensive.sum()),
            }
    return {
        'name': name, 'reference': 'tobler.area_weighted.area_interpolate; shapely.STRtree/intersection/area',
        'width': sourceZones.shape[1], 'height': sourceZones.shape[0],
        'sourceZoneCount': sourceCount, 'targetZoneCount': targetCount,
        'sourceZones': sourceZones.ravel().tolist(), 'targetZones': targetZones.ravel().tolist(),
        'sourceValues': values.tolist(), 'sourceRates': rates.tolist(),
        'offsets': matrix.indptr.tolist(), 'neighbors': matrix.indices.tolist(), 'areas': matrix.data.tolist(),
        'sourceAreas': sourceArea.tolist(),
        'targetUncoveredFractions': (1 - numpy.divide(targetOverlap, targetArea, out=numpy.zeros(targetCount), where=targetArea > 0)).tolist(),
        'invalidCount': int(numpy.sum(sourceArea == 0)), 'references': references,
        'inputSha256': hashlib.sha256(sourceZones.astype('<u4').tobytes() + targetZones.astype('<u4').tobytes() + values.astype('<f4').tobytes() + rates.astype('<f4').tobytes()).hexdigest(),
        'provenance': provenance,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', dest='outputDirectory', type=Path, default=Path(__file__).parent / 'fixtures')
    arguments = parser.parse_args()
    tractsPath = DATA / 'chicago-tracts/tracts.geojson'
    placesPath = DATA / 'chicago-places/tract.bin'
    tracts = geopandas.read_file(tractsPath).to_crs('EPSG:26916')
    width, height = 128, 160
    transform = from_bounds(*tracts.total_bounds, width, height)
    sourceZones = rasterize([(geometry, row) for row, geometry in enumerate(tracts.geometry)], out_shape=(height, width), fill=NO_ZONE, transform=transform, all_touched=False, dtype='uint32')
    rows, columns = numpy.indices((height, width))
    targetZones = ((rows // 16) * 8 + columns // 16).astype(numpy.uint32)
    # Trim the target system at the outer six cells; retain complete source support.
    targetZones[(columns < 6) | (columns >= width-6) | (rows < 6) | (rows >= height-6)] = NO_ZONE
    placesTracts = numpy.fromfile(placesPath, dtype='<u2')
    validPlaces = placesTracts < len(tracts)
    counts = numpy.bincount(placesTracts[validPlaces], minlength=len(tracts))
    provenance = {
        'sources': {str(path.relative_to(ROOT)): hashFile(path) for path in [tractsPath, placesPath, DATA / 'chicago-tracts/manifest.json', DATA / 'chicago-places/manifest.json']},
        'placesCount': len(placesTracts), 'assignedPlacesCount': int(validPlaces.sum()),
        'tractCount': len(tracts), 'selection': 'all tracts and all places; unassigned places have no source zone',
        'sourceCrs': 'EPSG:4326', 'targetCrs': 'EPSG:26916', 'rasterTransform': list(transform),
        'referenceGeometry': 'polygonized uploaded cell support in unit grid coordinates; not original tract boundaries',
    }
    cases = [makeCase('chicago-places-all-tracts-cell-support', sourceZones, targetZones, len(tracts), 81, counts, provenance)]
    cases.append(makeCase('holes-uncovered-empty-disconnected', numpy.array([
        [0, 0, 0, 1, 1, NO_ZONE], [0, NO_ZONE, 0, 1, 1, NO_ZONE],
        [2, 2, 0, 1, NO_ZONE, NO_ZONE], [2, 2, NO_ZONE, 2, 2, NO_ZONE],
    ], dtype=numpy.uint32), numpy.array([
        [0, 0, 1, 1, NO_ZONE, 3], [0, 0, 1, 1, NO_ZONE, 3],
        [NO_ZONE, 2, 2, 1, 3, 3], [NO_ZONE, 2, 2, 2, 2, 3],
    ], dtype=numpy.uint32), 4, 5, [100, 80, 45, 9], {'source': 'synthetic; absent source 3, uncovered target 3, absent target 4'}))
    fixture = {'generator': Path(__file__).name,
               'versions': {'python': platform.python_version(), 'numpy': numpy.__version__, 'scipy': scipy.__version__, 'tobler': tobler.__version__, 'shapely': shapely.__version__, 'geopandas': geopandas.__version__, 'rasterio': rasterio.__version__, 'pyproj': pyproj.__version__},
               'tolerance': {'profile': 'G/S', 'absoluteScale': 1e-6, 'relative': 1e-4}, 'cases': cases}
    arguments.outputDirectory.mkdir(parents=True, exist_ok=True)
    output = arguments.outputDirectory / 'parity-wp1-areal-interpolation.json'
    serialized = json.dumps(fixture, separators=(',', ':'), allow_nan=False) + '\n'
    assert len(serialized.encode()) <= 1500000
    output.write_text(serialized)
    print(output, len(serialized), 'bytes')
    for case in cases:
        print(case['name'], len(case['sourceZones']), 'cells;', len(case['areas']), 'pairs;', case['invalidCount'], 'empty sources')


if __name__ == '__main__':
    main()
