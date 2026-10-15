# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Native GDALGridCreate invdist/invdistnn reference via Rasterio's bundled GDAL.

Run with reference Python -I; no osgeo bindings or new installation required.
The ctypes structures/signature follow GDAL's PUBLIC 3.12 C API:
https://gdal.org/en/stable/api/gdal_alg.html
https://gdal.org/en/stable/programs/gdal_grid.html
Grid values come entirely from GDAL, never a local Shepard implementation.
SciPy cKDTree supplies independent counts and neighborhood ambiguity checks.

All 6,135 GHCN PRCP stations, projected by pyproj EPSG:5070, shifted [-1000,1000]
km, snapped to .125 km and converted to f32. This declared precision grid tests
planar interpolation, not projection/sphere equivalence. 64x64 full-extent
cells with dyadic centers, radius 80.0625 km, smoothing=0, power=2 or 1.5;
minimum 3 neighbors, all-radius or nearest 8. invdist max_points=0;
invdistnn max_points=8 (no quadrant constraints). Radius membership and k-cutoff
ties are explicitly enumerated; real points are never deduplicated.
GDAL does not promise the GPU's row-ID tie break at the nearest-k cutoff.
For only those queries, cKDTree membership is sorted by (distance, original row)
and native GDAL invdist is evaluated on exactly that k-set at the query center.
This explicit neighbor-selection adapter does not implement interpolation.
GDAL row 0 samples minY; GPU rows advance north, with an extra south-row case.
Synthetic radius-only cases exercise duplicate exact hits (first input wins),
closed radius, NaN samples, zero masks and insufficient neighbors. GDAL receives
only eligible finite samples in original order; GPU receives originals + mask.
No RNG/permutations, no antimeridian, no nonzero smoothing (API lacks smoothing),
no modified-Shepard taper. Profile L elementwise 1e-6*scale + 1e-4*abs(reference),
exact counts/nodata masks; weighted interpolation also respects the sample range.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import platform

import numpy as numpy
import pyproj
import rasterio
import scipy
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT/'examples/deck/spatial-analysis-showcase/public/data/ghcn-stations'


class InverseOptions(ctypes.Structure):
    _fields_ = [('size', ctypes.c_size_t), ('power', ctypes.c_double), ('smoothing', ctypes.c_double),
                ('anisotropyRatio', ctypes.c_double), ('anisotropyAngle', ctypes.c_double),
                ('radius1', ctypes.c_double), ('radius2', ctypes.c_double), ('angle', ctypes.c_double),
                ('maximumPoints', ctypes.c_uint32), ('minimumPoints', ctypes.c_uint32), ('nodata', ctypes.c_double)]


class NearestOptions(ctypes.Structure):
    _fields_ = [('size', ctypes.c_size_t), ('power', ctypes.c_double), ('radius', ctypes.c_double),
                ('smoothing', ctypes.c_double), ('maximumPoints', ctypes.c_uint32),
                ('minimumPoints', ctypes.c_uint32), ('nodata', ctypes.c_double),
                ('maximumQuadrant', ctypes.c_uint32), ('minimumQuadrant', ctypes.c_uint32)]


def loadGDAL():
    libraryDirectory = Path(rasterio.__file__).parent
    candidates = list((libraryDirectory/'.dylibs').glob('*gdal*.dylib')) + list((libraryDirectory.parent/'rasterio.libs').glob('*gdal*.so*'))
    assert len(candidates) == 1, candidates
    library = ctypes.CDLL(str(candidates[0]))
    pointer = ctypes.POINTER(ctypes.c_double)
    library.GDALGridCreate.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32, pointer, pointer, pointer,
                                       ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                       ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int, ctypes.c_void_p,
                                       ctypes.c_void_p, ctypes.c_void_p]
    library.GDALGridCreate.restype = ctypes.c_int
    library.GDALVersionInfo.argtypes = [ctypes.c_char_p]
    library.GDALVersionInfo.restype = ctypes.c_char_p
    return library


def encode(value):
    if isinstance(value, dict):
        return {key: encode(entry) for key, entry in value.items()}
    if isinstance(value, (list, tuple, numpy.ndarray)):
        return [encode(entry) for entry in value]
    if isinstance(value, numpy.integer):
        return int(value)
    if isinstance(value, (float, numpy.floating)):
        return float(value) if numpy.isfinite(value) else None
    return value


def makeDataset(positions, values, mask, provenance):
    positions = numpy.asarray(positions, dtype=numpy.float32).reshape(-1, 2)
    values = numpy.asarray(values, dtype=numpy.float32)
    return {'positions': positions.ravel(), 'values': values, 'mask': mask,
            'indexBounds': [float(positions[:, 0].min()-1), float(positions[:, 1].min()-1),
                            float(positions[:, 0].max()+1), float(positions[:, 1].max()+1)],
            'inputFloat32Sha256': {name: hashlib.sha256(array.astype('<f4').tobytes()).hexdigest()
                                  for name, array in [('positions', positions), ('values', values)]},
            'provenance': provenance}


def makeCase(library, name, datasetName, dataset, extent, width, height, radius, power, neighbors, minimum, direction='north'):
    positions = numpy.asarray(dataset['positions']).reshape(-1, 2)
    values = numpy.asarray(dataset['values'])
    valid = numpy.isfinite(values) & numpy.asarray(dataset['mask'], dtype=bool)
    coordinates = positions[valid].astype(float)
    channels = [numpy.ascontiguousarray(coordinates[:, 0]), numpy.ascontiguousarray(coordinates[:, 1]), numpy.ascontiguousarray(values[valid], dtype=float)]
    if neighbors:
        options = NearestOptions(ctypes.sizeof(NearestOptions), power, radius, 0, neighbors, minimum, numpy.nan, 0, 0)
        algorithm = 11
    else:
        options = InverseOptions(ctypes.sizeof(InverseOptions), power, 0, 0, 0, radius, radius, 0, 0, minimum, numpy.nan)
        algorithm = 1
    output = numpy.empty((height, width), dtype=numpy.float64)
    minimumX, minimumY, maximumX, maximumY = extent
    result = library.GDALGridCreate(algorithm, ctypes.byref(options), len(coordinates),
                                    *[channel.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) for channel in channels],
                                    minimumX, maximumX, minimumY, maximumY, width, height, 7,
                                    output.ctypes.data, None, None)
    assert result == 0, result
    columns = minimumX+(numpy.arange(width)+.5)*(maximumX-minimumX)/width
    rows = minimumY+(numpy.arange(height)+.5)*(maximumY-minimumY)/height
    centers = numpy.stack(numpy.meshgrid(columns, rows), axis=-1).reshape(-1, 2)
    tree = cKDTree(coordinates)
    memberships = tree.query_ball_point(centers, radius)
    counts = numpy.array([min(len(members), neighbors) if neighbors else len(members) for members in memberships])
    tiedQueries = []
    cutoffGap = None
    if neighbors:
        distances, _ = tree.query(centers, k=neighbors+1)
        relevant = distances[:, neighbors-1] <= radius
        gaps = (distances[:, neighbors]-distances[:, neighbors-1])[relevant]
        positive = gaps[gaps > 0]
        cutoffGap = float(positive.min()) if len(positive) else None
        tiedQueries = numpy.flatnonzero(relevant & (distances[:, neighbors] == distances[:, neighbors-1])).tolist()
        for query in tiedQueries:
            members = numpy.array(memberships[query])
            squaredDistances = numpy.sum((coordinates[members]-centers[query])**2, axis=1)
            selected = members[numpy.lexsort((members, squaredDistances))[:neighbors]]
            selected.sort()  # preserve original row order for exact duplicate hits
            localChannels = [numpy.ascontiguousarray(channel[selected]) for channel in channels]
            localOptions = InverseOptions(ctypes.sizeof(InverseOptions), power, 0, 0, 0, radius, radius, 0, 0, minimum, numpy.nan)
            localOutput = numpy.empty(1, dtype=float)
            centerX, centerY = centers[query]
            result = library.GDALGridCreate(1, ctypes.byref(localOptions), len(selected),
                *[channel.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) for channel in localChannels],
                centerX-.5, centerX+.5, centerY-.5, centerY+.5, 1, 1, 7, localOutput.ctypes.data, None, None)
            assert result == 0
            output.ravel()[query] = localOutput[0]
    # 0.125 km positions, dyadic centers and radius make the squared membership
    # arithmetic exactly representable here (max near-radius squared < 2^14).
    if datasetName == 'ghcn':
        outer = tree.query_ball_point(centers, radius+1e-7, return_length=True)
        inner = tree.query_ball_point(centers, radius-1e-7, return_length=True)
        assert numpy.array_equal(outer, inner)
    if direction == 'south':
        output = output[::-1]
        counts = counts.reshape(height, width)[::-1].ravel()
    finite = output[numpy.isfinite(output)]
    assert finite.min() >= values[valid].min()-1e-10 and finite.max() <= values[valid].max()+1e-10
    return {'name': name, 'dataset': datasetName, 'reference': f'GDALGridCreate/{"invdistnn" if neighbors else "invdist"}; scipy.spatial.cKDTree',
            'n': width*height, 'width': width, 'height': height, 'extent': extent,
            'searchRadius': radius, 'power': power, 'neighborCount': neighbors,
            'minimumNeighborCount': minimum, 'rowDirection': direction, 'minimumKCutoffGap': cutoffGap, 'adaptedKCutoffQueries': tiedQueries,
            'expected': {'values': output.ravel(), 'counts': counts}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', dest='outputDirectory', type=Path, default=Path(__file__).parent/'fixtures')
    arguments = parser.parse_args()
    library = loadGDAL()
    geographic = numpy.fromfile(DATA/'position.bin', dtype='<f4').reshape(-1, 2)
    values = numpy.fromfile(DATA/'prcp.bin', dtype='<f4')
    transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:5070', always_xy=True)
    positions = numpy.column_stack(transformer.transform(geographic[:, 0], geographic[:, 1]))/1000-[-1000, 1000]
    positions = numpy.rint(positions*8)/8
    provenance = {'source': str(DATA.relative_to(ROOT)), 'sourceCount': len(values), 'selection': 'all 6135 stations, original order',
                  'sourceSha256': {name: hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in ['manifest.json', 'position.bin', 'prcp.bin']},
                  'sourceCrs': 'EPSG:4326', 'targetCrs': 'EPSG:5070', 'units': 'kilometers', 'origin': [-1000, 1000],
                  'snapGridKilometers': .125, 'valueUnits': 'mm precipitation', 'attribution': 'NOAA NCEI GHCN-Daily'}
    datasets = {'ghcn': makeDataset(positions, values, [1]*len(values), provenance),
                'synthetic': makeDataset([[.5, .5], [.5, .5], [1.5, .5], [2.5, .5], [1.5, 1.5], [2.5, 1.5]],
                                         [10, 20, 30, 50, numpy.nan, 999], [1, 1, 1, 1, 1, 0], {'source': 'synthetic'})}
    extent = [float(numpy.floor(positions[:, 0].min()/64)*64), float(numpy.floor(positions[:, 1].min()/64)*64),
              float(numpy.ceil(positions[:, 0].max()/64)*64), float(numpy.ceil(positions[:, 1].max()/64)*64)]
    cases = [makeCase(library, 'ghcn-full-radius', 'ghcn', datasets['ghcn'], extent, 64, 64, 80.0625, 2, 0, 3),
             makeCase(library, 'ghcn-full-nearest-eight', 'ghcn', datasets['ghcn'], extent, 64, 64, 80.0625, 1.5, 8, 3)]
    for direction in ['north', 'south']:
        cases.append(makeCase(library, f'duplicates-mask-nodata-{direction}', 'synthetic', datasets['synthetic'],
                              [0, 0, 4, 3], 4, 3, 1, 2, 0, 3, direction))
    for height in [5, 7]:
        for direction in ['north', 'south']:
            cases.append(makeCase(library, f'duplicates-height-{height}-{direction}', 'synthetic', datasets['synthetic'],
                                  [0, 0, 4, height], 4, height, 1, 2, 0, 3, direction))
    fixture = {'generator': Path(__file__).name, 'versions': {'python': platform.python_version(),
               'numpy': numpy.__version__, 'scipy': scipy.__version__, 'pyproj': pyproj.__version__,
               'rasterio': rasterio.__version__, 'gdal': library.GDALVersionInfo(b'RELEASE_NAME').decode()},
               'datasets': datasets, 'cases': cases}
    output = arguments.outputDirectory/'parity-wp1-inverse-distance-weighting.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(encode(fixture), separators=(',', ':'), allow_nan=False)+'\n'
    assert len(serialized.encode()) <= 1500000
    output.write_text(serialized)
    print(output, len(serialized), 'bytes; GDAL', fixture['versions']['gdal'])
    for case in cases:
        print(case['name'], 'finite cells', numpy.isfinite(case['expected']['values']).sum(), 'k-gap', case['minimumKCutoffGap'])


if __name__ == '__main__':
    main()
