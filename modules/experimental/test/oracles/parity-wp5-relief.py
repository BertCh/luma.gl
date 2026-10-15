# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent RVT local dominance and SciPy multi-scale relief references.

Local dominance calls unmodified rvt.vis.local_dominance (Hesse 2016): pixel
radii, numpy half-even offsets, edge replication, missing neighbors contribute
zero but remain in normalization. Invalid centers are NaN. RVT forces float32;
observer heights/exaggerations here are binary-exact to avoid losing observer
height when adding it to Alpine elevations. Reference output retains RVT's
float32 reduction error; the GPU R budget is unchanged.

MSRM is a COMPOSITE reference of scipy.ndimage.uniform_filter in float64:
normalized valid-value/valid-count box convolution, edge replication, invalid
centers restored, then the published Orengo/Petrie 2018 mean of consecutive
scale differences. No GPU telescoping, integral table or quantization is ported.
The unmodified rvt.vis.msrm is also called on every case and checked with a
separately derived bound for its float32 intermediate surface rounding.

All GPU buffer values/parameters round to float32 first; CPU scale topology
retains float64, as in the API. Uniform local metres;
resolution defines the scale ladder, which uses square pixel windows (no
latitude-dependent ground spacing). Raw NaN, nodata=-9999 and explicit validity
are canonicalized before the independent functions. No random data/seed.
Decode the full Alps raster once; run the SciPy reference on the full raster.
Two adjacent 96x96 cores with 9-cell halos compare ALL 114x114 output pixels;
cores additionally agree with full-raster SciPy and a united-window RVT run.
This subsets the audit's 512x512 scale to keep each JSON below 1.5 MB.
"""
import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path

import numpy
import rasterio
import rvt.vis
from scipy.ndimage import uniform_filter

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).parent / 'fixtures'
DATA = ROOT / 'examples/deck/spatial-analysis-showcase/public/data/alps-dem'
VERSIONS = {name: importlib.metadata.version(name) for name in ['numpy', 'scipy', 'rasterio', 'rvt-py']}
VERSIONS.update(python=platform.python_version(), gdal=rasterio.__gdal_version__)
GEOMETRY = {'minimumRadius': 3, 'maximumRadius': 9, 'radiusIncrement': 2, 'angularResolution': 45}


def encode(value):
    if isinstance(value, dict):
        return {key: encode(entry) for key, entry in value.items()}
    if isinstance(value, (list, tuple, numpy.ndarray)):
        return [encode(entry) for entry in value]
    if isinstance(value, (numpy.integer, numpy.bool_)):
        return int(value)
    if isinstance(value, (float, numpy.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def computeMean(elevation, radius):
    if radius == 0:
        return elevation.copy()
    valid = numpy.isfinite(elevation)
    weights = uniform_filter(valid.astype(float), size=2 * radius + 1, mode='nearest')
    totals = uniform_filter(numpy.where(valid, elevation, 0), size=2 * radius + 1, mode='nearest')
    with numpy.errstate(invalid='ignore', divide='ignore'):
        result = totals / weights
    result[~valid] = numpy.nan
    return result


def computeMultiScale(elevation, scales, exaggeration):
    firstIndex = math.floor(((max(scales['featureMinimum'], scales['resolution']) - scales['resolution']) /
                            (2 * scales['resolution'])) ** (1 / scales['scalingFactor']))
    lastIndex = math.ceil(((scales['featureMaximum'] - scales['resolution']) /
                          (2 * scales['resolution'])) ** (1 / scales['scalingFactor']))
    radii = [index ** scales['scalingFactor'] for index in range(firstIndex, lastIndex + 1)]
    previous = computeMean(elevation * exaggeration, radii[0])
    result = numpy.zeros_like(elevation)
    for radius in radii[1:]:
        current = computeMean(elevation * exaggeration, radius)
        result += previous - current
        previous = current
    return result / (len(radii) - 1), radii


def computeDominance(elevation, geometry, observerHeight, exaggeration):
    return rvt.vis.local_dominance(elevation.copy(), min_rad=geometry['minimumRadius'],
        max_rad=geometry['maximumRadius'], rad_inc=geometry['radiusIncrement'],
        angular_res=geometry['angularResolution'], observer_height=observerHeight,
        ve_factor=exaggeration).astype(float)


def tolerance(values):
    finite = values[numpy.isfinite(values)]
    return {'profile': 'R', 'absolute': 1e-5 * max(1, float(numpy.ptp(finite)) if finite.size else 0),
            'relative': 1e-4}


def buildCase(name, rawElevation, scales, source, geometry=GEOMETRY, observerHeight=2,
              exaggeration=1, validity=None):
    rawElevation = numpy.asarray(rawElevation, dtype=numpy.float32).astype(float)
    validity = numpy.ones_like(rawElevation, dtype=int) if validity is None else validity
    elevation = numpy.where(numpy.isfinite(rawElevation) & (rawElevation != -9999) & (validity != 0), rawElevation, numpy.nan)
    observerHeight, exaggeration = map(lambda value: float(numpy.float32(value)), [observerHeight, exaggeration])
    dominance = computeDominance(elevation, geometry, observerHeight, exaggeration)
    relief, radii = computeMultiScale(elevation, scales, exaggeration)
    rvtRelief = rvt.vis.msrm(elevation.copy(), resolution=scales['resolution'],
        feature_min=scales['featureMinimum'], feature_max=scales['featureMaximum'],
        scaling_factor=scales['scalingFactor'], ve_factor=exaggeration)
    # RVT rounds each surface and each difference to f32. Each mean has error
    # <= eps*M; each subtraction adds <= 2*eps*M. Averaging any number of
    # differences therefore has error <= 4*eps*M. Binary-exact exaggeration
    # adds no input error. This checks the secondary RVT oracle ONLY; GPU
    # comparisons still use the original profile-R tolerance without widening.
    magnitude = max(1, float(numpy.nanmax(numpy.abs(elevation * exaggeration)))) if numpy.any(numpy.isfinite(elevation)) else 1
    rvtBound = 4 * numpy.finfo(numpy.float32).eps * magnitude
    numpy.testing.assert_allclose(rvtRelief, relief, atol=rvtBound, rtol=0, equal_nan=True)
    for result in [dominance, relief]:
        numpy.testing.assert_array_equal(numpy.isfinite(result), numpy.isfinite(elevation))
    residual = numpy.abs(rvtRelief - relief)
    return {'name': name, 'width': elevation.shape[1], 'height': elevation.shape[0],
            'source': source, 'scales': scales, 'geometry': geometry, 'radii': radii,
            'observerHeight': observerHeight, 'verticalExaggeration': exaggeration,
            'elevation': rawElevation.ravel(), 'validity': validity.ravel(),
            'expectedValidity': numpy.isfinite(elevation).astype(int).ravel(),
            'expected': {'dominance': dominance.ravel(), 'relief': relief.ravel()},
            'tolerances': {'dominance': tolerance(dominance), 'relief': tolerance(relief)},
            'rvtCrossCheck': {'maxAbs': float(numpy.nanmax(residual)) if numpy.any(numpy.isfinite(residual)) else 0,
                              'derivedAbsoluteBound': rvtBound}}


def writeFixture(topic, cases):
    payload = {'versions': VERSIONS, 'conventions': __doc__,
               'generatorSha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'references': {'dominance': 'rvt.vis.local_dominance',
                              'relief': 'Composite: scipy.ndimage.uniform_filter + Orengo/Petrie 2018 scale differences'},
               'citations': ['https://doi.org/10.1002/esp.4317', 'https://doi.org/10.3390/rs11070784',
                             'https://github.com/EarthObservation/RVT_py'], 'cases': cases}
    path = OUTPUT / f'parity-wp5-relief-{topic}.json'
    path.write_text(json.dumps(encode(payload), separators=(',', ':'), allow_nan=False) + '\n')
    assert path.stat().st_size <= 1_500_000, path.stat().st_size
    print(path.name, path.stat().st_size, flush=True)


def main():
    ladders = [dict(resolution=1, featureMinimum=1, featureMaximum=19, scalingFactor=1),
               dict(resolution=1, featureMinimum=3, featureMaximum=19, scalingFactor=2)]
    rows, columns = numpy.indices((17, 23))
    terrain = 500 + (columns - 11)**2 / 16 - (rows - 8)**2 / 8
    terrain[0, 0] = numpy.nan
    terrain[5:8, 6:9] = -9999
    validity = numpy.ones_like(terrain, dtype=int)
    validity[7:10, 12:14] = 0
    wideGeometry = dict(minimumRadius=10, maximumRadius=20, radiusIncrement=1, angularResolution=15)
    cases = [buildCase('flat-high-offset', numpy.full((11, 13), 8192.5), ladders[0], {'kind': 'synthetic'}, geometry=wideGeometry),
             buildCase('saddle-nodata-masks-negative-exaggeration', terrain, ladders[1], {'kind': 'synthetic'},
                       observerHeight=0.5, exaggeration=-0.5, validity=validity),
             buildCase('default-264-offsets-nodata', terrain, ladders[0], {'kind': 'synthetic'},
                       geometry=wideGeometry, observerHeight=4, exaggeration=2, validity=validity),
             buildCase('single-row-replicated-boundaries', numpy.array([[0, 1, 4, 1, 0, -2, 0]], dtype=float),
                       ladders[1], {'kind': 'synthetic'}, observerHeight=4, exaggeration=2),
             buildCase('all-invalid', numpy.full((5, 7), numpy.nan), ladders[0], {'kind': 'synthetic'})]
    numpy.testing.assert_allclose(cases[0]['expected']['dominance'], 1, atol=1e-6)
    numpy.testing.assert_array_equal(cases[0]['expected']['relief'], 0)
    writeFixture('synthetic', cases)
    with rasterio.open(DATA / 'dem.png') as dataset:
        channels = dataset.read().astype(float)
    full = (channels[0] * 256 + channels[1] + channels[2] / 256 - 32768).astype(numpy.float32).astype(float)
    del channels
    manifest = json.loads((DATA / 'manifest.json').read_text())
    source = {'kind': 'Alps adjacent halo tiles', 'path': str((DATA / 'dem.png').relative_to(ROOT)),
              'sha256': hashlib.sha256((DATA / 'dem.png').read_bytes()).hexdigest(),
              'manifestSha256': hashlib.sha256((DATA / 'manifest.json').read_bytes()).hexdigest(),
              'fullShape': list(full.shape), 'coreSize': 96, 'halo': 9,
              'groundCellSize': float(numpy.float32(manifest['raster']['cellSizeGroundM']))}
    # Full reference filters use the real ground resolution in the unchanged
    # square-pixel ladder. Exact integer multiples keep radii at 0..9 / 1,4,9.
    for ladderIndex, ladder in enumerate(ladders):
        spacing = source['groundCellSize']
        scales = {**ladder, 'resolution': spacing, 'featureMinimum': ladder['featureMinimum'] * spacing,
                  'featureMaximum': ladder['featureMaximum'] * spacing}
        fullRelief, _ = computeMultiScale(full, scales, 1)
        united = full[951:1065, 951:1161]
        unitedDominance = computeDominance(united, GEOMETRY, 2, 1)
        for tileIndex in range(2):
            column = 960 + 96 * tileIndex
            elevation = full[951:1065, column - 9:column + 105]
            case = buildCase(f'alps-ladder-{ladderIndex}-tile-{tileIndex}', elevation, scales,
                             {**source, 'coreOrigin': [960, column], 'fullFilterVerified': True,
                              'unitedDominanceVerified': True})
            localRelief = case['expected']['relief'].reshape(elevation.shape)[9:-9, 9:-9]
            numpy.testing.assert_allclose(localRelief, fullRelief[960:1056, column:column + 96], atol=1e-9, rtol=0)
            localDominance = case['expected']['dominance'].reshape(elevation.shape)[9:-9, 9:-9]
            numpy.testing.assert_array_equal(localDominance, unitedDominance[9:-9, 9 + 96 * tileIndex:105 + 96 * tileIndex])
            writeFixture(f'alps-{ladderIndex}-{tileIndex}', [case])


if __name__ == '__main__':
    main()
