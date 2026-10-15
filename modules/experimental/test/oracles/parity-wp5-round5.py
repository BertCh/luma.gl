# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent references for profile peaks, simple local relief, and relief blend.

SLRM: COMPOSITE scipy.ndimage.uniform_filter normalized valid-value convolution,
Hesse (2010), doi:10.1002/arp.374. Square pixel windows, edge replication, missing
neighbors excluded; invalid centers restored. Float64 reference on f32 inputs.
Unmodified rvt.vis.slrm is a secondary reference for its supported radii 10..50;
its f32 mean rounding is bounded separately, never added to GPU tolerance R.
Full Alps DEM decoded once, full-raster filters checked against adjacent 96x96
cores plus 20-cell halos. All exported pixels, including halo edges, are tested.

Blend: unmodified rvt.blend_func.normalize_lin, blend_images and render_images.
Normalize_lin returns f32; subsequent calculations use float64. Bottom-to-top,
linear stretches, single-band, opacity in [0,1], any nonfinite input invalidates
the pixel even for opacity zero. Copies preserve the pre-blend background for
overlay/soft-light opacity (RVT's kernels mutate it). This is kernel parity,
not equality with RVT BlenderCombination's in-place render orchestration.
An import-only matplotlib.cm.get_cmap compatibility alias is process-local;
no blend function or installed package is changed. RGB bytes allow one encoding
level under R; alpha/invalid bytes are exact. No percentile normalization.

Peaks: scipy.signal.find_peaks plateau left_edges (not midpoint), followed by
scipy.ndimage.maximum_filter1d size=5 and explicit earlier-equal exclusion.
scipy.signal.peak_prominences(wlen=2*window+1) supplies all prominence values.
Gaps split finite segments; minimumSide=1 and positive minimumProminence make
segment endpoints ineligible. Circular inputs are triplicated and the central
copy selected, with window capped at n-1. Empty CSR profiles are retained.
Refinement is a COMPOSITE 3-point quadratic fit via numpy.polynomial.polynomial
polyfit and derivative roots (Smith, Spectral Audio Signal Processing, quadratic
interpolation of spectral peaks). No GPU closed-form expression is translated.
Suppression is a COMPOSITE greedy nonmaximum selection using cKDTree radius
queries on refined positions, ranked by prominence then index (Neubeck/Van Gool
2006 doi:10.1109/ICPR.2006.479). SciPy's height-based integer-distance suppression
is deliberately not substituted. All dense outputs and compact IDs are exported;
width is not a GPU output. Eight full 2048-sample Alps transects, no resampling.

Uniform local pixel coordinates; elevation metres; profile distances samples.
All supplied data and numeric GPU settings round to f32 before reference work.
No random inputs. Every fixture is <=1.5 MB and records versions/provenance.
"""
import hashlib
import importlib.metadata
import json
import math
import os
import platform
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/parity-wp5-matplotlib')
import matplotlib
import matplotlib.cm
import numpy
import rasterio
import rvt.vis
from scipy.ndimage import uniform_filter, maximum_filter1d, label
from scipy.signal import find_peaks, peak_prominences
from scipy.spatial import cKDTree

if not hasattr(matplotlib.cm, 'get_cmap'):
    matplotlib.cm.get_cmap = matplotlib.colormaps.get_cmap
import rvt.blend_func

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).parent / 'fixtures'
DATA = ROOT / 'examples/deck/spatial-analysis-showcase/public/data/alps-dem'
VERSIONS = {name: importlib.metadata.version(name) for name in
            ['numpy', 'scipy', 'rasterio', 'rvt-py', 'matplotlib']}
VERSIONS.update(python=platform.python_version(), gdal=rasterio.__gdal_version__)


def encode(value):
    if isinstance(value, dict):
        return {key: encode(entry) for key, entry in value.items()}
    if isinstance(value, (list, tuple, numpy.ndarray)):
        return [encode(entry) for entry in value]
    if isinstance(value, (numpy.integer, numpy.bool_)):
        return int(value)
    if isinstance(value, (float, numpy.floating)):
        if numpy.isnan(value):
            return None
        if numpy.isinf(value):
            return 'Infinity' if value > 0 else '-Infinity'
        return float(value)
    return value


def roundInput(values):
    return numpy.asarray(values, dtype=numpy.float32).astype(float)


def tolerance(values):
    finite = numpy.asarray(values)[numpy.isfinite(values)]
    return {'profile': 'R', 'absolute': 1e-5 * max(1, float(numpy.ptp(finite)) if finite.size else 0),
            'relative': 1e-4}


def writeFixture(topic, cases):
    payload = {'versions': VERSIONS, 'conventions': __doc__,
               'generatorSha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'citations': ['https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_prominences.html',
                             'https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html',
                             'https://github.com/EarthObservation/RVT_py',
                             'https://www.dsprelated.com/freebooks/sasp/Quadratic_Interpolation_Spectral_Peaks.html',
                             'https://doi.org/10.1002/arp.374', 'https://doi.org/10.1109/ICPR.2006.479'],
               'cases': cases}
    path = OUTPUT / f'parity-wp5-round5-{topic}.json'
    path.write_text(json.dumps(encode(payload), separators=(',', ':'), allow_nan=False) + '\n')
    assert path.stat().st_size <= 1_500_000, (path, path.stat().st_size)
    print(path.name, path.stat().st_size, flush=True)


def computeRelief(elevation, radius, exaggeration):
    valid = numpy.isfinite(elevation)
    totals = uniform_filter(numpy.where(valid, elevation, 0), size=2 * radius + 1, mode='nearest')
    counts = uniform_filter(valid.astype(float), size=2 * radius + 1, mode='nearest')
    with numpy.errstate(divide='ignore', invalid='ignore'):
        result = exaggeration * (elevation - totals / counts)
    result[~valid] = numpy.nan
    return result


def buildRelief(name, rawElevation, radius, exaggeration, source, validity=None, defaultRadius=False):
    rawElevation = roundInput(rawElevation)
    validity = numpy.ones_like(rawElevation, dtype=int) if validity is None else validity
    elevation = numpy.where(numpy.isfinite(rawElevation) & (rawElevation != -9999) & (validity != 0), rawElevation, numpy.nan)
    exaggeration = float(numpy.float32(exaggeration))
    relief = computeRelief(elevation, radius, exaggeration)
    crossCheck = {'performed': radius >= 10, 'maxAbs': 0., 'derivedAbsoluteBound': 0.}
    if radius >= 10:
        reference = rvt.vis.slrm(elevation.copy(), radius_cell=radius, ve_factor=exaggeration)
        # RVT casts elevation*ve and the window mean to f32, then subtracts in
        # f32. For the binary-exact ve used here, the product is exact. Mean
        # rounding <= eps*M, final subtraction <= 2*eps*M: 3*eps*M suffices.
        # This checks ONLY the secondary RVT oracle; GPU profile R is unchanged.
        magnitude = max(1, float(numpy.nanmax(numpy.abs(elevation * exaggeration)))) if numpy.any(numpy.isfinite(elevation)) else 1
        bound = 3 * numpy.finfo(numpy.float32).eps * magnitude
        numpy.testing.assert_allclose(reference, relief, atol=bound, rtol=0, equal_nan=True)
        differences = numpy.abs(reference - relief)
        crossCheck.update(maxAbs=float(numpy.nanmax(differences)) if numpy.any(numpy.isfinite(differences)) else 0,
                          derivedAbsoluteBound=bound)
    return {'name': name, 'width': elevation.shape[1], 'height': elevation.shape[0], 'source': source,
            'elevation': rawElevation.ravel(), 'validity': validity.ravel(), 'maximumRadius': max(20, radius),
            'radius': radius, 'defaultRadius': defaultRadius, 'verticalExaggeration': exaggeration,
            'expected': relief.ravel(), 'expectedValidity': numpy.isfinite(elevation).astype(int).ravel(),
            'tolerance': tolerance(relief), 'rvtCrossCheck': crossCheck}


def buildBlend(name, images, settings, source):
    images = [roundInput(values) for values in images]
    settings = [{key: (float(numpy.float32(value)) if key in ['minimum', 'maximum', 'opacity'] else value)
                 for key, value in layer.items()} for layer in settings]
    valid = numpy.all(numpy.isfinite(images), axis=0)
    normalized = []
    for values, layer in zip(images, settings):
        normalizedImage = rvt.blend_func.normalize_lin(values.copy(), layer['minimum'], layer['maximum']).astype(float)
        normalized.append(1 - normalizedImage if layer.get('invert', False) else normalizedImage)
    result = normalized[0].copy()
    for active, layer in zip(normalized[1:], settings[1:]):
        # Preserve original background: overlay/soft_light mutate their input.
        blended = rvt.blend_func.blend_images(layer['blendMode'].replace('-', '_'), active.copy(), result.copy())
        result = rvt.blend_func.render_images(blended, result, layer['opacity'])
    result[~valid] = numpy.nan
    grey = numpy.floor(numpy.clip(numpy.nan_to_num(result), 0, 1) * 255 + 0.5).astype(int)
    color = numpy.where(valid, grey + (grey << 8) + (grey << 16) + (255 << 24), 0)
    return {'name': name, 'width': result.shape[1], 'height': result.shape[0], 'source': source,
            'layers': [values.ravel() for values in images], 'settings': settings,
            'expected': result.ravel(), 'expectedColor': color.ravel(), 'expectedValidity': valid.astype(int).ravel(),
            'tolerance': tolerance(result)}


def buildPeaks(name, profiles, window, suppressionRadius, minimumProminence, source, wrap=False, validities=None):
    profiles = [roundInput(profile) for profile in profiles]
    validities = [numpy.ones(len(profile), dtype=int) for profile in profiles] if validities is None else validities
    offsets = numpy.concatenate(([0], numpy.cumsum([len(profile) for profile in profiles])))
    values = numpy.concatenate(profiles)
    validity = numpy.concatenate(validities)
    result = {key: numpy.full(len(values), numpy.nan) for key in ['prominence', 'refinedIndex', 'refinedValue']}
    mask = numpy.zeros(len(values), dtype=int)
    candidateCount = 0
    suppressionMargin = math.inf
    for profileIndex, (profile, valid) in enumerate(zip(profiles, validities)):
        count = len(profile)
        if count < 3:
            continue
        data = numpy.where(numpy.isfinite(profile) & (valid != 0), profile, numpy.nan)
        extended = numpy.tile(data, 3) if wrap else data
        finite = numpy.isfinite(extended)
        segments, segmentCount = label(finite)
        candidates = []
        for segment in range(1, segmentCount + 1):
            locations = numpy.flatnonzero(segments == segment)
            start, stop = locations[0], locations[-1] + 1
            _, properties = find_peaks(extended[start:stop], plateau_size=(None, None))
            candidates.extend((properties['left_edges'] + start).tolist())
        maxima = maximum_filter1d(numpy.where(finite, extended, -numpy.inf), size=5, mode='constant', cval=-numpy.inf)
        ranked = []
        halfWindow = min(window, count - 1) if wrap else window
        for position in candidates:
            if wrap and not count <= position < 2 * count:
                continue
            if extended[position] != maxima[position]:
                continue
            if numpy.any(extended[max(0, position - 2):position] == extended[position]):
                continue
            segment = numpy.flatnonzero(segments == segments[position])
            start, stop = segment[0], segment[-1] + 1
            prominence = float(peak_prominences(extended[start:stop], [position - start], wlen=2 * halfWindow + 1)[0][0])
            if prominence < minimumProminence:
                continue
            localPosition = position % count
            # Fit differences from center, avoiding a large elevation offset in
            # a tiny three-point fit; the vertex of the polynomial is independent.
            coefficients = numpy.polynomial.polynomial.polyfit([-1., 0., 1.], extended[position - 1:position + 2] - extended[position], 2)
            delta = 0.
            refinedValue = extended[position]
            if coefficients[2] < 0:
                vertex = numpy.polynomial.polynomial.polyroots(numpy.polynomial.polynomial.polyder(coefficients))[0]
                delta = float(numpy.clip(vertex, -0.5, 0.5))
                refinedValue += numpy.polynomial.polynomial.polyval(delta, coefficients)
            refinedIndex = (localPosition + delta) % count if wrap else localPosition + delta
            # A negative sub-epsilon vertex offset can round modulo n to n;
            # canonicalize that equivalent circular coordinate to zero.
            if wrap and refinedIndex >= count:
                refinedIndex = 0.
            ranked.append((localPosition, prominence, refinedIndex, refinedValue))
        candidateCount += len(ranked)
        if not ranked:
            continue
        positions = numpy.asarray([peak[2] for peak in ranked])
        tree = cKDTree(positions[:, None], boxsize=count if wrap else None)
        blocked = set()
        for candidate in sorted(range(len(ranked)), key=lambda index: (-ranked[index][1], ranked[index][0])):
            if candidate in blocked:
                continue
            position, prominence, refinedIndex, refinedValue = ranked[candidate]
            distances = numpy.abs(positions - refinedIndex)
            if wrap:
                distances = numpy.minimum(distances, count - distances)
            if len(distances) > 1:
                suppressionMargin = min(suppressionMargin, float(numpy.min(numpy.abs(numpy.delete(distances, candidate) - suppressionRadius))))
            blocked.update(tree.query_ball_point([refinedIndex], suppressionRadius))
            index = offsets[profileIndex] + position
            mask[index] = 1
            for key, value in [('prominence', prominence), ('refinedIndex', refinedIndex), ('refinedValue', refinedValue)]:
                result[key][index] = value
    # Nonzero distances near the suppression boundary would need conditioned
    # comparisons; these deterministic fixtures deliberately keep a clear gap.
    assert not math.isfinite(suppressionMargin) or suppressionMargin > 0.001, suppressionMargin
    return {'name': name, 'source': source, 'values': values, 'validity': validity, 'offsets': offsets,
            'window': window, 'suppressionRadius': suppressionRadius, 'minimumProminence': minimumProminence,
            'wrap': wrap, 'expected': result, 'expectedMask': mask, 'expectedIds': numpy.flatnonzero(mask),
            'tolerances': {key: tolerance(value) for key, value in result.items()},
            'candidateCount': candidateCount, 'suppressionBoundaryMargin': suppressionMargin}


def main():
    with rasterio.open(DATA / 'dem.png') as raster:
        channels = raster.read().astype(float)
    full = roundInput(channels[0] * 256 + channels[1] + channels[2] / 256 - 32768)
    del channels
    source = {'kind': 'Alps DEM', 'path': str((DATA / 'dem.png').relative_to(ROOT)),
              'sha256': hashlib.sha256((DATA / 'dem.png').read_bytes()).hexdigest(),
              'manifestSha256': hashlib.sha256((DATA / 'manifest.json').read_bytes()).hexdigest(),
              'fullShape': list(full.shape)}
    for radius in [10, 20]:
        fullRelief = computeRelief(full, radius, 1)
        for tileIndex in range(2):
            column = 960 + 96 * tileIndex
            elevation = full[940:1076, column - 20:column + 116]
            case = buildRelief(f'alps-radius-{radius}-tile-{tileIndex}', elevation, radius, 1,
                               {**source, 'coreOrigin': [960, column], 'coreSize': 96, 'halo': 20, 'fullFilterVerified': True},
                               defaultRadius=radius == 20)
            numpy.testing.assert_allclose(case['expected'].reshape(elevation.shape)[20:-20, 20:-20],
                                          fullRelief[960:1056, column:column + 96], atol=2e-9, rtol=0)
            writeFixture(f'slrm-alps-{radius}-{tileIndex}', [case])
    rows, columns = numpy.indices((17, 23))
    terrain = 4096 + columns / 32 + rows / 64 + ((columns - 11)**2 - (rows - 8)**2) / 128
    terrain[0, 0] = numpy.nan
    terrain[6:9, 7:10] = -9999
    validity = numpy.ones_like(terrain, dtype=int)
    validity[8:11, 13:16] = 0
    synthetic = {'kind': 'synthetic'}
    reliefCases = [buildRelief('masked-high-offset-negative', terrain, 10, -0.5, synthetic, validity),
                   buildRelief('flat-high-offset', numpy.full((13, 15), 8192.5), 20, 2, synthetic),
                   buildRelief('all-invalid', numpy.full((5, 7), numpy.nan), 10, 1, synthetic),
                   buildRelief('single-row-replicated', [[0., 1., 8., numpy.nan, 2., 1.]], 1, 1, synthetic),
                   buildRelief('single-column-replicated', [[0.], [1.], [8.], [numpy.nan], [2.]], 1, 1, synthetic),
                   buildRelief('sub-quantum-values', [[0., 1e-6, 2e-5, -1e-5, 1e-4]], 1, 1, synthetic)]
    writeFixture('slrm-synthetic', reliefCases)
    modes = ['normal', 'multiply', 'screen', 'overlay', 'soft-light', 'luminosity']
    axis = numpy.array([-0.25, 0, 0.25, 0.499, 0.5, 0.501, 0.75, 1, 1.25])
    background, foreground = numpy.meshgrid(axis, axis)
    blendCases = []
    for mode in modes:
        for opacity in [0., 0.375, 1.]:
            blendCases.append(buildBlend(f'{mode}-opacity-{opacity}', [background, foreground],
                [dict(minimum=0, maximum=1, blendMode='screen', opacity=0),
                 dict(minimum=0, maximum=1, blendMode=mode, opacity=opacity, invert=True)], synthetic))
    invalidImages = [background.copy() for _ in range(5)]
    for index, image in enumerate(invalidImages):
        image.ravel()[index] = [numpy.nan, numpy.inf, -numpy.inf, numpy.nan, numpy.nan][index]
    blendCases.append(buildBlend('five-layer-invalid-even-zero-opacity', invalidImages,
        [dict(minimum=-0.25, maximum=1.25, blendMode=mode, opacity=0 if index == 4 else 0.5)
         for index, mode in enumerate(modes[:5])], synthetic))
    blendCases.append(buildBlend('single-bottom-ignores-opacity', [background],
        [dict(minimum=0, maximum=1, invert=True, blendMode='multiply', opacity=0)], synthetic))
    writeFixture('blend-synthetic', blendCases)
    for tileIndex in range(2):
        column = 960 + tileIndex * 64
        elevation = full[950:1034, column - 10:column + 74]
        relief = computeRelief(elevation, 10, 1)[10:-10, 10:-10]
        hillshade = rvt.vis.hillshade(elevation.copy(), resolution_x=1, resolution_y=1)[10:-10, 10:-10]
        slope = rvt.vis.slope_aspect(elevation.copy(), resolution_x=1, resolution_y=1, output_units='degree')['slope'][10:-10, 10:-10]
        images = [hillshade, slope, relief, elevation[10:-10, 10:-10]]
        settings = [dict(minimum=0, maximum=1, blendMode='normal', opacity=1),
                    dict(minimum=0, maximum=85, invert=True, blendMode='luminosity', opacity=0.5),
                    dict(minimum=-20, maximum=20, blendMode='overlay', opacity=0.375),
                    dict(minimum=1500, maximum=3500, blendMode='soft-light', opacity=0.75)]
        writeFixture(f'blend-alps-{tileIndex}', [buildBlend(f'alps-blend-{tileIndex}', images, settings,
            {**source, 'coreOrigin': [960, column], 'coreSize': 64, 'halo': 10,
             'layers': 'RVT hillshade, RVT slope degrees, SciPy SLRM, elevation; pixel resolution=1'})])
    profiles = [full[row, :] for row in [256, 768, 1024, 1536]] + [full[:, column] for column in [256, 768, 1024, 1536]]
    for window, suppressionRadius in [(32, 0), (128, 5)]:
        writeFixture(f'peaks-alps-{window}', [buildPeaks(f'alps-eight-full-transects-window-{window}', profiles, window,
            suppressionRadius, 0.5, {**source, 'rows': [256, 768, 1024, 1536], 'columns': [256, 768, 1024, 1536],
                                   'fullTransects': True})])
    plateauProfiles = [[], [4], [1, 2], [0, 1, 4, 4, 4, 1, 0], [0, 1, 4, 4, 1, 0],
                       [0, 1, 6, 1, 0, 1, 6, 1, 0, 1, 6, 1, 0], [3] * 20,
                       [0, 2, 7, 2, 0, numpy.nan, 0, 1, 5, 1, 0],
                       [0, 1, 9, 1, 0, numpy.inf, 0, 1, 4, 1, 0]]
    validities = [numpy.ones(len(profile), dtype=int) for profile in plateauProfiles]
    validities[-1][2] = 0
    samplePositions = numpy.arange(360)
    circular = 10 * numpy.cos(2 * numpy.pi * samplePositions / 360) + 2 * numpy.cos(8 * numpy.pi * samplePositions / 360)
    peakCases = [buildPeaks('plateaus-gaps-empty-and-short', plateauProfiles, 12, 0, 0.25, synthetic, validities=validities),
                 buildPeaks('prominence-ranked-tied-suppression', plateauProfiles, 12, 5, 0.25, synthetic, validities=validities),
                 buildPeaks('circular-seam-and-multiple-peaks', [circular, numpy.roll(circular, 53)], 120, 8, 0.25, synthetic, wrap=True)]
    writeFixture('peaks-synthetic', peakCases)


if __name__ == '__main__':
    main()
