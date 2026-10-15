# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent curvature references. Run with the parity venv's python -I.

Direct reference: xdem.terrain.get_terrain_attribute, geometric profile, planform
and tangential curvature, ZevenbergThorne and Florinsky fits, scipy engine,
float64 output; divide xDEM's 100/m outputs by 100 (convex positive).
SciPy ndimage.convolve applies Hessian stencils derived by numpy.linalg.pinv
from polynomial design matrices, NOT copied finite-difference coefficients.
Evans-Young has no xDEM implementation: its three geometric curvatures are a
composite reference (Evans 1979 / Minar et al. 2020) from that quadratic fit.
References: https://doi.org/10.1002/esp.3290120107 (partial biquadratic, 3x3),
https://doi.org/10.1177/0309133317733667 (Florinsky, total cubic, 5x5),
https://doi.org/10.1016/j.earscirev.2020.103414 (geometric curvatures),
https://xdem.readthedocs.io/en/latest/terrain.html (units and conventions).

Contract: float32-rounded elevations and parameters, float64 reference math;
pixel centers, east-positive columns, row direction explicitly recorded. Heights
and horizontal spacing are metres. All five derivatives are independently
computed, but the GPU API exposes only curvature outputs, so their trace is
compared as laplacian (1/m). No claim of direct individual derivative readback.
Flat gradient <= 1e-6 yields zero directional curvature. NaN elevations and
validity=0 invalidate the entire fitting window. clamp repeats boundary cells;
nodata invalidates boundary windows. No smoothing, resampling, or random seed.

Real input is the complete decoded showcase Alps Terrarium PNG. Two adjoining
96x96 cores at [row=960, col=960/1056], with a two-cell halo, fit the 1.5 MB
per-fixture cap. They are a reproducible subset, NOT full 512x512/full DEM parity.
The uniform 6.6390088 m local ground approximation is explicit: this does not
validate the showcase's row-dependent Web Mercator cell-size mode. Full input
tiles and all output cells (including halo/boundary validity) are serialized.
The generator checks the shared seam against a reference on the united window.
Synthetic cases exercise planes, mixed quadratics, cubic fits, a flat peak,
nodata and validity holes, anisotropic spacing, both row and border directions.
Multi-radius ring curvature and ancillary derived curvature kinds are not covered.
"""

import hashlib
import json
import math
import platform
import warnings
from pathlib import Path

import numpy
import rasterio
import scipy
from scipy.ndimage import convolve, minimum_filter
import xdem

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).parent / 'fixtures'
METHODS = ['evans-young', 'zevenbergen-thorne', 'florinsky']
KINDS = ['profile', 'plan', 'tangential', 'laplacian']
VERSIONS = {
    'python': platform.python_version(), 'numpy': numpy.__version__,
    'scipy': scipy.__version__, 'xdem': xdem.__version__,
    'rasterio': rasterio.__version__, 'gdal': rasterio.__gdal_version__,
}


def encode(value):
    if isinstance(value, dict):
        return {key: encode(entry) for key, entry in value.items()}
    if isinstance(value, (list, tuple, numpy.ndarray)):
        return [encode(entry) for entry in value]
    if isinstance(value, numpy.integer):
        return int(value)
    if isinstance(value, (float, numpy.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def getPartials(elevation, method, cellSize, rowDirection, borderMode, validity):
    radius = 2 if method == 'florinsky' else 1
    offsets = numpy.arange(-radius, radius + 1, dtype=numpy.float64)
    columns, rows = numpy.meshgrid(offsets, offsets)
    north = rows * (-1 if rowDirection == 'south' else 1)
    if method == 'zevenbergen-thorne':
        powers = [(east, northward) for east in range(3) for northward in range(3)]
    else:
        degree = 3 if method == 'florinsky' else 2
        powers = [(east, northward) for east in range(degree + 1)
                  for northward in range(degree + 1 - east)]
    design = numpy.stack([(columns ** east * north ** northward).ravel()
                          for east, northward in powers], axis=1)
    inverse = numpy.linalg.pinv(design)
    mode = 'nearest' if borderMode == 'clamp' else 'constant'
    windowValidity = minimum_filter(
        (numpy.isfinite(elevation) & (validity != 0)).astype(numpy.uint8),
        size=2 * radius + 1, mode=mode, cval=0).astype(bool)
    # A global constant subtraction improves conditioning without changing derivatives.
    centered = numpy.where(numpy.isfinite(elevation), elevation - 4000, 0)
    partials = {}
    for name, derivative in [('east', (1, 0)), ('north', (0, 1)),
                             ('eastEast', (2, 0)), ('eastNorth', (1, 1)),
                             ('northNorth', (0, 2))]:
        east, northward = derivative
        kernel = inverse[powers.index(derivative)].reshape(columns.shape)
        kernel = kernel * math.factorial(east) * math.factorial(northward)
        # Remove pseudoinverse roundoff; the center enforces zero response to constants.
        kernel[numpy.abs(kernel) < 1e-14] = 0
        kernel[radius, radius] -= kernel.sum()
        kernel /= cellSize[0] ** east * cellSize[1] ** northward
        # convolve reverses the kernel; the design describes offsets around the pixel.
        values = convolve(centered, kernel[::-1, ::-1], mode=mode, cval=0)
        partials[name] = numpy.where(windowValidity, values, numpy.nan)
    return partials, windowValidity


def computeReference(elevation, method, cellSize, rowDirection, borderMode, validity, zFactor=1):
    scaled = elevation.astype(numpy.float64) * zFactor
    partials, windowValidity = getPartials(
        scaled, method, cellSize, rowDirection, borderMode, validity)
    east, north = partials['east'], partials['north']
    eastEast, eastNorth, northNorth = (
        partials['eastEast'], partials['eastNorth'], partials['northNorth'])
    gradientSquared = east * east + north * north
    flat = gradientSquared <= float(numpy.float32(1e-6)) ** 2
    safeGradient = numpy.where(flat, 1, gradientSquared)
    normalSquared = 1 + gradientSquared
    # Composite Evans-Young geometric curvatures, Minar et al. (2020).
    along = east * east * eastEast + 2 * east * north * eastNorth + north * north * northNorth
    across = north * north * eastEast - 2 * east * north * eastNorth + east * east * northNorth
    outputs = {
        'profile': numpy.where(flat, 0, -along / (safeGradient * normalSquared ** 1.5)),
        'plan': numpy.where(flat, 0, -across / safeGradient ** 1.5),
        'tangential': numpy.where(flat, 0, -across / (safeGradient * numpy.sqrt(normalSquared))),
        'laplacian': eastEast + northNorth,
    }
    if method != 'evans-young' and cellSize[0] == cellSize[1]:
        radius = 2 if method == 'florinsky' else 1
        masked = numpy.where(validity != 0, scaled, numpy.nan)
        padded = (numpy.pad(masked, radius, mode='edge') if borderMode == 'clamp'
                  else numpy.pad(masked, radius, mode='constant', constant_values=numpy.nan))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            external = xdem.terrain.get_terrain_attribute(
                padded, ['profile_curvature', 'planform_curvature', 'tangential_curvature'],
                resolution=cellSize[0], surface_fit=('Florinsky' if method == 'florinsky'
                else 'ZevenbergThorne'), curv_method='geometric', engine='scipy', out_dtype=numpy.float64)
        for kind, values in zip(KINDS[:3], external):
            values = numpy.asarray(values)[radius:-radius, radius:-radius] / 100
            values = numpy.where(flat & windowValidity, 0, values)
            # Independent fits must agree before the fixture is allowed to be written.
            numpy.testing.assert_allclose(values, outputs[kind], atol=2e-10, rtol=2e-9, equal_nan=True)
            outputs[kind] = values
    return outputs, windowValidity, partials


def makeCase(name, elevation, method, cellSize, borderMode='nodata', rowDirection='south',
             validity=None, zFactor=1, source=None, includePartials=False):
    elevation = numpy.asarray(elevation, dtype=numpy.float32).astype(numpy.float64)
    cellSize = [float(numpy.float32(value)) for value in cellSize]
    zFactor = float(numpy.float32(zFactor))
    if validity is None:
        validity = numpy.ones(elevation.shape, dtype=numpy.uint8)
    outputs, windowValidity, partials = computeReference(
        elevation, method, cellSize, rowDirection, borderMode, validity, zFactor)
    direct = method != 'evans-young' and cellSize[0] == cellSize[1]
    references = {kind: ('xdem.terrain.get_terrain_attribute (geometric, /100)' if direct
                  else 'composite reference: Evans/Minar geometric curvature + scipy.ndimage.convolve')
                  for kind in KINDS[:3]}
    references['laplacian'] = 'scipy.ndimage.convolve (independent polynomial-fit Hessian trace)'
    result = {
        'name': name, 'width': elevation.shape[1], 'height': elevation.shape[0],
        'method': method, 'cellSize': cellSize, 'borderMode': borderMode,
        'rowDirection': rowDirection, 'zFactor': zFactor, 'references': references,
        'source': source or {'kind': 'synthetic', 'crs': 'local Cartesian metres'},
        'elevation': elevation.ravel(), 'inputValidity': validity.ravel(),
        'elevationFloat32Sha256': hashlib.sha256(elevation.astype('<f4').tobytes()).hexdigest(),
        'expectedValidity': windowValidity.astype(numpy.uint8).ravel(),
        'expected': {kind: values.ravel() for kind, values in outputs.items()},
        'tolerances': {kind: {'profile': 'R', 'absolute': 1e-5 * max(1, float(
            numpy.nanmax(values) - numpy.nanmin(values))), 'relative': 1e-4}
            for kind, values in outputs.items()},
    }
    if includePartials:
        result['partials'] = {name: values.ravel() for name, values in partials.items()}
    return result


def writeFixture(name, cases):
    payload = {'versions': VERSIONS, 'generator': Path(__file__).name,
               'generatorSha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'conventions': {
                   'inputs': 'float32-rounded elevation, cell sizes, zFactor; float64 reference arithmetic',
                   'units': 'metres for elevation and spacing; all compared curvatures in 1/metre',
                   'sign': 'convex positive for profile/plan/tangential; ordinary Hessian trace for laplacian',
                   'nodata': 'null decodes to NaN; any invalid fitting-window cell invalidates output',
                   'boundary': 'clamp repeats edge cells; nodata invalidates incomplete windows',
                   'flatGradient': float(numpy.float32(1e-6)),
                   'seamCheck': 'all real core outputs checked against a united window during generation',
                   'citations': ['https://doi.org/10.1002/esp.3290120107',
                                 'https://doi.org/10.1177/0309133317733667',
                                 'https://doi.org/10.1016/j.earscirev.2020.103414',
                                 'https://xdem.readthedocs.io/en/latest/terrain.html'],
               },
               'cases': cases}
    output = OUTPUT / f'parity-wp5-terrain-curvature-{name}.json'
    text = json.dumps(encode(payload), allow_nan=False, separators=(',', ':')) + '\n'
    assert len(text.encode()) <= 1_500_000, (output.name, len(text))
    output.write_text(text)
    print(f'{output.name}: {len(text)} bytes, {len(cases)} cases')


def main():
    dataset = ROOT / 'examples/deck/spatial-analysis-showcase/public/data/alps-dem'
    manifest = json.loads((dataset / 'manifest.json').read_text())
    imagePath = dataset / 'dem.png'
    with rasterio.open(imagePath) as image:
        channels = image.read().astype(numpy.float64)
    elevation = (channels[0] * 256 + channels[1] + channels[2] / 256 - 32768).astype(numpy.float32)
    source = {'path': str(imagePath.relative_to(ROOT)),
              'sha256': hashlib.sha256(imagePath.read_bytes()).hexdigest(),
              'manifestSha256': hashlib.sha256((dataset / 'manifest.json').read_bytes()).hexdigest(),
              'sourceShape': list(elevation.shape), 'sourceCrs': 'EPSG:3857',
              'computationCrs': 'local Cartesian metres; uniform manifest cellSizeGroundM',
              'encoding': 'Terrarium', 'nodata': 'none', 'halo': 2, 'coreSize': [96, 96]}
    cellSize = [manifest['raster']['cellSizeGroundM']] * 2
    united = elevation[958:1058, 958:1154]
    for method in METHODS:
        cases = []
        unitedOutputs, _, _ = computeReference(united, method,
            list(map(lambda value: float(numpy.float32(value)), cellSize)), 'south', 'nodata',
            numpy.ones(united.shape, dtype=numpy.uint8))
        for tile, column in enumerate([960, 1056]):
            case = makeCase(f'alps-adjacent-{tile}-{method}',
                elevation[958:1058, column - 2:column + 98], method, cellSize,
                source={**source, 'coreOrigin': [960, column]})
            # All core cells including the seam agree with the united-window reference.
            for kind in KINDS:
                core = case['expected'][kind].reshape(100, 100)[2:-2, 2:-2]
                numpy.testing.assert_allclose(core, unitedOutputs[kind][2:-2,
                    2 + tile * 96:98 + tile * 96], atol=1e-12, rtol=1e-12, equal_nan=True)
            cases.append(case)
            writeFixture(f'alps-{method}-{tile}', [case])

        rows, columns = numpy.indices((13, 15), dtype=numpy.float64)
        east, north = columns - 7, 6 - rows
        surfaces = {
            'mixed-quadratic': 4096 + east * 2 + north - east ** 2 / 4 + east * north / 8 + north ** 2 / 2,
            'cubic': 4096 + east ** 3 / 8 + east * north ** 2 / 4 - north ** 3 / 16,
            'flat': numpy.full(rows.shape, 4321.125),
            'dome': 4096 - east ** 2 / 4 - north ** 2 / 8,
            'nodata': 4096 + east ** 2 / 4 - north ** 2 / 8,
        }
        synthetic = []
        for name, surface in surfaces.items():
            validity = numpy.ones(surface.shape, dtype=numpy.uint8)
            if name == 'nodata':
                surface[4, 5] = numpy.nan
                validity[8, 10] = 0
            for borderMode in ['clamp', 'nodata']:
                for rowDirection in ['south', 'north']:
                    synthetic.append(makeCase(f'{name}-{borderMode}-{rowDirection}-{method}', surface,
                        method, [2, 2], borderMode, rowDirection, validity, zFactor=1.25,
                        includePartials=True))
        # Anisotropic Evans fit has no xDEM equivalent (xDEM requires square cells).
        if method == 'evans-young':
            synthetic.append(makeCase('anisotropic-evans-young', surfaces['mixed-quadratic'],
                method, [2, 3], 'clamp', 'south', zFactor=1.5, includePartials=True))
        writeFixture(f'synthetic-{method}', synthetic)


if __name__ == '__main__':
    main()
