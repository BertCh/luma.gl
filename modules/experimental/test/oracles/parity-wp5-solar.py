# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent solar references; run with the designated parity Python -I.

GPUSolarIrradiance: pvlib.irradiance.get_total_irradiance, isotropic sky,
albedo=0. Direct-normal irradiance is supplied, not inferred by the GPU. The
Meinel input uses pvlib.atmosphere.get_relative_airmass(kastenyoung1989) and
1353 * 0.7**(airmass**0.678), explicitly a composite reference. No seasonal
extraterrestrial correction: get_extra_radiation is NOT the Meinel contract.
Alps skyViewFactor=(1+cos(tilt))/2 matches isotropic diffuse transposition.
Synthetic arbitrary sky-view factors scale pvlib's HORIZONTAL diffuse term;
they are a custom composite, not a claim of pvlib terrain-obstruction parity.
Energy is the float64 rectangular sum of supplied midpoint samples (Wh/m2).

GPUSolarPosition: pvlib.solarposition.spa_python, UTC, pressure=0 (no
refraction), altitude=0 m, delta_t=67 seconds, numpy backend. All sites and
the split J2000 time are first rounded to f32; SPA evaluates the represented
instant, not an unrounded requested instant. Degrees clockwise from north;
daylight is geometric elevation > f32(-0.833). NOAA versus NREL SPA is an
intentional strict diagnostic, NOT assumed to satisfy the D budget. No model
error is absorbed into float tolerance. Refraction and observer height are
not covered because the GPU has no matching SPA atmosphere/height controls.

GPUSolarShadowMask: scipy.stats.semicircular.cdf gives the visible fraction
of a uniform circular disk cut by a straight horizon. This is an independent
library distribution, verified against scipy.integrate.quad of disk chords.
numpy.interp(period=360) supplies linear compass-sector interpolation. pvlib
get_total_irradiance supplies Lambertian incidence. This composite reference
tests the public horizon-to-mask operator, NOT DEM horizon generation or a
GRASS/GDAL viewshed. It has no observer/distance/CRS inputs. No existing
viewshed function implements this finite-solar-disk postprocessor.
Hard equality is shadowed (strict altitude>horizon). A disk entirely below
the astronomical horizon is zero; for a partially risen disk we preserve the
documented local-horizon convention. Invalid sampled horizons invalidate both
outputs; invalid slope/aspect/sky-view invalidate illumination only. Irradiance
instead uses sector-zero and all geometry validity for both outputs.

All GPU inputs are f32-rounded, reference operations f64. Real geometry uses
the complete decoded Alps DEM, uniform local ground metres, north-up pixel
centres; NumPy gradient generates input normals, NOT derivative parity.
Two adjacent 64x64 subsets have full-DEM halo/seam checks. The 1.5 MB cap
requires subsets rather than 512x512 arrays. Controlled horizons are synthetic
even on the real geometry; no real terrain-shadow parity is claimed. No RNG.
Every cell and every timestamp output is serialized. See fixture citations.
"""

import hashlib
import json
import math
import platform
from pathlib import Path

import numpy
import pandas
import pvlib
import rasterio
import scipy
from scipy.integrate import quad
from scipy.stats import semicircular

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).parent / 'fixtures'
DATA = ROOT / 'examples/deck/spatial-analysis-showcase/public/data/alps-dem'
VERSIONS = {name: module.__version__ for name, module in [
    ('numpy', numpy), ('pandas', pandas), ('pvlib', pvlib), ('scipy', scipy),
    ('rasterio', rasterio)]}
VERSIONS.update(python=platform.python_version(), gdal=rasterio.__gdal_version__)
CITATIONS = [
    'https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.irradiance.get_total_irradiance.html',
    'https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.atmosphere.get_relative_airmass.html',
    'https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.solarposition.spa_python.html',
    'https://doi.org/10.1016/j.solener.2003.12.003',
    'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.semicircular.html',
]


def roundFloat32(value):
    return numpy.asarray(value, dtype=numpy.float32).astype(numpy.float64)


def encode(value):
    if isinstance(value, dict):
        return {key: encode(entry) for key, entry in value.items()}
    if isinstance(value, (list, tuple, numpy.ndarray)):
        return [encode(entry) for entry in value]
    if isinstance(value, numpy.integer):
        return int(value)
    if isinstance(value, (float, numpy.floating)):
        return (float(value) if math.isfinite(value) else None if math.isnan(value)
                else 'Infinity' if value > 0 else '-Infinity')
    return value


def writeFixture(topic, cases):
    payload = {
        'versions': VERSIONS, 'generator': Path(__file__).name,
        'generatorSha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'citations': CITATIONS, 'conventions': __doc__, 'cases': cases,
    }
    path = OUTPUT / f'parity-wp5-solar-{topic}.json'
    path.write_text(json.dumps(encode(payload), separators=(',', ':'), allow_nan=False) + '\n')
    assert path.stat().st_size <= 1_500_000, (path.name, path.stat().st_size)
    print(path.name, path.stat().st_size, 'bytes', len(cases), 'cases')


def rasterTolerance(values):
    finite = numpy.asarray(values)[numpy.isfinite(values)]
    span = float(numpy.ptp(finite)) if finite.size else 0
    return {'profile': 'R', 'absolute': 1e-5 * max(1, span), 'relative': 1e-4}


def getVisibility(horizon, azimuth, altitude, radius):
    sectors = numpy.arange(horizon.shape[1]) * 360 / horizon.shape[1]
    sampled = numpy.array([numpy.interp(azimuth, sectors, pixel, period=360)
                           for pixel in horizon])
    # Distribution of vertical coordinates of points uniform in a unit disk.
    if radius > 0:
        visibility = semicircular.cdf((altitude - sampled) / radius)
    else:
        visibility = (altitude > sampled).astype(float)
    if altitude + radius <= 0:
        visibility[:] = 0
    visibility[~numpy.isfinite(sampled)] = numpy.nan
    return visibility


def getDirectIrradiance(slope, aspect, azimuth, altitude, directNormal):
    # Aspect=-1 denotes a horizontal surface, regardless of the slope input.
    tilt = numpy.where(aspect < 0, 0, slope)
    surfaceAzimuth = numpy.where(aspect < 0, 0, aspect)
    return pvlib.irradiance.get_total_irradiance(
        tilt, surfaceAzimuth, 90 - altitude, azimuth,
        dni=directNormal, ghi=0, dhi=0, albedo=0, model='isotropic')['poa_direct']


def makeIrradianceCase(name, geometry, sunTable, diffuse=60, radius=0, sampleCount=None):
    slope, aspect, skyView, horizon = [geometry[key] for key in
                                      ['slope', 'aspect', 'skyViewFactor', 'horizon']]
    table = roundFloat32(sunTable).reshape(-1, 4)
    diffuse, radius = float(roundFloat32(diffuse)), float(roundFloat32(radius))
    if sampleCount is None:
        sampleCount = len(table)
    pixelCount = len(slope)
    directEnergy = numpy.zeros(pixelCount)
    diffuseEnergy = numpy.zeros(pixelCount)
    hours = numpy.zeros(pixelCount)
    valid = (numpy.isfinite(horizon[:, 0]) & numpy.isfinite(slope)
             & numpy.isfinite(aspect) & numpy.isfinite(skyView))
    for azimuth, altitude, duration, directNormal in table[:sampleCount]:
        visibility = getVisibility(horizon, azimuth, altitude, radius)
        hours += visibility * duration
        directEnergy += getDirectIrradiance(slope, aspect, azimuth, altitude,
                                          directNormal) * visibility * duration
        if altitude > 0:
            horizontalDiffuse = pvlib.irradiance.get_total_irradiance(
                0, 0, 90 - altitude, azimuth, dni=0, ghi=diffuse, dhi=diffuse,
                albedo=0, model='isotropic')['poa_diffuse']
            diffuseEnergy += horizontalDiffuse * skyView * duration
    expected = {'sunHours': hours, 'direct': directEnergy, 'diffuse': diffuseEnergy,
                'total': directEnergy + diffuseEnergy}
    for values in expected.values():
        values[~valid] = numpy.nan
    return {**geometry, 'name': name, 'horizon': horizon.ravel(),
            'sunTable': table.ravel(), 'sampleCount': sampleCount,
            'diffuseIrradiance': diffuse, 'angularRadiusDegrees': radius,
            'expected': expected, 'expectedValidity': valid.astype(numpy.uint32),
            'tolerances': {key: rasterTolerance(values) for key, values in expected.items()},
            'reference': 'pvlib.irradiance.get_total_irradiance + scipy.stats.semicircular.cdf; float64 time sum'}


def makeMaskCase(name, geometry, azimuth, altitude, radius, sun=1.25, ambient=0.2):
    azimuth, altitude, radius, sun, ambient = map(float, roundFloat32(
        [azimuth, altitude, radius, sun, ambient]))
    slope, aspect, skyView, horizon = [geometry[key] for key in
                                      ['slope', 'aspect', 'skyViewFactor', 'horizon']]
    visibility = getVisibility(horizon, azimuth, altitude, radius)
    illumination = getDirectIrradiance(slope, aspect, azimuth, altitude, sun) * visibility
    illumination += ambient * skyView
    illumination[~(numpy.isfinite(slope) & numpy.isfinite(aspect)
                   & numpy.isfinite(skyView))] = numpy.nan
    # D angle budget propagated through the semicircle CDF. Its derivative is
    # at most 2/(pi*radius). This is a conditioned dimensionless bound, not a
    # relaxed angle budget. Hard visibility is exact, including ties.
    visibilityTolerance = ({'profile': 'E', 'absolute': 0, 'relative': 0} if radius == 0
                           else {'profile': 'D (disk CDF)',
                                 'absolute': 2 / (math.pi * radius) * 1e-4 + 1e-6,
                                 'relative': 0})
    illuminationTolerance = rasterTolerance(illumination)
    illuminationTolerance['absolute'] += sun * visibilityTolerance['absolute']
    illuminationTolerance['profile'] = 'R + propagated D disk error'
    return {**geometry, 'name': name, 'horizon': horizon.ravel(),
            'settings': {'azimuthDegrees': azimuth, 'altitudeDegrees': altitude,
                         'angularRadiusDegrees': radius, 'sunIntensity': sun,
                         'ambientIntensity': ambient},
            'expected': {'sunVisibility': visibility, 'illumination': illumination},
            'tolerances': {'sunVisibility': visibilityTolerance,
                           'illumination': illuminationTolerance},
            'reference': 'composite: numpy.interp(period=360), scipy.stats.semicircular.cdf, pvlib.irradiance.get_total_irradiance'}


def makeSunTable():
    times = pandas.date_range('2024-06-20T00:07:30Z', periods=96, freq='15min')
    solar = pvlib.solarposition.spa_python(times, 45.98, 7.75, altitude=0,
                                         pressure=0, delta_t=67, how='numpy')
    altitudes = solar.elevation.to_numpy()
    airMass = pvlib.atmosphere.get_relative_airmass(90 - altitudes,
                                                  model='kastenyoung1989')
    directNormal = numpy.where(altitudes > 0, 1353 * 0.7 ** (airMass ** 0.678), 0)
    return roundFloat32(numpy.column_stack((solar.azimuth, altitudes,
                                           numpy.full(96, 0.25), directNormal)))


def makeRealGeometry(manifest):
    with rasterio.open(DATA / 'dem.png') as dataset:
        channels = dataset.read().astype(numpy.float64)
    elevation = roundFloat32(channels[0] * 256 + channels[1] + channels[2] / 256 - 32768)
    cellSize = float(roundFloat32(manifest['raster']['cellSizeGroundM']))
    southGradient, eastGradient = numpy.gradient(elevation, cellSize, cellSize)
    source = {
        'kind': 'showcase DEM geometry with controlled synthetic horizon',
        'path': str((DATA / 'dem.png').relative_to(ROOT)),
        'sha256': hashlib.sha256((DATA / 'dem.png').read_bytes()).hexdigest(),
        'manifestSha256': hashlib.sha256((DATA / 'manifest.json').read_bytes()).hexdigest(),
        'crs': 'uniform local ground metres; north-up pixel centres', 'cellSize': cellSize,
        'fullDecodedShape': list(elevation.shape), 'coreSize': [64, 64], 'halo': 1,
        'seamCheck': 'both tile gradient cores equal full-DEM gradient, including shared seam',
    }
    for column in [960, 1024]:
        row = 960
        core = numpy.s_[row:row + 64, column:column + 64]
        tile = elevation[row - 1:row + 65, column - 1:column + 65]
        tileSouth, tileEast = numpy.gradient(tile, cellSize, cellSize)
        numpy.testing.assert_array_equal(tileSouth[1:-1, 1:-1], southGradient[core])
        numpy.testing.assert_array_equal(tileEast[1:-1, 1:-1], eastGradient[core])
        slope = roundFloat32(numpy.degrees(numpy.arctan(numpy.hypot(
            eastGradient[core], southGradient[core])))).ravel()
        aspect = roundFloat32(numpy.degrees(numpy.arctan2(
            -eastGradient[core], southGradient[core])) % 360).ravel()
        aspect[slope == 0] = -1
        skyView = roundFloat32((1 + numpy.cos(numpy.radians(slope))) / 2)
        yield {'width': 64, 'height': 64, 'directionCount': 4,
               'slope': slope, 'aspect': aspect, 'skyViewFactor': skyView,
               'horizon': numpy.zeros((64 * 64, 4)),
               'source': {**source, 'coreOrigin': [row, column],
                          'elevationFloat32Sha256': hashlib.sha256(
                              elevation[core].astype('<f4').tobytes()).hexdigest()}}


def makeSyntheticGeometry():
    pixelCount, directionCount = 17 * 9, 8
    row = numpy.arange(pixelCount)
    horizon = roundFloat32(20 + 0.65 * numpy.sin(row[:, None] * 0.37)
                          + 0.4 * numpy.cos(numpy.arange(directionCount)[None, :] * math.pi / 4))
    # Constant horizons make hard equality, adjacent f32 values, and disk tangencies unambiguous.
    radius = float(roundFloat32(0.2666))
    horizon[:7, :] = roundFloat32([20, numpy.nextafter(numpy.float32(20), numpy.float32(0)),
                                  numpy.nextafter(numpy.float32(20), numpy.float32(30)),
                                  20 - radius, 20 + radius, -1, 45])[:, None]
    slope = roundFloat32(row % 80)
    aspect = roundFloat32(row * 137 % 360)
    skyView = roundFloat32(0.25 + (row % 31) / 40)
    horizon[7, :] = numpy.nan
    slope[8] = numpy.nan
    aspect[9] = numpy.nan
    skyView[10] = numpy.nan
    aspect[11] = -1
    return {'width': 17, 'height': 9, 'directionCount': directionCount,
            'slope': slope, 'aspect': aspect, 'skyViewFactor': skyView, 'horizon': horizon,
            'source': {'kind': 'synthetic deterministic sectors, hard ties, penumbra, independent validity holes'}}


def makePositionCases(manifest):
    sites = [[entry['lon'], entry['lat']] for entry in manifest['properties']['observers'].values()]
    sites += [[longitude, latitude] for latitude in range(-75, 76, 15)
              for longitude in range(-180, 180, 15)]
    sites += [[180, 30], [-180, 30], [20, 89.5], [-60, -89.5],
              [10, numpy.nan], [10, 91], [numpy.inf, 10]]
    positions = roundFloat32(sites)
    valid = numpy.isfinite(positions).all(axis=1) & (numpy.abs(positions[:, 1]) <= 90)
    timestamps = ['2000-01-01T12:00:00Z', '2003-03-20T00:01:00Z',
                  '2024-03-20T06:15:00Z', '2024-06-20T12:00:00Z',
                  '2024-09-22T18:00:00Z', '2024-12-21T23:59:30Z',
                  '2026-10-04T06:15:00Z', '2041-08-09T17:42:11Z']
    cases = []
    for timestamp in timestamps:
        instant = pandas.Timestamp(timestamp)
        days = (instant - pandas.Timestamp('2000-01-01T12:00:00Z')).total_seconds() / 86400
        whole = math.floor(days)
        fraction = float(roundFloat32(days - whole))
        represented = pandas.Timestamp('2000-01-01T12:00:00Z') + pandas.Timedelta(
            seconds=(whole + fraction) * 86400)
        expected = {key: numpy.full(len(positions), numpy.nan) for key in
                    ['azimuth', 'altitude', 'zenith']}
        daylight = numpy.zeros(len(positions), dtype=numpy.uint32)
        for index in numpy.flatnonzero(valid):
            longitude, latitude = positions[index]
            solar = pvlib.solarposition.spa_python(pandas.DatetimeIndex([represented]),
                latitude, longitude, altitude=0, pressure=0, delta_t=67, how='numpy').iloc[0]
            expected['azimuth'][index] = solar.azimuth
            expected['altitude'][index] = solar.elevation
            expected['zenith'][index] = solar.zenith
            daylight[index] = solar.elevation > float(roundFloat32(-0.833))
        # All sites avoid zenith/nadir singularities; do not silently remove any output rows.
        assert numpy.nanmax(numpy.abs(expected['altitude'])) < 89
        cases.append({'name': timestamp, 'timestamp': instant.value / 1e6,
                      'representedTimestamp': represented.isoformat(),
                      'parameters': [whole, fraction, float(roundFloat32(-0.833)), 0],
                      'positions': positions.ravel(), 'expected': {**expected, 'daylight': daylight},
                      'source': {'kind': 'all three Alps observer sites + deterministic global grid + edge cases',
                                 'manifestSha256': hashlib.sha256((DATA / 'manifest.json').read_bytes()).hexdigest()},
                      'tolerance': {'profile': 'D', 'absolute': 1e-4, 'relative': 0},
                      'reference': 'pvlib.solarposition.spa_python (pressure=0, altitude=0, delta_t=67)'})
    return cases


def main():
    for coordinate in numpy.linspace(-1, 1, 21):
        integral = quad(lambda vertical: 2 * math.sqrt(max(0, 1 - vertical ** 2)) / math.pi,
                        -1, coordinate, epsabs=1e-12)[0]
        numpy.testing.assert_allclose(semicircular.cdf(coordinate), integral, atol=1e-10, rtol=0)
    manifest = json.loads((DATA / 'manifest.json').read_text())
    table = makeSunTable()
    for index, geometry in enumerate(makeRealGeometry(manifest)):
        writeFixture(f'irradiance-alps-{index}', [makeIrradianceCase(
            f'alps-{index}-isotropic-day', geometry, table)])
        # Real slopes/aspects, synthetic direction-dependent skyline, explicitly not DEM shadows.
        geometry['horizon'] = roundFloat32(20 + numpy.sin(numpy.arange(4096)[:, None] * 0.17)
                                          + numpy.array([0, 0.5, 0, -0.5])[None, :])
        writeFixture(f'shadow-alps-{index}', [makeMaskCase(
            f'alps-{index}-controlled-horizon', geometry, 337.5, 20, 0.2666)])
    geometry = makeSyntheticGeometry()
    samples = [[0, -5, 0.25, 0], [22.5, 20, 0.5, 700], [337.5, 20, 1, 900],
               [180, 65, 0.125, 1000], [90, 0, 0.5, 0], [270, -0.1, 0.25, 0],
               [90, 45, 1 / 12, 800]]
    writeFixture('irradiance-synthetic', [
        makeIrradianceCase('synthetic-penumbra-duration-nodata', geometry, samples, radius=0.2666),
        makeIrradianceCase('synthetic-one-hour-instantaneous', geometry, [[337.5, 20, 1, 900]], radius=0),
        makeIrradianceCase('synthetic-empty-table', geometry, samples, radius=0.2666, sampleCount=0)])
    settings = [(0, 20, 0), (22.5, 20, 0), (337.5, 20, 0.2666),
                (-22.5, 20, 0.2666), (697.5, 20, 0.2666), (360, 20, 0.2666),
                (0, 20, 0.25), (0, -0.25, 0.25), (0, -0.1, 0.25), (180, 65, 0.2666)]
    writeFixture('shadow-synthetic', [makeMaskCase(f'synthetic-mask-{index}', geometry, *setting)
                                       for index, setting in enumerate(settings)])
    writeFixture('position', makePositionCases(manifest))
    altitudes = roundFloat32([-1, 0, 0.1, 1, 5, 10, 30, 60, 89, 90])
    airMass = pvlib.atmosphere.get_relative_airmass(90 - altitudes, model='kastenyoung1989')
    writeFixture('meinel', [{'name': 'published-meinel-kasten-young', 'altitudes': altitudes,
                            'directNormal': numpy.where(altitudes > 0, 1353 * 0.7 ** (airMass ** 0.678), 0)}])


if __name__ == '__main__':
    main()
