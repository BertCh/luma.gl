# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Composite reference: NOAA published daily spreadsheet, columns G:AH.

Independent float64 NumPy evaluation of NOAA_Solar_Calculations_day.ods,
https://gml.noaa.gov/grad/solcalc/NOAA_Solar_Calculations_day.ods . Coefficients
and acos azimuth branches come from the publication, not repository code.
UTC, east-positive longitude, north-positive latitude, geometric sun center,
no atmospheric correction, no observer parallax. The spreadsheet's 20.772
refraction coefficient differs from current NOAA main.js (20.774); refraction
is deliberately disabled here. Inputs are the exact transported float32 values.

NREL SPA (pvlib, pressure=0, height=0, delta_t=67) remains a SEPARATE model
accuracy measurement. NOAA publishes sunrise/set accuracy of 60 s at |lat|<=72
and 600 s outside; it does not publish a universal angular accuracy. We test
those timing bounds directly using brentq crossings of the SAME -0.833 degree
geometric threshold, not arbitrary angular tolerances. All angle differences
are reported without calling model disagreement float32 error.
"""
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy
import pandas
import pvlib
import scipy
from scipy.optimize import brentq

OUTPUT = Path(__file__).parent / 'fixtures'
EPOCH = pandas.Timestamp('2000-01-01T12:00:00Z')


def computeSolarPositionReference(days, longitude, latitude):
    century = days / 36525
    meanLongitude = numpy.remainder(numpy.polynomial.polynomial.polyval(
        century, [280.46646, 36000.76983, 0.0003032]), 360)
    meanAnomaly = numpy.radians(numpy.polynomial.polynomial.polyval(
        century, [357.52911, 35999.05029, -0.0001537]))
    eccentricity = numpy.polynomial.polynomial.polyval(
        century, [0.016708634, -0.000042037, -0.0000001267])
    center = numpy.dot(numpy.sin(meanAnomaly * numpy.arange(1, 4)), [
        1.914602 - 0.004817 * century - 0.000014 * century**2,
        0.019993 - 0.000101 * century, 0.000289])
    omega = numpy.radians(125.04 - 1934.136 * century)
    apparentLongitude = numpy.radians(meanLongitude + center - 0.00569 - 0.00478 * numpy.sin(omega))
    obliquity = numpy.radians(23 + 26 / 60 + numpy.polynomial.polynomial.polyval(
        century, [21.448, -46.815, -0.00059, 0.001813]) / 3600 + 0.00256 * numpy.cos(omega))
    declination = numpy.arcsin(numpy.sin(obliquity) * numpy.sin(apparentLongitude))
    tangentSquared = numpy.tan(obliquity / 2)**2
    longitudeRadians = numpy.radians(meanLongitude)
    equationOfTime = 4 * numpy.degrees(numpy.dot([
        tangentSquared, -2 * eccentricity, 4 * eccentricity * tangentSquared,
        -0.5 * tangentSquared**2, -1.25 * eccentricity**2], [
        numpy.sin(2 * longitudeRadians), numpy.sin(meanAnomaly),
        numpy.sin(meanAnomaly) * numpy.cos(2 * longitudeRadians),
        numpy.sin(4 * longitudeRadians), numpy.sin(2 * meanAnomaly)]))
    solarMinutes = numpy.remainder(numpy.remainder(days + 0.5, 1) * 1440 + equationOfTime + 4 * longitude, 1440)
    hourAngle = solarMinutes / 4 - 180
    latitudeRadians = numpy.radians(latitude)
    zenith = numpy.arccos(numpy.clip(numpy.sin(latitudeRadians) * numpy.sin(declination) +
        numpy.cos(latitudeRadians) * numpy.cos(declination) * numpy.cos(numpy.radians(hourAngle)), -1, 1))
    azimuthArgument = ((numpy.sin(latitudeRadians) * numpy.cos(zenith) - numpy.sin(declination)) /
                      (numpy.cos(latitudeRadians) * numpy.sin(zenith)))
    azimuthArc = numpy.degrees(numpy.arccos(numpy.clip(azimuthArgument, -1, 1)))
    azimuth = numpy.remainder(numpy.where(hourAngle > 0, azimuthArc + 180, 540 - azimuthArc), 360)
    return azimuth, 90 - numpy.degrees(zenith)


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


def main():
    source = OUTPUT / 'parity-wp5-solar-position.json'
    original = json.loads(source.read_text())
    cases = []
    for fixture in original['cases']:
        sites = numpy.array(fixture['positions'], dtype=float).reshape(-1, 2)
        valid = numpy.isfinite(sites).all(axis=1) & (numpy.abs(sites[:, 1]) <= 90)
        days = sum(fixture['parameters'][:2])
        azimuth = numpy.full(len(sites), numpy.nan)
        altitude = azimuth.copy()
        azimuth[valid], altitude[valid] = computeSolarPositionReference(days, sites[valid, 0], sites[valid, 1])
        spaReference = fixture['expected']
        daylight = (altitude > fixture['parameters'][2]).astype(int)
        flips = numpy.flatnonzero(daylight != spaReference['daylight'])
        for index in flips:
            print('DAYLIGHT_MODEL_FLIP', fixture['name'], index, sites[index].tolist(),
                  'NOAA', altitude[index], 'SPA', spaReference['altitude'][index], 'threshold', fixture['parameters'][2])
        # Rounding analysis, NOT a model-accuracy allowance: whole-day products
        # have up to 2 ulps error (coefficient + multiply + add), reduced-angle
        # addition another ulp. 4 ulps covers both orbital angles propagated
        # through declination and equation of time; 1e-4 is the base D trig budget.
        # Azimuth/acos altitude are conditioned by 1/cos(altitude); no rows dropped.
        orbitalMagnitude = max(abs(280.46646 + days * 0.98564736), abs(357.52911 + days * 0.98560028))
        orbitalError = 4 * float(numpy.spacing(numpy.float32(orbitalMagnitude))) + 1e-4
        bounds = orbitalError / numpy.cos(numpy.radians(altitude))
        cases.append({**fixture, 'spaReference': spaReference, 'expected': {
            'azimuth': azimuth, 'altitude': altitude, 'zenith': 90 - altitude, 'daylight': daylight},
            'angleBounds': bounds, 'directionBound': orbitalError,
            'reference': 'Composite reference: NOAA published daily spreadsheet G:AH (Meeus)',
            'tolerance': {'profile': 'D (conditioned f32 orbital rounding)', 'absolute': orbitalError, 'relative': 0}})
    # Directly check the published sunrise/set timing bounds, including polar
    # latitude, equinox and solstice, without conflating degrees and seconds.
    events = []
    for date, longitude, latitude in [('2003-03-20', -90, 75), ('2024-06-20', 7.75, 45.98),
                                      ('2024-03-20', 0, 0), ('2041-08-09', -105, 40)]:
        midnight = pandas.Timestamp(date, tz='UTC')
        baseDays = (midnight - EPOCH).total_seconds() / 86400
        def noaa(seconds):
            return float(computeSolarPositionReference(baseDays + seconds / 86400, longitude, latitude)[1]) + 0.833
        def spaElevation(seconds):
            instant = midnight + pandas.Timedelta(seconds=seconds)
            return float(pvlib.solarposition.spa_python(pandas.DatetimeIndex([instant]), latitude,
                         longitude, altitude=0, pressure=0, delta_t=67).elevation.iloc[0]) + 0.833
        for start in range(0, 86400, 1800):
            if noaa(start) * noaa(start + 1800) >= 0:
                continue
            noaaSeconds = brentq(noaa, start, start + 1800, xtol=1e-5)
            spaSeconds = brentq(spaElevation, start - 600, start + 2400, xtol=1e-5)
            boundSeconds = 60 if abs(latitude) <= 72 else 600
            assert abs(noaaSeconds - spaSeconds) < boundSeconds
            events.append(dict(name=f'{date}/{longitude}/{latitude}/{start}', noaaSeconds=noaaSeconds,
                               spaSeconds=spaSeconds, boundSeconds=boundSeconds))
    payload = {'versions': {'python': platform.python_version(), 'numpy': numpy.__version__,
                             'pandas': pandas.__version__, 'pvlib': pvlib.__version__, 'scipy': scipy.__version__},
               'generatorSha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'sourceFixtureSha256': hashlib.sha256(source.read_bytes()).hexdigest(),
               'publication': {'url': 'https://gml.noaa.gov/grad/solcalc/NOAA_Solar_Calculations_day.ods', 'sha256': 'd013d001a7620645c3d1dd23b8d876f76d6fbae577df295330cb494a0c6cd3b5'},
               'conventions': __doc__, 'cases': cases, 'events': events}
    output = OUTPUT / 'parity-wp5-solar-noaa-position.json'
    output.write_text(json.dumps(encode(payload), separators=(',', ':'), allow_nan=False) + '\n')
    assert output.stat().st_size < 1_500_000
    print(output.name, output.stat().st_size, 'bytes', len(cases), 'cases', len(events), 'events')


if __name__ == '__main__':
    main()
