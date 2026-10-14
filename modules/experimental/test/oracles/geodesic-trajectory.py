# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Reference great-circle distances, bearings and interpolated points from pyproj, and the largest
searchable distance of the encounter lattice.

Writes fixtures/geodesic-trajectory.json next to this script, consumed (through
test/gpu-spatial-analysis/trajectory-analysis/geodesic-trajectory-oracle.ts) by
movement-spatial-context.node.spec.ts and gpu-trajectory-geodesic.spec.ts. Run from anywhere with
    <venv>/bin/python -I modules/experimental/test/oracles/geodesic-trajectory.py
in a venv with numpy and pyproj (versions are recorded in the fixture).

The sphere is `pyproj.Geod(a=6371008.8, f=0)`, the GPU_EARTH_MEAN_RADIUS_METERS sphere.
- `cases`: for each pair of points, `Geod.inv` distance (m) and forward azimuth (degrees clockwise
  from north), and `Geod.fwd` points at fractions 0.25, 0.5 and 0.75 of the distance, with the
  longitude made continuous with the start point (unwrapped, as the interpolation helper does).
  The over-the-pole pair is not part of the specs' cases and is left out.
- `encounterReach`: the encounter lattice claims the largest searchable angle
  asin(sin(cell) * cos(L)) for a cell of `cellDegrees` and a maximum |latitude| L. Walking that
  distance in 7201 azimuths from latitude +-L (`Geod.fwd`) gives the observed longitude and
  latitude reach in degrees: it must stay within one cell and be nearly tight. `maximumDistance`
  includes the `1 - 1e-6` safety scale of the operator.

Tolerances (stored in `tolerances`): the binary64 spherical formulas and pyproj's geodesic
algorithm on a sphere agree to rounding, about 1e-12 relative; the specs bound distances at 1e-6
relative and bearings and interpolated points at 1e-9 degrees, which leaves room for the
different evaluation orders without hiding a formula error (a wrong radius moves the distance by
1e-6 relative per 6 m).
"""

import json
import math
import os
import platform

import numpy as np
import pyproj

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'fixtures', 'geodesic-trajectory.json')
RADIUS = 6371008.8
FRACTIONS = [0.25, 0.5, 0.75]
PAIRS = [
    ('equator-east', (0.0, 0.0), (10.0, 0.0)),
    ('sf-la', (-122.4194, 37.7749), (-118.2437, 34.0522)),
    ('antimeridian', (179.5, 10.0), (-179.5, 12.0)),
    ('short-step', (8.5417, 47.3769), (8.5427, 47.3774)),
]
REACH_CASES = [(2.0, 60.0), (5.0, 0.0), (1.0, 80.0), (10.0, 45.0)]
AZIMUTH_COUNT = 7201
SAFETY_SCALE = 1 - 1e-6

geod = pyproj.Geod(a=RADIUS, f=0)

cases = []
for name, start, end in PAIRS:
    azimuth, _, distance = geod.inv(start[0], start[1], end[0], end[1])
    points = []
    for fraction in FRACTIONS:
        longitude, latitude, _ = geod.fwd(start[0], start[1], azimuth, distance * fraction)
        longitude = start[0] + ((longitude - start[0] + 180.0) % 360.0 - 180.0)
        points.append({'fraction': fraction, 'lon': longitude, 'lat': latitude})
    cases.append(
        {'name': name, 'a': list(start), 'b': list(end), 'distance': distance, 'bearingDegrees': azimuth, 'points': points}
    )

encounter_reach = []
for cell, maximum_latitude in REACH_CASES:
    claimed_angle = math.asin(math.sin(math.radians(cell)) * math.cos(math.radians(maximum_latitude)))
    claimed = RADIUS * claimed_angle
    azimuths = np.linspace(-180, 180, AZIMUTH_COUNT)
    longitude_reach = 0.0
    latitude_reach = 0.0
    for start_latitude in (maximum_latitude, -maximum_latitude):
        longitudes, latitudes, _ = geod.fwd(
            np.zeros_like(azimuths), np.full_like(azimuths, start_latitude), azimuths, np.full_like(azimuths, claimed)
        )
        longitude_reach = max(longitude_reach, float(np.max(np.abs((longitudes + 180) % 360 - 180))))
        latitude_reach = max(latitude_reach, float(np.max(np.abs(latitudes - start_latitude))))
    encounter_reach.append(
        {
            'cellDegrees': cell,
            'maximumLatitude': maximum_latitude,
            'unscaledMaximumDistance': claimed,
            'maximumDistance': claimed * SAFETY_SCALE,
            'observedLongitudeReachDegrees': longitude_reach,
            'observedLatitudeReachDegrees': latitude_reach,
        }
    )

fixture = {
    'generator': 'modules/experimental/test/oracles/geodesic-trajectory.py',
    'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pyproj': pyproj.__version__},
    'tolerances': {'distanceRelative': 1e-6, 'bearingDegrees': 1e-9, 'pointDegrees': 1e-9},
    'reference': {'radius': RADIUS, 'cases': cases, 'encounterReach': encounter_reach},
}
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, 'w') as handle:
    json.dump(fixture, handle, separators=(',', ':'))
    handle.write('\n')
print('wrote', OUTPUT)
