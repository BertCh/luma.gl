# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""PROJ Geod inverse/direct parity on full showcase track endpoints and routes.

Independent references: pyproj.Geod.inv/fwd, explicit sphere R=6371008.8 and
WGS84 separately. Inputs are f32 lon/lat degrees; direct bearings/distances are
also rounded BEFORE fwd. Output back azimuth is converted to forward arrival
bearing. Longitudes are unwrapped to the branch within 180 degrees of origin.
Midpoints are SPHERICAL in both GPU models, per public API. Coincident inverse
bearings have no geodetic meaning: pin API's 0/180 sphere and 0/0 WGS84 convention.
Exact antipodal bearings/midpoints have no unique answer: only distance and
WGS84 failure flag are compared there. The near-antipodal WGS84 test expects the
DOCUMENTED sphere fallback with converged=0, never labels it ellipsoid parity.

All 30000 ECCO tracks supply first/last endpoints (not every intermediate segment).
All 18930 OpenFlights routes supply origins and model-specific inverse distances/
bearings as direct inputs. Separate <=2048-row fixtures retain every output and
source index; no spatial subsampling. Each file <=1.5 MB. No local CPU oracle.

D profile: distance atol .001m + rtol 2e-6. Coordinate atol 8 ulps at angular
coordinate scale (degrees), rtol 0; angular error 1e-4 degrees away from antipodes.
Near-antipodal bearing/position condition factor 1/abs(sin(central angle)):
8*f32 epsilon radians propagated through atan2/spherical interpolation.
For the unit-sphere tangent components v=(east,north), ||v||=|sin(sigma)|.
The azimuth derivative is (-north,east)/||v||^2, so a component-vector error
||delta v|| <= 8*epsilon32 gives |delta bearing| <= 8*epsilon32/|sin(sigma)|
radians to first order. At the explicit near-antipode this is .008683 degrees;
the fixture adds the ordinary .0001 degree budget. The pole case uses the
colatitude cosine identity, which avoids loss of relative precision in cos(phi).
This finite conditioning bound does NOT cover atan2(0,0): the GPU must explicitly
implement the coincident convention (WGSL https://www.w3.org/TR/WGSL/#atan2
allows unbounded error at the origin). Nonfinite bearings always fail.
Direct sphere arrival bearing is reconstructed from its rounded destination;
its conditioning budget is 2*coordinate quantization radians / sin(distance/R),
converted to degrees, added to 1e-4. This accounts for subtracting two nearby
stored positions, and does not change the position bound.
"""
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import pyproj

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[3] / 'examples/deck/spatial-analysis-showcase/public/data'
RADIUS = 6371008.8
SPHERE = pyproj.Geod(a=RADIUS, f=0)
MODELS = {'sphere': SPHERE, 'wgs84': pyproj.Geod(ellps='WGS84')}
VERSIONS = dict(python=platform.python_version(), numpy=np.__version__, pyproj=pyproj.__version__, proj=pyproj.proj_version_str)


def wrap(values):
    return (np.asarray(values) + 180) % 360 - 180


def continuous(longitudes, origins):
    delta = wrap(np.asarray(longitudes) - origins)
    # The signed 180-degree branch is ambiguous; the due-north polar direct case
    # uses +180, matching the API's continuous origin + atan2 longitude.
    return origins + np.where(delta == -180, 180, delta)


def inverse(geod, origins, targets):
    initial, backward, distance = geod.inv(origins[:, 0], origins[:, 1], targets[:, 0], targets[:, 1])
    return np.asarray(initial), wrap(np.asarray(backward) + 180), np.asarray(distance)


def position_bound(positions):
    scale = max(1., float(np.max(np.abs(positions))))
    return max(1e-3 / (RADIUS*np.pi/180), 8*float(np.spacing(np.float32(scale))))


def make_pairs(name, origins, targets, fallback=None, singular=None):
    origins = np.asarray(origins, dtype=np.float32).astype(float)
    targets = np.asarray(targets, dtype=np.float32).astype(float)
    count = len(origins)
    fallback = np.zeros(count, dtype=bool) if fallback is None else np.asarray(fallback, dtype=bool)
    singular = np.zeros(count, dtype=bool) if singular is None else np.asarray(singular, dtype=bool)
    sphere_initial, sphere_final, sphere_distance = inverse(SPHERE, origins, targets)
    middle_longitude, middle_latitude, _ = SPHERE.fwd(origins[:, 0], origins[:, 1], sphere_initial, sphere_distance/2)
    midpoint = np.column_stack([continuous(middle_longitude, origins[:, 0]), middle_latitude])
    coincident = sphere_distance == 0
    midpoint[coincident] = origins[coincident]
    sine = np.abs(np.sin(sphere_distance / RADIUS))
    conditioned = (sphere_distance/RADIUS > 3.) & ~singular
    bearing_bounds = np.full(count, 1e-4)
    bearing_bounds[conditioned] += np.degrees(8*np.finfo(np.float32).eps / sine[conditioned])
    position_bounds = np.full(count, position_bound(np.concatenate([origins, targets, midpoint])))
    position_bounds[conditioned] += np.degrees(8*np.finfo(np.float32).eps / sine[conditioned])
    expected = {}
    for model, geod in MODELS.items():
        initial, final, distance = inverse(geod, origins, targets)
        if model == 'wgs84':
            initial[fallback], final[fallback], distance[fallback] = sphere_initial[fallback], sphere_final[fallback], sphere_distance[fallback]
        initial[coincident] = 0
        final[coincident] = 180 if model == 'sphere' else 0
        expected[model] = dict(distances=distance.tolist(), initialBearings=initial.tolist(), finalBearings=final.tolist(),
                               midpoints=midpoint.reshape(-1).tolist(), converged=(~fallback).astype(int).tolist())
    return dict(name=name, origins=origins.reshape(-1).tolist(), targets=targets.reshape(-1).tolist(), expected=expected,
                singular=singular.astype(int).tolist(), bearingBounds=bearing_bounds.tolist(), positionBounds=position_bounds.tolist())


def make_destination(name, origins, targets=None, bearings=None, distances=None):
    origins = np.asarray(origins, dtype=np.float32).astype(float)
    count = len(origins)
    cases = []
    for model, geod in MODELS.items():
        if targets is not None:
            initial, _, distance = inverse(geod, origins, targets)
        else:
            initial, distance = bearings, distances
        initial = np.asarray(initial, dtype=np.float32).astype(float)
        distance = np.asarray(distance, dtype=np.float32).astype(float)
        longitude, latitude, backward = geod.fwd(origins[:, 0], origins[:, 1], initial, distance)
        destination = np.column_stack([continuous(longitude, origins[:, 0]), latitude])
        final = wrap(np.asarray(backward)+180)
        # Zero distance has no inverse bearing on a sphere; its API returns 180.
        if model == 'sphere': final[distance == 0] = 180
        position_absolute = position_bound(np.concatenate([origins, destination]))
        bearing_bounds = np.full(count, 1e-4)
        if model == 'sphere':
            # Output rounding is the only extra conditioning budget; radius cancels.
            nonzero = distance != 0
            row_scale = np.maximum(1., np.maximum(np.abs(origins), np.abs(destination)).max(axis=1))
            quantization = np.radians(np.spacing(row_scale.astype(np.float32)).astype(float))
            bearing_bounds[nonzero] += np.degrees(2*quantization[nonzero] / np.maximum(np.abs(np.sin(distance[nonzero]/RADIUS)), 1e-15))
        cases.append(dict(name=name+'-'+model, model=model, origins=origins.reshape(-1).tolist(), bearings=initial.tolist(), distances=distance.tolist(),
                          expected=dict(destinations=destination.reshape(-1).tolist(), finalBearings=final.tolist(), converged=[1]*count),
                          positionAbsolute=position_absolute, bearingBounds=bearing_bounds.tolist()))
    return cases


def write(topic, cases, source):
    fixture = dict(generator=Path(__file__).name, reference='pyproj.Geod.inv/fwd (PROJ)', versions=VERSIONS, source=source,
                   conventions=dict(sphereRadius=RADIUS, ellipsoid='WGS84', inputPrecision='float32 before reference', units='meters; angles degrees clockwise from north',
                                    midpoint='sphere in both models', longitude='continuous with origin', fallback='only explicitly labelled antipodal inverse rows use sphere'),
                   tolerance=dict(profile='D', distanceAbsolute=.001, distanceRelative=2e-6, angles='1e-4 degrees; conditioned rows carry analytic bounds', position='8 angular-coordinate ulps'), cases=cases)
    output = HERE / f'fixtures/parity-wp2-{topic}.json'
    output.write_text(json.dumps(fixture, separators=(',', ':'), allow_nan=False)+'\n')
    assert output.stat().st_size <= 1500000, (output, output.stat().st_size)
    print(output.name, output.stat().st_size)


def source_metadata(dataset, filenames):
    return dict(dataset=dataset, hashes={filename: hashlib.sha256((DATA/dataset/filename).read_bytes()).hexdigest() for filename in filenames})

source = DATA/'poopdeck-ecco-currents'
offsets = np.fromfile(source/'pathOffsets.bin', '<u4')
vertices = np.fromfile(source/'vertices.bin', '<f4').reshape(-1, 2)
assert np.all(np.diff(offsets) > 0)
origins, targets = vertices[offsets[:-1]], vertices[offsets[1:]-1]
metadata = source_metadata('poopdeck-ecco-currents', ['manifest.json', 'pathOffsets.bin', 'vertices.bin'])
for start in range(0, len(origins), 2048):
    stop = min(start+2048, len(origins))
    write(f'geodesic-pairs-ecco-{start//2048:02}', [make_pairs(f'ecco-tracks-{start}-{stop}', origins[start:stop], targets[start:stop])],
          dict(**metadata, fullTrackCount=len(origins), sourceRange=[start,stop], selection='every track first/last endpoint; contiguous batches'))

edge_origins = [[179.5,20],[-179.5,-20],[0,89],[20,0],[10,10],[0,0],[0,0],[2.35,48.85]]
edge_targets = [[-179.5,21],[179.5,-21],[90,89],[21,0],[10,10],[179.7,.2],[180,0],[2.350013,48.850001]]
write('geodesic-pairs-edges', [make_pairs('date-line-pole-zero-near-and-exact-antipode-meter',edge_origins,edge_targets,
                                       [0,0,0,0,0,1,1,0],[0,0,0,0,0,0,1,0])], dict(dataset='synthetic; all rows explicit'))

source = DATA/'openflights'
positions = np.fromfile(source/'position.bin','<f4').reshape(-1,2)
origin_ids = np.fromfile(source/'origin.bin','<u4')
target_ids = np.fromfile(source/'destination.bin','<u4')
metadata = source_metadata('openflights',['manifest.json','position.bin','origin.bin','destination.bin'])
for start in range(0,len(origin_ids),2048):
    stop=min(start+2048,len(origin_ids))
    write(f'geodesic-destination-flights-{start//2048:02}',make_destination(f'flights-{start}-{stop}',positions[origin_ids[start:stop]],positions[target_ids[start:stop]]),
          dict(**metadata,fullRouteCount=len(origin_ids),sourceRange=[start,stop],selection='every route; contiguous batches'))
write('geodesic-destination-edges',make_destination('zero-meter-date-line-polar',[[10,10],[2.35,48.85],[179.5,20],[0,89]],
                                                  bearings=[35,45,90,0],distances=[0,1,500000,500000]),dict(dataset='synthetic; all rows explicit'))
