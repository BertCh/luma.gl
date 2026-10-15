# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Direct PROJ projection parity, full 876 Chicago facilities and edge frames.

pyproj.Transformer.from_crs(always_xy=True): spherical AEQD R=6371008.8,
ellipsoidal WGS84 LAEA; matching createGPULocalFrame model conventions.
Every real source row retained, no snap/resampling. Float32 and binary64 inputs
are separately rounded BEFORE PROJ. Boundary rows are inclusive; outside bounds
have validity=0 and output (0,0). Synthetic dateline uses continuous longitude,
and polar-nearby frames stay away from the antipodal projection singularity.
D budget: patch compiler's explicit 0.01m approximation budget + 8 output-scale
float32 ulps, rtol 2e-6. Binary64 inverse of f32-rounded PROJ outputs is included
for a supplementary CPU frame inverse check, not claimed as GPU inverse support.
"""
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import pyproj

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[3] / 'examples/deck/spatial-analysis-showcase/public/data/chicago-facilities'
positions = np.fromfile(DATA/'position.bin', '<f4').reshape(-1, 2).astype(float)
cases = []
scenes = [('all-chicago-facilities', [-87.63,41.88], [-87.85,41.65,-87.52,42.03], positions),
          ('continuous-date-line', [180,20], [179,19,181,21], np.array([[180,20],[179,19],[181,21],[179.999999,20.000001],[178.99,20],[181.01,20],[180,18.99],[180,21.01]])),
          ('near-north-pole', [35,89], [34,88.5,36,89.5], np.array([[35,89],[34,88.5],[36,89.5],[35.0000001,89.0000001],[34.5,89.25],[36.01,89]]))]
for name, origin, bounds, values in scenes:
    for kind, projection in [('azimuthal-equidistant', '+proj=aeqd +R=6371008.8'), ('lambert-equal-area', '+proj=laea +ellps=WGS84')]:
        crs = f'{projection} +lon_0={origin[0]} +lat_0={origin[1]} +units=m'
        forward = pyproj.Transformer.from_crs(4326, crs, always_xy=True)
        backward = pyproj.Transformer.from_crs(crs, 4326, always_xy=True)
        for precision in ['float32', 'float64']:
            points = values.astype(precision).astype(float)
            valid = ((points >= bounds[:2]) & (points <= bounds[2:])).all(axis=1)
            projected = np.asarray(forward.transform(points[:,0], points[:,1])).T
            projected[~valid] = 0
            scale = max(1, np.abs(projected).max())
            rounded = projected.astype(np.float32).astype(float)
            inverse = np.asarray(backward.transform(rounded[:,0], rounded[:,1])).T
            cases.append(dict(name=f'{name}-{kind}-{precision}', origin=origin, bounds=bounds, kind=kind, precision=precision,
                              positions=points.reshape(-1).tolist(), expected=projected.reshape(-1).tolist(), validity=valid.astype(int).tolist(),
                              inverseOfRounded=inverse.reshape(-1).tolist(), absolute=.01 + 8*float(np.spacing(np.float32(scale))), relative=2e-6))
fixture = dict(generator=Path(__file__).name, reference='pyproj.Transformer.from_crs(always_xy=True).transform',
               versions=dict(python=platform.python_version(), numpy=np.__version__, pyproj=pyproj.__version__, proj=pyproj.proj_version_str),
               source=dict(dataset='chicago-facilities', count=len(positions), selection='all source rows', hashes={name:hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in ['manifest.json','position.bin']}),
               conventions=dict(aeqd='sphere R=6371008.8', laea='WGS84 ellipsoid', units='meters', longitude='continuous within bounds', patchTolerance=.01),
               tolerance=dict(profile='D', absolute='0.01m patch budget + 8 ulp32(output scale)', relative=2e-6, validity='exact'), cases=cases)
path = HERE/'fixtures/parity-wp2-local-frame.json'
path.write_text(json.dumps(fixture,separators=(',', ':'),allow_nan=False)+'\n')
assert path.stat().st_size <= 1500000
print(path.name, path.stat().st_size, len(cases))
