# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent particle trajectories: scipy.integrate.solve_ivp (DOP853) with
scipy.interpolate.RegularGridInterpolator; landing speeds cross-checked with
scipy.ndimage.map_coordinates(order=1, mode='nearest', prefilter=False).

Run <reference-venv>/bin/python -I this-file. No library source is imported.
Cell centres are origin+(index+0.5)*spacing, row zero is south, samples clamp to
outer centres inside the half-open extent. Inputs/parameters/seeds are rounded
once to f32, the reference integrates in f64 at exactly the GPU frame times.
NOAA GFS frame 14 is decoded in full, reversed north->south, then the fixed
96x96 window [south row 32:128, column 56:152] is used. Components m/s -> km/s;
coordinates are local km with 1 km spacing (a planar test, NOT geodesic wind
transport). 256 seeds retain full trajectories; no summaries replace arrays.
The short 32 second window and 2 second step exercise the small-step ODE limit;
fixed midpoint discretization is NOT claimed identical to adaptive DOP853.
Profile L is not enlarged. A rigid rotation also exercises a curved trajectory.
The small edge case has clamped outer-half-cell samples, an invalid NaN corner,
and an outside seed. Respawn is made independent of RNG by a zero-area spawn
rectangle; ages/generations and every trail slot are checked by the spec.
References: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html
"""
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
import scipy
import rasterio
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DATA = ROOT / 'examples/deck/spatial-analysis-showcase/public/data'


def encode(value):
    if isinstance(value, dict):
        return {key: encode(entry) for key, entry in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [encode(entry) for entry in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if math.isnan(value) else float(value)
    return value


def make_case(name, field, seeds, step, frames, extent=(0, 0, 1, 1)):
    field = np.asarray(field, dtype=np.float32).astype(float)
    seeds = np.asarray(seeds, dtype=np.float32).astype(float)
    extent = np.asarray(extent, dtype=np.float32).astype(float)
    step = float(np.float32(step))
    height, width, _ = field.shape
    centers = [extent[1] + (np.arange(height) + .5) * extent[3],
               extent[0] + (np.arange(width) + .5) * extent[2]]
    interpolate = RegularGridInterpolator(centers, field, method='linear')

    def sample(points):
        coordinates = np.asarray(points)[..., ::-1].copy()
        coordinates[..., 0] = np.clip(coordinates[..., 0], centers[0][0], centers[0][-1])
        coordinates[..., 1] = np.clip(coordinates[..., 1], centers[1][0], centers[1][-1])
        return interpolate(coordinates)

    times = np.arange(frames + 1) * step
    solution = solve_ivp(lambda time, positions: sample(positions.reshape(-1, 2)).ravel(),
                         (0, times[-1]), seeds.ravel(), t_eval=times,
                         method='DOP853', rtol=1e-12, atol=1e-13, max_step=step / 4)
    assert solution.success
    positions = solution.y.T.reshape(frames + 1, len(seeds), 2)
    assert np.all(positions >= extent[:2])
    assert np.all(positions < extent[:2] + extent[2:] * [width, height])
    grid = (positions - extent[:2]) / extent[2:] - .5
    velocities = np.stack([map_coordinates(field[:, :, component],
        [grid[..., 1].ravel(), grid[..., 0].ravel()], order=1,
        mode='nearest', prefilter=False).reshape(frames + 1, len(seeds))
        for component in range(2)], axis=-1)
    np.testing.assert_allclose(velocities, sample(positions), atol=1e-14)
    return dict(name=name, width=width, height=height, extent=extent,
                field=field.ravel(), seeds=seeds.ravel(), timeStep=step,
                times=times[1:], positions=positions[1:].reshape(frames, -1),
                speeds=np.linalg.norm(velocities[1:], axis=-1),
                previousPositions=positions[:-1].reshape(frames, -1),
                generations=np.zeros(len(seeds), dtype=int),
                ages=[np.full(len(seeds), frame + 1) for frame in range(frames)],
                spawnBounds=[1.25, 1.25, 1.25, 1.25], trailLength=3,
                reference='scipy.integrate.solve_ivp + scipy.interpolate.RegularGridInterpolator + scipy.ndimage.map_coordinates')


source = DATA / 'gfs-wind/wind10m_14.png'
with rasterio.open(source) as dataset:
    pixels = dataset.read()[:2].transpose(1, 2, 0)[::-1]
wind = (-36 + pixels.astype(float) / 255 * [81, 77]) / 1000
seed_columns, seed_rows = np.meshgrid(np.arange(4, 96, 6) + .25, np.arange(4, 96, 6) + .25)
cases = [make_case('gfs-helene-frame14-window96', wind[32:128, 56:152],
                   np.column_stack([seed_columns.ravel(), seed_rows.ravel()]), 2, 16)]
rows, columns = np.mgrid[:16, :16] + .5
rotation = np.stack([-.125 * (2 * rows - 16), .125 * (columns - 8)], axis=-1)
cases.append(make_case('rigid-rotation-anisotropic-cells', rotation,
    [[4, 6], [8, 4], [10, 9], [9, 10]], .015625, 32, (0, 0, 1, 2)))
# Valid clamped samples and deterministic respawn on invalid interpolation support.
field = np.full((4, 4, 2), [.25, 0], dtype=float)
field[2, 2] = np.nan
seeds = np.array([[.125, .125], [3.875, .125], [2.25, 2.25], [-.125, .125]])
edge = dict(name='clamped-edge-nodata-outside', width=4, height=4,
    extent=[0, 0, 1, 1], field=field.ravel(), seeds=seeds.ravel(), timeStep=.25,
    times=[.25], positions=[[.1875, .125, 3.9375, .125, 1.25, 1.25, 1.25, 1.25]],
    previousPositions=[[.125, .125, 3.875, .125, 1.25, 1.25, 1.25, 1.25]],
    speeds=[[.25] * 4], generations=[0, 0, 1, 1], ages=[[1, 1, 0, 0]],
    spawnBounds=[1.25, 1.25, 1.25, 1.25], trailLength=3,
    reference='scipy.ndimage.map_coordinates + explicit half-open extent/NaN-support contract')
# Independently sample the valid landing points (the two invalid seeds respawn).
landings = np.array(edge['positions']).reshape(-1, 2)
for component in range(2):
    sampled = map_coordinates(field[:, :, component], [landings[:, 1]-.5, landings[:, 0]-.5],
                              order=1, mode='nearest', prefilter=False)
    np.testing.assert_allclose(sampled, [.25 if component == 0 else 0] * 4)
cases.append(edge)
fixture = dict(versions=dict(python=platform.python_version(), numpy=np.__version__,
    scipy=scipy.__version__, rasterio=rasterio.__version__),
    source=dict(path=str(source.relative_to(ROOT)), sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                window=[32, 128, 56, 152], originalShape=[153, 181], crs='local planar km',
                sourceCRS='EPSG:4326', rowOrder='south to north', nodata='NaN component'),
    tolerance=dict(profile='L', absoluteFactor=1e-6, relative=1e-4, scaledResidual=1e-4), cases=cases)
output = HERE / 'fixtures/parity-wp4-advection.json'
output.write_text(json.dumps(encode(fixture), separators=(',', ':'), allow_nan=False) + '\n')
assert output.stat().st_size <= 1_500_000
print(f'{output.name}: {output.stat().st_size} bytes; {len(cases)} cases')
