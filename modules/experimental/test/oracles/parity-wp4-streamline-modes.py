# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent streamline reference: scipy.integrate.solve_ivp (DOP853),
scipy.ndimage.map_coordinates, randomgen.Philox. Composite spacing reference:
Jobard & Lefer 1997, https://doi.org/10.1007/978-3-7091-6876-9_5, with the API's
explicit priority-ordered lattice/occupied-cell adaptation. There is no drop-in
library for that discrete adaptation. NumPy trajectories and Python cell sets
implement its declarative greedy rule; no TS/WGSL/CPU mirror is imported.

Planar world units; arclength ODE v/|v|, NOT velocity*time. f32 inputs, f64 ODE.
Philox4x32-10 counter=(seedIndex,0,0,0), key=(seed,2); high24 jitter; high16 word2
priority, lower index breaks ties. Seed placement rounds the specified f32
lattice coordinates. Randomgen increments before evaluation: counter-1.
Grid half-open; bilinear field clamps to outer centers. Horizontal case tests
boundary termination; wind/rotation stay inside field halo so event placement
is unambiguous. Zero/subthreshold/NaN seeds are rejected. Full candidate arrays
and full accepted CSR polylines are stored with lossless gzip/base64 transport. The wind case uses 32 steps of 1/32 world unit per direction (same 1-unit
horizon as the initial 8-step diagnostic). The short-step ODE regime uses the
unchanged L profile; it does not assert RK2 and adaptive integration equivalence.
"""
import hashlib
import base64
import gzip
import json
import platform
from pathlib import Path
import numpy as np
import scipy
import rasterio
import randomgen
from randomgen import Philox
from scipy.integrate import solve_ivp
from scipy.ndimage import map_coordinates

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / 'examples/deck/spatial-analysis-showcase/public/data/gfs-wind/wind10m_14.png'


def make_case(name, field, grid_size, lattice, steps, settings, straight=False):
    field = np.asarray(field, dtype=np.float32).astype(float)
    height, width, _ = field.shape
    origin = np.asarray(settings['gridExtent'][:2], dtype=np.float32)
    cells = np.asarray(settings['gridExtent'][2:], dtype=np.float32)
    field_origin = np.asarray(settings['fieldExtent'][:2])
    field_cells = np.asarray(settings['fieldExtent'][2:])
    seed_count = int(np.prod(lattice)); stride = 2 * steps + 1
    words = np.array([Philox(key=settings['seed'] + 2 * 2**32, counter=(index-1) % 2**128,
                            number=4, width=32).random_raw(4) for index in range(seed_count)], dtype=np.uint64)
    jitter = (words[:, :2] >> 8).astype(np.float32) / np.float32(2**24)
    columns = np.arange(seed_count) % lattice[0]; rows = np.arange(seed_count) // lattice[0]
    seed_cells = (cells * np.asarray(grid_size, dtype=np.float32) / np.asarray(lattice, dtype=np.float32)).astype(np.float32)
    starts = (origin + (np.column_stack([columns, rows]).astype(np.float32) + jitter) * seed_cells).astype(np.float32).astype(float)
    keys = ((words[:, 2] >> 16) << 16) | (65535-np.arange(seed_count, dtype=np.uint64))

    def sample(points):
        local = (points-field_origin)/field_cells
        coordinates = np.clip(local-.5, [0, 0], [width-1, height-1])
        sampled = np.stack([map_coordinates(field[..., component], coordinates.T[::-1], order=1,
                                           mode='nearest', prefilter=False) for component in range(2)], axis=-1)
        lower = np.floor(coordinates).astype(int); upper = np.minimum(lower+1, [width-1, height-1])
        invalid = np.any((local < 0) | (local >= [width, height]), axis=-1)
        for horizontal in [lower[:, 0], upper[:, 0]]:
            for vertical in [lower[:, 1], upper[:, 1]]:
                invalid |= np.isnan(field[vertical, horizontal]).any(axis=-1)
        sampled[invalid] = np.nan
        return sampled

    candidates = np.full((seed_count, stride, 2), np.nan)
    spans = np.zeros((seed_count, 2), dtype=np.uint32)
    speed = np.linalg.norm(sample(starts), axis=-1)
    valid = np.isfinite(speed) & (speed > 0) & (speed >= settings.get('minimumSpeed', 0))
    candidates[valid, steps] = starts[valid]
    times = np.arange(1, steps+1) * settings['stepLength']
    for direction_index, direction in enumerate([-1, 1]):
        if not valid.any(): continue
        if straight:
            velocities = sample(starts[valid]); velocities /= np.linalg.norm(velocities, axis=-1)[:, None]
            # Exact integral of a constant vector field, independently checked by solve_ivp below.
            traced = starts[valid][None] + direction * times[:, None, None] * velocities[None]
            check = solve_ivp(lambda time, point: direction*velocities[0], (0, times[-1]), starts[valid][0],
                              t_eval=times, method='DOP853', atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(check.y.T, traced[:, 0], atol=1e-12)
        else:
            def velocity(time, packed):
                vector = sample(packed.reshape(-1, 2))
                return (direction * vector / np.linalg.norm(vector, axis=-1)[:, None]).ravel()
            solution = solve_ivp(velocity, (0, times[-1]), starts[valid].ravel(), t_eval=times,
                                 method='DOP853', atol=1e-12, rtol=1e-12, max_step=settings['stepLength']/4)
            assert solution.success
            traced = solution.y.T.reshape(steps, -1, 2)
        inside = np.cumprod(np.all((traced >= origin) & (traced < origin + cells*grid_size), axis=-1), axis=0).astype(bool)
        for offset, seed_index in enumerate(np.flatnonzero(valid)):
            count = int(inside[:, offset].sum()); spans[seed_index, direction_index] = count
            for step_index in range(count):
                candidates[seed_index, steps+direction*(step_index+1)] = traced[step_index, offset]

    # Greedy cell-exclusion definition, independent of GPU parallel rounds and scan publication.
    occupancy = set(); accepted = {}; kept = np.zeros_like(spans)
    reciprocal = (1/cells).astype(np.float32)
    cell_coordinates = np.full(candidates.shape, -1, dtype=np.int64)
    finite = np.isfinite(candidates).all(axis=-1)
    local = ((candidates[finite].astype(np.float32)-origin)*reciprocal).astype(np.float32)
    cell_coordinates[finite] = np.floor(local).astype(int)
    margin = float(np.min(np.abs(local-np.rint(local)))) if finite.any() else 1
    for seed_index in np.argsort(keys)[::-1]:
        if not valid[seed_index]: continue
        row_cells = [tuple(point) for point in cell_coordinates[seed_index]]
        if row_cells[steps] in occupancy: continue
        selected = [steps]
        for direction_index, direction in enumerate([-1, 1]):
            for distance in range(1, int(spans[seed_index, direction_index])+1):
                slot = steps+direction*distance
                if row_cells[slot] in occupancy: break
                selected.append(slot)
        if len(selected) < settings['minimumPoints']: continue
        selected.sort()
        accepted[int(seed_index)] = candidates[seed_index, selected]
        kept[seed_index] = [steps-selected[0], selected[-1]-steps]
        occupancy.update(row_cells[slot] for slot in selected)
    ids = sorted(accepted); offsets = [0]
    for seed_index in ids: offsets.append(offsets[-1]+len(accepted[seed_index]))
    points = np.concatenate([accepted[seed_index] for seed_index in ids]) if ids else np.empty((0, 2))
    return dict(name=name, reference='scipy.integrate.solve_ivp + scipy.ndimage.map_coordinates + randomgen.Philox; composite priority spacing',
                field=field.ravel().tolist(), fieldWidth=width, fieldHeight=height,
                config=dict(gridWidth=grid_size[0], gridHeight=grid_size[1], seedColumns=lattice[0], seedRows=lattice[1],
                            stepsPerDirection=steps, lineCapacity=seed_count, pointCapacity=seed_count*stride), settings=settings,
                candidates=candidates.ravel().tolist(), spans=spans.ravel().tolist(), kept=kept.ravel().tolist(),
                ids=ids, offsets=offsets, points=points.ravel().tolist(), keys=keys.tolist(), minimumCellMargin=margin)


def settings(field=(0, 0, 1, 1), grid=(0, 0, 1, 1), step=.125, seed=17, minimum_speed=0):
    return dict(fieldExtent=list(field), gridExtent=list(grid), stepLength=step, seed=seed,
                minimumSpeed=minimum_speed, minimumPoints=2)


# Round 8: scene 48-step horizon, 0.4 occupancy-cell step, shortest-line
# endpoint/default settings and two seeds. The field has a full-domain halo.
cases=[]
field=np.zeros((32,64,2));field[...,0]=3
for seed in [0,99]:
 for minimum_points in [2,6,40]:
  for cell_width in [1,2]:
   options=settings(grid=(8,8,cell_width,1),step=.4*cell_width,seed=seed)
   options['minimumPoints']=minimum_points
   cases.append(make_case(f'showcase48-seed{seed}-points{minimum_points}-aspect{cell_width}',field,[16,8],[12,8],48,options,straight=True))

def sanitize(value):
 if isinstance(value,dict):return {key:sanitize(entry) for key,entry in value.items()}
 if isinstance(value,list):return [sanitize(entry) for entry in value]
 if isinstance(value,float) and np.isnan(value):return None
 return value
for case in cases:
 for key in ['field','candidates','points']:
  case[key]=base64.b64encode(gzip.compress(json.dumps(sanitize(case[key]),separators=(',',':'),allow_nan=False).encode(),mtime=0)).decode()
for seed in [0,99]:
 fixture=dict(reference='scipy.integrate.solve_ivp; randomgen.Philox; composite documented greedy occupancy',
  versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,randomgen=randomgen.__version__),
  cases=[case for case in cases if case['settings']['seed']==seed])
 output=HERE/f'fixtures/parity-wp4-streamline-modes-{seed}.json'
 output.write_text(json.dumps(sanitize(fixture),separators=(',',':'),allow_nan=False)+'\n')
 assert output.stat().st_size<1_500_000
 print(output.name,output.stat().st_size)
