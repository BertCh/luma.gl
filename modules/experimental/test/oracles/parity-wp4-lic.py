# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Composite discrete LIC reference, Cabral & Leedom, SIGGRAPH 1993,
https://doi.org/10.1145/166117.166151; animated window: Forssell, IEEE Vis 1994,
https://graphics.stanford.edu/papers/flowvis/.

The public API specifies a *sampled* Hann weighted sum, not the continuous
kernel integral: scipy.integrate.quad on a continuous noise reconstruction is
not an equal-answer oracle. Use scipy.signal.windows.hann and float64 weighted
reductions at signed integer trace samples. Straight flows independently use
scipy.ndimage.correlate1d with zero padding and normalization by retained kernel
mass. Curved wind uses adaptive scipy.integrate.solve_ivp (DOP853), not our RK2.
No library TS/WGSL or CPU mirror is imported or translated.

randomgen.Philox(number=4,width=32) supplies independent Philox4x32-10 noise:
counter=(column,row,0,0), key=(seed,1), high 24 bits / 2**24. Randomgen increments
before evaluating, so initialize the counter to desired_counter-1 modulo 2**128.
The published Random123 zero-counter known-answer vector is checked first.
https://bashtage.github.io/randomgen/bit_generators/philox.html

All inputs round once to f32; references are f64. Planar coordinates, pixel
centres origin+(index+.5)*cell, row zero south, output nearest-cell lookup,
field bilinear interpolation clamped to outer centres inside half-open bounds.
Any NaN support corner is invalid, even at zero interpolation weight. Domain
exit truncates and renormalizes the kernel. Zero/subthreshold speed keeps the
centre noise. Wind10m frame14 is decoded in FULL then a fixed 80x80 halo window
is used; all 32x32 outputs lie in its interior. Full output arrays are retained.
The short 1-pixel total integration length is a small-step ODE comparison under
unchanged profile L, not an equivalence of midpoint and adaptive integration.
"""
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy
import rasterio
import randomgen
from randomgen import Philox
from scipy.integrate import solve_ivp
from scipy.ndimage import correlate1d, map_coordinates
from scipy.signal.windows import hann

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / 'examples/deck/spatial-analysis-showcase/public/data/gfs-wind/wind10m_14.png'


def make_noise(width, height, seed):
    known = Philox(key=0, counter=2**128-1, number=4, width=32).random_raw(4)
    assert known.tolist() == [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8]
    words = [int(Philox(key=seed + 2**32,
                       counter=(column + row * 2**32 - 1) % 2**128,
                       number=4, width=32).random_raw())
             for row in range(height) for column in range(width)]
    return (np.asarray(words, dtype=np.uint64) >> 8).astype(float).reshape(height, width) / 2**24


def make_case(name, field, width, height, step_count, settings, mode):
    field = np.asarray(field, dtype=np.float32).astype(float)
    field_height, field_width, _ = field.shape
    for key in ['fieldExtent', 'outputExtent']:
        settings[key] = np.asarray(settings[key], dtype=np.float32).astype(float).tolist()
    for key in ['stepLength', 'minimumSpeed', 'phase', 'period']:
        settings[key] = float(np.float32(settings.get(key, 0)))
    origin = np.array(settings['outputExtent'][:2])
    cells = np.array(settings['outputExtent'][2:])
    field_origin = np.array(settings['fieldExtent'][:2])
    field_cells = np.array(settings['fieldExtent'][2:])
    rows, columns = np.indices((height, width))
    centres = origin + (np.stack([columns, rows], axis=-1).reshape(-1, 2) + .5) * cells
    noise = make_noise(width, height, settings['seed'])
    signed_steps = np.arange(-step_count, step_count + 1)
    weights = hann(2 * step_count + 3, sym=True)[1:-1]
    if settings['period'] > 0:
        weights *= .5 * (1 + np.cos(2 * np.pi * (signed_steps / settings['period'] - settings['phase'])))

    def sample(points):
        local = (points - field_origin) / field_cells
        coordinates = np.clip(local - .5, [0, 0], [field_width-1, field_height-1])
        sampled = np.stack([map_coordinates(field[:, :, component], coordinates.T[::-1],
                              order=1, mode='nearest', prefilter=False) for component in range(2)], axis=-1)
        # Explicit strict four-corner invalid support policy (also at exact centres).
        lower = np.floor(coordinates).astype(int)
        upper = np.minimum(lower + 1, [field_width-1, field_height-1])
        invalid = np.any((local < 0) | (local >= [field_width, field_height]), axis=-1)
        for horizontal in [lower[:, 0], upper[:, 0]]:
            for vertical in [lower[:, 1], upper[:, 1]]:
                invalid |= np.isnan(field[vertical, horizontal]).any(axis=-1)
        sampled[invalid] = np.nan
        return sampled

    speeds = np.linalg.norm(sample(centres), axis=-1)
    if mode in ['horizontal', 'vertical']:
        axis = 1 if mode == 'horizontal' else 0
        values = correlate1d(noise, weights, axis=axis, mode='constant', cval=0)
        values /= correlate1d(np.ones_like(noise), weights, axis=axis, mode='constant', cval=0)
    elif mode == 'stationary':
        values = noise.copy()
    else:
        assert np.isfinite(field).all() and np.min(speeds) > 0
        samples = [centres]
        sample_weights = [np.full(len(centres), weights[step_count])]
        minimum_pixel_margin = 1.0
        for direction in [-1, 1]:
            def velocity(time, packed):
                sampled = sample(packed.reshape(-1, 2))
                return (direction * cells * sampled / np.linalg.norm(sampled, axis=-1)[:, None]).ravel()
            times = np.arange(1, step_count + 1) * settings['stepLength']
            solution = solve_ivp(velocity, (0, times[-1]), centres.ravel(), t_eval=times,
                                 method='DOP853', atol=1e-12, rtol=1e-12,
                                 max_step=settings['stepLength'] / 4)
            assert solution.success
            positions = solution.y.T.reshape(step_count, -1, 2)
            inside = np.cumprod(np.all((positions >= origin) &
                       (positions < origin + cells * [width, height]), axis=-1), axis=0)
            for index, points in enumerate(positions):
                local = (points-origin)/cells
                minimum_pixel_margin = min(minimum_pixel_margin, float(np.min(np.abs(local-np.rint(local)))))
                samples.append(points)
                sample_weights.append(inside[index] * weights[step_count + direction * (index+1)])
        pixels = np.floor((np.array(samples)-origin)/cells).astype(int)
        pixels = np.clip(pixels, [0, 0], [width-1, height-1])
        selected_noise = noise[pixels[..., 1], pixels[..., 0]]
        retained_weights = np.array(sample_weights)
        values = (np.sum(selected_noise * retained_weights, axis=0) / np.sum(retained_weights, axis=0)).reshape(height, width)
        assert minimum_pixel_margin > 1e-6
    values = values.ravel()
    values[np.isnan(speeds)] = np.nan
    return dict(name=name, fieldWidth=field_width, fieldHeight=field_height,
                width=width, height=height, field=field.ravel().tolist(),
                stepCount=step_count, settings=settings, values=values.tolist(),
                speeds=speeds.tolist(), noise=noise.ravel().tolist(),
                reference=('scipy.ndimage.correlate1d + scipy.signal.windows.hann + randomgen.Philox'
                           if mode in ['horizontal', 'vertical'] else
                           'randomgen.Philox + scipy.ndimage.map_coordinates (centre-only stopped flow)'
                           if mode == 'stationary' else
                           'composite discrete LIC: scipy.integrate.solve_ivp + scipy.ndimage.map_coordinates + scipy.signal.windows.hann + randomgen.Philox'),
                **({'minimumPixelMargin': minimum_pixel_margin} if mode == 'wind' else {}))


def settings(seed=3, cells=(1, 1), **extra):
    return dict(fieldExtent=[0, 0, *cells], outputExtent=[0, 0, *cells],
                seed=seed, stepLength=1, **extra)


cases = []
horizontal = np.zeros((48, 64, 2)); horizontal[..., 0] = 3
cases.append(make_case('horizontal-hann-boundary-renormalization', horizontal, 64, 48, 12, settings(), 'horizontal'))
vertical = np.zeros((48, 64, 2)); vertical[..., 1] = 4
cases.append(make_case('vertical-anisotropic-animated-hann', vertical, 64, 48, 16,
                      settings(seed=4294967295, cells=(2, 3), period=7, phase=.25), 'vertical'))
edge = np.zeros((6, 8, 2)); edge[2, 3] = np.nan
cases.append(make_case('zero-speed-nan-support-outside', edge, 10, 8, 12,
                      {**settings(seed=17), 'outputExtent': [-1, -1, 1, 1]}, 'stationary'))
slow = np.full((7, 9, 2), .25)
cases.append(make_case('subthreshold-keeps-centre-noise', slow, 9, 7, 8,
                      settings(seed=0, minimumSpeed=.5), 'stationary'))
with rasterio.open(SOURCE) as dataset:
    decoded = dataset.read()[:2].transpose(1, 2, 0)[::-1]
wind = (-36 + decoded.astype(float) / 255 * [81, 77])[32:112, 48:128]
cases.append(make_case('gfs-frame14-halo-small-step-ode', wind, 32, 32, 8,
                      {**settings(seed=91, period=5, phase=.125),
                       'outputExtent': [24, 24, 1, 1], 'stepLength': .125}, 'wind'))


def sanitize(value):
    if isinstance(value, dict): return {key: sanitize(entry) for key, entry in value.items()}
    if isinstance(value, list): return [sanitize(entry) for entry in value]
    if isinstance(value, float) and np.isnan(value): return None
    return value


fixture = dict(versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                            rasterio=rasterio.__version__, randomgen=randomgen.__version__),
               source=dict(path=str(SOURCE.relative_to(ROOT)), sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                           fullShape=list(decoded.shape), window=[32, 112, 48, 128]), cases=cases)
output = HERE / 'fixtures/parity-wp4-lic.json'
output.write_text(json.dumps(sanitize(fixture), separators=(',', ':'), allow_nan=False) + '\n')
assert output.stat().st_size <= 1_500_000
print(output.name, output.stat().st_size, [(case['name'], len(case['values'])) for case in cases])
