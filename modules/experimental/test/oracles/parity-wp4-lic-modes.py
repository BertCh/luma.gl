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
import math
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
        # Exact constant-flow ODE lands at k*stepLength pixels. Aggregate Hann
        # taps by their independent nearest noise-cell displacement.
        displacements = np.floor(.5 + signed_steps * settings['stepLength']).astype(int)
        radius = int(np.max(np.abs(displacements)))
        kernel = np.bincount(displacements + radius, weights=weights, minlength=2*radius+1)
        values = correlate1d(noise, kernel, axis=axis, mode='constant', cval=0)
        values /= correlate1d(np.ones_like(noise), kernel, axis=axis, mode='constant', cval=0)
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


def add_rounding_intervals(case):
    """Enumerate discrete noise-cell ambiguity, never enlarge profile L.

    Uniform (3,0), unit cells: normalized direction (1,0) is exact. Each
    accepted step adds the f32 step once. Standard summation error is
    gamma_k*(|start|+k*|step|), gamma_k=k*u/(1-k*u), u=2^-24 (Higham,
    Accuracy and Stability of Numerical Algorithms, second ed., section 3.1).
    At most two floor results are possible; include either bordering pixel.
    A domain-edge tap may be omitted. Enumerate all such optional taps and
    bound the normalized weighted average. Later taps are strictly outside.
    Hann/ripple/reduction error retains unchanged L after this discrete test.
    """
    import itertools
    width,height=case['width'],case['height']
    noise=np.asarray(case['noise']).reshape(height,width)
    taps=case['stepCount']; step=case['settings']['stepLength']
    indices=np.arange(-taps,taps+1)
    weights=hann(2*taps+3,sym=True)[1:-1]
    if case['settings']['period']>0:
        weights*=.5*(1+np.cos(2*np.pi*(indices/case['settings']['period']-case['settings']['phase'])))
    lower=[];upper=[];ambiguous=[]
    for row,column in np.ndindex((height,width)):
        denominator=0.;minimum=0.;maximum=0.;optional=[];ambiguity=0
        for index,weight in zip(indices,weights):
            center=column+.5+index*step
            magnitude=abs(int(index));gamma=magnitude*2**-24/(1-magnitude*2**-24)
            bound=gamma*(column+.5+magnitude*abs(step))
            candidates=range(math.floor(center-bound),math.floor(center+bound)+1)
            inside=[candidate for candidate in candidates if 0<=candidate<width]
            if not inside: continue
            noise_min=float(np.min(noise[row,inside]));noise_max=float(np.max(noise[row,inside]))
            ambiguity+=int(len(candidates)>1)
            if len(inside)<len(candidates): optional.append((weight,noise_min*weight,noise_max*weight))
            else:
                denominator+=weight;minimum+=noise_min*weight;maximum+=noise_max*weight
        assert len(optional)<=2
        minima=[];maxima=[]
        for choices in itertools.product([False,True],repeat=len(optional)):
            chosen=[item for choice,item in zip(choices,optional) if choice]
            total=denominator+sum(item[0] for item in chosen)
            minima.append((minimum+sum(item[1] for item in chosen))/total)
            maxima.append((maximum+sum(item[2] for item in chosen))/total)
        lower.append(min(minima));upper.append(max(maxima));ambiguous.append(ambiguity)
    case.update(lower=lower,upper=upper,ambiguousTaps=ambiguous)
    assert np.all(np.asarray(case['values'])>=np.asarray(lower)-1e-12)
    assert np.all(np.asarray(case['values'])<=np.asarray(upper)+1e-12)

# Round 8: all three showcase tap choices x static/animated phase x slider
# endpoints/default step. Constant field makes the discrete reference exact;
# original wind ODE case retains independent curved-field coverage.
cases = []
field = np.zeros((17, 67, 2)); field[..., 0] = 3
for taps in [16, 32, 64]:
    for phase in [0., .25]:
        for step in [.3, .8, 1.6]:
            cases.append(make_case(f'showcase-taps{taps}-phase{phase}-step{step}', field, 67, 17, taps,
                         {**settings(seed=19, period=7, phase=phase), 'stepLength':step}, 'horizontal'))

def sanitize(value):
    if isinstance(value, dict): return {key: sanitize(entry) for key, entry in value.items()}
    if isinstance(value, list): return [sanitize(entry) for entry in value]
    if isinstance(value, float) and np.isnan(value): return None
    return value

for case in cases:
    add_rounding_intervals(case)

for taps in [16, 32, 64]:
    fixture = dict(versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, randomgen=randomgen.__version__),
                   conventions='f32 step; exact uniform horizontal ODE; nearest noise-cell samples; truncated Hann/ripple kernel; L unchanged',
                   cases=[case for case in cases if case['stepCount'] == taps])
    output = HERE / f'fixtures/parity-wp4-lic-modes-{taps}.json'
    output.write_text(json.dumps(sanitize(fixture), separators=(',', ':'), allow_nan=False) + '\n')
    assert output.stat().st_size <= 1_500_000
    print(output.name, output.stat().st_size)
