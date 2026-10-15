# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent particle lifecycle reference: randomgen Philox4x32-10 and SciPy ODE.

Philox is the established Random123 algorithm (Salmon et al., SC 2011,
https://doi.org/10.1145/2063384.2063405). Public counter/key contracts select
spawn (particle,generation; seed,4) and drop (particle,frame; seed,5).
A composite lifecycle adapter implements the documented age/reset/parking
contract, not the GPU integrator. Uniform fields make solve_ivp exact at every
frame, including negative/zero speed scales; no midpoint implementation is
copied. Half-open bounds, f32 inputs, exact age/generation comparisons, profile L
positions. This isolates lifecycle from the curved-field discretization test.
Scene topologies: speeds+trails (wind/terrain), state-only (ocean). Wind trail
lengths 8/16/32 and viewport/full spawn bounds are each exercised, plus optional
previous positions. Changing velocity/settings/seed and reset happen in one graph.
"""
import json
import platform
from pathlib import Path
import numpy as np
import scipy
import randomgen
from randomgen import Philox
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent

def random_words(particle, counter, seed, purpose):
    return Philox(key=seed + purpose * 2**32,
                  counter=(particle + counter * 2**32 - 1) % 2**128,
                  number=4, width=32).random_raw(4)


def make_case(name, bounds, trail_length, output_mode):
    count = 67  # crosses a workgroup boundary
    positions = np.column_stack([np.linspace(2, 6, count), np.full(count, 4.)]).astype(float)
    ages = np.zeros(count, dtype=int)
    generations = np.zeros(count, dtype=int)
    initial = positions.ravel().tolist()
    trails = np.zeros((count, trail_length, 2))
    frames = []
    # reset/stagger, random drop, expiry, reverse/zero, slow, mandatory drop,
    # changing field, domain exit, reset/recovery. Drop .05 is the scene maximum.
    sequence = [(True, 1., .05, 40, 0., [0.125, 0.]),
                (False, 1., .05, 40, 0., [0.125, 0.]),
                (False, 1., 0., 2, 0., [0.125, 0.]),
                (False, -1., 0., 0, 0., [0.125, 0.]),
                (False, 0., 0., 0, 0., [0.125, 0.]),
                (False, 1., 0., 0, 1., [0.125, 0.]),
                (False, 1., 1., 0, 0., [0.125, 0.]),
                (False, 2., 0., 0, 0., [0., 0.125]),
                (False, 1., 0., 0, 0., [16., 0.]),
                (True, 1., 0., 40, 0., [0.125, 0.])]
    for frame, (reset, speed_scale, drop_rate, maximum_age, minimum_speed, velocity) in enumerate(sequence):
        seed = 7 if frame < 9 else 31
        settings = dict(fieldExtent=[0, 0, 1, 1], timeStep=1,
                        speedScale=speed_scale, dropRate=float(np.float32(drop_rate)),
                        minimumSpeed=minimum_speed)
        if bounds is not None:
            settings['spawnBounds'] = bounds
        spawn = np.array(bounds if bounds is not None else [0, 0, 8, 8], dtype=float)
        words = dict(seed=seed, frame=frame + 31, maximumAge=maximum_age, reset=reset)
        previous = positions.copy()
        speeds = np.zeros(count)
        for particle in range(count):
            old = positions[particle].copy()
            next_position = solve_ivp(lambda time, point: np.array(velocity) * speed_scale,
                                      [0, 1], old, method='DOP853', rtol=1e-12, atol=1e-12).y[:, -1]
            valid = np.all((old >= 0) & (old < 8)) and np.all((next_position >= 0) & (next_position < 8))
            lottery = (int(random_words(particle, words['frame'], seed, 5)[0]) >> 8) / 2**24
            expired = maximum_age > 0 and ages[particle] + 1 >= maximum_age
            respawn = reset or not valid or expired or np.linalg.norm(velocity) * abs(speed_scale) < minimum_speed or lottery < settings['dropRate']
            if respawn:
                generations[particle] = 0 if reset else generations[particle] + 1
                random = (random_words(particle, int(generations[particle]), seed, 4).astype(np.uint64) >> 8).astype(float) / 2**24
                positions[particle] = spawn[:2] + random[:2] * (spawn[2:] - spawn[:2])
                previous[particle] = positions[particle]
                ages[particle] = min(int(np.float32(random[2]) * np.float32(maximum_age)), max(maximum_age, 1) - 1) if reset else 0
            else:
                positions[particle] = next_position
                ages[particle] += 1
            if np.all((positions[particle] >= 0) & (positions[particle] < 8)):
                speeds[particle] = np.linalg.norm(velocity) * abs(speed_scale)
            if reset or ages[particle] == 0:
                trails[particle] = positions[particle]
            else:
                trails[particle, words['frame'] % trail_length] = positions[particle]
        frames.append(dict(settings=settings, words=words, velocity=velocity,
                           positions=positions.ravel().tolist(), previous=previous.ravel().tolist(),
                           speeds=speeds.tolist(), ages=ages.tolist(), generations=generations.tolist(),
                           trails=trails.ravel().tolist() if output_mode != 'none' else []))
    return dict(name=name, width=8, height=8, initial=initial, trailLength=trail_length,
                outputMode=output_mode, frames=frames)

known = Philox(key=0, counter=2**128-1, number=4, width=32).random_raw(4)
assert known.tolist() == [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8]
for trail_length in [8, 16, 32]:
    cases = [make_case(f'wind-full-{trail_length}', None, trail_length, 'scene'),
             make_case(f'wind-viewport-{trail_length}', [2, 2, 6, 6], trail_length, 'all')]
    if trail_length == 8:
        cases.append(make_case('ocean-outside-parking-no-outputs', [-20, -20, -20, -20], 1, 'none'))
    fixture = dict(reference='randomgen.Philox; scipy.integrate.solve_ivp; composite documented lifecycle',
                   versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, randomgen=randomgen.__version__),
                   conventions=dict(inputPrecision='float32', bounds='half-open', field='uniform; planar world units/time'), cases=cases)
    for index, case in enumerate(cases):
        fixture['cases'] = [case]
        output = HERE / f'fixtures/parity-wp4-lifecycle-{trail_length}-{index}.json'
        output.write_text(json.dumps(fixture, separators=(',', ':')) + '\n')
        assert output.stat().st_size < 1_500_000
        print(output.name, output.stat().st_size)
