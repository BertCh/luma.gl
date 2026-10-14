# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Reference exact encounters (closest point of approach between linearly moving tracks) for
GPUTrajectoryClosestApproach.

Writes fixtures/trajectory-closest-approach.json next to this script, consumed by
test/gpu-spatial-analysis/trajectory-encounters/closest-approach-oracle.ts (and through it by the
node and GPU specs). Run from anywhere with
    <venv>/bin/python -I modules/experimental/test/oracles/trajectory-closest-approach.py
in a venv with numpy and shapely (versions recorded in the fixture).

No pip package implements MobilityDB's `tdwithin` / `nearestApproachDistance`, so this is a derived
reference, checked inside the script:
- The reference: tracks move linearly between samples. For every pair of segments of two tracks
  (brute force, no grid) the separation over their common time interval is `D + V tau`, the
  closest approach is at `tau = -(D . V) / |V|^2` clamped to the interval, and the time within the
  distance is the clamped interval between the roots of `|D + V tau|^2 = r^2`. Per track pair:
  the minimum distance, its time (earliest when tied), the first enter time and the total time
  within the distance. Positions and times are rounded to float32 first, then evaluated in binary64.
- Self-check 1 (sampling): every segment pair is sampled at 20001 instants; the sampled minimum
  distance is never below the analytic one (minus 1e-9) and converges to it, and the sampled
  measure of the within-distance set matches the analytic duration to the sampling step.
- Self-check 2 (shapely): for segment pairs with equal velocity (V = 0) the distance is constant,
  and for every pair the analytic minimum is at least the shapely distance between the two segment
  paths (a time-free lower bound).

Times in the fixture are relative to the first sample of the lower track of each pair, as the GPU
reports them.

Tolerances (stored in `tolerances`): the GPU evaluates the quadratic in float32 from float32 inputs,
so distances carry about 1e-5 of the coordinate magnitude and times about 1e-5 of the time span of
a segment (the roots divide by |V|^2); near-tangent passes (a minimum within 1e-3 of the distance)
are reported as ambiguous and excluded from the pair-set comparison.
"""

import json
import math
import os
import platform

import numpy as np
import shapely

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'fixtures', 'trajectory-closest-approach.json')


def f32(value):
    return float(np.float32(value))


def segment_pair(a0, a1, ta0, ta1, b0, b1, tb0, tb1, radius):
    start = max(ta0, tb0)
    end = min(ta1, tb1)
    if start > end:
        return None

    def position(p0, p1, t0, t1, t):
        fraction = (t - t0) / (t1 - t0)
        return (p0[0] + (p1[0] - p0[0]) * fraction, p0[1] + (p1[1] - p0[1]) * fraction)

    pa = position(a0, a1, ta0, ta1, start)
    pb = position(b0, b1, tb0, tb1, start)
    qa = position(a0, a1, ta0, ta1, end)
    qb = position(b0, b1, tb0, tb1, end)
    d = (pb[0] - pa[0], pb[1] - pa[1])
    span = end - start
    if span > 0:
        v = ((qb[0] - qa[0] - d[0]) / span, (qb[1] - qa[1] - d[1]) / span)
    else:
        v = (0.0, 0.0)
    a = v[0] * v[0] + v[1] * v[1]
    b = d[0] * v[0] + d[1] * v[1]
    c = d[0] * d[0] + d[1] * d[1] - radius * radius
    closest = min(max(-b / a, 0.0), span) if a > 0 else 0.0
    minimum = math.hypot(d[0] + v[0] * closest, d[1] + v[1] * closest)
    if a > 0:
        discriminant = b * b - a * c
        if discriminant < 0:
            return None
        root = math.sqrt(discriminant)
        enter = max((-b - root) / a, 0.0)
        leave = min((-b + root) / a, span)
        if enter > leave:
            return None
    else:
        if c > 0:
            return None
        enter, leave = 0.0, span
    return {
        'start': start,
        'span': span,
        'd': d,
        'v': v,
        'minimum': minimum,
        'closest': start + closest,
        'enter': start + enter,
        'exit': start + leave,
    }


def self_check(hit, radius, a_segment, b_segment):
    samples = 20001
    span = hit['span']
    taus = np.linspace(0.0, span, samples)
    distances = np.hypot(hit['d'][0] + hit['v'][0] * taus, hit['d'][1] + hit['v'][1] * taus)
    assert distances.min() >= hit['minimum'] - 1e-9, 'sampled minimum below the analytic one'
    step = span / (samples - 1) if span > 0 else 0.0
    assert distances.min() - hit['minimum'] <= max(1e-6, step * math.hypot(*hit['v'])), 'no convergence'
    measured = float(np.count_nonzero(distances <= radius)) * step
    assert abs(measured - (hit['exit'] - hit['enter'])) <= 2 * step + 1e-9, 'duration mismatch'
    path_distance = shapely.LineString(a_segment).distance(shapely.LineString(b_segment))
    assert hit['minimum'] >= path_distance - 1e-9, 'below the time-free path distance'
    if hit['v'] == (0.0, 0.0):
        assert abs(distances.max() - distances.min()) <= 1e-9


def reference(tracks, radius):
    pairs = []
    for track in range(len(tracks)):
        for partner in range(track + 1, len(tracks)):
            a = tracks[track]
            b = tracks[partner]
            hits = []
            for i in range(len(a['positions']) - 1):
                if not a['times'][i + 1] > a['times'][i]:
                    continue
                for j in range(len(b['positions']) - 1):
                    if not b['times'][j + 1] > b['times'][j]:
                        continue
                    hit = segment_pair(
                        a['positions'][i], a['positions'][i + 1], a['times'][i], a['times'][i + 1],
                        b['positions'][j], b['positions'][j + 1], b['times'][j], b['times'][j + 1],
                        radius,
                    )
                    if hit is None:
                        continue
                    self_check(hit, radius, [a['positions'][i], a['positions'][i + 1]],
                               [b['positions'][j], b['positions'][j + 1]])
                    hits.append(hit)
            if not hits:
                continue
            origin = a['times'][0]
            best = min(hits, key=lambda hit: (hit['minimum'], hit['closest']))
            pairs.append({
                'track': track,
                'partner': partner,
                'minimumDistance': best['minimum'],
                'closestTime': best['closest'] - origin,
                'firstTime': min(hit['enter'] for hit in hits) - origin,
                'duration': sum(hit['exit'] - hit['enter'] for hit in hits),
                'isNearTangent': abs(best['minimum'] - radius) < 1e-3 * max(1.0, radius),
            })
    return pairs


def make_track(points):
    return {
        'positions': [[f32(x), f32(y)] for x, y, _ in points],
        'times': [f32(t) for _, _, t in points],
    }


def main():
    scenes = []
    # Two tracks cross at x = 50 between their samples (t = 5): clock buckets at t = 0 and 10
    # see them 100 apart, the exact closest approach is 0.
    scenes.append({'name': 'cross-between-samples', 'distance': 2.0, 'tracks': [
        make_track([(0, 0, 0), (100, 0, 10)]),
        make_track([(50, -50, 0), (50, 50, 10)]),
    ]})
    # Parallel at a constant 3 units (same velocity): within for the whole common time [2, 10].
    scenes.append({'name': 'parallel-constant', 'distance': 3.5, 'tracks': [
        make_track([(0, 0, 0), (10, 0, 5), (20, 0, 10)]),
        make_track([(4, 3, 2), (24, 3, 12)]),
    ]})
    # Same path at different times: never close at the same instant.
    scenes.append({'name': 'same-path-different-times', 'distance': 1.0, 'tracks': [
        make_track([(0, 0, 0), (10, 0, 10)]),
        make_track([(0, 0, 20), (10, 0, 30)]),
    ]})
    # A stopped track (repeated position) and a passer-by at closest distance 4.
    scenes.append({'name': 'stopped', 'distance': 5.0, 'tracks': [
        make_track([(5, 4, 0), (5, 4, 100)]),
        make_track([(0, 0, 10), (10, 0, 20)]),
    ]})
    # Large absolute times (Unix seconds); partial time overlap.
    base = 1.7e9
    scenes.append({'name': 'large-times-partial-overlap', 'distance': 10.0, 'tracks': [
        make_track([(0, 0, base), (100, 0, base + 100)]),
        make_track([(60, 5, base + 50), (60, 5, base + 200)]),
    ]})
    rng = np.random.default_rng(20261014)
    tracks = []
    for _ in range(60):
        x, y, t = rng.uniform(0, 500), rng.uniform(0, 500), rng.uniform(0, 100)
        points = []
        heading = rng.uniform(0, 2 * math.pi)
        for _ in range(int(rng.integers(2, 30))):
            points.append((x, y, t))
            heading += rng.normal(0, 0.4)
            speed = rng.uniform(0.5, 3)
            dt = rng.uniform(1, 10)
            x += math.cos(heading) * speed * dt
            y += math.sin(heading) * speed * dt
            t += dt
        tracks.append(make_track(points))
    scenes.append({'name': 'random', 'distance': 15.0, 'tracks': tracks})
    for scene in scenes:
        scene['pairs'] = reference(scene['tracks'], scene['distance'])
    assert scenes[0]['pairs'] and scenes[0]['pairs'][0]['minimumDistance'] < 1e-6
    assert abs(scenes[1]['pairs'][0]['duration'] - 8.0) < 1e-9
    assert not scenes[2]['pairs']
    fixture = {
        'generator': 'test/oracles/trajectory-closest-approach.py',
        'versions': {
            'python': platform.python_version(),
            'numpy': np.__version__,
            'shapely': shapely.__version__,
        },
        'tolerances': {
            'distanceRelative': 1e-5,
            'distanceAbsolute': 1e-3,
            'timeAbsolute': 1e-2,
            'timeRelative': 1e-5,
        },
        'scenes': scenes,
    }
    with open(OUTPUT, 'w') as file:
        json.dump(fixture, file, indent=1)
        file.write('\n')
    print(f"wrote {OUTPUT}: {sum(len(scene['pairs']) for scene in scenes)} pairs")


if __name__ == '__main__':
    main()
