#!/usr/bin/env python3
# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Oracle generator for the relate engine on near-parallel polygon edges (review suspicion D S-1).

Reference implementation: shapely (GEOS) `relate`. Each pair is two quadrilaterals whose long edges
run within a few float32 ulps of each other: the right edge has its endpoints at, beyond, or shifted
along the left edge, perturbed by -3..3 ulps, and its quadrilateral lies on either side. Every
coordinate is rounded to float32 before shapely sees it, so the GPU and shapely relate the same
geometry. Pairs sit in [520, 1000), one 24-unit cell each (the spec compares each pair with its own cell only).

Locating the rounded midpoint of an edge piece put such pieces on the wrong side of the other
polygon's boundary (11 of the 400 pairs of seed 7 had a wrong DE-9IM matrix).

Usage (writes fixtures/geometry-relate-near-parallel.json):  <venv>/bin/python -I geometry-relate-near-parallel.py
Needs shapely and numpy (versions recorded in the fixture).
"""
import json
import math
import os
import random

import numpy as np
import shapely
from shapely.geometry import Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 7
PAIR_COUNT = 400


def to_float32(point):
    return [float(np.float32(point[0])), float(np.float32(point[1]))]


def main():
    random.seed(SEED)
    ulp = float(np.spacing(np.float32(800)))
    pairs = []
    for index in range(PAIR_COUNT):
        origin = (520 + 24 * (index % 20), 520 + 24 * (index // 20))
        a = (origin[0] + random.uniform(0, 4), origin[1] + random.uniform(0, 4))
        length = random.uniform(5, 15)
        angle = random.uniform(0, 2 * math.pi)
        direction = (math.cos(angle), math.sin(angle))
        normal = (-direction[1], direction[0])
        b = (a[0] + length * direction[0], a[1] + length * direction[1])
        a, b = to_float32(a), to_float32(b)
        left = [
            a,
            b,
            to_float32((b[0] + 5 * normal[0], b[1] + 5 * normal[1])),
            to_float32((a[0] + 5 * normal[0], a[1] + 5 * normal[1])),
        ]
        mode = random.choice(['endpoints', 'extended', 'shifted'])

        def perturb(point):
            return to_float32(
                (point[0] + random.randint(-3, 3) * ulp, point[1] + random.randint(-3, 3) * ulp)
            )

        if mode == 'endpoints':
            c, d = perturb(a), perturb(b)
        elif mode == 'extended':
            before, after = random.uniform(-0.3, 0.3) * length, random.uniform(-0.3, 0.3) * length
            c = perturb((a[0] - before * direction[0], a[1] - before * direction[1]))
            d = perturb((b[0] + after * direction[0], b[1] + after * direction[1]))
        else:
            shift = random.uniform(0.1, 0.6) * length
            c = perturb((a[0] + shift * direction[0], a[1] + shift * direction[1]))
            d = perturb((b[0] + shift * direction[0], b[1] + shift * direction[1]))
        side = random.choice([-1, 1])
        right = [
            c,
            d,
            to_float32((d[0] + side * 5 * normal[0], d[1] + side * 5 * normal[1])),
            to_float32((c[0] + side * 5 * normal[0], c[1] + side * 5 * normal[1])),
        ]
        left_polygon, right_polygon = Polygon(left), Polygon(right)
        if not (left_polygon.is_valid and right_polygon.is_valid):
            continue
        pairs.append(
            {
                'left': left,
                'right': right,
                'relate': shapely.relate(left_polygon, right_polygon),
                'mode': mode,
            }
        )
    fixture = {
        'generator': 'geometry-relate-near-parallel.py',
        'shapely': shapely.__version__,
        'geos': shapely.geos_version_string,
        'numpy': np.__version__,
        'seed': SEED,
        'note': 'relate = shapely.relate(Polygon(left), Polygon(right)); rings are open, float32 values.',
        'pairs': pairs,
    }
    with open(os.path.join(HERE, 'fixtures', 'geometry-relate-near-parallel.json'), 'w') as file:
        json.dump(fixture, file, indent=1)
        file.write('\n')


if __name__ == '__main__':
    main()
