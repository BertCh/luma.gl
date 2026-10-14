#!/usr/bin/env python3
"""Oracle generator for the GPUMaxPRegions region-count fixture.

Reference implementation: spopt.region.MaxPHeuristic (the PySAL Max-P heuristic of Duque, Anselin
and Rey 2012; construction by randomized sequential region growing plus enclave assignment, then
simulated annealing). Nothing about the GPU parallel construction is reimplemented: the fixture
records spopt's region counts, so the GPU construction can be required to reach a stated fraction
of them on the same instances.

spopt's heuristic is randomized, so each instance is solved with SEEDS different random seeds
(numpy.random.seed before every solve; spopt draws from the global numpy state) and the fixture
keeps every region count plus the objective and labels of the best (most regions) run. Instances:
  - rook lattices of 12 x 12, 20 x 20 and 30 x 30 rows, and
  - irregular planar adjacency (Delaunay triangulation of random points, a stand-in for tract-like
    polygons) of 150 and 400 rows,
each with one standardized attribute column and integer extensive values in [1, 5]. Thresholds
are chosen so a region holds about 4, 10 and 20 rows. Every spopt solution is verified here:
every region reaches the threshold and is contiguous under the weights.

Objective: the within-region sum of squared deviations of the standardized column (the quantity
GPUMaxPRegions reports as `objective`), computed here from spopt's labels. spopt runs
max_iterations_sa SA iterations (annealing lowers the objective, never the region count).

Usage (writes fixtures/statistics-max-p.json):
  python3 -I statistics-max-p.py
Needs numpy, pandas, scipy, libpysal and spopt (versions are recorded in the fixture).
"""
import json
import os
import sys
from importlib.metadata import version

import numpy as np
import pandas as pd
from libpysal.weights import W, lat2W
from scipy.spatial import Delaunay
from spopt.region import MaxPHeuristic

SEED = 20261014
SEEDS = 5
SA_ITERATIONS = 20


def to_csr(w):
    offsets = [0]
    neighbors = []
    for row in range(w.n):
        neighbors.extend(sorted(w.neighbors[row]))
        offsets.append(len(neighbors))
    return offsets, neighbors


def delaunay_weights(point_count, rng):
    points = rng.random((point_count, 2))
    triangulation = Delaunay(points)
    neighbors = {row: set() for row in range(point_count)}
    for simplex in triangulation.simplices:
        for a in simplex:
            for b in simplex:
                if a != b:
                    neighbors[int(a)].add(int(b))
    return W({row: sorted(items) for row, items in neighbors.items()})


def is_contiguous(w, members):
    members = set(members)
    stack = [next(iter(members))]
    seen = {stack[0]}
    while stack:
        row = stack.pop()
        for neighbor in w.neighbors[row]:
            if neighbor in members and neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return len(seen) == len(members)


def sum_of_squares(values, labels):
    total = 0.0
    for label in np.unique(labels):
        member_values = values[labels == label]
        total += float(((member_values - member_values.mean()) ** 2).sum())
    return total


def main():
    rng = np.random.default_rng(SEED)
    instances = []
    specs = [
        ('lattice-12', lambda: lat2W(12, 12, rook=True), 4),
        ('lattice-12', lambda: lat2W(12, 12, rook=True), 10),
        ('lattice-20', lambda: lat2W(20, 20, rook=True), 10),
        ('lattice-30', lambda: lat2W(30, 30, rook=True), 4),
        ('lattice-30', lambda: lat2W(30, 30, rook=True), 20),
        ('delaunay-150', lambda: delaunay_weights(150, rng), 10),
        ('delaunay-400', lambda: delaunay_weights(400, rng), 10),
        ('delaunay-400', lambda: delaunay_weights(400, rng), 20)
    ]
    for name, make_weights, rows_per_region in specs:
        w = make_weights()
        n = w.n
        raw = rng.normal(size=n)
        values = (raw - raw.mean()) / raw.std()
        extensive = rng.integers(1, 6, size=n).astype(float)
        threshold = float(np.ceil(extensive.mean() * rows_per_region))
        frame = pd.DataFrame({'value': values, 'extensive': extensive})
        runs = []
        for seed in range(SEEDS):
            np.random.seed(seed)
            model = MaxPHeuristic(
                frame,
                w,
                ['value'],
                'extensive',
                threshold,
                top_n=2,
                max_iterations_construction=99,
                max_iterations_sa=SA_ITERATIONS,
                verbose=False
            )
            model.solve()
            labels = np.array(model.labels_)
            sums = [extensive[labels == label].sum() for label in np.unique(labels)]
            assert min(sums) >= threshold, 'spopt region below the threshold'
            for label in np.unique(labels):
                assert is_contiguous(w, np.flatnonzero(labels == label)), 'spopt region not contiguous'
            runs.append(
                {
                    'seed': seed,
                    'p': int(model.p),
                    'objective': sum_of_squares(values, labels),
                    'labels': [int(label) for label in labels]
                }
            )
        offsets, neighbors = to_csr(w)
        best = max(runs, key=lambda run: (run['p'], -run['objective']))
        instances.append(
            {
                'name': f'{name}-{rows_per_region}-rows-per-region',
                'rows': n,
                'offsets': offsets,
                'neighbors': neighbors,
                'values': [float(value) for value in values],
                'extensive': [float(value) for value in extensive],
                'threshold': threshold,
                'upperBound': int(np.floor(extensive.sum() / threshold)),
                'spoptRegionCounts': [run['p'] for run in runs],
                'spoptBestRegionCount': best['p'],
                'spoptBestObjective': best['objective'],
                'spoptBestLabels': best['labels']
            }
        )
        print(name, rows_per_region, [run['p'] for run in runs], file=sys.stderr)
    fixture = {
        'generator': 'test/oracles/statistics-max-p.py',
        'versions': {
            'spopt': version('spopt'),
            'libpysal': version('libpysal'),
            'numpy': version('numpy'),
            'scipy': version('scipy'),
            'pandas': version('pandas')
        },
        'seed': SEED,
        'saIterations': SA_ITERATIONS,
        'instances': instances
    }
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures', 'statistics-max-p.json')
    with open(path, 'w') as handle:
        json.dump(fixture, handle, separators=(',', ':'), allow_nan=False)
        handle.write('\n')


if __name__ == '__main__':
    main()
