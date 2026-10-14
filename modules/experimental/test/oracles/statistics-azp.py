#!/usr/bin/env python3
"""Oracle generator for the GPUAZPRegions quality fixture.

Reference implementation: spopt.region.AZP (the PySAL automatic zoning procedure of Openshaw and
Rao, with spopt's default pairwise objective and move rule), started from the same contiguous
initial labels as the GPU contributor. Nothing about the GPU batched search is reimplemented: the
fixture records spopt's final labels and the within-region sum of squared deviations (the quantity
GPUAZPRegions reports as `objective`) of the initial and of spopt's final partition, so the GPU
result can be required to land within a stated factor of spopt's.

Both searches are local searches, so they reach different local optima; only quality is comparable,
never labels. Instances: rook lattices of 10 x 10, 14 x 14 and 18 x 18 rows with one standardized
attribute column; every initial partition is grown by randomized multi-source breadth-first search
(contiguous by construction); two starts per lattice. spopt's result is verified contiguous.

Usage (writes fixtures/statistics-azp.json):
  python3 -I statistics-azp.py
Needs numpy, pandas, libpysal and spopt (versions are recorded in the fixture).
"""
import json
import os
import sys
from importlib.metadata import version

import numpy as np
import pandas as pd
from libpysal.weights import lat2W
from spopt.region import AZP

SEED = 20261014


def to_csr(w):
    offsets = [0]
    neighbors = []
    for row in range(w.n):
        neighbors.extend(sorted(w.neighbors[row]))
        offsets.append(len(neighbors))
    return offsets, neighbors


def grow_partition(w, region_count, rng):
    labels = -np.ones(w.n, dtype=int)
    seeds = rng.choice(w.n, size=region_count, replace=False)
    frontiers = []
    for region, row in enumerate(seeds):
        labels[row] = region
        frontiers.append([int(row)])
    remaining = w.n - region_count
    while remaining > 0:
        region = int(rng.integers(region_count))
        open_rows = [
            (row, neighbor)
            for row in frontiers[region]
            for neighbor in w.neighbors[row]
            if labels[neighbor] == -1
        ]
        if not open_rows:
            continue
        row, neighbor = open_rows[int(rng.integers(len(open_rows)))]
        labels[neighbor] = region
        frontiers[region].append(int(neighbor))
        remaining -= 1
    return labels


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
    for width, region_count in [(10, 6), (14, 9), (18, 12)]:
        w = lat2W(width, width, rook=True)
        for start in range(2):
            raw = rng.normal(size=w.n)
            # A smooth trend plus noise, so contiguous regions have something to find.
            trend = np.array([(row % width) + (row // width) for row in range(w.n)], dtype=float)
            raw = raw + 0.4 * trend
            values = (raw - raw.mean()) / raw.std()
            initial = grow_partition(w, region_count, rng)
            frame = pd.DataFrame({'value': values})
            model = AZP(frame, w, ['value'], n_clusters=region_count, random_state=start,
                        initial_labels=initial.copy())
            model.solve()
            labels = np.array(model.labels_)
            assert len(np.unique(labels)) == region_count
            for label in np.unique(labels):
                assert is_contiguous(w, np.flatnonzero(labels == label)), 'spopt region not contiguous'
            offsets, neighbors = to_csr(w)
            instances.append(
                {
                    'name': f'lattice-{width}-{region_count}-regions-start-{start}',
                    'rows': w.n,
                    'regionCount': region_count,
                    'offsets': offsets,
                    'neighbors': neighbors,
                    'values': [float(value) for value in values],
                    'initialLabels': [int(label) for label in initial],
                    'initialObjective': sum_of_squares(values, initial),
                    'spoptLabels': [int(label) for label in labels],
                    'spoptObjective': sum_of_squares(values, labels)
                }
            )
            print(instances[-1]['name'], instances[-1]['initialObjective'],
                  instances[-1]['spoptObjective'], file=sys.stderr)
    fixture = {
        'generator': 'test/oracles/statistics-azp.py',
        'versions': {
            'spopt': version('spopt'),
            'libpysal': version('libpysal'),
            'numpy': version('numpy'),
            'pandas': version('pandas')
        },
        'seed': SEED,
        'instances': instances
    }
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures', 'statistics-azp.json')
    with open(path, 'w') as handle:
        json.dump(fixture, handle, separators=(',', ':'), allow_nan=False)
        handle.write('\n')


if __name__ == '__main__':
    main()
