#!/usr/bin/env python3
"""Oracle generator for GPUNeighborSearch on clustered, duplicate, collinear and invalid points.

Reference: scipy.spatial.cKDTree (float64). `query` gives the k-th distance of a row and
`query_ball_point` enumerates every target within it (or within the radius), so a tie at the k-th
distance is resolved here, as in the GPU definition, by `(squared distance, lowest id)`: scipy alone
returns tied neighbors in unspecified order.

Coordinates are integers below 2048, so float32 holds them exactly and every squared distance
(at most 2 * 2047^2 < 2^24) is an exact integer in both float32 and float64: the GPU and scipy
distances are identical up to the final sqrt, and equal distances are real ties, not rounding noise.
The generator asserts that scipy's neighbor sets equal a brute-force numpy evaluation of the
definition, so the fixture is the definition checked twice.

Scenes (seed 20261014), bounds [0, 0, 2047, 2047]:
- `clustered`: 600 points, 90% in a 24 x 24 integer box (about 2 points per lattice position, so
  many exact duplicates and tied distances), 10% uniform; 6 NaN rows, 6 outside the bounds, every
  seventh row masked out;
- `collinear`: 300 points on y = 100 with x = i // 2 (duplicates and exactly tied neighbors on a line);
- `cross`: the clustered targets queried by 120 other points (some NaN and outside the bounds).
Per scene and k in (1, 4, 32), `knn` holds the CSR; per radius, `radius` holds the CSR. Neighbors of a
row ascend by id; `distancesSquared` are the exact integer squared distances. Invalid rows (NaN, outside
the bounds, masked) are never targets and have empty rows as queries.

Usage (writes fixtures/geometry-neighbor-search-clustered.json):  python3 -I geometry-neighbor-search-clustered.py
Needs numpy and scipy (versions recorded in the fixture).
"""
import json
import math
import os
from importlib.metadata import version

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
GENERATOR = 'modules/experimental/test/oracles/geometry-neighbor-search-clustered.py'
BOUNDS = [0, 0, 2047, 2047]
KS = (1, 4, 32)


def is_valid(points, mask=None):
    x, y = points[:, 0], points[:, 1]
    inside = (
        np.isfinite(x) & np.isfinite(y) & (x >= BOUNDS[0]) & (x <= BOUNDS[2]) & (y >= BOUNDS[1]) & (y <= BOUNDS[3])
    )
    return inside if mask is None else inside & (mask != 0)


def finish_rows(rows, distances_squared):
    offsets = [0]
    neighbors, squared = [], []
    for ids, d2 in zip(rows, distances_squared):
        order = np.argsort(ids, kind='stable')
        neighbors.extend(int(ids[i]) for i in order)
        squared.extend(int(d2[i]) for i in order)
        offsets.append(len(neighbors))
    return {'offsets': offsets, 'neighbors': neighbors, 'distancesSquared': squared}


def brute_force_row(targets, target_ids, query, k, radius, skip):
    d2 = (targets[:, 0] - query[0]) ** 2 + (targets[:, 1] - query[1]) ** 2
    keep = target_ids != skip
    ids, d2 = target_ids[keep], d2[keep]
    if radius is not None:
        within = d2 <= radius * radius
        ids, d2 = ids[within], d2[within]
    order = np.lexsort((ids, d2))
    if k is not None:
        order = order[:k]
    return ids[order], d2[order]


def solve(positions, mask, queries, self_join, k=None, radius=None):
    """CSR of the k nearest (or radius) valid targets of every valid query, via cKDTree."""
    valid = is_valid(positions, mask)
    target_ids = np.flatnonzero(valid)
    targets = positions[target_ids]
    tree = cKDTree(targets)
    query_valid = valid if self_join else is_valid(queries)
    rows, squared = [], []
    for row in range(len(queries)):
        if not query_valid[row]:
            rows.append(np.empty(0, dtype=np.int64))
            squared.append(np.empty(0, dtype=np.int64))
            continue
        query = queries[row]
        skip = row if self_join else -1
        # Ask for k + 1 so a self match does not displace a neighbor, then enumerate the tie class.
        if k is not None:
            count = min(k + (1 if self_join else 0), len(targets))
            distance, _ = tree.query(query, k=count)
            reach = float(np.max(np.atleast_1d(distance)))
        else:
            reach = float(radius)
        found = np.array(sorted(tree.query_ball_point(query, reach * (1 + 1e-12) + 1e-9)), dtype=np.int64)
        ids = target_ids[found] if len(found) else np.empty(0, dtype=np.int64)
        d2 = (positions[ids, 0] - query[0]) ** 2 + (positions[ids, 1] - query[1]) ** 2
        keep = ids != skip
        ids, d2 = ids[keep], d2[keep]
        if radius is not None:
            within = d2 <= radius * radius
            ids, d2 = ids[within], d2[within]
        order = np.lexsort((ids, d2))
        if k is not None:
            order = order[:k]
        ids, d2 = ids[order], d2[order]
        expected_ids, expected_d2 = brute_force_row(targets, target_ids, query, k, radius, skip)
        assert np.array_equal(ids, expected_ids) and np.array_equal(d2, expected_d2), (row, k, radius)
        rows.append(ids)
        squared.append(d2)
    return finish_rows(rows, squared)


def encode(array):
    return [None if not math.isfinite(value) else int(value) for value in array.ravel()]


def make_scene(name, positions, mask, queries, radii):
    self_join = queries is None
    query_points = positions if self_join else queries
    scene = {
        'name': name,
        'selfJoin': self_join,
        'positions': encode(positions),
        'mask': None if mask is None else [int(value) for value in mask],
        'queryPositions': None if self_join else encode(queries),
        'knn': {},
        'radius': {},
    }
    for k in KS:
        scene['knn'][str(k)] = solve(positions, mask, query_points, self_join, k=k)
    for radius in radii:
        scene['radius'][str(radius)] = solve(positions, mask, query_points, self_join, radius=radius)
    return scene


def clustered_points(rng, count):
    dense = int(count * 0.9)
    box = rng.integers(900, 924, size=(dense, 2))
    spread = rng.integers(0, 2048, size=(count - dense, 2))
    points = np.vstack([box, spread]).astype(np.float64)
    return points[rng.permutation(count)]


def main():
    rng = np.random.default_rng(20261014)
    clustered = clustered_points(rng, 600)
    for row in rng.choice(600, 12, replace=False)[:6]:
        clustered[row] = [np.nan, 5.0] if row % 2 else [7.0, np.nan]
    for row in rng.choice(600, 12, replace=False)[:6]:
        clustered[row] = [-5.0, 100.0] if row % 2 else [3000.0, 20.0]
    mask = np.array([0 if row % 7 == 3 else 1 for row in range(600)], dtype=np.uint32)
    scenes = [make_scene('clustered', clustered, mask, None, (2, 5))]

    line = np.array([[i // 2, 100] for i in range(300)], dtype=np.float64)
    scenes.append(make_scene('collinear', line, None, None, (2, 4)))

    queries = clustered_points(rng, 120)
    queries[5] = [np.nan, np.nan]
    queries[9] = [-1.0, 4.0]
    scenes.append(make_scene('cross', clustered, mask, queries, (2, 5)))

    fixture = {
        'generator': GENERATOR,
        'versions': {'scipy': version('scipy'), 'numpy': version('numpy')},
        'tolerances': {
            'neighbors': 'exact ids and offsets (integer coordinates, exact float32 squared distances, ties by lowest id)',
            'distances': 'relative 2.4e-7 (two ulp) on sqrt of an exact integer squared distance',
        },
        'bounds': BOUNDS,
        'scenes': scenes,
    }
    path = os.path.join(HERE, 'fixtures', 'geometry-neighbor-search-clustered.json')
    with open(path, 'w') as handle:
        json.dump(fixture, handle)
    print('wrote', path, os.path.getsize(path), 'bytes')


if __name__ == '__main__':
    main()
