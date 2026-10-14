# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Reference HDBSCAN labels and core distances for GPUHierarchicalDensityClustering.

Writes fixtures/spatial-clustering-hdbscan.json next to this script, consumed by
test/gpu-spatial-analysis/spatial-clustering/hierarchical-density-clustering-oracle.ts (and through
it by the node and GPU specs). Run from anywhere with
    <venv>/bin/python -I modules/experimental/test/oracles/spatial-clustering-hdbscan.py
in a venv with numpy and scikit-learn (versions recorded in the fixture).

- Labels: `sklearn.cluster.HDBSCAN(min_cluster_size, min_samples)` with its defaults
  (`cluster_selection_method='eom'`, `allow_single_cluster=False`, `cluster_selection_epsilon=0`,
  Euclidean). The GPU numbers clusters differently, so specs compare partitions: labels are
  canonicalized by first occurrence and noise stays -1.
- Core distances: `NearestNeighbors(n_neighbors=min_samples).kneighbors(X)` distance to the
  `min_samples`-th neighbor, the point itself included (scikit-learn's core distance).
- `spanningTreeWeights`: the sorted merge heights of scikit-learn's single-linkage tree, which
  are the minimum spanning tree's weights and unique even where ties allow several trees.
- Ties: mutual reachability distances tie whenever one core distance dominates several edges, so
  several minimum spanning trees exist. scikit-learn orders tied merges by its own (unstable)
  sort; the GPU orders edges by `(weight, low row, high row)`. Tied merges may then condense
  differently for a few points, so specs require the same spanning-tree weights and an adjusted
  Rand index near 1 rather than identical labels. `test/oracles/spatial-clustering-hdbscan.py`'s
  condensing and excess-of-mass rules, applied to scikit-learn's own tree, were checked to give
  scikit-learn's labels exactly (the GPU formulation; see the model in the specs).
- Inputs are rounded to float32 first.
- `cpuSeconds` records scikit-learn's time on a 100k-point blob scene, the speed bar of the A/B.

Tolerances (stored in `tolerances`): core distances are one float32 distance (1e-5 relative plus
1e-6 absolute); labels are exact up to relabeling.
"""

import json
import os
import platform
import time

import numpy as np
import sklearn
from sklearn.cluster import HDBSCAN
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'fixtures', 'spatial-clustering-hdbscan.json')


def canonical(labels):
    mapping = {}
    result = []
    for label in labels:
        if label < 0:
            result.append(-1)
            continue
        if label not in mapping:
            mapping[label] = len(mapping)
        result.append(mapping[label])
    return result


def condense_and_select(tree, row_count, minimum_cluster_size):
    """The GPU's condensing and excess-of-mass rules applied to a scikit-learn single-linkage tree.

    A node of at least `minimum_cluster_size` points starts a cluster when it is the root or its
    parent is a true split; a point falls out of its cluster at its lowest such node; stability is
    the condensed-tree sum; clusters are selected in node order (children first). Returns labels.
    """
    node_count = 2 * row_count - 1
    parent = [-1] * node_count
    size = [1] * row_count + [0] * (row_count - 1)
    weight = [0.0] * node_count
    children = [None] * node_count
    for rank, row in enumerate(tree):
        node = row_count + rank
        left, right = int(row['left_node']), int(row['right_node'])
        children[node] = (left, right)
        size[node] = int(row['cluster_size'])
        weight[node] = float(row['value'])
        parent[left] = node
        parent[right] = node

    def density(node):
        return 1.0 / weight[node] if weight[node] > 0 else np.inf

    def is_big(node):
        return node >= 0 and size[node] >= minimum_cluster_size

    def is_start(node):
        if not is_big(node):
            return False
        up = parent[node]
        return up < 0 or (is_big(children[up][0]) and is_big(children[up][1]))

    def cluster_start(node):
        while not is_start(node):
            node = parent[node]
        return node

    def lowest_big(node):
        while node >= 0 and not is_big(node):
            node = parent[node]
        return node

    ids = {}
    for node in range(node_count):
        if is_start(node):
            ids[node] = len(ids)
    count = len(ids)
    birth = [0.0] * count
    parent_cluster = [-1] * count
    for node, cluster in ids.items():
        if parent[node] >= 0:
            birth[cluster] = density(parent[node])
            parent_cluster[cluster] = ids[cluster_start(parent[node])]
    stability = [0.0] * count
    low = [-1] * row_count
    for row in range(row_count):
        fall = lowest_big(row)
        if fall < 0:
            continue
        cluster = ids[cluster_start(fall)]
        low[row] = cluster
        stability[cluster] += density(fall) - birth[cluster]
    for node, cluster in ids.items():
        if parent_cluster[cluster] >= 0:
            stability[parent_cluster[cluster]] += size[node] * (birth[cluster] - birth[parent_cluster[cluster]])
    child_best = [0.0] * count
    selected = [False] * count
    for cluster in range(count):
        if parent_cluster[cluster] < 0:
            continue
        best = stability[cluster]
        if child_best[cluster] > stability[cluster]:
            best = child_best[cluster]
        else:
            selected[cluster] = True
        child_best[parent_cluster[cluster]] += best
    representative = [-1] * count
    for cluster in reversed(range(count)):
        inherited = representative[parent_cluster[cluster]] if parent_cluster[cluster] >= 0 else -1
        if inherited >= 0:
            selected[cluster] = False
            representative[cluster] = inherited
        elif selected[cluster]:
            representative[cluster] = cluster
    return [representative[low[row]] if low[row] >= 0 and representative[low[row]] >= 0 else -1 for row in range(row_count)]


def scene(name, points, minimum_cluster_size, minimum_samples=None):
    points = np.asarray(points, dtype=np.float32).astype(np.float64)
    samples = minimum_samples if minimum_samples is not None else minimum_cluster_size
    model = HDBSCAN(min_cluster_size=minimum_cluster_size, min_samples=minimum_samples)
    labels = model.fit_predict(points)
    # Self-check: the GPU's condensing and selection rules give scikit-learn's labels on its tree.
    assert canonical(condense_and_select(model._single_linkage_tree_, len(points), minimum_cluster_size)) == canonical(labels.tolist()), name
    # The minimum spanning tree's sorted weights are unique even when ties allow several trees.
    tree_weights = sorted(float(value) for value in model._single_linkage_tree_['value'])
    neighbors = NearestNeighbors(n_neighbors=min(samples, len(points))).fit(points)
    distances, _ = neighbors.kneighbors(points)
    core = distances[:, -1] if len(points) >= samples else np.full(len(points), np.inf)
    return {
        'name': name,
        'minimumClusterSize': minimum_cluster_size,
        'minimumSamples': samples,
        'positions': points.tolist(),
        'labels': canonical(labels.tolist()),
        'coreDistances': [float(value) if np.isfinite(value) else 'Infinity' for value in core],
        'spanningTreeWeights': [value if np.isfinite(value) else 'Infinity' for value in tree_weights],
    }


def blobs(rng, centers, sizes, sigmas):
    parts = [rng.normal(center, sigma, (size, 2)) for center, size, sigma in zip(centers, sizes, sigmas)]
    return np.concatenate(parts)


def main():
    rng = np.random.default_rng(20261014)
    scenes = []
    density = blobs(rng, [(0, 0), (10, 2), (4, 9), (-6, 8)], [400, 300, 250, 150], [0.6, 1.0, 1.4, 0.4])
    noise = rng.uniform(-12, 16, (200, 2))
    scenes.append(scene('blobs-and-noise', np.concatenate([density, noise]), 15))
    scenes.append(scene('blobs-and-noise-samples-5', np.concatenate([density, noise]), 15, 5))
    nested = np.concatenate([
        rng.normal((0, 0), 3.0, (500, 2)),
        rng.normal((1, 1), 0.3, (120, 2)),
        rng.normal((20, 0), 1.0, (300, 2)),
    ])
    scenes.append(scene('nested', nested, 25, 10))
    t = rng.uniform(0, np.pi, 300)
    moons = np.concatenate([
        np.stack([np.cos(t), np.sin(t)], axis=1),
        np.stack([1 - np.cos(t), 0.5 - np.sin(t)], axis=1),
    ]) + rng.normal(0, 0.05, (600, 2))
    scenes.append(scene('moons', moons, 20, 8))
    scenes.append(scene('tiny', rng.uniform(0, 1, (7, 2)), 10, 3))
    line = np.stack([np.arange(80) * 0.5, np.zeros(80)], axis=1) + rng.normal(0, 1e-3, (80, 2))
    scenes.append(scene('collinear', np.concatenate([line, line + np.array([60.0, 0.0])]), 10, 4))

    big = blobs(rng, [(0, 0), (30, 0), (0, 30), (30, 30)], [25000] * 4, [3, 4, 5, 2])
    start = time.perf_counter()
    HDBSCAN(min_cluster_size=50).fit(big)
    cpu_seconds = time.perf_counter() - start

    fixture = {
        'generator': 'test/oracles/spatial-clustering-hdbscan.py',
        'versions': {
            'python': platform.python_version(),
            'numpy': np.__version__,
            'scikit-learn': sklearn.__version__,
        },
        'tolerances': {'coreRelative': 1e-5, 'coreAbsolute': 1e-6},
        'cpuSeconds': {'blobs100k': cpu_seconds},
        'scenes': scenes,
    }
    with open(OUTPUT, 'w') as file:
        json.dump(fixture, file)
        file.write('\n')
    print(f'wrote {OUTPUT}: {len(scenes)} scenes, scikit-learn 100k in {cpu_seconds:.2f} s')


if __name__ == '__main__':
    main()
