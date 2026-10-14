# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Reference Ward agglomeration for GPUWardRegions: scipy.cluster.hierarchy and sklearn.

Writes fixtures/statistics-ward.json next to this script, consumed by
test/gpu-spatial-analysis/spatial-regionalization/gpu-ward-regions.spec.ts and
ward-mutual-nearest.node.spec.ts. Run from anywhere with
    <venv>/bin/python -I modules/experimental/test/oracles/statistics-ward.py
in a venv with numpy, scipy and scikit-learn (versions are recorded in the fixture).

Scenes (raw values, `standardize: false`; labels are canonical: the smallest row of each cluster):
- `separated`: complete adjacency, 28 rows in 4 tight clusters of 7 that pair up into 2 far-apart
  groups (rows shuffled). Constraint-free Ward is reducible and the cuts k = 4, 2, 1 are natural
  (every merge below the cut is cheaper than every merge above it), so the sequential schedule and
  the mutual-nearest schedule both equal `scipy.cluster.hierarchy.ward` + `fcluster(maxclust=k)`.
- `random`: complete adjacency, 40 uniform points, k = 5. Sequential equals scipy; the
  mutual-nearest schedule is not expected to (its cut is not natural).
- `lattice`: a 12 x 12 rook lattice with 2 random columns, k = 10, sklearn
  `AgglomerativeClustering(linkage='ward', connectivity=...)`. Sequential equals it.

`marginRatio` is the smallest relative gap between the cheapest and the second-cheapest adjacent
pair over the merges up to the cut, from an independent float64 greedy; seeds are chosen so it is
above `tolerances.marginRatio` and float32 cost rounding cannot flip a merge. `objective` is the
within-cluster sum of squared deviations of the labels.
"""

import json
import math
import os
import platform

import numpy as np
import scipy
import sklearn
from scipy.cluster.hierarchy import fcluster, ward
from sklearn.cluster import AgglomerativeClustering
from scipy.sparse import coo_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'fixtures', 'statistics-ward.json')
# float32 costs carry about 1e-6 relative error (moment sums of at most 150 values), so a 1e-4
# relative gap between the chosen and the runner-up merge cannot flip under float32.
MINIMUM_MARGIN_RATIO = 1e-4


def canonical(labels):
    first = {}
    for row, label in enumerate(labels):
        first.setdefault(label, row)
    return [first[label] for label in labels]


def within_ssd(values, labels):
    total = 0.0
    for label in set(labels):
        members = values[[i for i, l in enumerate(labels) if l == label]]
        total += float(((members - members.mean(axis=0)) ** 2).sum())
    return total


def greedy_margin(values, adjacency, cluster_count):
    """Independent float64 sequential Ward; returns (labels, smallest relative best/second gap)."""
    count = len(values)
    labels = list(range(count))
    size = {i: 1.0 for i in range(count)}
    total = {i: values[i].copy() for i in range(count)}
    neighbors = {i: set(adjacency[i]) for i in range(count)}
    margin = math.inf
    while len(size) > cluster_count:
        costs = []
        for a in size:
            for b in neighbors[a]:
                if a < b:
                    mean_difference = total[a] / size[a] - total[b] / size[b]
                    costs.append((size[a] * size[b] / (size[a] + size[b]) * float(mean_difference @ mean_difference), a, b))
        costs.sort()
        best = costs[0]
        if len(costs) > 1:
            margin = min(margin, (costs[1][0] - best[0]) / max(costs[1][0], 1e-300))
        _, a, b = best
        size[a] += size.pop(b)
        total[a] = total[a] + total.pop(b)
        neighbors[a] = (neighbors[a] | neighbors.pop(b)) - {a, b}
        for other in neighbors[a]:
            neighbors[other] = (neighbors[other] - {b}) | {a}
        labels = [a if label == b else label for label in labels]
    return canonical(labels), margin


def scene(values, adjacency, cut_counts, reference):
    result = {'rowCount': len(values), 'columnCount': values.shape[1], 'values': values.flatten().tolist(), 'cuts': []}
    for k in cut_counts:
        labels = canonical(reference(k))
        greedy_labels, margin = greedy_margin(values, adjacency, k)
        assert greedy_labels == labels, ('greedy disagrees with the reference', k)
        result['cuts'].append({
            'targetRegionCount': k,
            'labels': labels,
            'objective': within_ssd(values, labels),
            'marginRatio': margin
        })
        assert margin > MINIMUM_MARGIN_RATIO, ('merge margin too small', k, margin)
    return result


def complete_adjacency(count):
    return [[j for j in range(count) if j != i] for i in range(count)]


def scipy_reference(values):
    linkage = ward(values)
    return lambda k: fcluster(linkage, k, criterion='maxclust').tolist()


rng = np.random.default_rng(20261014)

centers = np.array([[0.0, 0.0], [5.0, 0.0], [40.0, 40.0], [45.0, 40.0]])
separated = np.concatenate([center + rng.normal(0, 0.3, (7, 2)) for center in centers])
separated = separated[rng.permutation(len(separated))]
separated_scene = scene(separated, complete_adjacency(len(separated)), [4, 2, 1], scipy_reference(separated))

random_values = rng.uniform(0, 10, (40, 2))
random_scene = scene(random_values, complete_adjacency(40), [5], scipy_reference(random_values))

WIDTH = 12
count = WIDTH * WIDTH
lattice_adjacency = []
for row in range(count):
    x = row % WIDTH
    lattice_adjacency.append([c for c in (row - WIDTH, row - 1 if x > 0 else -1, row + 1 if x < WIDTH - 1 else -1, row + WIDTH) if 0 <= c < count])
rows = [i for i in range(count) for _ in lattice_adjacency[i]]
columns = [j for i in range(count) for j in lattice_adjacency[i]]
connectivity = coo_matrix((np.ones(len(rows)), (rows, columns)), shape=(count, count))
lattice_values = rng.uniform(0, 10, (count, 2))
lattice_scene = scene(
    lattice_values,
    lattice_adjacency,
    [10],
    lambda k: AgglomerativeClustering(n_clusters=k, linkage='ward', connectivity=connectivity).fit_predict(lattice_values).tolist()
)
lattice_scene['width'] = WIDTH

fixture = {
    'generator': 'statistics-ward.py',
    'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'scikit-learn': sklearn.__version__},
    'tolerances': {'marginRatio': MINIMUM_MARGIN_RATIO, 'objectiveRelative': 1e-4},
    'separated': separated_scene,
    'random': random_scene,
    'lattice': lattice_scene
}
with open(OUTPUT, 'w') as handle:
    json.dump(fixture, handle, indent=1)
    handle.write('\n')
print('wrote', OUTPUT, [(name, [(c['targetRegionCount'], c['marginRatio']) for c in fixture[name]['cuts']]) for name in ('separated', 'random', 'lattice')])
