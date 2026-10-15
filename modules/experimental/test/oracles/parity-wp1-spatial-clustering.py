# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent DBSCAN reference: sklearn.cluster.DBSCAN, with indexed neighborhoods.

Run <reference-venv>/bin/python -I <this script> [--output-dir DIRECTORY].
All 43,557 Chicago nature observations are transformed EPSG:4326 -> EPSG:26916
(always_xy), shifted by (440000, 4600000) metres and snapped to a declared 0.25 m
coordinate grid before float32 conversion. No rows are sampled or deduplicated.
This tests native planar DBSCAN, not great-circle DBSCAN or projection accuracy.
The snapped local coordinates keep the complete fixture below 1.5 MB and avoid
large-coordinate subtraction error. Source and upload-byte hashes are recorded.

sklearn includes self in min_samples, uses closed Euclidean epsilon balls, and
counts duplicate observations separately. All points are finite/in bounds; no
weights or mask. Canonical IDs rank clusters by their smallest CORE input row.
A non-core point adjacent to multiple core components may join any such component;
all allowed IDs are computed with scipy.spatial.cKDTree, not our TS/WGSL.
The full dataset uses an off-grid squared radius; a separate exact, representable
synthetic boundary fixture tests closed-ball semantics and ambiguous borders.
No RNG, permutations, CRS wrap, or nodata. Profile E: masks/partitions exact.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as numpy
import pyproj
import scipy
from scipy.spatial import cKDTree
import sklearn
from sklearn.cluster import DBSCAN

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / 'examples/deck/spatial-analysis-showcase/public/data/chicago-nature'


def makeCase(name, positions, epsilon, minimumPoints, provenance):
    positions = numpy.asarray(positions, dtype=numpy.float32).reshape(-1, 2)
    coordinates = positions.astype(numpy.float64)
    model = DBSCAN(eps=epsilon, min_samples=minimumPoints, algorithm='kd_tree').fit(coordinates)
    core = numpy.zeros(len(positions), dtype=numpy.uint32)
    core[model.core_sample_indices_] = 1
    roots = sorted((int(numpy.flatnonzero((model.labels_ == label) & (core == 1))[0]), int(label))
                   for label in set(model.labels_) if label >= 0)
    mapping = {label: canonical for canonical, (_, label) in enumerate(roots)}
    labels = [mapping.get(int(label), -1) for label in model.labels_]
    tree = cKDTree(coordinates)
    ambiguous = {}
    for row in numpy.flatnonzero((core == 0) & (model.labels_ >= 0)):
        neighbors = tree.query_ball_point(coordinates[row], epsilon)
        allowed = sorted({labels[other] for other in neighbors if core[other]})
        assert labels[row] in allowed
        if len(allowed) > 1:
            ambiguous[str(row)] = allowed
    # On the 0.25 m grid, differences are exact (coordinates < 2^20),
    # and squared distances near this radius are multiples of 1/16 below
    # 2^20: subtraction, squares and addition are all exactly representable
    # in f32. epsilon=125.125 and its square are also exact, but epsilon^2
    # is NOT a multiple of 1/16. Thus no f32 threshold ambiguity exists.
    # This tiny shell checks only sklearn's float64 distance evaluation.
    assert numpy.abs(positions).max() < 2**20
    assert numpy.all(positions * 4 == numpy.rint(positions * 4))
    distanceGuard = 1e-7
    outer = tree.query_ball_point(coordinates, epsilon + distanceGuard, return_length=True)
    inner = tree.query_ball_point(coordinates, epsilon - distanceGuard, return_length=True)
    shellPairs = int(numpy.sum(outer - inner) // 2)
    if name == 'chicago-nature-full':
        # A pair at a threshold is separately enumerated, never silently excluded.
        # Fail generation if the chosen real-data radius is not unambiguous.
        assert shellPairs == 0, (epsilon, distanceGuard, shellPairs)
    return {
        'name': name,
        'reference': 'sklearn.cluster.DBSCAN',
        'n': len(positions),
        'positions': positions.ravel().tolist(),
        'bounds': [float(positions[:, 0].min() - 1), float(positions[:, 1].min() - 1),
                   float(positions[:, 0].max() + 1), float(positions[:, 1].max() + 1)],
        'epsilon': epsilon,
        'minimumPoints': minimumPoints,
        'labels': labels,
        'coreMask': core.tolist(),
        'clusterCount': len(roots),
        'allowedBorderLabels': ambiguous,
        'distanceGuard': distanceGuard,
        'boundaryShellPairs': shellPairs,
        'inputFloat32Sha256': hashlib.sha256(positions.astype('<f4').tobytes()).hexdigest(),
        'provenance': provenance,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', dest='outputDirectory', type=Path, default=Path(__file__).parent / 'fixtures')
    arguments = parser.parse_args()
    source = DATA / 'position.bin'
    sourceBytes = source.read_bytes()
    geographic = numpy.frombuffer(sourceBytes, dtype='<f4').reshape(-1, 2)
    transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:26916', always_xy=True)
    projected = numpy.column_stack(transformer.transform(geographic[:, 0], geographic[:, 1]))
    local = numpy.rint((projected - [440000, 4600000]) * 4) / 4
    provenance = {
        'source': str(source.relative_to(ROOT)),
        'sourceSha256': hashlib.sha256(sourceBytes).hexdigest(),
        'manifestSha256': hashlib.sha256((DATA / 'manifest.json').read_bytes()).hexdigest(),
        'sourceCount': len(geographic), 'selection': 'all rows in original order',
        'sourceCrs': 'EPSG:4326', 'targetCrs': 'EPSG:26916',
        'origin': [440000, 4600000], 'snapGridMeters': 0.25,
        'attribution': 'iNaturalist contributors; individual CC0, CC BY 4.0 or CC BY-NC 4.0; non-commercial use',
    }
    cases = [makeCase('chicago-nature-full', local, 125.125, 8, provenance)]
    cases.append(makeCase('duplicates-closed-boundary-shared-border', [
        [-1, 0], [-1.75, 0], [-1.75, 0], [-1.75, 0],
        [1, 0], [1.75, 0], [1.75, 0], [1.75, 0], [0, 0], [5, 5],
    ], 1, 4, {'source': 'synthetic; exact dyadic coordinates, shared border row 8'}))
    cases.append(makeCase('minimum-one-duplicates-and-island', [[0, 0], [0, 0], [2, 0]],
                           0.5, 1, {'source': 'synthetic'}))
    cases.append(makeCase('all-noise', [[0, 0], [0.5, 0], [2, 0]],
                           1, 4, {'source': 'synthetic; min_samples exceeds row count'}))
    fixture = {
        'generator': Path(__file__).name,
        'versions': {'python': platform.python_version(), 'numpy': numpy.__version__,
                     'scipy': scipy.__version__, 'sklearn': sklearn.__version__, 'pyproj': pyproj.__version__},
        'tolerance': {'profile': 'E', 'absolute': 0, 'relative': 0},
        'cases': cases,
    }
    arguments.outputDirectory.mkdir(parents=True, exist_ok=True)
    output = arguments.outputDirectory / 'parity-wp1-spatial-clustering.json'
    serialized = json.dumps(fixture, separators=(',', ':'), allow_nan=False) + '\n'
    assert len(serialized.encode()) <= 1500000
    output.write_text(serialized)
    print(output, len(serialized), 'bytes')
    for case in cases:
        print(case['name'], case['n'], 'rows;', case['clusterCount'], 'clusters;',
              len(case['allowedBorderLabels']), 'ambiguous borders')


if __name__ == '__main__':
    main()
