# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Partial GPUPointPatternIndices parity: rectangle, correction='none' only.

Run with the reference venv's python -I. Use all 105808 Chicago places, without
subsampling or losing cross-window neighbors. Source EPSG:4326 float32 lon/lat
are transformed by pyproj to EPSG:26916 metres, translated by [400000,4600000],
then rounded to f32. Reference distances use those exact rounded coordinates
in float64. The window is a fixed f32 rectangle enclosing all projected points.
No RNG, weights, or permutation inference. All three significance tails run.

pointpats.PointPattern.nnd / mean_nnd supplies distances/mean; scipy.cKDTree
supplies neighbor candidates and exact tie sets, resolved by lowest source row.
Every nearest distance and ID is compared, not just the mean. For non-exact
f64 ties whose first two distances could reverse under f32 arithmetic, the
fixture records candidate IDs separately; the spec reports candidate-set
membership explicitly, with strict IDs on every other row. This 8-epsilon
screen bounds two coordinate subtractions, two squares, addition and sqrt;
only these enumerated rows admit an alternative. Distances still use S.

Clark-Evans (1954), Ecology 35:445-453, doi:10.2307/1931034: homogeneous planar
Poisson nearest distance has a Rayleigh distribution with scale
1/sqrt(2*pi*intensity). scipy.stats.rayleigh.mean/std gives the CSR expectation
and standard error; this uses the exact constant sqrt((4-pi)/(4*pi)), whereas
the API rounds it to 0.26136 (relative difference < 7e-6, within S, not an
increased bound). scipy.stats.norm gives the significance of the standardized
mean. numpy.histogram2d gives quadrats with last-bin inclusive right endpoints,
transposed to GPU row-major y/x. scipy.stats.chisquare/chi2 gives the quadrat
statistic/tails; sample variance uses ddof=1. All undefined masks are explicit.

Synthetic cases include exact duplicate/distance ties, mask and window edges,
nonfinite/outside points, singleton and empty selections, and a 144-quadrat
CSR-like count vector exposing the df>100 Wilson-Hilferty approximation.
No Donnelly, guard, or polygon-corrected result is claimed: R/spatstat is absent.
No local TS/WGSL oracle is used.

Three base64 JSON array payloads retain all f32 inputs, f64 expected distances,
and uint32 IDs while keeping EACH fixture <=1.5 MB. Each payload records dtype,
count, SHA-256, dependency versions, source hashes, and conventions. These are
lossless full arrays, not summaries. Tolerance S, counts/IDs exact E.
"""

import base64
import hashlib
import json
import math
import os
import platform
import tempfile
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'parity-wp6-matplotlib'))

import libpysal
import numpy as np
import pointpats
import pyproj
import scipy
from scipy.spatial import cKDTree
from scipy.stats import chi2, chisquare, norm, rayleigh

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DATA = ROOT / 'examples/deck/spatial-analysis-showcase/public/data/chicago-places'
VERSIONS = {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'pointpats': pointpats.__version__, 'pyproj': pyproj.__version__, 'proj': pyproj.proj_version_str, 'libpysal': libpysal.__version__}
SOURCES = {str((DATA / name).relative_to(ROOT)): hashlib.sha256((DATA / name).read_bytes()).hexdigest() for name in ['manifest.json', 'position.bin']}
CONVENTIONS = {'crs': 'EPSG:4326 -> EPSG:26916; offset [400000,4600000] m; f32 then f64', 'seed': 'none', 'edgeCorrection': 'none', 'bounds': 'inclusive, maximum edge in last quadrat', 'counts': 'all included source rows, duplicates retained', 'ties': 'smallest original row ID among equal distances', 'quadratVarianceDof': 1, 'tolerance': 'S; IDs/counts E', 'notCovered': ['Donnelly', 'guard', 'polygon windows']}


def encode(value):
    if isinstance(value, dict):
        return {key: encode(entry) for key, entry in value.items()}
    if isinstance(value, (np.ndarray, list, tuple)):
        return [encode(entry) for entry in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return 'Infinity' if value > 0 else '-Infinity'
        return float(value)
    return value


def packArray(values, dtype):
    array = np.asarray(values, dtype=dtype).ravel()
    payload = array.tobytes()
    return {'dtype': dtype, 'length': len(array), 'sha256': hashlib.sha256(payload).hexdigest(), 'base64': base64.b64encode(payload).decode('ascii')}


def writeFixture(suffix, data):
    output = HERE / f'fixtures/parity-wp6-point-patterns-{suffix}.json'
    output.write_text(json.dumps(encode({'generator': Path(__file__).name, 'versions': VERSIONS, 'sources': SOURCES, 'conventions': CONVENTIONS, **data}), separators=(',', ':'), allow_nan=False) + '\n')
    assert output.stat().st_size <= 1_500_000
    print('wrote', output.name, output.stat().st_size)


def getTail(distribution, statistic, alternative):
    lower = distribution.cdf(statistic)
    upper = distribution.sf(statistic)
    return upper if alternative == 'greater' else lower if alternative == 'lesser' else min(1, 2 * min(lower, upper))


def computeCase(name, coordinates, bounds, quadratGrid, mask=None):
    coordinates = np.asarray(coordinates, dtype=np.float32).astype(np.float64)
    bounds = np.asarray(bounds, dtype=np.float32).astype(np.float64)
    mask = np.ones(len(coordinates), dtype=np.uint32) if mask is None else np.asarray(mask, dtype=np.uint32)
    included = (mask != 0) & np.all(np.isfinite(coordinates), axis=1) & np.all(coordinates >= bounds[:2], axis=1) & np.all(coordinates <= bounds[2:], axis=1)
    sourceRows = np.flatnonzero(included)
    points = coordinates[included]
    count = len(points)
    distances = np.full(len(coordinates), np.nan)
    neighborIds = np.full(len(coordinates), 0xffffffff, dtype=np.uint32)
    ambiguous = []
    if count > 1:
        pattern = pointpats.PointPattern(points)
        referenceDistances = np.asarray(pattern.nnd).ravel()
        distances[sourceRows] = referenceDistances
        tree = cKDTree(points)
        nearestDistances, nearestIndices = tree.query(points, k=min(3, count))
        np.testing.assert_allclose(referenceDistances, nearestDistances[:, 1], atol=1e-12, rtol=1e-12)
        # Query all equidistant candidates so cKDTree's arbitrary tie ordering is irrelevant.
        candidates = tree.query_ball_point(points, np.nextafter(referenceDistances, np.inf))
        for index, neighbors in enumerate(candidates):
            others = np.asarray([other for other in neighbors if other != index], dtype=int)
            assert len(others)
            candidateDistances = np.linalg.norm(points[others] - points[index], axis=1)
            minimum = candidateDistances.min()
            neighborIds[sourceRows[index]] = sourceRows[others[candidateDistances == minimum]].min()
            if count > 2 and 0 < nearestDistances[index, 2] - minimum <= 8 * np.finfo(np.float32).eps * minimum:
                ambiguous.append({'row': int(sourceRows[index]), 'candidate': int(sourceRows[nearestIndices[index, 2]]), 'distanceGap': nearestDistances[index, 2] - minimum})
        observed = pattern.mean_nnd
    else:
        observed = math.nan
    area = np.prod(bounds[2:] - bounds[:2])
    quadratCount = int(np.prod(quadratGrid))
    counts = np.histogram2d(points[:, 0], points[:, 1], bins=quadratGrid, range=[[bounds[0], bounds[2]], [bounds[1], bounds[3]]])[0].T.ravel().astype(np.uint32)
    assert counts.sum() == count
    mean = counts.mean()
    variance = counts.var(ddof=1) if quadratCount > 1 else math.nan
    ratio = variance / mean if count else math.nan
    statistic = chisquare(counts).statistic if count and quadratCount > 1 else math.nan
    runs = []
    for alternative in ['two-sided', 'greater', 'lesser']:
        clarkEvans = [count] + [math.nan] * 6
        if count > 1:
            distribution = rayleigh(scale=1 / math.sqrt(2 * math.pi * count / area))
            expected = distribution.mean()
            standardError = distribution.std() / math.sqrt(count)
            zScore = (observed - expected) / standardError
            clarkEvans = [count, observed, expected, observed / expected, standardError, zScore, getTail(norm, zScore, alternative)]
        quadratStatistics = [quadratCount, mean, variance, ratio, statistic, quadratCount - 1, getTail(chi2(quadratCount - 1), statistic, alternative) if math.isfinite(statistic) else math.nan]
        runs.append({'alternative': alternative, 'clarkEvans': clarkEvans, 'quadratStatistics': quadratStatistics})
    return {'name': name, 'positions': coordinates.ravel(), 'bounds': bounds, 'quadratGrid': quadratGrid, 'mask': mask, 'nearestNeighborDistances': distances, 'nearestNeighborIds': neighborIds, 'quadratCounts': counts, 'quadratAreas': np.full(quadratCount, area / quadratCount), 'ambiguousFloat32Neighbors': ambiguous, 'runs': runs}


source = np.fromfile(DATA / 'position.bin', dtype='<f4').reshape(-1, 2)
assert len(source) == 105808
transformer = pyproj.Transformer.from_crs(4326, 26916, always_xy=True)
projected = np.column_stack(transformer.transform(source[:, 0], source[:, 1])) - [400000, 4600000]
projected = projected.astype(np.float32).astype(float)
bounds = [math.floor(projected[:, 0].min()) - 16, math.floor(projected[:, 1].min()) - 16, math.ceil(projected[:, 0].max()) + 16, math.ceil(projected[:, 1].max()) + 16]
real = computeCase('chicago-places-all', projected, bounds, [8, 8])
writeFixture('positions', {'positions': packArray(real.pop('positions'), '<f4')})
writeFixture('distances', {'nearestNeighborDistances': packArray(real.pop('nearestNeighborDistances'), '<f8')})
writeFixture('ids', {'nearestNeighborIds': packArray(real.pop('nearestNeighborIds'), '<u4')})
real.pop('mask')
# Deterministic count variation around five events per cell; df=143 exercises approximation.
regularPoints = []
for row in range(12):
    for column in range(12):
        cellCount = [2, 4, 5, 6, 8][(row * 12 + column) % 5]
        for offset in range(cellCount):
            regularPoints.append([column + .2 + .05 * offset, row + .3 + .03 * offset])
synthetic = [
    computeCase('duplicates-ties-mask-and-boundaries', [[0, 0], [0, 0], [2, 0], [0, 2], [2, 2], [4, 4], [4, 0], [0, 4], [2, 4], [4, 2], [1, 1], [-1, 2], [math.nan, 2], [2, math.inf]], [0, 0, 4, 4], [2, 2], [1] * 10 + [0, 1, 1, 1]),
    computeCase('high-df-quadrats', regularPoints, [0, 0, 12, 12], [12, 12]),
    computeCase('singleton', [[1, 1], [2, 2]], [0, 0, 4, 4], [2, 2], [1, 0]),
    computeCase('empty-selection', [[1, 1], [2, 2]], [0, 0, 4, 4], [2, 2], [0, 0]),
]
writeFixture('statistics', {'real': real, 'synthetic': synthetic})
print('f32 ambiguous nearest rows:', len(real['ambiguousFloat32Neighbors']))
