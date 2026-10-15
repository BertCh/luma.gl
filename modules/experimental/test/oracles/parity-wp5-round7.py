# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent peak-neighborhood references; float64 arithmetic on float32 inputs.

COMPOSITE mathematical morphology reference: scipy.ndimage.maximum_filter on
unique (height,-row-major-index) ranks gives disc maxima; binary_erosion gives
the inner disc and valid-neighborhood mask. Ring = disc minus erosion by the
3x3 square (Serra, Image Analysis and Mathematical Morphology, 1982). A second
maximum_filter over the ring supplies the radius-limited drop. This is NOT
whole-plateau regional maxima or global topographic prominence. Ties use the
public lowest-index policy. Reject/ignore missing neighbors are explicit.

COMPOSITE snap reference: scipy.spatial.cKDTree.query_ball_point finds metric
neighborhoods, numpy.lexsort selects their maxima, and scipy.interpolate.
RegularGridInterpolator supplies linear DEM samples. The custom snap acceptance
policy (ring, unchanged, move, height) is applied explicitly, not claimed as a
standalone SciPy algorithm. The geometric ring uses morphological erosion of
the continuous-position lattice disc. Four bilinear corners must be valid,
including zero-weight corners, and the last row/column is replicated. Optional
catalogue heights/radii, fallback radii and radius caps follow the public API.
No TS/WGSL algorithm, pyramid traversal, scan or reduction is translated.

Uniform metric coordinates; candidate positions are pixel-center indices.
Raw -9999, NaN, infinity and explicit validity are excluded before float32
calibration. References use float64 on those calibrated values. Numeric snap
heights use unchanged R, distances unchanged D, positions/status E. Summit drop
is one f32 subtraction, so its independently calculated result is f32-rounded
and compared exactly (including NaN/+Infinity locations). Disc decisions are
guarded against nonexact near-boundary distances; exact binary arithmetic ties
are allowed. No tolerance is widened for a GPU result.

Decode the complete 2048x2048 Alps Terrarium once. Two adjacent 96x96 cores
with 6-cell halos are checked against full-raster morphology and a full-raster
KD tree for ALL selected snap queries, including the seam. JSON limits require
these reduced GPU tiles; no full-raster GPU parity is claimed. No random seed.
SciPy references: https://docs.scipy.org/doc/scipy/reference/ndimage.html,
https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query_ball_point.html,
https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html.
"""
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy
import rasterio
from scipy.ndimage import binary_erosion, maximum_filter
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / 'examples/deck/spatial-analysis-showcase/public/data/alps-dem'
OUTPUT = Path(__file__).parent / 'fixtures'
VERSIONS = {name: importlib.metadata.version(name) for name in ['numpy', 'scipy', 'rasterio']}
VERSIONS.update(python=platform.python_version(), gdal=rasterio.__gdal_version__)


def roundInput(values):
    return numpy.asarray(values, dtype=numpy.float32).astype(float)


def encode(value):
    if isinstance(value, dict): return {key: encode(entry) for key, entry in value.items()}
    if isinstance(value, (list, tuple, numpy.ndarray)): return [encode(entry) for entry in value]
    if isinstance(value, (numpy.integer, numpy.bool_)): return int(value)
    if isinstance(value, (float, numpy.floating)):
        if numpy.isnan(value): return None
        if numpy.isinf(value): return 'Infinity' if value > 0 else '-Infinity'
        return float(value)
    return value


def writeFixture(topic, cases):
    payload = dict(versions=VERSIONS, conventions=__doc__,
                   generatorSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), cases=cases)
    path = OUTPUT / f'parity-wp5-round7-{topic}.json'
    path.write_text(json.dumps(encode(payload), separators=(',', ':'), allow_nan=False) + '\n')
    assert path.stat().st_size <= 1500000, (path, path.stat().st_size)
    print(path.name, path.stat().st_size, 'cases', len(cases), flush=True)


def prepareCase(name, values, source, validity=None, scale=1, offset=0, cellSize=(1, 1)):
    values = roundInput(values)
    validity = numpy.ones(values.shape, dtype=int) if validity is None else numpy.asarray(validity)
    usable = numpy.isfinite(values) & (values != -9999) & (validity != 0)
    calibrated = roundInput(values * scale + offset)
    elevation = numpy.where(usable & numpy.isfinite(calibrated), calibrated, numpy.nan)
    return dict(name=name, width=values.shape[1], height=values.shape[0], source=source,
                elevation=values.ravel(), validity=validity.ravel(), scale=scale, offset=offset,
                cellSize=roundInput(cellSize),
                elevationSha256=hashlib.sha256(values.astype('<f4').tobytes()).hexdigest()), elevation


def makeDisc(radius, cellSize, center=(0, 0)):
    extent = int(numpy.ceil(radius / min(cellSize))) + 2
    rows, columns = numpy.mgrid[-extent:extent+1, -extent:extent+1]
    squared = ((columns-center[0])*cellSize[0])**2 + ((rows-center[1])*cellSize[1])**2
    difference = numpy.abs(squared-radius**2)
    # Nonzero separation > 32 eps32 bounds the short squared-distance expression.
    # Zero is accepted only with binary-exact coordinates/settings (synthetic ties).
    nonzero = difference[difference != 0]
    assert nonzero.min() > 32*numpy.finfo(numpy.float32).eps*max(1, radius**2)
    if numpy.any(difference == 0):
        assert all(float(value*8).is_integer() for value in [radius, *cellSize, *center])
    disc = squared <= radius**2
    ring = disc & ~binary_erosion(disc, structure=numpy.ones((3, 3)))
    return disc, ring, extent


def computeSummits(elevation, cellSize, radius, minimumDrop, policy):
    disc, ring, extent = makeDisc(radius, cellSize)
    ring[extent, extent] = False
    valid = numpy.isfinite(elevation)
    indices = numpy.arange(elevation.size)
    order = numpy.lexsort((-indices, numpy.where(valid, elevation, -numpy.inf).ravel()))
    ranks = numpy.zeros(elevation.size)
    ranks[order] = numpy.arange(1, elevation.size+1)
    ranks = numpy.where(valid, ranks.reshape(elevation.shape), 0)
    mask = valid & (ranks == maximum_filter(ranks, footprint=disc, mode='constant', cval=0))
    if policy == 'reject': mask &= binary_erosion(valid, structure=disc, border_value=0)
    ringMaximum = maximum_filter(numpy.where(valid, elevation, -numpy.inf), footprint=ring,
                                 mode='constant', cval=-numpy.inf) if ring.any() else numpy.full(elevation.shape, -numpy.inf)
    drop = roundInput(elevation-ringMaximum)
    mask &= drop >= minimumDrop
    return dict(mask=mask.astype(int).ravel(), drop=numpy.where(mask, drop, numpy.nan).ravel(),
                ids=numpy.flatnonzero(mask))


def buildSummits(name, values, source, radius=3.25, minimumDrop=0, policy='reject', maximumRadiusPixels=8, **options):
    case, elevation = prepareCase(name, values, source, **options)
    radius, minimumDrop = roundInput([radius, minimumDrop])
    maximumRadius = float(numpy.float32(maximumRadiusPixels * min(case['cellSize'])))
    expected = computeSummits(elevation, case['cellSize'], min(radius, maximumRadius), minimumDrop, policy)
    case.update(radius=radius, minimumDrop=minimumDrop, policy=policy, maximumRadiusPixels=maximumRadiusPixels,
                expected=expected, radiusClamped=int(radius > maximumRadius))
    return case


def makeTree(elevation, cellSize):
    indices = numpy.flatnonzero(numpy.isfinite(elevation))
    coordinates = numpy.column_stack((indices % elevation.shape[1], indices // elevation.shape[1]))
    return cKDTree(coordinates*cellSize), indices


def computeSnap(elevation, candidates, cellSize, settings, maximumRadiusPixels, candidateHeights=None, candidateRadii=None, treeData=None):
    height, width = elevation.shape
    tree, indices = makeTree(elevation, cellSize) if treeData is None else treeData
    # Padding makes singleton axes and edge replication explicit; invalid corners
    # are checked independently rather than relying on NaN interpolation behavior.
    padded = numpy.pad(numpy.nan_to_num(elevation, nan=0), ((0, 1), (0, 1)), mode='edge')
    interpolate = RegularGridInterpolator((numpy.arange(height+1), numpy.arange(width+1)), padded)
    positions = candidates.copy()
    heights = numpy.full(len(candidates), numpy.nan)
    status = numpy.full(len(candidates), 6)
    distances = numpy.zeros(len(candidates))
    radiusClamped = 0
    for candidateIndex, candidate in enumerate(candidates):
        if not numpy.isfinite(candidate).all() or numpy.any(candidate < 0) or numpy.any(candidate > [width-1, height-1]): continue
        base = numpy.floor(candidate).astype(int)
        corners = elevation[numpy.ix_([base[1], min(base[1]+1, height-1)], [base[0], min(base[0]+1, width-1)])]
        if numpy.isfinite(corners).all(): heights[candidateIndex] = interpolate(candidate[::-1][None, :])[0]
        requested = settings['radius']
        if candidateRadii is not None and numpy.isfinite(candidateRadii[candidateIndex]) and candidateRadii[candidateIndex] > 0:
            requested = candidateRadii[candidateIndex]
        maximumRadius = float(numpy.float32(maximumRadiusPixels*min(cellSize)))
        radiusClamped |= int(requested > maximumRadius)
        radius = min(requested, maximumRadius)
        # Build the lattice morphology around floor(candidate), independent of KD search.
        _, ring, extent = makeDisc(radius, cellSize, candidate-base)
        members = indices[tree.query_ball_point(candidate*cellSize, radius)]
        status[candidateIndex] = 5
        if not len(members): continue
        memberHeights = elevation.ravel()[members]
        bestIndex = members[numpy.lexsort((members, -memberHeights))[0]]
        bestPosition = numpy.array([bestIndex % width, bestIndex // width])
        bestHeight = elevation.ravel()[bestIndex]
        ringPosition = bestPosition-base+extent
        movement = numpy.linalg.norm((bestPosition-candidate)*cellSize)
        reference = heights[candidateIndex]
        if candidateHeights is not None and numpy.isfinite(candidateHeights[candidateIndex]): reference = candidateHeights[candidateIndex]
        if settings['interior'] and ring[ringPosition[1], ringPosition[0]]: status[candidateIndex] = 2
        elif numpy.array_equal(bestPosition, numpy.floor(candidate+0.5)): status[candidateIndex] = 0
        elif movement > settings['maximumMove']: status[candidateIndex] = 3
        elif numpy.isfinite(reference) and abs(bestHeight-reference) > settings['maximumHeightChange']: status[candidateIndex] = 4
        else:
            status[candidateIndex] = 1
            positions[candidateIndex] = bestPosition
            heights[candidateIndex] = bestHeight
            distances[candidateIndex] = movement
    finite = heights[numpy.isfinite(heights)]
    heightTolerance = dict(profile='R', absolute=1e-5*max(1, numpy.ptp(finite) if len(finite) else 0), relative=1e-4)
    return dict(positions=positions.ravel(), heights=heights, status=status, snapDistance=distances,
                radiusClamped=radiusClamped, heightTolerance=heightTolerance)


def buildSnap(name, values, source, candidates, radius=3.25, maximumMove=20, maximumHeightChange=1000,
              interior=False, maximumRadiusPixels=8, candidateHeights=None, candidateRadii=None, **options):
    case, elevation = prepareCase(name, values, source, **options)
    settings = dict(zip(['radius', 'maximumMove', 'maximumHeightChange'], roundInput([radius, maximumMove, maximumHeightChange])))
    settings['interior'] = interior
    candidates = roundInput(candidates)
    candidateHeights = roundInput(candidateHeights) if candidateHeights is not None else None
    candidateRadii = roundInput(candidateRadii) if candidateRadii is not None else None
    expected = computeSnap(elevation, candidates, case['cellSize'], settings, maximumRadiusPixels, candidateHeights, candidateRadii)
    case.update(candidates=candidates.ravel(), settings=settings, maximumRadiusPixels=maximumRadiusPixels, expected=expected)
    if candidateHeights is not None: case['candidateHeights'] = candidateHeights
    if candidateRadii is not None: case['candidateRadii'] = candidateRadii
    return case


def main():
    with rasterio.open(DATA/'dem.png') as dataset:
        channels = dataset.read().astype(float)
    terrain = roundInput(channels[0]*256+channels[1]+channels[2]/256-32768)
    cellSize = roundInput([6.6390088, 6.6390088])
    radius = float(numpy.float32(cellSize[0]*4.25))
    fullSummits = computeSummits(terrain, cellSize, radius, 0, 'reject')
    fullTree = makeTree(terrain, cellSize)
    for tile, coreColumn in enumerate([960, 1056]):
        origin = numpy.array([coreColumn-6, 954])
        values = terrain[954:1062, origin[0]:origin[0]+108]
        source = dict(kind='Alps DEM', path=str((DATA/'dem.png').relative_to(ROOT)),
                      sha256=hashlib.sha256((DATA/'dem.png').read_bytes()).hexdigest(),
                      manifestSha256=hashlib.sha256((DATA/'manifest.json').read_bytes()).hexdigest(),
                      fullShape=list(terrain.shape), coreSize=96, halo=6, origin=origin, fullReferenceVerified=True)
        summitCases = []
        for policy in ['reject', 'ignore']:
            case = buildSummits(f'alps-{tile}-{policy}', values, source, radius=radius, cellSize=cellSize, policy=policy)
            for metric in ['mask', 'drop']:
                core = case['expected'][metric].reshape(values.shape)[6:102, 6:102]
                full = fullSummits[metric].reshape(terrain.shape)[960:1056, coreColumn:coreColumn+96]
                numpy.testing.assert_array_equal(core, full)
            summitCases.append(case)
        writeFixture(f'summits-alps-{tile}', summitCases)
        # Includes columns .25 on either side of the tile seam; full-domain queries
        # retain the same disc because the six-cell halo exceeds the 4.25-cell radius.
        rows, columns = numpy.meshgrid(numpy.arange(6, 102, 4)+.375, numpy.arange(6, 102, 4)+.25, indexing='ij')
        candidates = numpy.column_stack((columns.ravel(), rows.ravel()))
        snapCases = []
        for interior in [False, True]:
            case = buildSnap(f'alps-{tile}-interior-{interior}', values, source, candidates, radius=radius,
                             maximumMove=30, maximumHeightChange=100, interior=interior, cellSize=cellSize)
            full = computeSnap(terrain, candidates+origin, cellSize, case['settings'], 8, treeData=fullTree)
            full['positions'] = (full['positions'].reshape(-1, 2)-origin).ravel()
            for metric in ['positions', 'heights', 'status', 'snapDistance']:
                numpy.testing.assert_allclose(case['expected'][metric], full[metric], atol=1e-10, rtol=0)
            snapCases.append(case)
        writeFixture(f'snap-alps-{tile}', snapCases)
    rows, columns = numpy.mgrid[:17, :19]
    hill = 100-(columns-9)**2-(rows-8)**2
    plateau = hill.copy(); plateau[7:10, 8:11] = 100
    holes = roundInput(plateau); holes[2, 3] = numpy.nan; holes[5, 6] = numpy.inf; holes[8, 9] = -9999
    validity = numpy.ones(holes.shape, dtype=int); validity[3, 2] = 0
    synthetic = dict(kind='analytic synthetic')
    summitCases = []
    for policy in ['reject', 'ignore']:
        for name, values, options in [
            ('hill', hill, {}), ('plateau', plateau, dict(radius=2.25)),
            ('masked-calibrated', holes, dict(validity=validity, scale=-2, offset=512, cellSize=(2, 3), radius=6.5)),
            ('drop-rejection', hill, dict(minimumDrop=100)),
            ('radius-clamped', hill, dict(radius=40, maximumRadiusPixels=4)),
            ('all-invalid', numpy.full((5, 7), numpy.nan), {}),
            ('singleton', [[42]], {}), ('small-disc', [[4, 3, 2], [1, 6, 5]], dict(radius=.25))]:
            summitCases.append(buildSummits(f'{name}-{policy}', values, synthetic, policy=policy, **options))
    writeFixture('summits-synthetic', summitCases)
    candidates = [[9, 8], [7.25, 8.375], [3.25, 4.375], [0, 0], [18, 16], [-1, 0], [numpy.nan, 3], [numpy.inf, 2]]
    snapCases = []
    for name, values, points, options in [
        ('hill', hill, candidates, {}), ('interior', hill, candidates, dict(interior=True)),
        ('move-rejection', hill, candidates, dict(maximumMove=.5)),
        ('height-rejection', hill, candidates, dict(maximumHeightChange=0)),
        ('plateau', plateau, candidates, {}),
        ('masked-calibrated', holes, candidates+[[6.25, 5.375]], dict(validity=validity, scale=2, offset=100)),
        ('catalogue-and-radii', hill, candidates, dict(candidateHeights=[100, 500, numpy.nan, numpy.inf, -1000, 0, 0, 0],
             candidateRadii=[1, 6.25, numpy.nan, -1, 0, numpy.inf, 40, 40], maximumHeightChange=20)),
        ('radius-clamped', hill, candidates, dict(radius=40, maximumRadiusPixels=4)),
        ('anisotropic', hill, candidates, dict(cellSize=(2, 3), radius=6.5)),
        ('all-invalid', numpy.full((5, 7), numpy.nan), [[2.25, 2.375], [6, 4]], {}),
        ('singleton', [[42]], [[0, 0], [1, 0]], {}),
        ('empty-disc', hill, [[7.25, 8.375]], dict(radius=.125)),
        ('invalid-interpolation', [[1, 2, 3], [4, -9999, 6], [7, 8, 9]], [[0, 0]], dict(radius=.25))]:
        snapCases.append(buildSnap(name, values, synthetic, points, **options))
    assert set(numpy.concatenate([entry['expected']['status'] for entry in snapCases])) == set(range(7))
    writeFixture('snap-synthetic', snapCases)


if __name__ == '__main__': main()
