# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Round 6 independent terrain references; float64 arithmetic on float32 inputs.

Contours: skimage.measure.find_contours on valid 2x2 cells. COMPOSITE asymptotic
connectivity adapter: evaluate the bilinear saddle elevation (Nielson/Hamann,
The Asymptotic Decider, 1991, doi:10.1109/VISUAL.1991.175782), then select
skimage's high/low connectivity; an exact saddle tie connects high. Unlike an
arithmetic-centre rule this is invariant under bilinear subdivision. All crossing
corners are strictly off-level; constant exact-level cells are also tested.
Local (column,row) pixel centers; area offset .5 or point offset 0. Whole cells
with a missing corner are omitted. Export ALL segments, edge pairs and closed
rings (Shapely polygonize), without dependence on emission order or orientation.
No corner-equality/zero-length segment parity is claimed. G = 8 coordinate ULPs.

Critical points: COMPOSITE lower-link topology (Edelsbrunner/Harer,
Computational Topology, PL Morse theory). scipy.ndimage.label finds connected
components of the lower link; its path ends are identified to form a circle.
Empty lower link = minimum; complete circle = maximum; multiple components =
saddle. Multiplicity = components-1 and signChanges = 2*components for mixed
links. Six-neighbor Freudenthal mesh or eight-neighbor Peucker/Douglas ring.
Lexicographic (height,row-major index) ranking supplies simulation of simplicity;
this is NOT skimage's whole-plateau maxima contract. Extrema independently checked
by scipy.ndimage maximum/minimum filters on unique ranks. No TS/WGSL is ported.
Edges or incomplete links = boundary; invalid center = nodata. All codes exact.

Flow: COMPOSITE numpy.gradient(edge_order=1) in uniform ground metres, then a
3D orthogonal tangent-plane projection via normalized normals and numpy.einsum.
This follows the documented projection theorem, not a hydrologic D8 direction or
GDAL Horn estimator. Rows increase in the wind's positive y direction. Singleton
axes have zero derivative. Invalid center or used gradient neighbor => both NaN.
Band scale/offset are applied then f32-rounded, as at the public band boundary.
Each case has two independently evaluated winds, for a per-frame parameter update.
Unchanged R bounds, per component, and exact nonfinite masks.

Full 2048x2048 Alps Terrarium PNG decoded once, no resampling. Adjacent 96x96
cores with one-cell halos for critical points and flow are checked against full
raster references, including the seam. Contours use adjacent 96x96 CELL domains
(97x97 samples) checked against a united domain. Reduced scale due to 1.5MB JSON
limit; neither full-DEM nor 512x512 GPU execution is claimed. Source/manifest and
generator hashes, versions, parameters and every output are retained. No RNG.
"""
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import numpy
import rasterio
from scipy.ndimage import label, maximum_filter, minimum_filter
from skimage.measure import find_contours
from shapely.geometry import LineString
from shapely.ops import polygonize

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / 'examples/deck/spatial-analysis-showcase/public/data/alps-dem'
OUTPUT = Path(__file__).parent / 'fixtures'
VERSIONS = {name: importlib.metadata.version(name) for name in ['numpy', 'scipy', 'scikit-image', 'shapely', 'rasterio']}
VERSIONS.update(python=platform.python_version(), gdal=rasterio.__gdal_version__)


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
    path = OUTPUT / f'parity-wp5-round6-{topic}.json'
    path.write_text(json.dumps(encode(payload), separators=(',', ':'), allow_nan=False) + '\n')
    assert path.stat().st_size <= 1500000, (path, path.stat().st_size)
    print(path.name, path.stat().st_size, flush=True)


def roundInput(values): return numpy.asarray(values, dtype=numpy.float32).astype(float)


def prepareCase(name, values, source, validity=None, scale=1, offset=0):
    values = roundInput(values)
    validity = numpy.ones(values.shape, dtype=int) if validity is None else numpy.asarray(validity)
    usable = numpy.isfinite(values) & (values != -9999) & (validity != 0)
    calibrated = roundInput(values * scale + offset)
    elevation = numpy.where(usable & numpy.isfinite(calibrated), calibrated, numpy.nan)
    return dict(name=name, width=values.shape[1], height=values.shape[0], source=source,
                elevation=values.ravel(), validity=validity.ravel(), scale=scale, offset=offset,
                elevationSha256=hashlib.sha256(values.astype('<f4').tobytes()).hexdigest()), elevation


def computeCritical(elevation, connectivity):
    height, width = elevation.shape
    indices = numpy.arange(elevation.size).reshape(elevation.shape)
    order = numpy.lexsort((indices.ravel(), elevation.ravel()))
    ranks = numpy.empty(elevation.size, dtype=int)
    ranks[order] = numpy.arange(elevation.size)
    ranks = ranks.reshape(elevation.shape)
    footprint = numpy.ones((3, 3), dtype=bool)
    if connectivity == 6: footprint[0, 2] = footprint[2, 0] = False
    peak = maximum_filter(ranks, footprint=footprint, mode='nearest') == ranks
    pit = minimum_filter(ranks, footprint=footprint, mode='nearest') == ranks
    # Ordered vertices of the link, from the documented grid triangulation.
    offsets = [(0, 1), (1, 1), (1, 0), (0, -1), (-1, -1), (-1, 0)] if connectivity == 6 else [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]
    classes = numpy.where(numpy.isfinite(elevation), 4, 5)
    changes = numpy.zeros(elevation.shape, dtype=int)
    if height < 3 or width < 3: return classes, changes
    # Bounded strips keep a full-DEM independent reference inexpensive in memory.
    for firstRow in range(1, height - 1, 64):
        lastRow = min(firstRow + 64, height - 1)
        center = ranks[firstRow:lastRow, 1:-1]
        lower = numpy.stack([ranks[firstRow+row:lastRow+row, 1+column:width-1+column] < center for row, column in offsets], axis=-1)
        complete = numpy.all(numpy.stack([numpy.isfinite(elevation[firstRow+row:lastRow+row, 1+column:width-1+column]) for row, column in offsets]), axis=0)
        links = lower.reshape(-1, connectivity)
        labels, _ = label(links, structure=[[0, 0, 0], [1, 1, 1], [0, 0, 0]])
        # Each row's component labels are consecutive; no vertical adjacency.
        minimum = numpy.where(labels > 0, labels, numpy.iinfo(labels.dtype).max).min(axis=1)
        maximum = labels.max(axis=1)
        components = numpy.where(maximum > 0, maximum - minimum + 1, 0)
        components -= links[:, 0] & links[:, -1] & (labels[:, 0] != labels[:, -1])
        components = components.reshape(center.shape)
        allLower = lower.all(axis=-1)
        noLower = ~lower.any(axis=-1)
        codes = numpy.select([allLower, noLower, components > 1], [1, 2, 3], default=0)
        accepted = complete & numpy.isfinite(elevation[firstRow:lastRow, 1:-1])
        numpy.testing.assert_array_equal((codes == 1)[accepted], peak[firstRow:lastRow, 1:-1][accepted])
        numpy.testing.assert_array_equal((codes == 2)[accepted], pit[firstRow:lastRow, 1:-1][accepted])
        classes[firstRow:lastRow, 1:-1][accepted] = codes[accepted]
        transitions = numpy.where(allLower | noLower, 0, 2 * components)
        changes[firstRow:lastRow, 1:-1][accepted] = transitions[accepted]
    return classes, changes


def buildCritical(name, values, source, connectivity, **options):
    case, elevation = prepareCase(name, values, source, **options)
    classes, changes = computeCritical(elevation, connectivity)
    case.update(connectivity=connectivity, classes=classes.ravel(), signChanges=changes.ravel(),
                counts=numpy.bincount(classes.ravel(), minlength=6))
    return case


def computeFlow(elevation, cellSize, wind, exaggeration):
    gradients = [numpy.gradient(elevation, cellSize[1-axis], axis=axis, edge_order=1) * exaggeration
                 if elevation.shape[axis] > 1 else numpy.zeros_like(elevation) for axis in [0, 1]]
    normals = numpy.stack([-gradients[1], -gradients[0], numpy.ones_like(elevation)], axis=-1)
    normals /= numpy.linalg.norm(normals, axis=-1, keepdims=True)
    windVector = numpy.array([*wind, 0.])
    projected = windVector - numpy.einsum('...i,i->...', normals, windVector)[..., None] * normals
    projected[~numpy.isfinite(elevation)] = numpy.nan
    return projected[..., :2]


def buildFlow(name, values, source, cellSize=(2, 3), wind=(7, -4), exaggeration=1, **options):
    case, elevation = prepareCase(name, values, source, **options)
    cellSize, wind = roundInput(cellSize), roundInput(wind)
    exaggeration = float(numpy.float32(exaggeration))
    winds = [wind, numpy.array([-wind[1], wind[0]])]
    outputs = [computeFlow(elevation, cellSize, selected, exaggeration) for selected in winds]
    tolerances = []
    for output in outputs:
        tolerances.append([dict(profile='R', absolute=1e-5 * max(1, float(numpy.ptp(finite)) if finite.size else 0), relative=1e-4)
                           for component in range(2) for finite in [output[..., component][numpy.isfinite(output[..., component])]]])
    case.update(cellSize=cellSize, winds=winds, exaggeration=exaggeration,
                expected=[output.ravel() for output in outputs], tolerances=tolerances)
    return case


def computeContours(elevation, level, pixelOffset):
    height, width = elevation.shape
    segments = []
    for row in range(height - 1):
        for column in range(width - 1):
            cell = elevation[row:row+2, column:column+2]
            if not numpy.isfinite(cell).all(): continue
            if numpy.all(cell == level): continue
            assert numpy.all(cell != level), 'Corner equality intentionally excluded'
            diagonalHigh = (cell[0, 0] > level) == (cell[1, 1] > level)
            ambiguous = diagonalHigh and (cell[0, 1] > level) == (cell[1, 0] > level) and (cell[0, 0] > level) != (cell[0, 1] > level)
            connectHigh = False
            if ambiguous:
                columnCoefficient = cell[0, 1] - cell[0, 0]
                rowCoefficient = cell[1, 0] - cell[0, 0]
                mixed = cell[1, 1] - cell[1, 0] - cell[0, 1] + cell[0, 0]
                saddle = cell[0, 0] - columnCoefficient * rowCoefficient / mixed
                connectHigh = saddle >= level
            for contour in find_contours(cell, level, fully_connected='high' if connectHigh else 'low'):
                assert len(contour) == 2
                coordinates = contour[:, ::-1]
                edgeIds = []
                for horizontal, vertical in coordinates:
                    if vertical in (0, 1): edgeIds.append((row + int(vertical)) * (width - 1) + column)
                    else:
                        assert horizontal in (0, 1)
                        edgeIds.append(height * (width - 1) + row * width + column + int(horizontal))
                coordinates += [column + pixelOffset, row + pixelOffset]
                if edgeIds[0] > edgeIds[1]: edgeIds.reverse(); coordinates = coordinates[::-1]
                segments.append([*edgeIds, *coordinates.ravel()])
    segments.sort(key=lambda segment: tuple(segment[:2]))
    # Shared f64 interpolation can differ at roundoff: canonicalize by exact grid edge ID.
    vertices = {}
    for segment in segments:
        for endpoint in range(2): vertices.setdefault(segment[endpoint], segment[2+2*endpoint:4+2*endpoint])
    lines = [LineString([vertices[segment[0]], vertices[segment[1]]]) for segment in segments]
    rings = [list(polygon.exterior.coords) for polygon in polygonize(lines)]
    return segments, rings


def buildContours(name, values, source, levels, point=False, **options):
    case, elevation = prepareCase(name, values, source, **options)
    expected = []
    for level in roundInput(levels):
        segments, rings = computeContours(elevation, level, 0 if point else .5)
        expected.append(dict(level=level, segments=segments, rings=rings))
    case.update(point=point, expected=expected,
                tolerance=dict(profile='G', absolute=8 * 2.**(int(numpy.floor(numpy.log2(max(elevation.shape)))) - 23), relative=0))
    return case


def main():
    with rasterio.open(DATA / 'dem.png') as dataset: channels = dataset.read().astype(float)
    full = roundInput(channels[0]*256 + channels[1] + channels[2]/256 - 32768)
    del channels
    source = dict(kind='Alps DEM', path=str((DATA / 'dem.png').relative_to(ROOT)),
                  sha256=hashlib.sha256((DATA / 'dem.png').read_bytes()).hexdigest(),
                  manifestSha256=hashlib.sha256((DATA / 'manifest.json').read_bytes()).hexdigest(), fullShape=list(full.shape))
    for connectivity in [6, 8]:
        fullClasses, fullChanges = computeCritical(full, connectivity)
        for tile in range(2):
            column = 960 + tile * 96
            case = buildCritical(f'alps-{connectivity}-{tile}', full[959:1057, column-1:column+97],
                                 dict(**source, coreOrigin=[960, column], coreSize=96, halo=1, fullReferenceVerified=True), connectivity)
            for key, reference in [('classes', fullClasses), ('signChanges', fullChanges)]:
                numpy.testing.assert_array_equal(case[key].reshape(98, 98)[1:-1, 1:-1], reference[960:1056, column:column+96])
            writeFixture(f'critical-alps-{connectivity}-{tile}', [case])
    cellSize = (float(numpy.float32(6.6390088)),) * 2
    fullFlow = computeFlow(full, cellSize, [7, -4], 1)
    for tile in range(2):
        column = 960 + tile * 96
        case = buildFlow(f'alps-flow-{tile}', full[959:1057, column-1:column+97],
                         dict(**source, coreOrigin=[960, column], coreSize=96, halo=1, fullReferenceVerified=True), cellSize=cellSize)
        numpy.testing.assert_allclose(case['expected'][0].reshape(98, 98, 2)[1:-1, 1:-1], fullFlow[960:1056, column:column+96], atol=1e-12, rtol=0)
        writeFixture(f'flow-alps-{tile}', [case])
    # These source windows have quarter-metre samples; half-steps avoid corner equality.
    levels = [2250.125, 2450.125, 2650.125]
    united = full[960:1057, 960:1153]
    unitedContours = [computeContours(united, level, .5)[0] for level in levels]
    for tile in range(2):
        case = buildContours(f'alps-contours-{tile}', united[:, tile*96:tile*96+97],
                             dict(**source, cellOrigin=[960, 960+96*tile], coreSize=96, unitedReferenceVerified=True), levels)
        for result, reference in zip(case['expected'], unitedContours):
            selected = [segment[2:] for segment in reference if tile*96 <= (segment[2]+segment[4])/2-.5 < (tile+1)*96]
            translated = [[segment[2]+tile*96, segment[3], segment[4]+tile*96, segment[5]] for segment in result['segments']]
            # Canonicalize orientation since horizontal/vertical edge ID ranges differ by width.
            def canonical(values): return sorted(tuple(sorted([tuple(entry[:2]), tuple(entry[2:])])) for entry in values)
            numpy.testing.assert_allclose(canonical(selected), canonical(translated), atol=3e-14, rtol=0)
        writeFixture(f'contours-alps-{tile}', [case])
    rows, columns = numpy.indices((19, 23))
    egg = roundInput(10*numpy.cos(rows*.7)*numpy.cos(columns*.9))
    plateau = numpy.zeros((11, 13)); plateau[3:8, 4:9] = 5
    holes = egg.copy(); holes[3, 4] = numpy.nan; holes[7, 8] = -9999; holes[10, 13] = numpy.inf
    validity = numpy.ones_like(holes, dtype=int); validity[6:8, 15:17] = 0
    synthetic = dict(kind='synthetic')
    criticalCases = []
    for connectivity in [6, 8]:
        for name, values, options in [('egg-crate', egg, {}), ('plateau', plateau, {}), ('holes', holes, dict(validity=validity)),
                                      ('all-invalid', numpy.full((5, 7), numpy.nan), {}), ('two-rows', [[1, 2, 3], [4, 5, 6]], {}),
                                      ('calibrated-reversal', egg, dict(scale=-2, offset=512))]:
            criticalCases.append(buildCritical(f'{name}-{connectivity}', values, synthetic, connectivity, **options))
    writeFixture('critical-synthetic', criticalCases)
    flowCases = [buildFlow('anisotropic-plane', 100 + 3*columns - 2*rows, synthetic),
                 buildFlow('steep-plane', 1000*columns + 4000*rows, synthetic, exaggeration=2),
                 buildFlow('flat', numpy.full((7, 9), 8192.5), synthetic),
                 buildFlow('zero-wind', egg, synthetic, wind=(0, 0)),
                 buildFlow('zero-exaggeration', egg, synthetic, exaggeration=0),
                 buildFlow('holes-calibrated', holes, synthetic, validity=validity, scale=2, offset=100),
                 buildFlow('single-row', [[0, 1, 4, 9, -9999, 2, 3]], synthetic),
                 buildFlow('single-column', [[0], [1], [4], [9], [numpy.inf], [2]], synthetic),
                 buildFlow('singleton', [[9]], synthetic),
                 buildFlow('all-invalid', numpy.full((3, 4), numpy.nan), synthetic)]
    writeFixture('flow-synthetic', flowCases)
    contourCases = [buildContours('closed-nested-hills', (columns-11)**2 + (rows-9)**2, synthetic, [12.5, 40.5, 75.5]),
                    buildContours('masked-point-grid', holes, synthetic, [.5, 2.5], point=True, validity=validity),
                    buildContours('flat-equal-level', numpy.full((5, 7), 3), synthetic, [3]),
                    buildContours('calibrated-plane', columns + rows, synthetic, [105.5, 120.5], scale=2, offset=100)]
    for tile, values in enumerate([[[4, -1], [-1, .125]], [[1, -4], [-.125, 1]], [[1, -1], [-1, 1]], [[-1, 1], [1, -1]]]):
        contourCases.append(buildContours(f'saddle-{tile}', values, synthetic, [0], point=tile % 2 == 1))
    writeFixture('contours-synthetic', contourCases)


if __name__ == '__main__': main()
