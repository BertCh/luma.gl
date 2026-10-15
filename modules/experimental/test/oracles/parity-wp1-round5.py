# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent Round 5 references; reference Python -I, optional --output-dir.

Neighborhoods: libpysal Queen on ALL 791 f32 Chicago tract shells, with counts
from ALL 105808 place assignments. pandas expanded membership preserves duplicate
slots for count/median; scipy CSR dot supplies weighted sums; numpy.average/cov
(ddof=0) supplies mean/deviation; pandas category frequencies + scipy.stats.entropy
supply category frequencies and natural-log entropy. Mode comparisons use exact
rational normalized frequencies (Fraction of uploaded f32 weights), so equal
fractions cannot become unequal through float64 division or summation. Positive off-diagonal
slots only, masks apply to both endpoints, finite values only for numeric outputs.
Row normalization is over eligible slots separately for numeric and categorical
members; an explicitly added focal member keeps its own weight (even zero).
Synthetic invalid neighbor IDs, self slots, duplicates, zero/negative weights,
NaN/Inf values, islands, median overflow and truncated-input status are adapters
of the documented public contract, not copies of the GPU algorithm. No RNG.

Partition: pandas groupby sizes and population variance times group size supplies
within SSD; numpy.var supplies total SSD, sklearn Calinski-Harabasz supplies
pseudo-F. Labels outside capacity are excluded. Directed CSR SLOT counts, not
weight magnitudes, define crossing fraction; sparse group indicator products
count links including duplicate/zero/self slots. This class does NOT expose
silhouettes or component counts; no such parity is claimed. Full 791 tract
attributes with deterministic poverty quartile labels, plus full 43557 nature
observations grouped by actual category. Empty label capacity is retained.
Degenerate pseudo-F is NaN for k<2 or n<=k, sklearn's 1 for zero within SSD.

Region: Shapely covers (rectangle), contains with explicit half-open boundary
adapter (lasso), distance<=radius (circle), pandas
aggregation and numpy.histogram. ALL nature observations, EPSG:4326->26916,
local [440000,4600000]m origin, km, snapped 1/1024 km and f32 before references.
Day-of-year is the stored local-clock day (floor seconds/86400), no timezone
claim. Rectangles/circles have closed boundaries. Lasso is an implicitly closed
ring with half-open membership, NOT Shapely covers: for these axis-aligned lassos,
left/bottom edges belong and right/top edges do not. Only exact GEOS boundary
points are evaluated after a small upper-right translation; both epsilon and
epsilon/2 must agree. Interior/exterior queries retain their unmodified inputs.
Finite values only, selected count includes invalid values. Empty summary floats are zero. Histogram includes rightmost endpoint;
constant domain maps its equal values to bin zero (GPUHistogram convention;
NumPy explicit repeated-edge bins would use the last bin instead). Fixed and selection
histogram domains, concave lasso, exact circle/rectangle boundaries, empty and
constant selections, nonfinite positions/values. Stable IDs are an explicit
scenario (reverse row IDs), not purported source IDs. Output capacity truncates
IDs only. Every mask/selected ID and all advertised summary outputs are retained.
No variance exists on this public class. No reduction outputs are subsampled.

References:
https://pysal.org/libpysal/generated/libpysal.weights.Queen.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.dot.html
https://numpy.org/doc/stable/reference/generated/numpy.cov.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.calinski_harabasz_score.html
https://shapely.readthedocs.io/en/stable/reference/shapely.covers.html
https://numpy.org/doc/stable/reference/generated/numpy.histogram.html
"""
import argparse
import base64
import hashlib
from fractions import Fraction
import json
import platform
from pathlib import Path
import warnings
import zlib
import numpy as numpy
import pandas
import scipy
from scipy.sparse import csr_matrix
from scipy.stats import entropy
import sklearn
from sklearn.metrics import calinski_harabasz_score
import shapely
from shapely import Polygon
import pyproj
import libpysal
from libpysal.weights import Queen

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / 'examples/deck/spatial-analysis-showcase/public/data'
NO_LABEL = 4294967295
VERSIONS = dict(python=platform.python_version(), numpy=numpy.__version__, pandas=pandas.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__, shapely=shapely.__version__, pyproj=pyproj.__version__, libpysal=libpysal.__version__)
STATISTICS = ['count', 'weightSum', 'sum', 'mean', 'min', 'max', 'standardDeviation', 'median']


def clean(value):
    if isinstance(value, numpy.ndarray): return clean(value.tolist())
    if isinstance(value, numpy.generic): return clean(value.item())
    if isinstance(value, float) and not numpy.isfinite(value): return None if numpy.isnan(value) else ('Infinity' if value > 0 else '-Infinity')
    if isinstance(value, dict): return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [clean(item) for item in value]
    return value


def writeFixture(directory, name, payload):
    raw = json.dumps(clean(payload), separators=(',', ':'), allow_nan=False).encode()
    wrapper = dict(generator=Path(__file__).name, versions=VERSIONS, encoding='zlib-json', sha256=hashlib.sha256(raw).hexdigest(), data=base64.b64encode(zlib.compress(raw, 9)).decode())
    serialized = json.dumps(wrapper, separators=(',', ':')) + '\n'
    assert len(serialized.encode()) <= 1500000, (name, len(serialized))
    (directory / f'parity-wp1-{name}.json').write_text(serialized)
    print(name, len(serialized), 'bytes', flush=True)


def provenance(files):
    return dict(sourceSha256={name: hashlib.sha256((DATA / name).read_bytes()).hexdigest() for name in files})


def packMatrix(matrix):
    matrix = matrix.tocsr()
    return dict(offsets=matrix.indptr, neighbors=matrix.indices, weights=matrix.data)


def float32(values):
    return numpy.asarray(values, dtype=numpy.float32).astype(float)


def hashFloats(*arrays):
    return hashlib.sha256(b''.join(numpy.asarray(array, dtype='<f4').tobytes() for array in arrays)).hexdigest()


def makeNeighborhood(name, matrix, values, categories, mask, normalize, includeFocal, focalWeight, maximumNeighbors, truncated, metadata, integerValues=False):
    values = float32(values)
    weights = float32(matrix.data)
    categories = numpy.asarray(categories)
    mask = numpy.asarray(mask, dtype=bool)
    rows = len(values)
    source = numpy.repeat(numpy.arange(rows), numpy.diff(matrix.indptr))
    target = matrix.indices
    valid = (source != target) & (target < rows) & (weights > 0)
    frame = pandas.DataFrame(dict(source=source[valid], target=target[valid], weight=weights[valid]))
    frame = frame[mask[frame.source] & mask[frame.target]]
    numeric = frame[numpy.isfinite(values[frame.target])].copy()
    categorical = frame.copy()
    for members in [numeric, categorical]:
        if normalize:
            members['weight'] /= members.groupby('source').weight.transform('sum')
    if includeFocal:
        focal = pandas.DataFrame(dict(source=numpy.flatnonzero(mask), target=numpy.flatnonzero(mask), weight=focalWeight))
        numeric = pandas.concat([focal[numpy.isfinite(values[focal.target])], numeric], ignore_index=True)
        categorical = pandas.concat([focal, categorical], ignore_index=True)
    numeric['value'] = values[numeric.target]
    grouped = numeric.groupby('source').value
    count = grouped.count().reindex(range(rows), fill_value=0).to_numpy()
    operator = csr_matrix((numeric.weight, (numeric.source, numeric.target)), shape=(rows, rows))
    sums = operator.dot(numpy.where(numpy.isfinite(values), values, 0))
    weightSums = numpy.asarray(operator.sum(axis=1)).ravel()
    means = numpy.full(rows, numpy.nan)
    deviations = numpy.full(rows, numpy.nan)
    for row, group in numeric.groupby('source'):
        if group.weight.sum() > 0:
            means[row] = numpy.average(group.value, weights=group.weight)
            deviations[row] = numpy.sqrt(numpy.cov(group.value, aweights=group.weight, ddof=0)) if len(group) > 1 else 0
    table = numpy.column_stack([count, weightSums, sums, means,
        grouped.min().reindex(range(rows)), grouped.max().reindex(range(rows)), deviations,
        grouped.median().reindex(range(rows))])
    overflow = bool(numpy.any(count > maximumNeighbors)) and not truncated
    table[count > maximumNeighbors, 7] = numpy.nan
    modes = numpy.full(rows, NO_LABEL, dtype=numpy.uint32)
    entropies = numpy.full(rows, numpy.nan)
    categorical['category'] = categories[categorical.target]
    frequencies = categorical.groupby(['source', 'category']).weight.sum()
    for row, frequenciesForRow in frequencies.groupby(level='source'):
        frequenciesForRow = frequenciesForRow.droplevel('source')
        # The mode is discrete: compare exact rational frequencies, not rounded
        # float64 normalized sums. Chicago row 689 has categories 1 and 3 at
        # exactly 3/5; pandas otherwise makes category 3 larger by one ulp.
        rawMembers = frame[frame.source == row]
        denominator = sum((Fraction(weight) for weight in rawMembers.weight), Fraction(0)) if normalize else Fraction(1)
        exactFrequencies = {}
        for member in rawMembers.itertuples():
            category = int(categories[member.target])
            exactFrequencies[category] = exactFrequencies.get(category, Fraction(0)) + Fraction(member.weight) / denominator
        if includeFocal:
            category = int(categories[row])
            exactFrequencies[category] = exactFrequencies.get(category, Fraction(0)) + Fraction(focalWeight)
        modes[row] = min(exactFrequencies, key=lambda category: (-exactFrequencies[category], category))
        if frequenciesForRow.sum() > 0: entropies[row] = entropy(frequenciesForRow.to_numpy())
    if truncated:
        table[:] = numpy.nan
        modes[:] = NO_LABEL
        entropies[:] = numpy.nan
    table[~mask] = numpy.nan
    table[~mask, 0] = 0
    return dict(name=name, reference='scipy.sparse.csr_matrix.dot; pandas.groupby; numpy.average/cov; scipy.stats.entropy', n=rows,
        **packMatrix(matrix), values=values, categories=categories, mask=mask.astype(int), normalize=normalize,
        includeFocal=includeFocal, focalWeight=focalWeight, maximumNeighbors=maximumNeighbors, truncated=truncated, integerValues=integerValues,
        statistics=STATISTICS, expected=dict(table=table.ravel(), modes=modes, entropy=entropies, overflow=[int(overflow)]),
        inputFloat32Sha256=hashFloats(values, weights), provenance=metadata)


def makePartition(name, values, labels, capacity, matrix, metadata):
    values = float32(values)
    labels = numpy.asarray(labels)
    valid = labels < capacity
    frame = pandas.DataFrame(values[valid])
    frame['label'] = labels[valid]
    grouped = frame.groupby('label')
    sizes = grouped.size().reindex(range(capacity), fill_value=0)
    within = grouped.var(ddof=0).sum(axis=1).mul(grouped.size()).reindex(range(capacity), fill_value=0).to_numpy()
    regionCount = int((sizes > 0).sum())
    total = float(numpy.var(values[valid], axis=0, ddof=0).sum() * valid.sum()) if valid.any() else 0.
    withinTotal = float(within.sum())
    between = max(0., total - withinTotal)
    pseudoF = calinski_harabasz_score(values[valid], labels[valid]) if 1 < regionCount < valid.sum() else numpy.nan
    crossFraction = 0.
    if matrix is not None:
        # Indicator-matrix aggregation counts slots, even duplicate/zero-weight links.
        # Build ones BEFORE sparse arithmetic: astype may coalesce duplicate
        # input slots, which would otherwise lose their multiplicity here.
        binary = csr_matrix((numpy.ones(matrix.nnz), matrix.indices.copy(), matrix.indptr.copy()), shape=matrix.shape)
        indicator = csr_matrix((numpy.ones(valid.sum()), (numpy.flatnonzero(valid), labels[valid])), shape=(len(labels), capacity))
        regionLinks = indicator.T @ binary @ indicator
        totalLinks = float(regionLinks.sum())
        if totalLinks: crossFraction = float(totalLinks - regionLinks.diagonal().sum()) / totalLinks
    nonempty = sizes[sizes > 0]
    summary = [regionCount, withinTotal, between, total, int(nonempty.min()) if regionCount else 0,
        int(nonempty.max()) if regionCount else 0, crossFraction, int((~valid).sum()), pseudoF]
    return dict(name=name, reference='pandas.groupby.var(ddof=0); numpy.var; sklearn.metrics.calinski_harabasz_score; scipy sparse indicator product',
        n=len(values), columns=values.shape[1], values=values.ravel(), labels=labels, labelCapacity=capacity,
        **({} if matrix is None else dict(graph=packMatrix(matrix))), expected=dict(summary=summary, regionSizes=sizes.to_numpy(), regionWithinSumOfSquares=within),
        inputFloat32Sha256=hashFloats(values), provenance=metadata)


def makeRegions(name, positions, values, selections, metadata):
    positions = float32(positions)
    values = float32(values)
    sourceIds = numpy.arange(len(values), dtype=numpy.uint32)[::-1] + 100000
    points = shapely.points(positions)
    queries = []
    for selection in selections:
        kind = selection['kind']
        if kind == 'rectangle':
            geometry = shapely.box(*selection['coordinates'])
            selected = shapely.covers(geometry, points)
        elif kind == 'polygon':
            geometry = Polygon(numpy.asarray(selection['coordinates']).reshape(-1, 2))
            assert geometry.is_valid
            # PNPOLY's half-open ownership on orthogonal edges differs from
            # covers. Adapt only exact boundary points, independently through
            # GEOS containment, without copying the crossing kernel. The input
            # snap spacing is 2^-10 km; epsilon=2^-25 is 32768 times smaller.
            # Halving epsilon must leave membership unchanged. Non-boundary
            # inputs are never moved, and the GPU receives the original points.
            assert numpy.all(numpy.any(numpy.diff(numpy.vstack([geometry.exterior.coords]), axis=0) == 0, axis=1))
            selected = shapely.contains(geometry, points)
            boundary = shapely.covers(geometry.boundary, points)
            boundaryPoints = positions[boundary]
            epsilon = 2**-25
            included = shapely.contains(geometry, shapely.points(boundaryPoints + epsilon))
            assert numpy.array_equal(included, shapely.contains(geometry, shapely.points(boundaryPoints + epsilon / 2)))
            selected[boundary] = included
        else:
            centerFirst, centerSecond, radius = selection['coordinates']
            distances = shapely.distance(shapely.Point(centerFirst, centerSecond), points)
            selected = distances <= radius
            if len(values) > 100:
                # A conservative 16 f32 ulps at squared-coordinate scale identifies
                # genuinely ambiguous near-circle inputs; real queries have none.
                squared = numpy.sum((positions - [centerFirst, centerSecond])**2, axis=1)
                bound = 16 * numpy.finfo(numpy.float32).eps * max(1, float(squared.max()), radius**2)
                assert numpy.min(numpy.abs(squared-radius**2)) > bound
        selected &= numpy.isfinite(positions).all(axis=1)
        finite = values[selected & numpy.isfinite(values)]
        aggregates = pandas.Series(finite).agg(['sum', 'mean', 'min', 'max']).to_numpy() if len(finite) else numpy.zeros(4)
        binCount = selection['binCount']
        domain = selection.get('domain', [float(finite.min()), float(finite.max())] if len(finite) else [0., 0.])
        if binCount:
            if domain[0] == domain[1]:
                histogram = numpy.zeros(binCount, dtype=int)
                histogram[0] = numpy.count_nonzero(finite == domain[0])
            else:
                histogram = numpy.histogram(finite, bins=numpy.linspace(*domain, binCount + 1))[0]
            outside = len(finite) - int(histogram.sum())
        else:
            histogram = []
            outside = 0
        queries.append(dict(**selection, expected=dict(mask=selected.astype(int), ids=sourceIds[selected],
            counts=[int(selected.sum()), len(finite), outside], statistics=aggregates, histogram=histogram)))
    return dict(name=name, reference='shapely.covers/contains/distance (half-open lasso adapter); pandas.Series.agg; numpy.histogram (bin-zero constant domain)', n=len(values), positions=positions.ravel(),
        values=values, sourceIds=sourceIds, queries=queries, inputFloat32Sha256=hashFloats(positions, values), provenance=metadata)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent / 'fixtures')
    directory = parser.parse_args().output_dir
    directory.mkdir(parents=True, exist_ok=True)
    vertices = numpy.fromfile(DATA / 'chicago-tracts/vertices.bin', dtype='<f4').reshape(-1, 2)
    ringOffsets = numpy.fromfile(DATA / 'chicago-tracts/ringOffsets.bin', dtype='<u4')
    polygons = [Polygon(vertices[start:end]) for start, end in zip(ringOffsets[:-1], ringOffsets[1:])]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        adjacency = Queen.from_iterable(polygons).sparse.tocsr()
    assignments = numpy.fromfile(DATA / 'chicago-places/tract.bin', dtype='<u2')
    counts = numpy.bincount(assignments[assignments < len(polygons)], minlength=len(polygons))
    population = numpy.fromfile(DATA / 'chicago-tracts/population.bin', dtype='<f4')
    poverty = numpy.fromfile(DATA / 'chicago-tracts/poverty150.bin', dtype='<f4')
    categories = numpy.asarray(pandas.qcut(poverty, 4, labels=False))
    source = provenance(['chicago-tracts/' + file for file in ['manifest.json', 'vertices.bin', 'ringOffsets.bin', 'population.bin', 'poverty150.bin']] + ['chicago-places/manifest.json', 'chicago-places/tract.bin'])
    source.update(sourceTracts=len(polygons), sourcePlaces=len(assignments), unassignedPlaces=int((assignments >= len(polygons)).sum()), categoryConvention='poverty150 pandas.qcut quartiles')
    neighborhoods = []
    for normalize in [False, True]:
        for masked in [False, True]:
            mask = numpy.arange(len(polygons)) % 11 != 0 if masked else numpy.ones(len(polygons), dtype=bool)
            neighborhoods.append(makeNeighborhood(f'chicago-all-tracts-row-{normalize}-masked-{masked}', adjacency, counts, categories, mask, normalize, masked, .5, 64, False, source, True))
    # Stored slots deliberately preserve repeated neighbor 1, a zero edge, an
    # out-of-range edge and a self edge; no CSR canonicalization is applied.
    small = csr_matrix((numpy.array([1, 1, 2, 0, -1, 1, 1, 3, .5, .5, 1, 1, 1]),
        numpy.array([1, 1, 2, 3, 4, 99, 0, 4, 0, 2, 1, 5, 2]),
        numpy.array([0, 7, 8, 10, 11, 12, 13, 13, 13])), shape=(8, 8))
    smallValues = [2, 4, 8, numpy.nan, numpy.inf, -4, 12, 24]
    for normalize in [False, True]:
        for includeFocal, focalWeight, maximumNeighbors, truncated in [(False, 1, 64, False), (True, .5, 2, False), (True, 0, 64, False), (True, 1, 64, True)]:
            neighborhoods.append(makeNeighborhood(f'edge-row-{normalize}-focal-{includeFocal}-{focalWeight}-cap-{maximumNeighbors}-truncated-{truncated}', small,
                smallValues, [9, 2, 7, 2, 9, 3, 7, 5], [1, 1, 1, 1, 1, 1, 0, 1], normalize, includeFocal, focalWeight, maximumNeighbors, truncated, {'source': 'synthetic'}))
    writeFixture(directory, 'round5-neighborhood', dict(cases=neighborhoods))

    attributes = numpy.column_stack([counts, population, poverty])
    attributes = float32((attributes - attributes.mean(axis=0)) / attributes.std(axis=0))
    partitions = [makePartition('chicago-all-tracts-poverty-quartiles', attributes, categories, 7, adjacency, source)]
    ignoredLabels = categories.copy()
    ignoredLabels[::11] = NO_LABEL
    ignoredLabels[1::17] = 11
    partitions.append(makePartition('chicago-ignored-labels', attributes, ignoredLabels, 7, adjacency, source))
    # Directed slots include duplicates, self loops and zero weights. Magnitudes
    # do not define crossLinkFraction: every listed labelled link is counted.
    partitionGraph = csr_matrix((numpy.array([0, 2, 1, -1, 3, 1, 1]), numpy.array([0, 1, 1, 2, 0, 4, 3]), numpy.array([0, 3, 4, 5, 6, 7, 7])), shape=(6, 6))
    smallAttributes = [[1, 4], [3, 2], [5, 0], [7, -2], [9, -4], [11, -6]]
    for name, labels, values in [
        ('gapped-labels-ignored', [0, 0, 2, 2, 8, NO_LABEL], smallAttributes),
        ('single-region', [2]*6, smallAttributes),
        ('all-ignored', [NO_LABEL]*6, smallAttributes),
        ('all-singletons', list(range(6)), smallAttributes),
        ('zero-within', [0, 0, 2, 2, 2, 2], [[1, 1]]*2 + [[4, 4]]*4)]:
        partitions.append(makePartition(name, values, labels, 8, partitionGraph, {'source': 'synthetic'}))

    rawPositions = numpy.fromfile(DATA / 'chicago-nature/position.bin', dtype='<f4').reshape(-1, 2)
    transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:26916', always_xy=True)
    projected = numpy.column_stack(transformer.transform(rawPositions[:, 0], rawPositions[:, 1]))
    positions = float32(numpy.rint((projected - [440000, 4600000]) / 1000 * 1024) / 1024)
    days = numpy.fromfile(DATA / 'chicago-nature/time.bin', dtype='<u4') // 86400
    natureCategories = numpy.fromfile(DATA / 'chicago-nature/category.bin', dtype='u1')
    natureSource = provenance(['chicago-nature/' + file for file in ['manifest.json', 'position.bin', 'time.bin', 'category.bin']])
    natureSource.update(sourceRows=len(days), coordinateConvention='EPSG:26916 local km snapped 1/1024 km then f32')
    partitions.append(makePartition('chicago-all-nature-categories', positions, natureCategories, int(natureCategories.max())+2, None, natureSource))
    writeFixture(directory, 'round5-partition', dict(cases=partitions))

    selections = [dict(name='rectangle-fixed-histogram', kind='rectangle', coordinates=[0, 20, 12, 36], binCount=12, domain=[0, 360]),
        dict(name='concave-lasso-selection-histogram', kind='polygon', coordinates=[-5, 0, 15, 0, 15, 15, 5, 15, 5, 35, -5, 35], binCount=16),
        dict(name='circle-selection-histogram', kind='radius', coordinates=[5, 25, 8], binCount=16),
        dict(name='whole-extent', kind='rectangle', coordinates=[float(positions[:, 0].min()-1), float(positions[:, 1].min()-1), float(positions[:, 0].max()+1), float(positions[:, 1].max()+1)], binCount=0)]
    writeFixture(directory, 'round5-region-real', makeRegions('chicago-all-nature', positions, days, selections, natureSource))
    smallPositions = [[0, 0], [2, 0], [2, 2], [0, 2], [1, 1], [3, 1], [1, 3], [-1, 1], [1, -1], [4, 4], [numpy.nan, 0], [0, numpy.inf], [1, 2]]
    selections = [dict(name='rectangle-boundaries', kind='rectangle', coordinates=[0, 0, 2, 2], binCount=4, domain=[0, 8]),
        dict(name='concave-boundaries', kind='polygon', coordinates=[0, 0, 3, 0, 3, 1, 1, 1, 1, 3, 0, 3], binCount=4),
        dict(name='circle-boundaries', kind='radius', coordinates=[1, 1, 2], binCount=4, domain=[0, 8]),
        dict(name='empty', kind='rectangle', coordinates=[20, 20, 22, 22], binCount=4),
        dict(name='constant-domain', kind='rectangle', coordinates=[4, 4, 4, 4], binCount=4),
        dict(name='only-invalid-value', kind='rectangle', coordinates=[0, 2, 0, 2], binCount=4)]
    writeFixture(directory, 'round5-region-small', makeRegions('boundaries-empty-invalid', smallPositions, [0, 2, 4, numpy.nan, 6, numpy.inf, -2, 8, 10, 5, 20, 30, 4], selections, {'source': 'synthetic exact dyadic coordinates'}))
    print('cases', len(neighborhoods), len(partitions), 'region queries', 10)

if __name__ == '__main__': main()
