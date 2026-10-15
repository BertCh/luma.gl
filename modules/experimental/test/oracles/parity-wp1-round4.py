# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent Round 4 references; run reference Python -I, optionally --output-dir.

Zonal: Shapely STRtree exact containment on uploaded f32 coordinates; smallest
feature row wins overlaps. Test open/closed boundaries, holes, multipolygons,
empty/one-value/zero-weight groups. pandas groupby supplies counts/sums/extrema/
mean/sample variance; weighted sample variance is numpy.cov(aweights, ddof=1).
Nonfinite values/weights are excluded from value reducers, not membership counts.
Full Chicago nature points and all 791 tract shells are projected EPSG:4326 ->
EPSG:26916, translated [440000,4600000]m, expressed in km, snapped to 1/1024 km,
then f32 BEFORE Shapely. This tests the quantized geometry, not original GIS data.
Time values are local-clock day-of-year (floor seconds/86400), no timezone claim.

Focal: decode full 2048x2048 Terrarium DEM, take [768:1280,768:1280]. Eight 64-row
stripes retain ceil(radius) halos. scipy.ndimage.generic_filter supplies all
window reductions in f64 using constant NaN padding (clipped windows), arbitrary
square/circle footprints, ddof=0. All outputs, including halo outputs, are stored;
every core cell is also compared to the unpartitioned 512x512 SciPy result at
generation time. Edge-of-halo outputs intentionally have local clipped support.
Masks/NaN/sentinel filter samples; infinities remain valid samples; minimum count and center propagation affect
floats but not counts. Negative radius means empty footprint. No RNG.

Geographic: all nature observations grouped by actual category, plus an empty
group; all rows or research-grade mask with explicit scenario weights 1+introduced
(not claimed scientific sampling weights). pointpats mean_center,
weighted_mean_center, std_distance and ellipse(method='yuill', crimestatCorr=False,
degfreedCorr=False): population convention, no n/(n-2) correction. Weighted
standard distance uses trace(numpy.cov(aweights,ddof=0)), cross-checked unweighted
against pointpats. Ellipse returns major/minor/major angle; convert to short-axis
angle modulo pi and apply GPU convention scale separately. Compare angles via
cos(2a),sin(2a) to avoid axial wrapping. Zero-spread angle is undefined and omitted
from angle comparison, but all axes, centers and generated vertices are checked.
Median centers/directional means are deliberately not covered. No CPU GPU-model
implementation is used. Every output float remains f64; JSON is losslessly zlib
compressed to respect the 1.5MB fixture cap, never rounded to fit or subsampled.

https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.var.html
https://numpy.org/doc/stable/reference/generated/numpy.cov.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.generic_filter.html
https://pysal.org/pointpats/generated/pointpats.ellipse.html
"""
import argparse
import base64
import hashlib
import json
import platform
from pathlib import Path
import warnings
import zlib
import numpy as numpy
import pandas
import scipy
from scipy.ndimage import generic_filter
import shapely
from shapely import Polygon, MultiPolygon, STRtree
import pyproj
import pointpats
from pointpats import centrography
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / 'examples/deck/spatial-analysis-showcase/public/data'
NO_FEATURE = 4294967295
VERSIONS = dict(python=platform.python_version(), numpy=numpy.__version__, pandas=pandas.__version__, scipy=scipy.__version__, shapely=shapely.__version__, pyproj=pyproj.__version__, pointpats=pointpats.__version__, pillow=Image.__version__)


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
    return {'sourceSha256': {name: hashlib.sha256((DATA / name).read_bytes()).hexdigest() for name in files}}


def project(points):
    transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:26916', always_xy=True)
    positions = numpy.column_stack(transformer.transform(points[:, 0], points[:, 1]))
    return (numpy.rint((positions - [440000, 4600000]) / 1000 * 1024) / 1024).astype(numpy.float32).astype(float)


def makeZonal(name, features, positions, values, weights, boundaryRule, metadata):
    positions = numpy.asarray(positions, dtype=numpy.float32).astype(float)
    values = numpy.asarray(values, dtype=numpy.float32).astype(float)
    weights = None if weights is None else numpy.asarray(weights, dtype=numpy.float32).astype(float)
    geometries = [MultiPolygon([Polygon(polygon[0], polygon[1:]) for polygon in feature]) for feature in features]
    pairs = STRtree(geometries).query(shapely.points(positions), predicate='covered_by' if boundaryRule == 'closed' else 'within')
    membership = numpy.full(len(positions), NO_FEATURE, dtype=numpy.uint32)
    numpy.minimum.at(membership, pairs[0], pairs[1].astype(numpy.uint32))
    featureCount = len(features)
    frame = pandas.DataFrame({'feature': membership, 'value': values})
    assigned = frame[frame.feature < featureCount]
    counts = assigned.groupby('feature').size().reindex(range(featureCount), fill_value=0).to_numpy()
    valid = numpy.isfinite(values) & (membership < featureCount)
    if weights is not None: valid &= numpy.isfinite(weights)
    validFrame = frame[valid].copy()
    grouped = validFrame.groupby('feature').value
    valueCounts = grouped.count().reindex(range(featureCount), fill_value=0).to_numpy()
    minima = grouped.min().reindex(range(featureCount)).to_numpy()
    maxima = grouped.max().reindex(range(featureCount)).to_numpy()
    weightSums = numpy.zeros(featureCount)
    if weights is None:
        sums = grouped.sum().reindex(range(featureCount), fill_value=0).to_numpy()
        means = grouped.mean().reindex(range(featureCount)).to_numpy()
        variances = grouped.var(ddof=1).reindex(range(featureCount)).to_numpy()
    else:
        validFrame['weight'] = weights[valid]
        validFrame['weighted'] = values[valid] * weights[valid]
        aggregates = validFrame.groupby('feature')[['weight', 'weighted']].sum().reindex(range(featureCount), fill_value=0)
        sums = aggregates.weighted.to_numpy(); weightSums = aggregates.weight.to_numpy()
        means = numpy.divide(sums, weightSums, out=numpy.full(featureCount, numpy.nan), where=weightSums != 0)
        variances = numpy.full(featureCount, numpy.nan)
        for feature, group in validFrame.groupby('feature'):
            if len(group) > 1 and group.weight.sum() > 0 and numpy.count_nonzero(group.weight) > 1:
                variances[feature] = numpy.cov(group.value, aweights=group.weight, ddof=1)
    areas = shapely.area(geometries)
    expected = dict(counts=counts, valueCounts=valueCounts, sums=sums, means=means, minima=minima, maxima=maxima, variances=variances, standardDeviations=numpy.sqrt(variances), featureAreas=areas, densities=counts / areas, pointFeatureRows=membership, overflow=[0], uncertainCount=[0])
    if weights is not None: expected.update(weightSums=weightSums, weightedDensities=weightSums / areas)
    finite = (counts > 0) & numpy.isfinite(means)
    expected['extent'] = [means[finite].min(), means[finite].max()] if finite.any() else [0, 0]
    return dict(name=name, reference='shapely.STRtree.query; pandas.DataFrame.groupby; numpy.cov', n=len(positions), featureCount=featureCount, features=features, positions=positions.ravel(), values=values, weights=[] if weights is None else weights, boundaryRule=boundaryRule, expected=expected, provenance=metadata, inputFloat32Sha256=hashlib.sha256(positions.astype('<f4').tobytes()+values.astype('<f4').tobytes()).hexdigest())


def generateZonal(directory, positions, source):
    vertices = project(numpy.fromfile(DATA / 'chicago-tracts/vertices.bin', dtype='<f4').reshape(-1, 2))
    offsets = numpy.fromfile(DATA / 'chicago-tracts/ringOffsets.bin', dtype='<u4')
    features = [[[[point.tolist() for point in vertices[start:end]]]] for start, end in zip(offsets[:-1], offsets[1:])]
    values = numpy.fromfile(DATA / 'chicago-nature/time.bin', dtype='<u4') // 86400
    metadata = {**provenance(['chicago-nature/position.bin', 'chicago-nature/manifest.json', 'chicago-tracts/vertices.bin', 'chicago-tracts/ringOffsets.bin', 'chicago-tracts/manifest.json', 'chicago-nature/time.bin']), 'selection': 'all 43557 observations and all 791 tracts', 'coordinateConvention': 'EPSG:26916 local km snapped 1/1024 km then f32'}
    writeFixture(directory, 'zonal-real', makeZonal('chicago-nature-all-tracts', features, positions, values, None, 'closed', metadata))
    shell = [[0, 0], [6, 0], [6, 6], [0, 6]]
    hole = [[2, 2], [2, 4], [4, 4], [4, 2]]
    small = [[ [shell, hole] ], [ [[[5, 0], [9, 0], [9, 4], [5, 4]]] ], [ [[[10, 0], [12, 0], [12, 2], [10, 2]]], [[[10, 4], [12, 4], [12, 6], [10, 6]]] ], [ [[[20, 0], [21, 0], [21, 1], [20, 1]]] ], [ [[[30, 0], [32, 0], [32, 2], [30, 2]]] ]]
    points = [[1, 1], [1, 2], [3, 3], [2, 3], [5.5, 1], [6, 1], [9, 4], [11, 1], [11, 5], [31, 1], [50, 50], [0, 0], [7, 2], [7, 3]]
    values = [2, 4, 99, 6, 8, 10, 12, 20, numpy.nan, 5, 55, numpy.inf, 14, 16]
    weights = [1, 2, 1, 0.5, 2, 3, 0, 1, 2, 0, 1, 1, numpy.nan, 1]
    cases = [makeZonal(f'holes-overlaps-{rule}-{mode}', small, points, values, weights if mode == 'weighted' else None, rule, {'source': 'synthetic exact dyadic geometry; empty feature 3, zero-weight feature 4'}) for rule in ['open', 'closed'] for mode in ['unweighted', 'weighted']]
    writeFixture(directory, 'zonal-small', dict(cases=cases))


def computeFocal(values, validity, settings, maximumRadius, noDataValue):
    valid = ~numpy.isnan(values) & (values != noDataValue) & (validity != 0)
    samples = numpy.where(valid, values, numpy.nan)
    radius = min(settings['radius'], maximumRadius)
    if radius < 0:
        return {**{key: numpy.full(values.shape, numpy.nan) for key in ['mean', 'sum', 'min', 'max', 'range', 'standardDeviation']}, 'counts': numpy.zeros(values.shape, dtype=int)}
    extent = int(numpy.floor(radius))
    rows, columns = numpy.mgrid[-extent:extent+1, -extent:extent+1]
    footprint = rows**2 + columns**2 <= radius**2 if settings.get('shape') == 'circle' else numpy.ones(rows.shape, dtype=bool)
    reducers = dict(mean=numpy.mean, sum=numpy.sum, min=numpy.min, max=numpy.max, range=numpy.ptp, standardDeviation=numpy.std)
    expected = {}
    counts = generic_filter(valid.astype(float), numpy.sum, footprint=footprint, mode='constant', cval=0).astype(int)
    usable = counts >= max(settings.get('minimumCount', 1), 1)
    if settings.get('propagateCenterNoData'): usable &= valid
    for key, reducer in reducers.items():
        def reduce(window, reducer=reducer):
            finite = window[~numpy.isnan(window)]
            return reducer(finite) if len(finite) else numpy.nan
        result = generic_filter(samples, reduce, footprint=footprint, mode='constant', cval=numpy.nan)
        expected[key] = numpy.where(usable, result, numpy.nan)
    return {**expected, 'counts': counts}


def makeFocal(name, values, validity, settings, maximumRadius, metadata):
    values = numpy.asarray(values, dtype=numpy.float32).astype(float)
    expected = computeFocal(values, validity, settings, maximumRadius, -9999)
    finite = values[numpy.isfinite(values) & (values != -9999) & (validity != 0)]
    return dict(name=name, reference='scipy.ndimage.generic_filter; numpy reducers ddof=0', n=values.size, width=values.shape[1], height=values.shape[0], maximumRadius=maximumRadius, settings=settings, noDataValue=-9999, values=values.ravel(), validity=validity.ravel(), expected={key: value.ravel() for key, value in expected.items()}, rangeScale=max(1.,float(numpy.ptp(finite))) if len(finite) else 1., provenance=metadata, inputFloat32Sha256=hashlib.sha256(values.astype('<f4').tobytes()).hexdigest())


def generateFocal(directory):
    image = numpy.asarray(Image.open(DATA / 'grand-canyon-dem/dem.png')).astype(float)
    elevation = image[:, :, 0] * 256 + image[:, :, 1] + image[:, :, 2] / 256 - 32768
    parent = elevation[768:1280, 768:1280]
    metadata = {**provenance(['grand-canyon-dem/dem.png', 'grand-canyon-dem/manifest.json']), 'sourceShape': list(elevation.shape), 'crop': [768, 768, 512, 512], 'unit': 'meters', 'boundary': 'constant NaN; clipped local footprint', 'selection': 'all 262144 crop cells, eight stripes with halos; no sampling'}
    for mode, settings in [('square', {'radius': 1, 'shape': 'square'}), ('circle', {'radius': 2.5, 'shape': 'circle'})]:
        full = computeFocal(parent, numpy.ones(parent.shape), settings, 3, -9999)
        for stripe in range(8):
            start = stripe * 64; stop = start + 64
            top = max(0, start-3); bottom = min(512, stop+3)
            values = parent[top:bottom]
            case = makeFocal(f'grand-canyon-512-{mode}-stripe-{stripe}', values, numpy.ones(values.shape, dtype=int), settings, 3, {**metadata, 'stripe': [top, bottom], 'core': [start-top, stop-top]})
            for key in full:
                actual = numpy.asarray(case['expected'][key]).reshape(values.shape)[start-top:stop-top]
                numpy.testing.assert_array_equal(actual, full[key][start:stop])
            case['coreSha256'] = {key: hashlib.sha256(full[key][start:stop].astype('<f8').tobytes()).hexdigest() for key in full}
            writeFixture(directory, f'focal-{mode}-{stripe}', case)
    values = numpy.arange(35, dtype=float).reshape(5, 7) + 1000
    values[1, 1] = numpy.nan; values[2, 3] = -9999; values[4, 0] = numpy.inf
    validity = numpy.ones(values.shape, dtype=int); validity[0, 1:4] = 0
    settings = [dict(radius=0), dict(radius=1), dict(radius=2.5, shape='circle'), dict(radius=2, minimumCount=24), dict(radius=1, propagateCenterNoData=True), dict(radius=-1), dict(radius=8, shape='square')]
    cases = [makeFocal(f'edge-nodata-policy-{index}', values, validity, setting, 3, {'source': 'synthetic; large offset, NaN/Inf/sentinel/mask, clipped 5x7 edges'}) for index, setting in enumerate(settings)]
    writeFixture(directory, 'focal-small', dict(cases=cases))


def makeGeographic(name, positions, groups, weights, mask, groupCount, metadata):
    positions = numpy.asarray(positions, dtype=numpy.float32).astype(float)
    weights = numpy.asarray(weights, dtype=numpy.float32).astype(float)
    groups = numpy.asarray(groups); mask = numpy.asarray(mask)
    expected = {key: [] for key in ['counts', 'weightSums', 'meanCenters', 'standardDistances', 'ellipses', 'angleDefined']}
    for group in range(groupCount):
        included = (groups == group) & (mask != 0) & numpy.isfinite(positions).all(axis=1) & numpy.isfinite(weights) & (weights > 0)
        points = positions[included]; selectedWeights = weights[included]
        expected['counts'].append(len(points)); expected['weightSums'].append(selectedWeights.sum())
        if not len(points):
            expected['meanCenters'].extend([numpy.nan]*2); expected['standardDistances'].append(numpy.nan); expected['ellipses'].extend([numpy.nan]*3); expected['angleDefined'].append(False)
            continue
        center = centrography.weighted_mean_center(points, selectedWeights)
        covariance = numpy.cov(points.T, aweights=selectedWeights, ddof=0) if len(points)>1 else numpy.zeros((2,2))
        distance = numpy.sqrt(numpy.trace(covariance))
        if numpy.all(selectedWeights == 1):
            numpy.testing.assert_allclose(center, centrography.mean_center(points), atol=1e-12)
            numpy.testing.assert_allclose(distance, centrography.std_distance(points), atol=1e-10)
        major, minor, angle = centrography.ellipse(points, weights=selectedWeights, method='yuill', crimestatCorr=False, degfreedCorr=False)
        eigenvalues, eigenvectors = numpy.linalg.eigh(covariance)
        numpy.testing.assert_allclose([minor**2, major**2], eigenvalues, atol=1e-10)
        defined = eigenvalues[1] - eigenvalues[0] > 1e-12
        shortAngle = (angle + numpy.pi/2 + numpy.pi/2) % numpy.pi - numpy.pi/2
        if shortAngle <= -numpy.pi/2: shortAngle += numpy.pi
        if not defined: shortAngle = numpy.pi/2  # declared representation for physically undefined orientation
        if defined:
            direction = numpy.array([numpy.cos(shortAngle), numpy.sin(shortAngle)])
            numpy.testing.assert_allclose(abs(direction @ eigenvectors[:,0]), 1, atol=1e-10)
        expected['meanCenters'].extend(center); expected['standardDistances'].append(distance); expected['ellipses'].extend([shortAngle, minor, major]); expected['angleDefined'].append(bool(defined))
    return dict(name=name, reference='pointpats.centrography.mean_center/weighted_mean_center/std_distance/ellipse; numpy.cov', n=len(positions), positions=positions.ravel(), groups=groups, weights=weights, mask=mask, groupCount=groupCount, expected=expected, provenance=metadata, inputFloat32Sha256=hashlib.sha256(positions.astype('<f4').tobytes()).hexdigest())


def generateGeographic(directory, positions, source):
    groups = numpy.fromfile(DATA / 'chicago-nature/category.bin', dtype='u1')
    grade = numpy.fromfile(DATA / 'chicago-nature/researchGrade.bin', dtype='u1')
    introduced = numpy.fromfile(DATA / 'chicago-nature/introduced.bin', dtype='u1')
    metadata = {**provenance(['chicago-nature/position.bin', 'chicago-nature/manifest.json', 'chicago-nature/category.bin','chicago-nature/researchGrade.bin','chicago-nature/introduced.bin']), 'selection': 'all rows, original order; masks applied by GPU and reference', 'coordinateConvention': 'EPSG:26916 local km snapped 1/1024 km then f32', 'attribution': 'iNaturalist contributors; CC0/CC BY/CC BY-NC per observation; non-commercial use'}
    for mode, weights, mask in [('all', numpy.ones(len(positions)), numpy.ones(len(positions), dtype=int)), ('weighted-masked', 1+introduced, grade)]:
        writeFixture(directory, f'geographic-{mode}', makeGeographic(f'chicago-nature-{mode}', positions, groups, weights, mask, 11, metadata))
    points = [[0,0],[4,1],[7,5],[2,2],[10,0],[12,0],[14,0],[50,50],[50,50],[numpy.nan,1],[2,3],[6,8],[9,9],[3,3],[4,4]]
    cases = [makeGeographic('degenerate-invalid-weights-island', points, [0,0,0,0,1,1,1,2,2,0,0,0,0,7,0], [1,2,3,4,1,2,1,1,1,1,0,-1,numpy.nan,1,1], [1]*14+[0], 4, {'source':'synthetic: anisotropic, collinear, coincident, empty; NaN position/weight, zero/negative weight, invalid group, mask'})]
    writeFixture(directory, 'geographic-small', dict(cases=cases))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent / 'fixtures'); arguments = parser.parse_args(); arguments.output_dir.mkdir(parents=True, exist_ok=True)
    positions = project(numpy.fromfile(DATA / 'chicago-nature/position.bin', dtype='<f4').reshape(-1,2))
    source = provenance(['chicago-nature/position.bin', 'chicago-nature/manifest.json'])
    generateZonal(arguments.output_dir, positions, source)
    generateGeographic(arguments.output_dir, positions, source)
    generateFocal(arguments.output_dir)


if __name__ == '__main__': main()
