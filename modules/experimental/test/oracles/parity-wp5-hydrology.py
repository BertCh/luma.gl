# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent HAND, hydrologic indices and Strahler references.

pysheds.Grid.compute_hand(return_index=False/True) supplies full HAND and
first-drainage indices. Grid.flowdir, accumulation and cell_slopes supply D8
area and maximum descent, then NumPy applies Beven/Kirkby (1979) ln(a/tan(beta))
and Moore/Grayson/Ladson (1991) a*tan(beta), a=A/sqrt(dx*dy). This COMPOSITE
reference deliberately does not call Horn/GDAL slope: that is a different
operator. All inputs/calibration are float32, reference math float64, north-up,
ESRI D8 codes, uniform metres, closed terminals 0, invalid direction 0xffffffff.
Nonfinite elevation/area gives NaN; slope floor and zero-area log limits explicit.

Strahler 1957 recurrence is independently evaluated over a NetworkX DAG
(topological_sort/predecessors), labelled COMPOSITE. pysheds 0.5 stream_order
is also called and its differences retained: its min_order==max_order rule is
incorrect for donor orders [2,2,1] (requires two maxima, not all equal). We do
not patch or silently use that result as an oracle. Binary nonterminal cells are checked
against pysheds (terminal self-receivers can spuriously increment order); mixed junctions are checked against the published definition.

Real data: decode full 2048x2048 showcase Alps DEM, run pysheds flowdir and
accumulation on the FULL graph (unfilled raw terrain, pits/flats are terminals).
Select three complete terminal catchments, nearest dataset center, 256..1800
cells. No arbitrary cropped global reference: catchment membership and every
accumulation value are verified against the full graph before export. Outside
catchment is nodata; rectangular bounds have a one-cell invalid rim. Each
fixture contains every cell including the rim. No seam or full-DEM parity claim.
Pysheds pops its outer rim: explicit invalid padding avoids losing valid cells.
NumPy 2.5 removed in1d; the process-local shim uses its documented replacement
isin (ravel inputs), without altering any pysheds numerical kernel.
"""
import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path

import networkx
import numpy
import rasterio
from affine import Affine
# Compatibility only, equivalent public NumPy operation, no algorithm changes.
if not hasattr(numpy, 'in1d'):
    numpy.in1d = lambda first, second: numpy.isin(numpy.ravel(first), numpy.ravel(second))
from pysheds.grid import Grid
from pysheds.view import Raster, ViewFinder

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).parent / 'fixtures'
DATA = ROOT / 'examples/deck/spatial-analysis-showcase/public/data/alps-dem'
INVALID = 4294967295
DIRECTIONS = {1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1),
              16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1)}
VERSIONS = {name: importlib.metadata.version(name) for name in ['numpy', 'pysheds', 'networkx', 'rasterio', 'affine']}
VERSIONS['python'] = platform.python_version()
VERSIONS['gdal'] = rasterio.__gdal_version__


def makeRaster(values, spacing, nodata):
    return Raster(numpy.array(values, copy=True), viewfinder=ViewFinder(
        affine=Affine(spacing[0], 0, 0, 0, -spacing[1], 0), shape=values.shape, nodata=nodata))


def encode(value):
    if isinstance(value, dict):
        return {key: encode(entry) for key, entry in value.items()}
    if isinstance(value, (list, tuple, numpy.ndarray)):
        return [encode(entry) for entry in value]
    if isinstance(value, (numpy.integer, numpy.bool_)):
        return int(value)
    if isinstance(value, (float, numpy.floating)):
        return float(value) if math.isfinite(value) else None if math.isnan(value) else 'Infinity' if value > 0 else '-Infinity'
    return value


def tolerance(values):
    finite = values[numpy.isfinite(values)]
    return {'profile': 'R', 'absolute': 1e-5 * max(1, float(numpy.ptp(finite)) if finite.size else 0), 'relative': 1e-4}


def computeOrder(directions, streams):
    graph = networkx.DiGraph()
    height, width = directions.shape
    graph.add_nodes_from(numpy.flatnonzero(streams))
    for donor in list(graph):
        row, column = divmod(int(donor), width)
        if int(directions[row, column]) not in DIRECTIONS:
            continue
        rowStep, columnStep = DIRECTIONS[int(directions[row, column])]
        targetRow, targetColumn = row + rowStep, column + columnStep
        if 0 <= targetRow < height and 0 <= targetColumn < width and streams[targetRow, targetColumn]:
            graph.add_edge(donor, targetRow * width + targetColumn)
    order = numpy.zeros(directions.size, dtype=numpy.int64)
    for target in networkx.topological_sort(graph):
        donors = [order[donor] for donor in graph.predecessors(target)]
        maximum = max(donors, default=1)
        order[target] = maximum + (donors.count(maximum) >= 2)
    return order.reshape(directions.shape)


def buildCase(name, elevation, spacing, source, directions=None, streams=None, minimumSlope=0.001, areaOverride=None):
    elevation = numpy.asarray(elevation, dtype=numpy.float32).astype(float)
    spacing = numpy.asarray(spacing, dtype=numpy.float32).astype(float)
    minimumSlope = float(numpy.float32(minimumSlope))
    valid = numpy.isfinite(elevation)
    elevationRaster = makeRaster(elevation, spacing, numpy.nan)
    grid = Grid.from_raster(elevationRaster)
    if directions is None:
        directions = numpy.asarray(grid.flowdir(elevationRaster, pits=0, flats=0, nodata_out=-9999)).copy()
    directions = numpy.asarray(directions, dtype=numpy.int64).copy()
    directions[~valid] = -9999
    flow = makeRaster(directions, spacing, -9999)
    counts = numpy.asarray(grid.accumulation(flow)).copy()
    if streams is None:
        streams = (counts >= 8) & valid
    streams = numpy.asarray(streams, dtype=bool) & valid
    channel = makeRaster(streams, spacing, False)
    hand = numpy.asarray(grid.compute_hand(flow, elevationRaster, channel)).copy()
    drainage = numpy.asarray(grid.compute_hand(flow, elevationRaster, channel, return_index=True)).copy()
    hand[~valid] = numpy.nan
    drainage[~valid] = -1
    drainage = numpy.where(drainage < 0, INVALID, drainage)
    order = computeOrder(directions, streams)
    pyshedsOrder = numpy.asarray(grid.stream_order(flow, channel)).copy()
    pyshedsOrder[~streams] = 0
    disagreement = numpy.flatnonzero(order.ravel() != pyshedsOrder.ravel())
    print(name, 'cells', elevation.size, 'streams', int(streams.sum()), 'order max', int(order.max()),
          'pysheds Strahler disagreements', len(disagreement), flush=True)
    # The scalar fixtures cover the known heterogeneous three-donor discrepancy.
    if 'binary' in name:
        numpy.testing.assert_array_equal(order[directions > 0], pyshedsOrder[directions > 0])
    # For manually supplied routing, use a fresh steepest-descent flow direction
    # for indices; these are elevation indices, independent of stream graph choice.
    steepestFlow = grid.flowdir(elevationRaster, pits=0, flats=0, nodata_out=-9999)
    slope = numpy.asarray(grid.cell_slopes(elevationRaster, steepestFlow)).copy()
    slope = numpy.maximum(numpy.nan_to_num(slope, nan=0), minimumSlope)
    area = numpy.asarray(counts * numpy.prod(spacing), dtype=numpy.float32).astype(float)
    if areaOverride is not None:
        area = numpy.asarray(areaOverride, dtype=numpy.float32).astype(float)
    catchment = area / numpy.sqrt(numpy.prod(spacing))
    with numpy.errstate(divide='ignore', invalid='ignore'):
        wetness = numpy.where(catchment <= 0, -numpy.inf, numpy.where(slope == 0, numpy.inf, numpy.log(catchment / slope)))
    expected = {'heightAboveDrainage': hand, 'drainageCells': drainage, 'streamOrder': order,
                'slope': slope, 'specificCatchmentArea': catchment, 'wetnessIndex': wetness,
                'streamPowerIndex': catchment * slope}
    for key in ['slope', 'specificCatchmentArea', 'wetnessIndex', 'streamPowerIndex']:
        expected[key][~valid | ~numpy.isfinite(area)] = numpy.nan
    for index in numpy.flatnonzero(streams):
        assert hand.ravel()[index] == 0 and drainage.ravel()[index] == index
    fixture = {'name': name, 'width': elevation.shape[1], 'height': elevation.shape[0],
               'cellSize': spacing, 'minimumSlope': minimumSlope, 'source': source,
               'elevation': elevation.ravel(), 'directions': numpy.where(directions == -9999, INVALID, directions).ravel(),
               'streams': streams.astype(int).ravel(), 'accumulation': area.ravel(),
               'expected': {key: values.ravel() for key, values in expected.items()},
               'pyshedsOrder': pyshedsOrder.ravel(), 'pyshedsOrderMismatchCells': disagreement,
               'tolerances': {key: tolerance(values) for key, values in expected.items() if key not in ['drainageCells', 'streamOrder']}}
    return fixture, counts


def writeFixture(topic, cases):
    payload = {'versions': VERSIONS, 'generatorSha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'conventions': __doc__, 'citations': [
                   'https://github.com/mdbartos/pysheds/blob/v0.5/pysheds/sgrid.py',
                   'https://doi.org/10.1029/TR038i006p00913',
                   'https://doi.org/10.1080/02626667909491834',
                   'https://doi.org/10.1002/hyp.3360050103'], 'cases': cases}
    path = OUTPUT / f'parity-wp5-hydrology-{topic}.json'
    path.write_text(json.dumps(encode(payload), separators=(',', ':'), allow_nan=False) + '\n')
    assert path.stat().st_size <= 1_500_000, (path.name, path.stat().st_size)
    print(path.name, path.stat().st_size, flush=True)


def main():
    # Complete synthetic domains, invalid outer rim, anisotropic metric geometry.
    rows, columns = numpy.indices((33, 65))
    elevation = 1000 + 2 * numpy.abs(columns - 32) + (32 - rows) * 3
    elevation = elevation.astype(float)
    elevation[[0, -1], :] = numpy.nan
    elevation[:, [0, -1]] = numpy.nan
    elevation[9:12, 20:23] = numpy.nan
    cases = [buildCase('valley-nodata-anisotropic', elevation, [2, 3], {'kind': 'complete synthetic domain'})[0]]
    flat = numpy.pad(numpy.ones((5, 7)) * 10, 1, constant_values=numpy.nan)
    areas = numpy.ones_like(flat) * 16
    areas[2, 2] = 0
    areas[2, 3] = numpy.nan
    cases.append(buildCase('flat-zero-area-no-stream', flat, [4, 4], {'kind': 'complete synthetic domain'},
                           streams=numpy.zeros_like(flat), minimumSlope=0, areaOverride=areas)[0])
    # Binary confluence with an optional third, lower-order donor into [2,2].
    directions = numpy.zeros((9, 11), dtype=numpy.int64)
    for row, column, code in [(2, 2, 2), (2, 4, 8), (3, 3, 2), (4, 4, 2),
                              (2, 6, 2), (2, 8, 8), (3, 7, 8), (4, 6, 8),
                              (5, 5, 4), (6, 5, 4)]:
        directions[row, column] = code
    channel = directions != 0
    channel[7, 5] = True
    terrain = numpy.broadcast_to((100 - numpy.arange(9) * 5)[:, None], (9, 11)).astype(float).copy()
    terrain[[0, -1], :] = numpy.nan
    terrain[:, [0, -1]] = numpy.nan
    for mixed in [False, True]:
        if mixed:
            directions[4, 5] = 4
            channel[4, 5] = True
        case, _ = buildCase('mixed-2-2-1-junction' if mixed else 'binary-junction', terrain, [1, 1],
                            {'kind': 'complete synthetic stream tree'}, directions, channel)
        assert case['expected']['streamOrder'].reshape(9, 11)[5, 5] == 3
        cases.append(case)
    writeFixture('synthetic', cases)

    manifest = json.loads((DATA / 'manifest.json').read_text())
    with rasterio.open(DATA / 'dem.png') as dataset:
        channels = dataset.read().astype(float)
    full = channels[0] * 256 + channels[1] + channels[2] / 256 - 32768
    del channels
    spacing = [float(numpy.float32(manifest['raster']['cellSizeGroundM']))] * 2
    # Pad the whole raster to preserve original edge cells in pysheds.
    elevationRaster = makeRaster(numpy.pad(full, 1, constant_values=numpy.nan), spacing, numpy.nan)
    grid = Grid.from_raster(elevationRaster)
    print('Computing full 2048x2048 D8 graph and accumulation', flush=True)
    flow = grid.flowdir(elevationRaster, pits=0, flats=0, nodata_out=-9999)
    counts = numpy.asarray(grid.accumulation(flow)).copy()
    candidates = numpy.argwhere((flow == 0) & (counts >= 256) & (counts <= 1800))
    candidates = sorted(candidates.tolist(), key=lambda location: ((location[0] - 1025)**2 + (location[1] - 1025)**2, location))
    selected = 0
    for row, column in candidates:
        membership = numpy.asarray(grid.catchment(column, row, flow, xytype='index')).astype(bool)
        memberRows, memberColumns = numpy.nonzero(membership)
        assert membership.sum() == counts[row, column]
        rowStart, rowEnd = int(memberRows.min()) - 1, int(memberRows.max()) + 2
        columnStart, columnEnd = int(memberColumns.min()) - 1, int(memberColumns.max()) + 2
        if rowEnd - rowStart > 100 or columnEnd - columnStart > 100:
            continue
        window = numpy.s_[rowStart:rowEnd, columnStart:columnEnd]
        localMembership = membership[window]
        elevation = numpy.where(localMembership, elevationRaster[window], numpy.nan)
        source = {'kind': 'complete catchment selected from full raw DEM flow graph',
                  'path': str((DATA / 'dem.png').relative_to(ROOT)),
                  'sha256': hashlib.sha256((DATA / 'dem.png').read_bytes()).hexdigest(),
                  'manifestSha256': hashlib.sha256((DATA / 'manifest.json').read_bytes()).hexdigest(),
                  'fullShape': list(full.shape), 'fullGraphCellCount': full.size,
                  'outletOriginalRowColumn': [row - 1, column - 1],
                  'windowOriginalRowColumn': [rowStart - 1, columnStart - 1],
                  'catchmentCells': int(membership.sum()), 'fullGraphAccumulationVerified': True}
        case, localCounts = buildCase(f'alps-complete-catchment-{selected}', elevation, spacing, source,
                                     numpy.where(localMembership, flow[window], -9999))
        numpy.testing.assert_array_equal(localCounts[localMembership], counts[window][localMembership])
        writeFixture(f'alps-{selected}', [case])
        selected += 1
        if selected == 3:
            break
    assert selected == 3


if __name__ == '__main__':
    main()
