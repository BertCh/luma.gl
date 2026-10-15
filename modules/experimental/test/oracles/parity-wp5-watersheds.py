# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""pysheds.Grid.catchment parity on the identical supplied ESRI D8 graph.

Reuse all seven Round-3 complete domains, including the three catchments whose
membership/accumulation were checked against the FULL 2048-square Alps graph.
Their files and original provenance are hashed/retained; no cropped routing is
recomputed. 0 is terminal, 0xffffffff invalid; there are no unresolved/cyclic
directions. North-up, uniform metres, cell indices at pixel centers. The invalid
outer rim prevents pysheds' rim removal from discarding valid cells.

Without pour points, disjoint catchment masks for every terminal supply exact
outlet-cell labels. With pour points, nested catchments are ordered by cardinality:
the smallest enclosing catchment owns the cell (nearest downstream pour point).
Equal/duplicate masks use lowest input point index. Invalid/outside points are
ignored, and cells not in a supplied catchment get 0xffffffff. This is explicit
set composition of the established catchment function, not GPU pointer tracing.
"""
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy
from affine import Affine
from pysheds.grid import Grid
from pysheds.view import Raster, ViewFinder

OUTPUT = Path(__file__).parent / 'fixtures'
INVALID = 4294967295
VERSIONS = {name: importlib.metadata.version(name) for name in ['numpy', 'pysheds', 'affine']}
VERSIONS['python'] = platform.python_version()


def buildCase(sourceCase, sourcePath):
    height, width = sourceCase['height'], sourceCase['width']
    directions = numpy.array(sourceCase['directions'], dtype=numpy.int64).reshape(height, width)
    valid = directions != INVALID
    flow = Raster(numpy.where(valid, directions, -9999), viewfinder=ViewFinder(
        affine=Affine(1, 0, 0, 0, -1, 0), shape=directions.shape, nodata=-9999))
    grid = Grid.from_raster(flow)

    def catchment(cell):
        row, column = divmod(int(cell), width)
        result = numpy.asarray(grid.catchment(column, row, flow, xytype='index')).astype(bool)
        assert result[row, column]
        assert not numpy.any(result & ~valid)
        return result

    terminals = numpy.flatnonzero(directions.ravel() == 0)
    basins = numpy.full(directions.shape, INVALID, dtype=numpy.int64)
    masks = {int(cell): catchment(cell) for cell in terminals}
    for cell, membership in masks.items():
        assert numpy.all(basins[membership] == INVALID)
        basins[membership] = cell
    numpy.testing.assert_array_equal(basins != INVALID, valid)
    largestTerminal = max(masks, key=lambda cell: (int(masks[cell].sum()), -cell))
    candidates = numpy.flatnonzero(valid.ravel() & (directions.ravel() != 0))
    area = numpy.array(sourceCase['accumulation'], dtype=float)
    candidates = sorted(candidates.tolist(), key=lambda cell: (numpy.nan_to_num(area[cell]), cell))
    points = [largestTerminal]
    if candidates:
        points += [candidates[-1], candidates[len(candidates) // 2]]
    points += [largestTerminal, 0, directions.size + 7, INVALID]
    pointMasks = {}
    for pointIndex, cell in enumerate(points):
        if cell < directions.size and valid.ravel()[cell]:
            pointMasks[pointIndex] = catchment(cell)
    labels = numpy.full(directions.shape, INVALID, dtype=numpy.int64)
    for pointIndex in sorted(pointMasks, key=lambda index: (int(pointMasks[index].sum()), index)):
        membership = pointMasks[pointIndex]
        labels[membership & (labels == INVALID)] = pointIndex
    # Preserve the raw independent masks, so node checks can verify the composition.
    return {'name': sourceCase['name'], 'width': width, 'height': height,
            'source': sourceCase['source'], 'sourceFixture': sourcePath.name,
            'sourceFixtureSha256': hashlib.sha256(sourcePath.read_bytes()).hexdigest(),
            'directions': directions.ravel().tolist(), 'pourPoints': points,
            'basinLabels': basins.ravel().tolist(), 'pourPointLabels': labels.ravel().tolist(),
            'catchments': [{'pointIndex': index, 'mask': mask.astype(int).ravel().tolist()}
                           for index, mask in pointMasks.items()]}


def main():
    for topic in ['alps-0', 'alps-1', 'alps-2', 'synthetic']:
        sourcePath = OUTPUT / f'parity-wp5-hydrology-{topic}.json'
        source = json.loads(sourcePath.read_text())
        cases = [buildCase(case, sourcePath) for case in source['cases']]
        payload = {'versions': VERSIONS, 'conventions': __doc__,
                   'generatorSha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   'reference': 'pysheds.Grid.catchment',
                   'citation': 'https://mattbartos.com/pysheds/catchments.html', 'cases': cases}
        path = OUTPUT / f'parity-wp5-watersheds-{topic}.json'
        path.write_text(json.dumps(payload, separators=(',', ':')) + '\n')
        assert path.stat().st_size <= 1_500_000
        print(path.name, path.stat().st_size, 'cells', sum(case['width'] * case['height'] for case in cases), flush=True)


if __name__ == '__main__':
    main()
