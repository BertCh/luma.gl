// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {cellToParent, getPentagons, getRes0Cells, latLngToCell, cellToChildren} from 'h3-js';
import {expect, it} from 'vitest';
import {
  GPUCellAggregation,
  GPUCellLevelSelection,
  GPUCellPyramid,
  GPUCellRollup,
  type GPUCellAggregationProps,
  type GPUCellTable
} from '../../../src/gpu-spatial-analysis/cell-aggregation';
import {getCellKeyLayout} from '../../../src/gpu-spatial-analysis/cell-aggregation/cell-keys';
import {
  createNullWebGPUDevice,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';
import {
  getQuadbinTileX,
  getQuadbinTileY,
  getQuadbinTileYFloat64,
  getScaledValue,
  h3ToBigInt,
  quadbinCellToParent,
  quadbinCellToTile,
  quadbinIsValidCell,
  quadbinPointToCell,
  quadbinTileToCell
} from './cell-aggregation-oracle';
import {createPointPositions, createRandom, getEdgeCasePoints} from './cell-aggregation-points';

let serial = 0;

function createTable(graph: GPUCommandGraph, capacity: number, columns = true): GPUCellTable {
  const view = <Format extends 'uint32' | 'float32' | 'uint32x2'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  return {
    cells: view('uint32x2', capacity),
    counts: view('uint32', capacity),
    ...(columns
      ? {
          sums: view('uint32x2', capacity),
          sumValues: view('float32', capacity),
          minimums: view('float32', capacity),
          maximums: view('float32', capacity)
        }
      : {}),
    status: {
      count: view('uint32', 1),
      overflow: view('uint32', 1)
    }
  };
}

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUCellAggregationProps> = {}
): GPUCellAggregationProps {
  return {
    family: 'quadbin',
    resolution: 8,
    positions: createTransientView(graph, `positions-${serial++}`, 'float32x2', 10),
    values: createTransientView(graph, `values-${serial++}`, 'float32', 10),
    output: createTable(graph, 16),
    ...overrides
  };
}

it('quadbin oracle matches published quadbin-js values', () => {
  // quadbin-js README: tileToCell({x: 0, y: 0, z: 0}).
  expect(quadbinTileToCell(0, 0, 0)).toBe(5192650370358181887n);
  // quadbin-py README: point_to_cell(-3.7038, 40.4168, 4).
  expect(quadbinPointToCell(-3.7038, 40.4168, 4)).toBe(5207251884775047167n);
  expect(quadbinCellToTile(5207251884775047167n)).toEqual({x: 7, y: 6, z: 4});
  expect(quadbinCellToParent(5207251884775047167n, 0)).toBe(5192650370358181887n);
  expect(quadbinIsValidCell(5207251884775047167n)).toBe(true);
  expect(quadbinIsValidCell(5207251884775047167n - 1n)).toBe(false);
  // Antimeridian wraps to column 0; latitude clips to the Mercator limit.
  expect(getQuadbinTileX(180, 10)).toBe(0);
  expect(getQuadbinTileX(-180, 10)).toBe(0);
  expect(getQuadbinTileX(Math.fround(179.99999), 10)).toBe(1023);
  expect(getQuadbinTileX(-1e-40, 10)).toBe(511);
  expect(getQuadbinTileX(0, 10)).toBe(512);
  expect(getQuadbinTileX(-0, 10)).toBe(512);
  expect(getQuadbinTileY(90, 10)).toBe(0);
  expect(getQuadbinTileY(-90, 10)).toBe(1023);
});

it('quadbin f32 keys agree with the f64 formula except within one tile of an edge', () => {
  const report: Record<number, number> = {};
  const errors: Record<number, number> = {};
  for (const resolution of [1, 4, 8, 12, 16, 20, 24, 26]) {
    const positions = createPointPositions(resolution, 20000, resolution);
    let mismatches = 0;
    let maximumError = 0;
    for (let row = 0; row < positions.length / 2; row++) {
      const latitude = positions[2 * row + 1];
      const y32 = getQuadbinTileY(latitude, resolution);
      const y64 = getQuadbinTileYFloat64(latitude, resolution);
      if (y32 !== y64) {
        // Rows past 20000 are the deliberate tile-edge cases; rate only the random points.
        mismatches += row < 20000 ? 1 : 0;
        maximumError = Math.max(maximumError, Math.abs(y32 - y64));
      }
    }
    report[resolution] = mismatches / 20000;
    errors[resolution] = maximumError;
    // f32 keeps about 1e-7 of the Mercator row range (an f32 latitude near the pole is itself
    // only that precise), so only levels finer than 22 may miss by more than one row.
    expect(maximumError, `resolution ${resolution}`).toBeLessThanOrEqual(
      Math.max(1, Math.ceil(2 ** resolution * 1e-7))
    );
  }
  // Resolution 12 is ~10 m tiles; the f32 row matches the f64 formula almost everywhere there.
  expect(report[12]).toBeLessThan(0.002);
  expect(report[4]).toBeLessThan(0.0005);
  // eslint-disable-next-line no-console
  console.log('quadbin f32 row vs f64: mismatch rate', report, 'max rows', errors);
});

it('H3 parents are bit truncations (pentagons, poles, antimeridian)', () => {
  const random = createRandom(3);
  const cells: string[] = [...getRes0Cells(), ...getPentagons(5), ...getPentagons(15)];
  for (let index = 0; index < 400; index++) {
    cells.push(latLngToCell(random() * 180 - 90, random() * 360 - 180, Math.floor(random() * 16)));
  }
  for (const [longitude, latitude] of getEdgeCasePoints(10)) {
    if (Number.isFinite(longitude) && Math.abs(longitude) <= 180) {
      cells.push(latLngToCell(latitude, longitude, 12));
    }
  }
  cells.push(...cellToChildren(getPentagons(2)[0], 4));
  for (const cell of cells) {
    const key = h3ToBigInt(cell);
    const resolution = Number((key >> 52n) & 0xfn);
    for (let parent = 0; parent <= resolution; parent++) {
      const layout = getCellKeyLayout('h3', parent);
      const compact = (key >> BigInt(layout.lowBit)) & ((1n << BigInt(layout.width)) - 1n);
      const rebuilt =
        (BigInt(layout.headerHigh | (parent << 20)) << 32n) |
        (compact << BigInt(layout.lowBit)) |
        ((1n << BigInt(layout.lowBit)) - 1n);
      expect(rebuilt.toString(16)).toBe(cellToParent(cell, parent));
    }
  }
});

it('fixed-point scaling rounds half to even and saturates', () => {
  expect(getScaledValue(0.5, 1)).toBe(0n);
  expect(getScaledValue(1.5, 1)).toBe(2n);
  expect(getScaledValue(-2.5, 1)).toBe(-2n);
  expect(getScaledValue(-0.25, 4)).toBe(-1n);
  expect(getScaledValue(3e38, 65536)).toBe(BigInt(Math.fround(2 ** 62 - 2 ** 38)));
});

it('GPUCellAggregation validates props', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const expectThrows = (overrides: Partial<GPUCellAggregationProps>, message: RegExp) =>
    expect(() => new GPUCellAggregation(createProps(graph, overrides))).toThrow(message);
  expectThrows({resolution: 27}, /resolution/);
  expectThrows({family: 'h3'}, /quadbin family only/);
  expectThrows(
    {cells: createTransientView(graph, 'cells-a', 'uint32x2', 10)},
    /exactly one of positions or cells/
  );
  expectThrows({values: undefined}, /needs values/);
  expectThrows({values: createTransientView(graph, 'values-b', 'float32', 9)}, /length/);
  expectThrows({sumScale: 0}, /sumScale/);
  expectThrows(
    {
      output: {
        ...createTable(graph, 16),
        counts: createTransientView(graph, 'counts-c', 'uint32', 15)
      }
    },
    /same length/
  );
  const positions = createTransientView(graph, 'positions-d', 'float32x2', 10);
  expectThrows(
    {
      positions,
      output: {
        ...createTable(graph, 16),
        status: {
          ...createTable(graph, 16).status,
          count: createTransientView(graph, 'x', 'uint32', 0)
        }
      }
    },
    /one uint32 row/
  );
  device.destroy();
});

it('GPUCellAggregation emits deterministic nodes with one or two key sorts', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const getIds = (resolution: number) => {
    const contributor = new GPUCellAggregation(
      createProps(graph, {id: `agg-${resolution}`, resolution})
    );
    const first = contributor.getCommandNodes(graph).map(node => node.id);
    return first;
  };
  const single = getIds(15);
  const double = getIds(16);
  expect(single.some(id => id.includes('-sort-high'))).toBe(false);
  expect(double.some(id => id.includes('-sort-high'))).toBe(true);
  expect(single[0]).toBe('agg-15-keys');
  expect(single.at(-1)).toBe('agg-15-publish');
  device.destroy();
});

it('GPUCellRollup and GPUCellPyramid validate levels and columns', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const source = createTable(graph, 16, false);
  expect(
    () =>
      new GPUCellRollup({
        family: 'quadbin',
        sourceResolution: 6,
        resolution: 4,
        source,
        output: createTable(graph, 8)
      })
  ).toThrow(/source.sums/);
  expect(
    () =>
      new GPUCellRollup({
        family: 'h3',
        sourceResolution: 4,
        resolution: 6,
        source,
        output: createTable(graph, 8, false)
      })
  ).toThrow(/must not exceed/);
  const props = createProps(graph);
  const {output: _output, resolution: _resolution, ...rowProps} = props;
  expect(
    () =>
      new GPUCellPyramid({
        ...rowProps,
        resolutions: [8, 8],
        output: {levels: [createTable(graph, 16), createTable(graph, 16)]}
      })
  ).toThrow(/strictly decrease/);
  // Coarse levels with sums get transient sums on intermediate levels.
  const pyramid = new GPUCellPyramid({
    ...rowProps,
    resolutions: [8, 6, 2],
    output: {
      levels: [
        createTable(graph, 16, false),
        createTable(graph, 16, false),
        createTable(graph, 16)
      ],
      levelCounts: createTransientView(graph, 'level-counts', 'uint32', 3)
    }
  });
  const ids = pyramid.getCommandNodes(graph).map(node => node.id);
  expect(ids.filter(id => id.endsWith('-accumulate-sums'))).toEqual([
    'cell-pyramid-level-0-accumulate-sums',
    'cell-pyramid-level-1-accumulate-sums',
    'cell-pyramid-level-2-accumulate-sums'
  ]);
  expect(pyramid.levelFirstRows).toEqual([0, 0, 0]);
  expect(
    () =>
      new GPUCellLevelSelection({
        levelCounts: createTransientView(graph, 'counts-e', 'uint32', 3),
        activeLevel: createTransientView(graph, 'active-e', 'uint32', 1),
        levelFirstRows: [0, 1],
        output: {status: {count: createTransientView(graph, 'count-e', 'uint32', 1)}}
      })
  ).toThrow(/levelFirstRows/);
  device.destroy();
});

it('GPUCellPyramid level publishes share one pipeline for N and N + 1 levels (rule 12)', () => {
  const getPipelineCount = (levelCount: number, idPattern: RegExp) => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const {output: _output, resolution: _resolution, ...rowProps} = createProps(graph);
    const resolutions = Array.from({length: levelCount}, (_, level) => 12 - level);
    const pyramid = new GPUCellPyramid({
      ...rowProps,
      resolutions,
      output: {
        levels: resolutions.map(() => createTable(graph, 16)),
        levelCounts: createTransientView(graph, `level-counts-${serial++}`, 'uint32', levelCount)
      }
    });
    const count = getGPUCommandNodePipelineKeys(pyramid.getCommandNodes(graph), {idPattern}).size;
    device.destroy();
    return count;
  };
  // Each level used to publish into its own row of the packed levelCounts view, baking a
  // different element offset into every level's publish kernel. (The roll-up key kernels still
  // bake each level's resolution.)
  const publish = /-level-\d+-publish$/;
  expect(getPipelineCount(5, publish)).toBe(getPipelineCount(4, publish));
  expect(getPipelineCount(4, publish)).toBeLessThanOrEqual(2);
  expect(getPipelineCount(5, /-level-counts$/)).toBe(1);
});

it('cell operations declare their coordinate space', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const planar = {coordinateSpace: 'planar', metric: 'native', units: 'native'} as const;
  const keyed = (overrides: Partial<GPUCellAggregationProps> = {}) => {
    const {positions: _positions, ...rest} = createProps(graph);
    return {
      ...rest,
      cells: createTransientView(graph, `cells-${serial++}`, 'uint32x2', 10),
      ...overrides
    };
  };
  // Points are longitude/latitude degrees: planar throws, the default is lon/lat.
  expect(new GPUCellAggregation(createProps(graph)).spatialContext).toMatchObject({
    coordinateSpace: 'longitude-latitude',
    metric: 'none'
  });
  expect(() => new GPUCellAggregation(createProps(graph, {spatialContext: planar}))).toThrow(
    /does not support planar coordinates/
  );
  expect(
    () =>
      new GPUCellAggregation(
        createProps(graph, {
          spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
        })
      )
  ).toThrow(/does not support the great-circle metric/);
  // Pre-keyed rows are discrete-grid; the family must match.
  expect(new GPUCellAggregation(keyed()).spatialContext).toMatchObject({
    coordinateSpace: 'discrete-grid',
    gridFamily: 'quadbin'
  });
  expect(() => new GPUCellAggregation(keyed({spatialContext: planar}))).toThrow(
    /does not support planar coordinates/
  );
  const grid = (gridFamily: string, gridResolution?: number) =>
    ({coordinateSpace: 'discrete-grid', metric: 'none', gridFamily, gridResolution}) as const;
  expect(() => new GPUCellAggregation(keyed({spatialContext: grid('h3')}))).toThrow(
    /gridFamily must be quadbin/
  );
  expect(() => new GPUCellAggregation(keyed({spatialContext: grid('Quadbin', 10)}))).not.toThrow();
  expect(() => new GPUCellAggregation(keyed({spatialContext: grid('quadbin', 6)}))).toThrow(
    /coarser than the output resolution 8/
  );
  const rollup = (spatialContext: typeof planar | ReturnType<typeof grid>) =>
    new GPUCellRollup({
      family: 'h3',
      sourceResolution: 6,
      resolution: 4,
      spatialContext,
      source: createTable(graph, 16),
      output: createTable(graph, 8)
    });
  expect(() => rollup(grid('h3', 6))).not.toThrow();
  expect(() => rollup(grid('quadbin', 6))).toThrow(/gridFamily must be h3/);
  expect(() => rollup(planar)).toThrow(/does not support planar coordinates/);
});
