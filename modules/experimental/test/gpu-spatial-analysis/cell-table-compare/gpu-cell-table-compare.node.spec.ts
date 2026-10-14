// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import type {GPUCellTable} from '../../../src/gpu-spatial-analysis/cell-aggregation/cell-table';
import {
  GPUCellTableCompare,
  type GPUCellTableCompareProps
} from '../../../src/gpu-spatial-analysis/cell-table-compare';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  CORE_WORKGROUP_STORAGE_LIMIT_BYTES,
  getStorageBindingCount,
  getWorkgroupMemoryBytes
} from '../../utils/gpu-contributor-test-utils';
import {
  compareCellTablesOnCPU,
  convertInt64ToFloat32,
  sumFixedOrder,
  type CompareInputCell
} from './cell-table-compare-oracle';

let serial = 0;

function createTable(graph: GPUCommandGraph, capacity: number, withSums = true): GPUCellTable {
  const view = <Format extends 'uint32' | 'uint32x2' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  return {
    cells: view('uint32x2', capacity),
    counts: view('uint32', capacity),
    sums: withSums ? view('uint32x2', capacity) : undefined,
    status: {count: view('uint32', 1), overflow: view('uint32', 1)}
  };
}

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUCellTableCompareProps> = {}
): GPUCellTableCompareProps {
  const view = <Format extends 'uint32' | 'uint32x2' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  return {
    before: createTable(graph, 4),
    after: createTable(graph, 6),
    output: {
      cells: view('uint32x2', 8),
      presence: view('uint32', 8),
      delta: view('float32', 8),
      zScore: view('float32', 8),
      status: {count: view('uint32', 1), overflow: view('uint32', 1)}
    },
    ...overrides
  };
}

function expectThrows(
  overrides: (graph: GPUCommandGraph) => Partial<GPUCellTableCompareProps>,
  message: RegExp
): void {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  expect(() => new GPUCellTableCompare(createProps(graph, overrides(graph)))).toThrow(message);
  device.destroy();
}

const cell = (key: bigint, count: number, sum = 0n): CompareInputCell => ({
  key,
  count,
  sum
});
const options = {
  measure: 'count',
  sumScale: 65536,
  zScore: 'poisson',
  capacity: 16
} as const;

it('GPUCellTableCompare validates its props', () => {
  expectThrows(() => ({measure: 'median' as 'sum'}), /measure/);
  expectThrows(() => ({zScore: 'x' as 'poisson'}), /zScore/);
  expectThrows(() => ({sumScale: 0}), /sumScale/);
  expectThrows(graph => ({measure: 'sum', before: createTable(graph, 4, false)}), /needs sums/);
  expectThrows(
    graph => ({
      output: {
        ...createProps(graph).output,
        delta: createTransientView(graph, 'short', 'float32', 3)
      }
    }),
    /needs output.cells.length rows/
  );
  expectThrows(graph => {
    const before = createTable(graph, 8);
    return {
      before,
      output: {...createProps(graph).output, cells: before.cells}
    };
  }, /outputs must not share/);
  expectThrows(
    graph => ({
      output: {
        ...createProps(graph).output,
        cells: createTransientView(graph, 'c', 'uint32', 8) as never
      }
    }),
    /uint32x2/
  );
});

it('GPUCellTableCompare names nodes deterministically and stays within 8 bindings', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, {measure: 'sum', zScore: 'standardized'});
  const nodes = new GPUCellTableCompare({
    ...props,
    id: 'cmp'
  }).getCommandNodes(graph);
  expect(nodes.map(node => node.id)).toEqual([
    'cmp-flags',
    'cmp-scan-level-0-scan',
    'cmp-merge',
    'cmp-measures',
    'cmp-statistics-sums',
    'cmp-statistics-squares',
    'cmp-statistics',
    'cmp-derived',
    'cmp-tail',
    'cmp-publish'
  ]);
  const sources = getGPUCommandNodeKernelSources(nodes);
  for (const {id, source} of sources) {
    expect(getStorageBindingCount(source), `${id} storage bindings`).toBeLessThanOrEqual(8);
    expect(getWorkgroupMemoryBytes(source), `${id} workgroup bytes`).toBeLessThanOrEqual(
      CORE_WORKGROUP_STORAGE_LIMIT_BYTES
    );
  }
  // The z-score reduction is three kernels, with no atomics, one workgroup tile per 256 slots.
  const statistics = sources.filter(({id}) => id.startsWith('cmp-statistics'));
  expect(statistics.map(({id}) => id)).toEqual([
    'cmp-statistics-sums',
    'cmp-statistics-squares',
    'cmp-statistics'
  ]);
  for (const {id, source} of statistics) {
    expect(source, id).not.toContain('atomic');
    // Capacity 8 needs one workgroup.
    expect(source, id).toContain('const GROUPS: u32 = 1u;');
  }
  device.destroy();
});

it('GPUCellTableCompare spreads the z-score reduction over workgroups for large capacities', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const capacity = 3000;
  const view = <Format extends 'uint32' | 'uint32x2' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `large-${serial++}`, format, length);
  const props = createProps(graph, {
    zScore: 'standardized',
    output: {
      cells: view('uint32x2', capacity),
      presence: view('uint32', capacity),
      delta: view('float32', capacity),
      zScore: view('float32', capacity),
      status: {count: view('uint32', 1), overflow: view('uint32', 1)}
    }
  });
  const nodes = new GPUCellTableCompare({...props, id: 'big'}).getCommandNodes(graph);
  const sources = getGPUCommandNodeKernelSources(nodes).filter(({id}) =>
    id.startsWith('big-statistics')
  );
  expect(sources).toHaveLength(3);
  // ceil(3000 / 256) = 12 workgroups.
  for (const {id, source} of sources) {
    expect(source, id).toContain('const GROUPS: u32 = 12u;');
    expect(getStorageBindingCount(source)).toBeLessThanOrEqual(8);
  }
  device.destroy();
});

it('compareCellTablesOnCPU builds the outer join', () => {
  const result = compareCellTablesOnCPU(
    [cell(1n, 4), cell(5n, 2), cell(9n, 7)],
    [cell(2n, 3), cell(5n, 6), cell(9n << 32n, 1)],
    options
  );
  expect(result.rows.map(row => [row.key, row.presence, row.before, row.after, row.delta])).toEqual(
    [
      [1n, 1, 4, 0, -4],
      [2n, 2, 0, 3, 3],
      [5n, 3, 2, 6, 4],
      [9n, 1, 7, 0, -7],
      [9n << 32n, 2, 0, 1, 1]
    ]
  );
  expect(result.total).toBe(5);
  expect(result.overflow).toBe(false);
  expect(result.rows[0].ratio).toBe(0);
  expect(result.rows[1].ratio).toBeNaN();
  expect(result.rows[1].percentChange).toBeNaN();
  expect(result.rows[2].percentChange).toBe(200);
  expect(result.rows[2].zScore).toBe(Math.fround(4 / Math.sqrt(8)));
});

it('compareCellTablesOnCPU bounds the union and flags overflow', () => {
  const result = compareCellTablesOnCPU([cell(1n, 1), cell(3n, 1)], [cell(2n, 1), cell(4n, 1)], {
    ...options,
    capacity: 3
  });
  expect(result.rows.map(row => row.key)).toEqual([1n, 2n, 3n]);
  expect([result.total, result.overflow]).toEqual([4, true]);
  expect(compareCellTablesOnCPU([], [], options)).toEqual({
    rows: [],
    total: 0,
    overflow: false
  });
});

it('compareCellTablesOnCPU computes sum deltas exactly in 64-bit fixed point', () => {
  // Two sums that differ by 1 fixed-point unit at 2^40 magnitude: f32 subtraction would give 0.
  const big = (1n << 40n) + 1n;
  const result = compareCellTablesOnCPU([cell(1n, 1, 1n << 40n)], [cell(1n, 1, big)], {
    ...options,
    measure: 'sum'
  });
  expect(result.rows[0].delta).toBe(Math.fround(1 / 65536));
  expect(result.rows[0].before).toBe(2 ** 40 / 65536);
  expect(convertInt64ToFloat32(-5n)).toBe(-5);
  expect(convertInt64ToFloat32((1n << 33n) + 3n)).toBe(Math.fround(2 ** 33 + 3));
});

it('compareCellTablesOnCPU standardizes deltas with the population standard deviation', () => {
  const before = [cell(1n, 0), cell(2n, 0), cell(3n, 0), cell(4n, 0)];
  const after = [cell(1n, 1), cell(2n, 2), cell(3n, 3), cell(4n, 4)];
  const result = compareCellTablesOnCPU(before, after, {
    ...options,
    zScore: 'standardized'
  });
  const sd = Math.sqrt(1.25);
  expect(result.rows[0].zScore).toBeCloseTo((1 - 2.5) / sd, 5);
  expect(result.rows[3].zScore).toBeCloseTo((4 - 2.5) / sd, 5);
  const same = compareCellTablesOnCPU([cell(1n, 1), cell(2n, 1)], [cell(1n, 1), cell(2n, 1)], {
    ...options,
    zScore: 'standardized'
  });
  expect(same.rows.map(row => row.zScore)).toEqual([0, 0]);
  expect(sumFixedOrder(Array.from({length: 1000}, (_, i) => i))).toBe(499500);
});
