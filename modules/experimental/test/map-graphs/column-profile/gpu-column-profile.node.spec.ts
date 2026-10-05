// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUColumnProfileParameterLength,
  getGPUColumnProfileParameterValues,
  GPU_COLUMN_PROFILE_STATISTIC,
  GPU_COLUMN_PROFILE_STATISTIC_COUNT,
  GPUColumnProfile,
  type GPUColumnProfileProps
} from '../../../src/map-graphs/column-profile';
import {createNullWebGPUDevice} from '../map-graph-test-utils';
import {
  computeColumnProfileOracle,
  estimateHyperLogLog,
  fmix32,
  getOracleBin,
  resolveOracleDomain
} from './column-profile-oracle';

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUColumnProfileProps> = {}
): GPUColumnProfileProps {
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  return {
    columns: [
      {values: view('float32', 100)},
      {values: view('uint32', 100), kind: 'category', categoryCount: 7}
    ],
    mask: view('uint32', 100),
    parameters: view('float32', 4),
    histogramBinCount: 8,
    hyperLogLogPrecision: 6,
    topCategoryCount: 3,
    output: {
      statistics: view('float32', 2 * GPU_COLUMN_PROFILE_STATISTIC_COUNT),
      counts: view('uint32', 4),
      histograms: view('uint32', 16),
      hyperLogLogRegisters: view('uint32', 128),
      topCategories: view('uint32', 6),
      topCategoryCounts: view('uint32', 6)
    },
    ...overrides
  };
}

function expectThrows(
  overrides: (graph: GPUCommandGraph) => Partial<GPUColumnProfileProps>,
  message: RegExp
): void {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  expect(() => new GPUColumnProfile(createProps(graph, overrides(graph)))).toThrow(message);
  device.destroy();
}

it('GPUColumnProfile parameter helper packs domains column-major', () => {
  expect(getGPUColumnProfileParameterLength(3)).toBe(6);
  const values = getGPUColumnProfileParameterValues([[1, 2], undefined, {lo: 5}], 3);
  expect(values[0]).toBe(1);
  expect(values[1]).toBe(2);
  expect(Number.isNaN(values[2])).toBe(true);
  expect(Number.isNaN(values[3])).toBe(true);
  expect(values[4]).toBe(5);
  expect(Number.isNaN(values[5])).toBe(true);
  expect(getGPUColumnProfileParameterValues([[1, 2]], 2).length).toBe(4);
  const target = new Float32Array(4);
  expect(
    getGPUColumnProfileParameterValues(
      [
        [3, 4],
        [5, 6]
      ],
      2,
      target
    )
  ).toBe(target);
  expect(Array.from(target)).toEqual([3, 4, 5, 6]);
  expect(() =>
    getGPUColumnProfileParameterValues(
      [
        [1, 2],
        [3, 4]
      ],
      1
    )
  ).toThrow(/exceed/);
  expect(() => getGPUColumnProfileParameterValues([], 0)).toThrow(/positive/);
  expect(() => getGPUColumnProfileParameterValues([[1, 2]], 2, new Float32Array(3))).toThrow(
    /must hold 4/
  );
});

it('GPUColumnProfile statistic layout is dense and unique', () => {
  const fields = Object.values(GPU_COLUMN_PROFILE_STATISTIC).sort((a, b) => a - b);
  expect(fields).toEqual(Array.from({length: GPU_COLUMN_PROFILE_STATISTIC_COUNT}, (_, i) => i));
});

it('GPUColumnProfile emits deterministic node IDs', () => {
  const ids = () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const recipe = new GPUColumnProfile(createProps(graph, {id: 'profile'}));
    const nodeIds = recipe.getCommandNodes(graph).map(node => node.id);
    device.destroy();
    return nodeIds;
  };
  const first = ids();
  expect(first).toEqual(ids());
  expect(new Set(first).size).toBe(first.length);
  expect(first).toContain('profile-init');
  expect(first).toContain('profile-column-0-moments-tile');
  expect(first).toContain('profile-column-1-moments-tile');
  expect(first).toContain('profile-moments-merge');
  expect(first).toContain('profile-domain');
  expect(first).toContain('profile-column-0-histogram');
  expect(first).not.toContain('profile-column-1-histogram');
  expect(first).toContain('profile-column-1-category-counts');
  expect(first).toContain('profile-column-1-top-categories');
  expect(first).toContain('profile-column-0-finish');
  expect(first.every(id => id.startsWith('profile-'))).toBe(true);
});

it('GPUColumnProfile skips optional nodes when outputs are absent', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph);
  const recipe = new GPUColumnProfile({
    ...props,
    mask: undefined,
    parameters: undefined,
    output: {statistics: props.output.statistics}
  });
  const ids = recipe.getCommandNodes(graph).map(node => node.id);
  expect(ids.some(id => id.includes('histogram'))).toBe(false);
  expect(ids.some(id => id.includes('domain'))).toBe(false);
  expect(ids.some(id => id.includes('category-counts'))).toBe(false);
  expect(ids.some(id => id.includes('top-categories'))).toBe(false);
  device.destroy();
});

it('GPUColumnProfile validates its properties', () => {
  expectThrows(() => ({columns: []}), /between 1 and 16 columns/);
  expectThrows(
    graph => ({
      columns: Array.from({length: 17}, () => ({
        values: createTransientView(graph, `many-${serial++}`, 'float32', 100)
      }))
    }),
    /between 1 and 16 columns/
  );
  expectThrows(() => ({histogramBinCount: 0}), /histogramBinCount/);
  expectThrows(() => ({histogramBinCount: 2.5}), /histogramBinCount/);
  expectThrows(() => ({hyperLogLogPrecision: 3}), /hyperLogLogPrecision/);
  expectThrows(() => ({hyperLogLogPrecision: 17}), /hyperLogLogPrecision/);
  expectThrows(() => ({topCategoryCount: 65}), /topCategoryCount/);
  expectThrows(() => ({topCategoryCount: 0}), /topCategoryCount/);
  expectThrows(
    graph => ({
      columns: [
        {values: createTransientView(graph, `a-${serial++}`, 'float32', 100)},
        {values: createTransientView(graph, `b-${serial++}`, 'float32', 99)}
      ]
    }),
    /length must equal/
  );
  expectThrows(
    graph => ({
      columns: [
        {
          values: createTransientView(graph, `c-${serial++}`, 'uint32', 100),
          kind: 'category',
          categoryCount: 0
        }
      ]
    }),
    /categoryCount/
  );
  expectThrows(
    graph => ({
      columns: [{values: createTransientView(graph, `d-${serial++}`, 'uint32', 100) as never}]
    }),
    /float32/
  );
  expectThrows(
    graph => ({mask: createTransientView(graph, `m-${serial++}`, 'uint32', 99)}),
    /mask length/
  );
  expectThrows(
    graph => ({parameters: createTransientView(graph, `p-${serial++}`, 'float32', 3)}),
    /parameters must hold 4/
  );
  expectThrows(
    graph => ({
      output: {statistics: createTransientView(graph, `s-${serial++}`, 'float32', 21)}
    }),
    /output.statistics must hold 22/
  );
  expectThrows(
    graph => ({
      output: {
        statistics: createTransientView(graph, `s-${serial++}`, 'float32', 22),
        histograms: createTransientView(graph, `h-${serial++}`, 'uint32', 15)
      }
    }),
    /output.histograms must hold 16/
  );
  expectThrows(
    graph => ({
      output: {
        statistics: createTransientView(graph, `s-${serial++}`, 'float32', 22),
        topCategories: createTransientView(graph, `t-${serial++}`, 'uint32', 6)
      }
    }),
    /go together/
  );
  expectThrows(graph => {
    const column = createTransientView(graph, `alias-${serial++}`, 'float32', 100);
    return {
      columns: [{values: column}],
      output: {statistics: column as never}
    };
  }, /outputs must not share buffers|output.statistics must hold/);
});

it('GPUColumnProfile rejects views from another graph', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const other = new GPUCommandGraph(device);
  const recipe = new GPUColumnProfile(createProps(graph));
  expect(() => recipe.getCommandNodes(other)).toThrow(/must belong to the target graph/);
  device.destroy();
});

it('column profile oracle computes moments, nulls, infinities and masks', () => {
  const values = Float32Array.from([1, 2, 3, NaN, Infinity, -Infinity, 4, -0, 0, 99]);
  const mask = Uint32Array.from([1, 1, 1, 1, 1, 1, 1, 1, 1, 0]);
  const [result] = computeColumnProfileOracle({
    columns: [{kind: 'numeric', values}],
    mask,
    histogramBinCount: 4,
    precision: 6,
    topCategoryCount: 2
  });
  expect(result.count).toBe(8);
  expect(result.nullCount).toBe(1);
  expect(result.minimum).toBe(-Infinity);
  expect(result.maximum).toBe(Infinity);
  expect(result.sum).toBe(10);
  expect(result.mean).toBeCloseTo(10 / 6, 12);
  expect(result.variance).toBeCloseTo(
    [1, 2, 3, 4, 0, 0].reduce((total, value) => total + (value - 10 / 6) ** 2, 0) / 6,
    12
  );
  // Auto domain is the finite range [-0, 4]; 6 finite rows land in bins of width 1.
  expect(result.histogram.reduce((a, b) => a + b, 0)).toBe(6);
  expect(Object.is(result.minimum, -Infinity)).toBe(true);
  const [signed] = computeColumnProfileOracle({
    columns: [{kind: 'numeric', values: Float32Array.from([0, -0, 0])}],
    histogramBinCount: 2,
    precision: 4,
    topCategoryCount: 1
  });
  expect(Object.is(signed.minimum, -0)).toBe(true);
  expect(signed.trueDistinct).toBe(1);
  expect(signed.histogram.reduce((a, b) => a + b, 0)).toBe(3);
});

it('column profile oracle handles empty selections and categories', () => {
  const [categories] = computeColumnProfileOracle({
    columns: [
      {
        kind: 'category',
        values: Uint32Array.from([2, 2, 5, 0xffffffff, 9, 1, 1, 1]),
        categoryCount: 6
      }
    ],
    histogramBinCount: 4,
    precision: 5,
    topCategoryCount: 4
  });
  expect(categories.count).toBe(7);
  expect(categories.nullCount).toBe(1);
  expect(categories.overflowCount).toBe(1);
  expect(categories.topCategories).toEqual([1, 2, 5, 0xffffffff]);
  expect(categories.topCategoryCounts).toEqual([3, 2, 1, 0]);
  expect(Array.from(categories.histogram)).toEqual([0, 3, 2, 0]);
  expect(categories.minimum).toBe(1);
  expect(categories.maximum).toBe(9);
  const [none] = computeColumnProfileOracle({
    columns: [{kind: 'numeric', values: Float32Array.from([1, 2])}],
    mask: Uint32Array.from([0, 0]),
    histogramBinCount: 3,
    precision: 4,
    topCategoryCount: 1
  });
  expect(none.count).toBe(0);
  expect(Number.isNaN(none.minimum)).toBe(true);
  expect(Number.isNaN(none.mean)).toBe(true);
  expect(Array.from(none.histogram)).toEqual([0, 0, 0]);
  expect(none.distinctEstimate).toBe(0);
});

it('column profile oracle estimates HyperLogLog from known register sets', () => {
  const precision = 6;
  const m = 64;
  // All registers zero: linear counting gives zero.
  expect(estimateHyperLogLog(new Uint32Array(m), precision)).toBe(0);
  // One occupied register of rank 1: linear counting m * ln(m / (m - 1)).
  const single = new Uint32Array(m);
  single[3] = 1;
  expect(estimateHyperLogLog(single, precision)).toBeCloseTo(m * Math.log(m / (m - 1)), 12);
  // Every register at rank 3: raw = alpha * m^2 / (m / 8), above the small-range threshold.
  const full = new Uint32Array(m).fill(3);
  expect(estimateHyperLogLog(full, precision)).toBeCloseTo((0.709 * m * m) / (m / 8), 9);
  // Mixed: ten zeros out of 64, raw below 2.5 m uses linear counting.
  const mixed = new Uint32Array(m).fill(1);
  for (let index = 0; index < 10; index++) {
    mixed[index] = 0;
  }
  expect(estimateHyperLogLog(mixed, precision)).toBeCloseTo(m * Math.log(m / 10), 12);
  // Large precision alpha.
  const big = new Uint32Array(4096).fill(10);
  expect(estimateHyperLogLog(big, 12)).toBeCloseTo(
    ((0.7213 / (1 + 1.079 / 4096)) * 4096 * 4096) / (4096 / 1024),
    6
  );
  // The estimate tracks the true distinct count within a few standard errors.
  const rows = Float32Array.from({length: 50000}, (_, row) => (row * 7919) % 20011);
  const [result] = computeColumnProfileOracle({
    columns: [{kind: 'numeric', values: rows}],
    histogramBinCount: 4,
    precision: 12,
    topCategoryCount: 1
  });
  expect(result.trueDistinct).toBe(20011);
  expect(Math.abs(result.distinctEstimate - 20011) / 20011).toBeLessThan((4 * 1.04) / 64);
  expect(fmix32(0)).toBe(0);
  expect(fmix32(1)).toBe(0x514e28b7);
});

it('column profile oracle bins mirror the corrected f32 formula', () => {
  const domain = resolveOracleDomain(0, 1, 10);
  expect(domain.mode).toBe(1);
  expect(getOracleBin(0, 0, domain.width, 10)).toBe(0);
  expect(getOracleBin(1, 0, domain.width, 10)).toBe(9);
  expect(getOracleBin(Math.fround(0.3), 0, domain.width, 10)).toBeGreaterThanOrEqual(2);
  expect(resolveOracleDomain(2, 2, 4).mode).toBe(2);
  expect(resolveOracleDomain(3, 2, 4).mode).toBe(0);
  expect(resolveOracleDomain(NaN, 2, 4).mode).toBe(0);
  expect(resolveOracleDomain(-3.4e38, 3.4e38, 4).mode).toBe(0);
});
