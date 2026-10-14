// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUTimeWindowFilter,
  type GPUTimeWindowFilterProps
} from '../../../src/gpu-dataframe/time-window-filter';
import {
  countDistinctPipelines,
  createNullWebGPUDevice,
  createVectorView,
  getParsedKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {
  scanFloatRange,
  normalizeRange,
  scanWordRange,
  searchFloatRange,
  searchWordRange,
  splitFloatWindow,
  type SortedWordWindow
} from './sorted-range-oracle';

function createProps(
  graph: GPUCommandGraph,
  rows: number = 8,
  overrides: Partial<GPUTimeWindowFilterProps> = {}
): GPUTimeWindowFilterProps {
  return {
    sorted: true,
    timestamps: createTransientView(graph, 'timestamps', 'float32', rows),
    window: createTransientView(graph, 'window', 'float32', 8),
    output: {
      ids: createTransientView(graph, 'ids', 'uint32', 4),
      status: {
        count: createTransientView(graph, 'count', 'uint32', 1),
        overflow: createTransientView(graph, 'overflow', 'uint32', 1),
        requiredCount: createTransientView(graph, 'required', 'uint32', 1)
      },
      rowRange: createTransientView(graph, 'range', 'uint32', 2),
      drawInstanceCount: createTransientView(graph, 'draw-count', 'uint32', 1),
      drawFirstInstance: createTransientView(graph, 'draw-first', 'uint32', 1)
    },
    ...overrides
  };
}

it('GPUTimeWindowFilter sorted schedules a range search, ids, and publish without O(n) nodes', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const nodes = new GPUTimeWindowFilter(createProps(graph)).getCommandNodes(graph);
  expect(nodes.map(node => node.id)).toEqual([
    'time-window-filter-range-search',
    'time-window-filter-ids',
    'time-window-filter-publish'
  ]);
  const sources = getParsedKernelSources(nodes);
  for (const source of sources.values()) {
    expect(getStorageBindingCount(source)).toBeLessThanOrEqual(8);
  }
  device.destroy();
});

it('GPUTimeWindowFilter sorted adds mask, fade, and order-check nodes only when asked', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const base = createProps(graph);
  const ids = new GPUTimeWindowFilter({
    ...base,
    validateOrder: true,
    output: {
      ...base.output,
      outputMask: createTransientView(graph, 'mask', 'uint32', 8),
      fadeWeights: createTransientView(graph, 'weights', 'float32', 8),
      orderViolations: createTransientView(graph, 'violations', 'uint32', 1)
    }
  })
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(ids.slice(0, 3)).toEqual([
    'time-window-filter-order-fill',
    'time-window-filter-order-check',
    'time-window-filter-range-search'
  ]);
  expect(ids).toContain('time-window-filter-classify');
  expect(ids.at(-1)).toBe('time-window-filter-publish');
  expect(new Set(ids).size).toBe(ids.length);
  device.destroy();
});

it('GPUTimeWindowFilter sorted generates parsing kernels in float, double-single, and word mode', () => {
  const device = createNullWebGPUDevice();
  for (const mode of ['float', 'double-single', 'words', 'empty'] as const) {
    const graph = new GPUCommandGraph(device);
    const rows = mode === 'empty' ? 0 : 16;
    const isWords = mode === 'words';
    const props = createProps(graph, rows, {
      timestamps: createTransientView(
        graph,
        `t-${mode}`,
        isWords ? 'uint32x2' : 'float32',
        rows
      ) as never,
      window: createTransientView(graph, `w-${mode}`, isWords ? 'uint32' : 'float32', 8) as never,
      ...(mode === 'double-single'
        ? {timestampsLow: createTransientView(graph, 'low', 'float32', rows)}
        : {}),
      sourceIds: createTransientView(graph, `s-${mode}`, 'uint32', rows)
    });
    const nodes = new GPUTimeWindowFilter(props).getCommandNodes(graph);
    const sources = getParsedKernelSources(nodes);
    // With no rows the ids kernel is skipped: the count is always 0.
    expect(sources.size, mode).toBe(mode === 'empty' ? 2 : 3);
    const search = sources.get('time-window-filter-range-search')!;
    expect(search.includes('loop {'), mode).toBe(mode !== 'empty');
    if (isWords) {
      expect(search).toContain('isTimeWordsAtLeast');
    } else if (mode !== 'empty') {
      expect(search).toContain('isTimeAtLeast');
    }
  }
  device.destroy();
});

it('GPUTimeWindowFilter sorted shares one range-search module across row counts', () => {
  const device = createNullWebGPUDevice();
  const count = (rows: number) => {
    const graph = new GPUCommandGraph(device);
    const nodes = new GPUTimeWindowFilter(createProps(graph, rows)).getCommandNodes(graph);
    return countDistinctPipelines(nodes);
  };
  // Rule 12: the row count is an override, never WGSL text.
  expect(count(1000)).toBe(count(1001));
  expect(count(1000)).toBe(3);
  device.destroy();
});

it('GPUTimeWindowFilter sorted validates its restrictions', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const base = createProps(graph);
  const create = (overrides: Partial<GPUTimeWindowFilterProps>) =>
    new GPUTimeWindowFilter({...base, ...overrides});

  expect(() => create({endTimestamps: createTransientView(graph, 'ends', 'float32', 8)})).toThrow(
    /sorted requires instant timestamps/
  );
  expect(() =>
    create({
      additionalPredicates: [
        {kind: 'selection', mask: createTransientView(graph, 'sel', 'uint32', 8)}
      ]
    })
  ).toThrow(/sorted does not support/);
  expect(() =>
    create({
      trackIds: createTransientView(graph, 'tracks', 'uint32', 8),
      output: {
        ...base.output,
        trackVisibleCounts: createTransientView(graph, 'track-counts', 'uint32', 2)
      }
    })
  ).toThrow(/sorted does not support/);
  expect(() =>
    create({
      timestamps: createVectorView('chunks', 'float32', [
        createTransientView(graph, 'c0', 'float32', 4),
        createTransientView(graph, 'c1', 'float32', 4)
      ])
    })
  ).toThrow(/single packed timestamps/);
  expect(() =>
    create({output: {...base.output, rowRange: createTransientView(graph, 'r1', 'uint32', 1)}})
  ).toThrow(/rowRange/);
  expect(() => create({validateOrder: true})).toThrow(/validateOrder and output.orderViolations/);
  expect(() =>
    create({
      output: {...base.output, orderViolations: createTransientView(graph, 'v', 'uint32', 1)}
    })
  ).toThrow(/validateOrder and output.orderViolations/);

  const unsorted = (overrides: Partial<GPUTimeWindowFilterProps>) =>
    new GPUTimeWindowFilter({...base, sorted: false, ...overrides});
  expect(() => unsorted({})).toThrow(/output.rowRange requires sorted/);
  expect(() => unsorted({output: {...base.output, rowRange: undefined}})).toThrow(
    /output.drawFirstInstance requires sorted/
  );
  expect(() =>
    unsorted({output: {...base.output, rowRange: undefined, drawFirstInstance: undefined}})
  ).not.toThrow();

  // Outputs must not alias inputs.
  const shared = graph.createTransientBuffer({id: 'shared', byteLength: 64, usage: 128});
  expect(() =>
    create({
      timestamps: graph.createDataView(shared, {format: 'float32', length: 8}),
      output: {
        ...base.output,
        rowRange: graph.createDataView(shared, {format: 'uint32', length: 2})
      }
    })
  ).toThrow(/share buffers/);
  device.destroy();
});

/** Sorted float rows with duplicates at the window edges. */
const FLOAT_ROWS = [-5, -1, 0, 0, 2, 2, 2, 3.5, 7, 7, 9, 12];

function sortedWords(): bigint[] {
  return [
    -(2n ** 62n),
    -3n,
    -3n,
    -1n,
    0n,
    0n,
    1n,
    1n,
    1n,
    4_294_967_295n,
    4_294_967_296n,
    4_294_967_296n,
    1_700_000_000_000n,
    2n ** 62n
  ];
}

it('sorted range oracle: binary searches agree with the mask path (float)', () => {
  const rows = {high: Float32Array.from(FLOAT_ROWS)};
  const windows: [number, number][] = [
    [0, 2], // duplicates at both edges
    [2, 2], // a single value with duplicates
    [-100, 100], // all in
    [100, 200], // all out above
    [-200, -100], // all out below
    [3, 2], // reversed: empty
    [2.5, 3], // between values
    [-5, -5], // first row only
    [12, 12], // last row only
    [7, 12],
    [-5, 12]
  ];
  for (const [start, end] of windows) {
    const window = splitFloatWindow(start, end);
    expect(normalizeRange(searchFloatRange(rows, window)), `${start}..${end}`).toEqual(
      normalizeRange(scanFloatRange(rows, window))
    );
  }
  expect(searchFloatRange(rows, splitFloatWindow(0, 2))).toEqual({firstRow: 2, rowCount: 5});
  expect(searchFloatRange(rows, splitFloatWindow(100, 200)).rowCount).toBe(0);
  expect(searchFloatRange(rows, splitFloatWindow(-100, 100))).toEqual({firstRow: 0, rowCount: 12});
  expect(searchFloatRange(rows, splitFloatWindow(3, 2)).rowCount).toBe(0);
});

it('sorted range oracle: empty input and a single row', () => {
  const window = splitFloatWindow(0, 1);
  expect(searchFloatRange({high: new Float32Array(0)}, window)).toEqual({firstRow: 0, rowCount: 0});
  expect(searchFloatRange({high: Float32Array.of(0.5)}, window)).toEqual({
    firstRow: 0,
    rowCount: 1
  });
  expect(searchFloatRange({high: Float32Array.of(2)}, window)).toEqual({firstRow: 0, rowCount: 0});
});

it('sorted range oracle: double-single low parts decide ties of the high part', () => {
  // All rows share the same f32 high part 1e8; the low parts order them.
  const base = 1e8;
  const highs = new Float32Array(8).fill(Math.fround(base));
  const exact = [base - 6, base - 4, base - 2, base - 2, base, base + 1, base + 3, base + 3];
  const lows = Float32Array.from(exact.map(value => Math.fround(value - Math.fround(base))));
  const rows = {high: highs, low: lows};
  for (const [start, end] of [
    [base - 2, base],
    [base - 2, base - 2],
    [base + 0.5, base + 3],
    [base - 100, base - 7],
    [base + 4, base + 100]
  ]) {
    const window = splitFloatWindow(start, end);
    expect(normalizeRange(searchFloatRange(rows, window)), `${start}..${end}`).toEqual(
      normalizeRange(scanFloatRange(rows, window))
    );
  }
  expect(searchFloatRange(rows, splitFloatWindow(base - 2, base - 2))).toEqual({
    firstRow: 2,
    rowCount: 2
  });
});

it('sorted range oracle: Int64 words with negative times and window fractions', () => {
  const times = sortedWords();
  const make = (
    start: bigint,
    end: bigint,
    startFraction = 0,
    endFraction = 0
  ): SortedWordWindow => ({start, startFraction, end, endFraction});
  const windows = [
    make(-3n, 0n), // negative edges with duplicates
    make(-(2n ** 63n), 2n ** 63n - 1n), // unbounded: all in
    make(1n, 1n), // duplicates only
    make(1n, 1n, 0.5, 0.5), // fractional playhead between integers: nothing
    make(0n, 0n, 0, 0.25), // fraction on the end keeps the integer row
    make(0n, 1n, 0.25, 0), // fraction on the start drops the integer row
    make(4_294_967_295n, 4_294_967_296n), // across the low-word wrap
    make(5n, 4n), // reversed
    make(2n ** 62n + 1n, 2n ** 63n - 1n), // all out above
    make(-(2n ** 63n), -(2n ** 62n) - 1n) // all out below
  ];
  for (const window of windows) {
    expect(
      normalizeRange(searchWordRange(times, window)),
      JSON.stringify(window, (_, v) => String(v))
    ).toEqual(normalizeRange(scanWordRange(times, window)));
  }
  expect(searchWordRange(times, make(-3n, 0n))).toEqual({firstRow: 1, rowCount: 5});
  expect(searchWordRange(times, make(1n, 1n, 0.5, 0.5)).rowCount).toBe(0);
  expect(searchWordRange(times, make(4_294_967_295n, 4_294_967_296n))).toEqual({
    firstRow: 9,
    rowCount: 3
  });
});

it('sorted range oracle: exhaustive windows over small sorted float inputs', () => {
  for (const rows of [[], [1], [1, 1], [1, 2], [1, 1, 1, 1, 1], [0, 1, 1, 2, 3, 3, 3, 5, 8]]) {
    const data = {high: Float32Array.from(rows)};
    for (let start = -1; start <= 9; start++) {
      for (let end = -1; end <= 9; end++) {
        for (const fraction of [0, 0.5]) {
          const window = splitFloatWindow(start + fraction, end);
          expect(normalizeRange(searchFloatRange(data, window))).toEqual(
            normalizeRange(scanFloatRange(data, window))
          );
        }
      }
    }
  }
});
