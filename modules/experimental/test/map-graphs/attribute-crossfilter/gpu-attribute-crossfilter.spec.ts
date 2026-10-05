// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUMapGraphParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/map-graphs';
import {
  getGPUAttributeCrossfilterHistogramLayout,
  getGPUAttributeCrossfilterParameterLength,
  getGPUAttributeCrossfilterParameterValues,
  GPUAttributeCrossfilter,
  type GPUAttributeCrossfilterDimensionState
} from '../../../src/map-graphs/attribute-crossfilter';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
} from '../map-graph-test-utils';
import {
  computeAttributeCrossfilterOracle,
  type AttributeCrossfilterOracleDimension
} from './attribute-crossfilter-oracle';

type ColumnSpec = {
  values: number[];
  binCount: number;
  isUint32?: boolean;
  domain?: 'auto' | 'parameters';
};

type Harness = {
  device: Device;
  compiled: CompiledGPUCommandGraph;
  parameters: GPUMapGraphParameterBuffer<'float32'>;
  liveMaskBuffer?: Buffer;
  liveMaskSource?: Uint32Array;
  histograms: Buffer;
  domains: Buffer;
  selectedCount: Buffer;
  liveCount: Buffer;
  selection: Buffer;
  ids: Buffer;
  count: Buffer;
  overflow: Buffer;
  totalCount: Buffer;
  columns: Buffer[];
  layout: ReturnType<typeof getGPUAttributeCrossfilterHistogramLayout>;
  destroy(): void;
};

/** Seeded generator so failures reproduce. */
function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

/** Values in 1/16 steps within [0, 64], with 0 and 64 pinned so the auto domain is [0, 64]. */
function createDyadicValues(rows: number, random: () => number): number[] {
  const values = Array.from({length: rows}, () => Math.floor(random() * 1025) / 16);
  values[0] = 0;
  values[1] = 64;
  return values;
}

function createHarness(
  device: Device,
  columns: ColumnSpec[],
  options: {liveMask?: number[]; capacity?: number; id?: string} = {}
): Harness {
  const rows = columns[0].values.length;
  const graph = new GPUCommandGraph(device, {
    id: options.id ?? 'attribute-crossfilter-test'
  });
  const layout = getGPUAttributeCrossfilterHistogramLayout(columns.map(c => c.binCount));
  const columnBuffers = columns.map(column =>
    createInputBuffer(
      device,
      column.isUint32 ? Uint32Array.from(column.values) : Float32Array.from(column.values)
    )
  );
  const liveMaskSource = options.liveMask ? Uint32Array.from(options.liveMask) : undefined;
  const liveMaskBuffer = liveMaskSource ? createInputBuffer(device, liveMaskSource) : undefined;
  const parameters = new GPUMapGraphParameterBuffer(device, {
    id: 'parameters',
    format: 'float32',
    length: getGPUAttributeCrossfilterParameterLength(columns.length)
  });
  const histograms = createOutputBuffer(device, layout.totalBinCount);
  const domains = createOutputBuffer(device, 2 * columns.length);
  const selectedCount = createOutputBuffer(device, 1);
  const liveCount = createOutputBuffer(device, 1);
  const selection = createOutputBuffer(device, rows);
  const capacity = options.capacity ?? rows;
  const ids = createOutputBuffer(device, capacity);
  const count = createOutputBuffer(device, 1);
  const overflow = createOutputBuffer(device, 1);
  const totalCount = createOutputBuffer(device, 1);
  graph.add(
    new GPUAttributeCrossfilter({
      dimensions: columns.map((column, index) => ({
        column: importGraphBuffer(
          graph,
          `column-${index}`,
          columnBuffers[index],
          column.isUint32 ? 'uint32' : 'float32',
          rows
        ) as never,
        binCount: column.binCount,
        domain: column.domain
      })),
      liveMask: liveMaskBuffer
        ? importGraphBuffer(graph, 'live', liveMaskBuffer, 'uint32', rows)
        : undefined,
      parameters: parameters.importToGraph(graph),
      histograms: importGraphBuffer(
        graph,
        'histograms',
        histograms,
        'uint32',
        layout.totalBinCount
      ),
      domains: importGraphBuffer(graph, 'domains', domains, 'float32', 2 * columns.length),
      selectedCount: importGraphBuffer(graph, 'selected', selectedCount, 'uint32', 1),
      liveCount: importGraphBuffer(graph, 'live-count', liveCount, 'uint32', 1),
      selection: importGraphBuffer(graph, 'selection', selection, 'uint32', rows),
      output: {
        ids: importGraphBuffer(graph, 'ids', ids, 'uint32', capacity),
        count: importGraphBuffer(graph, 'count', count, 'uint32', 1),
        overflow: importGraphBuffer(graph, 'overflow', overflow, 'uint32', 1),
        totalCount: importGraphBuffer(graph, 'total', totalCount, 'uint32', 1)
      }
    })
  );
  const compiled = graph.compile();
  const owned = [
    ...columnBuffers,
    histograms,
    domains,
    selectedCount,
    liveCount,
    selection,
    ids,
    count,
    overflow,
    totalCount,
    ...(liveMaskBuffer ? [liveMaskBuffer] : [])
  ];
  return {
    device,
    compiled,
    parameters,
    liveMaskBuffer,
    liveMaskSource,
    histograms,
    domains,
    selectedCount,
    liveCount,
    selection,
    ids,
    count,
    overflow,
    totalCount,
    columns: columnBuffers,
    layout,
    destroy() {
      compiled.destroy();
      parameters.destroy();
      for (const buffer of owned) {
        buffer.destroy();
      }
    }
  };
}

/** Writes parameters, encodes the compiled graph, and compares every output with the oracle. */
async function expectMatchesOracle(
  harness: Harness,
  columns: ColumnSpec[],
  states: GPUAttributeCrossfilterDimensionState[],
  capacity: number = columns[0].values.length
): Promise<ReturnType<typeof computeAttributeCrossfilterOracle>> {
  harness.parameters.write(getGPUAttributeCrossfilterParameterValues(states));
  submitGraph(harness.device, harness.compiled, undefined);
  const oracle = computeAttributeCrossfilterOracle(
    columns as AttributeCrossfilterOracleDimension[],
    states,
    harness.liveMaskSource
  );
  const rows = columns[0].values.length;
  expect(await readUint32(harness.histograms, harness.layout.totalBinCount)).toEqual(
    oracle.histograms
  );
  expect(await readFloat32(harness.domains, 2 * columns.length)).toEqual(oracle.domains);
  expect(await readUint32(harness.selectedCount, 1)).toEqual([oracle.selectedCount]);
  expect(await readUint32(harness.liveCount, 1)).toEqual([oracle.liveCount]);
  expect(await readUint32(harness.selection, rows)).toEqual(oracle.selection);
  const count = Math.min(oracle.selectedCount, capacity);
  expect(await readUint32(harness.count, 1)).toEqual([count]);
  expect(await readUint32(harness.totalCount, 1)).toEqual([oracle.selectedCount]);
  expect(await readUint32(harness.overflow, 1)).toEqual([oracle.selectedCount > capacity ? 1 : 0]);
  expect(await readUint32(harness.ids, count)).toEqual(oracle.selectedIds.slice(0, count));
  return oracle;
}

function createMixedColumns(rows: number, seed: number): ColumnSpec[] {
  const random = createRandom(seed);
  return [
    {values: createDyadicValues(rows, random), binCount: 16},
    {
      values: createDyadicValues(rows, random).map(value => Math.round(value)),
      binCount: 8,
      isUint32: true
    },
    {values: createDyadicValues(rows, random), binCount: 64},
    {
      values: createDyadicValues(rows, random).map(value => Math.round(value)),
      binCount: 32,
      isUint32: true,
      domain: 'parameters'
    }
  ];
}

const STATES: GPUAttributeCrossfilterDimensionState[] = [
  {brush: [8, 40]},
  {brush: [10, 50]},
  {},
  {domain: [0, 64]}
];

it('GPUAttributeCrossfilter matches the oracle for mixed columns and updates brushes without recompiling', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const rows = 3000;
  const columns = createMixedColumns(rows, 1);
  const harness = createHarness(device, columns);
  const first = await expectMatchesOracle(harness, columns, STATES);
  expect(first.selectedCount).toBeGreaterThan(0);
  expect(first.selectedCount).toBeLessThan(rows);

  const nodeCount = harness.compiled.nodes?.length;
  // Changed brushes, enabled flags, and parameter domain on the same compiled graph.
  const second = await expectMatchesOracle(harness, columns, [
    {brush: [0, 20]},
    {},
    {brush: [30, Infinity]},
    {brush: [4, 60], domain: [16, 48]}
  ]);
  expect(second.histograms).not.toEqual(first.histograms);
  expect(harness.compiled.nodes?.length).toBe(nodeCount);

  // Column contents change between encodings.
  const shifted = columns[0].values.map(value => (value + 16) % 64);
  shifted[0] = 0;
  shifted[1] = 64;
  columns[0] = {...columns[0], values: shifted};
  harness.columns[0].write(Float32Array.from(shifted));
  await expectMatchesOracle(harness, columns, STATES);
  harness.destroy();
});

it('GPUAttributeCrossfilter excludes dead rows from domains, histograms, counts, and selection', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const rows = 2048;
  const columns = createMixedColumns(rows, 2);
  const liveMask = Array.from({length: rows}, (_, row) => (row % 3 === 0 ? 0 : 1));
  // Extremes only in dead rows (row 0 and 3 are dead): the live domain must not widen.
  columns[0].values[0] = -1000;
  columns[0].values[3] = 5000;
  columns[2].values[6] = -1048576;
  columns[2].values[9] = 1048576;
  columns[0].values[1] = 0.5;
  columns[0].values[2] = 63.5;
  columns[2].values[1] = 1;
  columns[2].values[2] = 63;
  const harness = createHarness(device, columns, {liveMask});
  const oracle = await expectMatchesOracle(harness, columns, STATES);
  expect(oracle.domains.slice(0, 2)).not.toContain(-1000);
  expect(oracle.domains[1]).toBeLessThan(5000);
  expect(oracle.domains[0]).toBeGreaterThanOrEqual(0);
  expect(oracle.domains[4]).toBeGreaterThan(-1048576);
  expect(oracle.liveCount).toBeLessThan(rows);

  // Toggle liveness without recompiling.
  const next = liveMask.map(value => 1 - value);
  harness.liveMaskBuffer!.write(Uint32Array.from(next));
  harness.liveMaskSource!.set(next);
  await expectMatchesOracle(harness, columns, STATES);
  harness.destroy();
});

it("GPUAttributeCrossfilter ignores a dimension's own brush in its own histogram", async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const columns = createMixedColumns(1500, 3);
  const harness = createHarness(device, columns);
  const read = async (states: GPUAttributeCrossfilterDimensionState[]) => {
    const oracle = await expectMatchesOracle(harness, columns, states);
    return oracle.histograms;
  };
  const [, , , dimensionThreeStates] = [0, 0, 0, {domain: [0, 64] as const}];
  const base = [{brush: [10, 30] as const}, {}, {}, dimensionThreeStates];
  const histogramsA = await read(base);
  const histogramsB = await read([{brush: [40, 50]}, ...base.slice(1)]);
  const {offsets} = harness.layout;
  // Dimension 0's own histogram is unchanged by its own brush...
  expect(histogramsA.slice(offsets[0], offsets[1])).toEqual(
    histogramsB.slice(offsets[0], offsets[1])
  );
  // ...while other dimensions' histograms respond to it.
  expect(histogramsA.slice(offsets[2])).not.toEqual(histogramsB.slice(offsets[2]));
  harness.destroy();
});

it('GPUAttributeCrossfilter bins boundaries exactly and clamps the domain maximum into the last bin', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // Parameter domain [0, 16], 16 bins: integer values sit exactly on bin boundaries.
  const values = [-1, 0, 0.5, 1, 1.0625, 7.9375, 8, 15, 15.9375, 16, 16.5, 100];
  const columns: ColumnSpec[] = [{values, binCount: 16, domain: 'parameters'}];
  const harness = createHarness(device, columns, {capacity: 4});
  const oracle = await expectMatchesOracle(
    harness,
    columns,
    [{domain: [0, 16], brush: [0, 16]}],
    4
  );
  // Value 16 lands in the last bin of the histogram (own brush is ignored) but fails the exclusive brush maximum.
  expect(oracle.histograms[15]).toBe(3);
  expect(oracle.histograms[0]).toBe(2);
  expect(oracle.selectedCount).toBe(8);
  // Constant column: zero-width domain puts every row in bin 0.
  harness.destroy();
  const constant: ColumnSpec[] = [{values: [5, 5, 5, 5], binCount: 4}];
  const constantHarness = createHarness(device, constant);
  const constantOracle = await expectMatchesOracle(constantHarness, constant, [{}]);
  expect(constantOracle.histograms).toEqual([4, 0, 0, 0]);
  constantHarness.destroy();
});

it('GPUAttributeCrossfilter treats non-finite values as missing', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const columns: ColumnSpec[] = [
    {values: [0, 8, NaN, 16, 4, Infinity, 12, -Infinity], binCount: 4},
    {values: [1, 2, 3, 4, NaN, 6, 7, 8], binCount: 4, domain: 'parameters'}
  ];
  const harness = createHarness(device, columns);
  const oracle = await expectMatchesOracle(harness, columns, [{}, {domain: [0, 8]}]);
  // Rows 2, 4, 5, 7 are missing in at least one dimension: only rows 0, 1, 3, 6 are selected.
  expect(oracle.selectedIds).toEqual([0, 1, 3, 6]);
  expect(oracle.domains.slice(0, 2)).toEqual([0, 16]);
  expect(oracle.liveCount).toBe(8);
  harness.destroy();
});

it('GPUAttributeCrossfilter supports one dimension without outputs beyond histograms', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const values = createDyadicValues(700, createRandom(4));
  const graph = new GPUCommandGraph(device, {
    id: 'attribute-crossfilter-minimal'
  });
  const column = createInputBuffer(device, Float32Array.from(values));
  const histograms = createOutputBuffer(device, 16);
  const parameters = new GPUMapGraphParameterBuffer(device, {
    id: 'parameters',
    format: 'float32',
    length: 8,
    values: getGPUAttributeCrossfilterParameterValues([{brush: [10, 20]}])
  });
  graph.add(
    new GPUAttributeCrossfilter({
      dimensions: [
        {
          column: importGraphBuffer(graph, 'column', column, 'float32', 700),
          binCount: 16
        }
      ],
      parameters: parameters.importToGraph(graph),
      histograms: importGraphBuffer(graph, 'histograms', histograms, 'uint32', 16)
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const oracle = computeAttributeCrossfilterOracle([{values, binCount: 16}], [{brush: [10, 20]}]);
  expect(await readUint32(histograms, 16)).toEqual(oracle.histograms);
  compiled.destroy();
  parameters.destroy();
  column.destroy();
  histograms.destroy();
});

it('GPUAttributeCrossfilter measures brush-change latency', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const rows = 524288;
  const random = createRandom(5);
  const columns: ColumnSpec[] = Array.from({length: 4}, (_, index) => ({
    values: createDyadicValues(rows, random),
    binCount: 64,
    domain: index === 3 ? ('parameters' as const) : ('auto' as const)
  }));
  const harness = createHarness(device, columns, {capacity: 1024});
  const states = (shift: number): GPUAttributeCrossfilterDimensionState[] => [
    {brush: [shift, 32 + shift]},
    {},
    {brush: [8, 56]},
    {domain: [0, 64]}
  ];
  const durations: number[] = [];
  for (let run = -3; run < 20; run++) {
    const start = performance.now();
    harness.parameters.write(getGPUAttributeCrossfilterParameterValues(states(run & 7)));
    submitGraph(device, harness.compiled, undefined);
    await readUint32(harness.histograms, harness.layout.totalBinCount);
    if (run >= 0) {
      durations.push(performance.now() - start);
    }
  }
  durations.sort((a, b) => a - b);
  const median = durations[Math.floor(durations.length / 2)];
  const p95 = durations[Math.ceil(durations.length * 0.95) - 1];
  // eslint-disable-next-line no-console
  console.log(
    `GPUAttributeCrossfilter 524288 rows x 4 dims x 64 bins: median ${median.toFixed(2)} ms, p95 ${p95.toFixed(2)} ms (${device.info.gpu} ${device.info.gpuType})`
  );
  expect(median).toBeGreaterThan(0);
  harness.destroy();
}, 60000);
