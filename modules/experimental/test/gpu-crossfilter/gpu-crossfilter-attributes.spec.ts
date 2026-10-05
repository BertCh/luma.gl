// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, createTransientView} from '@luma.gl/gpgpu/gpu-core';
import {GPUCrossfilter} from '@luma.gl/experimental/gpu-crossfilter';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createNullWebGPUDevice,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../utils/gpu-contributor-test-utils';

/** Linked histograms over numeric attribute columns: live-row mask, missing values, auto domains. */

const BIN_COUNT = 16;

type Column = {values: number[]; isUint32?: boolean};
type Brush = readonly [number, number] | null;

type Harness = {
  device: Device;
  compiled: ReturnType<GPUCommandGraph['compile']>;
  filter: GPUCrossfilter;
  columns: Buffer[];
  liveMask?: Buffer;
  histograms: Buffer[];
  selectedCount: Buffer;
  selection: Buffer;
  destroy(): void;
};

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
  columns: Column[],
  options: {liveMask?: number[]} = {}
): Harness {
  const rows = columns[0].values.length;
  const graph = new GPUCommandGraph(device, {id: 'crossfilter-attributes'});
  const columnBuffers = columns.map(column =>
    createInputBuffer(
      device,
      column.isUint32 ? Uint32Array.from(column.values) : Float32Array.from(column.values)
    )
  );
  const liveMaskBuffer = options.liveMask
    ? createInputBuffer(device, Uint32Array.from(options.liveMask))
    : undefined;
  const histogramBuffers = columns.map(() => createOutputBuffer(device, BIN_COUNT));
  const selectedCount = createOutputBuffer(device, 1);
  const selection = createOutputBuffer(device, rows);

  const inputs = columns.map((column, index) =>
    importGraphBuffer(
      graph,
      `column-${index}`,
      columnBuffers[index],
      column.isUint32 ? 'uint32' : 'float32',
      rows
    )
  );
  const filter = new GPUCrossfilter(graph, {
    id: 'attributes',
    dimensions: inputs.map((input, index) => ({
      id: `d${index}`,
      kind: 'range' as const,
      input,
      rejectNonFinite: true,
      exclusiveMaximum: true
    })),
    liveMask: liveMaskBuffer
      ? importGraphBuffer(graph, 'live', liveMaskBuffer, 'uint32', rows)
      : undefined,
    views: [
      ...inputs.map((input, index) => ({
        id: `h${index}`,
        kind: 'histogram' as const,
        dimension: `d${index}`,
        input,
        domain: 'auto' as const,
        output: importGraphBuffer(graph, `h${index}`, histogramBuffers[index], 'uint32', BIN_COUNT)
      })),
      {
        id: 'selected',
        kind: 'count' as const,
        output: importGraphBuffer(graph, 'selected-count', selectedCount, 'uint32', 1)
      },
      {
        id: 'selection',
        kind: 'mask' as const,
        output: importGraphBuffer(graph, 'selection', selection, 'uint32', rows)
      }
    ]
  });
  filter.addToGraph(graph);
  const compiled = graph.compile();
  const owned = [
    ...columnBuffers,
    ...histogramBuffers,
    selectedCount,
    selection,
    ...(liveMaskBuffer ? [liveMaskBuffer] : [])
  ];
  return {
    device,
    compiled,
    filter,
    columns: columnBuffers,
    liveMask: liveMaskBuffer,
    histograms: histogramBuffers,
    selectedCount,
    selection,
    destroy() {
      compiled.destroy();
      filter.destroy();
      for (const buffer of owned) buffer.destroy();
    }
  };
}

/** CPU reference: half-open brushes, non-finite rows fail, auto domains over live finite rows. */
function computeOracle(columns: Column[], brushes: Brush[], liveMask?: number[]) {
  const rows = columns[0].values.length;
  const value = (dimension: number, row: number) => Math.fround(columns[dimension].values[row]);
  const live = (row: number) => (liveMask ? liveMask[row] !== 0 : true);
  const fails = (dimension: number, row: number) => {
    const v = value(dimension, row);
    const brush = brushes[dimension];
    return !Number.isFinite(v) || (brush !== null && !(v >= brush[0] && v < brush[1]));
  };
  const histograms = columns.map((_, dimension) => {
    let minimum = Infinity;
    let maximum = -Infinity;
    for (let row = 0; row < rows; row++) {
      if (live(row) && Number.isFinite(value(dimension, row))) {
        minimum = Math.min(minimum, value(dimension, row));
        maximum = Math.max(maximum, value(dimension, row));
      }
    }
    const bins = new Array<number>(BIN_COUNT).fill(0);
    for (let row = 0; row < rows; row++) {
      if (!live(row) || !Number.isFinite(value(dimension, row))) continue;
      if (columns.some((_, other) => other !== dimension && fails(other, row))) continue;
      const scaled = ((value(dimension, row) - minimum) * BIN_COUNT) / (maximum - minimum);
      bins[Math.min(Math.floor(scaled), BIN_COUNT - 1)]++;
    }
    return bins;
  });
  const selection = Array.from({length: rows}, (_, row) =>
    live(row) && !columns.some((_, dimension) => fails(dimension, row)) ? 1 : 0
  );
  return {histograms, selection, selectedCount: selection.filter(Boolean).length};
}

async function expectMatchesOracle(
  harness: Harness,
  columns: Column[],
  brushes: Brush[],
  liveMask?: number[]
) {
  columns.forEach((_, index) => {
    const brush = brushes[index];
    if (brush) harness.filter.setRange(`d${index}`, brush);
    else harness.filter.clear(`d${index}`);
  });
  submitGraph(harness.device, harness.compiled, undefined);
  const oracle = computeOracle(columns, brushes, liveMask);
  for (const [index, buffer] of harness.histograms.entries()) {
    expect(await readUint32(buffer, BIN_COUNT), `histogram ${index}`).toEqual(
      oracle.histograms[index]
    );
  }
  expect(await readUint32(harness.selectedCount, 1)).toEqual([oracle.selectedCount]);
  expect(await readUint32(harness.selection, columns[0].values.length)).toEqual(oracle.selection);
  return oracle;
}

it('GPUCrossfilter matches the attribute oracle and updates brushes without recompiling', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const random = createRandom(1);
  const rows = 3000;
  const columns: Column[] = [
    {values: createDyadicValues(rows, random)},
    {values: createDyadicValues(rows, random).map(Math.round), isUint32: true},
    {values: createDyadicValues(rows, random)}
  ];
  columns[0].values[5] = NaN;
  columns[2].values[7] = Infinity;
  const harness = createHarness(device, columns);
  console.log('GPUDEVICE', String(device && device.info.gpu));
  const first = await expectMatchesOracle(harness, columns, [[8, 40], [10, 50], null]);
  expect(first.selectedCount).toBeGreaterThan(0);
  expect(first.selectedCount).toBeLessThan(rows);
  const second = await expectMatchesOracle(harness, columns, [[0, 20], null, [30, 60]]);
  expect(second.histograms).not.toEqual(first.histograms);

  // Column contents change between encodings; auto domains follow.
  const shifted = columns[0].values.map(value => (value + 16) % 64);
  shifted[0] = 0;
  shifted[1] = 64;
  columns[0] = {values: shifted};
  harness.columns[0].write(Float32Array.from(shifted));
  await expectMatchesOracle(harness, columns, [[8, 40], [10, 50], null]);
  harness.destroy();
});

it('GPUCrossfilter keeps dead rows out of domains, histograms, counts, and selections', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const random = createRandom(2);
  const rows = 2048;
  const columns: Column[] = [
    {values: createDyadicValues(rows, random)},
    {values: createDyadicValues(rows, random)}
  ];
  const liveMask = Array.from({length: rows}, (_, row) => (row % 3 === 0 ? 0 : 1));
  // The pinned 0 and 64 extremes sit in dead rows 0 and 3? Row 0 is dead, row 1 is live:
  // plant out-of-domain extremes in dead rows so a live-blind domain would widen.
  columns[0].values[3] = 5000;
  columns[0].values[6] = -1000;
  columns[0].values[2] = 63.5;
  columns[0].values[1] = 0.5;
  const harness = createHarness(device, columns, {liveMask});
  const brushes: Brush[] = [[8, 40], null];
  const oracle = await expectMatchesOracle(harness, columns, brushes, liveMask);
  expect(oracle.selectedCount).toBeLessThan(rows);

  // Toggle liveness without recompiling.
  const next = liveMask.map(value => 1 - value);
  harness.liveMask!.write(Uint32Array.from(next));
  await expectMatchesOracle(harness, columns, brushes, next);
  harness.destroy();
});

it('GPUCrossfilter composes more dimensions than one pass can bind', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const random = createRandom(3);
  const rows = 1024;
  const columns: Column[] = Array.from({length: 12}, () => ({
    values: createDyadicValues(rows, random)
  }));
  const liveMask = Array.from({length: rows}, (_, row) => (row % 5 === 0 ? 0 : 1));
  const harness = createHarness(device, columns, {liveMask});
  const brushes: Brush[] = columns.map((_, index) => (index % 2 === 0 ? [4, 62] : null));
  await expectMatchesOracle(harness, columns, brushes, liveMask);
  harness.destroy();
});

it('GPUCrossfilter range options reject missing values and make the maximum exclusive', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const values = createTransientView(graph, 'values', 'float32', 4);
  const mask = createTransientView(graph, 'live', 'uint32', 4);
  const wrongLength = createTransientView(graph, 'live-short', 'uint32', 2);
  expect(
    () =>
      new GPUCrossfilter(graph, {
        dimensions: [{id: 'value', kind: 'range', input: values, rejectNonFinite: true}],
        liveMask: mask
      })
  ).not.toThrow();
  expect(
    () =>
      new GPUCrossfilter(graph, {
        dimensions: [{id: 'value2', kind: 'range', input: values}],
        liveMask: wrongLength
      })
  ).toThrow(/liveMask must preserve the same row count/);
  device.destroy();
});
