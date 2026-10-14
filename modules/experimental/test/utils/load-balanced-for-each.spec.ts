// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type GPUCommandNode} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../src/utils/gpu-contributor-utils';
import {
  createLoadBalancedForEachNode,
  createLoadBalancedSegmentSumNodes
} from '../../src/utils/load-balanced-for-each';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from './gpu-contributor-test-utils';
import {requireTestDevice} from './require-test-device';

/** Small deterministic generator so failures reproduce. */
function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

type SkewedCSR = {offsets: Uint32Array; values: Float32Array; hubRows: number[]};

/**
 * Leaves of degree 0 to 3 (a quarter empty), hubs of `hubDegree` items at the first, a middle and
 * the last row, and a run of empty rows after the middle hub.
 */
function createSkewedCSR(rowCount: number, hubDegree: number, seed: number): SkewedCSR {
  const random = createRandom(seed);
  const hubRows = [0, Math.floor(rowCount / 2), rowCount - 1];
  const degrees = new Uint32Array(rowCount);
  for (let row = 0; row < rowCount; row++) {
    degrees[row] = hubRows.includes(row) ? hubDegree : Math.floor(random() * 4);
  }
  for (let row = hubRows[1] + 1; row < Math.min(hubRows[1] + 3000, rowCount - 1); row++) {
    degrees[row] = 0;
  }
  const offsets = new Uint32Array(rowCount + 1);
  for (let row = 0; row < rowCount; row++) {
    offsets[row + 1] = offsets[row] + degrees[row];
  }
  const values = new Float32Array(offsets[rowCount]).map(() => Math.fround(random() * 2 - 1));
  return {offsets, values, hubRows};
}

async function runNodes(
  device: Device,
  build: (graph: GPUCommandGraph<undefined>) => GPUCommandNode<undefined>[]
): Promise<() => void> {
  const graph = new GPUCommandGraph<undefined>(device, {id: 'load-balanced'});
  for (const node of build(graph)) graph.add(node);
  const compiled = await graph.compileAsync();
  submitGraph(device, compiled, undefined);
  // The caller destroys the compiled graph after its readbacks.
  return () => compiled.destroy();
}

async function runSegmentSum(
  device: Device,
  csr: SkewedCSR,
  itemCapacity = csr.values.length
): Promise<{sums: number[]; bits: number[]}> {
  const rowCount = csr.offsets.length - 1;
  const offsetsBuffer = createInputBuffer(device, csr.offsets);
  const valuesBuffer = createInputBuffer(
    device,
    csr.values.length ? csr.values : new Float32Array(1)
  );
  const outputBuffer = createOutputBuffer(device, rowCount);
  const destroyGraph = await runNodes(device, graph =>
    createLoadBalancedSegmentSumNodes(graph, {
      id: 'sum',
      operation: 'LoadBalancedSpec',
      offsets: importGraphBuffer(graph, 'offsets', offsetsBuffer, 'uint32', rowCount + 1),
      itemCapacity,
      accumulatorType: 'f32',
      output: importGraphBuffer(graph, 'output', outputBuffer, 'float32', rowCount),
      bindings: [
        {
          name: 'itemValues',
          view: importGraphBuffer(
            graph,
            'values',
            valuesBuffer,
            'float32',
            Math.max(csr.values.length, 1)
          ),
          type: 'f32',
          access: 'read'
        }
      ],
      declarations: `fn getLoadBalancedContribution(row: u32, item: u32, lane: u32) -> f32 {
  return itemValues[itemValuesOffset + item];
}`
    })
  );
  const sums = await readFloat32(outputBuffer, rowCount);
  const bits = await readUint32(outputBuffer, rowCount);
  destroyGraph();
  for (const buffer of [offsetsBuffer, valuesBuffer, outputBuffer]) buffer.destroy();
  return {sums, bits};
}

/** CPU oracle in f64 with the absolute-value sum used by the tolerance. */
function sumRowsOnCpu(csr: SkewedCSR, itemCapacity = csr.values.length) {
  const rowCount = csr.offsets.length - 1;
  const sums = new Float64Array(rowCount);
  const magnitudes = new Float64Array(rowCount);
  const limit = Math.min(csr.offsets[rowCount], itemCapacity);
  for (let row = 0; row < rowCount; row++) {
    const end = Math.min(csr.offsets[row + 1], limit);
    for (let item = Math.min(csr.offsets[row], end); item < end; item++) {
      sums[row] += csr.values[item];
      magnitudes[row] += Math.abs(csr.values[item]);
    }
  }
  return {sums, magnitudes};
}

/**
 * Tolerance derivation: a row sum is a chain of at most 4 thread adds, 8 scan levels and about
 * `degree / 1024` serial carry adds (about 100 for the 1e5 hub), each rounding by at most 2^-24
 * of a partial bounded by `sum |v|`. 120 * 6e-8 = 7.2e-6, so 1e-5 * sum |v| bounds the error.
 */
function expectRowSums(actual: number[], expected: ReturnType<typeof sumRowsOnCpu>): void {
  expect(actual.length).toBe(expected.sums.length);
  let failures = 0;
  for (let row = 0; row < actual.length; row++) {
    const bound = 1e-5 * expected.magnitudes[row];
    if (Math.abs(actual[row] - expected.sums[row]) > bound) {
      failures++;
      if (failures < 5) {
        expect.soft(actual[row], `row ${row}`).toBeCloseTo(expected.sums[row], 4);
      }
    }
  }
  expect(failures).toBe(0);
}

it('createLoadBalancedSegmentSumNodes matches the CPU on a 1e5-item hub plus leaves, reproducibly', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const csr = createSkewedCSR(60_000, 100_000, 7);
  const expected = sumRowsOnCpu(csr);
  const first = await runSegmentSum(device, csr);
  expectRowSums(first.sums, expected);
  // Empty rows are written with zero (rule 11), every hub row is complete.
  expect(first.sums[csr.hubRows[1] + 1]).toBe(0);
  for (const hub of csr.hubRows) {
    expect(Math.abs(first.sums[hub] - expected.sums[hub])).toBeLessThan(
      1e-5 * expected.magnitudes[hub]
    );
  }
  // The association depends only on the offsets: a second run is bitwise identical.
  const second = await runSegmentSum(device, csr);
  expect(second.bits).toEqual(first.bits);
});

it('createLoadBalancedSegmentSumNodes handles a nonzero first offset, capacity clamps and empty input', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const base = createSkewedCSR(3000, 5000, 3);
  // Rows start at slot 17; the last offsets point past the capacity and are clamped.
  const shifted: SkewedCSR = {
    ...base,
    offsets: base.offsets.map(offset => offset + 17),
    values: new Float32Array(base.values.length + 17).map((_, index) =>
      index < 17 ? 1e6 : base.values[index - 17]
    )
  };
  const capacity = shifted.values.length - 2500;
  expectRowSums((await runSegmentSum(device, shifted, capacity)).sums, {
    ...sumRowsOnCpu(shifted, capacity)
  });
  // One row, no items.
  const empty: SkewedCSR = {
    offsets: new Uint32Array([0, 0]),
    values: new Float32Array(0),
    hubRows: []
  };
  expect((await runSegmentSum(device, empty, 0)).sums).toEqual([0]);
});

it('createLoadBalancedSegmentSumNodes sums vec2<u32> lanes exactly', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const csr = createSkewedCSR(5000, 20_000, 11);
  const rowCount = csr.offsets.length - 1;
  const laneCount = 3;
  const offsetsBuffer = createInputBuffer(device, csr.offsets);
  const outputBuffer = createOutputBuffer(device, rowCount * laneCount * 2);
  const destroyGraph = await runNodes(device, graph =>
    createLoadBalancedSegmentSumNodes(graph, {
      id: 'counts',
      operation: 'LoadBalancedSpec',
      offsets: importGraphBuffer(graph, 'offsets', offsetsBuffer, 'uint32', rowCount + 1),
      itemCapacity: csr.values.length,
      laneCount,
      accumulatorType: 'vec2<u32>',
      output: importGraphBuffer(graph, 'output', outputBuffer, 'uint32', rowCount * laneCount * 2),
      bindings: [],
      declarations: `fn getLoadBalancedContribution(row: u32, item: u32, lane: u32) -> vec2<u32> {
  return vec2<u32>(1u, (item % 7u) * (lane + 1u));
}`
    })
  );
  const actual = await readUint32(outputBuffer, rowCount * laneCount * 2);
  const expected: number[] = [];
  for (let row = 0; row < rowCount; row++) {
    for (let lane = 0; lane < laneCount; lane++) {
      let weighted = 0;
      for (let item = csr.offsets[row]; item < csr.offsets[row + 1]; item++) {
        weighted += (item % 7) * (lane + 1);
      }
      expected.push(csr.offsets[row + 1] - csr.offsets[row], weighted);
    }
  }
  expect(actual).toEqual(expected);
  destroyGraph();
  offsetsBuffer.destroy();
  outputBuffer.destroy();
});

it('createLoadBalancedForEachNode visits every item once with its row', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const csr = createSkewedCSR(40_000, 100_000, 5);
  const rowCount = csr.offsets.length - 1;
  const itemCount = csr.values.length;
  const offsetsBuffer = createInputBuffer(device, csr.offsets);
  const rowsBuffer: Buffer = createOutputBuffer(device, itemCount);
  const visitsBuffer: Buffer = createOutputBuffer(device, itemCount);
  const destroyGraph = await runNodes(device, graph => {
    const visits = importGraphBuffer(graph, 'visits', visitsBuffer, 'uint32', itemCount);
    return [
      createLoadBalancedForEachNode(graph, {
        id: 'visit',
        operation: 'LoadBalancedSpec',
        offsets: importGraphBuffer(graph, 'offsets', offsetsBuffer, 'uint32', rowCount + 1),
        itemCapacity: itemCount,
        bindings: [
          {
            name: 'rowOfItem',
            view: importGraphBuffer(graph, 'rows', rowsBuffer, 'uint32', itemCount),
            type: 'u32',
            access: 'read_write'
          },
          {name: 'visits', view: visits, type: 'atomic<u32>', access: 'read_write'}
        ],
        declarations: `fn visitLoadBalancedItem(row: u32, item: u32, lane: u32) {
  rowOfItem[rowOfItemOffset + item] = row + lane;
  atomicAdd(&visits[visitsOffset + item], 1u);
}`
      })
    ];
  });
  const rows = await readUint32(rowsBuffer, itemCount);
  const visits = await readUint32(visitsBuffer, itemCount);
  const expectedRows: number[] = [];
  for (let row = 0; row < rowCount; row++) {
    for (let item = csr.offsets[row]; item < csr.offsets[row + 1]; item++) expectedRows.push(row);
  }
  expect(rows).toEqual(expectedRows);
  expect(visits.every(count => count === 1)).toBe(true);
  destroyGraph();
  for (const buffer of [offsetsBuffer, rowsBuffer, visitsBuffer]) buffer.destroy();
});
