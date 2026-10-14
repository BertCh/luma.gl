// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPUCellRollup, type GPUCellTable} from '../../../src/gpu-spatial-analysis/cell-aggregation';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import fixtureJson from './fixtures/cell-rollup.json';
import {requireTestDevice} from '../../utils/require-test-device';

// Oracle: fixtures/cell-rollup.json, written by test/oracles/geometry-properties.py. The expected
// parents come from h3.cell_to_parent (h3-py) with counts, fixed-point sums and extremes folded in
// python, so the comparison does not use the kernel's own key arithmetic or any TypeScript H3 port.
// Counts, sums and extremes are integer or exact float32 values and are compared exactly.

type FixtureRow = {cell: string; count: number; sum: string; minimum: number; maximum: number};
type Fixture = {
  sumScale: number;
  sourceResolution: number;
  source: FixtureRow[];
  cases: {resolution: number; parents: FixtureRow[]}[];
};

const fixture: Fixture = fixtureJson as unknown as Fixture;

const EMPTY_KEY_WORD = 0xffffffff;

function splitWords(value: bigint): [number, number] {
  const unsigned = BigInt.asUintN(64, value);
  return [Number(unsigned & 0xffffffffn), Number(unsigned >> 32n)];
}

function createSourceBuffers(device: Device, buffers: Buffer[], capacity: number) {
  const cells = new Uint32Array(2 * capacity).fill(EMPTY_KEY_WORD);
  const counts = new Uint32Array(capacity);
  const sums = new Uint32Array(2 * capacity);
  const minimums = new Float32Array(capacity).fill(NaN);
  const maximums = new Float32Array(capacity).fill(NaN);
  for (const [row, source] of fixture.source.entries()) {
    [cells[2 * row], cells[2 * row + 1]] = splitWords(BigInt(`0x${source.cell}`));
    counts[row] = source.count;
    [sums[2 * row], sums[2 * row + 1]] = splitWords(BigInt(source.sum));
    minimums[row] = source.minimum;
    maximums[row] = source.maximum;
  }
  const input = (values: Float32Array | Uint32Array) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return buffer;
  };
  return {
    cells: input(cells),
    counts: input(counts),
    sums: input(sums),
    minimums: input(minimums),
    maximums: input(maximums),
    count: input(Uint32Array.of(fixture.source.length)),
    overflow: input(Uint32Array.of(0))
  };
}

async function runRollup(device: Device, resolution: number, outputCapacity: number) {
  const buffers: Buffer[] = [];
  const sourceCapacity = fixture.source.length + 3;
  const sourceBuffers = createSourceBuffers(device, buffers, sourceCapacity);
  const output = (length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return buffer;
  };
  const outputBuffers = {
    cells: output(2 * outputCapacity),
    counts: output(outputCapacity),
    sums: output(2 * outputCapacity),
    sumValues: output(outputCapacity),
    minimums: output(outputCapacity),
    maximums: output(outputCapacity),
    count: output(1),
    overflow: output(1),
    required: output(1)
  };
  const graph = new GPUCommandGraph(device, {id: 'cell-rollup-property'});
  const source: GPUCellTable = {
    cells: importGraphBuffer(
      graph,
      'source-cells',
      sourceBuffers.cells,
      'uint32x2',
      sourceCapacity
    ),
    counts: importGraphBuffer(
      graph,
      'source-counts',
      sourceBuffers.counts,
      'uint32',
      sourceCapacity
    ),
    sums: importGraphBuffer(graph, 'source-sums', sourceBuffers.sums, 'uint32x2', sourceCapacity),
    minimums: importGraphBuffer(
      graph,
      'source-minimums',
      sourceBuffers.minimums,
      'float32',
      sourceCapacity
    ),
    maximums: importGraphBuffer(
      graph,
      'source-maximums',
      sourceBuffers.maximums,
      'float32',
      sourceCapacity
    ),
    status: {
      count: importGraphBuffer(graph, 'source-count', sourceBuffers.count, 'uint32', 1),
      overflow: importGraphBuffer(graph, 'source-overflow', sourceBuffers.overflow, 'uint32', 1)
    }
  };
  const table: GPUCellTable = {
    cells: importGraphBuffer(graph, 'cells', outputBuffers.cells, 'uint32x2', outputCapacity),
    counts: importGraphBuffer(graph, 'counts', outputBuffers.counts, 'uint32', outputCapacity),
    sums: importGraphBuffer(graph, 'sums', outputBuffers.sums, 'uint32x2', outputCapacity),
    sumValues: importGraphBuffer(
      graph,
      'sum-values',
      outputBuffers.sumValues,
      'float32',
      outputCapacity
    ),
    minimums: importGraphBuffer(
      graph,
      'minimums',
      outputBuffers.minimums,
      'float32',
      outputCapacity
    ),
    maximums: importGraphBuffer(
      graph,
      'maximums',
      outputBuffers.maximums,
      'float32',
      outputCapacity
    ),
    status: {
      count: importGraphBuffer(graph, 'count', outputBuffers.count, 'uint32', 1),
      overflow: importGraphBuffer(graph, 'overflow', outputBuffers.overflow, 'uint32', 1),
      requiredCount: importGraphBuffer(graph, 'required', outputBuffers.required, 'uint32', 1)
    }
  };
  graph.add(
    new GPUCellRollup({
      family: 'h3',
      sourceResolution: fixture.sourceResolution,
      resolution,
      source,
      sumScale: fixture.sumScale,
      output: table
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    count: (await readUint32(outputBuffers.count, 1))[0],
    overflow: (await readUint32(outputBuffers.overflow, 1))[0],
    required: (await readUint32(outputBuffers.required, 1))[0],
    cells: await readUint32(outputBuffers.cells, 2 * outputCapacity),
    counts: await readUint32(outputBuffers.counts, outputCapacity),
    sums: await readUint32(outputBuffers.sums, 2 * outputCapacity),
    sumValues: await readFloat32(outputBuffers.sumValues, outputCapacity),
    minimums: await readFloat32(outputBuffers.minimums, outputCapacity),
    maximums: await readFloat32(outputBuffers.maximums, outputCapacity)
  };
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

it('GPUCellRollup equals h3 cell_to_parent folds of the source table (counts, int64 sums, extremes)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Teeth: the source has a pentagon child and sums beyond 2^32 so word carries are exercised.
  expect(fixture.source.some(row => BigInt(row.sum) > 2n ** 32n)).toBe(true);
  expect(fixture.source.some(row => BigInt(row.sum) < -(2n ** 32n))).toBe(true);
  for (const {resolution, parents} of fixture.cases) {
    const label = `res ${fixture.sourceResolution} -> ${resolution}`;
    expect(parents.length, `${label} folds children`).toBeLessThan(fixture.source.length);
    const capacity = parents.length + 4;
    const actual = await runRollup(device, resolution, capacity);
    expect(actual.overflow, `${label} overflow`).toBe(0);
    expect(actual.required, `${label} required`).toBe(parents.length);
    expect(actual.count, `${label} count`).toBe(parents.length);
    for (const [row, expected] of parents.entries()) {
      const key = (BigInt(actual.cells[2 * row + 1]) << 32n) | BigInt(actual.cells[2 * row]);
      expect(key.toString(16), `${label} key ${row}`).toBe(expected.cell);
      expect(actual.counts[row], `${label} count ${row}`).toBe(expected.count);
      const sum = BigInt.asIntN(
        64,
        (BigInt(actual.sums[2 * row + 1]) << 32n) | BigInt(actual.sums[2 * row])
      );
      expect(sum.toString(), `${label} sum ${row}`).toBe(expected.sum);
      // The extremes are float32 values by construction (the generator rounds them), but the JSON
      // module import can perturb the last printed digit of a float64 literal (seen in the headless
      // run: -931.7789306640625 arrived as -931.7789306640624). Math.fround restores the exact
      // float32 value, so the comparison stays exact.
      expect(actual.minimums[row], `${label} minimum ${row}`).toBe(Math.fround(expected.minimum));
      expect(actual.maximums[row], `${label} maximum ${row}`).toBe(Math.fround(expected.maximum));
      const sumValue = Number(BigInt(expected.sum)) / fixture.sumScale;
      expect(Math.abs(actual.sumValues[row] - sumValue)).toBeLessThanOrEqual(
        Math.abs(sumValue) * 1e-6 + 1e-6
      );
    }
    // Rows past the count hold the empty key and a zero count.
    for (let row = parents.length; row < capacity; row++) {
      expect(actual.cells[2 * row], `${label} tail key ${row}`).toBe(EMPTY_KEY_WORD);
      expect(actual.cells[2 * row + 1], `${label} tail key ${row}`).toBe(EMPTY_KEY_WORD);
      expect(actual.counts[row], `${label} tail count ${row}`).toBe(0);
    }
  }
  device.destroy?.();
});

it('GPUCellRollup reports the exact requirement when the output table is too small', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {resolution, parents} = fixture.cases[0];
  const capacity = Math.floor(parents.length / 2);
  const actual = await runRollup(device, resolution, capacity);
  expect(actual.overflow).toBe(1);
  expect(actual.required).toBe(parents.length);
  expect(actual.count).toBe(capacity);
  // Documented: the cells with the largest keys are dropped.
  for (let row = 0; row < capacity; row++) {
    const key = (BigInt(actual.cells[2 * row + 1]) << 32n) | BigInt(actual.cells[2 * row]);
    expect(key.toString(16)).toBe(parents[row].cell);
    expect(actual.counts[row]).toBe(parents[row].count);
  }
  device.destroy?.();
});
