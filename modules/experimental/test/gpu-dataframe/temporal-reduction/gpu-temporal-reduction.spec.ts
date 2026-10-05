// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPUTemporalReductionParameterValues,
  getGPUTemporalReductionWordParameterValues,
  GPUTemporalReduction,
  reduceTemporalBucketsOnCPU,
  type GPUTemporalReductionCPUResult
} from '../../../src/gpu-dataframe/temporal-reduction';
import {ADVERSARIAL_WIDTHS, createAdversarialEdgeScene} from './adversarial-edges';
import {getInt64TimeWords} from '../../../src/gpu-dataframe/time-window-filter';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
} from '../../utils/gpu-contributor-test-utils';

type Scene = {
  cellIds: Uint32Array;
  timestamps: Float32Array | BigInt64Array;
  values: Float32Array;
  mask?: Uint32Array;
  cellCount: number;
  bucketCount: number;
  capacity?: number;
};

/** Deterministic xorshift in [0, 1). */
function createRandom(seed: number): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return state / 4294967296;
  };
}

type Fixture = {
  run(origin: number | bigint, width: number | bigint): Promise<Result>;
  destroy(): void;
};

type Result = GPUTemporalReductionCPUResult & {
  occupiedCount: number;
  occupiedOverflow: number;
  occupiedIds: number[];
};

function createFixture(device: Device, scene: Scene): Fixture {
  const isWordMode = scene.timestamps instanceof BigInt64Array;
  const slotCount = scene.cellCount * scene.bucketCount;
  const rows = scene.values.length;
  const capacity = scene.capacity ?? slotCount;
  const graph = new GPUCommandGraph(device, {id: 'temporal-reduction-graph'});
  const buffers: Buffer[] = [];
  const input = (values: Float32Array | Uint32Array) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return buffer;
  };
  const output = (length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return buffer;
  };
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'bucket-parameters',
    format: isWordMode ? 'uint32' : 'float32',
    length: 4
  });
  const timestampsBuffer = input(
    scene.timestamps instanceof BigInt64Array
      ? getInt64TimeWords(scene.timestamps)
      : scene.timestamps
  );
  const outputs = {
    counts: output(slotCount),
    min: output(slotCount),
    max: output(slotCount),
    first: output(slotCount),
    last: output(slotCount),
    ids: output(capacity),
    count: output(1),
    overflow: output(1),
    total: output(1)
  };
  const parameters = isWordMode
    ? parameterBuffer.importToGraph(graph)
    : parameterBuffer.importToGraph(graph);
  graph.add(
    new GPUTemporalReduction({
      id: 'reduction',
      cellIds: importGraphBuffer(graph, 'cells', input(scene.cellIds), 'uint32', rows),
      timestamps: isWordMode
        ? importGraphBuffer(graph, 'times', timestampsBuffer, 'uint32x2', rows)
        : importGraphBuffer(graph, 'times', timestampsBuffer, 'float32', rows),
      values: importGraphBuffer(graph, 'values', input(scene.values), 'float32', rows),
      mask: scene.mask
        ? importGraphBuffer(graph, 'mask', input(scene.mask), 'uint32', rows)
        : undefined,
      parameters,
      cellCount: scene.cellCount,
      bucketCount: scene.bucketCount,
      output: {
        counts: importGraphBuffer(graph, 'o-counts', outputs.counts, 'uint32', slotCount),
        min: importGraphBuffer(graph, 'o-min', outputs.min, 'float32', slotCount),
        max: importGraphBuffer(graph, 'o-max', outputs.max, 'float32', slotCount),
        first: importGraphBuffer(graph, 'o-first', outputs.first, 'float32', slotCount),
        last: importGraphBuffer(graph, 'o-last', outputs.last, 'float32', slotCount),
        occupiedSlots: {
          ids: importGraphBuffer(graph, 'o-ids', outputs.ids, 'uint32', capacity),
          count: importGraphBuffer(graph, 'o-count', outputs.count, 'uint32', 1),
          overflow: importGraphBuffer(graph, 'o-overflow', outputs.overflow, 'uint32', 1),
          totalCount: importGraphBuffer(graph, 'o-total', outputs.total, 'uint32', 1)
        }
      }
    })
  );
  const compiled = graph.compile();
  return {
    async run(origin, width) {
      parameterBuffer.write(
        isWordMode
          ? getGPUTemporalReductionWordParameterValues(origin, width)
          : getGPUTemporalReductionParameterValues(Number(origin), Number(width))
      );
      submitGraph(device, compiled, undefined);
      const [occupiedCount] = await readUint32(outputs.count, 1);
      const [occupiedOverflow] = await readUint32(outputs.overflow, 1);
      return {
        count: Uint32Array.from(await readUint32(outputs.counts, slotCount)),
        min: Float32Array.from(await readFloat32(outputs.min, slotCount)),
        max: Float32Array.from(await readFloat32(outputs.max, slotCount)),
        first: Float32Array.from(await readFloat32(outputs.first, slotCount)),
        last: Float32Array.from(await readFloat32(outputs.last, slotCount)),
        occupiedSlots: [],
        occupiedCount,
        occupiedOverflow,
        occupiedIds: await readUint32(outputs.ids, occupiedCount)
      };
    },
    destroy() {
      compiled.destroy();
      parameterBuffer.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
}

/** Bit-exact comparison (NaN equals NaN) of every column and of the compact occupied list. */
function expectParity(
  actual: Result,
  scene: Scene,
  origin: number | bigint,
  width: number | bigint
) {
  const expected = reduceTemporalBucketsOnCPU({...scene, origin, width});
  expect(Array.from(actual.count)).toEqual(Array.from(expected.count));
  for (const name of ['min', 'max', 'first', 'last'] as const) {
    const actualBits = Array.from(new Uint32Array(actual[name].buffer));
    const expectedBits = Array.from(new Uint32Array(expected[name].buffer));
    const bits = (values: number[]) => values.map(bit => (bit === 0x7fc00000 ? 'nan' : bit));
    expect(bits(actualBits), name).toEqual(bits(expectedBits));
  }
  const capacity = scene.capacity ?? scene.cellCount * scene.bucketCount;
  expect(actual.occupiedIds).toEqual(expected.occupiedSlots.slice(0, capacity));
  expect(actual.occupiedCount).toBe(Math.min(expected.occupiedSlots.length, capacity));
  expect(actual.occupiedOverflow).toBe(expected.occupiedSlots.length > capacity ? 1 : 0);
  return expected;
}

function createRandomScene(
  seed: number,
  rows: number,
  cellCount: number,
  bucketCount: number
): Scene {
  const random = createRandom(seed);
  const cellIds = new Uint32Array(rows);
  const timestamps = new Float32Array(rows);
  const values = new Float32Array(rows);
  for (let row = 0; row < rows; row++) {
    const roll = random();
    // Mostly valid cells with a few skipped and out-of-range cell IDs.
    cellIds[row] =
      roll < 0.03 ? 0xffffffff : roll < 0.05 ? cellCount + 2 : Math.floor(random() * cellCount);
    // Quarter-unit times on a coarse grid force many exact ties; some fall outside the buckets.
    timestamps[row] = Math.floor(random() * (bucketCount * 8 + 16)) * 0.25 - 1;
    values[row] = random() < 0.02 ? NaN : Math.round((random() - 0.5) * 2000) / 8;
  }
  return {cellIds, timestamps, values, cellCount, bucketCount};
}

it('GPUTemporalReduction matches the CPU oracle on random float32 data', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  for (const seed of [1, 2, 3]) {
    const scene = createRandomScene(seed, 4000, 7, 12);
    scene.mask = Uint32Array.from({length: 4000}, (_, row) => (row % 5 === 3 ? 0 : 1));
    const fixture = createFixture(device, scene);
    // Width 2 keeps bucket edges exact in f32; origin 0 puts negative times out of range.
    const expected = expectParity(await fixture.run(0, 2), scene, 0, 2);
    expect(expected.occupiedSlots.length).toBeGreaterThan(20);
    fixture.destroy();
  }
});

it('GPUTemporalReduction handles ties, empty buckets, and out-of-range rows', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene: Scene = {
    cellIds: Uint32Array.from([0, 0, 0, 0, 0, 1, 1, 0xffffffff, 0, 0]),
    // Rows 0-3 tie: times 4, 4, 5, 5 inside bucket 1 (width 4); row 4 is before the origin and
    // row 9 is past the last bucket (dropped, not clamped).
    timestamps: Float32Array.from([4, 4, 5, 5, -1, 8, 8, 8, 12, 100]),
    values: Float32Array.from([3, -2, 7, 9, 100, 1, 2, 3, 4, 5]),
    cellCount: 2,
    bucketCount: 3
  };
  const fixture = createFixture(device, scene);
  const actual = await fixture.run(0, 4);
  const expected = expectParity(actual, scene, 0, 4);
  expect(Array.from(expected.count)).toEqual([0, 4, 0, 0, 0, 2]);
  // First = lowest row among earliest time (row 0, value 3); last = lowest row among latest (row 2).
  expect([expected.first[1], expected.last[1], expected.min[1], expected.max[1]]).toEqual([
    3, 7, -2, 9
  ]);
  expect(actual.occupiedIds).toEqual([1, 5]);
  fixture.destroy();
});

it('GPUTemporalReduction changes the bucketing between encodings without recompiling', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createRandomScene(9, 3000, 5, 8);
  const fixture = createFixture(device, scene);
  for (const [origin, width] of [
    [0, 4],
    [8, 2],
    [-1, 8],
    [-1, 3],
    [0.5, 0.3],
    [0, 4]
  ]) {
    expectParity(await fixture.run(origin, width), scene, origin, width);
  }
  // A non-finite width drops every row.
  const empty = await fixture.run(0, 0);
  expect(empty.occupiedCount).toBe(0);
  expect(Array.from(empty.count).every(count => count === 0)).toBe(true);
  fixture.destroy();
});

it('GPUTemporalReduction clamps the occupied list to capacity and reports overflow', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createRandomScene(5, 2000, 6, 10);
  scene.capacity = 9;
  const fixture = createFixture(device, scene);
  const actual = await fixture.run(0, 2);
  expectParity(actual, scene, 0, 2);
  expect(actual.occupiedCount).toBe(9);
  expect(actual.occupiedOverflow).toBe(1);
  fixture.destroy();
});

it('GPUTemporalReduction reduces exact Int64 word times', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const origin = 1_700_000_000_000n;
  const width = 3_600_000;
  const random = createRandom(77);
  const rows = 3000;
  const cellCount = 4;
  const bucketCount = 6;
  const cellIds = new Uint32Array(rows);
  const timestamps = new BigInt64Array(rows);
  const values = new Float32Array(rows);
  for (let row = 0; row < rows; row++) {
    cellIds[row] = random() < 0.03 ? 0xffffffff : Math.floor(random() * cellCount);
    // Millisecond times well past 2^24, a few before the origin and after the last bucket, with
    // frequent exact ties and exact bucket edges.
    const bucket = Math.floor(random() * (bucketCount + 2)) - 1;
    const offset =
      random() < 0.3 ? 0 : Math.floor(random() * 4) * 900_000 + (random() < 0.1 ? width - 1 : 0);
    timestamps[row] = origin + BigInt(bucket * width + offset);
    values[row] = Math.round((random() - 0.5) * 400) / 4;
  }
  const scene: Scene = {cellIds, timestamps, values, cellCount, bucketCount};
  const fixture = createFixture(device, scene);
  expectParity(await fixture.run(origin, width), scene, origin, width);
  // Shift the origin by one bucket with the same compiled graph.
  expectParity(
    await fixture.run(origin + BigInt(width), width),
    scene,
    origin + BigInt(width),
    width
  );
  fixture.destroy();
});

it('GPUTemporalReduction assigns adversarial bucket edges like the oracle', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  for (const width of ADVERSARIAL_WIDTHS) {
    const {edgeBuckets: _edgeBuckets, ...scene} = createAdversarialEdgeScene(width);
    const fixture = createFixture(device, scene);
    const actual = await fixture.run(0, width);
    const expected = expectParity(actual, scene, 0, width);
    expect(expected.occupiedSlots.length).toBe(scene.values.length);
    fixture.destroy();
  }
});
