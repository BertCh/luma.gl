// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPUTemporalReductionParameterValues,
  getGPUTemporalReductionWordParameterValues,
  GPUTemporalReduction
} from '../../../src/gpu-dataframe/temporal-reduction';
import {getInt64TimeWords} from '../../../src/gpu-dataframe/time-window-filter';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {createAdversarialEdgeScene, ADVERSARIAL_WIDTHS} from './adversarial-edges';

type Scene = {
  cellIds: Uint32Array;
  timestamps: Float32Array | BigInt64Array;
  values: Float32Array;
  mask?: Uint32Array;
  cellCount: number;
  bucketCount: number;
};

type Columns = {
  counts: number[];
  minimums: number[];
  maximums: number[];
  first: number[];
  last: number[];
  occupiedCount: number;
  occupiedOverflow: number;
  occupiedIds: number[];
  occupiedRequired: number;
};

type Fixture = {
  run(origin: number | bigint, width: number | bigint): Promise<Columns>;
  destroy(): void;
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

function createFixture(device: Device, scene: Scene, sorted: boolean): Fixture {
  const isWordMode = scene.timestamps instanceof BigInt64Array;
  const slotCount = scene.cellCount * scene.bucketCount;
  const rows = scene.values.length;
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
  const graph = new GPUCommandGraph(device, {id: `temporal-sorted-${sorted}`});
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'bucket-parameters',
    format: isWordMode ? 'uint32' : 'float32',
    length: 4
  });
  const outputs = {
    counts: output(slotCount),
    minimums: output(slotCount),
    maximums: output(slotCount),
    first: output(slotCount),
    last: output(slotCount),
    ids: output(slotCount),
    count: output(1),
    overflow: output(1),
    total: output(1)
  };
  const timestampsBuffer = input(
    scene.timestamps instanceof BigInt64Array
      ? getInt64TimeWords(scene.timestamps)
      : scene.timestamps
  );
  graph.add(
    new GPUTemporalReduction({
      id: 'reduction',
      sorted,
      cellIds: importGraphBuffer(graph, 'cells', input(scene.cellIds), 'uint32', rows),
      timestamps: isWordMode
        ? importGraphBuffer(graph, 'times', timestampsBuffer, 'uint32x2', rows)
        : importGraphBuffer(graph, 'times', timestampsBuffer, 'float32', rows),
      values: importGraphBuffer(graph, 'values', input(scene.values), 'float32', rows),
      mask: scene.mask
        ? importGraphBuffer(graph, 'mask', input(scene.mask), 'uint32', rows)
        : undefined,
      parameters: parameterBuffer.importToGraph(graph),
      cellCount: scene.cellCount,
      bucketCount: scene.bucketCount,
      output: {
        counts: importGraphBuffer(graph, 'o-counts', outputs.counts, 'uint32', slotCount),
        minimums: importGraphBuffer(graph, 'o-min', outputs.minimums, 'float32', slotCount),
        maximums: importGraphBuffer(graph, 'o-max', outputs.maximums, 'float32', slotCount),
        first: importGraphBuffer(graph, 'o-first', outputs.first, 'float32', slotCount),
        last: importGraphBuffer(graph, 'o-last', outputs.last, 'float32', slotCount),
        occupiedSlots: {
          ids: importGraphBuffer(graph, 'o-ids', outputs.ids, 'uint32', slotCount),
          status: {
            count: importGraphBuffer(graph, 'o-count', outputs.count, 'uint32', 1),
            overflow: importGraphBuffer(graph, 'o-overflow', outputs.overflow, 'uint32', 1),
            requiredCount: importGraphBuffer(graph, 'o-total', outputs.total, 'uint32', 1)
          }
        }
      }
    })
  );
  const compiled = graph.compile();
  const bits = (values: number[]) =>
    Array.from(new Uint32Array(Float32Array.from(values).buffer)).map(bit =>
      bit === 0x7fc00000 ? -1 : bit
    );
  return {
    async run(origin, width) {
      parameterBuffer.write(
        isWordMode
          ? getGPUTemporalReductionWordParameterValues(origin, width)
          : getGPUTemporalReductionParameterValues(Number(origin), Number(width))
      );
      submitGraph(device, compiled, undefined);
      const [occupiedCount] = await readUint32(outputs.count, 1);
      return {
        counts: await readUint32(outputs.counts, slotCount),
        // Bit patterns, so -0 and NaN compare exactly.
        minimums: bits(await readFloat32(outputs.minimums, slotCount)),
        maximums: bits(await readFloat32(outputs.maximums, slotCount)),
        first: bits(await readFloat32(outputs.first, slotCount)),
        last: bits(await readFloat32(outputs.last, slotCount)),
        occupiedCount,
        occupiedOverflow: (await readUint32(outputs.overflow, 1))[0],
        occupiedIds: await readUint32(outputs.ids, occupiedCount),
        occupiedRequired: (await readUint32(outputs.total, 1))[0]
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

function createSortedFloatScene(
  seed: number,
  rows: number,
  cellCount: number,
  bucketCount: number
): Scene {
  const random = createRandom(seed);
  // Duplicated times, so first/last ties break by row index inside the range.
  const timestamps = Float32Array.from({length: rows}, () => Math.floor(random() * 400) / 4).sort();
  return {
    cellIds: Uint32Array.from({length: rows}, () =>
      random() < 0.05 ? 0xffffffff : Math.floor(random() * cellCount)
    ),
    timestamps,
    values: Float32Array.from({length: rows}, () =>
      random() < 0.03 ? NaN : Math.fround(random() * 200 - 100)
    ),
    mask: Uint32Array.from({length: rows}, () => (random() < 0.9 ? 1 : 0)),
    cellCount,
    bucketCount
  };
}

function createSortedWordScene(
  seed: number,
  rows: number,
  cellCount: number,
  bucketCount: number
): Scene {
  const random = createRandom(seed);
  const values = Array.from({length: rows}, () => BigInt(Math.floor(random() * 6000) - 3000));
  values.push(-(2n ** 40n), 4_294_967_295n, 4_294_967_296n, 2n ** 40n);
  values.sort((left, right) => (left < right ? -1 : left > right ? 1 : 0));
  const total = values.length;
  return {
    cellIds: Uint32Array.from({length: total}, () => Math.floor(random() * cellCount)),
    timestamps: BigInt64Array.from(values),
    values: Float32Array.from({length: total}, () => Math.fround(random() * 50 - 25)),
    cellCount,
    bucketCount
  };
}

async function expectSortedEqualsUnsorted(
  device: Device,
  scene: Scene,
  parameters: [number | bigint, number | bigint][]
) {
  const unsorted = createFixture(device, scene, false);
  const sorted = createFixture(device, scene, true);
  for (const [origin, width] of parameters) {
    const expected = await unsorted.run(origin, width);
    const actual = await sorted.run(origin, width);
    expect(actual, `${origin} ${width}`).toEqual(expected);
  }
  unsorted.destroy();
  sorted.destroy();
}

it('GPUTemporalReduction sorted equals the unsorted path bit for bit (float times)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Few slots take the privatized extreme kernel, many the global-atomics one.
  for (const [cells, buckets] of [
    [3, 8],
    [40, 100]
  ]) {
    await expectSortedEqualsUnsorted(device, createSortedFloatScene(5, 3000, cells, buckets), [
      [0, 1], // range inside the data
      [20, 2.5], // range in the middle
      [-50, 1], // starts before the data
      [90, 1], // runs past the end
      [-1000, 1000], // everything in one bucket
      [1000, 1], // nothing in range
      [-1000, 1], // nothing in range (all above)
      [0, 0], // invalid width
      [0, NaN]
    ]);
  }
});

it('GPUTemporalReduction sorted equals the unsorted path on the adversarial bucket edges', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const width of ADVERSARIAL_WIDTHS) {
    const scene = createAdversarialEdgeScene(width);
    await expectSortedEqualsUnsorted(device, scene, [
      [0, width],
      [Math.fround(10 * width), width],
      [-3 * width, width]
    ]);
  }
});

it('GPUTemporalReduction sorted equals the unsorted path (Int64 word times)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const [cells, buckets] of [
    [3, 8],
    [40, 100]
  ]) {
    await expectSortedEqualsUnsorted(device, createSortedWordScene(9, 2000, cells, buckets), [
      [0n, 100n],
      [-3000n, 500n],
      [-(2n ** 40n), 2n ** 31n],
      [4_294_967_290n, 3n],
      [2n ** 40n, 1n],
      [5000n, 10n],
      [-(2n ** 63n), 0xffffffffn],
      [0n, 1n]
    ]);
  }
});

it('GPUTemporalReduction sorted A/B: full reduction versus range mode on 1e7 sorted rows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = 10_000_000;
  const cellCount = 16;
  const bucketCount = 64;
  const timestamps = new Float32Array(rows);
  const cellIds = new Uint32Array(rows);
  const values = new Float32Array(rows);
  for (let row = 0; row < rows; row++) {
    timestamps[row] = row / 16;
    cellIds[row] = row % cellCount;
    values[row] = (row * 2654435761) % 1000;
  }
  const scene: Scene = {cellIds, timestamps, values, cellCount, bucketCount};
  const unsorted = createFixture(device, scene, false);
  const sorted = createFixture(device, scene, true);
  // 64 buckets of width 100 cover 6400 time units, about 1% of the rows; each step scrubs one bucket.
  const time = async (fixture: Fixture) => {
    await fixture.run(250_000, 100);
    const samples: number[] = [];
    for (let iteration = 0; iteration < 10; iteration++) {
      const begin = performance.now();
      await fixture.run(250_000 + iteration * 100, 100);
      samples.push(performance.now() - begin);
    }
    return samples.sort((left, right) => left - right)[5];
  };
  const unsortedMilliseconds = await time(unsorted);
  const sortedMilliseconds = await time(sorted);
  expect(await sorted.run(250_000, 100)).toEqual(await unsorted.run(250_000, 100));
  console.log(
    `GPUTemporalReduction A/B ${rows} rows: unsorted ${unsortedMilliseconds.toFixed(2)} ms, ` +
      `sorted ${sortedMilliseconds.toFixed(2)} ms`
  );
  unsorted.destroy();
  sorted.destroy();
});
