// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPUKeyJoin, type GPUKeyJoinKind} from '../../../src/gpu-dataframe/key-join';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  createRandom,
  joinOnCPU,
  NO_KEY_32,
  NO_KEY_64,
  packKeys,
  type OracleAggregateOperation,
  type OracleJoinResult
} from './key-join-oracle';

const SUM_SCALE = 65536;
const AGGREGATES: {
  operation: OracleAggregateOperation;
  withColumn: boolean;
  sums: boolean;
}[] = [
  {operation: 'count', withColumn: false, sums: false},
  {operation: 'count', withColumn: true, sums: false},
  {operation: 'sum', withColumn: true, sums: true},
  {operation: 'mean', withColumn: true, sums: true},
  {operation: 'minimum', withColumn: true, sums: false},
  {operation: 'maximum', withColumn: true, sums: false}
];

type Scene = {
  keyBits: 32 | 64;
  leftKeys: bigint[];
  rightKeys: bigint[];
  leftMask?: Uint32Array;
  rightMask?: Uint32Array;
  /** Right-aligned f32 values with NaNs and negative zeros. */
  values: Float32Array;
  /** Second f32 column and two u32 columns for gathers. */
  values2: Float32Array;
  words: Uint32Array;
  words2: Uint32Array;
  kind: GPUKeyJoinKind;
  capacity: number;
};

function createScene(options: {
  seed: number;
  keyBits: 32 | 64;
  leftCount: number;
  rightCount: number;
  keyRange: number;
  kind?: GPUKeyJoinKind;
  capacity?: number;
  masks?: boolean;
}): Scene {
  const random = createRandom(options.seed);
  const noKey = options.keyBits === 32 ? NO_KEY_32 : NO_KEY_64;
  // 64-bit keys vary mostly in the high word; the low word has few values.
  const makeKey = (): bigint => {
    const roll = random();
    if (roll < 0.03) {
      return noKey;
    }
    const index = BigInt(Math.floor(random() * options.keyRange));
    if (options.keyBits === 32) {
      return index === 0n ? 0n : index * 7n;
    }
    // Keys that share the low word and differ only in the high word, plus all-ones words that are
    // valid keys because only the combined all-ones key is reserved.
    const high = index % 5n === 0n ? 0xffffffffn : index;
    const low = index % 3n === 0n ? 0xffffffffn : index % 2n;
    const key = (high << 32n) | low;
    return key === NO_KEY_64 ? 0n : key;
  };
  const leftKeys = Array.from({length: options.leftCount}, makeKey);
  const rightKeys = Array.from({length: options.rightCount}, makeKey);
  const mask = (length: number) =>
    options.masks ? Uint32Array.from({length}, () => (random() < 0.15 ? 0 : 1)) : undefined;
  const floats = (length: number) =>
    Float32Array.from({length}, () => {
      const roll = random();
      return roll < 0.04 ? NaN : roll < 0.07 ? -0 : Math.fround((random() - 0.4) * 1000);
    });
  return {
    keyBits: options.keyBits,
    leftKeys,
    rightKeys,
    leftMask: mask(options.leftCount),
    rightMask: mask(options.rightCount),
    values: floats(options.rightCount),
    values2: floats(options.rightCount),
    words: Uint32Array.from({length: options.rightCount}, () => Math.floor(random() * 2 ** 32)),
    words2: Uint32Array.from({length: options.rightCount}, (_, row) => row * 3),
    kind: options.kind ?? 'left',
    capacity: options.capacity ?? options.leftCount
  };
}

class Fixture {
  readonly buffers: Buffer[] = [];
  constructor(readonly device: Device) {}
  input(values: Float32Array | Uint32Array): Buffer {
    const buffer = createInputBuffer(this.device, values);
    this.buffers.push(buffer);
    return buffer;
  }
  output(length: number): Buffer {
    const buffer = createOutputBuffer(this.device, length);
    this.buffers.push(buffer);
    return buffer;
  }
  destroy(): void {
    for (const buffer of this.buffers) {
      buffer.destroy();
    }
  }
}

type Snapshot = {
  rightRows: number[];
  matchCounts: number[];
  matched: number[];
  rightMatched: number[];
  gathered: number[][];
  aggregateValues: number[][];
  aggregateSums: number[][];
  rows: number[];
  count: number;
  overflow: number;
  total: number;
};

/** One compiled join graph with writable inputs; every optional output is requested. */
function createJoin(fixture: Fixture, scene: Scene) {
  const {device} = fixture;
  const leftCount = scene.leftKeys.length;
  const rightCount = scene.rightKeys.length;
  const keyFormat = scene.keyBits === 64 ? 'uint32x2' : 'uint32';
  const graph = new GPUCommandGraph(device, {id: 'key-join-graph'});
  const leftKeysBuffer = fixture.input(packKeys(scene.leftKeys, scene.keyBits));
  const rightKeysBuffer = fixture.input(packKeys(scene.rightKeys, scene.keyBits));
  const leftMaskBuffer = fixture.input(scene.leftMask ?? new Uint32Array(leftCount).fill(1));
  const rightMaskBuffer = fixture.input(scene.rightMask ?? new Uint32Array(rightCount).fill(1));
  const valuesBuffer = fixture.input(scene.values);
  const columns = {
    values2: fixture.input(scene.values2),
    words: fixture.input(scene.words),
    words2: fixture.input(scene.words2)
  };
  const outputs = {
    rightRows: fixture.output(leftCount),
    matchCounts: fixture.output(leftCount),
    matched: fixture.output(leftCount),
    rightMatched: fixture.output(rightCount),
    ids: fixture.output(scene.capacity),
    count: fixture.output(1),
    overflow: fixture.output(1),
    total: fixture.output(1),
    gathered: [0, 1, 2, 3, 4].map(() => fixture.output(leftCount)),
    aggregates: AGGREGATES.map(() => fixture.output(leftCount)),
    sums: AGGREGATES.map(() => fixture.output(2 * leftCount))
  };
  const left = (name: string, buffer: Buffer) =>
    importGraphBuffer(graph, name, buffer, 'uint32', leftCount);
  const leftF32 = (name: string, buffer: Buffer) =>
    importGraphBuffer(graph, name, buffer, 'float32', leftCount);
  const rightF32 = (name: string, buffer: Buffer) =>
    importGraphBuffer(graph, name, buffer, 'float32', rightCount);
  const rightU32 = (name: string, buffer: Buffer) =>
    importGraphBuffer(graph, name, buffer, 'uint32', rightCount);
  const valuesView = rightF32('values', valuesBuffer);
  graph.add(
    new GPUKeyJoin({
      leftKeys: importGraphBuffer(graph, 'left-keys', leftKeysBuffer, keyFormat, leftCount),
      rightKeys: importGraphBuffer(graph, 'right-keys', rightKeysBuffer, keyFormat, rightCount),
      leftMask: left('left-mask', leftMaskBuffer),
      rightMask: rightU32('right-mask', rightMaskBuffer),
      kind: scene.kind,
      sumScale: SUM_SCALE,
      gather: [
        {
          column: valuesView,
          output: leftF32('gather-0', outputs.gathered[0])
        },
        {
          column: rightU32('words', columns.words),
          output: left('gather-1', outputs.gathered[1])
        },
        {
          column: rightF32('values2', columns.values2),
          output: leftF32('gather-2', outputs.gathered[2])
        },
        {
          column: rightU32('words2', columns.words2),
          output: left('gather-3', outputs.gathered[3])
        },
        {
          column: rightU32('words-again', columns.words),
          output: left('gather-4', outputs.gathered[4])
        }
      ],
      aggregates: AGGREGATES.map((aggregate, index) => ({
        operation: aggregate.operation,
        column: aggregate.withColumn ? valuesView : undefined,
        output: leftF32(`aggregate-${index}`, outputs.aggregates[index]),
        sums: aggregate.sums
          ? importGraphBuffer(graph, `sums-${index}`, outputs.sums[index], 'uint32x2', leftCount)
          : undefined
      })),
      output: {
        rightRows: left('right-rows', outputs.rightRows),
        matchCounts: left('match-counts', outputs.matchCounts),
        matched: left('matched', outputs.matched),
        rightMatched: rightU32('right-matched', outputs.rightMatched),
        rows:
          scene.kind === 'inner'
            ? {
                ids: importGraphBuffer(graph, 'ids', outputs.ids, 'uint32', scene.capacity),
                count: importGraphBuffer(graph, 'count', outputs.count, 'uint32', 1),
                overflow: importGraphBuffer(graph, 'overflow', outputs.overflow, 'uint32', 1),
                requiredCount: importGraphBuffer(graph, 'total', outputs.total, 'uint32', 1)
              }
            : undefined
      }
    })
  );
  const compiled = graph.compile();
  async function run(): Promise<Snapshot> {
    submitGraph(device, compiled, undefined);
    const count = scene.kind === 'inner' ? (await readUint32(outputs.count, 1))[0] : 0;
    return {
      rightRows: await readUint32(outputs.rightRows, leftCount),
      matchCounts: await readUint32(outputs.matchCounts, leftCount),
      matched: await readUint32(outputs.matched, leftCount),
      rightMatched: await readUint32(outputs.rightMatched, rightCount),
      gathered: await Promise.all(outputs.gathered.map(buffer => readUint32(buffer, leftCount))),
      // Compared as raw words so NaN payloads and -0 are bit exact.
      aggregateValues: await Promise.all(
        outputs.aggregates.map(buffer => readUint32(buffer, leftCount))
      ),
      aggregateSums: await Promise.all(
        outputs.sums.map(buffer => readUint32(buffer, 2 * leftCount))
      ),
      rows: scene.kind === 'inner' ? await readUint32(outputs.ids, scene.capacity) : [],
      count,
      overflow: scene.kind === 'inner' ? (await readUint32(outputs.overflow, 1))[0] : 0,
      total: scene.kind === 'inner' ? (await readUint32(outputs.total, 1))[0] : 0
    };
  }
  return {
    compiled,
    run,
    buffers: {
      leftMask: leftMaskBuffer,
      rightMask: rightMaskBuffer,
      values: valuesBuffer,
      leftKeys: leftKeysBuffer,
      rightKeys: rightKeysBuffer
    }
  };
}

function getOracle(scene: Scene): OracleJoinResult {
  return joinOnCPU({
    keyBits: scene.keyBits,
    leftKeys: scene.leftKeys,
    rightKeys: scene.rightKeys,
    leftMask: scene.leftMask,
    rightMask: scene.rightMask,
    kind: scene.kind,
    capacity: scene.capacity,
    sumScale: SUM_SCALE,
    gatherColumns: [
      {words: new Uint32Array(scene.values.buffer.slice(0)), isFloat: true},
      {words: scene.words, isFloat: false},
      {words: new Uint32Array(scene.values2.buffer.slice(0)), isFloat: true},
      {words: scene.words2, isFloat: false},
      {words: scene.words, isFloat: false}
    ],
    aggregates: AGGREGATES.map(aggregate => ({
      operation: aggregate.operation,
      values: aggregate.withColumn ? scene.values : undefined
    }))
  });
}

const f32Words = (value: number) => new Uint32Array(Float32Array.of(value).buffer)[0];

/** Compares a GPU snapshot with the oracle: exact except mean and sum decodes (stated below). */
function expectSnapshot(actual: Snapshot, expected: OracleJoinResult, scene: Scene, label: string) {
  expect(actual.rightRows, `${label} rightRows`).toEqual(expected.rightRows);
  expect(actual.matchCounts, `${label} matchCounts`).toEqual(expected.matchCounts);
  expect(actual.matched, `${label} matched`).toEqual(expected.matched);
  expect(actual.rightMatched, `${label} rightMatched`).toEqual(expected.rightMatched);
  for (const [index, words] of expected.gathered.entries()) {
    expect(actual.gathered[index], `${label} gather ${index}`).toEqual(words);
  }
  for (const [index, aggregate] of AGGREGATES.entries()) {
    const expectedAggregate = expected.aggregates[index];
    const words = actual.aggregateValues[index];
    for (const [row, expectedValue] of expectedAggregate.values.entries()) {
      const actualValue = new Float32Array(Uint32Array.of(words[row]).buffer)[0];
      if (Number.isNaN(expectedValue)) {
        expect(Number.isNaN(actualValue), `${label} aggregate ${index} row ${row} NaN`).toBe(true);
      } else if (aggregate.operation === 'sum' || aggregate.operation === 'mean') {
        // f32 decode and division: relative tolerance 2e-6 (the GPU may round division by 2.5 ulp).
        expect(
          Math.abs(actualValue - expectedValue),
          `${label} aggregate ${index} row ${row}`
        ).toBeLessThanOrEqual(Math.abs(expectedValue) * 2e-6 + 1e-6);
      } else {
        // Counts, minimum and maximum are exact, including the sign of zero.
        expect(words[row], `${label} aggregate ${index} row ${row}`).toBe(f32Words(expectedValue));
      }
    }
    if (aggregate.sums) {
      const sums = actual.aggregateSums[index];
      for (const [row, expectedSum] of expectedAggregate.sums.entries()) {
        const actualSum = BigInt.asIntN(
          64,
          (BigInt(sums[2 * row + 1]) << 32n) | BigInt(sums[2 * row])
        );
        expect(actualSum, `${label} sums ${index} row ${row}`).toBe(expectedSum);
      }
    }
  }
  if (scene.kind === 'inner') {
    expect(actual.total, `${label} total`).toBe(expected.innerTotal);
    expect(actual.count, `${label} count`).toBe(Math.min(expected.innerTotal, scene.capacity));
    expect(actual.overflow, `${label} overflow`).toBe(expected.innerOverflow);
    expect(actual.rows.slice(0, actual.count), `${label} rows`).toEqual(expected.innerRows);
    expect(
      actual.rows.slice(actual.count).every(id => id === 0xffffffff),
      `${label} rows tail`
    ).toBe(true);
  }
}

const SCENES: [string, Parameters<typeof createScene>[0]][] = [
  [
    'u32 keys, duplicates, masks',
    {
      seed: 11,
      keyBits: 32,
      leftCount: 1200,
      rightCount: 700,
      keyRange: 150,
      masks: true
    }
  ],
  [
    'u64 keys differing in the high word',
    {
      seed: 23,
      keyBits: 64,
      leftCount: 900,
      rightCount: 800,
      keyRange: 120,
      masks: true
    }
  ],
  [
    'u64 keys, unique right keys, no masks',
    {seed: 5, keyBits: 64, leftCount: 500, rightCount: 400, keyRange: 100000}
  ],
  [
    'right rows past 4096 and a heavy key',
    {
      seed: 31,
      keyBits: 64,
      leftCount: 3000,
      rightCount: 9000,
      keyRange: 25,
      masks: true
    }
  ],
  ['tiny tables', {seed: 2, keyBits: 32, leftCount: 3, rightCount: 2, keyRange: 3}]
];

for (const [name, options] of SCENES) {
  it(`GPUKeyJoin left join matches the CPU oracle: ${name}`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) {
      return;
    }
    const fixture = new Fixture(device);
    const scene = createScene(options);
    const join = createJoin(fixture, scene);
    const expected = getOracle(scene);
    expect(expected.matched.some(Boolean), 'scene has matches').toBe(true);
    const first = await join.run();
    expectSnapshot(first, expected, scene, name);
    // A second encoding of the same graph is bitwise identical.
    expect(await join.run()).toEqual(first);
    join.compiled.destroy();
    fixture.destroy();
  });
}

it('GPUKeyJoin inner join compacts matched rows and reports overflow', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  for (const [capacity, label] of [
    [4000, 'roomy capacity'],
    [37, 'overflowing capacity'],
    [1, 'single row capacity']
  ] as const) {
    const fixture = new Fixture(device);
    const scene = createScene({
      seed: 41,
      keyBits: 64,
      leftCount: 2500,
      rightCount: 500,
      keyRange: 400,
      masks: true,
      kind: 'inner',
      capacity
    });
    const join = createJoin(fixture, scene);
    const expected = getOracle(scene);
    const actual = await join.run();
    expect(expected.innerTotal).toBeGreaterThan(37);
    expect(expected.innerOverflow).toBe(capacity < expected.innerTotal ? 1 : 0);
    expectSnapshot(actual, expected, scene, label);
    expect(await join.run()).toEqual(actual);
    join.compiled.destroy();
    fixture.destroy();
  }
});

it('GPUKeyJoin follows per-frame input rewrites without recompiling', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const fixture = new Fixture(device);
  const scene = createScene({
    seed: 77,
    keyBits: 32,
    leftCount: 800,
    rightCount: 600,
    keyRange: 90,
    masks: true,
    kind: 'inner',
    capacity: 100
  });
  const join = createJoin(fixture, scene);
  const compiled = join.compiled;
  const totals: number[] = [];
  for (const seed of [1, 2, 3, 1]) {
    const next = createScene({
      seed,
      keyBits: 32,
      leftCount: 800,
      rightCount: 600,
      keyRange: 30 + 30 * seed,
      masks: true,
      kind: 'inner',
      capacity: 100
    });
    // Same buffers, new contents: keys, masks and values.
    join.buffers.leftKeys.write(packKeys(next.leftKeys, 32));
    join.buffers.rightKeys.write(packKeys(next.rightKeys, 32));
    join.buffers.leftMask.write(next.leftMask!);
    join.buffers.rightMask.write(next.rightMask!);
    join.buffers.values.write(next.values);
    const expected = getOracle({
      ...scene,
      ...next,
      words: scene.words,
      words2: scene.words2,
      values2: scene.values2
    });
    const actual = await join.run();
    // Gathered columns 1 to 4 keep the original right columns, which the oracle mirrors.
    expectSnapshot(actual, expected, scene, `seed ${seed}`);
    totals.push(actual.total);
    expect(join.compiled).toBe(compiled);
  }
  expect(new Set(totals).size).toBeGreaterThan(2);
  expect(totals[0]).toBe(totals[3]);
  compiled.destroy();
  fixture.destroy();
});

it('GPUKeyJoin handles keys with no valid right rows', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const fixture = new Fixture(device);
  const scene = createScene({
    seed: 9,
    keyBits: 64,
    leftCount: 40,
    rightCount: 30,
    keyRange: 6
  });
  scene.rightMask = new Uint32Array(30);
  const join = createJoin(fixture, scene);
  const actual = await join.run();
  expectSnapshot(actual, getOracle(scene), scene, 'masked right');
  expect(actual.matched.every(value => value === 0)).toBe(true);
  expect(actual.rightMatched.every(value => value === 0)).toBe(true);
  join.compiled.destroy();
  fixture.destroy();
});
