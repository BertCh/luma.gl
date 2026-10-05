// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUMapGraphParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/map-graphs';
import {
  GPUGroupStatistics,
  type GPUGroupStatistic,
  type GPUGroupStatisticsColumnOutput
} from '../../../src/map-graphs/group-statistics';
import {createInputBuffer, createOutputBuffer} from '../map-graph-test-utils';
import {
  canonicalizeValue,
  computeGroupStatisticsOnCPU,
  createRandom,
  type GroupStatisticsOracleResult
} from './group-statistics-oracle';

const SUM_SCALE = 65536;
const ALL_STATISTICS: GPUGroupStatistic[] = [
  'count',
  'sum',
  'mean',
  'minimum',
  'maximum',
  'variance',
  'standardDeviation',
  'skewness',
  'kurtosis',
  'median',
  'percentiles',
  'mode',
  'uniqueCount',
  'zScore'
];

/**
 * Stated float32 tolerances of the moments against the float64 oracle. Measured worst relative
 * errors on this data are about 1e-7 (mean, variance), 2e-6 (skewness, kurtosis) and 7e-6
 * (z-score); the tolerances leave an order of magnitude of headroom for other GPUs.
 */
const TOLERANCE = {
  mean: {relative: 1e-6, absolute: 1e-6},
  variance: {relative: 2e-5, absolute: 1e-4},
  skewness: {relative: 1e-4, absolute: 1e-4},
  kurtosis: {relative: 1e-4, absolute: 1e-4},
  zScore: {relative: 1e-4, absolute: 1e-4}
};

const FIELDS: Record<GPUGroupStatistic, (keyof GPUGroupStatisticsColumnOutput)[]> = {
  count: ['counts'],
  sum: ['sums', 'sumValues'],
  mean: ['means'],
  minimum: ['minimums'],
  maximum: ['maximums'],
  variance: ['variances'],
  standardDeviation: ['standardDeviations'],
  skewness: ['skewness'],
  kurtosis: ['kurtosis'],
  median: ['medians'],
  percentiles: ['percentiles'],
  mode: ['modes'],
  uniqueCount: ['uniqueCounts'],
  zScore: ['zScores']
};

type ColumnScene = {values: Float32Array; statistics: GPUGroupStatistic[]};

type Scene = {
  keyBits: 32 | 64;
  keys: bigint[];
  mask?: Uint32Array;
  columns: ColumnScene[];
  variance?: 'sample' | 'population';
  fractions: number[];
  capacity: number;
};

type ActualColumn = Partial<Record<keyof GPUGroupStatisticsColumnOutput, number[]>>;

type Actual = {
  count: number;
  overflow: number;
  total: number;
  keys: bigint[];
  counts: number[];
  columns: ActualColumn[];
  /** Raw words of every output buffer, in a fixed order. */
  words: number[];
};

class Harness {
  readonly buffers: Buffer[] = [];
  constructor(readonly device: Device) {}

  input(values: Float32Array | Uint32Array): Buffer {
    const buffer = createInputBuffer(this.device, values);
    this.buffers.push(buffer);
    return buffer;
  }

  output(words: number): Buffer {
    const buffer = createOutputBuffer(this.device, words);
    this.buffers.push(buffer);
    return buffer;
  }

  destroy(): void {
    for (const buffer of this.buffers) {
      buffer.destroy();
    }
  }
}

function packKeys(keys: readonly bigint[], keyBits: 32 | 64): Uint32Array {
  const words = new Uint32Array(keys.length * (keyBits === 64 ? 2 : 1));
  for (const [row, key] of keys.entries()) {
    if (keyBits === 64) {
      words[2 * row] = Number(key & 0xffffffffn);
      words[2 * row + 1] = Number(key >> 32n);
    } else {
      words[row] = Number(key);
    }
  }
  return words;
}

async function readWords(buffer: Buffer, length: number): Promise<Uint32Array> {
  const bytes = await buffer.readAsync();
  return new Uint32Array(bytes.slice(0, length * 4).buffer.slice(0, length * 4));
}

/** One compiled graph with rewritable keys, mask, values and percentile fractions. */
async function createRun(harness: Harness, scene: Scene) {
  const {device} = harness;
  const rows = scene.keys.length;
  const {capacity} = scene;
  const fractionCount = scene.fractions.length;
  const graph = new GPUCommandGraph(device, {id: 'group-statistics-graph'});
  const keyWordCount = scene.keyBits === 64 ? 2 : 1;
  const keyFormat = scene.keyBits === 64 ? 'uint32x2' : 'uint32';
  const keysBuffer = harness.input(packKeys(scene.keys, scene.keyBits));
  const maskBuffer = harness.input(scene.mask ?? new Uint32Array(rows).fill(1));
  const valueBuffers = scene.columns.map(column => harness.input(column.values));
  const parameters = fractionCount
    ? new GPUMapGraphParameterBuffer(device, {
        id: 'fractions',
        format: 'float32',
        length: fractionCount,
        values: Float32Array.from(scene.fractions)
      })
    : undefined;
  const outputBuffers = {
    keys: harness.output(capacity * keyWordCount),
    counts: harness.output(capacity),
    count: harness.output(1),
    overflow: harness.output(1),
    total: harness.output(1)
  };
  const columnOutputs: {
    views: GPUGroupStatisticsColumnOutput;
    buffers: Partial<Record<keyof GPUGroupStatisticsColumnOutput, Buffer>>;
    lengths: Partial<Record<keyof GPUGroupStatisticsColumnOutput, number>>;
  }[] = [];
  const columns = scene.columns.map((column, columnIndex) => {
    const views: Record<string, GraphDataView> = {};
    const buffers: Partial<Record<keyof GPUGroupStatisticsColumnOutput, Buffer>> = {};
    const lengths: Partial<Record<keyof GPUGroupStatisticsColumnOutput, number>> = {};
    for (const statistic of column.statistics) {
      for (const field of FIELDS[statistic]) {
        const length =
          field === 'zScores'
            ? rows
            : field === 'percentiles'
              ? capacity * fractionCount
              : capacity;
        const format =
          field === 'sums'
            ? 'uint32x2'
            : field === 'counts' || field === 'uniqueCounts'
              ? 'uint32'
              : 'float32';
        const buffer = harness.output(length * (field === 'sums' ? 2 : 1));
        buffers[field] = buffer;
        lengths[field] = length * (field === 'sums' ? 2 : 1);
        views[field] = importGraphBuffer(graph, `c${columnIndex}-${field}`, buffer, format, length);
      }
    }
    columnOutputs.push({
      views: views as GPUGroupStatisticsColumnOutput,
      buffers,
      lengths
    });
    return {
      values: importGraphBuffer(
        graph,
        `values-${columnIndex}`,
        valueBuffers[columnIndex],
        'float32',
        rows
      ),
      statistics: column.statistics,
      output: views as GPUGroupStatisticsColumnOutput
    };
  });
  const recipe = new GPUGroupStatistics({
    keys: importGraphBuffer(graph, 'keys', keysBuffer, keyFormat, rows) as never,
    mask: importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', rows),
    columns,
    variance: scene.variance,
    percentiles: parameters?.importToGraph(graph),
    sumScale: SUM_SCALE,
    output: {
      keys: importGraphBuffer(graph, 'out-keys', outputBuffers.keys, keyFormat, capacity) as never,
      counts: importGraphBuffer(graph, 'out-counts', outputBuffers.counts, 'uint32', capacity),
      count: importGraphBuffer(graph, 'out-count', outputBuffers.count, 'uint32', 1),
      overflow: importGraphBuffer(graph, 'out-overflow', outputBuffers.overflow, 'uint32', 1),
      totalCount: importGraphBuffer(graph, 'out-total', outputBuffers.total, 'uint32', 1)
    }
  });
  graph.add(recipe);
  const compiled = graph.compile();
  const allOutputs: [Buffer, number][] = [
    [outputBuffers.keys, capacity * keyWordCount],
    [outputBuffers.counts, capacity],
    [outputBuffers.count, 1],
    [outputBuffers.overflow, 1],
    [outputBuffers.total, 1]
  ];
  for (const output of columnOutputs) {
    for (const [field, buffer] of Object.entries(output.buffers)) {
      allOutputs.push([buffer, output.lengths[field as keyof GPUGroupStatisticsColumnOutput]!]);
    }
  }
  return {
    compiled,
    parameters,
    keysBuffer,
    maskBuffer,
    valueBuffers,
    poison(): void {
      for (const [buffer, length] of allOutputs) {
        buffer.write(new Uint32Array(length).fill(0xa5a5a5a5));
      }
    },
    async run(): Promise<Actual> {
      submitGraph(device, compiled, undefined);
      const words: number[] = [];
      for (const [buffer, length] of allOutputs) {
        words.push(...(await readWords(buffer, length)));
      }
      const [count] = await readWords(outputBuffers.count, 1);
      const [overflow] = await readWords(outputBuffers.overflow, 1);
      const [total] = await readWords(outputBuffers.total, 1);
      const keyWords = await readWords(outputBuffers.keys, capacity * keyWordCount);
      const keys: bigint[] = [];
      for (let row = 0; row < capacity; row++) {
        keys.push(
          scene.keyBits === 64
            ? (BigInt(keyWords[2 * row + 1]) << 32n) | BigInt(keyWords[2 * row])
            : BigInt(keyWords[row])
        );
      }
      const counts = Array.from(await readWords(outputBuffers.counts, capacity));
      const actualColumns: ActualColumn[] = [];
      for (const output of columnOutputs) {
        const column: ActualColumn = {};
        for (const [field, buffer] of Object.entries(output.buffers)) {
          const name = field as keyof GPUGroupStatisticsColumnOutput;
          const raw = await readWords(buffer, output.lengths[name]!);
          column[name] =
            name === 'counts' || name === 'uniqueCounts' || name === 'sums'
              ? Array.from(raw)
              : Array.from(new Float32Array(raw.buffer, raw.byteOffset, raw.length));
        }
        actualColumns.push(column);
      }
      return {
        count,
        overflow,
        total,
        keys,
        counts,
        columns: actualColumns,
        words
      };
    },
    destroy(): void {
      compiled.destroy();
      parameters?.destroy();
    }
  };
}

function expectClose(
  actual: number,
  expected: number,
  tolerance: {relative: number; absolute: number},
  label: string
): void {
  if (Number.isNaN(expected)) {
    expect(Number.isNaN(actual), `${label} should be NaN, got ${actual}`).toBe(true);
    return;
  }
  expect(Math.abs(actual - expected), `${label}: ${actual} vs ${expected}`).toBeLessThanOrEqual(
    tolerance.relative * Math.abs(expected) + tolerance.absolute
  );
}

function expectExactFloat(actual: number, expected: number, label: string): void {
  const wanted = Math.fround(expected);
  expect(
    Object.is(actual, wanted) || (Number.isNaN(actual) && Number.isNaN(wanted)),
    `${label}: ${actual} vs ${wanted}`
  ).toBe(true);
}

function expectMatchesOracle(
  actual: Actual,
  scene: Scene,
  label: string
): GroupStatisticsOracleResult {
  const expected = computeGroupStatisticsOnCPU({
    keys: scene.keys,
    keyBits: scene.keyBits,
    mask: scene.mask,
    columns: scene.columns.map(column => column.values),
    variance: scene.variance ?? 'sample',
    fractions: scene.fractions,
    sumScale: SUM_SCALE,
    capacity: scene.capacity
  });
  const groupCount = expected.groups.length;
  const emptyKey = scene.keyBits === 64 ? 0xffffffffffffffffn : 0xffffffffn;
  expect(actual.total, `${label} total`).toBe(expected.totalCount);
  expect(actual.count, `${label} count`).toBe(groupCount);
  expect(actual.overflow, `${label} overflow`).toBe(expected.totalCount > scene.capacity ? 1 : 0);
  expect(actual.keys.slice(0, groupCount), `${label} keys`).toEqual(
    expected.groups.map(group => group.key)
  );
  expect(actual.counts.slice(0, groupCount), `${label} counts`).toEqual(
    expected.groups.map(group => group.count)
  );
  for (let row = groupCount; row < scene.capacity; row++) {
    expect(actual.keys[row], `${label} tail key ${row}`).toBe(emptyKey);
    expect(actual.counts[row], `${label} tail count ${row}`).toBe(0);
  }
  const fractionCount = scene.fractions.length;
  for (const [columnIndex, column] of scene.columns.entries()) {
    const wanted = new Set(column.statistics);
    const out = actual.columns[columnIndex];
    for (let group = 0; group < scene.capacity; group++) {
      const isTail = group >= groupCount;
      const stats = isTail ? undefined : expected.groups[group].columns[columnIndex];
      const where = `${label} column ${columnIndex} group ${group}`;
      if (out.counts) {
        expect(out.counts[group], `${where} counts`).toBe(stats?.count ?? 0);
      }
      if (out.sums) {
        const sum = BigInt.asIntN(
          64,
          (BigInt(out.sums[2 * group + 1]) << 32n) | BigInt(out.sums[2 * group])
        );
        expect(sum, `${where} sums`).toBe(stats?.sum ?? 0n);
      }
      if (out.sumValues) {
        expectClose(
          out.sumValues[group],
          stats?.sumValue ?? 0,
          {relative: 1e-6, absolute: 1e-6},
          `${where} sumValues`
        );
      }
      const checks: [
        keyof GPUGroupStatisticsColumnOutput,
        number | undefined,
        'exact' | keyof typeof TOLERANCE
      ][] = [
        ['means', stats?.mean, 'mean'],
        ['minimums', stats?.minimum, 'exact'],
        ['maximums', stats?.maximum, 'exact'],
        ['variances', stats?.variance, 'variance'],
        [
          'standardDeviations',
          stats === undefined ? undefined : Math.sqrt(stats.variance),
          'variance'
        ],
        ['skewness', stats?.skewness, 'skewness'],
        ['kurtosis', stats?.kurtosis, 'kurtosis'],
        ['medians', stats?.median, 'exact'],
        ['modes', stats?.mode, 'exact']
      ];
      for (const [field, value, kind] of checks) {
        const values = out[field];
        if (!values) {
          continue;
        }
        const wantedValue = value ?? NaN;
        if (kind === 'exact') {
          expectExactFloat(values[group], wantedValue, `${where} ${field}`);
        } else if (field === 'standardDeviations') {
          // sd tolerance follows the variance tolerance in the square root domain.
          expectClose(
            values[group],
            wantedValue,
            {relative: TOLERANCE.variance.relative, absolute: 1e-3},
            `${where} ${field}`
          );
        } else {
          expectClose(values[group], wantedValue, TOLERANCE[kind], `${where} ${field}`);
        }
      }
      if (out.uniqueCounts) {
        expect(out.uniqueCounts[group], `${where} unique`).toBe(stats?.uniqueCount ?? 0);
      }
      if (out.percentiles) {
        for (let fraction = 0; fraction < fractionCount; fraction++) {
          expectExactFloat(
            out.percentiles[group * fractionCount + fraction],
            stats?.percentiles[fraction] ?? NaN,
            `${where} percentile ${scene.fractions[fraction]}`
          );
        }
      }
    }
    if (out.zScores) {
      for (let row = 0; row < scene.keys.length; row++) {
        const wantedValue = expected.zScores[columnIndex][row];
        expectClose(
          out.zScores[row],
          wantedValue,
          TOLERANCE.zScore,
          `${label} column ${columnIndex} z ${row}`
        );
      }
    }
    expect(wanted.size).toBeGreaterThan(0);
  }
  return expected;
}

/** Quantized value: a multiple of 1/64 so fixed-point sums are exact. */
function quantize(value: number): number {
  return Math.round(value * 64) / 64;
}

type SceneOptions = {
  seed: number;
  keyBits: 32 | 64;
  columns: GPUGroupStatistic[][];
  capacity?: number;
  mask?: boolean;
  variance?: 'sample' | 'population';
  fractions?: number[];
  bigGroupSize?: number;
  smallGroups?: number;
};

/**
 * Groups of size 1, 2, 3, ~300 and a big one, plus many small ones. Keys of the 64-bit scenes
 * differ in the high word only for some groups. Column 0 has quantized values with NaN, +-Infinity
 * and -0; column 1 has small integers (mode ties); column 2 has arbitrary f32 values.
 */
function createScene(options: SceneOptions): Scene {
  const random = createRandom(options.seed);
  const keys: bigint[] = [];
  const reserved = options.keyBits === 64 ? 0xffffffffffffffffn : 0xffffffffn;
  const groupKeys: bigint[] = [];
  const sizes = [1, 2, 3, 300, options.bigGroupSize ?? 5000];
  for (let group = 0; group < (options.smallGroups ?? 40); group++) {
    sizes.push(4 + Math.floor(random() * 40));
  }
  for (let group = 0; group < sizes.length; group++) {
    if (options.keyBits === 64) {
      // Some groups share the low word and differ in the high word only.
      const low = BigInt(group % 3 === 0 ? 7 : Math.floor(random() * 0xffffffff));
      groupKeys.push((BigInt(group * 5 + 1) << 32n) | low);
    } else {
      groupKeys.push(BigInt(group * 977 + 13 + (group === 1 ? 0xf0000000 : 0)));
    }
  }
  for (const [group, size] of sizes.entries()) {
    for (let row = 0; row < size; row++) {
      keys.push(groupKeys[group]);
    }
  }
  // Invalid keys.
  for (let index = 0; index < 25; index++) {
    keys.push(reserved);
  }
  // Fisher-Yates shuffle.
  for (let row = keys.length - 1; row > 0; row--) {
    const other = Math.floor(random() * (row + 1));
    [keys[row], keys[other]] = [keys[other], keys[row]];
  }
  const rows = keys.length;
  const columns: ColumnScene[] = options.columns.map((statistics, columnIndex) => {
    const values = new Float32Array(rows);
    for (let row = 0; row < rows; row++) {
      const roll = random();
      if (columnIndex === 1) {
        values[row] = roll < 0.03 ? NaN : 1 + Math.floor(random() * 5);
      } else if (columnIndex === 2) {
        values[row] = roll < 0.02 ? NaN : Math.fround((random() - 0.3) * 1000);
      } else {
        values[row] =
          roll < 0.03
            ? NaN
            : roll < 0.04
              ? Infinity
              : roll < 0.05
                ? -0
                : roll < 0.06
                  ? -Infinity
                  : quantize((random() - 0.4) * 600);
      }
    }
    return {values, statistics};
  });
  const mask = options.mask
    ? // The three smallest groups keep every row so sizes 1, 2 and 3 are exercised.
      Uint32Array.from({length: rows}, (_, row) =>
        groupKeys.slice(0, 3).includes(keys[row]) || random() >= 0.15 ? 1 : 0
      )
    : undefined;
  return {
    keyBits: options.keyBits,
    keys,
    mask,
    columns,
    variance: options.variance,
    fractions: options.fractions ?? [],
    capacity: options.capacity ?? sizes.length + 8
  };
}

it('GPUGroupStatistics computes every statistic for u32 keys, a mask and NaN values', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const harness = new Harness(device);
  const scene = createScene({
    seed: 11,
    keyBits: 32,
    mask: true,
    fractions: [0, 0.1, 0.5, 0.9, 1],
    columns: [ALL_STATISTICS]
  });
  const run = await createRun(harness, scene);
  const actual = await run.run();
  const expected = expectMatchesOracle(actual, scene, 'u32');
  // A few groups are tiny and the biggest holds more than 4096 rows.
  expect(Math.max(...expected.groups.map(group => group.count))).toBeGreaterThan(4096);
  expect(expected.groups.some(group => group.count === 1)).toBe(true);
  expect(expected.groups.some(group => group.count === 2)).toBe(true);
  expect(expected.groups.some(group => group.columns[0].count === 0)).toBe(false);
  run.destroy();
  harness.destroy();
});

it('GPUGroupStatistics matches the oracle for u64 keys that differ in the high word, 3 columns', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const harness = new Harness(device);
  const scene = createScene({
    seed: 23,
    keyBits: 64,
    mask: true,
    variance: 'population',
    fractions: [0.25, 0.5, 0.75],
    columns: [
      ['count', 'sum', 'mean', 'minimum', 'maximum', 'standardDeviation', 'median', 'zScore'],
      ['mode', 'uniqueCount', 'percentiles', 'variance', 'count'],
      ['kurtosis', 'skewness', 'sum', 'percentiles', 'zScore']
    ]
  });
  const run = await createRun(harness, scene);
  const actual = await run.run();
  const expected = expectMatchesOracle(actual, scene, 'u64');
  // Keys that share a low word and differ in the high word are separate groups.
  const lowWords = new Map<bigint, number>();
  for (const group of expected.groups) {
    lowWords.set(group.key & 0xffffffffn, (lowWords.get(group.key & 0xffffffffn) ?? 0) + 1);
  }
  expect(Math.max(...lowWords.values())).toBeGreaterThan(5);
  run.destroy();
  harness.destroy();
});

it('GPUGroupStatistics keeps the smallest keys on capacity overflow', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  for (const keyBits of [32, 64] as const) {
    const harness = new Harness(device);
    const scene = createScene({
      seed: 5 + keyBits,
      keyBits,
      bigGroupSize: 600,
      capacity: 13,
      fractions: [0.5],
      columns: [
        [
          'count',
          'mean',
          'minimum',
          'median',
          'percentiles',
          'mode',
          'uniqueCount',
          'variance',
          'zScore'
        ]
      ]
    });
    const run = await createRun(harness, scene);
    const actual = await run.run();
    const expected = expectMatchesOracle(actual, scene, `overflow ${keyBits}`);
    expect(actual.total).toBeGreaterThan(13);
    expect(actual.overflow).toBe(1);
    expect(actual.count).toBe(13);
    expect(expected.groups.length).toBe(13);
    // Rows of dropped groups have NaN z-scores.
    const zScores = actual.columns[0].zScores!;
    expect(zScores.some(Number.isNaN)).toBe(true);
    run.destroy();
    harness.destroy();
  }
});

it('GPUGroupStatistics percentile fractions, values and masks change without a rebuild', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const harness = new Harness(device);
  const scene = createScene({
    seed: 77,
    keyBits: 32,
    bigGroupSize: 800,
    smallGroups: 20,
    fractions: [0.1, 0.5, 0.9],
    columns: [['median', 'percentiles', 'count', 'mean', 'zScore']]
  });
  const run = await createRun(harness, scene);
  const compiled = run.compiled;
  const first = await run.run();
  expectMatchesOracle(first, scene, 'fractions A');
  // Rewrite only the percentile parameter view and re-encode the SAME compiled graph.
  const rewritten = {...scene, fractions: [0.25, 0, 1]};
  run.parameters!.write(Float32Array.from(rewritten.fractions));
  const second = await run.run();
  expectMatchesOracle(second, rewritten, 'fractions B');
  expect(run.compiled).toBe(compiled);
  expect(second.columns[0].percentiles).not.toEqual(first.columns[0].percentiles);
  // Out-of-range and non-finite fractions clamp / give NaN.
  const odd = {...scene, fractions: [-3, 7, NaN]};
  run.parameters!.write(Float32Array.from(odd.fractions));
  expectMatchesOracle(await run.run(), odd, 'fractions odd');
  // Rewrite values and the mask too.
  const random = createRandom(5);
  const values = Float32Array.from(scene.columns[0].values, () => quantize((random() - 0.5) * 80));
  const mask = Uint32Array.from(scene.keys, () => (random() < 0.3 ? 0 : 1));
  run.valueBuffers[0].write(values);
  run.maskBuffer.write(mask);
  const changed = {
    ...scene,
    fractions: [0.1, 0.5, 0.9],
    mask,
    columns: [{...scene.columns[0], values}]
  };
  run.parameters!.write(Float32Array.from(changed.fractions));
  expectMatchesOracle(await run.run(), changed, 'values and mask');
  expect(run.compiled).toBe(compiled);
  run.destroy();
  harness.destroy();
});

it('GPUGroupStatistics encodes bitwise identically twice and handles an all-masked input', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const harness = new Harness(device);
  const scene = createScene({
    seed: 91,
    keyBits: 64,
    bigGroupSize: 1500,
    smallGroups: 10,
    fractions: [0.3, 0.7],
    columns: [ALL_STATISTICS, ['mode', 'median', 'uniqueCount']]
  });
  const run = await createRun(harness, scene);
  const first = await run.run();
  run.poison();
  const second = await run.run();
  expect(second.words.length).toBe(first.words.length);
  expect(second.words.every((word, index) => word === first.words[index])).toBe(true);
  expectMatchesOracle(second, scene, 'determinism');
  // Everything masked off: no groups, an empty tail and NaN z-scores.
  run.maskBuffer.write(new Uint32Array(scene.keys.length));
  run.poison();
  const empty = await run.run();
  expect([empty.count, empty.total, empty.overflow]).toEqual([0, 0, 0]);
  expect(empty.columns[0].zScores!.every(Number.isNaN)).toBe(true);
  expect(empty.counts.every(count => count === 0)).toBe(true);
  expect(empty.columns[1].uniqueCounts!.every(count => count === 0)).toBe(true);
  expect(canonicalizeValue(-0)).toBe(0);
  run.destroy();
  harness.destroy();
});

it('GPUGroupStatistics handles every statistic alone and a recipe with no columns', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const harness = new Harness(device);
  const sets: GPUGroupStatistic[][] = [
    ...ALL_STATISTICS.map(statistic => [statistic]),
    ['mode', 'uniqueCount'],
    ['minimum', 'maximum'],
    ['median', 'mode']
  ];
  for (const statistics of sets) {
    const scene = createScene({
      seed: 3,
      keyBits: 32,
      bigGroupSize: 300,
      smallGroups: 8,
      mask: true,
      fractions: statistics.includes('percentiles') ? [0.2, 0.8] : [],
      columns: [statistics]
    });
    const run = await createRun(harness, scene);
    expectMatchesOracle(await run.run(), scene, statistics.join('+'));
    run.destroy();
  }
  // No value columns: only keys, counts, count, overflow and total.
  const scene = createScene({seed: 4, keyBits: 64, bigGroupSize: 100, smallGroups: 5, columns: []});
  const run = await createRun(harness, scene);
  expectMatchesOracle(await run.run(), scene, 'no columns');
  run.destroy();
  harness.destroy();
});

it('GPUGroupStatistics treats a large constant group as zero variance despite mean rounding', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const harness = new Harness(device);
  const rows = 6000;
  const keys = Array.from({length: rows}, (_, row) => (row < 5000 ? 9n : BigInt(row % 3)));
  const values = Float32Array.from({length: rows}, (_, row) =>
    row < 5000 ? Math.fround(100.1) : Math.fround(row * 0.37)
  );
  const scene: Scene = {
    keyBits: 32,
    keys,
    columns: [
      {
        values,
        statistics: ['mean', 'variance', 'standardDeviation', 'skewness', 'kurtosis', 'zScore']
      }
    ],
    fractions: [],
    capacity: 8
  };
  const run = await createRun(harness, scene);
  const actual = await run.run();
  expectMatchesOracle(actual, scene, 'constant group');
  const group = actual.keys.indexOf(9n);
  expect(actual.columns[0].skewness![group]).toBeNaN();
  expect(actual.columns[0].kurtosis![group]).toBeNaN();
  expect(actual.columns[0].zScores!.slice(0, 5000).every(z => z === 0)).toBe(true);
  run.destroy();
  harness.destroy();
});
