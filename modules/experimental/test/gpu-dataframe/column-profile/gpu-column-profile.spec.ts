// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPUColumnProfileParameterLength,
  getGPUColumnProfileParameterValues,
  GPU_COLUMN_PROFILE_STATISTIC,
  GPU_COLUMN_PROFILE_STATISTIC_COUNT,
  GPUColumnProfile,
  type GPUColumnProfileColumn
} from '../../../src/gpu-dataframe/column-profile';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  computeColumnProfileOracle,
  type ColumnProfileOracleColumn,
  type ColumnProfileOracleResult
} from './column-profile-oracle';

type Domains = (readonly [number, number] | undefined)[];

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

function createMask(seed: number, rows: number, keepProbability: number): Uint32Array {
  const random = createRandom(seed);
  return Uint32Array.from({length: rows}, () => (random() < keepProbability ? 1 : 0));
}

type NumericKind = 'special' | 'offset' | 'ties' | 'wide' | 'unique';

function createNumeric(kind: NumericKind, seed: number, rows: number): Float32Array {
  const random = createRandom(seed);
  const values = new Float32Array(rows);
  const specials = [NaN, Infinity, -Infinity, 0, -0, NaN, 7, -7];
  for (let row = 0; row < rows; row++) {
    switch (kind) {
      case 'special':
        values[row] =
          random() < 0.15 ? specials[Math.floor(random() * specials.length)] : random() * 20 - 10;
        break;
      case 'offset':
        values[row] = 1e6 + (random() - 0.5) * 200;
        break;
      case 'ties':
        values[row] = Math.round((random() - 0.5) * 60) / 4;
        break;
      case 'wide':
        values[row] = (random() - 0.4) * 10 ** Math.floor(random() * 8 - 3);
        break;
      case 'unique':
        values[row] = (row * 7919) % 100003;
        break;
    }
  }
  return values;
}

function createCategory(
  seed: number,
  rows: number,
  categoryCount: number,
  extras: {nullProbability: number; overflowProbability: number}
): Uint32Array {
  const random = createRandom(seed);
  return Uint32Array.from({length: rows}, () => {
    const roll = random();
    if (roll < extras.nullProbability) {
      return 0xffffffff;
    }
    if (roll < extras.nullProbability + extras.overflowProbability) {
      return categoryCount + Math.floor(random() * 5);
    }
    // Skewed so that the top list has clear and tied entries.
    return Math.floor(random() ** 2 * categoryCount);
  });
}

type Scene = {
  columns: ColumnProfileOracleColumn[];
  mask?: Uint32Array;
  histogramBinCount: number;
  precision: number;
  topCategoryCount: number;
};

type Result = {
  statistics: Float32Array;
  counts: Uint32Array;
  histograms: Uint32Array;
  registers: Uint32Array;
  topCategories: Uint32Array;
  topCategoryCounts: Uint32Array;
};

type Fixture = {
  readonly counters: {getCommandNodes: number; compile: number};
  readonly maskBuffer?: Buffer;
  readonly columnBuffers: Buffer[];
  encode(domains?: Domains): void;
  /** Resolves when all submitted frames finished (maps the small statistics output). */
  waitForGPU(): Promise<void>;
  read(): Promise<Result>;
  run(domains?: Domains): Promise<Result>;
  destroy(): void;
};

function createFixture(device: Device, scene: Scene): Fixture {
  const {columns, histogramBinCount, precision, topCategoryCount} = scene;
  const columnCount = columns.length;
  const rows = columns[0].values.length;
  const registerCount = 2 ** precision;
  const graph = new GPUCommandGraph(device, {id: 'column-profile-graph'});
  const counters = {getCommandNodes: 0, compile: 0};
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const columnBuffers = columns.map(column => track(createInputBuffer(device, column.values)));
  const maskBuffer = scene.mask ? track(createInputBuffer(device, scene.mask)) : undefined;
  const outputs = {
    statistics: track(createOutputBuffer(device, columnCount * GPU_COLUMN_PROFILE_STATISTIC_COUNT)),
    counts: track(createOutputBuffer(device, columnCount * 2)),
    histograms: track(createOutputBuffer(device, columnCount * histogramBinCount)),
    registers: track(createOutputBuffer(device, columnCount * registerCount)),
    topCategories: track(createOutputBuffer(device, columnCount * topCategoryCount)),
    topCategoryCounts: track(createOutputBuffer(device, columnCount * topCategoryCount))
  };
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'profile-parameters',
    format: 'float32',
    length: getGPUColumnProfileParameterLength(columnCount)
  });
  const contributor = new GPUColumnProfile({
    id: 'profile',
    columns: columns.map((column, columnIndex): GPUColumnProfileColumn => {
      const buffer = columnBuffers[columnIndex];
      return column.kind === 'numeric'
        ? {values: importGraphBuffer(graph, `column-${columnIndex}`, buffer, 'float32', rows)}
        : {
            values: importGraphBuffer(graph, `column-${columnIndex}`, buffer, 'uint32', rows),
            kind: 'category',
            categoryCount: column.categoryCount
          };
    }),
    mask: maskBuffer ? importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', rows) : undefined,
    parameters: parameterBuffer.importToGraph(graph),
    histogramBinCount,
    hyperLogLogPrecision: precision,
    topCategoryCount,
    output: {
      statistics: importGraphBuffer(
        graph,
        'o-statistics',
        outputs.statistics,
        'float32',
        columnCount * GPU_COLUMN_PROFILE_STATISTIC_COUNT
      ),
      counts: importGraphBuffer(graph, 'o-counts', outputs.counts, 'uint32', columnCount * 2),
      histograms: importGraphBuffer(
        graph,
        'o-histograms',
        outputs.histograms,
        'uint32',
        columnCount * histogramBinCount
      ),
      hyperLogLogRegisters: importGraphBuffer(
        graph,
        'o-registers',
        outputs.registers,
        'uint32',
        columnCount * registerCount
      ),
      topCategories: importGraphBuffer(
        graph,
        'o-top',
        outputs.topCategories,
        'uint32',
        columnCount * topCategoryCount
      ),
      topCategoryCounts: importGraphBuffer(
        graph,
        'o-top-counts',
        outputs.topCategoryCounts,
        'uint32',
        columnCount * topCategoryCount
      )
    }
  });
  const originalGetCommandNodes = contributor.getCommandNodes.bind(contributor);
  contributor.getCommandNodes = graphArgument => {
    counters.getCommandNodes++;
    return originalGetCommandNodes(graphArgument);
  };
  graph.add(contributor);
  counters.compile++;
  const compiled = graph.compile();
  const encode = (domains: Domains = []) => {
    parameterBuffer.write(getGPUColumnProfileParameterValues(domains, columnCount));
    submitGraph(device, compiled, undefined);
  };
  const read = async (): Promise<Result> => ({
    statistics: Float32Array.from(
      await readFloat32(outputs.statistics, columnCount * GPU_COLUMN_PROFILE_STATISTIC_COUNT)
    ),
    counts: Uint32Array.from(await readUint32(outputs.counts, columnCount * 2)),
    histograms: Uint32Array.from(
      await readUint32(outputs.histograms, columnCount * histogramBinCount)
    ),
    registers: Uint32Array.from(await readUint32(outputs.registers, columnCount * registerCount)),
    topCategories: Uint32Array.from(
      await readUint32(outputs.topCategories, columnCount * topCategoryCount)
    ),
    topCategoryCounts: Uint32Array.from(
      await readUint32(outputs.topCategoryCounts, columnCount * topCategoryCount)
    )
  });
  return {
    counters,
    maskBuffer,
    columnBuffers,
    encode,
    read,
    async waitForGPU() {
      await readFloat32(outputs.statistics, 1);
    },
    async run(domains) {
      encode(domains);
      return read();
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

const fround = Math.fround;

function expectClose(
  actual: number,
  expected: number,
  relativeTolerance: number,
  scale: number,
  label: string
): void {
  if (Number.isNaN(expected)) {
    expect(Number.isNaN(actual), label).toBe(true);
    return;
  }
  expect(Math.abs(actual - expected), `${label}: ${actual} vs ${expected}`).toBeLessThanOrEqual(
    relativeTolerance * Math.max(Math.abs(expected), scale) + 1e-30
  );
}

/** Integer outputs bitwise, float moments within tolerance. */
function expectParity(
  actual: Result,
  scene: Scene,
  domains: Domains | undefined,
  tolerance = {mean: 2e-6, variance: 1e-3, sum: 1e-5}
): ColumnProfileOracleResult[] {
  const expected = computeColumnProfileOracle({...scene, domains});
  const {histogramBinCount, topCategoryCount, precision} = scene;
  const registerCount = 2 ** precision;
  const stat = GPU_COLUMN_PROFILE_STATISTIC;
  for (const [column, oracle] of expected.entries()) {
    const label = `column ${column}`;
    const base = column * GPU_COLUMN_PROFILE_STATISTIC_COUNT;
    const statistics = (field: number) => actual.statistics[base + field];
    expect(statistics(stat.count), `${label} count`).toBe(oracle.count);
    expect(statistics(stat.nullCount), `${label} nullCount`).toBe(oracle.nullCount);
    expect(actual.counts[column * 2], `${label} counts[0]`).toBe(oracle.count);
    expect(actual.counts[column * 2 + 1], `${label} counts[1]`).toBe(oracle.nullCount);
    expect(statistics(stat.overflowCount), `${label} overflow`).toBe(oracle.overflowCount);
    for (const [name, field] of [
      ['minimum', stat.minimum],
      ['maximum', stat.maximum]
    ] as const) {
      const expectedValue = oracle[name];
      if (Number.isNaN(expectedValue)) {
        expect(Number.isNaN(statistics(field)), `${label} ${name}`).toBe(true);
      } else {
        expect(Object.is(statistics(field), fround(expectedValue)), `${label} ${name}`).toBe(true);
      }
    }
    const spread = Math.sqrt(Math.max(oracle.variance, 0)) || 1;
    expectClose(
      statistics(stat.sum),
      oracle.sum,
      tolerance.sum,
      spread * oracle.count * 0.1,
      `${label} sum`
    );
    // The mean is accurate relative to the spread plus a few f32 ULP of its magnitude.
    expectClose(statistics(stat.mean), oracle.mean, tolerance.mean, spread, `${label} mean`);
    expectClose(
      statistics(stat.variance),
      oracle.variance,
      tolerance.variance,
      0,
      `${label} variance`
    );
    expectClose(
      statistics(stat.sampleVariance),
      oracle.sampleVariance,
      tolerance.variance,
      0,
      `${label} sampleVariance`
    );
    expectClose(
      statistics(stat.standardDeviation),
      oracle.standardDeviation,
      tolerance.variance,
      0,
      `${label} standardDeviation`
    );
    expect(
      Array.from(actual.registers.subarray(column * registerCount, (column + 1) * registerCount)),
      `${label} registers`
    ).toEqual(Array.from(oracle.registers));
    expectClose(
      statistics(stat.distinctEstimate),
      fround(oracle.distinctEstimate),
      1e-5,
      1,
      `${label} distinct`
    );
    expect(
      Array.from(
        actual.histograms.subarray(column * histogramBinCount, (column + 1) * histogramBinCount)
      ),
      `${label} histogram`
    ).toEqual(Array.from(oracle.histogram));
    expect(
      Array.from(
        actual.topCategories.subarray(column * topCategoryCount, (column + 1) * topCategoryCount)
      ),
      `${label} topCategories`
    ).toEqual(oracle.topCategories);
    expect(
      Array.from(
        actual.topCategoryCounts.subarray(
          column * topCategoryCount,
          (column + 1) * topCategoryCount
        )
      ),
      `${label} topCategoryCounts`
    ).toEqual(oracle.topCategoryCounts);
  }
  return expected;
}

function createMixedScene(rows: number, seed: number): Scene {
  return {
    columns: [
      {kind: 'numeric', values: createNumeric('special', seed, rows)},
      {kind: 'numeric', values: createNumeric('offset', seed + 1, rows)},
      {
        kind: 'category',
        values: createCategory(seed + 2, rows, 20, {
          nullProbability: 0.05,
          overflowProbability: 0.03
        }),
        categoryCount: 20
      },
      {kind: 'numeric', values: createNumeric('ties', seed + 3, rows)},
      {kind: 'numeric', values: createNumeric('wide', seed + 4, rows)}
    ],
    mask: createMask(seed + 5, rows, 0.7),
    histogramBinCount: 16,
    precision: 8,
    topCategoryCount: 5
  };
}

it('GPUColumnProfile matches the CPU oracle on mixed columns with NaN, Infinity and masks', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  for (const rows of [1, 7, 2048, 2049, 20000]) {
    const scene = createMixedScene(rows, 100 + rows);
    const fixture = createFixture(device, scene);
    expectParity(await fixture.run(), scene, undefined);
    fixture.destroy();
  }
});

it('GPUColumnProfile is stable on a large offset with a small spread', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const rows = 100000;
  const scene: Scene = {
    columns: [{kind: 'numeric', values: createNumeric('offset', 5, rows)}],
    histogramBinCount: 10,
    precision: 10,
    topCategoryCount: 1
  };
  const fixture = createFixture(device, scene);
  const result = await fixture.run();
  const [oracle] = expectParity(result, scene, undefined, {mean: 1e-7, variance: 5e-4, sum: 1e-6});
  // A naive f32 sum of squares would be off by orders of magnitude; the pairwise merge is not.
  expect(oracle.variance).toBeGreaterThan(3000);
  const variance = result.statistics[GPU_COLUMN_PROFILE_STATISTIC.variance];
  expect(Math.abs(variance - oracle.variance) / oracle.variance).toBeLessThan(5e-4);
  fixture.destroy();
});

it('GPUColumnProfile handles all-masked columns', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createMixedScene(3000, 31);
  scene.mask = new Uint32Array(3000);
  const fixture = createFixture(device, scene);
  const result = await fixture.run();
  expectParity(result, scene, undefined);
  const stat = GPU_COLUMN_PROFILE_STATISTIC;
  for (let column = 0; column < scene.columns.length; column++) {
    const base = column * GPU_COLUMN_PROFILE_STATISTIC_COUNT;
    expect(result.statistics[base + stat.count]).toBe(0);
    expect(result.statistics[base + stat.nullCount]).toBe(0);
    for (const field of [stat.minimum, stat.maximum, stat.mean, stat.variance, stat.sum]) {
      expect(Number.isNaN(result.statistics[base + field])).toBe(true);
    }
    expect(result.statistics[base + stat.distinctEstimate]).toBe(0);
  }
  expect(result.histograms.every(count => count === 0)).toBe(true);
  expect(result.topCategoryCounts.every(count => count === 0)).toBe(true);
  expect(result.topCategories.every(code => code === 0xffffffff)).toBe(true);
  fixture.destroy();
});

it('GPUColumnProfile excludes Infinity from moments and keeps it in extremes and counts', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const values = Float32Array.from([1, 2, Infinity, 3, -Infinity, NaN, -0, 0]);
  const scene: Scene = {
    columns: [
      {kind: 'numeric', values},
      {kind: 'numeric', values: Float32Array.from([Infinity, Infinity, 5, 5, 5, 5, 5, 5])}
    ],
    histogramBinCount: 4,
    precision: 6,
    topCategoryCount: 1
  };
  const fixture = createFixture(device, scene);
  const result = await fixture.run();
  expectParity(result, scene, undefined);
  const stat = GPU_COLUMN_PROFILE_STATISTIC;
  expect(result.statistics[stat.count]).toBe(7);
  expect(result.statistics[stat.nullCount]).toBe(1);
  expect(result.statistics[stat.minimum]).toBe(-Infinity);
  expect(result.statistics[stat.maximum]).toBe(Infinity);
  expect(result.statistics[stat.sum]).toBe(6);
  // The second column has an all-equal finite range: lo == hi counts into bin 0.
  const second = GPU_COLUMN_PROFILE_STATISTIC_COUNT;
  expect(result.statistics[second + stat.count]).toBe(8);
  expect(result.statistics[second + stat.mean]).toBe(5);
  expect(Array.from(result.histograms.subarray(4, 8))).toEqual([6, 0, 0, 0]);
  fixture.destroy();
});

it('GPUColumnProfile supports automatic and fixed histogram domains', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createMixedScene(15000, 211);
  const fixture = createFixture(device, scene);
  const domainSets: Domains[] = [
    [],
    [[-5, 5], [999900, 1000100], undefined, [-3.3, 3.3], [-1, 1]],
    [[NaN, 2], [999950, NaN], undefined, [0.25, 0.25], [5, -5]],
    [[-100, -50], [0, 1], undefined, [-1e10, 1e10], [NaN, NaN]],
    [[-Infinity, 1], [NaN, Infinity], undefined, [1e-30, 2e-30], [-1e38, 1e38]]
  ];
  for (const domains of domainSets) {
    expectParity(await fixture.run(domains), scene, domains);
  }
  fixture.destroy();
});

it('GPUColumnProfile runs several frames on one compiled graph without rebuilding', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const rows = 12000;
  const scene = createMixedScene(rows, 311);
  const fixture = createFixture(device, scene);
  for (let frame = 0; frame < 5; frame++) {
    if (frame > 0) {
      // Per-frame inputs: mask contents, then column contents, then domains.
      scene.mask = createMask(400 + frame, rows, 0.2 + 0.2 * frame);
      fixture.maskBuffer!.write(scene.mask);
    }
    if (frame === 3) {
      scene.columns[0] = {kind: 'numeric', values: createNumeric('special', 999, rows)};
      fixture.columnBuffers[0].write((scene.columns[0] as {values: Float32Array}).values);
    }
    const domains: Domains =
      frame % 2 === 0 ? [] : [[-2 * frame, 2 * frame], undefined, undefined, [-frame, frame]];
    expectParity(await fixture.run(domains), scene, domains);
  }
  expect(fixture.counters).toEqual({getCommandNodes: 1, compile: 1});
  fixture.destroy();
});

it('GPUColumnProfile is bitwise identical across repeated runs', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createMixedScene(60000, 511);
  const fixture = createFixture(device, scene);
  const first = await fixture.run();
  for (let repeat = 0; repeat < 3; repeat++) {
    const next = await fixture.run();
    for (const name of Object.keys(first) as (keyof Result)[]) {
      const bits = (values: Float32Array | Uint32Array) =>
        Array.from(new Uint32Array(values.buffer, values.byteOffset, values.length));
      expect(bits(next[name]), name).toEqual(bits(first[name]));
    }
  }
  fixture.destroy();
  // A second graph over the same data reproduces the first one's bits too.
  const again = createFixture(device, scene);
  const other = await again.run();
  expect(Array.from(new Uint32Array(other.statistics.buffer))).toEqual(
    Array.from(new Uint32Array(first.statistics.buffer))
  );
  again.destroy();
});

it('GPUColumnProfile HyperLogLog registers match bitwise and the estimate is accurate', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const rows = 200000;
  for (const precision of [4, 12, 14]) {
    const scene: Scene = {
      columns: [
        {kind: 'numeric', values: createNumeric('unique', 3, rows)},
        {
          kind: 'category',
          values: Uint32Array.from({length: rows}, (_, row) => (row * 2654435761) % 50021),
          categoryCount: 50021
        },
        {kind: 'numeric', values: createNumeric('ties', 4, rows)}
      ],
      histogramBinCount: 4,
      precision,
      topCategoryCount: 3
    };
    const fixture = createFixture(device, scene);
    const result = await fixture.run();
    const expected = expectParity(result, scene, undefined);
    const standardError = 1.04 / Math.sqrt(2 ** precision);
    for (const [column, oracle] of expected.entries()) {
      const estimate = result.statistics[column * GPU_COLUMN_PROFILE_STATISTIC_COUNT + 9];
      const tolerance = Math.max(4 * standardError, 0.02) * oracle.trueDistinct;
      expect(
        Math.abs(estimate - oracle.trueDistinct),
        `p=${precision} column ${column}`
      ).toBeLessThan(tolerance);
    }
    fixture.destroy();
  }
});

it('GPUColumnProfile ranks top categories with ties and large dictionaries', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const rows = 30000;
  const categoryCount = 100000;
  const random = createRandom(77);
  // Counts per code form many exact ties.
  const codes = Uint32Array.from({length: rows}, () =>
    random() < 0.02 ? 0xffffffff : (Math.floor(random() * 3000) * 31) % categoryCount
  );
  const scene: Scene = {
    columns: [
      {kind: 'category', values: codes, categoryCount},
      {
        kind: 'category',
        values: Uint32Array.from({length: rows}, (_, row) => row % 3),
        categoryCount: 3
      }
    ],
    mask: createMask(78, rows, 0.9),
    histogramBinCount: 12,
    precision: 9,
    topCategoryCount: 64
  };
  const fixture = createFixture(device, scene);
  const result = await fixture.run();
  expectParity(result, scene, undefined);
  fixture.destroy();
});

it('GPUColumnProfile times one million rows by four columns', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const rows = 1_000_000;
  const scene: Scene = {
    columns: [
      {kind: 'numeric', values: createNumeric('wide', 1, rows)},
      {kind: 'numeric', values: createNumeric('offset', 2, rows)},
      {kind: 'numeric', values: createNumeric('special', 3, rows)},
      {
        kind: 'category',
        values: createCategory(4, rows, 500, {nullProbability: 0.01, overflowProbability: 0}),
        categoryCount: 500
      }
    ],
    mask: createMask(5, rows, 0.8),
    histogramBinCount: 64,
    precision: 12,
    topCategoryCount: 10
  };
  const fixture = createFixture(device, scene);
  const result = await fixture.run();
  const frameMilliseconds: number[] = [];
  for (let frame = 0; frame < 5; frame++) {
    const start = performance.now();
    fixture.encode();
    await fixture.waitForGPU();
    frameMilliseconds.push(performance.now() - start);
  }
  await fixture.read();
  // eslint-disable-next-line no-console
  console.log(
    `GPUColumnProfile 1M rows x 4 columns: ${frameMilliseconds.map(value => value.toFixed(1)).join(', ')} ms per encode+sync`
  );
  expectParity(result, scene, undefined);
  fixture.destroy();
});
