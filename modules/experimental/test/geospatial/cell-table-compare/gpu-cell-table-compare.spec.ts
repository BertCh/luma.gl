// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer, submitGraph} from '../../../src/utils/gpu-contributor-utils';
import {GPUCellAggregation} from '../../../src/geospatial/cell-aggregation';
import type {GPUCellTable} from '../../../src/geospatial/cell-aggregation/cell-table';
import {GPUCellTableCompare} from '../../../src/geospatial/cell-table-compare';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
} from '../../utils/gpu-contributor-test-utils';
import {
  aggregateCellsOnCPU,
  getQuadbinPointKeys,
  joinCellKey,
  splitCellKey
} from '../cell-aggregation/cell-aggregation-oracle';
import {createPointPositions, createRandom} from '../cell-aggregation/cell-aggregation-points';
import {
  compareCellTablesOnCPU,
  type CompareInputCell,
  type CompareResult
} from './cell-table-compare-oracle';

const SUM_SCALE = 65536;
const EMPTY_KEY = 0xffffffff;

/**
 * Tolerances. `delta`, keys, presence, before and after are bit exact. Ratio, percent change and
 * the Poisson z-score divide in f32 (WGSL allows a few ULP of error), so they use 1e-5 relative.
 * The standardized z-score also depends on an f32 reduction whose multiply-adds a compiler may
 * fuse, so it uses 1e-3 relative.
 */
const DIVISION_TOLERANCE = 1e-5;
const STANDARDIZED_TOLERANCE = 1e-3;

type Measure = 'count' | 'sum';
type ZScore = 'poisson' | 'standardized';

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

/** Input table buffers of a fixed capacity; tail rows hold garbage that `count` must hide. */
type InputTable = {
  capacity: number;
  cells: Buffer;
  counts: Buffer;
  sums: Buffer;
  count: Buffer;
  overflow: Buffer;
};

function createInputTable(fixture: Fixture, capacity: number): InputTable {
  return {
    capacity,
    cells: fixture.input(new Uint32Array(2 * capacity)),
    counts: fixture.input(new Uint32Array(capacity)),
    sums: fixture.input(new Uint32Array(2 * capacity)),
    count: fixture.input(new Uint32Array(1)),
    overflow: fixture.input(new Uint32Array(1))
  };
}

function writeInputTable(table: InputTable, rows: readonly CompareInputCell[], overflow = 0): void {
  const cells = new Uint32Array(2 * table.capacity);
  const counts = new Uint32Array(table.capacity);
  const sums = new Uint32Array(2 * table.capacity);
  for (let row = 0; row < table.capacity; row++) {
    const entry = rows[row];
    // Past `count`, a decoy key and values that must never be read.
    const key = entry ? entry.key : BigInt(row + 1);
    const [low, high] = splitCellKey(key);
    cells[2 * row] = low;
    cells[2 * row + 1] = high;
    counts[row] = entry ? entry.count : 777;
    const [sumLow, sumHigh] = splitCellKey(BigInt.asUintN(64, entry ? entry.sum : 99n));
    sums[2 * row] = sumLow;
    sums[2 * row + 1] = sumHigh;
  }
  table.cells.write(cells);
  table.counts.write(counts);
  table.sums.write(sums);
  table.count.write(new Uint32Array([rows.length]));
  table.overflow.write(new Uint32Array([overflow]));
}

function importInputTable(graph: GPUCommandGraph, name: string, table: InputTable): GPUCellTable {
  const {capacity} = table;
  return {
    cells: importGraphBuffer(graph, `${name}-cells`, table.cells, 'uint32x2', capacity),
    counts: importGraphBuffer(graph, `${name}-counts`, table.counts, 'uint32', capacity),
    sums: importGraphBuffer(graph, `${name}-sums`, table.sums, 'uint32x2', capacity),
    count: importGraphBuffer(graph, `${name}-count`, table.count, 'uint32', 1),
    overflow: importGraphBuffer(graph, `${name}-overflow`, table.overflow, 'uint32', 1)
  };
}

type CompareBuffers = Record<
  | 'cells'
  | 'presence'
  | 'before'
  | 'after'
  | 'delta'
  | 'ratio'
  | 'percentChange'
  | 'zScore'
  | 'count'
  | 'overflow'
  | 'total',
  Buffer
>;

function createCompareBuffers(fixture: Fixture, capacity: number): CompareBuffers {
  return {
    cells: fixture.output(2 * capacity),
    presence: fixture.output(capacity),
    before: fixture.output(capacity),
    after: fixture.output(capacity),
    delta: fixture.output(capacity),
    ratio: fixture.output(capacity),
    percentChange: fixture.output(capacity),
    zScore: fixture.output(capacity),
    count: fixture.output(1),
    overflow: fixture.output(1),
    total: fixture.output(1)
  };
}

function importCompareOutput(graph: GPUCommandGraph, buffers: CompareBuffers, capacity: number) {
  const view = <Format extends 'uint32' | 'float32'>(
    name: keyof CompareBuffers,
    format: Format,
    length = capacity
  ) => importGraphBuffer(graph, `out-${name}`, buffers[name], format, length);
  return {
    cells: importGraphBuffer(graph, 'out-cells', buffers.cells, 'uint32x2', capacity),
    presence: view('presence', 'uint32'),
    before: view('before', 'float32'),
    after: view('after', 'float32'),
    delta: view('delta', 'float32'),
    ratio: view('ratio', 'float32'),
    percentChange: view('percentChange', 'float32'),
    zScore: view('zScore', 'float32'),
    count: view('count', 'uint32', 1),
    overflow: view('overflow', 'uint32', 1),
    totalCount: view('total', 'uint32', 1)
  };
}

type ActualCompare = {
  count: number;
  overflow: number;
  total: number;
  keys: bigint[];
  columns: Record<
    'presence' | 'before' | 'after' | 'delta' | 'ratio' | 'percentChange' | 'zScore',
    number[]
  >;
  tailIsEmpty: boolean;
};

async function readCompare(buffers: CompareBuffers, capacity: number): Promise<ActualCompare> {
  const [count] = await readUint32(buffers.count, 1);
  const [overflow] = await readUint32(buffers.overflow, 1);
  const [total] = await readUint32(buffers.total, 1);
  const cells = await readUint32(buffers.cells, 2 * capacity);
  const presence = await readUint32(buffers.presence, capacity);
  const floats = async (
    name: 'before' | 'after' | 'delta' | 'ratio' | 'percentChange' | 'zScore'
  ) => readFloat32(buffers[name], capacity);
  const columns = {
    presence,
    before: await floats('before'),
    after: await floats('after'),
    delta: await floats('delta'),
    ratio: await floats('ratio'),
    percentChange: await floats('percentChange'),
    zScore: await floats('zScore')
  };
  let tailIsEmpty = true;
  for (let row = count; row < capacity; row++) {
    tailIsEmpty &&=
      cells[2 * row] === EMPTY_KEY &&
      cells[2 * row + 1] === EMPTY_KEY &&
      presence[row] === 0 &&
      columns.before[row] === 0 &&
      columns.after[row] === 0 &&
      Number.isNaN(columns.delta[row]) &&
      Number.isNaN(columns.ratio[row]) &&
      Number.isNaN(columns.percentChange[row]) &&
      Number.isNaN(columns.zScore[row]);
  }
  return {
    count,
    overflow,
    total,
    keys: Array.from({length: count}, (_, row) => joinCellKey(cells[2 * row], cells[2 * row + 1])),
    columns,
    tailIsEmpty
  };
}

function expectClose(actual: number, expected: number, tolerance: number, label: string): void {
  if (Number.isNaN(expected)) {
    expect(actual, label).toBeNaN();
    return;
  }
  expect(Math.abs(actual - expected), `${label}: ${actual} vs ${expected}`).toBeLessThanOrEqual(
    tolerance * Math.abs(expected) + 1e-30
  );
}

function expectMatchesOracle(
  actual: ActualCompare,
  expected: CompareResult,
  zScore: ZScore,
  label: string
): void {
  expect(actual.total, `${label} total`).toBe(expected.total);
  expect(actual.count, `${label} count`).toBe(expected.rows.length);
  expect(actual.overflow, `${label} overflow`).toBe(expected.overflow ? 1 : 0);
  expect(actual.tailIsEmpty, `${label} empty tail`).toBe(true);
  expect(actual.keys.map(String), `${label} keys`).toEqual(
    expected.rows.map(row => String(row.key))
  );
  const {columns} = actual;
  const zTolerance = zScore === 'poisson' ? DIVISION_TOLERANCE : STANDARDIZED_TOLERANCE;
  for (const [row, entry] of expected.rows.entries()) {
    expect(columns.presence[row], `${label} presence ${row}`).toBe(entry.presence);
    // Object.is also checks the sign of zero.
    for (const name of ['before', 'after', 'delta'] as const) {
      expect(Object.is(columns[name][row], entry[name]), `${label} ${name} ${row}`).toBe(true);
    }
    expectClose(columns.ratio[row], entry.ratio, DIVISION_TOLERANCE, `${label} ratio ${row}`);
    expectClose(
      columns.percentChange[row],
      entry.percentChange,
      DIVISION_TOLERANCE,
      `${label} percent ${row}`
    );
    expectClose(columns.zScore[row], entry.zScore, zTolerance, `${label} z ${row}`);
  }
}

type CompareFixture = {
  before: InputTable;
  after: InputTable;
  buffers: CompareBuffers;
  compiled: ReturnType<GPUCommandGraph['compile']>;
  graph: GPUCommandGraph;
  run(
    before: readonly CompareInputCell[],
    after: readonly CompareInputCell[],
    overflow?: [number, number]
  ): Promise<ActualCompare>;
};

function createCompareFixture(
  fixture: Fixture,
  props: {
    beforeCapacity: number;
    afterCapacity: number;
    capacity: number;
    measure: Measure;
    zScore: ZScore;
  }
): CompareFixture {
  const graph = new GPUCommandGraph(fixture.device, {
    id: 'cell-table-compare-graph'
  });
  const before = createInputTable(fixture, props.beforeCapacity);
  const after = createInputTable(fixture, props.afterCapacity);
  const buffers = createCompareBuffers(fixture, props.capacity);
  graph.add(
    new GPUCellTableCompare({
      before: importInputTable(graph, 'before', before),
      after: importInputTable(graph, 'after', after),
      measure: props.measure,
      sumScale: SUM_SCALE,
      zScore: props.zScore,
      output: importCompareOutput(graph, buffers, props.capacity)
    })
  );
  const compiled = graph.compile();
  return {
    before,
    after,
    buffers,
    compiled,
    graph,
    async run(beforeRows, afterRows, overflow = [0, 0]) {
      writeInputTable(before, beforeRows, overflow[0]);
      writeInputTable(after, afterRows, overflow[1]);
      submitGraph(fixture.device, compiled, undefined);
      return readCompare(buffers, props.capacity);
    }
  };
}

const cell = (key: bigint, count: number, sum = 0n): CompareInputCell => ({
  key,
  count,
  sum
});

/** Ascending distinct random cells; keys span both words, some differ only in the high word. */
function createRandomTable(seed: number, rows: number, universe: number): CompareInputCell[] {
  const random = createRandom(seed);
  const keys = new Set<bigint>();
  while (keys.size < rows) {
    const slot = BigInt(Math.floor(random() * universe));
    // Every third slot shares its low word with another slot and differs in the high word only.
    const key = slot % 3n === 0n ? slot << 32n : (slot << 8n) | 0xa000000000000000n;
    keys.add(key);
  }
  return [...keys]
    .sort((left, right) => (left < right ? -1 : 1))
    .map(key => {
      const magnitude = BigInt(Math.floor(random() * 1e6));
      // Sums span signs and exceed 2^32 so both words and the sign matter.
      const sum = (random() < 0.5 ? -1n : 1n) * (magnitude << BigInt(Math.floor(random() * 30)));
      return cell(key, 1 + Math.floor(random() * 500), sum);
    });
}

type Scenario = {
  name: string;
  before: CompareInputCell[];
  after: CompareInputCell[];
};

function getScenarios(): Scenario[] {
  const interleavedBefore = [1n, 3n, 5n, 7n, 9n].map((key, i) =>
    cell(key, i + 1, BigInt(i * 1000 - 2000))
  );
  const interleavedAfter = [2n, 3n, 6n, 7n, 10n, 11n].map((key, i) =>
    cell(key, 10 - i, BigInt(i * 70000))
  );
  return [
    {
      name: 'disjoint',
      before: [cell(1n, 3), cell(2n, 4)],
      after: [cell(10n, 1), cell(11n, 9)]
    },
    {
      name: 'disjoint reversed',
      before: [cell(10n, 1), cell(11n, 9)],
      after: [cell(1n, 3), cell(2n, 4)]
    },
    {
      name: 'identical',
      before: [cell(4n, 5, 100n), cell(8n, 6, -7n)],
      after: [cell(4n, 5, 100n), cell(8n, 6, -7n)]
    },
    {name: 'interleaved', before: interleavedBefore, after: interleavedAfter},
    {name: 'before empty', before: [], after: [cell(5n, 2), cell(6n, 3)]},
    {name: 'after empty', before: [cell(5n, 2), cell(6n, 3)], after: []},
    {name: 'both empty', before: [], after: []},
    {
      name: 'high word only',
      before: [cell(7n, 1), cell((1n << 32n) | 7n, 2), cell(3n << 32n, 3)],
      after: [
        cell(7n, 4),
        cell((2n << 32n) | 7n, 5),
        cell(3n << 32n, 6),
        cell(0xfffffffen << 32n, 8)
      ]
    },
    {
      name: 'one and one',
      before: [cell(0n, 2, 1n << 40n)],
      after: [cell(0n, 5, (1n << 40n) + 1n)]
    },
    {
      name: 'random overlapping',
      before: createRandomTable(11, 700, 2000),
      after: createRandomTable(12, 900, 2000)
    },
    {
      name: 'large (more than 256 rows each)',
      before: createRandomTable(21, 3000, 6000),
      after: createRandomTable(22, 2500, 6000)
    }
  ];
}

for (const [measure, zScore] of [
  ['count', 'poisson'],
  ['sum', 'standardized'],
  ['count', 'standardized'],
  ['sum', 'poisson']
] as const) {
  it(`GPUCellTableCompare ${measure} measure with ${zScore} z-score matches the CPU oracle`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) {
      return;
    }
    const fixture = new Fixture(device);
    // Table and union capacities are larger than every scenario, so the tails are exercised.
    const scenarios = getScenarios();
    const maximumRows = 6000;
    const compare = createCompareFixture(fixture, {
      beforeCapacity: 3100,
      afterCapacity: 2600,
      capacity: maximumRows,
      measure,
      zScore
    });
    for (const scenario of scenarios) {
      const actual = await compare.run(scenario.before, scenario.after);
      const expected = compareCellTablesOnCPU(scenario.before, scenario.after, {
        measure,
        sumScale: SUM_SCALE,
        zScore,
        capacity: maximumRows
      });
      expectMatchesOracle(actual, expected, zScore, `${measure}/${scenario.name}`);
    }
    compare.compiled.destroy();
    fixture.destroy();
  });
}

it('GPUCellTableCompare bounds the union, keeps the smallest keys, and ORs input overflow', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const fixture = new Fixture(device);
  const before = createRandomTable(31, 300, 900);
  const after = createRandomTable(32, 300, 900);
  for (const measure of ['count', 'sum'] as const) {
    const zScore = measure === 'count' ? 'poisson' : 'standardized';
    for (const capacity of [1, 7, 256, 400, 1000]) {
      const compare = createCompareFixture(fixture, {
        beforeCapacity: 300,
        afterCapacity: 300,
        capacity,
        measure,
        zScore
      });
      const actual = await compare.run(before, after);
      const expected = compareCellTablesOnCPU(before, after, {
        measure,
        sumScale: SUM_SCALE,
        zScore,
        capacity
      });
      expect(expected.overflow).toBe(capacity < expected.total);
      expectMatchesOracle(actual, expected, zScore, `${measure}/capacity ${capacity}`);
      compare.compiled.destroy();
    }
  }
  // An overflowed input table raises overflow even when the union fits.
  const compare = createCompareFixture(fixture, {
    beforeCapacity: 4,
    afterCapacity: 4,
    capacity: 16,
    measure: 'count',
    zScore: 'poisson'
  });
  for (const flags of [
    [1, 0],
    [0, 1],
    [0, 0]
  ] as [number, number][]) {
    const actual = await compare.run([cell(1n, 1)], [cell(2n, 2)], flags);
    expect(actual.overflow).toBe(flags[0] | flags[1]);
    expect(actual.count).toBe(2);
    expect(actual.total).toBe(2);
  }
  compare.compiled.destroy();
  fixture.destroy();
});

it('GPUCellTableCompare follows rewritten tables without recompiling and is deterministic', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const fixture = new Fixture(device);
  const capacity = 1500;
  const compare = createCompareFixture(fixture, {
    beforeCapacity: 800,
    afterCapacity: 800,
    capacity,
    measure: 'sum',
    zScore: 'standardized'
  });
  const compiled = compare.compiled;
  const totals: number[] = [];
  for (const seed of [1, 2, 3, 1]) {
    const before = createRandomTable(seed * 3, 100 + seed * 150, 1200);
    const after = createRandomTable(seed * 3 + 1, 700 - seed * 100, 1200);
    const first = await compare.run(before, after);
    const expected = compareCellTablesOnCPU(before, after, {
      measure: 'sum',
      sumScale: SUM_SCALE,
      zScore: 'standardized',
      capacity
    });
    expectMatchesOracle(first, expected, 'standardized', `seed ${seed}`);
    totals.push(first.total);
    // Re-encoding the same graph gives bitwise identical output, including the z-scores.
    const bitsOf = async () => ({
      words: await Promise.all(
        (
          [
            'cells',
            'presence',
            'before',
            'after',
            'delta',
            'ratio',
            'percentChange',
            'zScore'
          ] as const
        ).map(async name =>
          readUint32(compare.buffers[name], name === 'cells' ? 2 * capacity : capacity)
        )
      )
    });
    const firstBits = await bitsOf();
    submitGraph(device, compiled, undefined);
    expect(await bitsOf()).toEqual(firstBits);
    expect(compare.compiled).toBe(compiled);
  }
  expect(new Set(totals).size).toBeGreaterThan(2);
  expect(totals[0]).toBe(totals[3]);
  compiled.destroy();
  fixture.destroy();
});

it('GPUCellTableCompare compares two real GPUCellAggregation periods in one graph', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const resolution = 6;
  for (const measure of ['count', 'sum'] as const) {
    const fixture = new Fixture(device);
    const positionsBefore = createPointPositions(41, 1500, resolution);
    const positionsAfter = createPointPositions(42, 1800, resolution);
    // Periods share clusters but differ elsewhere; a few NaN coordinates are skipped.
    positionsBefore[4] = NaN;
    const rowsBefore = positionsBefore.length / 2;
    const rowsAfter = positionsAfter.length / 2;
    const createValues = (seed: number, rows: number) => {
      const random = createRandom(seed);
      return Float32Array.from({length: rows}, () => Math.fround((random() - 0.4) * 500));
    };
    const valuesBefore = createValues(51, rowsBefore);
    const valuesAfter = createValues(52, rowsAfter);
    const graph = new GPUCommandGraph(device, {id: 'two-periods'});
    const tableCapacity = 2048;
    const createTable = (name: string): GPUCellTable => {
      const view = <Format extends 'uint32' | 'uint32x2'>(
        suffix: string,
        format: Format,
        length: number
      ) =>
        importGraphBuffer(
          graph,
          `${name}-${suffix}`,
          fixture.output(format === 'uint32x2' ? 2 * length : length),
          format,
          length
        );
      return {
        cells: view('cells', 'uint32x2', tableCapacity),
        counts: view('counts', 'uint32', tableCapacity),
        sums: view('sums', 'uint32x2', tableCapacity),
        count: view('count', 'uint32', 1),
        overflow: view('overflow', 'uint32', 1)
      };
    };
    const periodBefore = createTable('period-a');
    const periodAfter = createTable('period-b');
    const aggregate = (
      id: string,
      positions: Float32Array,
      values: Float32Array,
      output: GPUCellTable
    ) =>
      new GPUCellAggregation({
        id,
        family: 'quadbin',
        resolution,
        positions: importGraphBuffer(
          graph,
          `${id}-positions`,
          fixture.input(positions),
          'float32x2',
          positions.length / 2
        ),
        values: importGraphBuffer(
          graph,
          `${id}-values`,
          fixture.input(values),
          'float32',
          values.length
        ),
        sumScale: SUM_SCALE,
        output
      });
    graph.add(aggregate('aggregate-a', positionsBefore, valuesBefore, periodBefore));
    graph.add(aggregate('aggregate-b', positionsAfter, valuesAfter, periodAfter));
    const capacity = 4096;
    const buffers = createCompareBuffers(fixture, capacity);
    const zScore = measure === 'count' ? 'poisson' : 'standardized';
    graph.add(
      new GPUCellTableCompare({
        before: periodBefore,
        after: periodAfter,
        measure,
        sumScale: SUM_SCALE,
        zScore,
        output: importCompareOutput(graph, buffers, capacity)
      })
    );
    const compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    const actual = await readCompare(buffers, capacity);
    const aggregateOnCPU = (positions: Float32Array, values: Float32Array): CompareInputCell[] =>
      aggregateCellsOnCPU({
        family: 'quadbin',
        resolution,
        keys: getQuadbinPointKeys(positions, resolution),
        values,
        sumScale: SUM_SCALE
      }).map(entry => ({
        key: entry.key,
        count: entry.count,
        sum: entry.sum
      }));
    const expectedBefore = aggregateOnCPU(positionsBefore, valuesBefore);
    const expectedAfter = aggregateOnCPU(positionsAfter, valuesAfter);
    expect(expectedBefore.length).toBeLessThan(tableCapacity);
    expect(expectedAfter.length).toBeLessThan(tableCapacity);
    const expected = compareCellTablesOnCPU(expectedBefore, expectedAfter, {
      measure,
      sumScale: SUM_SCALE,
      zScore,
      capacity
    });
    // Both periods contribute: cells only in one period and cells in both.
    const presences = new Set(expected.rows.map(row => row.presence));
    expect([...presences].sort()).toEqual([1, 2, 3]);
    expectMatchesOracle(actual, expected, zScore, `real ${measure}`);
    compiled.destroy();
    fixture.destroy();
  }
});
