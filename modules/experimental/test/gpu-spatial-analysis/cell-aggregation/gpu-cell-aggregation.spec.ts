// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {cellToChildren, getPentagons, latLngToCell} from 'h3-js';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPUCellAggregation,
  GPUCellLevelSelection,
  GPUCellPyramid,
  type GPUCellAggregationProps,
  type GPUCellFamily,
  type GPUCellTable
} from '../../../src/gpu-spatial-analysis/cell-aggregation';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  aggregateCellsOnCPU,
  getCellResolution,
  getQuadbinPointKeys,
  isValidCellKey,
  h3ToBigInt,
  joinCellKey,
  quadbinPointToCell,
  splitCellKey,
  type OracleCell
} from './cell-aggregation-oracle';
import {createPointPositions, createRandom} from './cell-aggregation-points';

const SUM_SCALE = 65536;

type TableBuffers = Record<
  | 'cells'
  | 'counts'
  | 'sums'
  | 'sumValues'
  | 'minimums'
  | 'maximums'
  | 'count'
  | 'overflow'
  | 'total',
  Buffer
> & {capacity: number};

type ActualTable = {
  count: number;
  overflow: number;
  total: number;
  rows: OracleCell[];
  sumValues: number[];
  /** True when every row past `count` holds the empty key and a zero count. */
  hasEmptyTail: boolean;
};

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

  createTable(capacity: number): TableBuffers {
    return {
      capacity,
      cells: this.output(2 * capacity),
      counts: this.output(capacity),
      sums: this.output(2 * capacity),
      sumValues: this.output(capacity),
      minimums: this.output(capacity),
      maximums: this.output(capacity),
      count: this.output(1),
      overflow: this.output(1),
      total: this.output(1)
    };
  }

  destroy(): void {
    for (const buffer of this.buffers) {
      buffer.destroy();
    }
  }
}

function importTable(graph: GPUCommandGraph, prefix: string, table: TableBuffers): GPUCellTable {
  const {capacity} = table;
  return {
    cells: importGraphBuffer(graph, `${prefix}-cells`, table.cells, 'uint32x2', capacity),
    counts: importGraphBuffer(graph, `${prefix}-counts`, table.counts, 'uint32', capacity),
    sums: importGraphBuffer(graph, `${prefix}-sums`, table.sums, 'uint32x2', capacity),
    sumValues: importGraphBuffer(
      graph,
      `${prefix}-sum-values`,
      table.sumValues,
      'float32',
      capacity
    ),
    minimums: importGraphBuffer(graph, `${prefix}-minimums`, table.minimums, 'float32', capacity),
    maximums: importGraphBuffer(graph, `${prefix}-maximums`, table.maximums, 'float32', capacity),
    count: importGraphBuffer(graph, `${prefix}-count`, table.count, 'uint32', 1),
    overflow: importGraphBuffer(graph, `${prefix}-overflow`, table.overflow, 'uint32', 1),
    totalCount: importGraphBuffer(graph, `${prefix}-total`, table.total, 'uint32', 1)
  };
}

async function readTable(table: TableBuffers): Promise<ActualTable> {
  const {capacity} = table;
  const [count] = await readUint32(table.count, 1);
  const [overflow] = await readUint32(table.overflow, 1);
  const [total] = await readUint32(table.total, 1);
  const cells = await readUint32(table.cells, 2 * capacity);
  const counts = await readUint32(table.counts, capacity);
  const sums = await readUint32(table.sums, 2 * capacity);
  const sumValues = await readFloat32(table.sumValues, capacity);
  const minimums = await readFloat32(table.minimums, capacity);
  const maximums = await readFloat32(table.maximums, capacity);
  const rows: OracleCell[] = [];
  for (let row = 0; row < count; row++) {
    rows.push({
      key: joinCellKey(cells[2 * row], cells[2 * row + 1]),
      count: counts[row],
      sum: BigInt.asIntN(64, joinCellKey(sums[2 * row], sums[2 * row + 1])),
      minimum: minimums[row],
      maximum: maximums[row]
    });
  }
  let hasEmptyTail = true;
  for (let row = count; row < capacity; row++) {
    hasEmptyTail &&=
      cells[2 * row] === 0xffffffff && cells[2 * row + 1] === 0xffffffff && counts[row] === 0;
  }
  return {
    count,
    overflow,
    total,
    rows,
    sumValues: sumValues.slice(0, count),
    hasEmptyTail
  };
}

function expectTable(actual: ActualTable, expected: OracleCell[], capacity: number, label: string) {
  expect(actual.total, `${label} total`).toBe(expected.length);
  expect(actual.count, `${label} count`).toBe(Math.min(expected.length, capacity));
  expect(actual.overflow, `${label} overflow`).toBe(expected.length > capacity ? 1 : 0);
  expect(actual.hasEmptyTail, `${label} empty tail`).toBe(true);
  const bounded = expected.slice(0, capacity);
  const format = (cells: OracleCell[]) =>
    cells.map(cell => `${cell.key.toString(16)} ${cell.count} ${cell.sum}`);
  expect(format(actual.rows), `${label} keys, counts, sums`).toEqual(format(bounded));
  // Extremes are bit exact (Object.is distinguishes -0).
  for (const [row, cell] of bounded.entries()) {
    expect(
      Object.is(actual.rows[row].minimum, Math.fround(cell.minimum)),
      `${label} min ${row}`
    ).toBe(true);
    expect(
      Object.is(actual.rows[row].maximum, Math.fround(cell.maximum)),
      `${label} max ${row}`
    ).toBe(true);
    const sumValue = Number(cell.sum) / SUM_SCALE;
    expect(Math.abs(actual.sumValues[row] - sumValue)).toBeLessThanOrEqual(
      Math.abs(sumValue) * 1e-6 + 1e-6
    );
  }
}

function createValues(seed: number, rows: number): Float32Array {
  const random = createRandom(seed);
  return Float32Array.from({length: rows}, () => {
    const roll = random();
    return roll < 0.01 ? NaN : roll < 0.02 ? -0 : Math.fround((random() - 0.4) * 1000);
  });
}

type AggregationScene = {
  family: GPUCellFamily;
  resolution: number;
  positions?: Float32Array;
  cells?: Uint32Array;
  wordOrder?: GPUCellAggregationProps['wordOrder'];
  values: Float32Array;
  capacity: number;
};

/** One compiled aggregation graph with writable values and mask. */
function createAggregation(fixture: Fixture, scene: AggregationScene) {
  const {device} = fixture;
  const rows = scene.values.length;
  const graph = new GPUCommandGraph(device, {id: 'cell-aggregation-graph'});
  const valuesBuffer = fixture.input(scene.values);
  const maskBuffer = fixture.input(new Uint32Array(rows).fill(1));
  const table = fixture.createTable(scene.capacity);
  graph.add(
    new GPUCellAggregation({
      family: scene.family,
      resolution: scene.resolution,
      positions: scene.positions
        ? importGraphBuffer(graph, 'positions', fixture.input(scene.positions), 'float32x2', rows)
        : undefined,
      cells: scene.cells
        ? importGraphBuffer(graph, 'cells', fixture.input(scene.cells), 'uint32x2', rows)
        : undefined,
      wordOrder: scene.wordOrder,
      values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', rows),
      mask: importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', rows),
      sumScale: SUM_SCALE,
      output: importTable(graph, 'table', table)
    })
  );
  const compiled = graph.compile();
  return {
    compiled,
    table,
    async run(values: Float32Array, mask: Uint32Array): Promise<ActualTable> {
      valuesBuffer.write(values);
      maskBuffer.write(mask);
      submitGraph(device, compiled, undefined);
      return readTable(table);
    }
  };
}

it('GPUCellAggregation quadbin point keys and tables match the CPU reference', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  for (const resolution of [0, 1, 7, 15, 16, 21, 26]) {
    const fixture = new Fixture(device);
    const positions = createPointPositions(100 + resolution, 6000, resolution);
    // A few NaN coordinates are skipped.
    positions[10] = NaN;
    positions[33] = NaN;
    const rows = positions.length / 2;
    const values = createValues(resolution, rows);
    const keys = getQuadbinPointKeys(positions, resolution);
    const expected = aggregateCellsOnCPU({
      family: 'quadbin',
      resolution,
      keys,
      values,
      sumScale: SUM_SCALE
    });
    const aggregation = createAggregation(fixture, {
      family: 'quadbin',
      resolution,
      positions,
      values,
      capacity: rows
    });
    const actual = await aggregation.run(values, new Uint32Array(rows).fill(1));
    expectTable(actual, expected, rows, `resolution ${resolution}`);
    if (resolution >= 7) {
      expect(expected.length, 'fine levels hold many cells').toBeGreaterThan(rows / 4);
    }
    aggregation.compiled.destroy();
    fixture.destroy();
  }
});

it('GPUCellAggregation follows per-frame masks and values without recompiling', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const fixture = new Fixture(device);
  const resolution = 9;
  const positions = createPointPositions(7, 4000, resolution);
  const rows = positions.length / 2;
  const keys = getQuadbinPointKeys(positions, resolution);
  const aggregation = createAggregation(fixture, {
    family: 'quadbin',
    resolution,
    positions,
    values: createValues(1, rows),
    capacity: rows
  });
  const compiled = aggregation.compiled;
  const totals: number[] = [];
  for (const seed of [1, 2, 3, 1]) {
    const random = createRandom(seed * 17);
    const mask = Uint32Array.from({length: rows}, () => (random() < 0.2 * seed ? 0 : 1));
    const values = createValues(seed, rows);
    const actual = await aggregation.run(values, mask);
    const expected = aggregateCellsOnCPU({
      family: 'quadbin',
      resolution,
      keys,
      values,
      mask,
      sumScale: SUM_SCALE
    });
    expectTable(actual, expected, rows, `seed ${seed}`);
    totals.push(actual.total);
    expect(aggregation.compiled).toBe(compiled);
  }
  expect(new Set(totals).size).toBeGreaterThan(2);
  expect(totals[0]).toBe(totals[3]);
  // An all-zero mask leaves an empty table.
  const empty = await aggregation.run(createValues(1, rows), new Uint32Array(rows));
  expect([empty.count, empty.total, empty.overflow, empty.hasEmptyTail]).toEqual([0, 0, 0, true]);
  compiled.destroy();
  fixture.destroy();
});

/** Pre-keyed H3 rows: random cells, pentagons, poles and antimeridian, finer, coarser and invalid. */
function createH3Keys(seed: number, rows: number, resolution: number): (bigint | null)[] {
  const random = createRandom(seed);
  const keys: bigint[] = [];
  const special = [
    ...getPentagons(resolution + 1),
    ...cellToChildren(getPentagons(resolution)[3], resolution + 2),
    latLngToCell(89.9, 10, resolution + 1),
    latLngToCell(-89.9, -170, resolution + 3),
    latLngToCell(0, 180, resolution + 1),
    latLngToCell(0, -179.9999, resolution + 1),
    latLngToCell(65.5, 179.99, resolution + 2),
    latLngToCell(-16.2, -179.999, resolution + 2)
  ];
  keys.push(...special.map(h3ToBigInt));
  while (keys.length < rows) {
    const roll = random();
    const latitude = random() * 180 - 90;
    const longitude = random() * 360 - 180;
    if (roll < 0.05) {
      // Coarser than the table: skipped.
      keys.push(h3ToBigInt(latLngToCell(latitude, longitude, Math.max(resolution - 2, 0))));
    } else if (roll < 0.08) {
      // Invalid: garbage words, a used digit set to 7, and the wrong mode.
      const cell = h3ToBigInt(latLngToCell(latitude, longitude, resolution + 1));
      keys.push(
        [0x123456789abcdefn, cell | (7n << BigInt(3 * (15 - 1))), cell ^ (3n << 59n)][
          Math.floor(random() * 3)
        ]
      );
    } else if (roll < 0.4) {
      // Clustered fine cells.
      keys.push(
        h3ToBigInt(latLngToCell(40 + random() * 0.3, -74 + random() * 0.3, resolution + 3))
      );
    } else {
      keys.push(
        h3ToBigInt(
          latLngToCell(latitude, longitude, Math.min(resolution + Math.floor(random() * 4), 15))
        )
      );
    }
  }
  return keys.slice(0, rows);
}

function packKeys(keys: readonly (bigint | null)[], highLow = false): Uint32Array {
  const words = new Uint32Array(keys.length * 2);
  for (const [row, key] of keys.entries()) {
    const [low, high] = splitCellKey(key ?? 0n);
    words[2 * row] = highLow ? high : low;
    words[2 * row + 1] = highLow ? low : high;
  }
  return words;
}

it('GPUCellAggregation aggregates pre-keyed H3 and Quadbin rows (CARTO columns)', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const random = createRandom(5);
  const scenes: {
    family: GPUCellFamily;
    resolution: number;
    keys: (bigint | null)[];
    highLow?: boolean;
  }[] = [
    {family: 'h3', resolution: 5, keys: createH3Keys(1, 3000, 5)},
    // Width 7 + 30 bits: the two-word sort path.
    {family: 'h3', resolution: 10, keys: createH3Keys(2, 3000, 10)},
    {
      family: 'h3',
      resolution: 0,
      keys: createH3Keys(3, 1000, 0),
      highLow: true
    },
    {
      family: 'quadbin',
      resolution: 12,
      keys: Array.from({length: 3000}, (_, row) => {
        const longitude = random() * 360 - 180;
        const latitude = random() * 170 - 85;
        const level = row % 10 === 0 ? 8 : 12 + Math.floor(random() * 15);
        const cell = quadbinPointToCell(longitude, latitude, Math.min(level, 26));
        return row % 37 === 0 ? cell - 1n : cell;
      })
    }
  ];
  for (const scene of scenes) {
    const fixture = new Fixture(device);
    const rows = scene.keys.length;
    const values = createValues(scene.resolution + 40, rows);
    const expected = aggregateCellsOnCPU({
      family: scene.family,
      resolution: scene.resolution,
      keys: scene.keys,
      values,
      sumScale: SUM_SCALE
    });
    const aggregation = createAggregation(fixture, {
      family: scene.family,
      resolution: scene.resolution,
      cells: packKeys(scene.keys, scene.highLow),
      wordOrder: scene.highLow ? 'high-low' : 'little-endian',
      values,
      capacity: rows
    });
    const actual = await aggregation.run(values, new Uint32Array(rows).fill(1));
    expectTable(actual, expected, rows, `${scene.family} ${scene.resolution}`);
    expect(expected.length).toBeGreaterThan(scene.resolution === 0 ? 50 : 300);
    aggregation.compiled.destroy();
    fixture.destroy();
  }
});

it('GPUCellPyramid roll-ups equal direct aggregation and switch levels per frame', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  for (const {family, levels} of [
    {family: 'quadbin' as const, levels: [18, 14, 9, 3, 0]},
    {family: 'h3' as const, levels: [9, 6, 2]}
  ]) {
    const fixture = new Fixture(device);
    const finest = levels[0];
    let keys: (bigint | null)[];
    let positions: Float32Array | undefined;
    if (family === 'quadbin') {
      positions = createPointPositions(77, 8000, finest);
      keys = getQuadbinPointKeys(positions, finest);
    } else {
      keys = createH3Keys(9, 8000, finest);
    }
    const rows = keys.length;
    const values = createValues(8, rows);
    // Capacities hold every cell here; the overflow test covers truncated levels.
    const capacities = levels.map((_, levelIndex) => rows + 2 + levelIndex);
    const tables = capacities.map(capacity => fixture.createTable(capacity));
    const levelCountsBuffer = fixture.output(levels.length);
    const graph = new GPUCommandGraph(device, {id: 'cell-pyramid-graph'});
    const levelCounts = importGraphBuffer(
      graph,
      'level-counts',
      levelCountsBuffer,
      'uint32',
      levels.length
    );
    const pyramid = new GPUCellPyramid({
      family,
      positions: positions
        ? importGraphBuffer(graph, 'positions', fixture.input(positions), 'float32x2', rows)
        : undefined,
      cells: positions
        ? undefined
        : importGraphBuffer(graph, 'cells', fixture.input(packKeys(keys)), 'uint32x2', rows),
      values: importGraphBuffer(graph, 'values', fixture.input(values), 'float32', rows),
      sumScale: SUM_SCALE,
      levels: levels.map((resolution, levelIndex) => ({
        resolution,
        output: importTable(graph, `level-${levelIndex}`, tables[levelIndex])
      })),
      levelCounts
    });
    graph.add(pyramid);
    const compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    const counts: number[] = [];
    // The pyramid only sees rows that reach the finest level; coarser pre-keyed rows are skipped.
    const finestKeys = keys.map(key =>
      key !== null && isValidCellKey(family, key) && getCellResolution(family, key) >= finest
        ? key
        : null
    );
    for (const [levelIndex, resolution] of levels.entries()) {
      const expected = aggregateCellsOnCPU({
        family,
        resolution,
        keys: finestKeys,
        values,
        sumScale: SUM_SCALE
      });
      const actual = await readTable(tables[levelIndex]);
      expectTable(actual, expected, capacities[levelIndex], `${family} level ${resolution}`);
      counts.push(actual.count);
    }
    expect(await readUint32(levelCountsBuffer, levels.length)).toEqual(counts);

    // Level selection runs in its own per-frame graph: rewrite one uint32, no rebuild.
    const selectionGraph = new GPUCommandGraph(device, {
      id: 'cell-level-selection-graph'
    });
    const activeLevel = new GPUParameterBuffer(device, {
      id: 'active-level',
      format: 'uint32',
      length: 1
    });
    const countBuffer = fixture.output(1);
    const drawBuffer = fixture.output(4);
    drawBuffer.write(Uint32Array.from([6, 0, 0, 0]));
    const levelFirstRows = levels.map((_, levelIndex) => levelIndex * 100);
    selectionGraph.add(
      new GPUCellLevelSelection({
        levelCounts: importGraphBuffer(
          selectionGraph,
          'level-counts',
          levelCountsBuffer,
          'uint32',
          levels.length
        ),
        activeLevel: activeLevel.importToGraph(selectionGraph),
        levelFirstRows,
        output: {
          count: importGraphBuffer(selectionGraph, 'count', countBuffer, 'uint32', 1),
          drawArguments: importGraphBuffer(selectionGraph, 'draw', drawBuffer, 'uint32', 4)
        }
      })
    );
    const selection = selectionGraph.compile();
    for (const level of [2, 0, levels.length - 1, 99]) {
      activeLevel.write(Uint32Array.from([level]));
      submitGraph(device, selection, undefined);
      const selected = Math.min(level, levels.length - 1);
      expect(await readUint32(countBuffer, 1)).toEqual([counts[selected]]);
      expect(await readUint32(drawBuffer, 4)).toEqual([
        6,
        counts[selected],
        0,
        levelFirstRows[selected]
      ]);
    }
    selection.destroy();
    activeLevel.destroy();
    compiled.destroy();
    fixture.destroy();
  }
});

it('GPUCellAggregation and roll-ups clamp to capacity and propagate overflow', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const fixture = new Fixture(device);
  const levels = [10, 6, 1];
  const positions = createPointPositions(31, 3000, levels[0]);
  const rows = positions.length / 2;
  const keys = getQuadbinPointKeys(positions, levels[0]);
  const values = createValues(31, rows);
  const capacities = [25, 1000, 8];
  const tables = capacities.map(capacity => fixture.createTable(capacity));
  const graph = new GPUCommandGraph(device, {id: 'cell-overflow-graph'});
  graph.add(
    new GPUCellPyramid({
      family: 'quadbin',
      positions: importGraphBuffer(graph, 'positions', fixture.input(positions), 'float32x2', rows),
      values: importGraphBuffer(graph, 'values', fixture.input(values), 'float32', rows),
      sumScale: SUM_SCALE,
      levels: levels.map((resolution, levelIndex) => ({
        resolution,
        output: importTable(graph, `level-${levelIndex}`, tables[levelIndex])
      }))
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const finest = aggregateCellsOnCPU({
    family: 'quadbin',
    resolution: levels[0],
    keys,
    values,
    sumScale: SUM_SCALE
  });
  expect(finest.length).toBeGreaterThan(capacities[0]);
  const level0 = await readTable(tables[0]);
  // The 25 smallest keys survive, exactly.
  expectTable(level0, finest, capacities[0], 'finest');
  // Level 1 rolls up only the surviving cells and inherits the overflow flag.
  const surviving = new Set(finest.slice(0, capacities[0]).map(cell => cell.key));
  const survivingKeys = keys.map(key => (key !== null && surviving.has(key) ? key : null));
  const level1Expected = aggregateCellsOnCPU({
    family: 'quadbin',
    resolution: levels[1],
    keys: survivingKeys,
    values,
    sumScale: SUM_SCALE
  });
  const level1 = await readTable(tables[1]);
  expect(level1.overflow).toBe(1);
  expect(level1.rows.map(cell => [cell.key, cell.count, cell.sum])).toEqual(
    level1Expected.map(cell => [cell.key, cell.count, cell.sum])
  );
  const level2 = await readTable(tables[2]);
  expect(level2.overflow).toBe(1);
  compiled.destroy();
  fixture.destroy();
});
