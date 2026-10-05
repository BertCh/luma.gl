// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {DrawCommandBuffer, GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {afterEach, expect, it, vi} from 'vitest';
import {
  GPUParameterBuffer,
  importGraphBuffer,
  submitGraph
} from '../../../src/utils/gpu-contributor-utils';
import {GPUPointDensity} from '../../../src/geospatial/point-density';
import {
  decodeGPURegionStatistics,
  getGPURegionStatisticsSummaryLength,
  GPURegionStatistics,
  type GPURegionStatisticsResult
} from '../../../src/geospatial/region-statistics';
import {GPUResidentRowSelection} from '../../../src/gpu-tables/residency-arena';
import {
  getGPUTimeWindowParameterValues,
  GPU_TIME_WINDOW_PARAMETER_LENGTH,
  GPUTimeWindowFilter
} from '../../../src/gpu-dataframe/time-window-filter';
import type {GPUCompactOutput} from '../../../src/utils/gpu-contributor-types';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
} from '../../utils/gpu-contributor-test-utils';
import {
  ArenaMirror,
  createRandom,
  getLiveRows,
  type MakeTileData
} from './residency-arena-composition-oracle';

const PAGE_ROW_COUNT = 64;
const ROW_CAPACITY = PAGE_ROW_COUNT * 64;
const MAX_TILE_COUNT = 64;
const STATE_COUNT = 12;

const compileSpy = vi.spyOn(GPUCommandGraph.prototype, 'compile');

afterEach(() => {
  compileSpy.mockClear();
});

type CompactOutputBuffers = {
  output: GPUCompactOutput;
  ids: Buffer;
  count: Buffer;
  overflow: Buffer;
  total: Buffer;
};

function createCompactOutput(
  device: Device,
  graph: GPUCommandGraph,
  name: string
): CompactOutputBuffers {
  const ids = createOutputBuffer(device, ROW_CAPACITY);
  const count = createOutputBuffer(device, 1);
  const overflow = createOutputBuffer(device, 1);
  const total = createOutputBuffer(device, 1);
  return {
    ids,
    count,
    overflow,
    total,
    output: {
      ids: importGraphBuffer(graph, `${name}-ids`, ids, 'uint32', ROW_CAPACITY),
      count: importGraphBuffer(graph, `${name}-count`, count, 'uint32', 1),
      overflow: importGraphBuffer(graph, `${name}-overflow`, overflow, 'uint32', 1),
      totalCount: importGraphBuffer(graph, `${name}-total`, total, 'uint32', 1)
    }
  };
}

async function readCompact(buffers: CompactOutputBuffers): Promise<{ids: number[]; count: number}> {
  const [count] = await readUint32(buffers.count, 1);
  return {count, ids: await readUint32(buffers.ids, count)};
}

async function readDrawInstanceCount(drawCommands: DrawCommandBuffer): Promise<number> {
  const bytes = await drawCommands.buffer.readAsync(drawCommands.getInstanceCountByteOffset(0), 4);
  return new Uint32Array(bytes.buffer, bytes.byteOffset, 1)[0];
}

function destroyAll(resources: {destroy(): void}[]): void {
  for (const resource of resources) {
    resource.destroy();
  }
}

type TimeWindow = {start: number; end: number};

function createRandomWindow(random: () => number): TimeWindow {
  const start = Math.floor(random() * 900);
  return {start, end: start + Math.floor(random() * 500)};
}

/** Time-window tiles: integer times in [0, 1000) and unique stable IDs. */
function createTimeTileFactory(random: () => number): MakeTileData {
  let nextId = 100000;
  return rowCount => ({
    rowCount,
    columns: {
      time: Float32Array.from({length: rowCount}, () => Math.floor(random() * 1000)),
      ids: Uint32Array.from({length: rowCount}, () => nextId++)
    }
  });
}

function isInWindow(time: number, window: TimeWindow): boolean {
  return time >= window.start && time <= window.end;
}

it('arena columns drive GPUTimeWindowFilter with an indirect draw; the live predicate excludes stale rows', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const random = createRandom(11);
  const makeTile = createTimeTileFactory(random);
  const mirror = new ArenaMirror(device, {
    id: 'composition-time',
    rowCapacity: ROW_CAPACITY,
    pageRowCount: PAGE_ROW_COUNT,
    columns: [
      {name: 'time', format: 'float32'},
      {name: 'ids', format: 'uint32'}
    ]
  });
  const window = new GPUParameterBuffer(device, {
    id: 'composition-window',
    format: 'float32',
    length: GPU_TIME_WINDOW_PARAMETER_LENGTH,
    values: getGPUTimeWindowParameterValues({start: 0, end: 0})
  });
  const drawCommands = new DrawCommandBuffer(device, {
    id: 'composition-draw',
    type: 'draw',
    commands: [{vertexCount: 6, instanceCount: 0}]
  });
  const graph = new GPUCommandGraph(device, {id: 'composition-time-window'});
  const views = mirror.arena.importToGraph(graph);
  const windowView = window.importToGraph(graph);
  const timestamps = views.columns.time as GraphDataView<'float32'>;
  const sourceIds = views.columns.ids as GraphDataView<'uint32'>;
  const live = createCompactOutput(device, graph, 'live');
  const stale = createCompactOutput(device, graph, 'stale');
  graph.add(
    new GPUTimeWindowFilter({
      id: 'live-filter',
      timestamps,
      sourceIds,
      additionalPredicates: [{kind: 'selection', mask: views.liveMask}],
      window: windowView,
      output: live.output,
      drawInstanceCount: graph.importGPUData(
        'composition-draw-count',
        drawCommands.getInstanceCountData(0)
      )
    })
  );
  // Negative control: the same filter without the live predicate sees stale rows.
  graph.add(
    new GPUTimeWindowFilter({
      id: 'stale-filter',
      timestamps,
      sourceIds,
      window: windowView,
      output: stale.output
    })
  );
  const compiled = graph.compile();
  expect(compileSpy).toHaveBeenCalledTimes(1);
  const nodeCount = compiled.stats.nodeOrder.length;
  console.log(
    `[residency-arena-composition] time-window + indirect draw graph: ${nodeCount} nodes`
  );

  let sawStaleOvercount = false;
  let sawEviction = false;
  const nonEmptyStates = new Set<number>();
  for (let state = 0; state < STATE_COUNT; state++) {
    const keysBefore = mirror.getKeys().length;
    mirror.churn(random, makeTile);
    sawEviction ||= keysBefore > 0 && mirror.arena.version > 1;
    const timeWindow = createRandomWindow(random);
    window.write(getGPUTimeWindowParameterValues(timeWindow));
    submitGraph(device, compiled, undefined);

    const liveRows = getLiveRows(mirror);
    const times = mirror.columns.time as Float32Array;
    const ids = mirror.columns.ids as Uint32Array;
    const expectedIds = liveRows
      .filter(row => isInWindow(times[row], timeWindow))
      .map(row => ids[row]);
    const result = await readCompact(live);
    expect(result.count).toBe(expectedIds.length);
    expect(result.ids).toEqual(expectedIds);
    expect(await readUint32(live.overflow, 1)).toEqual([0]);
    expect(await readDrawInstanceCount(drawCommands)).toBe(expectedIds.length);
    nonEmptyStates.add(expectedIds.length);

    // Without the live predicate every row, evicted or never written, is classified by its time.
    let expectedStaleCount = 0;
    for (let row = 0; row < ROW_CAPACITY; row++) {
      expectedStaleCount += isInWindow(times[row], timeWindow) ? 1 : 0;
    }
    const [staleCount] = await readUint32(stale.count, 1);
    expect(staleCount).toBe(expectedStaleCount);
    sawStaleOvercount ||= staleCount > expectedIds.length;
  }
  expect(sawEviction).toBe(true);
  expect(nonEmptyStates.size).toBeGreaterThan(3);
  // The dead-row bug: stale in-window rows are counted without the live predicate.
  expect(sawStaleOvercount).toBe(true);
  // One graph, never recompiled, same node count.
  expect(compileSpy).toHaveBeenCalledTimes(1);
  expect(compiled.stats.nodeOrder.length).toBe(nodeCount);

  compiled.destroy();
  destroyAll([window, drawCommands, mirror]);
  destroyAll([...Object.values(live).filter(isBuffer), ...Object.values(stale).filter(isBuffer)]);
});

function isBuffer(value: unknown): value is Buffer {
  return typeof (value as Buffer)?.destroy === 'function';
}

it('GPUTimeWindowFilter outputMask composes with GPUResidentRowSelection tile gating over arena columns', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const random = createRandom(23);
  const makeTile = createTimeTileFactory(random);
  const mirror = new ArenaMirror(device, {
    id: 'composition-gate',
    rowCapacity: ROW_CAPACITY,
    pageRowCount: PAGE_ROW_COUNT,
    maxTileCount: MAX_TILE_COUNT,
    columns: [
      {name: 'time', format: 'float32'},
      {name: 'ids', format: 'uint32'}
    ]
  });
  const window = new GPUParameterBuffer(device, {
    id: 'composition-gate-window',
    format: 'float32',
    length: GPU_TIME_WINDOW_PARAMETER_LENGTH,
    values: getGPUTimeWindowParameterValues({start: 0, end: 0})
  });
  const drawCommands = new DrawCommandBuffer(device, {
    id: 'composition-gate-draw',
    type: 'draw',
    commands: [{vertexCount: 6, instanceCount: 0}]
  });
  const tileMaskBuffer = createInputBuffer(device, new Uint32Array(MAX_TILE_COUNT));
  const timeMaskBuffer = createOutputBuffer(device, ROW_CAPACITY);
  const graph = new GPUCommandGraph(device, {id: 'composition-gate'});
  const views = mirror.arena.importToGraph(graph);
  const timeMask = importGraphBuffer(graph, 'time-mask', timeMaskBuffer, 'uint32', ROW_CAPACITY);
  const timeOnly = createCompactOutput(device, graph, 'time-only');
  const selected = createCompactOutput(device, graph, 'selected');
  graph.add(
    new GPUTimeWindowFilter({
      id: 'time-only-filter',
      timestamps: views.columns.time as GraphDataView<'float32'>,
      window: window.importToGraph(graph),
      output: timeOnly.output,
      outputMask: timeMask
    })
  );
  graph.add(
    new GPUResidentRowSelection({
      id: 'gated-selection',
      liveMask: views.liveMask,
      tileVisibility: {
        rowTileSlots: views.rowTileSlots,
        tileMask: importGraphBuffer(graph, 'tile-mask', tileMaskBuffer, 'uint32', MAX_TILE_COUNT)
      },
      additionalPredicates: [{kind: 'time-range', mask: timeMask}],
      sourceIds: views.columns.ids as GraphDataView<'uint32'>,
      output: selected.output,
      drawInstanceCount: graph.importGPUData(
        'composition-gate-draw-count',
        drawCommands.getInstanceCountData(0)
      )
    })
  );
  const compiled = graph.compile();
  expect(compileSpy).toHaveBeenCalledTimes(1);
  const nodeCount = compiled.stats.nodeOrder.length;
  console.log(
    `[residency-arena-composition] time mask + tile-gated resident selection graph: ${nodeCount} nodes`
  );

  let sawGateEffect = false;
  for (let state = 0; state < STATE_COUNT; state++) {
    mirror.churn(random, makeTile);
    const timeWindow = createRandomWindow(random);
    window.write(getGPUTimeWindowParameterValues(timeWindow));
    const tileMask = Uint32Array.from({length: MAX_TILE_COUNT}, () => (random() < 0.5 ? 1 : 0));
    tileMaskBuffer.write(tileMask);
    submitGraph(device, compiled, undefined);

    const times = mirror.columns.time as Float32Array;
    const ids = mirror.columns.ids as Uint32Array;
    const slots = mirror.arena.allocator.getRowTileSlots();
    const liveMask = mirror.getLiveMask();
    const expectedIds: number[] = [];
    let timeAndLiveCount = 0;
    const expectedTimeMask: number[] = [];
    for (let row = 0; row < ROW_CAPACITY; row++) {
      const timeIn = isInWindow(times[row], timeWindow);
      expectedTimeMask.push(timeIn ? 1 : 0);
      if (liveMask[row] && timeIn) {
        timeAndLiveCount++;
        if (slots[row] < MAX_TILE_COUNT && tileMask[slots[row]]) {
          expectedIds.push(ids[row]);
        }
      }
    }
    sawGateEffect ||= expectedIds.length < timeAndLiveCount;

    // The time mask covers every arena row (stale rows included); liveness is applied downstream.
    expect(await readUint32(timeMaskBuffer, ROW_CAPACITY)).toEqual(expectedTimeMask);
    const result = await readCompact(selected);
    expect(result.count).toBe(expectedIds.length);
    expect(result.ids).toEqual(expectedIds);
    expect(await readUint32(selected.overflow, 1)).toEqual([0]);
    expect(await readDrawInstanceCount(drawCommands)).toBe(expectedIds.length);
  }
  expect(sawGateEffect).toBe(true);
  expect(compileSpy).toHaveBeenCalledTimes(1);
  expect(compiled.stats.nodeOrder.length).toBe(nodeCount);

  compiled.destroy();
  destroyAll([window, drawCommands, mirror, tileMaskBuffer, timeMaskBuffer]);
  destroyAll([
    ...Object.values(timeOnly).filter(isBuffer),
    ...Object.values(selected).filter(isBuffer)
  ]);
});

it('arena columns drive GPUPointDensity; NaN dead positions exclude dead rows without a mask', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const random = createRandom(37);
  const makeTile: MakeTileData = rowCount => ({
    rowCount,
    columns: {
      // 0.013 keeps every point away from cell edges (multiples of 12.5) so f32 and f64 agree.
      positions: Float32Array.from(
        {length: rowCount * 2},
        () => Math.floor(random() * 1000) / 10 + 0.013
      ),
      weights: Float32Array.from({length: rowCount}, () => 1 + Math.floor(random() * 5))
    }
  });
  const mirror = new ArenaMirror(device, {
    id: 'composition-density',
    rowCapacity: ROW_CAPACITY,
    pageRowCount: PAGE_ROW_COUNT,
    columns: [
      {name: 'positions', format: 'float32x2', deadValue: Number.NaN},
      {name: 'weights', format: 'float32', deadValue: 0}
    ]
  });
  const GRID = 8;
  const CELL_COUNT = GRID * GRID;
  const countsBuffer = createOutputBuffer(device, CELL_COUNT);
  const sumsBuffer = createOutputBuffer(device, CELL_COUNT);
  const valuesBuffer = createOutputBuffer(device, CELL_COUNT);
  const graph = new GPUCommandGraph(device, {id: 'composition-density'});
  const views = mirror.arena.importToGraph(graph);
  graph.add(
    new GPUPointDensity({
      id: 'arena-density',
      positions: views.columns.positions as GraphDataView<'float32x2'>,
      weights: views.columns.weights as GraphDataView<'float32'>,
      bounds: [0, 0, 100, 100],
      gridSize: [GRID, GRID],
      statistic: 'sum',
      output: {
        values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', CELL_COUNT),
        counts: importGraphBuffer(graph, 'counts', countsBuffer, 'uint32', CELL_COUNT),
        sums: importGraphBuffer(graph, 'sums', sumsBuffer, 'float32', CELL_COUNT)
      }
    })
  );
  const compiled = graph.compile();
  expect(compileSpy).toHaveBeenCalledTimes(1);
  const nodeCount = compiled.stats.nodeOrder.length;
  console.log(`[residency-arena-composition] point-density graph: ${nodeCount} nodes`);

  let sawEvictionWithLiveRows = false;
  for (let state = 0; state < STATE_COUNT; state++) {
    mirror.churn(random, makeTile);
    submitGraph(device, compiled, undefined);

    // CPU oracle over live rows only.
    const positions = mirror.columns.positions as Float32Array;
    const weights = mirror.columns.weights as Float32Array;
    const expectedCounts = new Array<number>(CELL_COUNT).fill(0);
    const expectedSums = new Array<number>(CELL_COUNT).fill(0);
    const liveRows = getLiveRows(mirror);
    for (const row of liveRows) {
      const column = Math.min(Math.floor((positions[row * 2] / 100) * GRID), GRID - 1);
      const gridRow = Math.min(Math.floor((positions[row * 2 + 1] / 100) * GRID), GRID - 1);
      expectedCounts[gridRow * GRID + column]++;
      expectedSums[gridRow * GRID + column] += weights[row];
    }
    const counts = await readUint32(countsBuffer, CELL_COUNT);
    expect(counts).toEqual(expectedCounts);
    expect(counts.reduce((total, count) => total + count, 0)).toBe(mirror.arena.liveRowCount);
    const sums = await readFloat32(sumsBuffer, CELL_COUNT);
    for (let cell = 0; cell < CELL_COUNT; cell++) {
      expect(sums[cell]).toBeCloseTo(expectedSums[cell], 3);
    }
    sawEvictionWithLiveRows ||= liveRows.length > 0 && mirror.arena.version > 2;
  }
  expect(sawEvictionWithLiveRows).toBe(true);
  expect(compileSpy).toHaveBeenCalledTimes(1);
  expect(compiled.stats.nodeOrder.length).toBe(nodeCount);

  compiled.destroy();
  destroyAll([mirror, countsBuffer, sumsBuffer, valuesBuffer]);
});

const HISTOGRAM_BIN_COUNT = 8;
const POISON_VALUE = 1e6;

/**
 * CPU statistics with the `GPUHistogram` rule for float32 input: inclusive domain, last bin closed,
 * `bin = min(u32((value - min) / (max - min) * binCount), binCount - 1)`.
 *
 * The GPU divides in f32 (possibly through a reciprocal), so a value whose scaled position lies
 * within 1e-5 of a bin edge may land on either side. Those values are counted into
 * `ambiguousBins` for both neighbouring bins; every other value is counted in `histogram`, which
 * is exact.
 */
function computeStatisticsOracle(values: number[]): {
  valueCount: number;
  sum: number;
  minimum: number;
  maximum: number;
  histogram: number[];
  ambiguousBins: number[];
} {
  const minimum = values.length ? Math.min(...values) : 0;
  const maximum = values.length ? Math.max(...values) : 0;
  const histogram = new Array<number>(HISTOGRAM_BIN_COUNT).fill(0);
  const ambiguousBins = new Array<number>(HISTOGRAM_BIN_COUNT).fill(0);
  for (const value of values) {
    if (value === maximum) {
      histogram[HISTOGRAM_BIN_COUNT - 1]++;
      continue;
    }
    const scaled = ((value - minimum) / (maximum - minimum)) * HISTOGRAM_BIN_COUNT;
    const nearestEdge = Math.round(scaled);
    if (
      nearestEdge >= 1 &&
      nearestEdge < HISTOGRAM_BIN_COUNT &&
      Math.abs(scaled - nearestEdge) < 1e-5
    ) {
      ambiguousBins[nearestEdge - 1]++;
      ambiguousBins[nearestEdge]++;
    } else {
      histogram[Math.min(Math.floor(scaled), HISTOGRAM_BIN_COUNT - 1)]++;
    }
  }
  return {
    valueCount: values.length,
    sum: values.reduce((total, value) => total + value, 0),
    minimum,
    maximum,
    histogram,
    ambiguousBins
  };
}

function expectStatistics(
  actual: GPURegionStatisticsResult,
  rows: number[],
  values: Float32Array
): void {
  const expected = computeStatisticsOracle(rows.map(row => values[row]));
  expect(actual.selectedCount).toBe(rows.length);
  expect(actual.valueCount).toBe(expected.valueCount);
  expect(Math.abs(actual.sum - expected.sum)).toBeLessThanOrEqual(1);
  expect(actual.minimum).toBe(expected.minimum);
  expect(actual.maximum).toBe(expected.maximum);
  for (let bin = 0; bin < HISTOGRAM_BIN_COUNT; bin++) {
    expect(actual.histogram[bin]).toBeGreaterThanOrEqual(expected.histogram[bin]);
    expect(actual.histogram[bin]).toBeLessThanOrEqual(
      expected.histogram[bin] + expected.ambiguousBins[bin]
    );
  }
  expect(actual.histogram.reduce((total, count) => total + count, 0)).toBe(expected.valueCount);
  expect(actual.histogramOutsideCount).toBe(0);
}

it('GPURegionStatistics automatic histogram domain ignores dead rows holding stale extreme values', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const random = createRandom(53);
  const makeTile: MakeTileData = rowCount => ({
    rowCount,
    columns: {
      values: Float32Array.from({length: rowCount}, () => Math.floor(random() * 1001) - 500)
    }
  });
  const makePoisonTile: MakeTileData = rowCount => ({
    rowCount,
    columns: {values: new Float32Array(rowCount).fill(POISON_VALUE)}
  });
  const mirror = new ArenaMirror(device, {
    id: 'composition-stats',
    rowCapacity: ROW_CAPACITY,
    pageRowCount: PAGE_ROW_COUNT,
    maxTileCount: MAX_TILE_COUNT,
    columns: [{name: 'values', format: 'float32'}]
  });
  const summaryLength = getGPURegionStatisticsSummaryLength(HISTOGRAM_BIN_COUNT);
  const onesBuffer = createInputBuffer(device, new Uint32Array(ROW_CAPACITY).fill(1));
  const tileMaskBuffer = createInputBuffer(device, new Uint32Array(MAX_TILE_COUNT));
  const gateMaskBuffer = createOutputBuffer(device, ROW_CAPACITY);
  const summaries = {
    live: createOutputBuffer(device, summaryLength),
    ones: createOutputBuffer(device, summaryLength),
    gated: createOutputBuffer(device, summaryLength)
  };
  const graph = new GPUCommandGraph(device, {id: 'composition-stats'});
  const views = mirror.arena.importToGraph(graph);
  const values = views.columns.values as GraphDataView<'float32'>;
  const gate = createCompactOutput(device, graph, 'gate');
  const gateMask = importGraphBuffer(graph, 'gate-mask', gateMaskBuffer, 'uint32', ROW_CAPACITY);
  const statistics = (
    name: keyof typeof summaries,
    mask: GraphDataView<'uint32'>
  ): GPURegionStatistics =>
    new GPURegionStatistics({
      id: `stats-${name}`,
      selection: {kind: 'mask', mask},
      values,
      histogram: {binCount: HISTOGRAM_BIN_COUNT},
      summary: importGraphBuffer(graph, `summary-${name}`, summaries[name], 'uint32', summaryLength)
    });
  graph.add(statistics('live', views.liveMask));
  // Negative control: an all-ones mask is the "automatic domain counted dead rows" failure.
  graph.add(
    statistics('ones', importGraphBuffer(graph, 'ones', onesBuffer, 'uint32', ROW_CAPACITY))
  );
  graph.add(
    new GPUResidentRowSelection({
      id: 'stats-gate',
      liveMask: views.liveMask,
      tileVisibility: {
        rowTileSlots: views.rowTileSlots,
        tileMask: importGraphBuffer(graph, 'tile-mask', tileMaskBuffer, 'uint32', MAX_TILE_COUNT)
      },
      output: gate.output,
      outputMask: gateMask
    })
  );
  graph.add(statistics('gated', gateMask));
  const compiled = graph.compile();
  expect(compileSpy).toHaveBeenCalledTimes(1);
  const nodeCount = compiled.stats.nodeOrder.length;
  console.log(`[residency-arena-composition] region-statistics graph: ${nodeCount} nodes`);

  const readSummary = async (buffer: Buffer) =>
    decodeGPURegionStatistics(await buffer.readAsync(0, summaryLength * 4));
  let sawPoisonVisibleWithoutMask = false;
  for (let state = 0; state < STATE_COUNT; state++) {
    mirror.churn(random, makeTile);
    // Leave 1e6 behind in dead rows: insert a poison tile, then evict it. No deadValue, so it stays.
    if (!mirror.arena.allocator.canInsert(150)) {
      mirror.evict(mirror.getKeys()[0]);
    }
    const poisonKey = mirror.insert(makePoisonTile(150));
    expect(poisonKey).toBeDefined();
    mirror.evict(poisonKey as string);
    const tileMask = Uint32Array.from({length: MAX_TILE_COUNT}, () => (random() < 0.6 ? 1 : 0));
    tileMaskBuffer.write(tileMask);
    submitGraph(device, compiled, undefined);

    const cpuValues = mirror.columns.values as Float32Array;
    const liveRows = getLiveRows(mirror);
    const slots = mirror.arena.allocator.getRowTileSlots();
    const gatedRows = liveRows.filter(row => tileMask[slots[row]]);

    const liveSummary = await readSummary(summaries.live);
    expectStatistics(liveSummary, liveRows, cpuValues);
    expect(liveSummary.selectedCount).toBe(mirror.arena.liveRowCount);
    expect(liveSummary.maximum).toBeLessThan(POISON_VALUE);
    expectStatistics(await readSummary(summaries.gated), gatedRows, cpuValues);

    const onesSummary = await readSummary(summaries.ones);
    expect(onesSummary.selectedCount).toBe(ROW_CAPACITY);
    expect(onesSummary.maximum).toBe(POISON_VALUE);
    expect(onesSummary.histogram[HISTOGRAM_BIN_COUNT - 1]).toBeGreaterThan(0);
    sawPoisonVisibleWithoutMask ||= onesSummary.maximum === POISON_VALUE;
  }
  expect(sawPoisonVisibleWithoutMask).toBe(true);
  expect(compileSpy).toHaveBeenCalledTimes(1);
  expect(compiled.stats.nodeOrder.length).toBe(nodeCount);

  compiled.destroy();
  destroyAll([mirror, onesBuffer, tileMaskBuffer, gateMaskBuffer]);
  destroyAll([...Object.values(summaries), ...Object.values(gate).filter(isBuffer)]);
});
