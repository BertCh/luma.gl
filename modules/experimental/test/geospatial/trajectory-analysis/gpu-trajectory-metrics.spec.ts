// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPUTrajectoryMetricsParameterValues,
  GPUTrajectoryMetrics,
  GPU_TRAJECTORY_METRICS_PARAMETER_LENGTH,
  type GPUTrajectoryStopParameters
} from '../../../src/geospatial/trajectory-analysis';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  computeTrajectoryOracle,
  createTracksFromRows,
  generateTrajectoryTracks,
  type TrajectoryTracks
} from './trajectory-metrics-oracle';

const SENTINEL = 0xffffffff;
const GARBAGE = 0x7f7f7f7f;

type StopRows = {
  count: number;
  overflow: number;
  total: number;
  ids: number[];
  startRows: number[];
  endRows: number[];
  durations: number[];
  centroids: number[];
};

type MetricsResult = {
  trackLengths: number[];
  trackDurations: number[];
  averageSpeeds: number[];
  maximumSpeeds: number[];
  trackStopCounts: number[];
  stops: StopRows;
};

/** Builds one compiled graph for `tracks` that can be encoded repeatedly with new parameters. */
function createHarness(device: Device, tracks: TrajectoryTracks, capacity: number) {
  const rowCount = tracks.timestamps.length;
  const trackCount = tracks.trackOffsets.length - 1;
  const buffers: Buffer[] = [];
  const createGarbageBuffer = (length: number): Buffer => {
    const buffer = createOutputBuffer(device, length);
    buffer.write(new Uint32Array(Math.max(length, 1)).fill(GARBAGE));
    buffers.push(buffer);
    return buffer;
  };
  const positionsBuffer = createInputBuffer(
    device,
    tracks.positions.length > 0 ? tracks.positions : new Float32Array(2)
  );
  const timestampsBuffer = createInputBuffer(
    device,
    rowCount > 0 ? tracks.timestamps : new Float32Array(1)
  );
  const offsetsBuffer = createInputBuffer(device, Uint32Array.from(tracks.trackOffsets));
  buffers.push(positionsBuffer, timestampsBuffer, offsetsBuffer);
  const parameters = new GPUParameterBuffer(device, {
    id: 'trajectory-parameters',
    format: 'float32',
    length: GPU_TRAJECTORY_METRICS_PARAMETER_LENGTH
  });
  const outputs = {
    lengths: createGarbageBuffer(trackCount),
    durations: createGarbageBuffer(trackCount),
    averages: createGarbageBuffer(trackCount),
    maxima: createGarbageBuffer(trackCount),
    stopCounts: createGarbageBuffer(trackCount),
    stopIds: createGarbageBuffer(capacity),
    stopStarts: createGarbageBuffer(capacity),
    stopEnds: createGarbageBuffer(capacity),
    stopDurations: createGarbageBuffer(capacity),
    stopCentroids: createGarbageBuffer(2 * capacity),
    count: createGarbageBuffer(1),
    overflow: createGarbageBuffer(1),
    total: createGarbageBuffer(1)
  };
  const graph = new GPUCommandGraph(device, {id: 'trajectory-metrics-test'});
  const importView = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    id: string,
    buffer: Buffer,
    format: Format,
    length: number
  ) => importGraphBuffer(graph, id, buffer, format, length);
  graph.add(
    new GPUTrajectoryMetrics({
      positions: importView('positions', positionsBuffer, 'float32x2', rowCount),
      timestamps: importView('timestamps', timestampsBuffer, 'float32', rowCount),
      trackOffsets: importView('offsets', offsetsBuffer, 'uint32', trackCount + 1),
      parameters: parameters.importToGraph(graph),
      trackLengths: importView('lengths', outputs.lengths, 'float32', trackCount),
      trackDurations: importView('durations', outputs.durations, 'float32', trackCount),
      averageSpeeds: importView('averages', outputs.averages, 'float32', trackCount),
      maximumSpeeds: importView('maxima', outputs.maxima, 'float32', trackCount),
      trackStopCounts: importView('stop-counts', outputs.stopCounts, 'uint32', trackCount),
      stops: {
        output: {
          ids: importView('stop-ids', outputs.stopIds, 'uint32', capacity),
          count: importView('count', outputs.count, 'uint32', 1),
          overflow: importView('overflow', outputs.overflow, 'uint32', 1),
          totalCount: importView('total', outputs.total, 'uint32', 1)
        },
        startRows: importView('stop-starts', outputs.stopStarts, 'uint32', capacity),
        endRows: importView('stop-ends', outputs.stopEnds, 'uint32', capacity),
        durations: importView('stop-durations', outputs.stopDurations, 'float32', capacity),
        centroids: importView('stop-centroids', outputs.stopCentroids, 'float32x2', capacity)
      }
    })
  );
  const compiled = graph.compile();
  return {
    async run(stopParameters: GPUTrajectoryStopParameters): Promise<MetricsResult> {
      parameters.write(getGPUTrajectoryMetricsParameterValues(stopParameters));
      submitGraph(device, compiled, undefined);
      const [count] = await readUint32(outputs.count, 1);
      const [overflow] = await readUint32(outputs.overflow, 1);
      const [total] = await readUint32(outputs.total, 1);
      return {
        trackLengths: await readFloat32(outputs.lengths, trackCount),
        trackDurations: await readFloat32(outputs.durations, trackCount),
        averageSpeeds: await readFloat32(outputs.averages, trackCount),
        maximumSpeeds: await readFloat32(outputs.maxima, trackCount),
        trackStopCounts: await readUint32(outputs.stopCounts, trackCount),
        stops: {
          count,
          overflow,
          total,
          ids: await readUint32(outputs.stopIds, capacity),
          startRows: await readUint32(outputs.stopStarts, capacity),
          endRows: await readUint32(outputs.stopEnds, capacity),
          durations: await readFloat32(outputs.stopDurations, capacity),
          centroids: await readFloat32(outputs.stopCentroids, 2 * capacity)
        }
      };
    },
    destroy(): void {
      compiled.destroy();
      parameters.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
}

function expectCloseRelative(actual: number[], expected: number[], tolerance: number = 1e-4): void {
  expect(actual.length).toBe(expected.length);
  for (const [index, value] of expected.entries()) {
    expect(Math.abs(actual[index] - value)).toBeLessThanOrEqual(
      tolerance * Math.max(1, Math.abs(value))
    );
  }
}

/** Compares a GPU result with the oracle for the given parameters and stop capacity. */
function expectMatchesOracle(
  result: MetricsResult,
  tracks: TrajectoryTracks,
  parameters: GPUTrajectoryStopParameters,
  capacity: number
): void {
  const expected = computeTrajectoryOracle(
    tracks,
    parameters.stopSpeedThreshold,
    parameters.stopMinimumDuration
  );
  expectCloseRelative(result.trackLengths, expected.trackLengths);
  expectCloseRelative(result.maximumSpeeds, expected.maximumSpeeds);
  expectCloseRelative(result.averageSpeeds, expected.averageSpeeds);
  expect(result.trackDurations).toEqual(expected.trackDurations);
  expect(result.trackStopCounts).toEqual(expected.trackStopCounts);

  const count = Math.min(expected.stops.length, capacity);
  const {stops} = result;
  expect(stops.total).toBe(expected.stops.length);
  expect(stops.count).toBe(count);
  expect(stops.overflow).toBe(expected.stops.length > capacity ? 1 : 0);
  for (let index = 0; index < capacity; index++) {
    const stop = expected.stops[index];
    if (index >= count) {
      expect(stops.ids[index]).toBe(SENTINEL);
      expect(stops.startRows[index]).toBe(SENTINEL);
      expect(stops.endRows[index]).toBe(SENTINEL);
      expect(stops.durations[index]).toBe(0);
      expect(stops.centroids.slice(2 * index, 2 * index + 2)).toEqual([0, 0]);
      continue;
    }
    expect(stops.ids[index]).toBe(stop.track);
    expect(stops.startRows[index]).toBe(stop.startRow);
    expect(stops.endRows[index]).toBe(stop.endRow);
    expect(stops.durations[index]).toBe(stop.duration);
    for (const axis of [0, 1]) {
      expect(Math.abs(stops.centroids[2 * index + axis] - stop.centroid[axis])).toBeLessThanOrEqual(
        0.02 + 1e-4 * Math.abs(stop.centroid[axis])
      );
    }
  }
}

const PARAMETERS: GPUTrajectoryStopParameters = {stopSpeedThreshold: 1, stopMinimumDuration: 4};

it('GPUTrajectoryMetrics matches the oracle on randomized tracks and new parameters', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  for (const seed of [1, 2, 3]) {
    const tracks = generateTrajectoryTracks(seed, 40, 5, 200, PARAMETERS.stopSpeedThreshold);
    const parameterSets = [
      PARAMETERS,
      {stopSpeedThreshold: 0.5, stopMinimumDuration: 7},
      {stopSpeedThreshold: 2, stopMinimumDuration: 2}
    ];
    const stopCounts = parameterSets.map(
      set =>
        computeTrajectoryOracle(tracks, set.stopSpeedThreshold, set.stopMinimumDuration).stops
          .length
    );
    expect(stopCounts[0]).toBeGreaterThan(10);
    expect(new Set(stopCounts).size).toBe(3);
    const capacity = Math.max(...stopCounts) + 8;
    const harness = createHarness(device, tracks, capacity);
    // Same compiled graph, different thresholds: no recompilation.
    for (const set of [...parameterSets, PARAMETERS]) {
      expectMatchesOracle(await harness.run(set), tracks, set, capacity);
    }
    harness.destroy();
  }
  device.destroy?.();
});

it('GPUTrajectoryMetrics handles dwell boundaries and minimum duration equality', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const parameters = {stopSpeedThreshold: 1, stopMinimumDuration: 3};
  const tracks = createTracksFromRows([
    // Track 0: dwell at the start (rows 0..3, duration 3), then moves, then dwell at the end
    // (rows 5..8, duration exactly 3).
    [
      [0, 0, 0],
      [0, 0, 1],
      [0.1, 0, 2],
      [0.1, 0.1, 3],
      [20, 0, 4],
      [40, 0, 5],
      [40, 0, 6],
      [40, 0.1, 7],
      [40.1, 0.1, 8]
    ],
    // Track 1 begins with a dwell right after track 0's final dwell: no merge across tracks.
    [
      [40.1, 0.1, 9],
      [40.1, 0.1, 10],
      [40.1, 0.2, 11],
      [40.1, 0.2, 12],
      [90, 0.2, 13]
    ],
    // Track 2: a dwell shorter than the minimum duration (2) does not qualify.
    [
      [0, 0, 0],
      [0, 0, 1],
      [0, 0, 2],
      [30, 0, 3]
    ],
    // Track 3: duplicate timestamps. Zero distance with deltaTime 0 is slow; nonzero distance
    // with deltaTime 0 is not slow and creates no speed spike.
    [
      [5, 5, 0],
      [5, 5, 0],
      [5, 5, 1],
      [5, 5, 2],
      [5, 5, 3],
      [9, 5, 3],
      [9, 5, 4]
    ]
  ]);
  const expected = computeTrajectoryOracle(tracks, 1, 3);
  expect(expected.stops.map(stop => [stop.track, stop.startRow, stop.endRow])).toEqual([
    [0, 0, 3],
    [0, 5, 8],
    [1, 9, 12],
    [3, 18, 22]
  ]);
  expect(expected.maximumSpeeds[3]).toBe(0);
  const harness = createHarness(device, tracks, 6);
  const result = await harness.run(parameters);
  expectMatchesOracle(result, tracks, parameters, 6);
  expect(result.stops.count).toBe(4);
  expect(result.stops.ids.slice(0, 4)).toEqual([0, 0, 1, 3]);
  harness.destroy();
});

it('GPUTrajectoryMetrics reports capacity overflow with unclamped totals', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const tracks = generateTrajectoryTracks(11, 40, 20, 100, 1);
  const expected = computeTrajectoryOracle(tracks, 1, 4);
  expect(expected.stops.length).toBeGreaterThan(5);
  const capacity = 3;
  const harness = createHarness(device, tracks, capacity);
  const result = await harness.run(PARAMETERS);
  expectMatchesOracle(result, tracks, PARAMETERS, capacity);
  expect(result.stops.count).toBe(capacity);
  expect(result.stops.overflow).toBe(1);
  expect(result.stops.total).toBe(expected.stops.length);
  expect(result.trackStopCounts.reduce((sum, value) => sum + value, 0)).toBe(expected.stops.length);
  harness.destroy();
});

it('GPUTrajectoryMetrics handles empty and single-row input', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const emptyTracks: TrajectoryTracks = {
    positions: new Float32Array(0),
    timestamps: new Float32Array(0),
    trackOffsets: [0, 0]
  };
  const emptyHarness = createHarness(device, emptyTracks, 4);
  const empty = await emptyHarness.run(PARAMETERS);
  expectMatchesOracle(empty, emptyTracks, PARAMETERS, 4);
  expect(empty.stops.count).toBe(0);
  expect(empty.trackLengths).toEqual([0]);
  emptyHarness.destroy();

  const singleTracks = createTracksFromRows([[[3, 4, 5]]]);
  const singleHarness = createHarness(device, singleTracks, 4);
  const single = await singleHarness.run(PARAMETERS);
  expectMatchesOracle(single, singleTracks, PARAMETERS, 4);
  expect(single.stops.count).toBe(0);
  expect(single.trackDurations).toEqual([0]);
  singleHarness.destroy();

  // Only empty tracks over nonempty rows.
  const ignoredTracks = createTracksFromRows([[], []], 3);
  const ignoredHarness = createHarness(device, ignoredTracks, 2);
  expectMatchesOracle(await ignoredHarness.run(PARAMETERS), ignoredTracks, PARAMETERS, 2);
  ignoredHarness.destroy();
});

it('GPUTrajectoryMetrics ignores rows before the first track offset', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const generated = generateTrajectoryTracks(5, 12, 5, 40, 1);
  const leadingRowCount = 6;
  const tracks: TrajectoryTracks = {
    // Leading rows are identical zero-distance samples that would form a long dwell.
    positions: Float32Array.from([
      ...new Array(2 * leadingRowCount).fill(1),
      ...generated.positions
    ]),
    timestamps: Float32Array.from([
      ...Array.from({length: leadingRowCount}, (_, row) => row * 10),
      ...generated.timestamps
    ]),
    trackOffsets: generated.trackOffsets.map(offset => offset + leadingRowCount)
  };
  const capacity = 64;
  const harness = createHarness(device, tracks, capacity);
  const result = await harness.run(PARAMETERS);
  expectMatchesOracle(result, tracks, PARAMETERS, capacity);
  expect(
    result.stops.startRows.slice(0, result.stops.count).every(row => row >= leadingRowCount)
  ).toBe(true);
  harness.destroy();
});

it('GPUTrajectoryMetrics supports minimal output subsets', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const tracks = generateTrajectoryTracks(21, 20, 5, 60, 1);
  const rowCount = tracks.timestamps.length;
  const trackCount = tracks.trackOffsets.length - 1;
  const expected = computeTrajectoryOracle(tracks, 1, 4);
  const positions = createInputBuffer(device, tracks.positions);
  const timestamps = createInputBuffer(device, tracks.timestamps);
  const offsets = createInputBuffer(device, Uint32Array.from(tracks.trackOffsets));
  const parameters = new GPUParameterBuffer(device, {
    id: 'subset-parameters',
    format: 'float32',
    length: GPU_TRAJECTORY_METRICS_PARAMETER_LENGTH,
    values: getGPUTrajectoryMetricsParameterValues(PARAMETERS)
  });
  const averages = createOutputBuffer(device, trackCount);
  const stopCounts = createOutputBuffer(device, trackCount);
  const stopIds = createOutputBuffer(device, 4);
  const count = createOutputBuffer(device, 1);
  const overflow = createOutputBuffer(device, 1);
  const outputs = [averages, stopCounts, stopIds, count, overflow];

  // Average speeds only (lengths are internal scratch).
  const averageGraph = new GPUCommandGraph(device, {id: 'subset-average'});
  averageGraph.add(
    new GPUTrajectoryMetrics({
      positions: importGraphBuffer(averageGraph, 'p', positions, 'float32x2', rowCount),
      timestamps: importGraphBuffer(averageGraph, 't', timestamps, 'float32', rowCount),
      trackOffsets: importGraphBuffer(averageGraph, 'o', offsets, 'uint32', trackCount + 1),
      averageSpeeds: importGraphBuffer(averageGraph, 'a', averages, 'float32', trackCount)
    })
  );
  const averageCompiled = averageGraph.compile();
  submitGraph(device, averageCompiled, undefined);
  const gotAverages = await readFloat32(averages, trackCount);
  expectCloseRelative(gotAverages, expected.averageSpeeds);
  averageCompiled.destroy();

  // Per-track stop counts only.
  const countGraph = new GPUCommandGraph(device, {id: 'subset-count'});
  countGraph.add(
    new GPUTrajectoryMetrics({
      positions: importGraphBuffer(countGraph, 'p', positions, 'float32x2', rowCount),
      timestamps: importGraphBuffer(countGraph, 't', timestamps, 'float32', rowCount),
      trackOffsets: importGraphBuffer(countGraph, 'o', offsets, 'uint32', trackCount + 1),
      parameters: parameters.importToGraph(countGraph),
      trackStopCounts: importGraphBuffer(countGraph, 's', stopCounts, 'uint32', trackCount)
    })
  );
  const countCompiled = countGraph.compile();
  submitGraph(device, countCompiled, undefined);
  expect(await readUint32(stopCounts, trackCount)).toEqual(expected.trackStopCounts);
  countCompiled.destroy();

  // Stop track IDs only, with a small capacity.
  const idGraph = new GPUCommandGraph(device, {id: 'subset-ids'});
  idGraph.add(
    new GPUTrajectoryMetrics({
      positions: importGraphBuffer(idGraph, 'p', positions, 'float32x2', rowCount),
      timestamps: importGraphBuffer(idGraph, 't', timestamps, 'float32', rowCount),
      trackOffsets: importGraphBuffer(idGraph, 'o', offsets, 'uint32', trackCount + 1),
      parameters: parameters.importToGraph(idGraph),
      stops: {
        output: {
          ids: importGraphBuffer(idGraph, 'i', stopIds, 'uint32', 4),
          count: importGraphBuffer(idGraph, 'c', count, 'uint32', 1),
          overflow: importGraphBuffer(idGraph, 'f', overflow, 'uint32', 1)
        }
      }
    })
  );
  const idCompiled = idGraph.compile();
  submitGraph(device, idCompiled, undefined);
  const expectedCount = Math.min(4, expected.stops.length);
  expect(await readUint32(count, 1)).toEqual([expectedCount]);
  expect(await readUint32(overflow, 1)).toEqual([expected.stops.length > 4 ? 1 : 0]);
  expect(await readUint32(stopIds, expectedCount)).toEqual(
    expected.stops.slice(0, expectedCount).map(stop => stop.track)
  );
  idCompiled.destroy();

  parameters.destroy();
  for (const buffer of [positions, timestamps, offsets, ...outputs]) {
    buffer.destroy();
  }
});
