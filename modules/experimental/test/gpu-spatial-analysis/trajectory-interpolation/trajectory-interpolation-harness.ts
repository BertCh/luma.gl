// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {getInt64TimeWords} from '../../../src/gpu-dataframe/time-window-filter/time-words';
import {
  getGPUTrajectoryPlayheadParameterValues,
  getGPUTrajectoryPlayheadWordParameterValues,
  GPUTrajectoryPlayhead,
  GPUTrajectoryResample,
  GPU_TRAJECTORY_PLAYHEAD_PARAMETER_LENGTH,
  type GPUTrajectoryResampleSpacing
} from '../../../src/gpu-spatial-analysis/trajectory-interpolation';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import type {
  OracleTracks,
  PlayheadOracleResult,
  ResampleOracleResult
} from './trajectory-interpolation-oracle';

/** GPU playhead readback plus the compact active list. */
export type PlayheadGPUResult = PlayheadOracleResult & {
  activeCount: number;
  activeOverflow: number;
  activeTotal: number;
  drawInstanceCount: number;
};

/** One compiled playhead graph reused across playheads. */
export type PlayheadFixture = {
  run(playhead: number | bigint, maxGap?: number): Promise<PlayheadGPUResult>;
  /** Number of times the contributor built its command nodes (1 after compile). */
  getBuildCount(): number;
  destroy(): void;
};

function createBufferTracker(device: Device) {
  const buffers: Buffer[] = [];
  return {
    buffers,
    input(values: Float32Array | Uint32Array): Buffer {
      const buffer = createInputBuffer(device, values);
      buffers.push(buffer);
      return buffer;
    },
    output(length: number): Buffer {
      const buffer = createOutputBuffer(device, length);
      // Poison outputs so unwritten words are caught.
      buffer.write(new Uint32Array(Math.max(length, 1)).fill(0x7f7f7f7f));
      buffers.push(buffer);
      return buffer;
    }
  };
}

function getTimestampData(tracks: OracleTracks): Float32Array | Uint32Array {
  return tracks.times.kind === 'words'
    ? getInt64TimeWords(BigInt64Array.from(tracks.times.values))
    : tracks.times.values;
}

/** Builds and compiles a playhead graph with every output over `tracks`. */
export function createPlayheadFixture(
  device: Device,
  tracks: OracleTracks,
  capacity: number
): PlayheadFixture {
  const isWordMode = tracks.times.kind === 'words';
  const rowCount = tracks.positions.length / 2;
  const trackCount = tracks.trackOffsets.length - 1;
  const graph = new GPUCommandGraph(device, {id: 'trajectory-playhead-test'});
  const tracker = createBufferTracker(device);
  const parameterBuffer = isWordMode
    ? new GPUParameterBuffer(device, {
        id: 'playhead',
        format: 'uint32',
        length: GPU_TRAJECTORY_PLAYHEAD_PARAMETER_LENGTH
      })
    : new GPUParameterBuffer(device, {
        id: 'playhead',
        format: 'float32',
        length: GPU_TRAJECTORY_PLAYHEAD_PARAMETER_LENGTH
      });
  const out = {
    positions: tracker.output(2 * trackCount),
    elevations: tracker.output(trackCount),
    headings: tracker.output(trackCount),
    speeds: tracker.output(trackCount),
    status: tracker.output(trackCount),
    segmentRows: tracker.output(trackCount),
    segmentFractions: tracker.output(trackCount),
    ids: tracker.output(capacity),
    count: tracker.output(1),
    overflow: tracker.output(1),
    total: tracker.output(1),
    draw: tracker.output(1)
  };
  const timestampBuffer = tracker.input(getTimestampData(tracks));
  const contributor = new GPUTrajectoryPlayhead({
    id: 'playhead',
    positions: importGraphBuffer(
      graph,
      'positions',
      tracker.input(tracks.positions),
      'float32x2',
      rowCount
    ),
    elevations: tracks.elevations
      ? importGraphBuffer(
          graph,
          'elevations',
          tracker.input(tracks.elevations),
          'float32',
          rowCount
        )
      : undefined,
    timestamps: isWordMode
      ? importGraphBuffer(graph, 'timestamps', timestampBuffer, 'uint32x2', rowCount)
      : importGraphBuffer(graph, 'timestamps', timestampBuffer, 'float32', rowCount),
    trackOffsets: importGraphBuffer(
      graph,
      'offsets',
      tracker.input(Uint32Array.from(tracks.trackOffsets)),
      'uint32',
      trackCount + 1
    ),
    parameters: parameterBuffer.importToGraph(graph),
    currentPositions: importGraphBuffer(
      graph,
      'o-positions',
      out.positions,
      'float32x2',
      trackCount
    ),
    currentElevations: tracks.elevations
      ? importGraphBuffer(graph, 'o-elevations', out.elevations, 'float32', trackCount)
      : undefined,
    headings: importGraphBuffer(graph, 'o-headings', out.headings, 'float32', trackCount),
    speeds: importGraphBuffer(graph, 'o-speeds', out.speeds, 'float32', trackCount),
    status: importGraphBuffer(graph, 'o-status', out.status, 'uint32', trackCount),
    segmentRows: importGraphBuffer(graph, 'o-segment-rows', out.segmentRows, 'uint32', trackCount),
    segmentFractions: importGraphBuffer(
      graph,
      'o-segment-fractions',
      out.segmentFractions,
      'float32',
      trackCount
    ),
    activeTracks: {
      ids: importGraphBuffer(graph, 'o-ids', out.ids, 'uint32', capacity),
      count: importGraphBuffer(graph, 'o-count', out.count, 'uint32', 1),
      overflow: importGraphBuffer(graph, 'o-overflow', out.overflow, 'uint32', 1),
      requiredCount: importGraphBuffer(graph, 'o-total', out.total, 'uint32', 1)
    },
    drawInstanceCount: importGraphBuffer(graph, 'o-draw', out.draw, 'uint32', 1)
  });
  let buildCount = 0;
  const getCommandNodes = contributor.getCommandNodes.bind(contributor);
  contributor.getCommandNodes = currentGraph => {
    buildCount++;
    return getCommandNodes(currentGraph);
  };
  graph.add(contributor);
  const compiled = graph.compile();
  return {
    async run(playhead, maxGap = 0) {
      if (isWordMode) {
        parameterBuffer.write(getGPUTrajectoryPlayheadWordParameterValues({playhead, maxGap}));
      } else {
        parameterBuffer.write(getGPUTrajectoryPlayheadParameterValues({playhead, maxGap}));
      }
      submitGraph(device, compiled, undefined);
      const [activeCount] = await readUint32(out.count, 1);
      const [activeOverflow] = await readUint32(out.overflow, 1);
      const [activeTotal] = await readUint32(out.total, 1);
      const [drawInstanceCount] = await readUint32(out.draw, 1);
      return {
        positions: await readFloat32(out.positions, 2 * trackCount),
        elevations: tracks.elevations
          ? await readFloat32(out.elevations, trackCount)
          : new Array(trackCount).fill(0),
        headings: await readFloat32(out.headings, trackCount),
        speeds: await readFloat32(out.speeds, trackCount),
        status: await readUint32(out.status, trackCount),
        segmentRows: await readUint32(out.segmentRows, trackCount),
        segmentFractions: await readFloat32(out.segmentFractions, trackCount),
        activeTracks: await readUint32(out.ids, activeCount),
        activeCount,
        activeOverflow,
        activeTotal,
        drawInstanceCount
      };
    },
    getBuildCount: () => buildCount,
    destroy() {
      compiled.destroy();
      parameterBuffer.destroy();
      for (const buffer of tracker.buffers) {
        buffer.destroy();
      }
    }
  };
}

/** Runs one resample graph and reads every output. */
export async function runResample(
  device: Device,
  tracks: OracleTracks,
  sampleCount: number,
  spacing: GPUTrajectoryResampleSpacing
): Promise<ResampleOracleResult> {
  const isWordMode = tracks.times.kind === 'words';
  const rowCount = tracks.positions.length / 2;
  const trackCount = tracks.trackOffsets.length - 1;
  const outputLength = trackCount * sampleCount;
  const graph = new GPUCommandGraph(device, {id: 'trajectory-resample-test'});
  const tracker = createBufferTracker(device);
  const out = {
    samples: tracker.output(2 * outputLength),
    elevations: tracker.output(outputLength),
    times: tracker.output(outputLength)
  };
  const timestampBuffer = tracker.input(getTimestampData(tracks));
  graph.add(
    new GPUTrajectoryResample({
      id: 'resample',
      positions: importGraphBuffer(
        graph,
        'positions',
        tracker.input(tracks.positions),
        'float32x2',
        rowCount
      ),
      elevations: tracks.elevations
        ? importGraphBuffer(
            graph,
            'elevations',
            tracker.input(tracks.elevations),
            'float32',
            rowCount
          )
        : undefined,
      timestamps: isWordMode
        ? importGraphBuffer(graph, 'timestamps', timestampBuffer, 'uint32x2', rowCount)
        : importGraphBuffer(graph, 'timestamps', timestampBuffer, 'float32', rowCount),
      trackOffsets: importGraphBuffer(
        graph,
        'offsets',
        tracker.input(Uint32Array.from(tracks.trackOffsets)),
        'uint32',
        trackCount + 1
      ),
      sampleCount,
      spacing,
      samples: importGraphBuffer(graph, 'o-samples', out.samples, 'float32x2', outputLength),
      sampleElevations: tracks.elevations
        ? importGraphBuffer(graph, 'o-elevations', out.elevations, 'float32', outputLength)
        : undefined,
      sampleTimes: importGraphBuffer(graph, 'o-times', out.times, 'float32', outputLength)
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result: ResampleOracleResult = {
    samples: await readFloat32(out.samples, 2 * outputLength),
    sampleElevations: tracks.elevations
      ? await readFloat32(out.elevations, outputLength)
      : new Array(outputLength).fill(0),
    sampleTimes: await readFloat32(out.times, outputLength)
  };
  compiled.destroy();
  for (const buffer of tracker.buffers) {
    buffer.destroy();
  }
  return result;
}

/** Expects `actual` within `absolute + relative * |expected|` of `expected`, element-wise. */
export function expectClose(
  actual: readonly number[],
  expected: readonly number[],
  absolute: number,
  relative: number,
  label: string
): void {
  expect(actual.length, label).toBe(expected.length);
  for (const [index, value] of expected.entries()) {
    const tolerance = absolute + relative * Math.abs(value);
    if (!(Math.abs(actual[index] - value) <= tolerance)) {
      expect.fail(`${label}[${index}]: ${actual[index]} != ${value} (tolerance ${tolerance})`);
    }
  }
}

/** Checks a GPU playhead result against the oracle: exact integers, tight float tolerances. */
export function expectPlayheadParity(
  actual: PlayheadGPUResult,
  expected: PlayheadOracleResult,
  capacity: number,
  label: string
): void {
  expect(actual.status, `${label} status`).toEqual(expected.status);
  expect(actual.segmentRows, `${label} segmentRows`).toEqual(expected.segmentRows);
  expectClose(actual.segmentFractions, expected.segmentFractions, 1e-6, 1e-6, `${label} fractions`);
  expectClose(actual.positions, expected.positions, 1e-4, 1e-6, `${label} positions`);
  expectClose(actual.elevations, expected.elevations, 1e-4, 1e-6, `${label} elevations`);
  expectClose(actual.headings, expected.headings, 1e-5, 1e-6, `${label} headings`);
  expectClose(actual.speeds, expected.speeds, 1e-5, 1e-5, `${label} speeds`);
  const count = Math.min(expected.activeTracks.length, capacity);
  expect(actual.activeTotal, `${label} total`).toBe(expected.activeTracks.length);
  expect(actual.activeCount, `${label} count`).toBe(count);
  expect(actual.drawInstanceCount, `${label} draw count`).toBe(count);
  expect(actual.activeOverflow, `${label} overflow`).toBe(
    expected.activeTracks.length > capacity ? 1 : 0
  );
  expect(actual.activeTracks, `${label} active ids`).toEqual(
    expected.activeTracks.slice(0, capacity)
  );
}
