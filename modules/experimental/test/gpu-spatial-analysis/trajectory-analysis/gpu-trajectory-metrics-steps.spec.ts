// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPUTrajectoryMetricsParameterValues,
  GPUTrajectoryMetrics,
  GPU_TRAJECTORY_METRICS_PARAMETER_LENGTH
} from '../../../src/gpu-spatial-analysis/trajectory-analysis';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {generateTrajectoryTracks} from './trajectory-metrics-oracle';

const CAPACITY = 64;

it('GPUTrajectoryMetrics writes per-step speed, heading, acceleration and the draw count', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // Leading rows are outside every track and must stay zero.
  const tracks = generateTrajectoryTracks(21, 30, 3, 40, 1);
  const rowCount = tracks.timestamps.length;
  const trackCount = tracks.trackOffsets.length - 1;
  const positionsBuffer = createInputBuffer(device, tracks.positions);
  const timestampsBuffer = createInputBuffer(device, tracks.timestamps);
  const offsetsBuffer = createInputBuffer(device, Uint32Array.from(tracks.trackOffsets));
  const parameters = new GPUParameterBuffer(device, {
    id: 'trajectory-steps-parameters',
    format: 'float32',
    length: GPU_TRAJECTORY_METRICS_PARAMETER_LENGTH,
    values: getGPUTrajectoryMetricsParameterValues({stopSpeedThreshold: 1, stopMinimumDuration: 2})
  });
  const outputs = {
    speeds: createOutputBuffer(device, rowCount),
    headings: createOutputBuffer(device, rowCount),
    accelerations: createOutputBuffer(device, rowCount),
    ids: createOutputBuffer(device, CAPACITY),
    count: createOutputBuffer(device, 1),
    overflow: createOutputBuffer(device, 1),
    draw: createOutputBuffer(device, 1),
    stopCounts: createOutputBuffer(device, trackCount)
  };
  for (const buffer of Object.values(outputs)) {
    buffer.write(new Uint32Array(Math.max(1, buffer.byteLength / 4)).fill(0x7f7f7f7f));
  }
  const graph = new GPUCommandGraph<void>(device, {id: 'trajectory-steps'});
  graph.add(
    new GPUTrajectoryMetrics({
      positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', rowCount),
      timestamps: importGraphBuffer(graph, 'timestamps', timestampsBuffer, 'float32', rowCount),
      trackOffsets: importGraphBuffer(graph, 'offsets', offsetsBuffer, 'uint32', trackCount + 1),
      parameters: parameters.importToGraph(graph),
      stepSpeeds: importGraphBuffer(graph, 'speeds', outputs.speeds, 'float32', rowCount),
      stepHeadings: importGraphBuffer(graph, 'headings', outputs.headings, 'float32', rowCount),
      stepAccelerations: importGraphBuffer(
        graph,
        'accelerations',
        outputs.accelerations,
        'float32',
        rowCount
      ),
      trackStopCounts: importGraphBuffer(
        graph,
        'stop-counts',
        outputs.stopCounts,
        'uint32',
        trackCount
      ),
      stops: {
        output: {
          ids: importGraphBuffer(graph, 'ids', outputs.ids, 'uint32', CAPACITY),
          count: importGraphBuffer(graph, 'count', outputs.count, 'uint32', 1),
          overflow: importGraphBuffer(graph, 'overflow', outputs.overflow, 'uint32', 1)
        },
        drawInstanceCount: importGraphBuffer(graph, 'draw', outputs.draw, 'uint32', 1)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);

  const speeds = await readFloat32(outputs.speeds, rowCount);
  const headings = await readFloat32(outputs.headings, rowCount);
  const accelerations = await readFloat32(outputs.accelerations, rowCount);
  const {positions, timestamps} = tracks;
  const stepSpeed = (row: number) => {
    const deltaTime = timestamps[row] - timestamps[row - 1];
    return deltaTime > 0
      ? Math.hypot(
          positions[2 * row] - positions[2 * row - 2],
          positions[2 * row + 1] - positions[2 * row - 1]
        ) / deltaTime
      : 0;
  };
  let checkedAccelerations = 0;
  for (let track = 0; track < trackCount; track++) {
    const start = tracks.trackOffsets[track];
    const end = tracks.trackOffsets[track + 1];
    for (let row = start; row < end; row++) {
      let speed = 0;
      let heading = 0;
      let acceleration = 0;
      if (row > start) {
        speed = stepSpeed(row);
        const deltaX = positions[2 * row] - positions[2 * row - 2];
        const deltaY = positions[2 * row + 1] - positions[2 * row - 1];
        heading = deltaX !== 0 || deltaY !== 0 ? Math.atan2(deltaY, deltaX) : 0;
        const deltaTime = timestamps[row] - timestamps[row - 1];
        if (row - 1 > start && deltaTime > 0) {
          acceleration = (speed - stepSpeed(row - 1)) / deltaTime;
          checkedAccelerations++;
        }
      }
      expect(Math.abs(speeds[row] - speed)).toBeLessThan(1e-3 * Math.max(1, speed));
      const angleDifference = Math.atan2(
        Math.sin(headings[row] - heading),
        Math.cos(headings[row] - heading)
      );
      expect(Math.abs(angleDifference)).toBeLessThan(1e-3);
      expect(Math.abs(accelerations[row] - acceleration)).toBeLessThan(
        2e-3 * Math.max(1, Math.abs(acceleration))
      );
    }
  }
  expect(checkedAccelerations).toBeGreaterThan(100);

  const [count] = await readUint32(outputs.count, 1);
  const [draw] = await readUint32(outputs.draw, 1);
  expect(count).toBeGreaterThan(0);
  expect(draw).toBe(count);

  compiled.destroy();
  parameters.destroy();
  for (const buffer of [
    positionsBuffer,
    timestampsBuffer,
    offsetsBuffer,
    ...Object.values(outputs)
  ]) {
    buffer.destroy();
  }
});
