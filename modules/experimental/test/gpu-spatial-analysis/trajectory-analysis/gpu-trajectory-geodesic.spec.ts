// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPUTrajectoryMetrics} from '../../../src/gpu-spatial-analysis/trajectory-analysis/index';
import {GPUTrajectoryEncounters} from '../../../src/gpu-spatial-analysis/trajectory-encounters/index';
import {
  getGPUTrajectoryPlayheadParameterValues,
  GPUTrajectoryPlayhead,
  GPUTrajectoryResample,
  GPU_TRAJECTORY_PLAYHEAD_PARAMETER_LENGTH
} from '../../../src/gpu-spatial-analysis/trajectory-interpolation/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  GEODESIC_ORACLE_CASES,
  getGreatCircleDistance,
  getInitialBearingDegrees,
  interpolateGreatCircle
} from './geodesic-trajectory-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

const GREAT_CIRCLE = {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const;
const TRACK_DURATION = 100;
const fround = Math.fround;

/** Two-row tracks, one per oracle case, times `0` and `TRACK_DURATION`. */
function createOracleTracks() {
  const positions = new Float32Array(GEODESIC_ORACLE_CASES.length * 4);
  const timestamps = new Float32Array(GEODESIC_ORACLE_CASES.length * 2);
  const trackOffsets = [0];
  GEODESIC_ORACLE_CASES.forEach((testCase, track) => {
    positions.set([...testCase.a, ...testCase.b], track * 4);
    timestamps.set([0, TRACK_DURATION], track * 2);
    trackOffsets.push((track + 1) * 2);
  });
  return {positions, timestamps, trackOffsets};
}

/** The case endpoints after the float32 upload, which is what the kernels see. */
function getUploadedEndpoints(index: number) {
  const {a, b} = GEODESIC_ORACLE_CASES[index];
  return {a: a.map(fround), b: b.map(fround)};
}

function createBuffers(device: Device) {
  const buffers: Buffer[] = [];
  return {
    buffers,
    input(values: Float32Array | Uint32Array) {
      const buffer = createInputBuffer(device, values);
      buffers.push(buffer);
      return buffer;
    },
    output(length: number) {
      const buffer = createOutputBuffer(device, length);
      buffer.write(new Uint32Array(Math.max(length, 1)).fill(0x7f7f7f7f));
      buffers.push(buffer);
      return buffer;
    },
    destroy() {
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
}

it('GPUTrajectoryMetrics measures great-circle meters, speeds and bearings against pyproj', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const tracks = createOracleTracks();
  const rowCount = tracks.timestamps.length;
  const trackCount = tracks.trackOffsets.length - 1;
  const io = createBuffers(device);
  const outputs = {
    lengths: io.output(trackCount),
    maximumSpeeds: io.output(trackCount),
    speeds: io.output(rowCount),
    headings: io.output(rowCount)
  };
  const graph = new GPUCommandGraph<void>(device, {id: 'trajectory-geodesic-metrics'});
  graph.add(
    new GPUTrajectoryMetrics({
      spatialContext: GREAT_CIRCLE,
      positions: importGraphBuffer(
        graph,
        'positions',
        io.input(tracks.positions),
        'float32x2',
        rowCount
      ),
      timestamps: importGraphBuffer(
        graph,
        'timestamps',
        io.input(tracks.timestamps),
        'float32',
        rowCount
      ),
      trackOffsets: importGraphBuffer(
        graph,
        'offsets',
        io.input(Uint32Array.from(tracks.trackOffsets)),
        'uint32',
        trackCount + 1
      ),
      output: {
        trackLengths: importGraphBuffer(graph, 'lengths', outputs.lengths, 'float32', trackCount),
        maximumSpeeds: importGraphBuffer(
          graph,
          'maximum',
          outputs.maximumSpeeds,
          'float32',
          trackCount
        ),
        stepSpeeds: importGraphBuffer(graph, 'speeds', outputs.speeds, 'float32', rowCount),
        stepHeadings: importGraphBuffer(graph, 'headings', outputs.headings, 'float32', rowCount)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const lengths = await readFloat32(outputs.lengths, trackCount);
  const maximumSpeeds = await readFloat32(outputs.maximumSpeeds, trackCount);
  const speeds = await readFloat32(outputs.speeds, rowCount);
  const headings = await readFloat32(outputs.headings, rowCount);
  for (let track = 0; track < trackCount; track++) {
    const {a, b} = getUploadedEndpoints(track);
    const distance = getGreatCircleDistance(a, b);
    const tolerance = 1e-4 * distance + 0.05;
    expect(Math.abs(lengths[track] - distance)).toBeLessThan(tolerance);
    expect(Math.abs(speeds[2 * track + 1] - distance / TRACK_DURATION)).toBeLessThan(
      tolerance / TRACK_DURATION
    );
    expect(Math.abs(maximumSpeeds[track] - distance / TRACK_DURATION)).toBeLessThan(
      tolerance / TRACK_DURATION
    );
    // The first row of every track has no step.
    expect(speeds[2 * track]).toBe(0);
    expect(headings[2 * track]).toBe(0);
    // Compared on the circle.
    // Headings use the planar convention: pi / 2 - bearing, counterclockwise from east.
    const bearing = Math.PI / 2 - (getInitialBearingDegrees(a, b) * Math.PI) / 180;
    const difference = Math.atan2(
      Math.sin(headings[2 * track + 1] - bearing),
      Math.cos(headings[2 * track + 1] - bearing)
    );
    expect(Math.abs(difference)).toBeLessThan(1e-4);
  }
  compiled.destroy();
  io.destroy();
});

it('GPUTrajectoryResample slerps along great circles and keeps longitudes continuous', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const tracks = createOracleTracks();
  const rowCount = tracks.timestamps.length;
  const trackCount = tracks.trackOffsets.length - 1;
  const sampleCount = 5;
  const io = createBuffers(device);
  const samples = io.output(2 * trackCount * sampleCount);
  const graph = new GPUCommandGraph<void>(device, {id: 'trajectory-geodesic-resample'});
  graph.add(
    new GPUTrajectoryResample({
      spatialContext: GREAT_CIRCLE,
      positions: importGraphBuffer(
        graph,
        'positions',
        io.input(tracks.positions),
        'float32x2',
        rowCount
      ),
      timestamps: importGraphBuffer(
        graph,
        'timestamps',
        io.input(tracks.timestamps),
        'float32',
        rowCount
      ),
      trackOffsets: importGraphBuffer(
        graph,
        'offsets',
        io.input(Uint32Array.from(tracks.trackOffsets)),
        'uint32',
        trackCount + 1
      ),
      sampleCount,
      spacing: 'time',
      output: {
        samples: importGraphBuffer(graph, 'samples', samples, 'float32x2', trackCount * sampleCount)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const values = await readFloat32(samples, 2 * trackCount * sampleCount);
  for (let track = 0; track < trackCount; track++) {
    const {a, b} = getUploadedEndpoints(track);
    for (let sample = 0; sample < sampleCount; sample++) {
      const fraction = sample / (sampleCount - 1);
      const row = track * sampleCount + sample;
      let expected: number[];
      if (sample === 0) {
        expected = a;
      } else if (sample === sampleCount - 1) {
        // The end row, shifted by whole turns to continue the start longitude.
        expected = [b[0] - 360 * Math.round((b[0] - a[0]) / 360), b[1]];
      } else {
        expected = interpolateGreatCircle(a, b, fraction);
      }
      // About 50 m: float32 degrees near 180 resolve 1.5e-5 degrees.
      expect(Math.abs(values[2 * row] - expected[0])).toBeLessThan(5e-4);
      expect(Math.abs(values[2 * row + 1] - expected[1])).toBeLessThan(5e-4);
    }
  }
  // The antimeridian track continues past 180 instead of jumping to -179.5.
  const antimeridianTrack = GEODESIC_ORACLE_CASES.findIndex(c => c.name === 'antimeridian');
  expect(values[2 * (antimeridianTrack * sampleCount + sampleCount - 1)]).toBeGreaterThan(180);
  compiled.destroy();
  io.destroy();
});

it('GPUTrajectoryPlayhead interpolates positions, bearing and meters per time unit', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const tracks = createOracleTracks();
  const rowCount = tracks.timestamps.length;
  const trackCount = tracks.trackOffsets.length - 1;
  const io = createBuffers(device);
  const outputs = {
    positions: io.output(2 * trackCount),
    headings: io.output(trackCount),
    speeds: io.output(trackCount)
  };
  const parameters = new GPUParameterBuffer(device, {
    id: 'trajectory-geodesic-playhead-parameters',
    format: 'float32',
    length: GPU_TRAJECTORY_PLAYHEAD_PARAMETER_LENGTH,
    values: getGPUTrajectoryPlayheadParameterValues({playhead: TRACK_DURATION / 2})
  });
  const graph = new GPUCommandGraph<void>(device, {id: 'trajectory-geodesic-playhead'});
  graph.add(
    new GPUTrajectoryPlayhead({
      spatialContext: GREAT_CIRCLE,
      positions: importGraphBuffer(
        graph,
        'positions',
        io.input(tracks.positions),
        'float32x2',
        rowCount
      ),
      timestamps: importGraphBuffer(
        graph,
        'timestamps',
        io.input(tracks.timestamps),
        'float32',
        rowCount
      ),
      trackOffsets: importGraphBuffer(
        graph,
        'offsets',
        io.input(Uint32Array.from(tracks.trackOffsets)),
        'uint32',
        trackCount + 1
      ),
      parameters: parameters.importToGraph(graph),
      output: {
        currentPositions: importGraphBuffer(
          graph,
          'current',
          outputs.positions,
          'float32x2',
          trackCount
        ),
        headings: importGraphBuffer(graph, 'headings', outputs.headings, 'float32', trackCount),
        speeds: importGraphBuffer(graph, 'speeds', outputs.speeds, 'float32', trackCount)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const positions = await readFloat32(outputs.positions, 2 * trackCount);
  const headings = await readFloat32(outputs.headings, trackCount);
  const speeds = await readFloat32(outputs.speeds, trackCount);
  for (let track = 0; track < trackCount; track++) {
    const {a, b} = getUploadedEndpoints(track);
    const [longitude, latitude] = interpolateGreatCircle(a, b, 0.5);
    expect(Math.abs(positions[2 * track] - longitude)).toBeLessThan(5e-4);
    expect(Math.abs(positions[2 * track + 1] - latitude)).toBeLessThan(5e-4);
    const distance = getGreatCircleDistance(a, b);
    expect(Math.abs(speeds[track] - distance / TRACK_DURATION)).toBeLessThan(
      (1e-4 * distance + 0.05) / TRACK_DURATION
    );
    // Headings use the planar convention: pi / 2 - bearing, counterclockwise from east.
    const bearing = Math.PI / 2 - (getInitialBearingDegrees(a, b) * Math.PI) / 180;
    expect(
      Math.abs(Math.atan2(Math.sin(headings[track] - bearing), Math.cos(headings[track] - bearing)))
    ).toBeLessThan(1e-4);
  }
  compiled.destroy();
  parameters.destroy();
  io.destroy();
});

it('GPUTrajectoryEncounters finds great-circle neighbors across the antimeridian in meters', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Two buckets. Tracks 0 and 1 straddle the antimeridian at latitude 50: 143 m apart in bucket 0
  // (longitude gap 0.002 degrees * cos 50 * 111.2 km) and 71 m in bucket 1. Track 2 is far away.
  const samples = Float32Array.of(
    179.999,
    50,
    179.9995,
    50, // track 0
    -179.999,
    50,
    -179.9995,
    50, // track 1
    0,
    0,
    0,
    0 // track 2
  );
  const trackCount = 3;
  const bucketCount = 2;
  const pairCapacity = 8;
  const io = createBuffers(device);
  const outputs = {
    ids: io.output(pairCapacity),
    partners: io.output(pairCapacity),
    distances: io.output(pairCapacity),
    counts: io.output(pairCapacity),
    count: io.output(1),
    overflow: io.output(1),
    candidateOverflow: io.output(1)
  };
  const graph = new GPUCommandGraph<void>(device, {id: 'trajectory-geodesic-encounters'});
  const encounters = new GPUTrajectoryEncounters({
    spatialContext: GREAT_CIRCLE,
    samples: importGraphBuffer(
      graph,
      'samples',
      io.input(samples),
      'float32x2',
      samples.length / 2
    ),
    trackCount,
    bucketCount,
    distance: importGraphBuffer(graph, 'distance', io.input(Float32Array.of(500)), 'float32', 1),
    cellSize: 2,
    bounds: [-180, -60, 180, 60],
    hitCapacity: 64,
    output: {
      pairs: {
        ids: importGraphBuffer(graph, 'ids', outputs.ids, 'uint32', pairCapacity),
        status: {
          count: importGraphBuffer(graph, 'count', outputs.count, 'uint32', 1),
          overflow: importGraphBuffer(graph, 'overflow', outputs.overflow, 'uint32', 1)
        },
        partners: importGraphBuffer(graph, 'partners', outputs.partners, 'uint32', pairCapacity),
        minimumDistances: importGraphBuffer(
          graph,
          'distances',
          outputs.distances,
          'float32',
          pairCapacity
        ),
        bucketCounts: importGraphBuffer(graph, 'counts', outputs.counts, 'uint32', pairCapacity)
      },
      status: {
        candidateOverflow: importGraphBuffer(
          graph,
          'candidate-overflow',
          outputs.candidateOverflow,
          'uint32',
          1
        )
      }
    }
  });
  graph.add(encounters);
  // 500 m is far below the searchable maximum of this lattice.
  expect(encounters.maximumDistance).toBeGreaterThan(500);
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const [count] = await readUint32(outputs.count, 1);
  const [overflow] = await readUint32(outputs.overflow, 1);
  const [candidateOverflow] = await readUint32(outputs.candidateOverflow, 1);
  const ids = await readUint32(outputs.ids, pairCapacity);
  const partners = await readUint32(outputs.partners, pairCapacity);
  const counts = await readUint32(outputs.counts, pairCapacity);
  const distances = await readFloat32(outputs.distances, pairCapacity);
  expect([count, overflow, candidateOverflow]).toEqual([1, 0, 0]);
  expect([ids[0], partners[0], counts[0]]).toEqual([0, 1, 2]);
  const expected = Math.min(
    getGreatCircleDistance([fround(179.999), 50], [fround(-179.999), 50]),
    getGreatCircleDistance([fround(179.9995), 50], [fround(-179.9995), 50])
  );
  expect(expected).toBeGreaterThan(60);
  expect(expected).toBeLessThan(80);
  expect(Math.abs(distances[0] - expected)).toBeLessThan(0.05 * expected);
  compiled.destroy();
  io.destroy();
});
