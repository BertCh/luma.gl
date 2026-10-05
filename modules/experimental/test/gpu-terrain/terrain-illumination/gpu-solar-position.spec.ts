// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPUSolarPositionParameterValues,
  GPUSolarPosition
} from '../../../src/gpu-terrain/terrain-illumination/gpu-solar-position';
import {getSolarPosition} from '../../../src/gpu-terrain/terrain-illumination/solar-position';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {createRandom} from './terrain-horizon-oracle';
import {getAngleDifference} from './solar-position-oracle';

const TIMESTAMPS = [
  Date.UTC(2026, 9, 4, 6, 15),
  Date.UTC(2026, 5, 21, 12),
  Date.UTC(2026, 11, 21, 23, 59, 30),
  Date.UTC(2003, 2, 20, 0, 1),
  Date.UTC(2041, 7, 9, 17, 42, 11)
];

it('GPUSolarPosition matches getSolarPosition for many locations and per-frame times', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const random = createRandom(42);
  const rowCount = 128;
  const positions = new Float32Array(rowCount * 2);
  for (let row = 0; row < rowCount; row++) {
    positions[row * 2] = -180 + random() * 360;
    positions[row * 2 + 1] = -89 + random() * 178;
  }
  // Polar day and night, the equator, and invalid rows.
  positions.set([20, 89.5, -60, -89.5, 0, 0, 10, NaN, 10, 91, Infinity, 10], 0);
  for (const refraction of [true, false]) {
    const positionBuffer = createInputBuffer(device, positions);
    const outputs = [
      createOutputBuffer(device, rowCount),
      createOutputBuffer(device, rowCount),
      createOutputBuffer(device, rowCount)
    ];
    const settings = new GPUParameterBuffer(device, {
      id: 'solar-settings',
      format: 'float32',
      length: 4,
      values: getGPUSolarPositionParameterValues({timestamp: TIMESTAMPS[0]})
    });
    const graph = new GPUCommandGraph(device, {id: 'solar-position-test'});
    graph.add(
      new GPUSolarPosition({
        positions: importGraphBuffer(graph, 'positions', positionBuffer, 'float32x2', rowCount),
        settings: settings.importToGraph(graph),
        azimuth: importGraphBuffer(graph, 'azimuth', outputs[0], 'float32', rowCount),
        altitude: importGraphBuffer(graph, 'altitude', outputs[1], 'float32', rowCount),
        daylight: importGraphBuffer(graph, 'daylight', outputs[2], 'uint32', rowCount),
        refraction
      })
    );
    const compiled = graph.compile();
    const compileCount = 1;
    let worstAltitude = 0;
    let worstAzimuth = 0;
    for (const timestamp of TIMESTAMPS) {
      settings.write(getGPUSolarPositionParameterValues({timestamp}));
      submitGraph(device, compiled, undefined);
      const azimuth = await readFloat32(outputs[0], rowCount);
      const altitude = await readFloat32(outputs[1], rowCount);
      const daylight = await readUint32(outputs[2], rowCount);
      for (let row = 0; row < rowCount; row++) {
        const longitude = positions[row * 2];
        const latitude = positions[row * 2 + 1];
        if (!Number.isFinite(longitude) || !Number.isFinite(latitude) || Math.abs(latitude) > 90) {
          expect(Number.isNaN(azimuth[row])).toBe(true);
          expect(Number.isNaN(altitude[row])).toBe(true);
          expect(daylight[row]).toBe(0);
          continue;
        }
        const expected = getSolarPosition(timestamp, longitude, latitude, {
          refraction
        });
        worstAltitude = Math.max(worstAltitude, Math.abs(altitude[row] - expected.altitudeDegrees));
        if (expected.altitudeDegrees < 89) {
          worstAzimuth = Math.max(
            worstAzimuth,
            getAngleDifference(azimuth[row], expected.azimuthDegrees)
          );
        }
        if (Math.abs(expected.geometricAltitudeDegrees + 0.833) > 0.05) {
          expect(daylight[row]).toBe(expected.geometricAltitudeDegrees > -0.833 ? 1 : 0);
        }
      }
    }
    expect(worstAltitude).toBeLessThan(0.02);
    expect(worstAzimuth).toBeLessThan(0.05);
    expect(compileCount).toBe(1);
    compiled.destroy();
    settings.destroy();
    for (const buffer of [positionBuffer, ...outputs]) buffer.destroy();
  }
});
