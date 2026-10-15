// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUVector} from '@luma.gl/gpgpu/gpu-data';
import {GPUGraphForceLayout, type GPUGraphTopology} from '@luma.gl/gpgpu/gpu-graph';

const KILOMETERS_PER_DEGREE = (Math.PI / 180) * 6371;

/** Input preparation: equirectangular lon/lat to planar km centered on the mean station. */
function projectToLocalKm(lonLat: Float32Array): Float32Array {
  const count = lonLat.length / 2;
  let meanLongitude = 0;
  let meanLatitude = 0;
  for (let index = 0; index < count; index++) {
    meanLongitude += lonLat[2 * index] / count;
    meanLatitude += lonLat[2 * index + 1] / count;
  }
  const scaleX = Math.cos((meanLatitude * Math.PI) / 180) * KILOMETERS_PER_DEGREE;
  const projected = new Float32Array(lonLat.length);
  for (let index = 0; index < count; index++) {
    projected[2 * index] = (lonLat[2 * index] - meanLongitude) * scaleX;
    projected[2 * index + 1] = (lonLat[2 * index + 1] - meanLatitude) * KILOMETERS_PER_DEGREE;
  }
  return projected;
}

/** Declares the force-layout run used by the result figure, without submitting work. */
export async function runExample(
  device: Device,
  input: {
    commandGraph: GPUCommandGraph<void>;
    railTopology: GPUGraphTopology;
    stations: {vertexCount: number; lonLat: Float32Array};
  }
) {
  const createColumn = (name: string, values: Float32Array, extraUsage = 0) => {
    const usage = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST | extraUsage;
    const buffer = device.createBuffer({byteLength: values.byteLength, usage});
    buffer.write(values);
    return new GPUVector({
      type: 'buffer',
      name,
      format: 'float32x2',
      length: values.length / 2,
      ownsBuffer: true,
      buffer
    });
  };
  // #region example
  const {commandGraph, railTopology, stations} = input;
  const projectedKm = projectToLocalKm(stations.lonLat);
  const startPositions = createColumn('layout-start', projectedKm);
  const positions = createColumn('layout-positions', projectedKm, Buffer.VERTEX);
  const velocities = createColumn('layout-velocities', new Float32Array(projectedKm.length));
  const forceLayout = new GPUGraphForceLayout({
    topology: railTopology,
    positions,
    velocities,
    iterationsPerFrame: 100,
    repulsion: 5,
    attraction: 0.02,
    gravity: 0.05,
    maxVelocity: 2
  });
  forceLayout.addToGraph(commandGraph);
  return {startPositions, positions: forceLayout.positions, velocities: forceLayout.velocities};
  // #endregion example
}
