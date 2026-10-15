// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUVector} from '@luma.gl/gpgpu/gpu-data';
import {
  GPUGraphForceLayout,
  GPUGraphSpatialForceLayout,
  type GPUGraphTopology
} from '@luma.gl/gpgpu/gpu-graph';

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

/** Input preparation: the start extent grown by its full width and height on every side. */
function getPaddedBounds(positions: Float32Array): [number, number, number, number] {
  const bounds: [number, number, number, number] = [Infinity, Infinity, -Infinity, -Infinity];
  for (let index = 0; index < positions.length; index += 2) {
    bounds[0] = Math.min(bounds[0], positions[index]);
    bounds[1] = Math.min(bounds[1], positions[index + 1]);
    bounds[2] = Math.max(bounds[2], positions[index]);
    bounds[3] = Math.max(bounds[3], positions[index + 1]);
  }
  const width = bounds[2] - bounds[0];
  const height = bounds[3] - bounds[1];
  return [bounds[0] - width, bounds[1] - height, bounds[2] + width, bounds[3] + height];
}

/** Declares the spatial force-layout run used by the result figure, without submitting work. */
export async function runExample(
  device: Device,
  input: {
    commandGraph: GPUCommandGraph<void>;
    railTopology: GPUGraphTopology;
    stations: {vertexCount: number; lonLat: Float32Array};
  }
) {
  const createColumn = <Format extends 'uint32' | 'float32x2'>(
    name: string,
    format: Format,
    values: Uint32Array | Float32Array,
    extraUsage = 0
  ) => {
    const usage = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST | extraUsage;
    const buffer = device.createBuffer({byteLength: values.byteLength, usage});
    buffer.write(values);
    const length = values.length / (format === 'float32x2' ? 2 : 1);
    return new GPUVector({type: 'buffer', name, format, length, ownsBuffer: true, buffer});
  };
  const gridSize = [16, 16] as const;
  const cellCount = gridSize[0] * gridSize[1];
  // Caller-owned grid index storage; the layout rebuilds it from current positions each iteration.
  const createGridColumns = () => ({
    cellOffsets: createColumn('grid-offsets', 'uint32', new Uint32Array(cellCount + 1)),
    vertexIds: createColumn(
      'grid-vertex-ids',
      'uint32',
      new Uint32Array(input.stations.vertexCount)
    ),
    cellCenters: createColumn('grid-centers', 'float32x2', new Float32Array(2 * cellCount)),
    count: createColumn('grid-count', 'uint32', new Uint32Array(1)),
    overflow: createColumn('grid-overflow', 'uint32', new Uint32Array(1))
  });
  // #region example
  const {commandGraph, railTopology, stations} = input;
  const projectedKm = projectToLocalKm(stations.lonLat);
  const startPositions = createColumn('start', 'float32x2', projectedKm);
  const positions = createColumn('positions', 'float32x2', projectedKm, Buffer.VERTEX);
  const velocities = createColumn('velocities', 'float32x2', new Float32Array(projectedKm.length));
  const layout = new GPUGraphForceLayout({
    topology: railTopology,
    positions,
    velocities,
    iterationsPerFrame: 64,
    repulsion: 5,
    attraction: 0.02,
    gravity: 0.05,
    maxVelocity: 2
  });
  const grid = createGridColumns();
  const spatialLayout = new GPUGraphSpatialForceLayout({
    layout,
    gridSize,
    bounds: getPaddedBounds(projectedKm),
    ...grid
  });
  spatialLayout.addToGraph(commandGraph);
  return {startPositions, positions: spatialLayout.layout.positions, velocities, ...grid};
  // #endregion example
}
