// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {type GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  type GPURasterMetadata,
  type GPUStreamlinesOutput,
  GPUParameterBuffer,
  GPUStreamlines,
  GPU_STREAMLINES_PARAMETER_LENGTH,
  GPU_STREAMLINES_UINT32_PARAMETER_LENGTH,
  getGPUStreamlinesParameterValues,
  getGPUStreamlinesUint32ParameterValues
} from '@luma.gl/experimental/gpu-raster';

/** Graph, wind grid and caller-owned views for streamlines. */
export type StreamlinesExampleInput = {
  graph: GPUCommandGraph;
  /** Wind grid: 0.25 degree cells, rows north to south. */
  grid: GPURasterMetadata;
  /** Wind in degrees per hour, rows from the south edge as the contributor expects. */
  velocities: GraphDataView<'float32x2'>;
  /** Line ids and status words, sized to the expected result. */
  lines: GPUStreamlinesOutput['lines'];
  lineOffsets: GraphDataView<'uint32'>;
  positions: GraphDataView<'float32x2'>;
  positionCount: GraphDataView<'uint32'>;
};

/** Adds the recorded streamlines computation; returns its owned parameter buffers. */
export async function runExample(device: Device, input: StreamlinesExampleInput) {
  const {graph, grid, velocities, lines, lineOffsets, positions, positionCount} = input;
  const parameters = new GPUParameterBuffer(device, {
    id: 'streamlines-parameters',
    format: 'float32',
    length: GPU_STREAMLINES_PARAMETER_LENGTH
  });
  const words = new GPUParameterBuffer(device, {
    id: 'streamlines-words',
    format: 'uint32',
    length: GPU_STREAMLINES_UINT32_PARAMETER_LENGTH
  });
  // #region example
  const [west, cell] = [grid.affine[2], grid.affine[0]];
  const south = grid.affine[5] - grid.height * cell; // field rows start at the south edge
  parameters.write(
    getGPUStreamlinesParameterValues({
      fieldExtent: [west, south, cell, cell], // degrees
      gridExtent: [west, south, 0.5, 0.5], // 0.5 degree separation between lines
      stepLength: 0.1 // degrees per step
    })
  );
  words.write(getGPUStreamlinesUint32ParameterValues({seed: 1, minimumPoints: 8}));
  graph.add(
    new GPUStreamlines({
      velocities, // degrees per hour
      fieldWidth: grid.width,
      fieldHeight: grid.height,
      gridWidth: 32, // 16 degrees of longitude
      gridHeight: 24, // 12 degrees of latitude
      seedColumns: 6, // 6 by 5 = 30 jittered seeds
      seedRows: 5,
      stepsPerDirection: 40,
      parameters: parameters.importToGraph(graph),
      uint32Parameters: words.importToGraph(graph),
      output: {lines, lineOffsets, positions, positionCount} // polylines in CSR form
    })
  );
  // #endregion example
  return {parameters: [parameters, words]};
}
