// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {createTransientView, type GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPURipleyDistanceFunctionsParameterValues,
  GPUParameterBuffer,
  GPURipleyDistanceFunctions,
  GPU_RIPLEY_DISTANCE_FUNCTIONS_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-dataframe';

/** Observation columns the lens scenario reads from `f-chicago-nature.json`. */
export type RipleyDistanceFunctionsExampleInput = {
  graph: GPUCommandGraph<undefined>;
  observations: {
    /** Interleaved projected x, y in metres. */
    positions: Float32Array;
  };
  /** Study window `[minX, minY, maxX, maxY]` in metres, from the data file. */
  studyWindow: readonly [number, number, number, number];
};

/** G, F and J of iNaturalist observations at 10 radii of 200 m, border-corrected. */
export async function runExample(
  device: Device,
  {graph, observations, studyWindow}: RipleyDistanceFunctionsExampleInput
) {
  const buffers: Buffer[] = [];
  const buffer = device.createBuffer({
    id: 'observations',
    data: observations.positions,
    usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
  });
  buffers.push(buffer);
  const positions = graph.createDataView(
    graph.importBuffer(
      {id: 'observations', byteLength: buffer.byteLength, usage: buffer.usage},
      buffer
    ),
    {format: 'float32x2', length: observations.positions.length / 2}
  );
  const view = (id: string, length: number) =>
    createTransientView(graph, id, 'float32', length, Buffer.STORAGE | Buffer.COPY_SRC);
  // #region example
  const radiusCount = 10;
  const parameters = new GPUParameterBuffer(device, {
    id: 'distance-function-parameters',
    format: 'float32',
    length: GPU_RIPLEY_DISTANCE_FUNCTIONS_PARAMETER_LENGTH,
    values: getGPURipleyDistanceFunctionsParameterValues({
      bounds: studyWindow, // the study window from the data file
      maximumDistance: 2000, // 10 radii of 200 m
      edgeCorrection: 'border'
    })
  });
  const [g, f, j, radii] = ['g', 'f', 'j', 'radii'].map(id => view(id, radiusCount));
  graph.add(
    new GPURipleyDistanceFunctions({
      positions,
      parameters: parameters.importToGraph(graph),
      gridSize: [32, 32],
      referenceGrid: [53, 47], // coprime with the window, F's reference locations
      radiusCount,
      output: {g, f, j, radii}
    })
  );
  // g[b], f[b], j[b]: nearest-event and empty-space functions at radii[b]; j = (1 - g) / (1 - f)
  // #endregion example
  return {positions, g, f, j, radii, parameters, buffers};
}
