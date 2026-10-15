// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {createTransientView, type GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPURipleyParameterValues,
  GPUParameterBuffer,
  GPURipley,
  GPU_RIPLEY_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-dataframe';

/** Observation columns the lens scenario reads from `f-chicago-nature.json`. */
export type RipleyExampleInput = {
  graph: GPUCommandGraph<undefined>;
  observations: {
    /** Interleaved projected x, y in metres. */
    positions: Float32Array;
  };
  /** Study window `[minX, minY, maxX, maxY]` in metres, from the data file. */
  studyWindow: readonly [number, number, number, number];
};

/** Ripley's K and L of iNaturalist observations at 10 radii of 200 m, edge-corrected. */
export async function runExample(
  device: Device,
  {graph, observations, studyWindow}: RipleyExampleInput
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
  const view = <Format extends 'float32' | 'uint32'>(id: string, format: Format, length: number) =>
    createTransientView(graph, id, format, length, Buffer.STORAGE | Buffer.COPY_SRC);
  // #region example
  const radiusCount = 10;
  const parameters = new GPUParameterBuffer(device, {
    id: 'ripley-parameters',
    format: 'float32',
    length: GPU_RIPLEY_PARAMETER_LENGTH,
    values: getGPURipleyParameterValues({
      bounds: studyWindow, // the study window from the data file
      maximumDistance: 2000, // 10 radii of 200 m
      edgeCorrection: 'isotropic'
    })
  });
  const [k, l, lMinusR, radii] = ['k', 'l', 'lMinusR', 'radii'].map(id =>
    view(id, 'float32', radiusCount)
  );
  const pairCounts = view('pairCounts', 'uint32', radiusCount);
  graph.add(
    new GPURipley({
      positions,
      parameters: parameters.importToGraph(graph),
      gridSize: [32, 32],
      radiusCount,
      output: {k, l, lMinusR, radii, pairCounts}
    })
  );
  // k[b], l[b]: edge-corrected K and L = sqrt(K / pi) at radii[b]
  // #endregion example
  return {positions, k, l, lMinusR, radii, pairCounts, parameters, buffers};
}
