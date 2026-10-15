// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {createTransientView, type GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPURipleyParameterValues,
  GPUParameterBuffer,
  GPURipley,
  GPURipleyEnvelope,
  GPU_RIPLEY_ENVELOPE_STATISTICS_LENGTH,
  GPU_RIPLEY_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-dataframe';

/** Observation columns the lens scenario reads from `f-chicago-nature.json`. */
export type RipleyEnvelopeExampleInput = {
  graph: GPUCommandGraph<undefined>;
  observations: {
    /** Interleaved projected x, y in metres. */
    positions: Float32Array;
  };
  /** Study window `[minX, minY, maxX, maxY]` in metres, from the data file. */
  studyWindow: readonly [number, number, number, number];
};

/** Observed L - r against a 39-simulation complete-spatial-randomness envelope, fixed seed. */
export async function runExample(
  device: Device,
  {graph, observations, studyWindow}: RipleyEnvelopeExampleInput
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
  const radiusCount = 10; // 10 radii of 200 m
  const views = (ids: string[], length = radiusCount) =>
    ids.map(id =>
      createTransientView(graph, id, 'float32', length, Buffer.STORAGE | Buffer.COPY_SRC)
    );
  const [observedK, observedLMinusR, radii] = views(['observedK', 'observedLMinusR', 'radii']);
  const [lower, upper, globalLower, globalUpper] = views([
    'lower',
    'upper',
    'globalLower',
    'globalUpper'
  ]);
  const [statistics] = views(['statistics'], GPU_RIPLEY_ENVELOPE_STATISTICS_LENGTH);
  // #region example
  const parameters = new GPUParameterBuffer(device, {
    id: 'envelope-parameters',
    format: 'float32',
    length: GPU_RIPLEY_PARAMETER_LENGTH,
    values: getGPURipleyParameterValues({bounds: studyWindow, maximumDistance: 2000, seed: 2024})
  });
  const common = {
    positions,
    parameters: parameters.importToGraph(graph),
    gridSize: [32, 32] as const,
    radiusCount
  };
  graph.add(new GPURipley({...common, output: {k: observedK, lMinusR: observedLMinusR, radii}}));
  graph.add(
    new GPURipleyEnvelope({
      ...common,
      simulationCount: 39,
      observedK,
      output: {lower, upper, globalLower, globalUpper, statistics}
    })
  );
  // lower[b], upper[b]: pointwise CSR band of L - r; statistics[4]: Monte Carlo p-value
  // #endregion example
  return {
    positions,
    observedK,
    observedLMinusR,
    radii,
    lower,
    upper,
    globalLower,
    globalUpper,
    statistics,
    parameters,
    buffers
  };
}
