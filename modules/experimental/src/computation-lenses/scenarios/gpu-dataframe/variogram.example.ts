// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {createTransientView, type GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPUVariogramParameterValues,
  GPUParameterBuffer,
  GPUVariogram,
  GPU_VARIOGRAM_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-dataframe';

/** Station columns the lens scenario reads from `f-ghcn-stations.json`. */
export type VariogramExampleInput = {
  graph: GPUCommandGraph<undefined>;
  stations: {
    /** Interleaved projected x, y in metres. */
    positions: Float32Array;
    /** Daily precipitation in millimetres. */
    precipitation: Float32Array;
  };
  /** Inclusive `[minX, minY, maxX, maxY]` extent of the stations, in metres. */
  bounds: readonly [number, number, number, number];
};

/** Empirical semivariogram of rainfall at 12 lags of 25 km, omnidirectional. */
export async function runExample(device: Device, {graph, stations, bounds}: VariogramExampleInput) {
  const buffers: Buffer[] = [];
  const importColumn = <Format extends 'float32' | 'float32x2'>(
    id: string,
    data: Float32Array,
    format: Format
  ) => {
    const buffer = device.createBuffer({
      id,
      data,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    buffers.push(buffer);
    const handle = graph.importBuffer(
      {id, byteLength: buffer.byteLength, usage: buffer.usage},
      buffer
    );
    return graph.createDataView(handle, {
      format,
      length: data.length / (format === 'float32x2' ? 2 : 1)
    });
  };
  const view = <Format extends 'float32' | 'uint32'>(id: string, format: Format, length: number) =>
    createTransientView(graph, id, format, length, Buffer.STORAGE | Buffer.COPY_SRC);
  // #region example
  const positions = importColumn('positions', stations.positions, 'float32x2');
  const rainfall = importColumn('rainfall', stations.precipitation, 'float32');
  const lagCount = 12;
  const parameters = new GPUParameterBuffer(device, {
    id: 'variogram-parameters',
    format: 'float32',
    length: GPU_VARIOGRAM_PARAMETER_LENGTH,
    values: getGPUVariogramParameterValues({bounds, maximumDistance: 300_000}) // 12 lags of 25 km
  });
  const semivariances = view('semivariances', 'float32', lagCount);
  const pairCounts = view('pairCounts', 'uint32', lagCount);
  const meanDistances = view('meanDistances', 'float32', lagCount);
  graph.add(
    new GPUVariogram({
      positions,
      values: rainfall,
      parameters: parameters.importToGraph(graph),
      gridSize: [64, 64],
      lagCount,
      output: {semivariances, pairCounts, meanDistances}
    })
  );
  // semivariances[h]: half the mean squared rainfall difference of the pairs in lag h
  // #endregion example
  return {positions, rainfall, semivariances, pairCounts, meanDistances, parameters, buffers};
}
