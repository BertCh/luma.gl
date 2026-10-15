// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {createTransientView, type GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPUSpatialCorrelogramParameterValues,
  GPUParameterBuffer,
  GPUSpatialCorrelogram,
  GPU_SPATIAL_CORRELOGRAM_PARAMETER_LENGTH,
  GPU_SPATIAL_CORRELOGRAM_STATISTICS_LENGTH
} from '@luma.gl/experimental/gpu-dataframe';

/** Station columns the lens scenario reads from `f-ghcn-stations.json`. */
export type SpatialCorrelogramExampleInput = {
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

/** Moran's I of rainfall at 8 cumulative distance bands of 25 km. */
export async function runExample(
  device: Device,
  {graph, stations, bounds}: SpatialCorrelogramExampleInput
) {
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
  const positions = importColumn('positions', stations.positions, 'float32x2');
  const rainfall = importColumn('rainfall', stations.precipitation, 'float32');
  // #region example
  const bandCount = 8;
  const parameters = new GPUParameterBuffer(device, {
    id: 'correlogram-parameters',
    format: 'float32',
    length: GPU_SPATIAL_CORRELOGRAM_PARAMETER_LENGTH,
    values: getGPUSpatialCorrelogramParameterValues({bounds, maximumDistance: 200_000}) // 8 bands of 25 km
  });
  const [moransI, expectedI, zScores, pValues] = ['moransI', 'expectedI', 'zScores', 'pValues'].map(
    id => view(id, 'float32', bandCount)
  );
  const pairCounts = view('pairCounts', 'uint32', bandCount);
  const statistics = view('statistics', 'float32', GPU_SPATIAL_CORRELOGRAM_STATISTICS_LENGTH);
  graph.add(
    new GPUSpatialCorrelogram({
      positions,
      values: rainfall,
      parameters: parameters.importToGraph(graph),
      gridSize: [64, 64],
      bandCount,
      bandMode: 'cumulative',
      output: {moransI, expectedI, zScores, pValues, pairCounts, statistics}
    })
  );
  // moransI[b]: Moran's I over all station pairs within band b's upper distance
  // #endregion example
  return {
    positions,
    rainfall,
    moransI,
    expectedI,
    zScores,
    pValues,
    pairCounts,
    statistics,
    parameters,
    buffers
  };
}
