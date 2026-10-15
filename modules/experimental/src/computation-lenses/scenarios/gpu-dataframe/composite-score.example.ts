// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// biome-ignore-all format: the shown example is hand-formatted to fit the 25-line card limit

import {Buffer, type Device} from '@luma.gl/core';
import {createTransientView, type GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPUCompositeScoreParameterValues,
  GPUCompositeScore,
  GPUParameterBuffer,
  GPU_COMPOSITE_SCORE_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-dataframe';

/** Tract rates the lens scenario reads from `f-chicago-tracts.json`; suppressed estimates are NaN. */
export type CompositeScoreExampleInput = {
  graph: GPUCommandGraph<undefined>;
  tracts: {
    /** Row-major `row * 4 + column`: poverty, unemployment, uninsured and no-vehicle percentages. */
    indicators: Float32Array;
  };
};

/** Equal-weight deprivation score of four tract rates, each min-max scaled to [0, 1]. */
export async function runExample(device: Device, {graph, tracts}: CompositeScoreExampleInput) {
  const rowCount = tracts.indicators.length / 4;
  const view = <Format extends 'float32' | 'uint32'>(id: string, format: Format, length: number) =>
    createTransientView(graph, id, format, length, Buffer.STORAGE | Buffer.COPY_SRC);
  const usage = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST;
  const indicatorBuffer = device.createBuffer({id: 'indicators', data: tracts.indicators, usage});
  const indicatorHandle = graph.importBuffer(
    {id: 'indicators', byteLength: indicatorBuffer.byteLength, usage},
    indicatorBuffer
  );
  // #region example
  const indicators = graph.createDataView(indicatorHandle, {format: 'float32', length: tracts.indicators.length});
  const parameters = new GPUParameterBuffer(device, {
    id: 'composite-score-parameters',
    format: 'float32',
    length: GPU_COMPOSITE_SCORE_PARAMETER_LENGTH,
    // directions default to +1: higher is worse
    values: getGPUCompositeScoreParameterValues({scaler: 'min-max', aggregation: 'weighted-sum', weights: [1, 1, 1, 1]})
  });
  const output = {
    score: view('score', 'float32', rowCount), // NaN for tracts with a suppressed indicator
    columnStatistics: view('columnStatistics', 'float32', 16) // [min, max, mean, sd] per indicator
  };
  graph.add(new GPUCompositeScore({
    indicators,
    indicatorCount: 4,
    parameters: parameters.importToGraph(graph),
    output
  }));
  const {score, columnStatistics} = output;
  // #endregion example
  return {indicators, score, columnStatistics, parameters, indicatorBuffer};
}