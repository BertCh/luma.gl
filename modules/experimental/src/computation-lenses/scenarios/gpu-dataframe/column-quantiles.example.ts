// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// biome-ignore-all format: the shown example is hand-formatted to fit the 25-line card limit

import {Buffer, type Device} from '@luma.gl/core';
import {createTransientView, type GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPUColumnQuantilesParameterLength,
  getGPUColumnQuantilesParameterValues,
  GPUColumnQuantiles,
  GPUParameterBuffer
} from '@luma.gl/experimental/gpu-dataframe';

/** Tract columns the lens scenario reads from `f-chicago-tracts.json`; suppressed estimates are NaN. */
export type ColumnQuantilesExampleInput = {
  graph: GPUCommandGraph<undefined>;
  tracts: {poverty150Pct: Float32Array};
};

/** Quartiles of tract poverty plus a 5-95 percentile row filter, with no readback and no sort. */
export async function runExample(device: Device, {graph, tracts}: ColumnQuantilesExampleInput) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST;
  const povertyBuffer = device.createBuffer({
    id: 'poverty150Pct',
    data: tracts.poverty150Pct,
    usage
  });
  const povertyHandle = graph.importBuffer(
    {id: 'poverty150Pct', byteLength: povertyBuffer.byteLength, usage},
    povertyBuffer
  );
  const createOutput = <Format extends 'float32' | 'uint32'>(
    id: string,
    format: Format,
    length: number
  ) => createTransientView(graph, id, format, length, Buffer.STORAGE | Buffer.COPY_SRC);
  // #region example
  const poverty = graph.createDataView(povertyHandle, {format: 'float32', length: tracts.poverty150Pct.length});
  const probabilities = [0.25, 0.5, 0.75];
  const parameters = new GPUParameterBuffer(device, {
    id: 'quantile-parameters',
    format: 'float32',
    length: getGPUColumnQuantilesParameterLength(probabilities.length),
    values: getGPUColumnQuantilesParameterValues({quantiles: probabilities, filterRange: [0.05, 0.95]})
  });
  const parameterView = parameters.importToGraph(graph);
  const output = {
    quantiles: createOutput('quantiles', 'float32', probabilities.length),
    validCount: createOutput('validCount', 'uint32', 1),
    outputMask: createOutput('outputMask', 'uint32', poverty.length), // 1 inside the 5-95 % range
    filterBounds: createOutput('filterBounds', 'float32', 2)
  };
  graph.add(new GPUColumnQuantiles({values: poverty, parameters: parameterView, quantileCount: 3, output}));
  const {quantiles, validCount, outputMask, filterBounds} = output;
  // #endregion example
  return {
    poverty,
    parameterView,
    quantiles,
    validCount,
    outputMask,
    filterBounds,
    destroy() {
      parameters.destroy();
      povertyBuffer.destroy();
    }
  };
}