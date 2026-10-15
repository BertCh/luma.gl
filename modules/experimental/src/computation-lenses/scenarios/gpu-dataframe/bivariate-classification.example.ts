// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// biome-ignore-all format: the shown example is hand-formatted to fit the 25-line card limit

import {Buffer, type Device} from '@luma.gl/core';
import {createTransientView, type GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPUBivariateClassificationParameterValues,
  GPUBivariateClassification,
  GPUParameterBuffer,
  packGPUColor
} from '@luma.gl/experimental/gpu-dataframe';

/** Tract columns the lens scenario reads from `f-chicago-tracts.json`; suppressed estimates are NaN. */
export type BivariateClassificationExampleInput = {
  graph: GPUCommandGraph<undefined>;
  tracts: {poverty150Pct: Float32Array; uninsuredPct: Float32Array};
};

/** A 3 by 3 bivariate map of poverty against lack of health insurance. */
export async function runExample(
  device: Device,
  {graph, tracts}: BivariateClassificationExampleInput
) {
  const buffers: Buffer[] = [];
  const parameterBuffers: GPUParameterBuffer[] = [];
  const importColumn = <Format extends 'float32' | 'uint32'>(
    id: string,
    data: Float32Array | Uint32Array,
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
    return graph.createDataView(handle, {format, length: data.length});
  };
  const createOutput = (id: string, length: number) =>
    createTransientView(graph, id, 'uint32', length, Buffer.STORAGE | Buffer.COPY_SRC);
  const createParameters = (id: string, values: Float32Array) => {
    const parameters = new GPUParameterBuffer(device, {
      id,
      format: 'float32',
      length: values.length,
      values
    });
    parameterBuffers.push(parameters);
    return parameters.importToGraph(graph);
  };
  // #region example
  const poverty = importColumn('poverty150Pct', tracts.poverty150Pct, 'float32');
  const uninsured = importColumn('uninsuredPct', tracts.uninsuredPct, 'float32');
  // Stevens 3 x 3 palette, row-major: palette[uninsuredClass * 3 + povertyClass]
  const stevens = [0xe8e8e8, 0xace4e4, 0x5ac8c8, 0xdfb0d6, 0xa5add3, 0x5698b9, 0xbe64ac, 0x8c62aa, 0x3b4994]
    .map(rgb => packGPUColor(rgb >> 16, rgb >> 8, rgb));
  const output = {
    classIds: createOutput('classIds', poverty.length),
    colors: createOutput('colors', poverty.length),
    classCounts: createOutput('classCounts', 9)
  };
  graph.add(new GPUBivariateClassification({
    valuesX: poverty,
    valuesY: uninsured,
    breaksX: importColumn('breaksX', Float32Array.of(0, 20, 35, 100), 'float32'), // percent
    breaksY: importColumn('breaksY', Float32Array.of(0, 8, 16, 100), 'float32'),
    palette: importColumn('palette', Uint32Array.from(stevens), 'uint32'),
    parameters: createParameters('bivariate-parameters', getGPUBivariateClassificationParameterValues({classCountX: 3, classCountY: 3})),
    maximumClassCount: 3,
    output
  }));
  const {classIds, colors} = output;
  // #endregion example
  return {
    poverty,
    uninsured,
    classIds,
    colors,
    classCounts: output.classCounts,
    destroy() {
      for (const parameters of parameterBuffers) parameters.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  };
}