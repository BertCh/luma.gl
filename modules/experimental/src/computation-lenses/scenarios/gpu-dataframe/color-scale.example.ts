// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// biome-ignore-all format: the shown example is hand-formatted to fit the 25-line card limit

import {Buffer, type Device} from '@luma.gl/core';
import {createTransientView, type GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPUColorScaleParameterValues,
  GPUColorScale,
  GPUParameterBuffer,
  packGPUColor
} from '@luma.gl/experimental/gpu-dataframe';

/** Tract columns the lens scenario reads from `f-chicago-tracts.json`; suppressed estimates are NaN. */
export type ColorScaleExampleInput = {
  graph: GPUCommandGraph<undefined>;
  tracts: {poverty150Pct: Float32Array};
};

/** Five poverty bands coloured with a Blues palette, written as a packed `rgba8` column. */
export async function runExample(device: Device, {graph, tracts}: ColorScaleExampleInput) {
  const buffers: Buffer[] = [];
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
  const parameterBuffers: GPUParameterBuffer[] = [];
  const createParameters = (id: string, values: Float32Array) => {
    const parameters = new GPUParameterBuffer(device, {id, format: 'float32', length: values.length, values});
    parameterBuffers.push(parameters);
    return parameters.importToGraph(graph);
  };
  // #region example
  const poverty = importColumn('poverty150Pct', tracts.poverty150Pct, 'float32');
  const bands = [0, 10, 20, 30, 40, 100]; // percent; the four inner edges are thresholds
  const blues = [0xc6dbef, 0x8abfdd, 0x4a98c9, 0x1c6bb0, 0x083d7f].map(c => packGPUColor(c >> 16, c >> 8, c));
  const scale = {scale: 'threshold', domainCount: bands.length, paletteCount: blues.length} as const;
  const output = {
    classIndices: createOutput('classIndices', poverty.length),
    colors: createOutput('colors', poverty.length),
    classCounts: createOutput('classCounts', blues.length)
  };
  graph.add(new GPUColorScale({
    values: poverty,
    domain: importColumn('domain', Float32Array.from(bands), 'float32'),
    palette: importColumn('palette', Uint32Array.from(blues), 'uint32'),
    parameters: createParameters('color-scale-parameters', getGPUColorScaleParameterValues(scale)),
    maximumDomainCount: bands.length,
    maximumPaletteCount: blues.length,
    output
  }));
  const {classIndices, colors} = output;
  // #endregion example
  return {
    poverty,
    bands,
    classIndices,
    colors,
    classCounts: output.classCounts,
    destroy() {
      for (const parameters of parameterBuffers) parameters.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  };
}