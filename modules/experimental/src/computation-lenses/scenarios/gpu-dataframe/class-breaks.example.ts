// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// biome-ignore-all format: the shown example is hand-formatted to fit the 25-line card limit

import {Buffer, type Device} from '@luma.gl/core';
import {createTransientView, type GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPUClassBreaksParameterLength,
  getGPUClassBreaksParameterValues,
  GPUClassBreaks,
  GPUParameterBuffer
} from '@luma.gl/experimental/gpu-dataframe';

/** Tract columns the lens scenario reads from `f-chicago-tracts.json`; suppressed estimates are NaN. */
export type ClassBreaksExampleInput = {
  graph: GPUCommandGraph<undefined>;
  tracts: {poverty150Pct: Float32Array};
};

/** Five equal-interval classes of tract poverty; quantile breaks are the compiled alternative. */
export async function runExample(device: Device, {graph, tracts}: ClassBreaksExampleInput) {
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
  const classCount = 5;
  const parameters = new GPUParameterBuffer(device, {
    id: 'class-break-parameters',
    format: 'float32',
    length: getGPUClassBreaksParameterLength(classCount),
    values: getGPUClassBreaksParameterValues({method: 'equal-interval', classCount}, classCount)
  });
  const output = {
    breaks: createOutput('breaks', 'float32', classCount + 1), // e[0..k], minimum to maximum
    classCount: createOutput('classCount', 'uint32', 1),
    classCounts: createOutput('classCounts', 'uint32', classCount)
  };
  graph.add(new GPUClassBreaks({
    values: poverty,
    parameters: parameters.importToGraph(graph),
    maximumClassCount: classCount,
    methods: ['equal-interval', 'quantile'], // compile only the methods you switch between
    output
  }));
  const {breaks, classCounts} = output;
  // #endregion example
  return {
    poverty,
    breaks,
    classCount: output.classCount,
    classCounts,
    destroy() {
      parameters.destroy();
      povertyBuffer.destroy();
    }
  };
}