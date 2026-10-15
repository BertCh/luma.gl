// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUVector} from '@luma.gl/gpgpu/gpu-data';
import {GPUGraphLabelPropagation, type GPUGraphTopology} from '@luma.gl/gpgpu/gpu-graph';

/** Declares the label-propagation run used by the result figure, without submitting work. */
export async function runExample(
  device: Device,
  input: {
    commandGraph: GPUCommandGraph<void>;
    railTopology: GPUGraphTopology;
    stations: {vertexCount: number};
  }
) {
  // #region example
  const {commandGraph, railTopology, stations} = input;
  const usage = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST;
  const createColumn = (name: string, length: number) =>
    new GPUVector({
      type: 'buffer',
      name,
      format: 'uint32',
      length,
      ownsBuffer: true,
      buffer: device.createBuffer({byteLength: length * 4, usage})
    });
  const communities = createColumn('communities', stations.vertexCount);
  const converged = createColumn('converged', 1);
  const labelPropagation = new GPUGraphLabelPropagation({
    topology: railTopology,
    output: communities,
    iterations: 16,
    converged
  });
  labelPropagation.addToGraph(commandGraph);
  return {communities: labelPropagation.output, converged: labelPropagation.converged};
  // #endregion example
}
