// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUVector} from '@luma.gl/gpgpu/gpu-data';
import {GPUGraphDegree, type GPUGraphTopology} from '@luma.gl/gpgpu/gpu-graph';

/** Declares outgoing station degrees from the rail CSR offsets, without submitting work. */
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
  const outDegrees = new GPUVector({
    type: 'buffer',
    name: 'out-degrees',
    format: 'uint32',
    length: stations.vertexCount,
    ownsBuffer: true,
    buffer: device.createBuffer({
      byteLength: stations.vertexCount * 4,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    })
  });
  const degree = new GPUGraphDegree({
    topology: railTopology,
    output: outDegrees,
    direction: 'outgoing'
  });
  degree.addToGraph(commandGraph);
  return {outDegrees: degree.output};
  // #endregion example
}
