// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUVector} from '@luma.gl/gpgpu/gpu-data';
import {GPUGraphLocalClusteringCoefficient, type GPUGraphTopology} from '@luma.gl/gpgpu/gpu-graph';

/** Declares local clustering coefficients over the undirected rail CSR, without submitting work. */
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
  const clustering = new GPUVector({
    type: 'buffer',
    name: 'clustering',
    format: 'float32',
    length: stations.vertexCount,
    ownsBuffer: true,
    buffer: device.createBuffer({
      byteLength: stations.vertexCount * 4,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    })
  });
  const localClustering = new GPUGraphLocalClusteringCoefficient({
    topology: railTopology,
    output: clustering
  });
  localClustering.addToGraph(commandGraph);
  return {clustering: localClustering.output};
  // #endregion example
}
