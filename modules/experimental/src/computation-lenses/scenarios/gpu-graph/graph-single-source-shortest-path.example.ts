// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUVector} from '@luma.gl/gpgpu/gpu-data';
import {GPUGraphSingleSourceShortestPath, type GPUGraphTopology} from '@luma.gl/gpgpu/gpu-graph';

/** Declares travel times from Utrecht Centraal over weighted rail CSR, without submitting work. */
export async function runExample(
  device: Device,
  input: {
    commandGraph: GPUCommandGraph<void>;
    railTopology: GPUGraphTopology;
    stations: {vertexCount: number; names: readonly string[]};
  }
) {
  // #region example
  const {commandGraph, railTopology, stations} = input;
  const usage = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST;
  const createColumn = <Format extends 'uint32' | 'float32'>(name: string, format: Format) =>
    new GPUVector({
      type: 'buffer',
      name,
      format,
      length: stations.vertexCount,
      ownsBuffer: true,
      buffer: device.createBuffer({byteLength: stations.vertexCount * 4, usage})
    });
  const shortestPaths = new GPUGraphSingleSourceShortestPath({
    topology: railTopology,
    sourceVertex: stations.names.indexOf('Utrecht Centraal'),
    distances: createColumn('travel-seconds', 'float32'),
    predecessors: createColumn('parents', 'uint32'),
    maxIterations: 24
  });
  shortestPaths.addToGraph(commandGraph);
  return {seconds: shortestPaths.distances, parents: shortestPaths.predecessors};
  // #endregion example
}
