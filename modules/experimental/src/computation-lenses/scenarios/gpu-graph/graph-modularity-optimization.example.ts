// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUVector} from '@luma.gl/gpgpu/gpu-data';
import {
  GPUGraphLabelPropagation,
  GPUGraphModularityOptimization,
  type GPUGraphTopology
} from '@luma.gl/gpgpu/gpu-graph';

/** Declares label propagation seeding modularity local moves, without submitting work. */
export async function runExample(
  device: Device,
  input: {
    commandGraph: GPUCommandGraph<void>;
    railTopology: GPUGraphTopology;
    stations: {vertexCount: number};
  }
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST;
  const createColumn = <Format extends 'uint32' | 'float32'>(
    name: string,
    format: Format,
    length: number
  ) =>
    new GPUVector({
      type: 'buffer',
      name,
      format,
      length,
      ownsBuffer: true,
      buffer: device.createBuffer({byteLength: length * 4, usage})
    });
  // #region example
  const {commandGraph, railTopology, stations} = input;
  const seedCommunities = createColumn('seed-communities', 'uint32', stations.vertexCount);
  const communities = createColumn('communities', 'uint32', stations.vertexCount);
  const modularity = createColumn('modularity', 'float32', 1);
  const converged = createColumn('converged', 'uint32', 1);
  const valid = createColumn('valid', 'uint32', 1);
  new GPUGraphLabelPropagation({
    topology: railTopology,
    output: seedCommunities,
    iterations: 16
  }).addToGraph(commandGraph);
  const optimization = new GPUGraphModularityOptimization({
    topology: railTopology,
    initialCommunities: seedCommunities,
    output: communities,
    modularity,
    converged,
    valid,
    iterations: 64
  });
  optimization.addToGraph(commandGraph);
  return {seedCommunities, communities: optimization.output, modularity, converged, valid};
  // #endregion example
}
