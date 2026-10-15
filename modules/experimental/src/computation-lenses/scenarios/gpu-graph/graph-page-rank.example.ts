// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUVector} from '@luma.gl/gpgpu/gpu-data';
import {GPUGraphPageRank, type GPUGraphTopology} from '@luma.gl/gpgpu/gpu-graph';

/** Declares the PageRank run used by the result figure, without submitting work. */
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
  const createRankColumn = (name: string, length: number) =>
    new GPUVector({
      type: 'buffer',
      name,
      format: 'float32',
      length,
      ownsBuffer: true,
      buffer: device.createBuffer({byteLength: length * 4, usage})
    });
  const ranks = createRankColumn('ranks', stations.vertexCount);
  const residual = createRankColumn('residual', 1);
  const pageRank = new GPUGraphPageRank({
    topology: railTopology,
    output: ranks,
    damping: 0.85,
    iterations: 40,
    residual
  });
  pageRank.addToGraph(commandGraph);
  return {ranks: pageRank.output, residual: pageRank.residual};
  // #endregion example
}
