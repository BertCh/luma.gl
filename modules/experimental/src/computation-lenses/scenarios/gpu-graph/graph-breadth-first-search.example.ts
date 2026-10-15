// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUVector} from '@luma.gl/gpgpu/gpu-data';
import {GPUGraphBreadthFirstSearch, type GPUGraphTopology} from '@luma.gl/gpgpu/gpu-graph';

/** Declares hop distances from Utrecht Centraal over the rail CSR, without submitting work. */
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
  const createColumn = (name: string, length: number, data?: Uint32Array) =>
    new GPUVector({
      type: 'buffer',
      name,
      format: 'uint32',
      length,
      ownsBuffer: true,
      buffer: device.createBuffer(data ? {data, usage} : {byteLength: length * 4, usage})
    });
  const utrecht = stations.names.indexOf('Utrecht Centraal');
  const search = new GPUGraphBreadthFirstSearch({
    topology: railTopology,
    seeds: createColumn('seeds', 1, Uint32Array.of(utrecht)),
    distances: createColumn('hops', stations.vertexCount),
    predecessors: createColumn('parents', stations.vertexCount),
    maxDepth: 16
  });
  search.addToGraph(commandGraph);
  return {hops: search.distances, parents: search.predecessors, seeds: search.seeds};
  // #endregion example
}
