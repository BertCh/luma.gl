// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert, Buffer, type Device} from '@luma.gl/core';
import {createTransientView, type GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPULatticeWeights,
  GPUSpatialWeightsSummary
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** A recording graph and the lattice shape; the caller owns the graph and its uploads. */
export type LatticeWeightsInput = {graph: GPUCommandGraph; width: number; height: number};

/** Runs the public-API example shown below the lattice weights result figure. */
export async function runExample(device: Device, input: LatticeWeightsInput) {
  const {graph, width, height} = input;
  // Every view must belong to the supplied device's graph.
  assert(graph.device === device);
  const count = width * height;
  // A queen lattice has at most eight neighbors per cell.
  const capacity = count * 8;
  const usage = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST;
  const createFloat = (name: string, length = count) =>
    createTransientView(graph, name, 'float32', length, usage);
  const createInteger = (name: string, length = count) =>
    createTransientView(graph, name, 'uint32', length, usage);
  const weightOutput = {
    offsets: createInteger('weightsRowOffsets', count + 1),
    neighbors: createInteger('weightsNeighbors', capacity),
    weights: createFloat('weightsValues', capacity),
    status: {overflow: createInteger('weightsOverflow', 1)}
  };
  const statistics = createFloat('statistics', 3);
  const counts = createInteger('counts', 6);
  const neighborCounts = createInteger('neighborCounts');
  // #region example
  const lattice = new GPULatticeWeights({
    width,
    height,
    criterion: 'queen',
    radius: 1,
    output: weightOutput
  });
  graph.add(lattice);
  const weights = lattice.outputWeights;
  // #endregion example
  graph.add(
    new GPUSpatialWeightsSummary({
      weights,
      output: {statistics, counts, cardinality: neighborCounts}
    })
  );
  return {
    graph,
    weights,
    views: {
      weightsRowOffsets: weights.offsets,
      weightsNeighbors: weights.neighbors,
      weightsValues: weights.weights,
      weightsOverflow: weights.status!.overflow!,
      statistics,
      counts,
      neighborCounts
    }
  };
}
