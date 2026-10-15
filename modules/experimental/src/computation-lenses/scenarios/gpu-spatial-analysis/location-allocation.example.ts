// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert, Buffer, type Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPULocationAllocation} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Uploaded demand-by-facility kilometre costs and the population weight of each demand tract. */
export type LocationAllocationInput = {
  graph: GPUCommandGraph;
  costs: GraphDataView<'float32'>;
  demandWeights: GraphDataView<'float32'>;
};

/** Runs the exact public-API example shown below the location-allocation result figure. */
export async function runExample(device: Device, input: LocationAllocationInput) {
  const {graph, costs, demandWeights} = input;
  // Every view must belong to the supplied device's graph.
  assert(graph.device === device);
  const count = demandWeights.length;
  const usage = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST;
  const createFloat = (name: string, length = count) =>
    createTransientView(graph, name, 'float32', length, usage);
  const createInteger = (name: string, length = count) =>
    createTransientView(graph, name, 'uint32', length, usage);
  const output = {
    selectedFacilities: createInteger('selectedFacilities'),
    assignments: createInteger('assignments'),
    assignedCosts: createFloat('assignedCosts'),
    objective: createFloat('objective', 1),
    status: createInteger('status', 7)
  };
  // #region example
  // Every tract is both a demand row and a candidate facility; choose five to minimize
  // population-weighted distance, then improve the greedy choice by up to eight swaps.
  graph.add(
    new GPULocationAllocation({
      costs,
      facilityCount: count,
      demandWeights,
      operation: 'p-median',
      maximumFacilityCount: 5,
      maximumSwapIterations: 8,
      output
    })
  );
  const {assignments, selectedFacilities, objective} = output;
  // #endregion example
  return {
    graph,
    views: {
      demandWeights,
      ...output,
      assignments,
      selectedFacilities,
      objective
    }
  };
}
