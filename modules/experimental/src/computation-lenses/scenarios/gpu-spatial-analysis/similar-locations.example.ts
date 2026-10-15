// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert, Buffer, type Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUSimilarLocations} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Tract attributes the example compares, in matrix column order. */
export const SIMILAR_LOCATIONS_COLUMNS = [
  'poverty150Pct',
  'medianHouseholdIncome',
  'uninsuredPct',
  'noVehiclePct',
  'unemployedPct'
] as const;

/** Row-major attribute matrix, the one-tract reference selection and the packed settings. */
export type SimilarLocationsExampleInput = {
  graph: GPUCommandGraph;
  /** Five attributes of each of the 136 study tracts, row-major. */
  values: GraphDataView<'float32'>;
  /** 1 for the reference tract, else 0. */
  referenceMask: GraphDataView<'uint32'>;
  /** `getGPUSimilarLocationsParameterValues({resultCount: 10}, 5)`. */
  parameters: GraphDataView<'float32'>;
};

/** Runs the exact public-API example shown below the similar locations result figure. */
export async function runExample(device: Device, input: SimilarLocationsExampleInput) {
  const {graph, values, referenceMask, parameters} = input;
  // Every view must belong to the supplied device's graph.
  assert(graph.device === device);
  const count = referenceMask.length;
  const usage = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST;
  const output = {
    ranks: createTransientView(graph, 'ranks', 'uint32', count, usage),
    distances: createTransientView(graph, 'distances', 'float32', count, usage),
    topIds: createTransientView(graph, 'topIds', 'uint32', 32, usage),
    status: {count: createTransientView(graph, 'topCount', 'uint32', 1, usage)}
  };
  // #region example
  // Rank every tract by its distance to the reference tract in standardized attribute space.
  graph.add(
    new GPUSimilarLocations({
      values,
      columnCount: 5,
      referenceMask,
      parameters,
      standardization: 'zscore',
      output
    })
  );
  const {ranks, distances, topIds} = output;
  // #endregion example
  return {
    graph,
    views: {
      values,
      referenceMask,
      distances,
      ranks,
      topIds,
      topCount: output.status.count
    }
  };
}
