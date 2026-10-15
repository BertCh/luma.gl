// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert, Buffer, type Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUGeographicDistribution} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Number of vertices of the drawn ellipse ring. */
export const GEOGRAPHIC_DISTRIBUTION_RING_VERTICES = 32;

/** Planar tract centroids (metres), population weights and packed settings; the caller owns the graph. */
export type GeographicDistributionInput = {
  graph: GPUCommandGraph;
  positions: GraphDataView<'float32x2'>;
  weights: GraphDataView<'float32'>;
  parameters: GraphDataView<'float32'>;
};

/** Runs the exact public-API example shown below the geographic distribution result figure. */
export async function runExample(device: Device, input: GeographicDistributionInput) {
  const {graph, positions, weights, parameters} = input;
  // Every view must belong to the supplied device's graph.
  assert(graph.device === device);
  const usage = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST;
  const counts = createTransientView(graph, 'counts', 'uint32', 1, usage);
  const weightSums = createTransientView(graph, 'weightSums', 'float32', 1, usage);
  const meanCenters = createTransientView(graph, 'meanCenters', 'float32x2', 1, usage);
  const standardDistances = createTransientView(graph, 'standardDistances', 'float32', 1, usage);
  const ellipses = createTransientView(graph, 'ellipses', 'float32', 3, usage);
  const ellipseVertices = createTransientView(
    graph,
    'ellipseVertices',
    'float32x2',
    GEOGRAPHIC_DISTRIBUTION_RING_VERTICES,
    usage
  );
  // #region example
  // One group: the population-weighted centre, spread and ellipse of all tract centroids.
  graph.add(
    new GPUGeographicDistribution({
      positions,
      weights,
      parameters,
      polygonVertexCount: GEOGRAPHIC_DISTRIBUTION_RING_VERTICES,
      output: {counts, weightSums, meanCenters, standardDistances, ellipses, ellipseVertices}
    })
  );
  // #endregion example
  return {
    graph,
    encodeOptions: {parameters: undefined},
    views: {
      positions,
      weights,
      counts,
      weightSums,
      meanCenters,
      standardDistances,
      ellipses,
      ellipseVertices
    }
  };
}
