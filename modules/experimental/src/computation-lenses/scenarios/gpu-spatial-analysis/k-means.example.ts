// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUKMeans, type GPUSpatialContext} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Number of clusters of the example. */
export const K_MEANS_CLUSTER_COUNT = 6;

/** Inputs the lens scenario uploads from `s2-chicago-north-nature.json`. */
export type KMeansExampleInput = {
  /** iNaturalist observations, projected to local-frame metres. */
  observations: GraphDataView<'float32x2'>;
  /** Planar context of the local frame. */
  spatialContext: GPUSpatialContext;
  /** `getGPUKMeansParameterValues({seed: 3})`, uploaded as `uint32` words. */
  parameters: GraphDataView<'uint32'>;
};

/** Six k-means clusters of observations, seeded k-means++ and at most 20 Lloyd iterations. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {observations, spatialContext, parameters}: KMeansExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const create = <Format extends 'float32' | 'uint32' | 'float32x2'>(
    id: string,
    format: Format,
    length: number
  ) => createTransientView(graph, id, format, length, usage);
  const labels = create('labels', 'uint32', observations.length);
  const centers = create('centers', 'float32x2', K_MEANS_CLUSTER_COUNT);
  const sizes = create('sizes', 'uint32', K_MEANS_CLUSTER_COUNT);
  const squaredDistances = create('squaredDistances', 'float32', observations.length);
  const inertia = create('inertia', 'float32', 1);
  const iterationCount = create('iterationCount', 'uint32', 1);
  const converged = create('converged', 'uint32', 1);
  // #region example
  graph.add(
    new GPUKMeans({
      spatialContext,
      positions: observations,
      k: 6,
      maximumIterations: 20,
      initialization: 'kmeans++',
      parameters,
      output: {
        labels,
        centers,
        sizes,
        squaredDistances,
        inertia,
        status: {iterationCount, converged}
      }
    })
  );
  // inertia is the sum of squaredDistances; converged is 1 when the centers stopped moving
  // #endregion example
  return {labels, centers, sizes, squaredDistances, inertia, iterationCount, converged};
}
