// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  GPULocalOutlierFactor,
  GPUNeighborSearch,
  type GPUSpatialContext
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Neighbours per observation in the kNN table. */
export const LOCAL_OUTLIER_FACTOR_NEIGHBOR_COUNT = 10;

/** Inputs the lens scenario uploads from `s2-chicago-north-nature.json`. */
export type LocalOutlierFactorExampleInput = {
  /** iNaturalist observations, projected to local-frame metres. */
  observations: GraphDataView<'float32x2'>;
  /** Planar context of the local frame. */
  spatialContext: GPUSpatialContext;
  /** `getGPUNeighborSearchParameterValues({bounds})` of the observation extent. */
  searchParameters: GraphDataView<'float32'>;
  /** `getGPULocalOutlierFactorParameterValues({threshold: 3, densityFloor: 10})`. */
  parameters: GraphDataView<'float32'>;
};

/** Local outlier factor of observations over their ten nearest neighbours. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {observations, spatialContext, searchParameters, parameters}: LocalOutlierFactorExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const count = observations.length;
  const slots = count * LOCAL_OUTLIER_FACTOR_NEIGHBOR_COUNT;
  const create = <Format extends 'float32' | 'uint32'>(
    id: string,
    format: Format,
    length: number
  ) => createTransientView(graph, id, format, length, usage);
  const table = {
    offsets: create('neighborOffsets', 'uint32', count + 1),
    neighbors: create('neighborIds', 'uint32', slots),
    weights: create('neighborWeights', 'float32', slots),
    distances: create('neighborDistances', 'float32', slots),
    status: {overflow: create('neighborOverflow', 'uint32', 1)}
  };
  const output = {
    kDistance: create('kDistance', 'float32', count),
    localReachabilityDensity: create('density', 'float32', count),
    lof: create('lof', 'float32', count),
    outlier: create('outlier', 'uint32', count),
    outlierCount: create('outlierCount', 'uint32', 1)
  };
  // #region example
  // A true kNN list with distances: the self join of the observations, ten neighbours each.
  const search = new GPUNeighborSearch({
    spatialContext,
    mode: 'knn',
    k: 10,
    positions: observations,
    parameters: searchParameters,
    gridSize: [32, 32],
    output: table
  });
  graph.add(search);
  graph.add(
    new GPULocalOutlierFactor({
      spatialContext,
      neighbors: search.outputWeights,
      parameters,
      output
    })
  );
  // lof near 1 is as dense as the neighbours; outlier flags lof above the chosen threshold
  // #endregion example
  return {...output, neighborOverflow: table.status.overflow};
}
