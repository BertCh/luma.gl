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
  getGPUNeighborSearchParameterValues,
  GPUCatchmentAccessibility,
  GPUNeighborSearch,
  GPUParameterBuffer,
  type GPUSpatialContext
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Inputs the lens scenario uploads from the facility and tract data files. */
export type CatchmentAccessibilityExampleInput = {
  /** Library branches, projected to local-frame metres. */
  libraries: GraphDataView<'float32x2'>;
  /** Supply per library (1 for every branch). */
  supply: GraphDataView<'float32'>;
  /** Census-tract centroids, projected to local-frame metres. */
  tractCentroids: GraphDataView<'float32x2'>;
  /** Residents per tract. */
  population: GraphDataView<'float32'>;
  /** Planar context of the local frame. */
  spatialContext: GPUSpatialContext;
  /** Search extent `[minX, minY, maxX, maxY]` in frame metres, covering every point. */
  bounds: readonly [number, number, number, number];
};

/** Two-step floating catchment accessibility of tract residents to libraries, 2 km Euclidean. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {
    libraries,
    supply,
    tractCentroids,
    population,
    spatialContext,
    bounds
  }: CatchmentAccessibilityExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const slotCapacity = 1024;
  const createRows = (name: string, rowCount: number) => ({
    offsets: createTransientView(graph, `${name}Offsets`, 'uint32', rowCount + 1, usage),
    neighbors: createTransientView(graph, `${name}Neighbors`, 'uint32', slotCapacity, usage),
    weights: createTransientView(graph, `${name}Weights`, 'float32', slotCapacity, usage),
    status: {overflow: createTransientView(graph, `${name}Overflow`, 'uint32', 1, usage)}
  });
  const facilityRows = createRows('facility', libraries.length);
  const tractRows = createRows('tract', tractCentroids.length);
  const accessibility = createTransientView(
    graph,
    'accessibility',
    'float32',
    tractCentroids.length,
    usage
  );
  const ratios = createTransientView(graph, 'ratios', 'float32', libraries.length, usage);
  const reachableFacilities = createTransientView(
    graph,
    'reachableFacilities',
    'uint32',
    tractCentroids.length,
    usage
  );
  const searchParameters = new GPUParameterBuffer(graph.device, {
    id: 'catchment-search',
    format: 'float32',
    length: 12,
    values: getGPUNeighborSearchParameterValues({bounds, radius: 2000, weightKind: 'binary'})
  });
  const search = {
    mode: 'radius',
    gridSize: [16, 16],
    spatialContext,
    weightTransform: 'binary',
    rowStandardize: false,
    parameters: searchParameters.importToGraph(graph)
  } as const;
  const libraryCatchments = new GPUNeighborSearch({
    ...search,
    id: 'library-catchments',
    positions: tractCentroids,
    queryPositions: libraries,
    output: facilityRows
  });
  const tractReach = new GPUNeighborSearch({
    ...search,
    id: 'tract-reach',
    positions: libraries,
    queryPositions: tractCentroids,
    output: tractRows
  });
  graph.add(libraryCatchments);
  graph.add(tractReach);
  // #region example
  // facilityWeights and demandWeights: binary 2 km radius searches (Euclidean, not network paths)
  graph.add(
    new GPUCatchmentAccessibility({
      method: '2sfca',
      supply,
      demand: population,
      facilityWeights: libraryCatchments.outputWeights,
      demandWeights: tractReach.outputWeights,
      output: {accessibility, ratios, reachableFacilities}
    })
  );
  // accessibility: libraries per resident reachable from each tract centroid
  // #endregion example
  return {
    accessibility,
    ratios,
    reachableFacilities,
    facilityOverflow: facilityRows.status.overflow,
    tractOverflow: tractRows.status.overflow,
    searchParameters
  };
}
