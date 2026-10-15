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
  getGPUSpatialScanStatisticParameterValues,
  GPU_SPATIAL_SCAN_STATISTIC_PARAMETER_LENGTH,
  GPUParameterBuffer,
  GPUSpatialScanStatistic,
  type GPUSpatialContext
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Inputs the lens scenario uploads from `s2-midwest-counties.json`. */
export type SpatialScanStatisticExampleInput = {
  /** County centroids projected to local-frame metres. */
  centroids: GraphDataView<'float32x2'>;
  /** Planar context of the local frame. */
  spatialContext: GPUSpatialContext;
  /** Traffic deaths 2017-2023 (FARS) per county. */
  deaths: GraphDataView<'uint32'>;
  /** Person-years at risk per county (seven times the 2020 census population). */
  personYears: GraphDataView<'float32'>;
};

/** Circular windows of counties whose traffic death rate is higher than elsewhere. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {centroids, spatialContext, deaths, personYears}: SpatialScanStatisticExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const counties = centroids.length;
  const clusterIndices = createTransientView(graph, 'clusterIndices', 'uint32', 20, usage);
  const clusterStatistics = createTransientView(graph, 'clusterStatistics', 'float32', 40, usage);
  const statistics = createTransientView(graph, 'statistics', 'float32', 200, usage);
  const summary = createTransientView(graph, 'summary', 'uint32', 4, usage);
  const zoneStatistics = createTransientView(graph, 'zoneStatistics', 'float32', counties, usage);
  // #region example
  const scanParameters = new GPUParameterBuffer(graph.device, {
    id: 'scan-parameters',
    format: 'uint32',
    length: GPU_SPATIAL_SCAN_STATISTIC_PARAMETER_LENGTH,
    values: getGPUSpatialScanStatisticParameterValues({
      seed: 1,
      permutationCount: 199,
      maximumPopulationFraction: 0.5,
      rateDirection: 'high'
    })
  });
  graph.add(
    new GPUSpatialScanStatistic({
      positions: centroids,
      spatialContext,
      model: 'poisson',
      cases: deaths,
      baseline: personYears,
      maximumPermutations: 199,
      maximumClusters: 5,
      parameters: scanParameters.importToGraph(graph),
      output: {clusterIndices, clusterStatistics, statistics, summary, zoneStatistics}
    })
  );
  // zoneStatistics: best log-likelihood ratio of any window centred on each county
  // #endregion example
  return {
    clusterIndices,
    clusterStatistics,
    statistics,
    summary,
    zoneStatistics,
    scanParameters
  };
}
