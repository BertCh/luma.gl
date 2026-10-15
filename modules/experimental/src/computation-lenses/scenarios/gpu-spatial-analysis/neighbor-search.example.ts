// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert, Buffer, type Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUGeometryMeasures,
  GPUSpatialWeightsSummary,
  GPUNeighborSearch,
  type GPUContiguityWeightsProps
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Runs the public-API example used by this weights card. */
export async function runExample(
  device: Device,
  input: {
    graph: GPUCommandGraph;
    tracts: GPUContiguityWeightsProps['polygons'] & {positions: GraphDataView<'float32x2'>};
    values: GraphDataView<'float32'>;
    searchParameters: GraphDataView<'float32'>;
    colorParameters: GraphDataView<'uint32'>;
  }
) {
  const {graph, tracts, searchParameters} = input;
  // Inputs and outputs must belong to this device.
  assert(graph.device === device);
  const count = tracts.featureOffsets.length - 1;
  const capacity = 32768;
  const usage = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST;
  const createFloat = (name: string, length = count) =>
    createTransientView(graph, name, 'float32', length, usage);
  const createInteger = (name: string, length = count) =>
    createTransientView(graph, name, 'uint32', length, usage);
  const createWeights = (prefix: string) => ({
    offsets: createInteger(`${prefix}RowOffsets`, count + 1),
    neighbors: createInteger(`${prefix}Neighbors`, capacity),
    weights: createFloat(`${prefix}Values`, capacity),
    status: {overflow: createInteger(`${prefix}Overflow`, 1)}
  });
  const weightOutput = {...createWeights('weights'), distances: createFloat('distances', capacity)};
  const statistics = createFloat('statistics', 3);
  const counts = createInteger('counts', 6);
  const neighborCounts = createInteger('neighborCounts');
  const summaryOutput = {statistics, counts, cardinality: neighborCounts};
  const centroids = createTransientView(graph, 'centroids', 'float32x2', count, usage);
  const spatialContext = {
    coordinateSpace: 'longitude-latitude',
    metric: 'great-circle',
    units: 'meters'
  } as const;
  // #region example
  graph.add(
    new GPUGeometryMeasures({
      geometryType: 'polygons',
      positions: tracts.positions,
      ringOffsets: tracts.ringOffsets,
      featureOffsets: tracts.featureOffsets,
      polygonOffsets: tracts.polygonOffsets,
      spatialContext,
      output: {centroids}
    })
  );
  const search = new GPUNeighborSearch({
    positions: centroids,
    spatialContext,
    mode: 'knn',
    k: 4,
    gridSize: [16, 16],
    parameters: searchParameters,
    rowStandardize: false,
    weightTransform: 'binary',
    output: weightOutput
  });
  graph.add(search);
  const weights = search.outputWeights;
  const {offsets: weightsRowOffsets, neighbors: weightsNeighbors, weights: weightsValues} = weights;
  // #endregion example
  graph.add(new GPUSpatialWeightsSummary({weights, output: summaryOutput}));
  return {
    graph,
    weights,
    views: {
      weightsRowOffsets,
      weightsNeighbors,
      weightsValues,
      weightsOverflow: weights.status!.overflow!,
      statistics: statistics,
      counts: counts,
      neighborCounts: neighborCounts,
      distances: weightOutput.distances,
      centroids: centroids
    }
  };
}
