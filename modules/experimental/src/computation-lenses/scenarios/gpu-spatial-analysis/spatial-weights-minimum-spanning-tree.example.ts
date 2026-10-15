// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert, Buffer, type Device} from '@luma.gl/core';
import {
  createTransientView,
  GPUGather,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUContiguityWeights,
  GPUGeometryMeasures,
  GPUSpatialWeightsMinimumSpanningTree,
  type GPUContiguityWeightsProps
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Uploaded tracts and one poverty column; the caller owns the recording graph. */
export type MinimumSpanningTreeInput = {
  graph: GPUCommandGraph;
  tracts: GPUContiguityWeightsProps['polygons'] & {positions: GraphDataView<'float32x2'>};
  values: GraphDataView<'float32'>;
};

/** Runs the exact public-API example shown below the minimum spanning tree result figure. */
export async function runExample(device: Device, input: MinimumSpanningTreeInput) {
  const {graph, tracts, values} = input;
  // Every view must belong to the supplied device's graph.
  assert(graph.device === device);
  const count = values.length;
  const usage = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST;
  const createFloat = (name: string, length = count) =>
    createTransientView(graph, name, 'float32', length, usage);
  const createInteger = (name: string, length = count) =>
    createTransientView(graph, name, 'uint32', length, usage);
  // 2K slots cover the queen topology of the 136 study tracts; a connected graph has count - 1 tree edges.
  const weightOutput = {
    offsets: createInteger('weightsRowOffsets', count + 1),
    neighbors: createInteger('weightsNeighbors', 2048),
    weights: createFloat('weightsValues', 2048),
    status: {overflow: createInteger('weightsOverflow', 1)}
  };
  const output = {
    treeEdgeFlags: createInteger('treeEdgeFlags', 2048),
    componentLabels: createInteger('componentLabels'),
    standardizedValues: createFloat('standardizedValues'),
    edges: {
      ids: createInteger('edgeIds', count - 1),
      status: {count: createInteger('edgeCount', 1), overflow: createInteger('edgeOverflow', 1)}
    },
    edgeEndpoints: createInteger('edgeEndpoints', 2 * (count - 1)),
    edgeDissimilarities: createFloat('edgeDissimilarities', count - 1)
  };
  const centroids = createTransientView(graph, 'centroids', 'float32x2', count, usage);
  const treeSegments = createTransientView(
    graph,
    'treeSegments',
    'float32x2',
    2 * (count - 1),
    usage
  );
  // #region example
  const queen = new GPUContiguityWeights({
    polygons: tracts,
    criterion: 'queen',
    output: weightOutput
  });
  graph.add(queen);
  // The edge cost is the squared difference of standardized poverty between two tracts.
  graph.add(
    new GPUSpatialWeightsMinimumSpanningTree({weights: queen.outputWeights, values, output})
  );
  const {treeEdgeFlags, edgeDissimilarities} = output;
  // #endregion example
  graph.add(
    new GPUGeometryMeasures({
      geometryType: 'polygons',
      positions: tracts.positions,
      ringOffsets: tracts.ringOffsets,
      featureOffsets: tracts.featureOffsets,
      polygonOffsets: tracts.polygonOffsets,
      spatialContext: {
        coordinateSpace: 'longitude-latitude',
        metric: 'great-circle',
        units: 'meters'
      },
      output: {centroids}
    })
  );
  // Each tree edge becomes a two-vertex segment between the centroids of its tracts.
  graph.add(
    new GPUGather({source: centroids, indices: output.edgeEndpoints, output: treeSegments})
  );
  return {
    graph,
    weights: queen.outputWeights,
    views: {
      values,
      treeEdgeFlags,
      componentLabels: output.componentLabels,
      standardizedValues: output.standardizedValues,
      edgeEndpoints: output.edgeEndpoints,
      edgeDissimilarities,
      treeSegments,
      weightsRowOffsets: weightOutput.offsets,
      weightsNeighbors: weightOutput.neighbors,
      weightsOverflow: weightOutput.status.overflow,
      edgeOverflow: output.edges.status.overflow
    }
  };
}
