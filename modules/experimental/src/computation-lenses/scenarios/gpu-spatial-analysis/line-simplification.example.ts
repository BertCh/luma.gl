// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView,
  GPUGather,
  GPUScan
} from '@luma.gl/gpgpu/gpu-core';
import {
  GPULineSimplification,
  type GPULineGeometryFields
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Uploaded Chicago geometry and parameters for GPULineSimplification. */
export type LineSimplificationExampleInput = {
  roads: GPULineGeometryFields;
  parameters: GraphDataView<'float32'>;
};

/** Runs GPULineSimplification and returns the views drawn or inspected by its card. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  input: LineSimplificationExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const {roads, parameters} = input;
  const vertexCount = roads.positions.length;
  const roadCount = roads.lineOffsets.length - 1;
  const importance = createTransientView(graph, 'importance', 'float32', vertexCount, usage);
  const countStorage = createTransientView(graph, 'lineCounts', 'uint32', roadCount + 1, usage);
  const lineCounts = graph.createDataView(countStorage.buffer, {
    format: 'uint32',
    length: roadCount
  });
  const output = {
    ids: createTransientView(graph, 'ids', 'uint32', vertexCount, usage),
    lineCounts,
    status: {
      count: createTransientView(graph, 'count', 'uint32', 1, usage),
      overflow: createTransientView(graph, 'overflow', 'uint32', 1, usage),
      converged: createTransientView(graph, 'converged', 'uint32', 1, usage),
      iterationCount: createTransientView(graph, 'iterationCount', 'uint32', 1, usage)
    }
  };
  const simplifiedRoads = createTransientView(
    graph,
    'simplifiedRoads',
    'float32x2',
    vertexCount,
    usage
  );
  const simplifiedOffsets = createTransientView(
    graph,
    'simplifiedOffsets',
    'uint32',
    roadCount + 1,
    usage
  );
  // Exclusive scan includes a terminal row; its input value cannot affect any output prefix.
  // #region example
  // Keep endpoints and vertices whose Douglas-Peucker importance exceeds 5 m.
  graph.add(
    new GPULineSimplification({
      ...roads,
      parameters,
      importance,
      method: 'douglas-peucker',
      metric: 'segment',
      maximumIterations: 64,
      computeImportance: true,
      output
    })
  );
  graph.add(new GPUGather({source: roads.positions, indices: output.ids, output: simplifiedRoads}));
  graph.add(new GPUScan({input: countStorage, output: simplifiedOffsets, mode: 'exclusive'}));
  const simplified = {positions: simplifiedRoads, lineOffsets: simplifiedOffsets};
  // #endregion example
  return {...output, importance, simplified};
}
