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
  GPUCoverageSimplification,
  type GPUPolygonGeometryFields
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Uploaded Chicago community areas and tolerance for GPUCoverageSimplification. */
export type CoverageSimplificationExampleInput = {
  areas: GPUPolygonGeometryFields;
  parameters: GraphDataView<'float32'>;
};

/** Runs GPUCoverageSimplification and returns the views drawn or inspected by its card. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  input: CoverageSimplificationExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const {areas, parameters} = input;
  const output = {
    positions: createTransientView(
      graph,
      'simplifiedAreas',
      'float32x2',
      areas.positions.length,
      usage
    ),
    ringOffsets: createTransientView(
      graph,
      'simplifiedOffsets',
      'uint32',
      areas.ringOffsets.length,
      usage
    ),
    outputMask: createTransientView(graph, 'outputMask', 'uint32', areas.positions.length, usage),
    topologyStatistics: createTransientView(graph, 'topologyStatistics', 'uint32', 4, usage),
    status: {
      overflow: createTransientView(graph, 'overflow', 'uint32', 1, usage),
      converged: createTransientView(graph, 'converged', 'uint32', 1, usage)
    }
  };
  // #region example
  // Parameters: 50 m Douglas-Peucker tolerance; areas are UTM 16N metres.
  graph.add(
    new GPUCoverageSimplification({
      ...areas,
      parameters,
      maximumIterations: 32,
      topologyIterationCount: 4,
      output
    })
  );
  const {positions: simplifiedAreas, ringOffsets} = output;
  // #endregion example
  return {...output, simplifiedAreas, ringOffsets};
}
