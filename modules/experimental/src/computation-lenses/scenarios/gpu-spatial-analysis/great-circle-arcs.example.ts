// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUGreatCircleArcs} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Uploaded O'Hare routes and parameters for GPUGreatCircleArcs. */
export type GreatCircleArcsExampleInput = {
  sources: GraphDataView<'float32x2'>;
  targets: GraphDataView<'float32x2'>;
  parameters: GraphDataView<'float32'>;
  vertexCapacity: number;
};

/** Runs GPUGreatCircleArcs and returns the views drawn or inspected by its card. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  input: GreatCircleArcsExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const {sources, targets, parameters, vertexCapacity} = input;
  const pairCount = sources.length;
  const output = {
    positions: createTransientView(graph, 'arcs', 'float32x2', vertexCapacity, usage),
    lineOffsets: createTransientView(graph, 'arcOffsets', 'uint32', pairCount + 1, usage),
    sourceLineIndices: createTransientView(graph, 'arcSources', 'uint32', pairCount, usage),
    status: {
      count: createTransientView(graph, 'count', 'uint32', 1, usage),
      overflow: createTransientView(graph, 'overflow', 'uint32', 1, usage)
    }
  };
  // #region example
  // Longitude/latitude degrees; the parameters set no length limit and 32 segments per arc.
  graph.add(
    new GPUGreatCircleArcs({
      sources,
      targets,
      parameters,
      spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'},
      output
    })
  );
  const {positions: arcs, lineOffsets} = output;
  // #endregion example
  return {...output, arcs, lineOffsets};
}
