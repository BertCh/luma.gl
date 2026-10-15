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
  GPULineSegmentize,
  type GPULineGeometryFields
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Uploaded Chicago geometry and parameters for GPULineSegmentize. */
export type LineSegmentizeExampleInput = {
  roads: GPULineGeometryFields;
  parameters: GraphDataView<'float32'>;
  vertexCapacity: number;
  lineCapacity: number;
};

/** Runs GPULineSegmentize and returns the views drawn or inspected by its card. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  input: LineSegmentizeExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const {roads, parameters, vertexCapacity, lineCapacity} = input;
  const output = {
    positions: createTransientView(graph, 'densified', 'float32x2', vertexCapacity, usage),
    lineOffsets: createTransientView(graph, 'outputOffsets', 'uint32', lineCapacity + 1, usage),
    sourceLineIndices: createTransientView(
      graph,
      'sourceLineIndices',
      'uint32',
      lineCapacity,
      usage
    ),
    status: {
      count: createTransientView(graph, 'count', 'uint32', 1, usage),
      overflow: createTransientView(graph, 'overflow', 'uint32', 1, usage),
      lineCount: createTransientView(graph, 'lineCount', 'uint32', 1, usage)
    }
  };
  // #region example
  // Parameters: 25 m maximum segment length; roads are UTM 16N metres.
  graph.add(
    new GPULineSegmentize({...roads, parameters, maximumPieceCount: 64, closedLines: false, output})
  );
  const {positions: densified, lineOffsets} = output;
  // #endregion example
  return {...output, densified, lineOffsets};
}
