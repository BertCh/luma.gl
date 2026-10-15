// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPULineChunk, type GPULineGeometryFields} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Uploaded Chicago geometry and parameters for GPULineChunk. */
export type LineChunkExampleInput = {
  roads: GPULineGeometryFields;
  parameters: GraphDataView<'float32'>;
  vertexCapacity: number;
  lineCapacity: number;
};

/** Runs GPULineChunk and returns the views drawn or inspected by its card. */
export async function runExample(graph: GPUCommandGraph<undefined>, input: LineChunkExampleInput) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const {roads, parameters, vertexCapacity, lineCapacity} = input;
  const output = {
    positions: createTransientView(graph, 'chunks', 'float32x2', vertexCapacity, usage),
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
  // Parameters: 100 m chunks; roads are UTM 16N metres.
  graph.add(new GPULineChunk({...roads, parameters, mode: 'chunk', output}));
  const {positions: chunks, lineOffsets} = output;
  // #endregion example
  return {...output, chunks, lineOffsets};
}
