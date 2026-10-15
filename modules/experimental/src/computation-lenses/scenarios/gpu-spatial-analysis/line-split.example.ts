// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, type GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPULineSplit, type GPULineGeometryFields} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Uploaded Chicago geometry and parameters for GPULineSplit. */
export type LineSplitExampleInput = {
  roads: GPULineGeometryFields;
  pieceCapacity: number;
  vertexCapacity: number;
  intersectionCapacity: number;
};

/** Runs GPULineSplit and returns the views drawn or inspected by its card. */
export async function runExample(graph: GPUCommandGraph<undefined>, input: LineSplitExampleInput) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const {roads, pieceCapacity, vertexCapacity, intersectionCapacity} = input;
  const pieces = {
    geometry: {
      kind: 'lines' as const,
      positions: createTransientView(graph, 'piecePositions', 'float32x2', vertexCapacity, usage),
      lineOffsets: createTransientView(graph, 'pieceOffsets', 'uint32', pieceCapacity + 1, usage)
    },
    sourceIds: createTransientView(graph, 'sourceIds', 'uint32', pieceCapacity, usage),
    status: {
      count: createTransientView(graph, 'count', 'uint32', 1, usage),
      overflow: createTransientView(graph, 'overflow', 'uint32', 1, usage),
      candidateOverflow: createTransientView(graph, 'candidateOverflow', 'uint32', 1, usage),
      vertexCount: createTransientView(graph, 'vertexCount', 'uint32', 1, usage)
    }
  };
  const uncertainCount = createTransientView(graph, 'uncertainCount', 'uint32', 1, usage);
  // #region example
  graph.add(
    new GPULineSplit({
      lines: {kind: 'lines', ...roads},
      intersectionCapacity,
      output: {pieces, status: {uncertainCount}}
    })
  );
  const {positions: piecePositions, lineOffsets: pieceOffsets} = pieces.geometry;
  // #endregion example
  return {pieces, piecePositions, pieceOffsets, uncertainCount};
}
