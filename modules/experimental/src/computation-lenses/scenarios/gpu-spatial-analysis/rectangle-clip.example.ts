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
  GPURectangleClip,
  type GPULineGeometryFields
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Roads in UTM 16N metres (planar), the clip rectangle and the output capacities. */
export type RectangleClipExampleInput = {
  roads: GPULineGeometryFields;
  /** `[minX, minY, maxX, maxY]` from `getGPURectangleClipParameterValues`, in metres. */
  parameters: GraphDataView<'float32'>;
  vertexCapacity: number;
  lineCapacity: number;
};

/** Streets clipped to a 900 m by 700 m rectangle in the middle of Wicker Park. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {roads, parameters, vertexCapacity, lineCapacity}: RectangleClipExampleInput
) {
  // #region example
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const output = {
    positions: createTransientView(graph, 'clipped', 'float32x2', vertexCapacity, usage),
    lineOffsets: createTransientView(graph, 'clippedOffsets', 'uint32', lineCapacity + 1, usage),
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
  graph.add(new GPURectangleClip({...roads, geometryType: 'lines', parameters, output}));
  // sourceLineIndices: the road each clipped line came from; a road can leave and re-enter
  const {positions: clipped, lineOffsets, sourceLineIndices} = output;
  // #endregion example
  return {...output, clipped, lineOffsets, sourceLineIndices};
}
