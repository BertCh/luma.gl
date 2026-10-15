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
  GPUShapeGenerator,
  getGPUShapeVertexCount
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Circle centres and radii in UTM 16N metres (planar), and the per-frame segment count. */
export type ShapeGeneratorExampleInput = {
  centers: GraphDataView<'float32x2'>;
  /** One radius per centre, in metres. */
  radii: GraphDataView<'float32'>;
  /** `[segmentCount, radiusScale]` from `getGPUShapeGeneratorParameterValues`. */
  parameters: GraphDataView<'float32'>;
};

/** A circle ring of 24 segments around every selected place. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {centers, radii, parameters}: ShapeGeneratorExampleInput
) {
  // #region example
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const maximumSegments = 32;
  const output = {
    positions: createTransientView(
      graph,
      'circleVertices',
      'float32x2',
      centers.length * getGPUShapeVertexCount('circle', maximumSegments),
      usage
    ),
    offsets: createTransientView(graph, 'circleOffsets', 'uint32', centers.length + 1, usage),
    vertexCount: createTransientView(graph, 'vertexCount', 'uint32', 1, usage)
  };
  graph.add(
    new GPUShapeGenerator({shape: 'circle', maximumSegments, centers, radii, parameters, output})
  );
  // positions: segmentCount + 1 vertices per circle; offsets: where each circle's ring starts
  const {positions, offsets} = output;
  // #endregion example
  return {...output, positions, offsets};
}
