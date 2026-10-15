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
  GPUOutlineGeometry,
  getGPUOutlineGeometryVerticesPerInput,
  type GPUPolygonGeometryFields
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Parks in UTM 16N metres and the packed outline distance (100 m) for GPUOutlineGeometry. */
export type OutlineGeometryExampleInput = {
  parks: GPUPolygonGeometryFields;
  parameters: GraphDataView<'float32'>;
};

/** Outline triangles of every park, drawn as a round-joined stroke. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {parks, parameters}: OutlineGeometryExampleInput
) {
  // #region example
  // Parameters: 100 m outline distance; UTM 16N metres, so the context is planar.
  const joinSegments = 8;
  const rowCount = parks.positions.length * getGPUOutlineGeometryVerticesPerInput(joinSegments);
  const output = {
    outline: createTransientView(
      graph,
      'outline',
      'float32x2',
      rowCount,
      Buffer.STORAGE | Buffer.COPY_SRC
    )
  };
  graph.add(
    new GPUOutlineGeometry({
      ...parks,
      geometryType: 'polygons',
      joinSegments,
      parameters,
      output: {positions: output.outline}
    })
  );
  // outline: a triangle list, 3 * (joinSegments + 2) vertices for every park vertex
  const {outline} = output;
  // #endregion example
  return {outline};
}
