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
  GPUVertexSnap,
  type GPUPolygonGeometryFields
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Parks, community-area vertices (UTM 16N metres) and the packed snap tolerance (50 m). */
export type VertexSnapExampleInput = {
  parks: GPUPolygonGeometryFields;
  areaVertices: GraphDataView<'float32x2'>;
  parameters: GraphDataView<'float32'>;
};

/** Snaps park vertices onto nearby community-area vertices. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {parks, areaVertices, parameters}: VertexSnapExampleInput
) {
  // #region example
  // Parameters: tolerance 50 m; a vertex moves only if an area vertex is strictly closer.
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const vertexCount = parks.positions.length;
  const output = {
    positions: createTransientView(graph, 'snappedParks', 'float32x2', vertexCount, usage),
    referenceRows: createTransientView(graph, 'referenceRows', 'uint32', vertexCount, usage)
  };
  graph.add(
    new GPUVertexSnap({
      positions: parks.positions,
      referencePositions: areaVertices,
      parameters,
      output
    })
  );
  // referenceRows: the area vertex a park vertex moved to, or 0xffffffff when it stayed
  const {positions: snappedParks, referenceRows} = output;
  // #endregion example
  return {snappedParks, referenceRows};
}
