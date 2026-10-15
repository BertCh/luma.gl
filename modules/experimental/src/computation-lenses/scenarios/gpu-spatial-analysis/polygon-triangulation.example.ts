// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, GPUGather, type GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUPolygonTriangulation,
  getPolygonTriangulationIndexCount,
  type GPUPolygonGeometryFields
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Parks uploaded by the lens scenario, in UTM 16N metres (planar, as the element requires). */
export type PolygonTriangulationExampleInput = {parks: GPUPolygonGeometryFields};

/** Earcut triangle indices of every park, plus the triangle corners gathered for display. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {parks}: PolygonTriangulationExampleInput
) {
  // #region example
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const vertexCount = parks.positions.length;
  const ringCount = parks.ringOffsets.length - 1;
  const parkCount = parks.polygonOffsets.length - 1;
  const output = {
    indices: createTransientView(
      graph,
      'triangleIndices',
      'uint32',
      getPolygonTriangulationIndexCount(vertexCount, ringCount, parkCount),
      usage
    ),
    valid: createTransientView(graph, 'valid', 'uint32', parkCount, usage),
    triangleCount: createTransientView(graph, 'triangleCount', 'uint32', parkCount, usage),
    triangleOffsets: createTransientView(graph, 'triangleOffsets', 'uint32', parkCount + 1, usage)
  };
  graph.add(new GPUPolygonTriangulation({...parks, output}));
  // indices: three rows of parks.positions per triangle; park p starts at 3 * triangleOffsets[p]
  const {indices, valid, triangleCount} = output;
  // #endregion example
  // Display only: gather the corners of every triangle on the GPU.
  const triangles = createTransientView(graph, 'triangles', 'float32x2', indices.length, usage);
  graph.add(new GPUGather({source: parks.positions, indices, output: triangles}));
  return {...output, indices, valid, triangleCount, triangles};
}
