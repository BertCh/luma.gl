// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  GPUGather,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUDelaunayTessellation} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Places in UTM 16N metres (planar, as the element requires) and the uploaded triangle rows. */
export type DelaunayTessellationExampleInput = {
  places: GraphDataView<'float32x2'>;
  /** Triangle capacity rows, uploaded as row 0 so slots past the written count stay degenerate. */
  triangleSlots: GraphDataView<'uint32x3'>;
};

/** Delaunay triangles of the places, plus the triangle corners gathered for display. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {places, triangleSlots}: DelaunayTessellationExampleInput
) {
  // #region example
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const scalar = (id: string) => createTransientView(graph, id, 'uint32', 1, usage);
  const output = {
    triangles: triangleSlots,
    status: {
      count: scalar('count'),
      requiredCount: scalar('requiredCount'),
      overflow: scalar('overflow'),
      duplicateCount: scalar('duplicateCount')
    }
  };
  graph.add(new GPUDelaunayTessellation({positions: places, output}));
  // triangles: counter-clockwise triples of place rows; the first status.count rows are written
  const {triangles} = output;
  // #endregion example
  // Display only: gather the corners of every triangle on the GPU.
  const triangleCorners = createTransientView(
    graph,
    'triangleCorners',
    'float32x2',
    3 * triangles.length,
    usage
  );
  const indices = graph.createDataView(triangles.buffer, {
    format: 'uint32',
    length: 3 * triangles.length
  });
  graph.add(new GPUGather({source: places, indices, output: triangleCorners}));
  return {...output, triangles, triangleCorners};
}
