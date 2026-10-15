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
  GPUDelaunayTessellation,
  GPUVoronoiDiagram
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Places in UTM 16N metres (planar) and the uploaded output slots of the two elements. */
export type VoronoiDiagramExampleInput = {
  places: GraphDataView<'float32x2'>;
  /** Triangle capacity rows, uploaded as row 0; only the written prefix is read. */
  triangleSlots: GraphDataView<'uint32x3'>;
  /** Segment capacity rows, uploaded as NaN so unwritten slots are not drawn. */
  segmentSlots: GraphDataView<'float32x4'>;
  /** One row `[minX, minY, maxX, maxY]` that hull rays are clipped to, in metres. */
  clipBounds: GraphDataView<'float32x4'>;
};

/** Voronoi cell edges of the places, from their Delaunay triangles. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {places, triangleSlots, segmentSlots, clipBounds}: VoronoiDiagramExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const scalar = (id: string) => createTransientView(graph, id, 'uint32', 1, usage);
  // The Voronoi diagram is the dual of the Delaunay triangulation of the same places.
  const triangleCount = scalar('triangleCount');
  const delaunay = {
    triangles: triangleSlots,
    status: {
      count: triangleCount,
      requiredCount: scalar('triangleRequiredCount'),
      overflow: scalar('triangleOverflow')
    }
  };
  graph.add(new GPUDelaunayTessellation({positions: places, output: delaunay}));
  // #region example
  const output = {
    segments: segmentSlots,
    siteIds: createTransientView(graph, 'siteIds', 'uint32x2', segmentSlots.length, usage),
    status: {
      count: scalar('count'),
      requiredCount: scalar('requiredCount'),
      overflow: scalar('overflow')
    }
  };
  const {triangles} = delaunay;
  graph.add(
    new GPUVoronoiDiagram({positions: places, triangles, triangleCount, clipBounds, output})
  );
  // segments: (startX, startY, endX, endY) in m; siteIds: the two places each segment separates
  const {segments, siteIds} = output;
  // #endregion example
  return {...output, segments, siteIds, triangleCount};
}
