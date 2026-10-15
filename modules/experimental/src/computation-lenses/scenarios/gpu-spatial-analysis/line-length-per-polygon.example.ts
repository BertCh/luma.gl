// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, type GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPULineLengthPerPolygon,
  type GPULineGeometryFields,
  type GPUPolygonGeometryFields
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Roads and community areas uploaded by the lens scenario, in UTM 16N metres (planar). */
export type LineLengthPerPolygonExampleInput = {
  roads: GPULineGeometryFields;
  areas: GPUPolygonGeometryFields;
};

/** Metres of street inside each community area, and the street segments counted there. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {roads, areas}: LineLengthPerPolygonExampleInput
) {
  // #region example
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const areaCount = areas.featureOffsets.length - 1;
  const output = {
    lengths: createTransientView(graph, 'lengths', 'float32', areaCount, usage),
    segmentCounts: createTransientView(graph, 'segmentCounts', 'uint32', areaCount, usage),
    status: {overflow: createTransientView(graph, 'overflow', 'uint32', 1, usage)}
  };
  graph.add(
    new GPULineLengthPerPolygon({
      ...roads,
      polygons: {kind: 'polygons', ...areas},
      maximumCrossingCount: 64,
      pairCapacity: 4096,
      output
    })
  );
  // lengths: m of road clipped to each area; segmentCounts: road segments with length inside
  const {lengths, segmentCounts} = output;
  // #endregion example
  return {...output, lengths, segmentCounts};
}
