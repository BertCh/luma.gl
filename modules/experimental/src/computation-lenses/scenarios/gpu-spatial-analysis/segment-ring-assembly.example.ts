// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUSegmentRingAssembly} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Directed boundary segments (x0, y0, x1, y1) in UTM 16N metres, and the ring capacity. */
export type SegmentRingAssemblyExampleInput = {
  segments: GraphDataView<'float32x4'>;
  ringCapacity: number;
};

/** Chains shuffled community-area edges into closed rings, cancelling shared borders. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {segments, ringCapacity}: SegmentRingAssemblyExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const make = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    id: string,
    format: Format,
    length: number
  ) => createTransientView(graph, id, format, length, usage);
  // #region example
  const output = {
    ringOffsets: make('ringOffsets', 'uint32', ringCapacity + 1),
    positions: make('ringPositions', 'float32x2', segments.length + ringCapacity),
    ringAreas: make('ringAreas', 'float32', ringCapacity),
    ringIsHole: make('ringIsHole', 'uint32', ringCapacity),
    status: {
      count: make('ringCount', 'uint32', 1),
      overflow: make('overflow', 'uint32', 1),
      openSegmentCount: make('openSegmentCount', 'uint32', 1)
    }
  };
  graph.add(
    new GPUSegmentRingAssembly({
      endpoints: segments,
      vertexTolerance: 0.01, // metres
      interiorSide: 'right', // the areas' rings run clockwise
      normalizeWinding: true, // shells counter-clockwise, holes clockwise
      cancelOpposingSegments: true, // a shared border is walked once per neighbour
      output
    })
  );
  const {ringOffsets, positions, ringAreas} = output; // ring r: positions[ringOffsets[r]..ringOffsets[r+1])
  // #endregion example
  return {...output, ringOffsets, positions, ringAreas};
}
