// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, type GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUMinimumBounds,
  type GPUPolygonGeometryFields
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Parks uploaded by the lens scenario, in UTM 16N metres (planar, as the element requires). */
export type MinimumBoundsExampleInput = {parks: GPUPolygonGeometryFields};

/** Minimum-area rotated rectangle and minimum bounding circle of every park. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {parks}: MinimumBoundsExampleInput
) {
  // #region example
  const parkCount = parks.featureOffsets.length - 1;
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const output = {
    rectangleCorners: createTransientView(
      graph,
      'rectangleCorners',
      'float32x2',
      parkCount * 4,
      usage
    ),
    rectangleSizes: createTransientView(graph, 'rectangleSizes', 'float32x4', parkCount, usage),
    circles: createTransientView(graph, 'circles', 'float32x3', parkCount, usage),
    status: {overflow: createTransientView(graph, 'overflow', 'uint32', 1, usage)}
  };
  graph.add(new GPUMinimumBounds({...parks, output}));
  // rectangleCorners: 4 corners per park; rectangleSizes: [width, height, angle, area]
  const {rectangleCorners, rectangleSizes, circles} = output;
  // #endregion example
  return {...output, rectangleCorners, rectangleSizes, circles};
}
