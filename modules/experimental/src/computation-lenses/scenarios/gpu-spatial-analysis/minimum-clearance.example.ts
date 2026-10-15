// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, type GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUMinimumClearance,
  type GPUMinimumClearanceProps,
  type GPUSpatialContext
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Inputs the lens scenario uploads from `s2-chicago-north-parks.json`. */
export type MinimumClearanceExampleInput = {
  /** Park polygons, projected to local-frame metres. */
  parks: Extract<GPUMinimumClearanceProps['geometry'], {kind: 'polygons'}>;
  /** Planar context of the local frame. */
  spatialContext: GPUSpatialContext;
};

/** Minimum clearance of every park polygon, and the vertex that sets it. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {parks, spatialContext}: MinimumClearanceExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const parkCount = parks.featureOffsets.length - 1;
  const clearances = createTransientView(graph, 'clearances', 'float32', parkCount, usage);
  const vertexIds = createTransientView(graph, 'vertexIds', 'uint32', parkCount, usage);
  // #region example
  graph.add(
    new GPUMinimumClearance({
      spatialContext,
      geometry: parks,
      output: {clearances, vertexIds}
    })
  );
  // clearances: metres a single vertex could move before its park becomes invalid
  // #endregion example
  return {clearances, vertexIds};
}
