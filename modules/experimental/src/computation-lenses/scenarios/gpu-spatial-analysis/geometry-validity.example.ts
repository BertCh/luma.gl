// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, type GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUGeometryValidity,
  type GPUGeometryValidityProps,
  type GPUSpatialContext
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Inputs the lens scenario uploads from `s2-chicago-north-parks.json`. */
export type GeometryValidityExampleInput = {
  /** Park polygons, projected to local-frame metres. */
  parks: NonNullable<GPUGeometryValidityProps['polygons']>;
  /** Planar context of the local frame. */
  spatialContext: GPUSpatialContext;
};

/** Structural validity bitmask of every park polygon. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {parks, spatialContext}: GeometryValidityExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const mask = createTransientView(graph, 'mask', 'uint32', parks.featureOffsets.length - 1, usage);
  const overflow = createTransientView(graph, 'overflow', 'uint32', 1, usage);
  const intersectionCount = createTransientView(graph, 'intersectionCount', 'uint32', 1, usage);
  // #region example
  graph.add(
    new GPUGeometryValidity({
      spatialContext,
      polygons: parks,
      orientation: 'ignore', // winding is a convention, not a validity rule
      intersectionCapacity: 4096,
      output: {mask, status: {overflow, intersectionCount}}
    })
  );
  // mask: 0 for a valid park, otherwise GPU_GEOMETRY_VALIDITY_BIT flags
  // #endregion example
  return {mask, overflow, intersectionCount};
}
