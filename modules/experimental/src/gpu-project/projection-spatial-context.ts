// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileComment: Independently implemented for WebGPU; inspired by NVIDIA RAPIDS cuProj.

import type {GPUSpatialContext} from '../analysis-core/contracts/spatial-context';

/**
 * Planar context of rows projected into the projected CRS `targetCrs`, or `undefined` when no
 * target is given. The frame `{kind: 'crs', id: targetCrs, crs: targetCrs}` makes rows projected
 * into different CRSs incompatible. `units` is left undeclared (native): the CRS linear unit is
 * not parsed here, so execution bundles stay free of CRS providers.
 *
 * @internal
 */
export function getGPUProjectionSpatialContext(targetCrs?: string): GPUSpatialContext | undefined {
  if (!targetCrs) {
    return undefined;
  }
  return Object.freeze({
    coordinateSpace: 'planar',
    metric: 'native',
    frame: Object.freeze({kind: 'crs', id: targetCrs, crs: targetCrs})
  });
}
