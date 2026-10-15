// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUSpatialEdgeModel} from '../../analysis-core/contracts/index';

/** Number of float32 words in the per-frame preparation parameters. */
export const GPU_LON_LAT_POLYGON_PREPARATION_PARAMETER_LENGTH = 4;

/** Per-frame controls for longitude/latitude polygon preparation. */
export type GPULonLatPolygonPreparationSettings = {
  /** Maximum longitude or central-angle step in degrees; zero disables densification. */
  maximumSegmentDegrees?: number;
};

/** Packs the degree step, defaulting to 0.1 for great-circle edges and zero for linear edges. */
export function getGPULonLatPolygonPreparationParameterValues(
  edgeModel: GPUSpatialEdgeModel,
  settings: GPULonLatPolygonPreparationSettings = {},
  target: Float32Array = new Float32Array(GPU_LON_LAT_POLYGON_PREPARATION_PARAMETER_LENGTH)
): Float32Array {
  const maximumSegmentDegrees =
    settings.maximumSegmentDegrees ?? (edgeModel === 'great-circle' ? 0.1 : 0);
  if (!(maximumSegmentDegrees >= 0)) {
    throw new Error(
      'getGPULonLatPolygonPreparationParameterValues maximumSegmentDegrees must be >= 0'
    );
  }
  if (target.length < GPU_LON_LAT_POLYGON_PREPARATION_PARAMETER_LENGTH) {
    throw new Error('getGPULonLatPolygonPreparationParameterValues target must hold 4 elements');
  }
  target.set([maximumSegmentDegrees, 0, 0, 0]);
  return target;
}
