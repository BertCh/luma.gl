// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  type GPUSpatialContext,
  warnGPUShortLongitudeLatitudeDistance
} from '../../analysis-core/contracts/spatial-context';

import {validateCapacity} from '../../utils/gpu-contributor-utils';
import type {GPURasterMetadata} from '../../gpu-raster/types';
import {getRasterGridCorners} from './spatial-interpolation-parameters';
import type {VariogramModelType} from '../../gpu-dataframe/pair-statistics/variogram-model';

/** Number of float32 elements in a {@link GPUKriging} parameter view. */
export const GPU_KRIGING_PARAMETER_LENGTH = 12;

/** Per-frame settings of {@link GPUKriging}. */
export type GPUKrigingSettings = {
  /**
   * Spatial context of the positions; when it is longitude/latitude, a distance under 10 float32
   * quanta (about 17 m) logs a warning (see Precision of longitude/latitude positions). Does not
   * change the packed values.
   */
  spatialContext?: GPUSpatialContext;
  /** Search radius in position units. `Infinity` searches the whole index domain. */
  searchRadius: number;
  /**
   * Neighborhood size `k`. Clamped to the contributor's compile-time `maximumNeighborCount`.
   * Defaults to that capacity.
   */
  neighborCount?: number;
  /** Cells with fewer contributing samples (and no exact hit) are nodata. Defaults to 3. */
  minimumNeighborCount?: number;
  /**
   * Fitted variogram, for example the result of `fitVariogramModel`: `gamma(h) = nugget + sill *
   * shape(h / range)` for `h > 0`, with the same effective-range convention. The nugget plus the
   * partial sill must be positive.
   */
  variogram: {
    /** Model family. */
    model: VariogramModelType;
    /** Nugget, at least 0. */
    nugget: number;
    /** Partial sill, at least 0. */
    sill: number;
    /** Effective range, positive. */
    range: number;
  };
};

const MODEL_CODES: Record<VariogramModelType, number> = {spherical: 0, exponential: 1, gaussian: 2};

/**
 * Packs per-frame {@link GPUKriging} parameters.
 *
 * Layout (float32): `[startX, startY, endX, endY, searchRadius, neighborCount, modelCode (0 spherical,
 * 1 exponential, 2 Gaussian), nugget, sill, range, minimumNeighborCount, reserved]`. The first
 * four values are the outer corners of the output grid in grid order: `(startX, startY)` is the
 * outer corner of cell `(0, 0)` and `(endX, endY)` the opposite corner, so a north-up grid
 * (`grid.affine[4] < 0`) has `endY < startY` and row 0 at the top, and a grid whose rows advance
 * north has row 0 at `minY`.
 *
 * @param settings Radius, neighborhood and variogram.
 * @param grid The output grid: axis-aligned; its width and height must match the contributor's.
 * @param target Optional destination of at least 12 elements.
 * @throws If the grid is rotated or sheared or invalid, the target is too short, or a setting is
 *   out of range.
 */
export function getGPUKrigingParameterValues(
  settings: GPUKrigingSettings,
  grid: GPURasterMetadata,
  target: Float32Array = new Float32Array(GPU_KRIGING_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_KRIGING_PARAMETER_LENGTH) {
    throw new Error(`Kriging parameter target must hold ${GPU_KRIGING_PARAMETER_LENGTH} elements`);
  }
  const {variogram} = settings;
  const modelCode = MODEL_CODES[variogram.model];
  if (modelCode === undefined) {
    throw new Error(`Unknown variogram model ${String(variogram.model)}`);
  }
  if (
    !(variogram.nugget >= 0) ||
    !(variogram.sill >= 0) ||
    !(variogram.nugget + variogram.sill > 0)
  ) {
    throw new Error('Kriging variogram nugget and sill must be non-negative with a positive total');
  }
  if (!(variogram.range > 0) || !Number.isFinite(variogram.range)) {
    throw new Error('Kriging variogram range must be positive and finite');
  }
  const neighborCount = settings.neighborCount ?? 16;
  const minimumNeighborCount = settings.minimumNeighborCount ?? 3;
  for (const [name, count] of [
    ['neighborCount', neighborCount],
    ['minimumNeighborCount', minimumNeighborCount]
  ] as const) {
    validateCapacity('Kriging', name, count, 0);
  }
  warnGPUShortLongitudeLatitudeDistance(
    'GPUKriging',
    settings.spatialContext,
    settings.searchRadius
  );
  warnGPUShortLongitudeLatitudeDistance(
    'GPUKriging',
    settings.spatialContext,
    settings.variogram.range
  );
  target.fill(0);
  target.set(getRasterGridCorners('getGPUKrigingParameterValues', grid), 0);
  target[4] = settings.searchRadius;
  target[5] = neighborCount;
  target[6] = modelCode;
  target[7] = variogram.nugget;
  target[8] = variogram.sill;
  target[9] = variogram.range;
  target[10] = minimumNeighborCount;
  return target;
}
