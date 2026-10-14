// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineGPUSpatialParameterSchema} from '../../analysis-core/contracts/index';

/** Number of float32 elements in a {@link GPUNeighborSearch} parameter view. */
export const GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH = 12;

/** Declarative layout of the packed neighbor-search view. */
export const GPU_NEIGHBOR_SEARCH_PARAMETER_SCHEMA = defineGPUSpatialParameterSchema({
  id: 'neighbor-search',
  format: 'float32',
  wordLength: GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
  fields: [
    {name: 'minimumX', format: 'float32', wordOffset: 0, defaultValue: 0, dynamic: true},
    {name: 'minimumY', format: 'float32', wordOffset: 1, defaultValue: 0, dynamic: true},
    {name: 'maximumX', format: 'float32', wordOffset: 2, defaultValue: 0, dynamic: true},
    {name: 'maximumY', format: 'float32', wordOffset: 3, defaultValue: 0, dynamic: true},
    {
      name: 'radius',
      format: 'float32',
      wordOffset: 4,
      defaultValue: Infinity,
      allowNonFinite: true,
      units: 'position-units',
      dynamic: true
    },
    {name: 'weightKind', format: 'float32', wordOffset: 5, defaultValue: 0, dynamic: true},
    {name: 'power', format: 'float32', wordOffset: 6, defaultValue: 1, dynamic: true},
    {
      name: 'distanceFloor',
      format: 'float32',
      wordOffset: 7,
      defaultValue: 0,
      minimum: 0,
      units: 'position-units',
      dynamic: true
    },
    {name: 'kernel', format: 'float32', wordOffset: 8, defaultValue: 1, dynamic: true},
    {
      name: 'rowStandardize',
      format: 'float32',
      wordOffset: 9,
      defaultValue: 0,
      minimum: 0,
      maximum: 1,
      dynamic: true
    }
  ]
});

/** Encoded `weightKind` values stored in parameter slot 5. */
export const GPU_NEIGHBOR_SEARCH_WEIGHT_KIND = {
  /** `w = 1`. */
  binary: 0,
  /** `w = max(d, distanceFloor)^-power`. */
  inverseDistance: 1,
  /** `w = K(d / bandwidth)` for the selected kernel. */
  kernel: 2
} as const;

/** Encoded `kernel` values stored in parameter slot 8. */
export const GPU_NEIGHBOR_SEARCH_KERNEL = {
  /** `exp(-z^2 / 2) / sqrt(2 pi)`. */
  gaussian: 0,
  /** `1 - z`. */
  triangular: 1,
  /** `3/4 (1 - z^2)`, PySAL's `quadratic`. */
  epanechnikov: 2,
  /** `15/16 (1 - z^2)^2`, PySAL's `quartic`. */
  bisquare: 3,
  /** `1/2`. */
  uniform: 4
} as const;

/** Weight function applied to each neighbor distance. */
export type GPUNeighborSearchWeightKind = keyof typeof GPU_NEIGHBOR_SEARCH_WEIGHT_KIND;

/** Kernel profile used when `weightKind` is `'kernel'`. */
export type GPUNeighborSearchKernel = keyof typeof GPU_NEIGHBOR_SEARCH_KERNEL;

/** Per-frame settings of {@link GPUNeighborSearch}, packed by {@link getGPUNeighborSearchParameterValues}. */
export type GPUNeighborSearchSettings = {
  /**
   * `[minX, minY, maxX, maxY]` extent of the search lattice. Targets and queries outside it (or
   * with non-finite coordinates) are excluded: they become isolates and never a neighbor, so it
   * must cover every point of interest (count the dropped rows with `invalidCount`). For a
   * great-circle search these are degrees `[minLongitude, minLatitude, maxLongitude,
   * maxLatitude]` with a longitude span of at most 360 (`[-180, -90, 180, 90]` for the world).
   */
  bounds: readonly [number, number, number, number];
  /**
   * In the units of the spatial context (position units, or meters/kilometers of great-circle
   * distance). Radius mode: the distance band (neighbors satisfy `d <= radius`); a non-positive or non-finite
   * radius yields empty rows. kNN mode: an optional maximum distance; `Infinity` (the default) or
   * any non-positive value means unbounded.
   */
  radius?: number;
  /** Weight function. Defaults to `'binary'`. */
  weightKind?: GPUNeighborSearchWeightKind;
  /** Inverse-distance exponent. Defaults to 1. */
  power?: number;
  /**
   * Inverse distance uses `max(d, distanceFloor)`. Defaults to 0, in which case coincident points
   * (`d = 0`) get weight 0 because their weight is not finite.
   */
  distanceFloor?: number;
  /** Kernel profile for `weightKind: 'kernel'`. Defaults to `'triangular'`. */
  kernel?: GPUNeighborSearchKernel;
  /** Divides each row's weights by their sum (zero-sum rows stay zero). Defaults to false. */
  rowStandardize?: boolean;
};

/**
 * The weight transform {@link GPUNeighborSearchSettings} select, for
 * `GPUNeighborSearchProps.weightTransform`: `'row'` with `rowStandardize`, else `'binary'`,
 * `'raw'` (inverse distance) or `'kernel'` from `weightKind`.
 */
export function getGPUNeighborSearchWeightTransform(
  settings: Pick<GPUNeighborSearchSettings, 'weightKind' | 'rowStandardize'>
): 'binary' | 'raw' | 'kernel' | 'row' {
  const weightKind = settings.weightKind ?? 'binary';
  if (!(weightKind in GPU_NEIGHBOR_SEARCH_WEIGHT_KIND)) {
    throw new Error(`GPUNeighborSearch unknown weightKind ${weightKind}`);
  }
  if (settings.rowStandardize) {
    return 'row';
  }
  return weightKind === 'inverseDistance' ? 'raw' : weightKind === 'kernel' ? 'kernel' : 'binary';
}

/**
 * Packs {@link GPUNeighborSearchSettings} into the float32 layout read by the kernels.
 *
 * Layout: `[minX, minY, maxX, maxY, radius, weightKind, power, distanceFloor, kernel,
 * rowStandardize, 0, 0]`.
 */
export function getGPUNeighborSearchParameterValues(
  settings: GPUNeighborSearchSettings
): Float32Array {
  const values = new Float32Array(GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH);
  const [minimumX, minimumY, maximumX, maximumY] = settings.bounds;
  const weightKind = settings.weightKind ?? 'binary';
  const kernel = settings.kernel ?? 'triangular';
  if (!(weightKind in GPU_NEIGHBOR_SEARCH_WEIGHT_KIND)) {
    throw new Error(`GPUNeighborSearch unknown weightKind ${weightKind}`);
  }
  if (!(kernel in GPU_NEIGHBOR_SEARCH_KERNEL)) {
    throw new Error(`GPUNeighborSearch unknown kernel ${kernel}`);
  }
  values[0] = minimumX;
  values[1] = minimumY;
  values[2] = maximumX;
  values[3] = maximumY;
  values[4] = settings.radius ?? Infinity;
  values[5] = GPU_NEIGHBOR_SEARCH_WEIGHT_KIND[weightKind];
  values[6] = settings.power ?? 1;
  values[7] = settings.distanceFloor ?? 0;
  values[8] = GPU_NEIGHBOR_SEARCH_KERNEL[kernel];
  values[9] = settings.rowStandardize ? 1 : 0;
  return values;
}
