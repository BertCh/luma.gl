// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Rectangle bounds or the bounding box of a compile-time polygon study window. */
export type GPUPairStatisticsBounds = readonly [number, number, number, number] | 'study-window';

/**
 * Validates the bounds and maximum distance shared by the pair-statistics parameter packers.
 *
 * @internal
 */
export function validatePairStatisticsBounds<Bounds extends GPUPairStatisticsBounds>(
  name: string,
  bounds: Bounds,
  maximumDistance: number
): void {
  if (bounds !== 'study-window' && (typeof bounds === 'string' || bounds.length !== 4)) {
    throw new Error(`${name} bounds must be [minX, minY, maxX, maxY] or 'study-window'`);
  }
  for (const value of [...(bounds === 'study-window' ? [] : bounds), maximumDistance]) {
    if (!Number.isFinite(value)) {
      throw new Error(`${name} bounds and maximumDistance must be finite`);
    }
  }
  if (bounds !== 'study-window' && (bounds[2] < bounds[0] || bounds[3] < bounds[1])) {
    throw new Error(`${name} bounds must satisfy minX <= maxX and minY <= maxY`);
  }
  if (maximumDistance <= 0) {
    throw new Error(`${name} maximumDistance must be positive`);
  }
}
