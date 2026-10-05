// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Validates the bounds and maximum distance shared by the pair-statistics parameter packers.
 *
 * @internal
 */
export function validatePairStatisticsBounds(
  name: string,
  bounds: readonly [number, number, number, number],
  maximumDistance: number
): void {
  if (bounds.length !== 4) {
    throw new Error(`${name} bounds must be [minX, minY, maxX, maxY]`);
  }
  for (const value of [...bounds, maximumDistance]) {
    if (!Number.isFinite(value)) {
      throw new Error(`${name} bounds and maximumDistance must be finite`);
    }
  }
  if (bounds[2] < bounds[0] || bounds[3] < bounds[1]) {
    throw new Error(`${name} bounds must satisfy minX <= maxX and minY <= maxY`);
  }
  if (maximumDistance <= 0) {
    throw new Error(`${name} maximumDistance must be positive`);
  }
}
