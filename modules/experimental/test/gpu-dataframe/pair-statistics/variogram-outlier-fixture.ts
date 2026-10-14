// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRandom} from './pair-statistics-oracle';

/** Fixture of the variogram outlier specs: bounds, distance and lags of the pinned bins. */
export const VARIOGRAM_OUTLIER_FRAME = {
  // Wide enough to include the isolated outlier at (500, 500).
  bounds: [0, 0, 600, 600] as const,
  maximumDistance: 10,
  lagCount: 8
};

/** Row replaced by the outlier. */
export const VARIOGRAM_OUTLIER_ROW = 17;

/**
 * 4000 seeded uniform points in the 100 x 100 square with N(0, 1) values (Box-Muller from the
 * shared xorshift, seed 2024). With `outlierStandardDeviations`, row `VARIOGRAM_OUTLIER_ROW` takes
 * that value (a sentinel-like outlier): `'isolated'` also moves it to (500, 500), where it has no
 * neighbor within the maximum distance (it inflates the value range but joins no pair, so every
 * pair term and the exact semivariances equal the clean scene's), `'embedded'` keeps its position
 * so its neighbors' terms are huge.
 */
export function createVariogramOutlierScene(
  outlierStandardDeviations?: number,
  placement: 'isolated' | 'embedded' = 'isolated'
): {
  positions: Float32Array;
  values: Float32Array;
} {
  const rows = 4000;
  const random = createRandom(2024);
  const positions = new Float32Array(rows * 2);
  const values = new Float32Array(rows);
  for (let row = 0; row < rows; row++) {
    positions[row * 2] = random() * 100;
    positions[row * 2 + 1] = random() * 100;
    const radius = Math.sqrt(-2 * Math.log(1 - random()));
    values[row] = radius * Math.cos(2 * Math.PI * random());
  }
  if (outlierStandardDeviations !== undefined) {
    values[VARIOGRAM_OUTLIER_ROW] = outlierStandardDeviations;
    if (placement === 'isolated') {
      positions[VARIOGRAM_OUTLIER_ROW * 2] = 500;
      positions[VARIOGRAM_OUTLIER_ROW * 2 + 1] = 500;
    }
  }
  return {positions, values};
}
