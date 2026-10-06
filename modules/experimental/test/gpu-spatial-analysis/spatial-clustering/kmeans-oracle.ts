// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Result of {@link computeKMeansOracle}. */
export type KMeansOracleResult = {
  labels: number[];
  centers: number[];
  sizes: number[];
  squaredDistances: number[];
};

const NOISE = 0xffffffff;

/**
 * f64 Lloyd oracle with the same fixed rules as `GPUKMeans` for `'first-valid'` initialization:
 * first `k` finite points as centers, nearest center with lowest-ID ties, empty centers keep their
 * position, a final assignment after the last update.
 */
export function computeKMeansOracle(
  positions: Float32Array,
  k: number,
  iterations: number
): KMeansOracleResult {
  const rows = positions.length / 2;
  const valid = (row: number) =>
    Number.isFinite(positions[2 * row]) && Number.isFinite(positions[2 * row + 1]);
  const centers = new Array<number>(2 * k).fill(NaN);
  let found = 0;
  for (let row = 0; row < rows && found < k; row++) {
    if (valid(row)) {
      centers[2 * found] = positions[2 * row];
      centers[2 * found + 1] = positions[2 * row + 1];
      found++;
    }
  }
  const labels = new Array<number>(rows).fill(NOISE);
  const squaredDistances = new Array<number>(rows).fill(NaN);
  const assign = () => {
    for (let row = 0; row < rows; row++) {
      labels[row] = NOISE;
      squaredDistances[row] = NaN;
      if (!valid(row)) continue;
      for (let center = 0; center < k; center++) {
        const distance =
          (centers[2 * center] - positions[2 * row]) ** 2 +
          (centers[2 * center + 1] - positions[2 * row + 1]) ** 2;
        if (distance === distance && (labels[row] === NOISE || distance < squaredDistances[row])) {
          labels[row] = center;
          squaredDistances[row] = distance;
        }
      }
    }
  };
  for (let iteration = 0; iteration < iterations; iteration++) {
    assign();
    const sums = new Array<number>(2 * k).fill(0);
    const counts = new Array<number>(k).fill(0);
    for (let row = 0; row < rows; row++) {
      if (labels[row] !== NOISE) {
        sums[2 * labels[row]] += positions[2 * row];
        sums[2 * labels[row] + 1] += positions[2 * row + 1];
        counts[labels[row]]++;
      }
    }
    for (let center = 0; center < k; center++) {
      if (counts[center] > 0) {
        centers[2 * center] = sums[2 * center] / counts[center];
        centers[2 * center + 1] = sums[2 * center + 1] / counts[center];
      }
    }
  }
  assign();
  const sizes = new Array<number>(k).fill(0);
  for (const label of labels) if (label !== NOISE) sizes[label]++;
  return {labels, centers, sizes, squaredDistances};
}
