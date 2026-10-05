// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  computeTopographicPosition,
  type TopographicPositionOracleScale
} from './topographic-position-oracle';

/** Oracle result of the Weiss (2001) classification. */
export type WeissLandformsOracleResult = {
  landforms: number[];
  /** Two planes: small then large standardized position. */
  standardizedPosition: number[];
  slope: number[];
  validity: number[];
};

/**
 * Float64 Weiss (2001) / Jenness Land Facets classification on uniform cells.
 *
 * TPI uses the quantized oracle (exactly the GPU's integer statistics), global standardization
 * uses the grid mean and population standard deviation of each scale over valid cells, and slope
 * is Horn's with clamped borders. `'local'` standardization uses DEV.
 */
export function computeWeissLandforms(
  values: ArrayLike<number>,
  valid: ArrayLike<number> | undefined,
  width: number,
  height: number,
  options: {
    scales: readonly [TopographicPositionOracleScale, TopographicPositionOracleScale];
    cellSize: number;
    zFactor?: number;
    standardThreshold?: number;
    slopeThresholdDegrees?: number;
    standardization?: 'global' | 'local';
    quantum?: number;
  }
): WeissLandformsOracleResult {
  const pixelCount = width * height;
  const isGlobal = (options.standardization ?? 'global') === 'global';
  const position = computeTopographicPosition(
    values,
    valid,
    width,
    height,
    options.scales,
    options.quantum ?? 1 / 256
  );
  const planes = isGlobal ? position.topographicPositionIndex : position.deviationFromMean;
  const isValidSample = (index: number) =>
    (valid ? valid[index] !== 0 : true) && Number.isFinite(values[index]);
  const zFactor = options.zFactor ?? 1;
  const slope = new Array(pixelCount).fill(Number.NaN);
  const mask = new Array(pixelCount).fill(0);
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      const index = row * width + column;
      let ok = isValidSample(index);
      let eastSum = 0;
      let southSum = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          const sampleRow = Math.min(Math.max(row + dy, 0), height - 1);
          const sampleColumn = Math.min(Math.max(column + dx, 0), width - 1);
          const sampleIndex = sampleRow * width + sampleColumn;
          ok = ok && isValidSample(sampleIndex);
          const weight = dx === 0 || dy === 0 ? 2 : 1;
          const difference = values[sampleIndex] - values[index];
          eastSum += dx * weight * difference;
          southSum += dy * weight * difference;
        }
      }
      const gradient = Math.hypot(eastSum, southSum) * (zFactor / (8 * options.cellSize));
      const small = planes[index];
      const large = planes[pixelCount + index];
      ok = ok && Number.isFinite(small) && Number.isFinite(large) && position.validity[index] === 1;
      if (ok) {
        slope[index] = (Math.atan(gradient) * 180) / Math.PI;
        mask[index] = 1;
      }
    }
  }
  const standardize = [0, 1].map(plane => {
    if (!isGlobal) return (value: number) => value;
    let count = 0;
    let sum = 0;
    for (let index = 0; index < pixelCount; index++) {
      if (mask[index]) {
        count++;
        sum += planes[plane * pixelCount + index];
      }
    }
    const mean = sum / count;
    let squareSum = 0;
    for (let index = 0; index < pixelCount; index++) {
      if (mask[index]) squareSum += (planes[plane * pixelCount + index] - mean) ** 2;
    }
    const deviation = Math.sqrt(squareSum / count);
    return (value: number) => (deviation > 0 ? (value - mean) / deviation : 0);
  });
  const threshold = options.standardThreshold ?? 1;
  const slopeThreshold = options.slopeThresholdDegrees ?? 5;
  const getClass = (value: number) => (value <= -threshold ? 0 : value >= threshold ? 2 : 1);
  const result: WeissLandformsOracleResult = {
    landforms: new Array(pixelCount).fill(0),
    standardizedPosition: new Array(2 * pixelCount).fill(Number.NaN),
    slope,
    validity: mask
  };
  for (let index = 0; index < pixelCount; index++) {
    if (!mask[index]) continue;
    const small = standardize[0](planes[index]);
    const large = standardize[1](planes[pixelCount + index]);
    result.standardizedPosition[index] = small;
    result.standardizedPosition[pixelCount + index] = large;
    const smallClass = getClass(small);
    const largeClass = getClass(large);
    let landform = 1 + largeClass;
    if (smallClass === 1) {
      landform =
        largeClass === 0 ? 4 : largeClass === 2 ? 7 : slope[index] <= slopeThreshold ? 5 : 6;
    } else if (smallClass === 2) {
      landform = 8 + largeClass;
    }
    result.landforms[index] = landform;
  }
  return result;
}
