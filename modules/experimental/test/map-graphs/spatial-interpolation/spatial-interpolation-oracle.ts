// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Inputs of {@link interpolateInverseDistanceWeightingOnCPU}. */
export type InverseDistanceWeightingOracleInput = {
  /** Interleaved `x, y` sample positions. */
  positions: Float32Array;
  /** Sample values; NaN skips the sample. */
  values: Float32Array;
  /** Optional sample mask; zero skips the sample. */
  mask?: ArrayLike<number>;
  /** Output raster width. */
  width: number;
  /** Output raster height. */
  height: number;
  /** Inclusive sample domain of the grid index; samples outside it are ignored. */
  indexBounds: readonly [number, number, number, number];
  /** Output extent `[minX, minY, maxX, maxY]`. */
  extent: readonly [number, number, number, number];
  /** Search radius. */
  searchRadius: number;
  /** Distance power. */
  power: number;
  /** Effective nearest-neighbor limit; zero means every sample within the radius. */
  neighborCount: number;
  /** Minimum contributing samples for a non-exact cell. */
  minimumNeighborCount: number;
};

/** Dense result of the IDW oracle. */
export type InverseDistanceWeightingOracleResult = {
  /** Interpolated value per cell, NaN for nodata. */
  values: Float32Array;
  /** Contributing samples per cell. */
  counts: Uint32Array;
};

const f = Math.fround;

/**
 * CPU oracle for `GPUInverseDistanceWeighting`.
 *
 * Candidate selection (cell centers, `d^2`, the radius test, and `(d^2, row)` order) mirrors the
 * kernel's f32 arithmetic with `Math.fround`, assuming no fused multiply-add. Weights and sums are
 * float64, normalized by the nearest contributor, so GPU parity is within f32 rounding, not exact.
 */
export function interpolateInverseDistanceWeightingOnCPU(
  input: InverseDistanceWeightingOracleInput
): InverseDistanceWeightingOracleResult {
  const {positions, values, width, height, extent, indexBounds} = input;
  const cellCount = width * height;
  const result = new Float32Array(cellCount).fill(NaN);
  const counts = new Uint32Array(cellCount);
  const [minX, minY, maxX, maxY] = extent.map(f);
  const radius = f(input.searchRadius);
  const power = f(input.power);
  const radiusSquared = f(radius * radius);
  const cellWidth = f(f(maxX - minX) / width);
  const cellHeight = f(f(maxY - minY) / height);
  const minimumNeighbors = Math.max(input.minimumNeighborCount, 1);
  const parametersValid = radius >= 0 && Number.isFinite(power) && power >= 0;

  const samples: number[] = [];
  for (let row = 0; row < values.length; row++) {
    const x = positions[2 * row];
    const y = positions[2 * row + 1];
    if (
      !Number.isFinite(x) ||
      !Number.isFinite(y) ||
      x < indexBounds[0] ||
      x > indexBounds[2] ||
      y < indexBounds[1] ||
      y > indexBounds[3] ||
      Number.isNaN(values[row]) ||
      (input.mask && input.mask[row] === 0)
    ) {
      continue;
    }
    samples.push(row);
  }

  for (let rasterRow = 0; rasterRow < height; rasterRow++) {
    for (let column = 0; column < width; column++) {
      const cell = rasterRow * width + column;
      const centerX = f(minX + f(f(column + 0.5) * cellWidth));
      const centerY = f(minY + f(f(rasterRow + 0.5) * cellHeight));
      if (!parametersValid || !Number.isFinite(centerX) || !Number.isFinite(centerY)) {
        continue;
      }
      let candidates: {row: number; distanceSquared: number}[] = [];
      for (const row of samples) {
        const dx = f(positions[2 * row] - centerX);
        const dy = f(positions[2 * row + 1] - centerY);
        const distanceSquared = f(f(dx * dx) + f(dy * dy));
        if (distanceSquared <= radiusSquared) {
          candidates.push({row, distanceSquared});
        }
      }
      let exactRow = -1;
      for (const candidate of candidates) {
        if (candidate.distanceSquared === 0 && (exactRow < 0 || candidate.row < exactRow)) {
          exactRow = candidate.row;
        }
      }
      if (input.neighborCount > 0) {
        candidates.sort(
          (left, right) => left.distanceSquared - right.distanceSquared || left.row - right.row
        );
        candidates = candidates.slice(0, input.neighborCount);
      }
      counts[cell] = candidates.length;
      if (exactRow >= 0) {
        result[cell] = values[exactRow];
        continue;
      }
      if (candidates.length < minimumNeighbors) {
        continue;
      }
      const nearest = Math.min(...candidates.map(candidate => candidate.distanceSquared));
      let weightSum = 0;
      let weightedSum = 0;
      for (const candidate of candidates) {
        const weight = Math.pow(nearest / candidate.distanceSquared, power / 2);
        weightSum += weight;
        weightedSum += weight * values[candidate.row];
      }
      result[cell] = weightedSum / weightSum;
    }
  }
  return {values: result, counts};
}

/** Inputs of {@link computeFocalStatisticsOnCPU}. */
export type FocalStatisticsOracleInput = {
  /** Row-major raster. NaN cells are nodata. */
  values: Float32Array;
  /** Optional validity; zero marks nodata. */
  validity?: ArrayLike<number>;
  /** Optional exact nodata sentinel. */
  noDataValue?: number;
  /** Raster width. */
  width: number;
  /** Raster height. */
  height: number;
  /** Compile-time radius cap. */
  maximumRadius: number;
  /** Per-frame radius. */
  radius: number;
  /** Window shape. */
  shape: 'square' | 'circle';
  /** Minimum valid cells for a defined statistic. */
  minimumCount: number;
  /** Whether a nodata center cell yields nodata statistics. */
  propagateCenterNoData: boolean;
};

/** Dense result of the focal statistics oracle; float columns are NaN for nodata. */
export type FocalStatisticsOracleResult = {
  mean: Float32Array;
  sum: Float32Array;
  min: Float32Array;
  max: Float32Array;
  range: Float32Array;
  standardDeviation: Float32Array;
  count: Uint32Array;
};

/**
 * CPU oracle for `GPUFocalStatistics`.
 *
 * Sums are accumulated in the kernel's row-major window order with `Math.fround` after every
 * addition, so `sum`, `min`, `max`, `range`, and `count` match the GPU exactly when the GPU does not
 * reassociate. The mean (one f32 division) and the centered standard deviation are computed in
 * float64 from that sum, so parity is within f32 rounding.
 */
export function computeFocalStatisticsOnCPU(
  input: FocalStatisticsOracleInput
): FocalStatisticsOracleResult {
  const {values, width, height} = input;
  const cellCount = width * height;
  const createFloat = () => new Float32Array(cellCount).fill(NaN);
  const result: FocalStatisticsOracleResult = {
    mean: createFloat(),
    sum: createFloat(),
    min: createFloat(),
    max: createFloat(),
    range: createFloat(),
    standardDeviation: createFloat(),
    count: new Uint32Array(cellCount)
  };
  const radius = f(input.radius);
  const windowRadius = radius >= 0 ? Math.min(Math.floor(radius), input.maximumRadius) : -1;
  const radiusSquared = f(radius * radius);
  const minimumCount = Math.max(input.minimumCount, 1);
  const isValid = (cell: number) =>
    !Number.isNaN(values[cell]) &&
    (input.noDataValue === undefined || values[cell] !== f(input.noDataValue)) &&
    (!input.validity || input.validity[cell] !== 0);

  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      const cell = row * width + column;
      const windowValues: number[] = [];
      let sum = 0;
      for (let dy = -windowRadius; dy <= windowRadius; dy++) {
        const sampleRow = row + dy;
        if (sampleRow < 0 || sampleRow >= height) {
          continue;
        }
        for (let dx = -windowRadius; dx <= windowRadius; dx++) {
          const sampleColumn = column + dx;
          if (sampleColumn < 0 || sampleColumn >= width) {
            continue;
          }
          if (input.shape === 'circle' && dx * dx + dy * dy > radiusSquared) {
            continue;
          }
          const sampleCell = sampleRow * width + sampleColumn;
          if (!isValid(sampleCell)) {
            continue;
          }
          windowValues.push(values[sampleCell]);
          sum = f(sum + values[sampleCell]);
        }
      }
      const count = windowValues.length;
      result.count[cell] = count;
      if (count < minimumCount || (input.propagateCenterNoData && !isValid(cell))) {
        continue;
      }
      const mean = sum / count;
      let squaredDeviationSum = 0;
      for (const value of windowValues) {
        squaredDeviationSum += (value - mean) ** 2;
      }
      const minimum = Math.min(...windowValues);
      const maximum = Math.max(...windowValues);
      result.sum[cell] = sum;
      result.mean[cell] = mean;
      result.min[cell] = minimum;
      result.max[cell] = maximum;
      result.range[cell] = maximum - minimum;
      result.standardDeviation[cell] = Math.sqrt(squaredDeviationSum / count);
    }
  }
  return result;
}
