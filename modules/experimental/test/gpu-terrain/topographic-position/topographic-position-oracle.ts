// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** One oracle scale, mirroring `GPUTerrainTopographicPositionScale`. */
export type TopographicPositionOracleScale = {radius: number; innerRadius?: number};

/** Oracle result: planes are `scale * pixelCount + index`. */
export type TopographicPositionOracleResult = {
  topographicPositionIndex: number[];
  deviationFromMean: number[];
  maximumDeviation: number[];
  maximumDeviationRadius: number[];
  validity: number[];
};

/** IEEE round-half-to-even, matching WGSL `round`. */
export function roundHalfEven(value: number): number {
  const floor = Math.floor(value);
  const fraction = value - floor;
  if (fraction > 0.5) return floor + 1;
  if (fraction < 0.5) return floor;
  return floor % 2 === 0 ? floor : floor + 1;
}

/**
 * Float64 oracle of multiscale TPI, DEV, and DEVmax (Lindsay et al. 2015) with windows clipped to
 * the raster and to valid cells.
 *
 * With `quantum`, elevations are first quantized exactly like the GPU (`round(z / quantum)`,
 * half to even), so every sum below is an exact integer in float64 for test-sized windows and the
 * result isolates the GPU's float32 rounding. Without it, the oracle is the plain definition.
 */
export function computeTopographicPosition(
  values: ArrayLike<number>,
  valid: ArrayLike<number> | undefined,
  width: number,
  height: number,
  scales: readonly TopographicPositionOracleScale[],
  quantum?: number
): TopographicPositionOracleResult {
  const pixelCount = width * height;
  const isValid = (index: number) =>
    (valid ? valid[index] !== 0 : true) && Number.isFinite(values[index]);
  const level = (index: number) =>
    quantum === undefined ? values[index] : roundHalfEven(values[index] / quantum);
  const unit = quantum ?? 1;
  const result: TopographicPositionOracleResult = {
    topographicPositionIndex: new Array(scales.length * pixelCount).fill(Number.NaN),
    deviationFromMean: new Array(scales.length * pixelCount).fill(Number.NaN),
    maximumDeviation: new Array(pixelCount).fill(Number.NaN),
    maximumDeviationRadius: new Array(pixelCount).fill(0),
    validity: new Array(pixelCount).fill(0)
  };
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      const index = row * width + column;
      if (!isValid(index)) {
        continue;
      }
      result.validity[index] = 1;
      const centre = level(index);
      let maximumAbsolute = -1;
      for (const [scaleIndex, scale] of scales.entries()) {
        const innerRadius = scale.innerRadius ?? 0;
        let count = 0;
        let sum = 0;
        let squareSum = 0;
        let annulusCount = 0;
        let annulusSum = 0;
        for (let dy = -scale.radius; dy <= scale.radius; dy++) {
          for (let dx = -scale.radius; dx <= scale.radius; dx++) {
            const sampleRow = row + dy;
            const sampleColumn = column + dx;
            if (sampleRow < 0 || sampleColumn < 0 || sampleRow >= height || sampleColumn >= width) {
              continue;
            }
            const sampleIndex = sampleRow * width + sampleColumn;
            if (!isValid(sampleIndex)) {
              continue;
            }
            const difference = level(sampleIndex) - centre;
            count++;
            sum += difference;
            squareSum += difference * difference;
            if (Math.max(Math.abs(dx), Math.abs(dy)) > innerRadius) {
              annulusCount++;
              annulusSum += difference;
            }
          }
        }
        const meanOffset = sum / count;
        const variance = squareSum / count - meanOffset * meanOffset;
        const deviation = variance > 0 ? -meanOffset / Math.sqrt(variance) : 0;
        result.deviationFromMean[scaleIndex * pixelCount + index] = deviation;
        if (annulusCount > 0) {
          result.topographicPositionIndex[scaleIndex * pixelCount + index] =
            (-annulusSum / annulusCount) * unit;
        }
        if (Math.abs(deviation) > maximumAbsolute) {
          maximumAbsolute = Math.abs(deviation);
          result.maximumDeviation[index] = deviation;
          result.maximumDeviationRadius[index] = scale.radius;
        }
      }
    }
  }
  return result;
}

/**
 * Deterministic test terrain: a tilted ridge-and-valley surface plus hash noise, offset by `base`.
 */
export function createTopographicTerrain(
  width: number,
  height: number,
  base: number,
  relief: number
): Float32Array {
  return Float32Array.from({length: width * height}, (_, index) => {
    const column = index % width;
    const row = Math.floor(index / width);
    const hash = Math.sin(column * 12.9898 + row * 78.233) * 43758.5453;
    const noise = hash - Math.floor(hash) - 0.5;
    return (
      base +
      relief *
        (0.4 * Math.sin(column * 0.31) * Math.cos(row * 0.23) + 0.002 * column * row + 0.15 * noise)
    );
  });
}
