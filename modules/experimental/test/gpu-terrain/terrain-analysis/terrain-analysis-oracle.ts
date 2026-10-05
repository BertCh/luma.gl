// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

const DEGREES = 180 / Math.PI;
const SOBEL_X = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
const SOBEL_Y = [-1, -2, -1, 0, 0, 0, 1, 2, 1];

/** Sobel derivative with strict validity propagation. */
export function computeSobel(
  values: ArrayLike<number>,
  valid: ArrayLike<number> | undefined,
  width: number,
  height: number,
  direction: 'x' | 'y',
  borderMode: 'clamp' | 'nodata'
): {values: number[]; validity: number[]} {
  const kernel = direction === 'x' ? SOBEL_X : SOBEL_Y;
  const isValid = (column: number, row: number) =>
    (valid ? valid[row * width + column] !== 0 : true) &&
    Number.isFinite(values[row * width + column]);
  const output: number[] = [];
  const validity: number[] = [];
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      let sum = 0;
      let ok = isValid(column, row);
      for (let ky = -1; ky <= 1; ky++) {
        for (let kx = -1; kx <= 1; kx++) {
          const weight = kernel[(ky + 1) * 3 + kx + 1];
          if (weight === 0) {
            continue;
          }
          let sampleColumn = column + kx;
          let sampleRow = row + ky;
          const outside =
            sampleColumn < 0 || sampleRow < 0 || sampleColumn >= width || sampleRow >= height;
          if (outside && borderMode === 'nodata') {
            ok = false;
            continue;
          }
          sampleColumn = Math.min(Math.max(sampleColumn, 0), width - 1);
          sampleRow = Math.min(Math.max(sampleRow, 0), height - 1);
          if (!isValid(sampleColumn, sampleRow)) {
            ok = false;
            continue;
          }
          sum += weight * values[sampleRow * width + sampleColumn];
        }
      }
      output.push(ok ? sum : Number.NaN);
      validity.push(ok ? 1 : 0);
    }
  }
  return {values: output, validity};
}

/** Options mirroring `GPUTerrainDerivatives` topology props. */
export type TerrainDerivativeOptions = {
  cellSizeMode?: 'uniform' | 'web-mercator' | 'geographic';
  slopeUnits?: 'degrees' | 'percent';
  rowDirection?: 'south' | 'north';
  borderMode?: 'clamp' | 'nodata';
};

/** CPU mirror of the shade kernel. */
export function computeTerrainDerivatives(
  values: ArrayLike<number>,
  valid: ArrayLike<number> | undefined,
  width: number,
  height: number,
  settings: ArrayLike<number>,
  options: TerrainDerivativeOptions = {}
): {slope: number[]; aspect: number[]; hillshade: number[]; validity: number[]} {
  const borderMode = options.borderMode ?? 'clamp';
  const gradientX = computeSobel(values, valid, width, height, 'x', borderMode);
  const gradientY = computeSobel(values, valid, width, height, 'y', borderMode);
  const rowSign = (options.rowDirection ?? 'south') === 'south' ? -1 : 1;
  const result = {
    slope: [] as number[],
    aspect: [] as number[],
    hillshade: [] as number[],
    validity: [] as number[]
  };
  for (let index = 0; index < width * height; index++) {
    const row = Math.floor(index / width);
    let cellX = settings[0];
    let cellY = settings[1];
    const mode = options.cellSizeMode ?? 'uniform';
    if (mode !== 'uniform') {
      const fraction = (row + 0.5) / height;
      const edge = settings[5] + (settings[6] - settings[5]) * fraction;
      if (mode === 'web-mercator') {
        const scale = Math.cosh(Math.PI * (1 - 2 * edge));
        cellX /= scale;
        cellY /= scale;
      } else {
        cellX = cellX * 111319.49079327357 * Math.cos((edge * Math.PI) / 180);
        cellY = cellY * 111319.49079327357;
      }
    }
    const zFactor = settings[2];
    const east = (zFactor * gradientX.values[index]) / (8 * cellX);
    const north = (rowSign * zFactor * gradientY.values[index]) / (8 * cellY);
    const magnitude = Math.hypot(east, north);
    const isValid = gradientX.validity[index] === 1 && Number.isFinite(magnitude);
    const slope =
      options.slopeUnits === 'percent' ? magnitude * 100 : Math.atan(magnitude) * DEGREES;
    let aspect = -1;
    if (magnitude > 0) {
      aspect = Math.atan2(-east, -north) * DEGREES;
      if (aspect < 0) aspect += 360;
    }
    const azimuth = (settings[3] * Math.PI) / 180;
    const altitude = (settings[4] * Math.PI) / 180;
    const length = Math.hypot(east, north, 1);
    const shade = Math.max(
      ((-east / length) * Math.sin(azimuth) + (-north / length) * Math.cos(azimuth)) *
        Math.cos(altitude) +
        Math.sin(altitude) / length,
      0
    );
    result.slope.push(isValid ? slope : Number.NaN);
    result.aspect.push(isValid ? aspect : Number.NaN);
    result.hillshade.push(isValid ? shade : Number.NaN);
    result.validity.push(isValid ? 1 : 0);
  }
  return result;
}

/** Bilinear sample at a clamped pixel-center position. */
export function sampleBilinear(
  values: ArrayLike<number>,
  width: number,
  height: number,
  x: number,
  y: number
): number {
  const clampedX = Math.min(Math.max(x, 0), width - 1);
  const clampedY = Math.min(Math.max(y, 0), height - 1);
  const baseX = Math.floor(clampedX);
  const baseY = Math.floor(clampedY);
  const nextX = Math.min(baseX + 1, width - 1);
  const nextY = Math.min(baseY + 1, height - 1);
  const fractionX = clampedX - baseX;
  const fractionY = clampedY - baseY;
  const top =
    values[baseY * width + baseX] * (1 - fractionX) + values[baseY * width + nextX] * fractionX;
  const bottom =
    values[nextY * width + baseX] * (1 - fractionX) + values[nextY * width + nextX] * fractionX;
  return top * (1 - fractionY) + bottom * fractionY;
}

/** CPU mirror of the viewshed kernel. */
export function computeTerrainViewshed(
  values: ArrayLike<number>,
  valid: ArrayLike<number> | undefined,
  width: number,
  height: number,
  settings: ArrayLike<number>
): number[] {
  const isValid = (column: number, row: number) =>
    valid ? valid[row * width + column] !== 0 : true;
  const sample = (x: number, y: number): number | undefined => {
    const clampedX = Math.min(Math.max(x, 0), width - 1);
    const clampedY = Math.min(Math.max(y, 0), height - 1);
    const baseX = Math.floor(clampedX);
    const baseY = Math.floor(clampedY);
    const nextX = Math.min(baseX + 1, width - 1);
    const nextY = Math.min(baseY + 1, height - 1);
    if (
      !isValid(baseX, baseY) ||
      !isValid(nextX, baseY) ||
      !isValid(baseX, nextY) ||
      !isValid(nextX, nextY)
    ) {
      return undefined;
    }
    return sampleBilinear(values, width, height, x, y);
  };
  const [observerX, observerY, observerHeight, targetHeight, maxDistance, cellX, cellY, curvature] =
    Array.from(settings);
  const observerInside =
    Number.isFinite(observerX) &&
    Number.isFinite(observerY) &&
    observerX >= 0 &&
    observerY >= 0 &&
    observerX <= width - 1 &&
    observerY <= height - 1;
  const observerSample = observerInside ? sample(observerX, observerY) : undefined;
  const result: number[] = [];
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      if (observerSample === undefined || !isValid(column, row)) {
        result.push(3);
        continue;
      }
      const observerElevation = observerSample + observerHeight;
      const deltaX = column - observerX;
      const deltaY = row - observerY;
      const targetDistance = Math.hypot(deltaX * cellX, deltaY * cellY);
      if (maxDistance > 0 && targetDistance > maxDistance) {
        result.push(2);
        continue;
      }
      if (targetDistance === 0) {
        result.push(1);
        continue;
      }
      const targetElevation =
        values[row * width + column] + targetHeight - curvature * targetDistance * targetDistance;
      const targetSlope = (targetElevation - observerElevation) / targetDistance;
      const stepCount = Math.ceil(Math.max(Math.abs(deltaX), Math.abs(deltaY)));
      let visible = true;
      for (let step = 1; step < stepCount; step++) {
        const fraction = step / stepCount;
        const elevation = sample(observerX + deltaX * fraction, observerY + deltaY * fraction);
        if (elevation === undefined) {
          continue;
        }
        const distance = targetDistance * fraction;
        if (
          (elevation - curvature * distance * distance - observerElevation) / distance >
          targetSlope
        ) {
          visible = false;
          break;
        }
      }
      result.push(visible ? 1 : 0);
    }
  }
  return result;
}

/** Marching-squares segment count for one level; cells with an invalid corner emit nothing. */
export function countContourSegments(
  values: ArrayLike<number>,
  width: number,
  height: number,
  level: number,
  valid?: ArrayLike<number>
): number {
  let count = 0;
  for (let row = 0; row + 1 < height; row++) {
    for (let column = 0; column + 1 < width; column++) {
      const corners = [
        row * width + column,
        row * width + column + 1,
        (row + 1) * width + column + 1,
        (row + 1) * width + column
      ];
      if (valid && corners.some(corner => valid[corner] === 0)) {
        continue;
      }
      const caseIndex = corners.reduce(
        (bits, corner, bit) => bits | (values[corner] >= level ? 1 << bit : 0),
        0
      );
      count +=
        caseIndex === 0 || caseIndex === 15 ? 0 : caseIndex === 5 || caseIndex === 10 ? 2 : 1;
    }
  }
  return count;
}

/** Result of {@link computeTerrainSightLine}. */
export type TerrainSightLineResult = {
  /** `GPU_TERRAIN_VISIBILITY` code. */
  code: number;
  /** Metres the target could sink and stay visible; NaN for noData / outOfRange. */
  clearance: number;
  /** Highest tested sample slope (-Infinity when none). */
  maxSlope: number;
  /** Target slope, or NaN when the pair is noData / outOfRange / zero distance. */
  targetSlope: number;
  /** Tolerance band half-width in slope units. */
  band: number;
};

/**
 * Float64 CPU mirror of the sight-line model of `GPUTerrainLineOfSight` including tolerance.
 *
 * `settings` uses the 12-float sight-line layout. The target elevation is the bilinear sample at
 * `target` plus the target height. Always evaluates every sample (no early exit).
 */
export function computeTerrainSightLine(
  values: ArrayLike<number>,
  valid: ArrayLike<number> | undefined,
  width: number,
  height: number,
  observer: readonly [number, number],
  target: readonly [number, number],
  settings: ArrayLike<number>,
  heights?: readonly [number, number]
): TerrainSightLineResult {
  const isValid = (column: number, row: number) =>
    valid ? valid[row * width + column] !== 0 : true;
  const sample = (x: number, y: number): number | undefined => {
    const clampedX = Math.min(Math.max(x, 0), width - 1);
    const clampedY = Math.min(Math.max(y, 0), height - 1);
    const baseX = Math.floor(clampedX);
    const baseY = Math.floor(clampedY);
    const nextX = Math.min(baseX + 1, width - 1);
    const nextY = Math.min(baseY + 1, height - 1);
    if (
      !isValid(baseX, baseY) ||
      !isValid(nextX, baseY) ||
      !isValid(baseX, nextY) ||
      !isValid(nextX, nextY)
    ) {
      return undefined;
    }
    return sampleBilinear(values, width, height, x, y);
  };
  const [observerHeight, targetHeight, maxDistance, cellX, cellY, curvature] = Array.from(settings);
  const [toleranceMeters, tolerancePerKilometer, ignoreDistance, ignoreFraction] = Array.from(
    settings
  ).slice(6, 10);
  const inside = ([x, y]: readonly [number, number]) =>
    Number.isFinite(x) && Number.isFinite(y) && x >= 0 && y >= 0 && x <= width - 1 && y <= height - 1;
  const noData = {code: 3, clearance: Number.NaN, maxSlope: -Infinity, targetSlope: Number.NaN, band: 0};
  if (!inside(observer) || !inside(target)) {
    return noData;
  }
  const observerSample = sample(observer[0], observer[1]);
  const targetSample = sample(target[0], target[1]);
  if (observerSample === undefined || targetSample === undefined) {
    return noData;
  }
  const eye = observerSample + (heights ? heights[0] : observerHeight);
  const deltaX = target[0] - observer[0];
  const deltaY = target[1] - observer[1];
  const distance = Math.hypot(deltaX * cellX, deltaY * cellY);
  if (maxDistance > 0 && distance > maxDistance) {
    return {...noData, code: 2};
  }
  if (distance === 0) {
    return {code: 1, clearance: 3.4028234663852886e38, maxSlope: -Infinity, targetSlope: Number.NaN, band: 0};
  }
  const targetTop = targetSample + (heights ? heights[1] : targetHeight);
  const targetSlope = (targetTop - curvature * distance * distance - eye) / distance;
  const band = (toleranceMeters + (tolerancePerKilometer * distance) / 1000) / distance;
  const stopDistance = distance - Math.max(ignoreDistance, ignoreFraction * distance);
  const stepCount = Math.ceil(Math.max(Math.abs(deltaX), Math.abs(deltaY)));
  let maxSlope = -Infinity;
  for (let step = 1; step < stepCount; step++) {
    const fraction = step / stepCount;
    const sampleDistance = distance * fraction;
    if (sampleDistance > stopDistance) {
      break;
    }
    const elevation = sample(observer[0] + deltaX * fraction, observer[1] + deltaY * fraction);
    if (elevation === undefined) {
      continue;
    }
    maxSlope = Math.max(
      maxSlope,
      (elevation - curvature * sampleDistance * sampleDistance - eye) / sampleDistance
    );
  }
  const code = maxSlope > targetSlope + band ? 0 : maxSlope <= targetSlope - band ? 1 : 4;
  return {
    code,
    clearance: maxSlope === -Infinity ? 3.4028234663852886e38 : (targetSlope - maxSlope) * distance,
    maxSlope,
    targetSlope,
    band
  };
}

/** Float64 viewshed with tolerance: one {@link computeTerrainSightLine} per pixel. */
export function computeTerrainViewshedWithTolerance(
  values: ArrayLike<number>,
  valid: ArrayLike<number> | undefined,
  width: number,
  height: number,
  viewshedSettings: ArrayLike<number>,
  tolerance: ArrayLike<number>
): TerrainSightLineResult[] {
  const [observerX, observerY, observerHeight, targetHeight, maxDistance, cellX, cellY, curvature] =
    Array.from(viewshedSettings);
  const sightSettings = [
    observerHeight,
    targetHeight,
    maxDistance,
    cellX,
    cellY,
    curvature,
    tolerance[0],
    tolerance[1],
    tolerance[2],
    tolerance[3],
    0,
    0
  ];
  const result: TerrainSightLineResult[] = [];
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      const line = computeTerrainSightLine(
        values,
        valid,
        width,
        height,
        [observerX, observerY],
        [column, row],
        sightSettings
      );
      result.push(line);
    }
  }
  return result;
}

/** Deterministic seeded value-noise fractal terrain (mulberry32), amplitude in metres. */
export function createFractalTerrain(
  width: number,
  height: number,
  seed: number,
  amplitude: number
): Float32Array {
  let state = seed >>> 0;
  const random = () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const result = new Float32Array(width * height);
  let octaveAmplitude = amplitude;
  for (let cellSize = 32; cellSize >= 2; cellSize /= 2) {
    const lattice = new Float64Array((Math.ceil(width / cellSize) + 2) * (Math.ceil(height / cellSize) + 2));
    for (let index = 0; index < lattice.length; index++) {
      lattice[index] = random() * 2 - 1;
    }
    const stride = Math.ceil(width / cellSize) + 2;
    for (let row = 0; row < height; row++) {
      for (let column = 0; column < width; column++) {
        const x = column / cellSize;
        const y = row / cellSize;
        const x0 = Math.floor(x);
        const y0 = Math.floor(y);
        const smooth = (t: number) => t * t * (3 - 2 * t);
        const fx = smooth(x - x0);
        const fy = smooth(y - y0);
        const top = lattice[y0 * stride + x0] * (1 - fx) + lattice[y0 * stride + x0 + 1] * fx;
        const bottom = lattice[(y0 + 1) * stride + x0] * (1 - fx) + lattice[(y0 + 1) * stride + x0 + 1] * fx;
        result[row * width + column] += octaveAmplitude * (top * (1 - fy) + bottom * fy);
      }
    }
    octaveAmplitude *= 0.55;
  }
  return result;
}
