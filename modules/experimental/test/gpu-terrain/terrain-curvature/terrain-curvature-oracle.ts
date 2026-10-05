// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getOracleGroundCellSize} from '../terrain-oracle-utils';

export type OracleCurvatureMethod = 'evans-young' | 'zevenbergen-thorne' | 'florinsky';

export const ORACLE_CURVATURE_KINDS = [
  'profile',
  'plan',
  'tangential',
  'mean',
  'gaussian',
  'minimal',
  'maximal',
  'unsphericity',
  'difference',
  'horizontal-excess',
  'vertical-excess',
  'accumulation',
  'ring',
  'rotor',
  'laplacian'
] as const;

export type OracleCurvatureKind = (typeof ORACLE_CURVATURE_KINDS)[number];

/** Options mirroring `GPUTerrainCurvature` topology props. */
export type TerrainCurvatureOptions = {
  method?: OracleCurvatureMethod;
  cellSizeMode?: 'uniform' | 'web-mercator' | 'geographic';
  rowDirection?: 'south' | 'north';
  borderMode?: 'clamp' | 'nodata';
  ringRadii?: readonly number[];
  ringSquash?: 'none' | 'pade-tanh';
};

/** Partial derivatives `p = dz/dx` (east), `q = dz/dy` (north), `r`, `s`, `t`. */
export type TerrainPartials = {p: number; q: number; r: number; s: number; t: number};

/** All curvature kinds from partial derivatives (Florinsky 2016, Whitebox signs), float64. */
export function computeCurvaturesFromPartials(
  {p, q, r, s, t}: TerrainPartials,
  flatGradient: number = 1e-6
): Record<OracleCurvatureKind, number> {
  const gradientSquared = p * p + q * q;
  const weight = 1 + gradientSquared;
  const isFlat = gradientSquared <= flatGradient * flatGradient;
  const planNumerator = q * q * r - 2 * p * q * s + p * p * t;
  const profileNumerator = p * p * r + 2 * p * q * s + q * q * t;
  const tangential = isFlat ? 0 : -planNumerator / (gradientSquared * Math.sqrt(weight));
  const plan = isFlat ? 0 : -planNumerator / Math.pow(gradientSquared, 1.5);
  const profile = isFlat ? 0 : -profileNumerator / (gradientSquared * Math.pow(weight, 1.5));
  const mean = -((1 + q * q) * r - 2 * p * q * s + (1 + p * p) * t) / (2 * Math.pow(weight, 1.5));
  const gaussian = (r * t - s * s) / (weight * weight);
  const unsphericity = Math.sqrt(Math.max(mean * mean - gaussian, 0));
  const minimal = mean - unsphericity;
  const maximal = mean + unsphericity;
  const horizontalExcess = isFlat ? 0 : tangential - minimal;
  const verticalExcess = isFlat ? 0 : profile - minimal;
  return {
    profile,
    plan,
    tangential,
    mean,
    gaussian,
    minimal,
    maximal,
    unsphericity,
    difference: (profile - tangential) / 2,
    'horizontal-excess': horizontalExcess,
    'vertical-excess': verticalExcess,
    accumulation: tangential * profile,
    ring: horizontalExcess * verticalExcess,
    rotor: isFlat ? 0 : ((p * p - q * q) * s - p * q * (r - t)) / Math.pow(gradientSquared, 1.5),
    laplacian: r + t
  };
}

/**
 * Partial derivatives from centre-relative samples. `sample(dx, north)` is the elevation
 * difference to the cell `dx` columns east and `north` cells toward geographic north.
 */
export function computePartials(
  method: OracleCurvatureMethod,
  sample: (dx: number, north: number) => number,
  wx: number,
  wy: number,
  zFactor: number
): TerrainPartials {
  let p: number;
  let q: number;
  let r: number;
  let s: number;
  let t: number;
  if (method === 'florinsky') {
    // z[n]: dx = n % 5 - 2, north = 2 - floor(n / 5) (WhiteboxTools lists the north row first).
    const z = Array.from({length: 25}, (_, n) => sample((n % 5) - 2, 2 - Math.floor(n / 5)));
    const sum = (...indices: number[]) => indices.reduce((total, index) => total + z[index], 0);
    r =
      (2 * sum(0, 4, 5, 9, 10, 14, 15, 19, 20, 24) -
        2 * sum(2, 7, 12, 17, 22) -
        sum(1, 3, 6, 8, 11, 13, 16, 18, 21, 23)) /
      (35 * wx * wx);
    t =
      (2 * sum(0, 1, 2, 3, 4, 20, 21, 22, 23, 24) -
        2 * sum(10, 11, 12, 13, 14) -
        sum(5, 6, 7, 8, 9, 15, 16, 17, 18, 19)) /
      (35 * wy * wy);
    s =
      (z[8] +
        z[16] -
        z[6] -
        z[18] +
        4 * (z[4] + z[20] - z[0] - z[24]) +
        2 * (z[3] + z[9] + z[15] + z[21] - z[1] - z[5] - z[19] - z[23])) /
      (100 * wx * wy);
    p =
      (44 * (z[3] + z[23] - z[1] - z[21]) +
        31 * (z[0] + z[20] - z[4] - z[24] + 2 * (z[8] + z[18] - z[6] - z[16])) +
        17 * (z[14] - z[10] + 4 * (z[13] - z[11])) +
        5 * (z[9] + z[19] - z[5] - z[15])) /
      (420 * wx);
    q =
      (44 * (z[5] + z[9] - z[15] - z[19]) +
        31 * (z[20] + z[24] - z[0] - z[4] + 2 * (z[6] + z[8] - z[16] - z[18])) +
        17 * (z[2] - z[22] + 4 * (z[7] - z[17])) +
        5 * (z[1] + z[3] - z[21] - z[23])) /
      (420 * wy);
  } else {
    const z1 = sample(-1, 1);
    const z2 = sample(0, 1);
    const z3 = sample(1, 1);
    const z4 = sample(-1, 0);
    const z6 = sample(1, 0);
    const z7 = sample(-1, -1);
    const z8 = sample(0, -1);
    const z9 = sample(1, -1);
    s = (z3 + z7 - z1 - z9) / (4 * wx * wy);
    if (method === 'zevenbergen-thorne') {
      p = (z6 - z4) / (2 * wx);
      q = (z2 - z8) / (2 * wy);
      r = (z4 + z6) / (wx * wx);
      t = (z2 + z8) / (wy * wy);
    } else {
      p = (z3 + z6 + z9 - z1 - z4 - z7) / (6 * wx);
      q = (z1 + z2 + z3 - z7 - z8 - z9) / (6 * wy);
      r = (z1 + z3 + z4 + z6 + z7 + z9 - 2 * (z2 + z8)) / (3 * wx * wx);
      t = (z1 + z2 + z3 + z7 + z8 + z9 - 2 * (z4 + z6)) / (3 * wy * wy);
    }
  }
  return {p: p * zFactor, q: q * zFactor, r: r * zFactor, s: s * zFactor, t: t * zFactor};
}

export type TerrainCurvatureResult = {
  curvatures: Record<OracleCurvatureKind, number[]>;
  ring: number[];
  /** 1 where the window curvatures are valid. */
  windowValidity: number[];
  /** 1 where the ring curvature is valid. */
  ringValidity: number[];
};

/** Float64 CPU mirror of `GPUTerrainCurvature`. */
export function computeTerrainCurvature(
  values: ArrayLike<number>,
  valid: ArrayLike<number> | undefined,
  width: number,
  height: number,
  settings: ArrayLike<number>,
  options: TerrainCurvatureOptions = {}
): TerrainCurvatureResult {
  const method = options.method ?? 'evans-young';
  const mode = options.cellSizeMode ?? 'uniform';
  const clampBorder = (options.borderMode ?? 'clamp') === 'clamp';
  const northSign = (options.rowDirection ?? 'south') === 'south' ? -1 : 1;
  const radii = options.ringRadii ?? [2, 8];
  const flatGradient = settings[5];
  const result: TerrainCurvatureResult = {
    curvatures: Object.fromEntries(
      ORACLE_CURVATURE_KINDS.map(kind => [kind, [] as number[]])
    ) as never,
    ring: [],
    windowValidity: [],
    ringValidity: []
  };
  const isUsable = (index: number) =>
    (valid ? valid[index] !== 0 : true) && Number.isFinite(values[index]);
  for (let index = 0; index < width * height; index++) {
    const row = Math.floor(index / width);
    const column = index % width;
    const [wx, wy] = getOracleGroundCellSize(settings, row, height, mode);
    const zFactor = settings[2];
    const cellsValid = wx > 0 && wy > 0 && Number.isFinite(zFactor);
    const centreValid = isUsable(index) && cellsValid;
    const centre = values[index];

    let windowValid = centreValid;
    const sampleWindow = (dx: number, north: number): number => {
      let sampleColumn = column + dx;
      let sampleRow = row + north * northSign;
      if (clampBorder) {
        sampleColumn = Math.min(Math.max(sampleColumn, 0), width - 1);
        sampleRow = Math.min(Math.max(sampleRow, 0), height - 1);
      } else if (
        sampleColumn < 0 ||
        sampleRow < 0 ||
        sampleColumn >= width ||
        sampleRow >= height
      ) {
        windowValid = false;
        return 0;
      }
      const sampleIndex = sampleRow * width + sampleColumn;
      if (!isUsable(sampleIndex)) {
        windowValid = false;
        return 0;
      }
      return values[sampleIndex] - centre;
    };
    const partials = computePartials(method, sampleWindow, wx, wy, zFactor);
    const curvatures = computeCurvaturesFromPartials(partials, flatGradient);
    windowValid &&= Object.values(partials).every(Number.isFinite);
    for (const kind of ORACLE_CURVATURE_KINDS) {
      result.curvatures[kind].push(windowValid ? curvatures[kind] : Number.NaN);
    }
    result.windowValidity.push(windowValid ? 1 : 0);

    let ringValid = centreValid;
    let ringValue = 0;
    const sampleRing = (dx: number, dy: number): number => {
      const sampleColumn = column + dx;
      const sampleRow = row + dy;
      if (sampleColumn < 0 || sampleRow < 0 || sampleColumn >= width || sampleRow >= height) {
        ringValid = false;
        return 0;
      }
      const sampleIndex = sampleRow * width + sampleColumn;
      if (!isUsable(sampleIndex)) {
        ringValid = false;
        return 0;
      }
      return values[sampleIndex] - centre;
    };
    for (const [ringIndex, radius] of radii.entries()) {
      const diagonal = Math.max(1, Math.round(radius / Math.SQRT2));
      let sum = 0;
      for (const [dx, dy] of [
        [radius, 0],
        [-radius, 0],
        [0, radius],
        [0, -radius],
        [diagonal, diagonal],
        [diagonal, -diagonal],
        [-diagonal, diagonal],
        [-diagonal, -diagonal]
      ]) {
        sum += sampleRing(dx, dy);
      }
      ringValue += (settings[6 + ringIndex] * -sum) / (8 * radius * Math.sqrt(wx * wy));
    }
    ringValue *= zFactor;
    if (options.ringSquash === 'pade-tanh') {
      const x = Math.min(3, Math.max(-3, ringValue));
      ringValue = (x * (27 + x * x)) / (27 + 9 * x * x);
    }
    ringValid &&= Number.isFinite(ringValue);
    result.ring.push(ringValid ? ringValue : Number.NaN);
    result.ringValidity.push(ringValid ? 1 : 0);
  }
  return result;
}
