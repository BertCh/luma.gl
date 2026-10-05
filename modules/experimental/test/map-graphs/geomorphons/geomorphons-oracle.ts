// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Float64 CPU oracle mirroring GRASS r.geomorphon with literal arctangent comparisons. */

export type GeomorphonOracleOptions = {
  searchRadius: number;
  skipRadius?: number;
  comparison?: 'anglev1' | 'anglev2' | 'anglev2-distance';
  cellSizeMode?: 'uniform' | 'web-mercator' | 'geographic';
  rowDirection?: 'south' | 'north';
};

export type GeomorphonOracleResult = {
  forms: number[];
  ternary: number[];
  pattern: number[];
  validity: number[];
};

// Directions in GRASS order: NE, N, NW, W, SW, S, SE, E as (column, north) steps.
const DIRECTION_COLUMN = [1, 0, -1, -1, -1, 0, 1, 1];
const DIRECTION_NORTH = [1, 1, 1, 0, -1, -1, -1, 0];

// Rows are minus counts 0..8, columns are plus counts 0..8; 0 marks impossible pairs.
const FORM_TABLE = [
  [1, 1, 1, 8, 8, 9, 9, 9, 10],
  [1, 1, 8, 8, 8, 9, 9, 9, 0],
  [1, 4, 6, 6, 7, 7, 9, 0, 0],
  [4, 4, 6, 6, 6, 7, 0, 0, 0],
  [4, 4, 5, 6, 6, 0, 0, 0, 0],
  [3, 3, 5, 5, 0, 0, 0, 0, 0],
  [3, 3, 3, 0, 0, 0, 0, 0, 0],
  [3, 3, 0, 0, 0, 0, 0, 0, 0],
  [2, 0, 0, 0, 0, 0, 0, 0, 0]
];

/** Returns the landform code for the given minus and plus counts (0 when impossible). */
export function getGeomorphonForm(minusCount: number, plusCount: number): number {
  return FORM_TABLE[minusCount][plusCount];
}

/** Rotation- and mirror-invariant ternary class of a 6561-valued pattern code. */
export function getRotatedTernaryCode(value: number): number {
  const pattern: number[] = [];
  let remaining = value;
  for (let index = 0; index < 8; index++) {
    pattern.push(remaining % 3);
    remaining = Math.floor(remaining / 3);
  }
  const mirrored = pattern.slice().reverse();
  let best = Infinity;
  for (const digits of [pattern, mirrored]) {
    for (let shift = 0; shift < 8; shift++) {
      let code = 0;
      let power = 1;
      for (let index = 0; index < 8; index++) {
        code += digits[(index - shift + 8) % 8] * power;
        power *= 3;
      }
      best = Math.min(best, code);
    }
  }
  return best;
}

function getGroundCellSize(
  settings: ArrayLike<number>,
  height: number,
  row: number,
  mode: string
): [number, number] {
  const cellX = settings[0];
  const cellY = settings[1];
  if (mode === 'uniform') {
    return [cellX, cellY];
  }
  const fraction = (row + 0.5) / height;
  const edge = settings[3] + (settings[4] - settings[3]) * fraction;
  if (mode === 'web-mercator') {
    const scale = Math.cosh(Math.PI * (1 - 2 * edge));
    return [cellX / scale, cellY / scale];
  }
  const metresPerDegree = 111319.49079327357;
  return [cellX * metresPerDegree * Math.cos((edge * Math.PI) / 180), cellY * metresPerDegree];
}

// Angles that differ by less than this are equal: exact ratios such as 2/20 and 3/30 arctangent
// to values one ulp apart in float64, and GRASS resolves those ties by luck of rounding.
const ANGLE_TIE_TOLERANCE = 1e-12;
const isLess = (left: number, right: number) => right - left > ANGLE_TIE_TOLERANCE;
const isGreater = (left: number, right: number) => left - right > ANGLE_TIE_TOLERANCE;

function compareMultiple(
  nadirAngle: number,
  zenithAngle: number,
  nadirThreshold: number,
  zenithThreshold: number,
  nadirDistance: number,
  zenithDistance: number
): number {
  const nadirOver = nadirAngle > nadirThreshold;
  const zenithOver = zenithAngle > zenithThreshold;
  if (!nadirOver && !zenithOver) return 0;
  if (!nadirOver && zenithOver) return 1;
  if (nadirOver && !zenithOver) return -1;
  if (isLess(nadirAngle, zenithAngle)) return 1;
  if (isGreater(nadirAngle, zenithAngle)) return -1;
  if (nadirDistance < zenithDistance) return 1;
  if (nadirDistance > zenithDistance) return -1;
  return 1;
}

/** Computes forms, ternary classes, pattern codes, and validity for a whole grid. */
export function computeGeomorphons(
  elevation: ArrayLike<number>,
  mask: ArrayLike<number> | undefined,
  width: number,
  height: number,
  settings: ArrayLike<number>,
  options: GeomorphonOracleOptions
): GeomorphonOracleResult {
  const skipRadius = options.skipRadius ?? 0;
  const comparison = options.comparison ?? 'anglev1';
  const result: GeomorphonOracleResult = {forms: [], ternary: [], pattern: [], validity: []};
  const isValid = (column: number, row: number) =>
    Number.isFinite(elevation[row * width + column]) && (!mask || mask[row * width + column] !== 0);
  const flatThreshold = (settings[5] * Math.PI) / 180;
  const flatDistance = settings[6];
  const flatThresholdHeight = Math.tan(flatThreshold) * flatDistance;
  const zFactor = settings[2];
  const northSign = (options.rowDirection ?? 'south') === 'south' ? -1 : 1;

  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      const inside =
        row >= skipRadius + 1 &&
        row < height - (skipRadius + 1) &&
        column >= skipRadius + 1 &&
        column < width - (skipRadius + 1);
      if (!inside || !isValid(column, row)) {
        result.forms.push(0);
        result.ternary.push(0);
        result.pattern.push(0);
        result.validity.push(0);
        continue;
      }
      const [cellX, cellY] = getGroundCellSize(
        settings,
        height,
        row,
        options.cellSizeMode ?? 'uniform'
      );
      // Mirrors the GPU's 1e-5 relative guard that excludes samples exactly at the search limit.
      const searchDistance = options.searchRadius * cellY * 0.99999;
      const center = elevation[row * width + column];
      const digits: number[] = [];
      let minusCount = 0;
      let plusCount = 0;
      for (let direction = 0; direction < 8; direction++) {
        digits.push(0);
        const columnStep = DIRECTION_COLUMN[direction];
        const rowStep = DIRECTION_NORTH[direction] * northSign;
        if (!isValid(column + columnStep, row + rowStep)) continue;
        const step = Math.hypot(columnStep * cellX, DIRECTION_NORTH[direction] * cellY);
        let zenithAngle = -Infinity;
        let nadirAngle = Infinity;
        let zenithDistance = 0;
        let nadirDistance = 0;
        let count = skipRadius + 1;
        while (count * step < searchDistance) {
          const sampleColumn = column + count * columnStep;
          const sampleRow = row + count * rowStep;
          if (sampleColumn < 0 || sampleColumn >= width || sampleRow < 0 || sampleRow >= height) {
            break;
          }
          if (isValid(sampleColumn, sampleRow)) {
            const sampleHeight = zFactor * (elevation[sampleRow * width + sampleColumn] - center);
            const distance = count * step;
            const angle = Math.atan2(sampleHeight, distance);
            if (isGreater(angle, zenithAngle)) {
              zenithAngle = angle;
              zenithDistance = distance;
            }
            if (isLess(angle, nadirAngle)) {
              nadirAngle = angle;
              nadirDistance = distance;
            }
          }
          count++;
        }
        if (zenithDistance === 0) continue;
        const zenithThreshold =
          flatDistance > 0 && flatDistance < zenithDistance
            ? Math.atan2(flatThresholdHeight, zenithDistance)
            : flatThreshold;
        const nadirThreshold =
          flatDistance > 0 && flatDistance < nadirDistance
            ? Math.atan2(flatThresholdHeight, nadirDistance)
            : flatThreshold;
        let digit = 0;
        if (comparison === 'anglev1') {
          if (Math.abs(zenithAngle) > zenithThreshold || Math.abs(nadirAngle) > nadirThreshold) {
            if (isLess(Math.abs(nadirAngle), Math.abs(zenithAngle))) digit = 1;
            if (isGreater(Math.abs(nadirAngle), Math.abs(zenithAngle))) digit = -1;
          }
        } else {
          digit = compareMultiple(
            Math.abs(nadirAngle),
            Math.abs(zenithAngle),
            nadirThreshold,
            zenithThreshold,
            comparison === 'anglev2-distance' ? nadirDistance : 0,
            comparison === 'anglev2-distance' ? zenithDistance : 0
          );
        }
        digits[direction] = digit;
        if (digit > 0) plusCount++;
        if (digit < 0) minusCount++;
      }
      let code = 0;
      let power = 1;
      for (const digit of digits) {
        code += (digit + 1) * power;
        power *= 3;
      }
      result.forms.push(getGeomorphonForm(minusCount, plusCount));
      result.ternary.push(getRotatedTernaryCode(code));
      result.pattern.push(code);
      result.validity.push(1);
    }
  }
  return result;
}
