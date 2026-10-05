// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateParameterTarget} from './raster-algebra-utils';

/** Number of float32 elements in a {@link GPURasterCellStatistics} parameter view. */
export const GPU_RASTER_CELL_STATISTICS_PARAMETER_LENGTH = 4;
/** Number of float32 elements in a {@link GPURasterConditional} parameter view. */
export const GPU_RASTER_CONDITIONAL_PARAMETER_LENGTH = 8;
/** Number of float32 elements in a {@link GPURasterArithmetic} parameter view. */
export const GPU_RASTER_ARITHMETIC_PARAMETER_LENGTH = 8;

/** Per-frame settings of {@link GPURasterCellStatistics}. */
export type GPURasterCellStatisticsSettings = {
  /**
   * `'ignore'` (default, ArcGIS "DATA"): statistics use the valid layers of each cell.
   * `'propagate'`: any nodata layer makes the cell nodata.
   */
  noDataPolicy?: 'ignore' | 'propagate';
  /** Cells with fewer valid layers are nodata. Defaults to 1. */
  minimumValidCount?: number;
};

/**
 * Packs per-frame {@link GPURasterCellStatistics} parameters.
 *
 * Layout (float32): `[propagateNoData, minimumValidCount, 0, 0]`.
 *
 * @param settings Nodata policy and minimum valid layer count.
 * @param target Optional destination of at least 4 elements.
 */
export function getGPURasterCellStatisticsParameterValues(
  settings: GPURasterCellStatisticsSettings = {},
  target: Float32Array = new Float32Array(GPU_RASTER_CELL_STATISTICS_PARAMETER_LENGTH)
): Float32Array {
  validateParameterTarget(
    'Raster cell statistics',
    target,
    GPU_RASTER_CELL_STATISTICS_PARAMETER_LENGTH
  );
  const minimumValidCount = settings.minimumValidCount ?? 1;
  if (!Number.isSafeInteger(minimumValidCount) || minimumValidCount < 0) {
    throw new Error('Raster cell statistics minimumValidCount must be a non-negative integer');
  }
  target[0] = settings.noDataPolicy === 'propagate' ? 1 : 0;
  target[1] = minimumValidCount;
  target[2] = 0;
  target[3] = 0;
  return target;
}

/** Comparison applied to `conditionValues` by {@link GPURasterConditional}. */
export type GPURasterComparison = '<' | '<=' | '>' | '>=' | '==' | '!=' | 'between';

const COMPARISON_CODES: Record<GPURasterComparison, number> = {
  '<': 0,
  '<=': 1,
  '>': 2,
  '>=': 3,
  '==': 4,
  '!=': 5,
  between: 6
};

/** Per-frame settings of {@link GPURasterConditional}. */
export type GPURasterConditionalSettings = {
  /** Comparison of `conditionValues` against `threshold`. Ignored with a `mask`. Defaults to `'>'`. */
  comparison?: GPURasterComparison;
  /** Threshold, or the inclusive lower bound for `'between'`. Defaults to 0. */
  threshold?: number;
  /** Inclusive upper bound for `'between'`. Defaults to `threshold`. */
  upperThreshold?: number;
  /** Value used where the condition holds when no `a` view is bound. Defaults to 1. */
  constantA?: number;
  /** Value used where the condition fails when no `b` view is bound. Defaults to 0. */
  constantB?: number;
};

/**
 * Packs per-frame {@link GPURasterConditional} parameters.
 *
 * Layout (float32): `[comparison, threshold, upperThreshold, constantA, constantB, 0, 0, 0]` with
 * comparison codes `<` 0, `<=` 1, `>` 2, `>=` 3, `==` 4, `!=` 5, `between` 6.
 *
 * @param settings Comparison, thresholds, and constants.
 * @param target Optional destination of at least 8 elements.
 */
export function getGPURasterConditionalParameterValues(
  settings: GPURasterConditionalSettings = {},
  target: Float32Array = new Float32Array(GPU_RASTER_CONDITIONAL_PARAMETER_LENGTH)
): Float32Array {
  validateParameterTarget('Raster conditional', target, GPU_RASTER_CONDITIONAL_PARAMETER_LENGTH);
  const comparison = settings.comparison ?? '>';
  if (!(comparison in COMPARISON_CODES)) {
    throw new Error(`Raster conditional comparison ${comparison} is not supported`);
  }
  const threshold = settings.threshold ?? 0;
  target.fill(0, 0, GPU_RASTER_CONDITIONAL_PARAMETER_LENGTH);
  target[0] = COMPARISON_CODES[comparison];
  target[1] = threshold;
  target[2] = settings.upperThreshold ?? threshold;
  target[3] = settings.constantA ?? 1;
  target[4] = settings.constantB ?? 0;
  return target;
}

/** Binary operations of {@link GPURasterArithmetic} on `a'` and `b'`. */
export type GPURasterBinaryOperation =
  | 'add'
  | 'subtract'
  | 'multiply'
  | 'divide'
  | 'minimum'
  | 'maximum'
  | 'power'
  | 'absoluteDifference'
  | 'normalizedDifference';

/** Unary operations of {@link GPURasterArithmetic} on `a'`. */
export type GPURasterUnaryOperation =
  | 'absolute'
  | 'squareRoot'
  | 'logarithm'
  | 'exponential'
  | 'floor'
  | 'ceil'
  | 'round';

/** Operation codes written into the {@link GPURasterArithmetic} parameter view. */
export const GPU_RASTER_ARITHMETIC_OPERATION_CODES: Readonly<
  Record<GPURasterBinaryOperation | GPURasterUnaryOperation, number>
> = {
  add: 0,
  subtract: 1,
  multiply: 2,
  divide: 3,
  minimum: 4,
  maximum: 5,
  power: 6,
  absoluteDifference: 7,
  normalizedDifference: 8,
  absolute: 16,
  squareRoot: 17,
  logarithm: 18,
  exponential: 19,
  floor: 20,
  ceil: 21,
  round: 22
};

/** Per-frame settings of {@link GPURasterArithmetic}. */
export type GPURasterArithmeticSettings = {
  /** Operation applied to the scaled operands. */
  operation: GPURasterBinaryOperation | GPURasterUnaryOperation;
  /** `a' = a * scaleA + offsetA`. Defaults to 1. */
  scaleA?: number;
  /** Defaults to 0. */
  offsetA?: number;
  /** `b' = b * scaleB + offsetB`. Defaults to 1. */
  scaleB?: number;
  /** Defaults to 0. */
  offsetB?: number;
  /** `b` when no `b` view is bound. Defaults to 0. */
  constantB?: number;
  /** Optional output clamp lower bound. Defaults to `-Infinity`. */
  clampMin?: number;
  /** Optional output clamp upper bound. Defaults to `+Infinity`. */
  clampMax?: number;
};

/**
 * Packs per-frame {@link GPURasterArithmetic} parameters.
 *
 * Layout (float32): `[operation, scaleA, offsetA, scaleB, offsetB, constantB, clampMin, clampMax]`
 * with codes from {@link GPU_RASTER_ARITHMETIC_OPERATION_CODES}.
 *
 * @param settings Operation, operand scales, constant, and clamp.
 * @param target Optional destination of at least 8 elements.
 */
export function getGPURasterArithmeticParameterValues(
  settings: GPURasterArithmeticSettings,
  target: Float32Array = new Float32Array(GPU_RASTER_ARITHMETIC_PARAMETER_LENGTH)
): Float32Array {
  validateParameterTarget('Raster arithmetic', target, GPU_RASTER_ARITHMETIC_PARAMETER_LENGTH);
  const code = GPU_RASTER_ARITHMETIC_OPERATION_CODES[settings.operation];
  if (code === undefined) {
    throw new Error(`Raster arithmetic operation ${settings.operation} is not supported`);
  }
  target[0] = code;
  target[1] = settings.scaleA ?? 1;
  target[2] = settings.offsetA ?? 0;
  target[3] = settings.scaleB ?? 1;
  target[4] = settings.offsetB ?? 0;
  target[5] = settings.constantB ?? 0;
  target[6] = settings.clampMin ?? -Infinity;
  target[7] = settings.clampMax ?? Infinity;
  return target;
}
