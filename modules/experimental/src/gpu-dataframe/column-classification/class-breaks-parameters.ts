// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateCapacity} from '../../utils/gpu-contributor-utils';

/** Classification methods supported by `GPUClassBreaks`. */
export type GPUClassBreaksMethod =
  | 'equal-interval'
  | 'quantile'
  | 'standard-deviation'
  | 'head-tail'
  | 'box-plot'
  | 'maximum-breaks'
  | 'natural-breaks'
  | 'custom';

/** Per-frame method codes stored in element 0 of the `GPUClassBreaks` parameter view. */
export const GPU_CLASS_BREAKS_METHOD_CODES: Readonly<Record<GPUClassBreaksMethod, number>> = {
  'equal-interval': 0,
  quantile: 1,
  'standard-deviation': 2,
  'head-tail': 3,
  'box-plot': 4,
  'maximum-breaks': 5,
  'natural-breaks': 6,
  custom: 7
};

/** Every method, in code order. */
export const GPU_CLASS_BREAKS_METHODS: readonly GPUClassBreaksMethod[] = Object.keys(
  GPU_CLASS_BREAKS_METHOD_CODES
) as GPUClassBreaksMethod[];

/** Number of float32 header elements before the method data in the parameter view. */
export const GPU_CLASS_BREAKS_PARAMETER_HEADER_LENGTH = 8;

/** Number of classes a box plot always produces. */
export const GPU_CLASS_BREAKS_BOX_PLOT_CLASS_COUNT = 6;

/**
 * Returns the float32 parameter length of a `GPUClassBreaks` with `maximumClassCount` classes.
 *
 * @param maximumClassCount Compile-time class capacity of the contributor.
 */
export function getGPUClassBreaksParameterLength(maximumClassCount: number): number {
  return GPU_CLASS_BREAKS_PARAMETER_HEADER_LENGTH + maximumClassCount + 1;
}

/** Per-frame settings packed by {@link getGPUClassBreaksParameterValues}. */
export type GPUClassBreaksSettings = {
  /** Classification method. It must be one of the contributor's compiled `methods`. */
  method: GPUClassBreaksMethod;
  /**
   * Requested class count `k`. Ignored by `'box-plot'` (always 6) and `'custom'` (one less than
   * the number of edges). For `'head-tail'` and `'maximum-breaks'` it is an upper bound: fewer
   * classes are produced when the data runs out of heads or distinct gaps.
   */
  classCount?: number;
  /** `'standard-deviation'`: width of one class in standard deviations. Default 1. */
  standardDeviationInterval?: number;
  /**
   * `'head-tail'`: largest head fraction that keeps splitting. Default 0.4 (Jiang's 40% rule).
   * Use 1 for mapclassify's rule, which splits until a head has a single distinct value.
   */
  headTailRatio?: number;
  /** `'box-plot'`: whisker length in interquartile ranges. Default 1.5. */
  boxPlotHinge?: number;
  /** `'custom'`: ascending class edges `e[0..k]`, at most `maximumClassCount + 1` of them. */
  customEdges?: ArrayLike<number>;
};

/**
 * Packs per-frame `GPUClassBreaks` parameters.
 *
 * Layout (float32): `[methodCode, classCount, standardDeviationInterval, headTailRatio,
 * boxPlotHinge, 0, 0, 0, data[0..maximumClassCount]]`. `data` holds the quantile probabilities
 * `fround(i / k)` for `i = 1..k-1` (`'quantile'`), `[0.25, 0.5, 0.75]` (`'box-plot'`), or the
 * custom edges (`'custom'`), and NaN elsewhere. Probabilities are computed here on the CPU so the
 * GPU quantiles match a CPU oracle bit for bit.
 *
 * @param settings Per-frame settings.
 * @param maximumClassCount Compile-time class capacity of the contributor.
 * @param target Optional destination of at least `getGPUClassBreaksParameterLength(maximumClassCount)` elements.
 * @throws If the class count is out of range, custom edges do not fit or one is NaN, or a method
 * setting (interval, ratio, hinge) is not finite.
 */
export function getGPUClassBreaksParameterValues(
  settings: GPUClassBreaksSettings,
  maximumClassCount: number,
  target: Float32Array = new Float32Array(getGPUClassBreaksParameterLength(maximumClassCount))
): Float32Array {
  const length = getGPUClassBreaksParameterLength(maximumClassCount);
  if (target.length < length) {
    throw new Error(`GPUClassBreaks parameter target must hold ${length} elements`);
  }
  const {method} = settings;
  const methodCode = GPU_CLASS_BREAKS_METHOD_CODES[method];
  if (methodCode === undefined) {
    throw new Error(`GPUClassBreaks unknown method ${String(method)}`);
  }
  let classCount = settings.classCount ?? 5;
  if (method === 'box-plot') {
    classCount = GPU_CLASS_BREAKS_BOX_PLOT_CLASS_COUNT;
  } else if (method === 'custom') {
    const edgeCount = settings.customEdges?.length ?? 0;
    if (edgeCount < 2) {
      throw new Error('GPUClassBreaks custom needs at least two edges');
    }
    classCount = edgeCount - 1;
  }
  validateCapacity('GPUClassBreaks', 'classCount', classCount, 1, maximumClassCount);
  for (const [name, value] of [
    ['standardDeviationInterval', settings.standardDeviationInterval],
    ['headTailRatio', settings.headTailRatio],
    ['boxPlotHinge', settings.boxPlotHinge]
  ] as const) {
    if (value !== undefined && !Number.isFinite(value)) {
      throw new Error(`GPUClassBreaks ${name} must be finite`);
    }
  }
  if (method === 'custom' && Array.from(settings.customEdges!).some(Number.isNaN)) {
    throw new Error('GPUClassBreaks custom edges must not be NaN');
  }
  target.fill(NaN, 0, length);
  target[0] = methodCode;
  target[1] = classCount;
  target[2] = settings.standardDeviationInterval ?? 1;
  target[3] = settings.headTailRatio ?? 0.4;
  target[4] = settings.boxPlotHinge ?? 1.5;
  target[5] = 0;
  target[6] = 0;
  target[7] = 0;
  const data = GPU_CLASS_BREAKS_PARAMETER_HEADER_LENGTH;
  if (method === 'quantile') {
    for (let index = 1; index < classCount; index++) {
      target[data + index - 1] = index / classCount;
    }
  } else if (method === 'box-plot') {
    target[data] = 0.25;
    target[data + 1] = 0.5;
    target[data + 2] = 0.75;
  } else if (method === 'custom') {
    const edges = settings.customEdges!;
    for (let index = 0; index < edges.length; index++) {
      target[data + index] = edges[index];
    }
  }
  return target;
}
