// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// CPU oracles for the raster-algebra recipes. Not exported from src. Every f32 operation is
// rounded with Math.fround in the same association as the WGSL kernels.

const f = Math.fround;

/** Returns whether a value is nodata: NaN or the optional sentinel. */
export function isNoData(value: number, noDataValue?: number): boolean {
  return Number.isNaN(value) || (noDataValue !== undefined && value === f(noDataValue));
}

/** Number of ascending breaks `<= value` (or `< value` when `closedRight`). */
export function countBreaksBelow(
  breaks: ArrayLike<number>,
  first: number,
  count: number,
  value: number,
  closedRight: boolean
): number {
  let low = 0;
  let high = count;
  while (low < high) {
    const middle = (low + high) >>> 1;
    const boundary = breaks[first + middle];
    const isBelow = closedRight ? boundary < value : boundary <= value;
    if (isBelow) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  return low;
}

export type ReclassifyScene = {
  values: Float32Array;
  validity?: Uint32Array;
  noDataValue?: number;
  breaks: Float32Array;
  classValues?: Float32Array;
};

export type ReclassifyResult = {
  classes: Uint32Array;
  reclassified: Float32Array;
  classCounts: Uint32Array;
};

/** Reclassifies `values` through the first `breakCount` breaks. */
export function reclassifyOnCPU(
  scene: ReclassifyScene,
  breakCount: number,
  closedRight: boolean
): ReclassifyResult {
  const maximumBreakCount = scene.breaks.length;
  const activeCount = Math.min(breakCount, maximumBreakCount);
  const classes = new Uint32Array(scene.values.length);
  const reclassified = new Float32Array(scene.values.length);
  const classCounts = new Uint32Array(maximumBreakCount + 1);
  for (let row = 0; row < scene.values.length; row++) {
    const value = scene.values[row];
    const isValid =
      !isNoData(value, scene.noDataValue) && (!scene.validity || scene.validity[row] !== 0);
    if (!isValid) {
      classes[row] = 0xffffffff;
      reclassified[row] = NaN;
      continue;
    }
    const classIndex = countBreaksBelow(scene.breaks, 0, activeCount, value, closedRight);
    classes[row] = classIndex;
    reclassified[row] = scene.classValues ? scene.classValues[classIndex] : NaN;
    classCounts[classIndex]++;
  }
  return {classes, reclassified, classCounts};
}

export type WeightedOverlayScene = {
  stack: Float32Array;
  layerCount: number;
  cellCount: number;
  noDataValue?: number;
  remapBreaks?: Float32Array;
  remapValues?: Float32Array;
  maximumBreakCount?: number;
};

export type WeightedOverlayResult = {
  score: Float32Array;
  validity: Uint32Array;
  scoreRange: [number, number];
  /** Per cell `sum(|weight * remapped|)` (divided by the weight sum when normalized), for tolerances. */
  magnitude: Float64Array;
};

/** Evaluates a weighted overlay from the packed float32 parameter view. */
export function weightedOverlayOnCPU(
  scene: WeightedOverlayScene,
  parameters: Float32Array
): WeightedOverlayResult {
  const {layerCount, cellCount} = scene;
  const maximumBreakCount = scene.remapBreaks ? (scene.maximumBreakCount ?? 0) : 0;
  const score = new Float32Array(cellCount);
  const validity = new Uint32Array(cellCount);
  const magnitude = new Float64Array(cellCount);
  let minimum = Infinity;
  let maximum = -Infinity;
  const normalizeWeights = parameters[0] !== 0;
  const ignoreNoData = parameters[1] !== 0;
  for (let cell = 0; cell < cellCount; cell++) {
    let sum = 0;
    let weightSum = 0;
    let usedCount = 0;
    let isDefined = true;
    for (let layer = 0; layer < layerCount; layer++) {
      const base = 8 + layer * 8;
      const weight = parameters[base];
      const value = scene.stack[layer * cellCount + cell];
      if (isNoData(value, scene.noDataValue)) {
        if (!ignoreNoData) {
          isDefined = false;
        }
        continue;
      }
      let remapped: number;
      if (parameters[base + 1] === 0) {
        const inputMin = parameters[base + 2];
        if (parameters[base + 7] !== 0) {
          remapped = value >= inputMin ? 1 : 0;
        } else {
          remapped = Math.min(Math.max(f(f(value - inputMin) * parameters[base + 3]), 0), 1);
        }
        if (parameters[base + 4] !== 0) {
          remapped = f(1 - remapped);
        }
      } else if (scene.remapBreaks && scene.remapValues) {
        const breakCount = Math.min(parameters[base + 5], maximumBreakCount);
        const classIndex = countBreaksBelow(
          scene.remapBreaks,
          layer * maximumBreakCount,
          breakCount,
          value,
          parameters[base + 6] !== 0
        );
        remapped = scene.remapValues[layer * (maximumBreakCount + 1) + classIndex];
      } else {
        remapped = NaN;
      }
      if (Number.isNaN(remapped)) {
        isDefined = false;
      }
      sum = f(sum + f(weight * remapped));
      magnitude[cell] += Math.abs(weight * remapped);
      weightSum = f(weightSum + Math.abs(weight));
      usedCount++;
    }
    isDefined = isDefined && usedCount > 0;
    let cellScore = sum;
    if (normalizeWeights) {
      isDefined = isDefined && weightSum > 0;
      cellScore = f(sum / weightSum);
      magnitude[cell] /= weightSum;
    }
    if (!isDefined || Number.isNaN(cellScore)) {
      cellScore = NaN;
    }
    score[cell] = cellScore;
    validity[cell] = Number.isNaN(cellScore) ? 0 : 1;
    if (!Number.isNaN(cellScore)) {
      minimum = Math.min(minimum, cellScore);
      maximum = Math.max(maximum, cellScore);
    }
  }
  const scoreRange: [number, number] = minimum <= maximum ? [minimum, maximum] : [NaN, NaN];
  return {score, validity, scoreRange, magnitude};
}

export type CellStatisticsResult = {
  minimum: Float32Array;
  maximum: Float32Array;
  range: Float32Array;
  sum: Float32Array;
  mean: Float32Array;
  standardDeviation: Float32Array;
  majority: Float32Array;
  minority: Float32Array;
  variety: Uint32Array;
  count: Uint32Array;
};

/** Per-cell statistics across a band-sequential stack. */
export function computeCellStatisticsOnCPU(
  stack: Float32Array,
  layerCount: number,
  cellCount: number,
  options: {noDataValue?: number; propagateNoData?: boolean; minimumValidCount?: number}
): CellStatisticsResult {
  const result: CellStatisticsResult = {
    minimum: new Float32Array(cellCount),
    maximum: new Float32Array(cellCount),
    range: new Float32Array(cellCount),
    sum: new Float32Array(cellCount),
    mean: new Float32Array(cellCount),
    standardDeviation: new Float32Array(cellCount),
    majority: new Float32Array(cellCount),
    minority: new Float32Array(cellCount),
    variety: new Uint32Array(cellCount),
    count: new Uint32Array(cellCount)
  };
  const minimumCount = Math.max(options.minimumValidCount ?? 1, 1);
  for (let cell = 0; cell < cellCount; cell++) {
    const values: number[] = [];
    for (let layer = 0; layer < layerCount; layer++) {
      const value = stack[layer * cellCount + cell];
      if (!isNoData(value, options.noDataValue)) {
        values.push(value);
      }
    }
    const count = values.length;
    result.count[cell] = count;
    const isDefined = count >= minimumCount && (!options.propagateNoData || count === layerCount);
    if (!isDefined) {
      for (const name of [
        'minimum',
        'maximum',
        'range',
        'sum',
        'mean',
        'standardDeviation',
        'majority',
        'minority'
      ] as const) {
        result[name][cell] = NaN;
      }
      result.variety[cell] = 0;
      continue;
    }
    let sum = 0;
    let minimum = values[0];
    let maximum = values[0];
    for (const value of values) {
      sum = f(sum + value);
      minimum = Math.min(minimum, value);
      maximum = Math.max(maximum, value);
    }
    const mean = f(sum / count);
    let squaredDeviationSum = 0;
    for (const value of values) {
      const deviation = f(value - mean);
      squaredDeviationSum = f(squaredDeviationSum + f(deviation * deviation));
    }
    result.sum[cell] = sum;
    result.mean[cell] = mean;
    result.standardDeviation[cell] = f(Math.sqrt(f(squaredDeviationSum / count)));
    result.minimum[cell] = minimum;
    result.maximum[cell] = maximum;
    result.range[cell] = f(maximum - minimum);
    // Frequencies: first occurrence in layer order defines a distinct value.
    let majority = 0;
    let majorityCount = 0;
    let minority = 0;
    let minorityCount = Infinity;
    let variety = 0;
    for (let index = 0; index < values.length; index++) {
      const value = values[index];
      let frequency = 0;
      let isFirst = true;
      for (let other = 0; other < values.length; other++) {
        // `==` treats -0 and +0 as equal, like WGSL.
        if (values[other] === value) {
          frequency++;
          if (other < index) {
            isFirst = false;
          }
        }
      }
      if (!isFirst) {
        continue;
      }
      variety++;
      if (frequency > majorityCount || (frequency === majorityCount && value < majority)) {
        majority = value;
        majorityCount = frequency;
      }
      if (frequency < minorityCount || (frequency === minorityCount && value < minority)) {
        minority = value;
        minorityCount = frequency;
      }
    }
    result.majority[cell] = majority;
    result.minority[cell] = minority;
    result.variety[cell] = variety;
  }
  return result;
}

/** `where(condition, a, b)` with the packed conditional parameters. */
export function conditionalOnCPU(
  cellCount: number,
  inputs: {
    mask?: Uint32Array;
    conditionValues?: Float32Array;
    a?: Float32Array;
    b?: Float32Array;
    noDataValue?: number;
  },
  parameters: Float32Array
): {values: Float32Array; mask: Uint32Array} {
  const values = new Float32Array(cellCount);
  const mask = new Uint32Array(cellCount);
  const [comparison, threshold, upperThreshold, constantA, constantB] = parameters;
  for (let cell = 0; cell < cellCount; cell++) {
    let conditionDefined = true;
    let condition: boolean;
    if (inputs.mask) {
      condition = inputs.mask[cell] !== 0;
    } else {
      const value = inputs.conditionValues![cell];
      conditionDefined = !isNoData(value, inputs.noDataValue);
      condition =
        [
          value < threshold,
          value <= threshold,
          value > threshold,
          value >= threshold,
          value === threshold,
          value !== threshold
        ][comparison] ??
        (value >= threshold && value <= upperThreshold);
    }
    const aValue = inputs.a ? inputs.a[cell] : constantA;
    const bValue = inputs.b ? inputs.b[cell] : constantB;
    const chosen = condition ? aValue : bValue;
    values[cell] = conditionDefined && !isNoData(chosen, inputs.noDataValue) ? chosen : NaN;
    mask[cell] = conditionDefined && condition ? 1 : 0;
  }
  return {values, mask};
}

/** Round half to even, like WGSL `round`. */
function roundHalfToEven(value: number): number {
  const floorValue = Math.floor(value);
  const difference = value - floorValue;
  let rounded: number;
  if (difference > 0.5) {
    rounded = floorValue + 1;
  } else if (difference < 0.5) {
    rounded = floorValue;
  } else {
    rounded = floorValue % 2 === 0 ? floorValue : floorValue + 1;
  }
  // Keep the sign of zero, like IEEE roundTiesToEven.
  return rounded === 0 && (value < 0 || Object.is(value, -0)) ? -0 : rounded;
}

/**
 * Raster arithmetic with the packed parameters. Returns f64-evaluated transcendental results
 * rounded to f32, so tests compare those with a tolerance.
 */
export function arithmeticOnCPU(
  a: Float32Array,
  b: Float32Array | undefined,
  parameters: Float32Array,
  noDataValue?: number
): Float32Array {
  const [operation, scaleA, offsetA, scaleB, offsetB, constantB, clampMin, clampMax] = parameters;
  const result = new Float32Array(a.length);
  for (let cell = 0; cell < a.length; cell++) {
    const aRaw = a[cell];
    const bRaw = b ? b[cell] : constantB;
    const isUnary = operation >= 16;
    const isDefined = !isNoData(aRaw, noDataValue) && (isUnary || !isNoData(bRaw, noDataValue));
    const x = f(f(aRaw * scaleA) + offsetA);
    const y = f(f(bRaw * scaleB) + offsetB);
    let value: number;
    switch (operation) {
      case 0:
        value = f(x + y);
        break;
      case 1:
        value = f(x - y);
        break;
      case 2:
        value = f(x * y);
        break;
      case 3:
        value = y === 0 ? NaN : f(x / y);
        break;
      case 4:
        value = Math.min(x, y);
        break;
      case 5:
        value = Math.max(x, y);
        break;
      case 6:
        value = x > 0 ? f(x ** y) : x === 0 && y > 0 ? 0 : NaN;
        break;
      case 7:
        value = Math.abs(f(x - y));
        break;
      case 8: {
        const total = f(x + y);
        value = total === 0 ? NaN : f(f(x - y) / total);
        break;
      }
      case 16:
        value = Math.abs(x);
        break;
      case 17:
        value = x < 0 ? NaN : f(Math.sqrt(x));
        break;
      case 18:
        value = x <= 0 ? NaN : f(Math.log(x));
        break;
      case 19:
        value = f(Math.exp(x));
        break;
      case 20:
        value = Math.floor(x);
        break;
      case 21:
        value = Math.ceil(x);
        break;
      case 22:
        value = roundHalfToEven(x);
        break;
      default:
        value = NaN;
    }
    result[cell] =
      isDefined && !Number.isNaN(value) ? Math.min(Math.max(value, clampMin), clampMax) : NaN;
  }
  return result;
}
