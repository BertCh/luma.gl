// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  NATURAL_BREAKS_CELL_COUNT,
  NATURAL_BREAKS_QUANTIZATION_BITS,
  NATURAL_BREAKS_SUBBIN_COUNT,
  NATURAL_BREAKS_MAXIMUM_ROUND_COUNT,
  NATURAL_BREAKS_SETTLED_ZOOM_LEVEL,
  NATURAL_BREAKS_ZOOM_CELLS
} from '../../../src/gpu-dataframe/column-classification/class-breaks-kernels';
import {getOrderedFloat32Key} from '../../../src/gpu-dataframe/column-classification/column-classification-shared';

const fround = Math.fround;

/** A 64-bit unsigned integer as `[low, high]` 32-bit words, as the WGSL `vec2<u32>`. */
export type Wide = readonly [low: number, high: number];

/** Wraps a number to an unsigned 32-bit integer. */
const unsigned32 = (value: number): number => value >>> 0;

/** Port of the WGSL `addWide`. */
export function addWide(left: Wide, right: Wide): Wide {
  const low = unsigned32(left[0] + right[0]);
  const carry = low < left[0] ? 1 : 0;
  return [low, unsigned32(left[1] + right[1] + carry)];
}

/** Port of the WGSL `subtractWide`. */
export function subtractWide(left: Wide, right: Wide): Wide {
  const low = unsigned32(left[0] - right[0]);
  const borrow = left[0] < right[0] ? 1 : 0;
  return [low, unsigned32(left[1] - right[1] - borrow)];
}

/** Port of the WGSL `multiplyWide`: the full 32 x 32 -> 64 bit product from 16-bit limbs. */
export function multiplyWide(left: number, right: number): Wide {
  const leftLow = left & 0xffff;
  const leftHigh = left >>> 16;
  const rightLow = right & 0xffff;
  const rightHigh = right >>> 16;
  const lowProduct = leftLow * rightLow;
  const crossA = leftLow * rightHigh;
  const crossB = leftHigh * rightLow;
  const highProduct = leftHigh * rightHigh;
  const cross = unsigned32(crossA + crossB);
  const crossCarry = cross < crossA ? 1 : 0;
  const low = unsigned32(lowProduct + ((cross & 0xffff) << 16));
  const lowCarry = low < lowProduct ? 1 : 0;
  return [low, unsigned32(highProduct + (cross >>> 16) + (crossCarry << 16) + lowCarry)];
}

/** Port of the WGSL `multiplyWideLow`: the product modulo 2^64. */
export function multiplyWideLow(left: Wide, right: Wide): Wide {
  const product = multiplyWide(left[0], right[0]);
  return [
    product[0],
    unsigned32(product[1] + Math.imul(left[0], right[1]) + Math.imul(left[1], right[0]))
  ];
}

/** Splits a non-negative BigInt below 2^64, or a negative one in two's complement, into words. */
export function toWide(value: bigint): Wide {
  const bits = BigInt.asUintN(64, value);
  return [Number(bits & 0xffffffffn), Number(bits >> 32n)];
}

/** Joins words into an unsigned BigInt. */
export function fromWide(value: Wide): bigint {
  return (BigInt(value[1]) << 32n) | BigInt(value[0]);
}

function wideToSignedFloat(value: Wide): number {
  // Two's complement of a small negative number has a huge low word, so convert the magnitude.
  if ((value[1] | 0) < 0) {
    return -wideToUnsignedFloat(subtractWide([0, 0], value));
  }
  return wideToUnsignedFloat(value);
}

function wideToUnsignedFloat(value: Wide): number {
  return fround(fround(value[1] * 4294967296) + fround(value[0]));
}

/** Rounds half to even, as the WGSL `round`. */
function roundHalfEven(value: number): number {
  const floor = Math.floor(value);
  const difference = value - floor;
  if (difference < 0.5) {
    return floor;
  }
  if (difference > 0.5) {
    return floor + 1;
  }
  return floor % 2 === 0 ? floor : floor + 1;
}

/**
 * Port of the WGSL `getClassCost`: `S2 - S1^2 / count` of integer power sums by recentring on
 * `round(S1 / count)`, in the same integer and f32 operations.
 */
export function getClassCost(count: number, sum: Wide, squares: Wide): number {
  if (count === 0) {
    return 0;
  }
  const countFloat = fround(count);
  const center = roundHalfEven(fround(wideToSignedFloat(sum) / countFloat));
  const centerBits = unsigned32(center);
  const centerWide: Wide = [centerBits, center < 0 ? 0xffffffff : 0];
  const centeredSum = unsigned32(sum[0] - Math.imul(centerBits, count)) | 0;
  const crossTerm = multiplyWideLow(centerWide, sum);
  const centerSquared = multiplyWideLow(centerWide, centerWide);
  const centeredSquares = addWide(
    subtractWide(squares, addWide(crossTerm, crossTerm)),
    multiplyWideLow(centerSquared, [count, 0])
  );
  const centeredSumFloat = fround(centeredSum);
  const correction = fround(fround(centeredSumFloat * centeredSumFloat) / countFloat);
  return Math.max(fround(wideToUnsignedFloat(centeredSquares) - correction), 0);
}

/** Exact `S2 - S1^2 / count` of integer power sums in f64, for tests of the f32 evaluation. */
export function getExactClassCost(count: number, sum: bigint, squares: bigint): number {
  if (count === 0) {
    return 0;
  }
  const numerator = squares * BigInt(count) - sum * sum;
  return Number(numerator) / count;
}

/** Bin of `value` exactly as the GPU kernel computes it. */
export function getBinIndex(
  value: number,
  minimum: number,
  width: number,
  binCount: number
): number {
  const difference = fround(value - minimum);
  let bin = Math.floor(fround(difference / width));
  bin = Math.min(Math.max(bin, 0), binCount - 1);
  if (difference < fround(bin * width)) {
    bin -= 1;
  } else if (difference >= fround((bin + 1) * width)) {
    bin += 1;
  }
  return Math.min(Math.max(bin, 0), binCount - 1);
}

/** Intermediate results of {@link computeNaturalBreaksMirror}, for the specs. */
export type NaturalBreaksMirrorResult = {
  /** Inner edges `e[1..k-1]` after removing empty classes (ascending, strictly). */
  edges: number[];
  /** Produced class count: the requested count, or the number of distinct values if smaller. */
  classCount: number;
  /** Coarse dynamic program class starts (bin index of every class after the first). */
  coarseStarts: number[];
  /** Total cost of the coarse partition, from the coarse f32 cost. */
  coarseCost: number;
  /** Bin histogram. */
  histogram: Uint32Array;
  /** Edges after the coarse stage, before refinement (`min + start * width`). */
  coarseEdges: number[];
  /** Number of row passes the refinement ran (extents plus two per round). */
  passCount: number;
  /** Per-bin sums of the integer positions and their squares (exact), for the specs. */
  binSums: bigint[];
  binSquares: bigint[];
};

type Element = {
  count: number;
  /** Sum of `q`, as a signed integer. */
  sum: bigint;
  /** Sum of `q^2`. */
  squares: bigint;
  minimumKey: number;
  maximumKey: number;
};

type Plan = {
  anchor: number;
  cellWidth: number;
  inverseUnit: number;
  halfWidth: number;
  elements: Element[];
};

const CELL_COUNT = NATURAL_BREAKS_CELL_COUNT;
const SUBBIN_COUNT = NATURAL_BREAKS_SUBBIN_COUNT;
const QUANTIZATION = 2 ** NATURAL_BREAKS_QUANTIZATION_BITS;
const NO_MINIMUM = 0xffffffff;
const NO_MAXIMUM = 0;

function createElement(): Element {
  return {count: 0, sum: 0n, squares: 0n, minimumKey: NO_MINIMUM, maximumKey: NO_MAXIMUM};
}

type CoarseStage = {
  histogram: Uint32Array;
  binSums: bigint[];
  binSquares: bigint[];
  coarseStarts: number[];
  coarseCost: number;
  coarseEdges: number[];
};

/**
 * Coarse stage: histogram of the values in `binCount` equal-width bins of `[minimum, maximum]`,
 * exact prefix sums, and the monotone divide-and-conquer dynamic program over the bins. Values
 * outside the range are skipped (the GPU runs the stage again on the range of one class).
 */
function computeCoarseStage(
  values: ArrayLike<number>,
  minimum: number,
  maximum: number,
  requestedClassCount: number,
  binCount: number
): CoarseStage {
  const classCount = Math.min(requestedClassCount, binCount);
  const width = fround(fround(maximum - minimum) * fround(1 / binCount));
  const histogram = new Uint32Array(binCount);
  const binSums = new Array<bigint>(binCount).fill(0n);
  const binSquares = new Array<bigint>(binCount).fill(0n);
  const inverseWidth = width > 0 ? fround(SUBBIN_COUNT / width) : 0;
  for (let row = 0; row < values.length; row++) {
    const bin = width > 0 ? getBinIndex(values[row], minimum, width, binCount) : 0;
    // Position in 1/SUBBIN_COUNT bins: the bin index plus the fixed-point offset inside the bin.
    let offset = 0;
    if (width > 0) {
      const remainder = fround(fround(values[row] - minimum) - fround(bin * width));
      offset = Math.min(
        Math.max(Math.floor(fround(remainder * inverseWidth)), 0),
        SUBBIN_COUNT - 1
      );
    }
    const position = BigInt(bin * SUBBIN_COUNT + offset);
    histogram[bin]++;
    binSums[bin] += position;
    binSquares[bin] += position * position;
  }

  // Exact prefix sums of count, position and position^2 over the bins (modulo 2^64).
  const prefixCount = new Array<number>(binCount + 1).fill(0);
  const prefixSum = new Array<bigint>(binCount + 1).fill(0n);
  const prefixSquares = new Array<bigint>(binCount + 1).fill(0n);
  for (let bin = 0; bin < binCount; bin++) {
    prefixCount[bin + 1] = prefixCount[bin] + histogram[bin];
    prefixSum[bin + 1] = BigInt.asUintN(64, prefixSum[bin] + binSums[bin]);
    prefixSquares[bin + 1] = BigInt.asUintN(64, prefixSquares[bin] + binSquares[bin]);
  }
  const getRangeCost = (first: number, last: number): number =>
    getClassCost(
      prefixCount[last + 1] - prefixCount[first],
      toWide(prefixSum[last + 1] - prefixSum[first]),
      toWide(prefixSquares[last + 1] - prefixSquares[first])
    );

  // Layer c holds the best split of bins [0, j] into c + 1 classes, in `totals[c][j]`, with the
  // start of the last class in `starts[c][j]` (lowest start on ties).
  const totals: number[][] = [];
  const starts: number[][] = [];
  totals.push(Array.from({length: binCount}, (_, end) => getRangeCost(0, end)));
  starts.push(new Array<number>(binCount).fill(0));
  for (let layer = 1; layer < classCount; layer++) {
    const layerTotals = new Array<number>(binCount).fill(NaN);
    const layerStarts = new Array<number>(binCount).fill(0);
    const count = binCount - layer;
    let levels = 0;
    while (2 ** levels - 1 < count) {
      levels++;
    }
    for (let level = 0; level < levels; level++) {
      const stride = 2 ** (levels - 1 - level);
      for (let node = 0; node < 2 ** level; node++) {
        const position = (2 * node + 1) * stride;
        if (position > count) {
          continue;
        }
        const end = layer + position - 1;
        const lowBound = position === stride ? layer : Math.max(layerStarts[end - stride], layer);
        const highBound = position + stride > count ? binCount - 1 : layerStarts[end + stride];
        const low = lowBound;
        const high = Math.min(end, Math.max(highBound, low));
        let bestCost = 3.0e38;
        let bestStart = 0xffffffff;
        for (let start = low; start <= high; start++) {
          const cost = fround(totals[layer - 1][start - 1] + getRangeCost(start, end));
          if (cost < bestCost) {
            bestCost = cost;
            bestStart = start;
          }
        }
        layerTotals[end] = bestCost;
        layerStarts[end] = bestStart;
      }
    }
    totals.push(layerTotals);
    starts.push(layerStarts);
  }
  const coarseStarts: number[] = [];
  let end = binCount - 1;
  for (let layer = classCount - 1; layer > 0; layer--) {
    const start = starts[layer][end];
    coarseStarts.unshift(start);
    end = start - 1;
  }
  const coarseCost = totals[classCount - 1][binCount - 1];
  const coarseEdges = coarseStarts.map(start => fround(minimum + fround(start * width)));
  return {histogram, binSums, binSquares, coarseStarts, coarseCost, coarseEdges};
}

/**
 * CPU mirror of the GPU natural-breaks algorithm, in the same integer and f32 operations: coarse
 * histogram, exact-integer prefix sums, monotone divide-and-conquer Fisher-Jenks over the bins
 * (the level-synchronous tree the GPU kernel walks), the class extents of the coarse partition,
 * then up to `NATURAL_BREAKS_MAXIMUM_ROUND_COUNT` rounds of red-black coordinate descent of every
 * break on the data (stopping early once settled), and the removal of empty classes. See `natural-breaks-nodes.ts` for the design.
 *
 * `values` are the finite (unmasked) values; `minimum` and `maximum` their extremes.
 */
export function computeNaturalBreaksMirror(
  values: ArrayLike<number>,
  minimum: number,
  maximum: number,
  requestedClassCount: number,
  binCount: number,
  options: {maximumRoundCount?: number; trace?: string[]} = {}
): NaturalBreaksMirrorResult {
  const maximumRoundCount = options.maximumRoundCount ?? NATURAL_BREAKS_MAXIMUM_ROUND_COUNT;
  const classCount = Math.min(requestedClassCount, binCount);
  const {histogram, binSums, binSquares, coarseStarts, coarseCost, coarseEdges} =
    computeCoarseStage(values, minimum, maximum, classCount, binCount);

  // Refinement. edges[j], j = 1..classCount - 1, are the cuts (index 0 unused); a class i holds the
  // values with i cuts at or below them.
  const edges = [minimum, ...coarseEdges];
  const valid = Array.from(values).filter(Number.isFinite);
  const keys = valid.map(getOrderedFloat32Key);
  const classMinimum = new Array<number>(classCount).fill(NO_MINIMUM);
  const classMaximum = new Array<number>(classCount).fill(NO_MAXIMUM);
  const halfWidths = new Array<number>(classCount).fill(0);
  const zoomLevels = new Array<number>(classCount).fill(0);
  // How far each cut moved at its last update; a break keeps a window wide enough to follow the
  // neighbours that moved since its own last update. Infinity until the first update.
  const movements = new Array<number>(classCount + 1).fill(Infinity);
  movements[0] = 0;
  movements[classCount] = 0;
  let passCount = 0;
  let movedInRound = false;
  const getRowClass = (key: number, cuts: number[]): number => {
    let low = 1;
    let high = classCount;
    while (low < high) {
      const middle = (low + high) >> 1;
      if (cuts[middle] <= key) {
        low = middle + 1;
      } else {
        high = middle;
      }
    }
    return low - 1;
  };
  if (classCount >= 2 && minimum < maximum) {
    // Class extents of the coarse partition. A cut beside an empty class means the bins could
    // not tell structure inside the most populated class apart (an extreme outlier collapses the
    // data into a few bins): that class is split by the same coarse stage over its own range,
    // with the classes the empty ones would have held.
    const classCounts = new Array<number>(classCount).fill(0);
    const measureExtents = () => {
      classMinimum.fill(NO_MINIMUM);
      classMaximum.fill(NO_MAXIMUM);
      classCounts.fill(0);
      const cuts = edges.map(edge => getOrderedFloat32Key(edge));
      for (const key of keys) {
        const rowClass = getRowClass(key, cuts);
        classMinimum[rowClass] = Math.min(classMinimum[rowClass], key);
        classMaximum[rowClass] = Math.max(classMaximum[rowClass], key);
        classCounts[rowClass]++;
      }
      passCount++;
    };
    measureExtents();
    const emptyCount = classMinimum.filter(key => key === NO_MINIMUM).length;
    if (emptyCount > 0) {
      let bulk = 0;
      for (let index = 1; index < classCount; index++) {
        if (classCounts[index] > classCounts[bulk]) {
          bulk = index;
        }
      }
      const bulkLow = decodeKey(classMinimum[bulk]);
      const bulkHigh = decodeKey(classMaximum[bulk]);
      if (bulkLow < bulkHigh) {
        const inner = computeCoarseStage(
          valid.filter(value => value >= bulkLow && value <= bulkHigh),
          bulkLow,
          bulkHigh,
          1 + emptyCount,
          binCount
        );
        const starts: number[] = [];
        for (let index = 0; index < classCount; index++) {
          if (classMinimum[index] === NO_MINIMUM) {
            continue;
          }
          starts.push(decodeKey(classMinimum[index]));
          if (index === bulk) {
            starts.push(...inner.coarseEdges);
          }
        }
        edges.splice(0, edges.length, minimum, ...starts.slice(1));
        measureExtents();
      }
    }
    // Every non-empty class starts at its smallest value, so cuts are data values.
    for (let index = 1; index < classCount; index++) {
      if (classMinimum[index] !== NO_MINIMUM) {
        edges[index] = decodeKey(classMinimum[index]);
      }
    }

    for (let half = 0; half < 2 * maximumRoundCount; half++) {
      if (half % 2 === 0 && half > 0) {
        // A round that moved no cut, with every break zoomed to the final resolution, is settled.
        const settled =
          !movedInRound &&
          zoomLevels.every(
            (level, index) => index === 0 || level >= NATURAL_BREAKS_SETTLED_ZOOM_LEVEL
          );
        if (settled) {
          break;
        }
        movedInRound = false;
      }
      const wanted = 1 - (half & 1);
      const rowCuts = edges.map(edge => getOrderedFloat32Key(edge));
      const plans = new Map<number, Plan>();
      for (let breakIndex = wanted === 1 ? 1 : 2; breakIndex < classCount; breakIndex += 2) {
        const lowKey = Math.min(classMinimum[breakIndex - 1], classMinimum[breakIndex]);
        const highKey = Math.max(classMaximum[breakIndex - 1], classMaximum[breakIndex]);
        if (lowKey === NO_MINIMUM) {
          zoomLevels[breakIndex] = 99;
          continue;
        }
        // The window is anchored at the cut, clamped into the data extent of the two classes: a
        // cut beside an empty class sits outside the data and would waste the window.
        const lowValue = decodeKey(lowKey);
        const highValue = decodeKey(highKey);
        const anchor = Math.min(Math.max(edges[breakIndex], lowValue), highValue);
        const span = Math.max(fround(anchor - lowValue), fround(highValue - anchor));
        if (!(span > 0)) {
          zoomLevels[breakIndex] = 99;
          continue;
        }
        const wanted_ =
          halfWidths[breakIndex] > 0
            ? Math.max(
                halfWidths[breakIndex],
                fround(2 * Math.max(movements[breakIndex - 1], movements[breakIndex + 1]))
              )
            : span;
        const halfWidth = Math.min(wanted_, span);
        plans.set(breakIndex, {
          anchor,
          halfWidth,
          cellWidth: fround(halfWidth * (2 / CELL_COUNT)),
          inverseUnit: fround(QUANTIZATION / span),
          elements: Array.from({length: CELL_COUNT + 2}, createElement)
        });
      }
      for (let row = 0; row < valid.length; row++) {
        const rowClass = getRowClass(keys[row], rowCuts);
        let breakIndex = 0;
        if (rowClass >= 1 && (rowClass & 1) === wanted) {
          breakIndex = rowClass;
        } else if (rowClass + 1 < classCount && ((rowClass + 1) & 1) === wanted) {
          breakIndex = rowClass + 1;
        }
        const plan = plans.get(breakIndex);
        if (!plan) {
          continue;
        }
        const offset = fround(valid[row] - plan.anchor);
        let cell =
          Math.floor(
            Math.min(Math.max(fround(offset / plan.cellWidth), -2 * CELL_COUNT), 2 * CELL_COUNT)
          ) +
          CELL_COUNT / 2;
        if (rowClass === breakIndex) {
          cell = Math.max(cell, CELL_COUNT / 2);
        } else {
          cell = Math.min(cell, CELL_COUNT / 2 - 1);
        }
        const slot = cell < 0 ? 0 : cell >= CELL_COUNT ? CELL_COUNT + 1 : cell + 1;
        const quantized = roundHalfEven(
          Math.min(Math.max(fround(offset * plan.inverseUnit), -QUANTIZATION), QUANTIZATION)
        );
        const element = plan.elements[slot];
        element.count++;
        element.sum += BigInt(quantized);
        element.squares += BigInt(quantized * quantized);
        element.minimumKey = Math.min(element.minimumKey, keys[row]);
        element.maximumKey = Math.max(element.maximumKey, keys[row]);
      }
      passCount++;
      for (const [breakIndex, plan] of plans) {
        solveBreak(breakIndex, plan);
      }
      options.trace?.push(
        `half ${half} edges ${edges.slice(1).join(' ')} | classMin ${classMinimum.map(key => (key === NO_MINIMUM ? '-' : decodeKey(key))).join(' ')} | halfWidths ${halfWidths.slice(1).join(' ')}`
      );
    }
  }

  // Remove empty classes: a cut is kept when its class holds a value. Without refinement (one
  // class, or a constant column) there are no cuts.
  const finalEdges: number[] = [];
  for (let breakIndex = 1; breakIndex < classCount; breakIndex++) {
    if (classMinimum[breakIndex] !== NO_MINIMUM) {
      finalEdges.push(edges[breakIndex]);
    }
  }
  return {
    edges: finalEdges,
    classCount: finalEdges.length + 1,
    coarseStarts,
    coarseCost,
    histogram,
    coarseEdges,
    passCount,
    binSums,
    binSquares
  };

  /** Chooses the best cell boundary of one break and updates its cut, extents and window. */
  function solveBreak(breakIndex: number, plan: Plan): void {
    const {elements} = plan;
    const current = CELL_COUNT / 2;
    let totalCount = 0;
    let totalSum = 0n;
    let totalSquares = 0n;
    for (const element of elements) {
      totalCount += element.count;
      totalSum += element.sum;
      totalSquares += element.squares;
    }
    const costOf = (count: number, sum: bigint, squares: bigint) =>
      getClassCost(count, toWide(sum), toWide(squares));
    let leftCount = elements[0].count;
    let leftSum = elements[0].sum;
    let leftSquares = elements[0].squares;
    let bestCost = Infinity;
    let bestBoundary = current;
    let currentCost = Infinity;
    let currentLeftCount = 0;
    for (let boundary = 0; boundary <= CELL_COUNT; boundary++) {
      if (boundary > 0) {
        const element = elements[boundary];
        leftCount += element.count;
        leftSum += element.sum;
        leftSquares += element.squares;
      }
      const cost = fround(
        costOf(leftCount, leftSum, leftSquares) +
          costOf(totalCount - leftCount, totalSum - leftSum, totalSquares - leftSquares)
      );
      if (boundary === current) {
        currentCost = cost;
        currentLeftCount = leftCount;
      }
      if (cost < bestCost) {
        bestCost = cost;
        bestBoundary = boundary;
      }
    }
    if (currentCost <= bestCost) {
      bestBoundary = current;
    }
    // Elements 0..bestBoundary form the left set, the rest the right set.
    let leftMinimum = NO_MINIMUM;
    let leftMaximum = NO_MAXIMUM;
    let rightMinimum = NO_MINIMUM;
    let rightMaximum = NO_MAXIMUM;
    elements.forEach((element, index) => {
      if (element.count === 0) {
        return;
      }
      if (index <= bestBoundary) {
        leftMinimum = Math.min(leftMinimum, element.minimumKey);
        leftMaximum = Math.max(leftMaximum, element.maximumKey);
      } else {
        rightMinimum = Math.min(rightMinimum, element.minimumKey);
        rightMaximum = Math.max(rightMaximum, element.maximumKey);
      }
    });
    classMinimum[breakIndex - 1] = leftMinimum;
    classMaximum[breakIndex - 1] = leftMaximum;
    classMinimum[breakIndex] = rightMinimum;
    classMaximum[breakIndex] = rightMaximum;
    if (rightMinimum !== NO_MINIMUM) {
      const cut = decodeKey(rightMinimum);
      movements[breakIndex] = Math.abs(fround(cut - edges[breakIndex]));
      if (movements[breakIndex] > 0) {
        movedInRound = true;
      }
      edges[breakIndex] = cut;
    } else {
      movements[breakIndex] = 0;
    }
    // Zoom around the chosen boundary unless a class beside the break is empty (the window must
    // then cover the extent again), or the optimum sits at the window edge (it may continue).
    const degenerate = currentLeftCount === 0 || totalCount - currentLeftCount === 0;
    const atEdge = Math.abs(bestBoundary - current) >= CELL_COUNT / 2 - NATURAL_BREAKS_ZOOM_CELLS;
    halfWidths[breakIndex] = degenerate
      ? 0
      : atEdge
        ? plan.halfWidth
        : fround(plan.halfWidth * ((2 * NATURAL_BREAKS_ZOOM_CELLS) / CELL_COUNT));
    zoomLevels[breakIndex] = degenerate
      ? 0
      : atEdge
        ? zoomLevels[breakIndex]
        : zoomLevels[breakIndex] + 1;
  }
}

const scratchFloat = new Float32Array(1);
const scratchBits = new Uint32Array(scratchFloat.buffer);

function decodeKey(key: number): number {
  scratchBits[0] = ((key & 0x80000000) !== 0 ? key ^ 0x80000000 : ~key) >>> 0;
  return scratchFloat[0];
}
