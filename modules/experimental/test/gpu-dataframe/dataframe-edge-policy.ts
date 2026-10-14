// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Edge-policy tables (quality roadmap rule 22) of the dataframe statistics changed in the
 * 2026-10-14 round: one table per statistic, covering NaN, +-Infinity, -0, out-of-range parameters
 * and empty input. Every row states the documented behavior and the expected value derived from it
 * by hand (or from numpy where named), so the CPU oracles (node specs) and the GPU kernels (GPU
 * specs) are checked against the same rows instead of against each other.
 */

/** One order-statistic row, shared by `GPUColumnQuantiles`, `GPUGroupStatistics` and `GPUClassBreaks`. */
export type OrderStatisticEdgePolicyRow = {
  name: string;
  values: number[];
  /** Linear (R-7) probability. */
  probability: number;
  /** Expected quantile; NaN for "no value". */
  expected: number;
};

/**
 * Order statistics (B-14): the sample is the finite unmasked values, `-0` equals `+0`, and an
 * invalid probability or an empty sample gives NaN. Expected values are
 * `numpy.quantile(x[numpy.isfinite(x)], p)` where numpy defines them.
 */
export const ORDER_STATISTIC_EDGE_POLICY: readonly OrderStatisticEdgePolicyRow[] = [
  {name: 'NaN is missing', values: [1, NaN, 3], probability: 0.5, expected: 2},
  // numpy keeps infinities (it would give 2); rule 8 makes them missing in interpolating kernels.
  {name: '+Infinity is missing', values: [1, 2, Infinity], probability: 0.5, expected: 1.5},
  {name: '-Infinity is missing', values: [-Infinity, 1, 2], probability: 0.5, expected: 1.5},
  {name: '-0 equals +0', values: [-0, 0, -0], probability: 0, expected: 0},
  {name: 'p below 0', values: [1, 2, 3], probability: -0.25, expected: NaN},
  {name: 'p above 1', values: [1, 2, 3], probability: 1.5, expected: NaN},
  {name: 'p NaN', values: [1, 2, 3], probability: NaN, expected: NaN},
  {name: 'empty sample', values: [NaN, Infinity, -Infinity], probability: 0.5, expected: NaN},
  {name: 'p = 1 is the maximum', values: [4, 1, 9, 2], probability: 1, expected: 9},
  // h = 0.5 between -3e38 and 3e38: the difference overflows f32, the convex form does not.
  {name: 'difference overflow', values: [-3e38, 3e38], probability: 0.5, expected: 0}
];

/** One class-breaks row. */
export type ClassBreaksEdgePolicyRow = {
  name: string;
  values: number[];
  method: 'equal-interval' | 'quantile' | 'standard-deviation' | 'custom';
  /** Raw per-frame class count written to the parameter view (out-of-range values allowed). */
  classCount: number;
  customEdges?: number[];
  /** Raw `standardDeviationInterval` (slot 2), written past the packer, which refuses NaN. */
  standardDeviationInterval?: number;
  expectedClassCount: number;
  /** Expected `e[0..expectedClassCount]`; NaN entries are "NaN". */
  expectedBreaks: number[];
};

/**
 * `GPUClassBreaks`: NaN rows are skipped, infinities are missing for every edge (so `e[0]` and
 * `e[k]` are the finite extremes), no finite value gives 0 classes and NaN edges, all-equal finite
 * values give one class, a per-frame class count is clamped to `[1, maximumClassCount]` on the
 * GPU (the CPU packer refuses it), and a NaN edge from a hand-written parameter view makes the
 * classification undefined (0 classes, NaN edges: every row unclassified, never a silent wrong
 * class). Rows run with `maximumClassCount` 4.
 */
export const CLASS_BREAKS_EDGE_POLICY: readonly ClassBreaksEdgePolicyRow[] = [
  {
    name: 'infinities are missing for edges',
    values: [-Infinity, 0, 10, 20, Infinity, NaN],
    method: 'equal-interval',
    classCount: 2,
    expectedClassCount: 2,
    expectedBreaks: [0, 10, 20]
  },
  {
    name: 'quantile ignores infinities',
    values: [Infinity, 1, 2, 3, 4, -Infinity],
    method: 'quantile',
    classCount: 2,
    expectedClassCount: 2,
    expectedBreaks: [1, 2.5, 4]
  },
  {
    name: 'no finite value',
    values: [NaN, Infinity, -Infinity],
    method: 'quantile',
    classCount: 3,
    expectedClassCount: 0,
    expectedBreaks: [NaN, NaN, NaN, NaN, NaN]
  },
  {
    name: 'all equal is one class',
    values: [5, 5, 5, 5],
    method: 'equal-interval',
    classCount: 3,
    expectedClassCount: 1,
    expectedBreaks: [5, 5]
  },
  {
    name: 'class count above the maximum clamps on the GPU',
    values: [0, 1, 2, 3, 4, 5, 6, 7, 8],
    method: 'equal-interval',
    classCount: 9,
    expectedClassCount: 4,
    expectedBreaks: [0, 2, 4, 6, 8]
  },
  {
    name: 'NaN custom edge is no classification',
    values: [0, 4, 8],
    method: 'custom',
    classCount: 2,
    customEdges: [0, NaN, 8],
    expectedClassCount: 0,
    expectedBreaks: [NaN, NaN, NaN, NaN, NaN]
  },
  {
    name: 'NaN standard deviation interval is no classification',
    values: [0, 4, 8],
    method: 'standard-deviation',
    classCount: 3,
    standardDeviationInterval: NaN,
    expectedClassCount: 0,
    expectedBreaks: [NaN, NaN, NaN, NaN, NaN]
  },
  {
    name: 'class count below 1 is one class',
    values: [0, 4, 8],
    method: 'equal-interval',
    classCount: 0,
    expectedClassCount: 1,
    expectedBreaks: [0, 8]
  }
];

/** One variogram row: a tiny scene, its frame and the expected bins (lag-major, 2 lags). */
export type VariogramEdgePolicyRow = {
  name: string;
  positions: number[];
  values: number[];
  mask?: number[];
  /** Raw parameter values `[minX, minY, maxX, maxY, maximumDistance]` (invalid ones allowed). */
  parameters: [number, number, number, number, number];
  expectedPairCounts: [number, number];
  /** NaN for an empty bin. */
  expectedSemivariances: [number, number];
};

/**
 * `GPUVariogram`: a row is included when its mask is nonzero, its value is finite and its finite
 * position lies inside the inclusive bounds; `-0` and `+0` values are equal; a non-positive maximum
 * distance excludes every row; empty bins are NaN with a 0 count. Every row uses 2 lags.
 */
export const VARIOGRAM_EDGE_POLICY: readonly VariogramEdgePolicyRow[] = [
  {
    // Pair (0, 1) at distance 1 falls in lag floor(1 / 3 * 2) = 0, gamma = (2 - 0)^2 / 2.
    name: 'NaN and +-Infinity values are missing',
    positions: [0, 0, 1, 0, 0, 1, 1, 1, 2, 2],
    values: [0, 2, NaN, Infinity, -Infinity],
    parameters: [0, 0, 2, 2, 3],
    expectedPairCounts: [1, 0],
    expectedSemivariances: [2, NaN]
  },
  {
    name: '-0 equals +0',
    positions: [0, 0, 1, 0],
    values: [-0, 0],
    parameters: [0, 0, 2, 2, 3],
    expectedPairCounts: [1, 0],
    expectedSemivariances: [0, NaN]
  },
  {
    // Row 2 has a NaN coordinate, row 3 lies outside the bounds; row 1 at distance 2 is in lag 1.
    name: 'non-finite or outside positions are excluded',
    positions: [0, 0, 2, 0, NaN, 0, 9, 9],
    values: [1, 3, 5, 7],
    parameters: [0, 0, 2, 2, 3],
    expectedPairCounts: [0, 1],
    expectedSemivariances: [NaN, 2]
  },
  {
    name: 'masked rows are excluded (empty input)',
    positions: [0, 0, 1, 0],
    values: [1, 3],
    mask: [0, 0],
    parameters: [0, 0, 2, 2, 3],
    expectedPairCounts: [0, 0],
    expectedSemivariances: [NaN, NaN]
  },
  {
    name: 'a non-positive maximum distance excludes every row',
    positions: [0, 0, 1, 0],
    values: [1, 3],
    parameters: [0, 0, 2, 2, 0],
    expectedPairCounts: [0, 0],
    expectedSemivariances: [NaN, NaN]
  }
];

/** One time-window row: interval rows, a window and the expected mask. */
export type TimeWindowEdgePolicyRow = {
  name: string;
  starts: number[];
  ends: number[];
  window: {start: number; end: number};
  expectedMask: number[];
};

/**
 * `GPUTimeWindowFilter`: rows overlap the inclusive window `[start, end]`; a reversed window
 * (`end < start`) accepts nothing, interval rows included (B-13); NaN timestamps are never accepted.
 */
export const TIME_WINDOW_EDGE_POLICY: readonly TimeWindowEdgePolicyRow[] = [
  {
    name: 'reversed window rejects a spanning interval',
    starts: [5, 12],
    ends: [25, 18],
    window: {start: 20, end: 10},
    expectedMask: [0, 0]
  },
  {
    name: 'edges are inclusive',
    starts: [0, 20],
    ends: [10, 30],
    window: {start: 10, end: 20},
    expectedMask: [1, 1]
  },
  {
    name: 'NaN timestamps are never accepted',
    starts: [NaN, 12, 12],
    ends: [15, NaN, 15],
    window: {start: 10, end: 20},
    expectedMask: [0, 0, 1]
  }
];
