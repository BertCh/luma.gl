// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  validatePackedUint32View,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {assert} from '@luma.gl/core';
import {
  createFillNode,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import type {
  GPUInferenceAlternative,
  GPUInferenceMetadata,
  GPUMultipleTestingMethod
} from '../../analysis-core/contracts/index';
import {
  getGPUSpatialWeightsNeighborLoopWGSL,
  resolveGPUIslandPolicy,
  type GPUIslandPolicy,
  type GPUSpatialWeights
} from '../spatial-weights/spatial-weights';
import {
  GPUMultipleTestingAdjustment,
  resolveGPUMultipleTestingMethod
} from '../multiple-testing/index';
import {
  BLOCK_ROWS,
  getColumnSumNodes,
  getPermutationInputNodes,
  getPermutationParameterWGSL,
  PERMUTATION_FLOAT_WGSL,
  PERMUTATION_INVALID_POSITION,
  PERMUTATION_TOTAL,
  validatePermutationInputs
} from './permutation-inference-kernels';
import {PERMUTATION_RANDOM_WGSL} from './permutation-random';
import {
  getAlternativeMultiplier,
  getExceedanceCountWGSL,
  resolvePermutationAlternative,
  resolveStopAfterExceedances
} from './permutation-alternative';
import {
  getGPUPermutationParameterValues,
  type GPUPermutationSettings
} from './permutation-parameters';
import {PHILOX_PURPOSES} from '../../utils/philox-random';
import type {GPUFloat32OrUint32Values} from '../spatial-weights/float32-values';

const OPERATION = 'GPULocalPermutationTest';

/** Philox counter word `c` that tags local conditional-permutation draws. @internal */
export const LOCAL_PERMUTATION_RANDOM_TAG = PHILOX_PURPOSES.localPermutationTest.id;

/** `exceedances` value of a row that was not tested. */
export const GPU_LOCAL_PERMUTATION_NOT_TESTED = 0xffffffff;

/** Largest supported compile-time `maximumNeighbors`. */
export const GPU_LOCAL_PERMUTATION_MAXIMUM_NEIGHBORS = 64;

/**
 * A local statistic tested by {@link GPULocalPermutationTest}.
 *
 * - `'localMoran'`, `'localGeary'`, `'localG'`, `'localGStar'`: univariate (`values`).
 * - `'localBivariateMoran'` (esda `Moran_Local_BV`): `I_i = (n-1) c^x_i sum_j w_ij c^y_j / sqrt(SS_x SS_y)`;
 *   `secondValues` is the permuted variable.
 * - `'localLeeL'` (esda `Spatial_Pearson_Local`): `L_i = n (sum_j w_ij c^x_j)(sum_j w_ij c^y_j) /
 *   sqrt(SS_x SS_y)`; both variables are drawn together (a drawn row supplies its `x` and `y`).
 * - `'localJoinCount'` (esda `Join_Counts_Local`): `LJC_i = x_i sum_j x_j` over binary `values`.
 * - `'localBivariateJoinCount'` (esda `Join_Counts_Local_BV`): the count selected by `joinCase`.
 */
export type GPULocalPermutationStatistic =
  | 'localMoran'
  | 'localGeary'
  | 'localG'
  | 'localGStar'
  | 'localBivariateMoran'
  | 'localLeeL'
  | 'localJoinCount'
  | 'localBivariateJoinCount';

/**
 * Bivariate local join count case (Anselin and Li 2019; esda `Join_Counts_Local_BV`):
 * `'BJC'` (no colocation) counts neighbors `j` with `x_j = 0, y_j = 1` for a focus with `x_i = 1,
 * y_i = 0`; `'CLC'` (co-location) counts neighbors with `x_j = y_j = 1` for a focus with `x_i =
 * y_i = 1`.
 */
export type GPULocalJoinCountCase = 'BJC' | 'CLC';

/**
 * Values of the optional `classes` output for `'localGeary'` (Anselin 2019, esda `Geary_Local`
 * `labs`, with `0` for esda's NaN).
 */
export const GPU_LOCAL_GEARY_CLASS = {
  /** Not tested, not an analysis row, or `c_i`/`x_i` exactly at the mean (esda NaN). */
  undefined: 0,
  /** Positive association (`c_i` below its mean), significant, `x_i` above the mean (esda 1). */
  highHigh: 1,
  /** Positive association, significant, `x_i` below the mean (esda 2). */
  lowLow: 2,
  /** Negative association: significant with `c_i` above its mean (esda 3, "other"). */
  negativeAssociation: 3,
  /** Not significant (esda 4). */
  notSignificant: 4
} as const;

/** Per-statistic WGSL: the per-draw terms, the row term and the data it needs. @internal */
type LocalStatisticSpec = {
  centerX: boolean;
  centerY: boolean;
  /** Needs `secondValues`. */
  bivariate: boolean;
  /** Counts joins: binary weights, rows with a zero count are not tested. */
  joinCount: boolean;
  /** WGSL of the first lag's per-draw term in `fx, fy, dx, dy` (focus and drawn x, y). */
  drawnA: string;
  /** WGSL of the second lag's per-draw term (Lee's L only). */
  drawnB?: string;
  /** WGSL of the row term in `fx, fy, lagA, lagB`; the statistic up to a positive per-row constant. */
  term: string;
};

/** Resolves the WGSL pieces of a statistic. @internal */
export function getLocalStatisticSpec(
  statistic: GPULocalPermutationStatistic,
  joinCase?: GPULocalJoinCountCase
): LocalStatisticSpec {
  const univariate = {bivariate: false, joinCount: false, centerY: true, centerX: false};
  switch (statistic) {
    case 'localMoran':
      return {...univariate, centerX: true, drawnA: 'dx', term: 'fx * lagA'};
    case 'localGeary':
      return {...univariate, centerX: true, drawnA: '(fx - dx) * (fx - dx)', term: 'lagA'};
    case 'localG':
      return {...univariate, drawnA: 'dx', term: 'lagA'};
    case 'localGStar':
      return {...univariate, drawnA: 'dx', term: 'lagA + fx'};
    case 'localBivariateMoran':
      return {
        ...univariate,
        centerX: true,
        bivariate: true,
        drawnA: 'dy',
        term: 'fx * lagA'
      };
    case 'localLeeL':
      return {
        ...univariate,
        centerX: true,
        bivariate: true,
        drawnA: 'dx',
        drawnB: 'dy',
        term: 'lagA * lagB'
      };
    case 'localJoinCount':
      return {
        ...univariate,
        joinCount: true,
        drawnA: 'isOne(dx)',
        term: 'isOne(fx) * lagA'
      };
    case 'localBivariateJoinCount':
      return joinCase === 'BJC'
        ? {
            ...univariate,
            joinCount: true,
            bivariate: true,
            centerY: false,
            drawnA: '(1.0 - isOne(dx)) * isOne(dy)',
            term: 'isOne(fx) * (1.0 - isOne(fy)) * lagA'
          }
        : {
            ...univariate,
            joinCount: true,
            bivariate: true,
            centerY: false,
            drawnA: 'isOne(dx) * isOne(dy)',
            term: 'isOne(fx) * isOne(fy) * lagA'
          };
    default:
      // The constructor rejects unknown statistics before any spec is resolved.
      assert(false);
  }
}

/** Caller-owned outputs of {@link GPULocalPermutationTest}. */
export type GPULocalPermutationTestOutput = {
  /**
   * Caller-owned exceedance count `M` per row (numerator of the pseudo p-value; see
   * {@link GPUInferenceAlternative}), or `GPU_LOCAL_PERMUTATION_NOT_TESTED`.
   */
  exceedances: GraphDataView<'uint32'>;
  /** Caller-owned pseudo p-value `min(m (exceedances + 1), P + 1) / (P + 1)` per row (`m` is 2 for `'two-sided'`, else 1), quiet NaN if not tested. These are the unadjusted p-values; `adjustedPValues` holds the multiple-testing adjusted ones. */
  pseudoPValues: GraphDataView<'float32'>;
  /** Optional caller-owned observed local statistic per row (esda scaling), NaN if not tested. */
  observed?: GraphDataView<'float32'>;
  /** Optional caller-owned mask: 1 when the pseudo p-value is significant, else 0. */
  significant?: GraphDataView<'uint32'>;
  /**
   * `'localGeary'` only: optional caller-owned `GPU_LOCAL_GEARY_CLASS` code per row. A row is
   * classified by whether its local Geary value lies below (positive association) or above
   * (negative association) the mean of the local Geary values over the analysis rows, whether it
   * is significant (`p <= level`, or the adjusted p-value for `multipleTesting`) and, for positive
   * association, whether `x_i` is above (high-high) or below (low-low) the mean (esda
   * `Geary_Local(labels=True)`). The classification uses the p-values of the chosen
   * `alternative`: with `'two-sided'` (default) clusters and negative association can both be
   * significant, with `'lesser'` (the tail of clustering) negative association is never
   * significant; `'greater'` finds only negative association.
   */
  classes?: GraphDataView<'uint32'>;
  /**
   * `'localBivariateMoran'` only: optional caller-owned quadrant per row (esda `Moran_Local_BV.q`,
   * PySAL scheme): 1 high-high, 2 low-high, 3 low-low, 4 high-low in (`x_i`, spatial lag of `y`),
   * 0 for rows that are not analyzed. The quadrant is not gated by significance.
   */
  quadrants?: GraphDataView<'uint32'>;
  /**
   * Optional caller-owned adjusted p-value per row for `multipleTesting` (`'none'` copies the
   * pseudo p-value), NaN if not tested. Must not alias `pseudoPValues`.
   */
  adjustedPValues?: GraphDataView<'float32'>;
  /**
   * Optional caller-owned permutation count `L` per row (uint32): the permutations the row used.
   * `P` for a row that did not stop, the stopping index for a row that did (see
   * `stopAfterExceedances`), `GPU_LOCAL_PERMUTATION_NOT_TESTED` for a row that was not tested.
   * Requires `stopAfterExceedances`.
   */
  permutationCounts?: GraphDataView<'uint32'>;
  /** Optional caller-owned one-row uint32 output: the number of islands among the included rows. */
  islandCount?: GraphDataView<'uint32'>;
  /** Completeness status. */
  status: {
    /**
     * Caller-owned one-row flag, 1 when the result is incomplete or invalid: an included row has
     * more than `maximumNeighbors` neighbors (that row is not tested), the weights report
     * `status.overflow`, or the per-frame permutation count is outside `[1, maximumPermutations]`
     * (no row is tested and all p-values are NaN; the count is never clamped).
     */
    overflow: GraphDataView<'uint32'>;
  };
};

/**
 * Properties for {@link GPULocalPermutationTest}.
 *
 * Per-frame (no rebuild or recompile): the contents of `weights`, `values`, `mask` and
 * `parameters` (seed, permutation count, significance level). Compile-time: the row count, slot
 * capacity, `statistic`, `maximumPermutations`, `maximumNeighbors`, `alternative`, `islands`,
 * `multipleTesting` and which optional views exist.
 */
export type GPULocalPermutationTestProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'local-permutation-test'`. */
  id?: string;
  /** Square spatial weights; each row's slot order assigns the drawn values to weights. */
  weights: GPUSpatialWeights;
  /**
   * Analysis values, one per row, `float32` or `uint32` (converted to f32 by one node, exact below
   * 2^24). Rows with a non-finite value are excluded.
   */
  values: GPUFloat32OrUint32Values;
  /**
   * Second variable `y`, one per row. Required by `'localBivariateMoran'`, `'localLeeL'` and
   * `'localBivariateJoinCount'` (rows with a non-finite `y` are excluded), rejected otherwise. It is
   * the permuted variable of the bivariate Moran and join counts; Lee's L draws `x` and `y`
   * together. `float32` or `uint32`.
   */
  secondValues?: GPUFloat32OrUint32Values;
  /** Optional row selection: nonzero includes the row. */
  mask?: GraphDataView<'uint32'>;
  /** Local statistic. Compile-time. */
  statistic: GPULocalPermutationStatistic;
  /** Join-count case of `'localBivariateJoinCount'`: `'BJC'` or `'CLC'`. Compile-time; required for it. */
  joinCase?: GPULocalJoinCountCase;
  /**
   * Tail of the pseudo p-value (see {@link GPUInferenceAlternative}). Compile-time. Defaults to
   * `'two-sided'`. `'directed'` is esda's legacy default and is anti-conservative (about twice
   * too small under the null); it is kept for parity with older esda output only.
   */
  alternative?: GPUInferenceAlternative;
  /**
   * Treatment of islands (included rows with no positive-weight included neighbor). Compile-time.
   * `'exclude'` (default) removes them from the analysis: NaN outputs, not counted in `n`, the
   * mean or the variance, and not neighbors of other rows. There is no cascade: a row whose only
   * neighbors were islands stays in `n` but has no neighbor and is not tested. `'keep'` (esda)
   * keeps islands in `n` and the moments; their statistic is the formula value with a zero lag
   * (`observed`), but they are never tested (`exceedances` `GPU_LOCAL_PERMUTATION_NOT_TESTED`,
   * p-value NaN).
   */
  islands?: GPUIslandPolicy;
  /**
   * Per-frame uint32 parameters of at least `GPU_PERMUTATION_PARAMETER_LENGTH` elements written with
   * `getGPUPermutationParameterValues`.
   */
  parameters: GraphDataView<'uint32'>;
  /** Upper bound of the per-frame permutation count, at most 2^20. Compile-time. */
  maximumPermutations: number;
  /**
   * Opt-in Besag-Clifford sequential stopping (Besag and Clifford 1991, Biometrika 78(2)): an
   * integer `h` in `[1, maximumPermutations]`. Each row consumes its permutations in index order
   * and stops at the first `L` whose exceedance count (the count the p-value uses, for the chosen
   * `alternative`) reaches `h`; then `pseudoPValues` is `min(1, m h / L)` with `m` the alternative's
   * multiplier (2 for `'two-sided'`, else 1), `exceedances` is `h` and `L` is reported in
   * `output.permutationCounts`. A row that never reaches `h` runs all `P` permutations and keeps
   * the standard `min(m (M + 1), P + 1) / (P + 1)`, so with `h > P` the result equals the
   * unstopped one bit for bit. The Philox stream of permutation `p` is unchanged, so `L` and the
   * p-value do not depend on the dispatch shape. Saves work for rows that are clearly not
   * extreme; typical use is `h` near 10 to 20 with large `P`. Compile-time; omitted (default) runs
   * every permutation and is unchanged. Not supported with `alternative: 'folded'`, whose
   * statistic needs the mean of the full run first.
   */
  stopAfterExceedances?: number;
  /**
   * Largest neighbor count a tested row may have, in `[1, 64]`. Compile-time. Rows with more
   * included neighbors are not tested and set `overflow`. Defaults to 32.
   */
  maximumNeighbors?: number;
  /**
   * Compile-time multiple-testing correction of the pseudo p-values, default `'none'`. The family
   * is every tested row (finite p-value), never a display mask. With a method other than `'none'`,
   * `significant` is `adjustedPValue <= level` and `adjustedPValues` holds the adjusted values;
   * with `'none'`, `significant` is the exact integer comparison `p <= level`. Needs `significant`
   * or `adjustedPValues`. `'directed'` p-values are not super-uniform, so combining them with a
   * correction does not control the error rate.
   */
  multipleTesting?: GPUMultipleTestingMethod;
  /** Caller-owned outputs. */
  output: GPULocalPermutationTestOutput;
};

/**
 * Conditional-permutation (pseudo p-value) inference for local Moran's I, local Geary (with the
 * optional `classes` output), local Getis-Ord G / G*, local bivariate Moran, Lee's local L and the
 * univariate and bivariate local join counts, all with the same conditional-randomization scheme.
 *
 * Definitions, over the `n` analysis rows:
 * - A row is *included* when its mask is nonzero and its value is finite. An *active slot* of
 *   row `i` has a valid neighbor other than `i`, a positive weight and an included neighbor;
 *   `k_i` is their count (zero-weight slots are not neighbors, are not counted and are never
 *   drawn). An *island* is an included row with `k_i = 0`. Under `islands: 'exclude'` (default)
 *   islands are not analysis rows and every later definition ignores them; under `'keep'` they
 *   stay in `n` and the moments but are never tested. Rows with `k_i > maximumNeighbors` are not
 *   tested and set `overflow`.
 * - Row-standardized weights (`weights.information.transform === 'row'`) are re-standardized over each
 *   row's active slots, so masked, missing or excluded neighbors drop out of the lag instead of
 *   shrinking it toward 0 (identical to the given weights when nothing is dropped). Other
 *   transforms are used as given.
 * - Statistics: local Moran `I_i = (n-1) c_i sum_s w_s c_s / sum c^2` with `c = x - mean`; local
 *   Geary `c_i = n sum_s w_s (c_i - c_s)^2 / sum c^2`; `G_i = sum_s w_s x_s / (sum x - x_i)`;
 *   `G*_i = (x_i + sum_s w_s x_s) / sum x`, that is, a self weight of 1 (esda's `star=True` with
 *   binary weights). The mean is a tree sum with a second-pass correction and `n` is an exact
 *   uint32. The bivariate Moran, Lee's L and the (bivariate) join counts are defined at
 *   {@link GPULocalPermutationStatistic}; join counts use binary weights, test only rows with a
 *   nonzero count and draw the pair `(x, y)` of each drawn row together.
 * - Each permutation draws an ordered sample of `k_i` distinct analysis rows uniformly from the
 *   other `n - 1` (a partial Fisher-Yates shuffle of their compacted positions) and assigns them
 *   to the row's slots in slot order, as esda does. Draws come from Philox 4x32-10 streams keyed
 *   by the seed with counter `(block, row, permutation, tag)`, so they do not depend on the
 *   dispatch shape or on other rows.
 * - Tail counts `g = #{simulated >= observed}`, `l = #{simulated <= observed}` and the
 *   `alternative` define `M` and `p = min(m (M + 1), P + 1) / (P + 1)`; see
 *   {@link GPUInferenceAlternative}. The default `'two-sided'` is `m = 2, M = min(g, l)`, so a
 *   row whose simulations all tie the observed value has `p = 1`. The observed and simulated terms
 *   use the same f32 expression, and positive constant factors are dropped, so the counts are
 *   exact integers, bitwise reproducible for a seed.
 * - The permutation count is never clamped. A per-frame count outside `[1, maximumPermutations]`
 *   leaves every row untested (NaN p-values) and sets `overflow`; use `getParameterValues` to
 *   validate on the CPU.
 * - `significant`: `p <= level` (exact cross-multiplied integer comparison, no division) for
 *   `multipleTesting: 'none'`, else `adjustedPValue <= level` over the family of all tested rows.
 *
 * - With `stopAfterExceedances` (Besag-Clifford sequential stopping) a row leaves the loop at the
 *   first permutation whose exceedance count reaches the target; see the prop.
 *
 * Cost is `P * sum_i k_i` draws (less with `stopAfterExceedances`). Each row is one invocation looping
 * over the permutations.
 */
export class GPULocalPermutationTest implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPULocalPermutationTestProps;

  constructor(props: GPULocalPermutationTestProps) {
    const id = props.id ?? 'local-permutation-test';
    this.id = id;
    this.props = props;
    const rows = validatePermutationInputs({
      ...props,
      id,
      operation: OPERATION,
      islandCount: props.output.islandCount
    });
    if (
      ![
        'localMoran',
        'localGeary',
        'localG',
        'localGStar',
        'localBivariateMoran',
        'localLeeL',
        'localJoinCount',
        'localBivariateJoinCount'
      ].includes(props.statistic)
    ) {
      throw new Error(`${id} unknown statistic ${props.statistic}`);
    }
    if (props.statistic === 'localBivariateJoinCount') {
      if (props.joinCase !== 'BJC' && props.joinCase !== 'CLC') {
        throw new Error(`${id} localBivariateJoinCount requires joinCase 'BJC' or 'CLC'`);
      }
    } else if (props.joinCase !== undefined) {
      throw new Error(`${id} joinCase applies to localBivariateJoinCount only`);
    }
    const bivariate = getLocalStatisticSpec(props.statistic, props.joinCase).bivariate;
    if (bivariate && !props.secondValues) {
      throw new Error(`${id} ${props.statistic} requires secondValues`);
    }
    if (!bivariate && props.secondValues) {
      throw new Error(`${id} ${props.statistic} does not take secondValues`);
    }
    if (props.output.classes) {
      if (props.statistic !== 'localGeary') {
        throw new Error(`${id} classes applies to localGeary only`);
      }
      validatePackedUint32View(props.output.classes, `${id} classes`);
    }
    if (props.output.quadrants) {
      if (props.statistic !== 'localBivariateMoran') {
        throw new Error(`${id} quadrants applies to localBivariateMoran only`);
      }
      validatePackedUint32View(props.output.quadrants, `${id} quadrants`);
    }
    resolveStopAfterExceedances(
      id,
      props.stopAfterExceedances,
      resolvePermutationAlternative(id, props.alternative),
      props.maximumPermutations
    );
    if (props.output.permutationCounts) {
      if (props.stopAfterExceedances === undefined) {
        throw new Error(`${id} permutationCounts requires stopAfterExceedances`);
      }
      validatePackedUint32View(props.output.permutationCounts, `${id} permutationCounts`);
    }
    resolveGPUIslandPolicy(id, props.islands);
    validateCapacity(
      id,
      'maximumNeighbors',
      props.maximumNeighbors ?? 32,
      1,
      GPU_LOCAL_PERMUTATION_MAXIMUM_NEIGHBORS
    );
    validatePackedUint32View(props.output.exceedances, `${id} exceedances`);
    validatePackedView(props.output.pseudoPValues, ['float32'], `${id} pseudoPValues`);
    if (props.output.observed) {
      validatePackedView(props.output.observed, ['float32'], `${id} observed`);
    }
    if (props.output.significant) {
      validatePackedUint32View(props.output.significant, `${id} significant`);
    }
    if (props.output.adjustedPValues) {
      validatePackedView(props.output.adjustedPValues, ['float32'], `${id} adjustedPValues`);
    }
    for (const [name, view] of [
      ['exceedances', props.output.exceedances],
      ['permutationCounts', props.output.permutationCounts],
      ['pseudoPValues', props.output.pseudoPValues],
      ['observed', props.output.observed],
      ['significant', props.output.significant],
      ['adjustedPValues', props.output.adjustedPValues],
      ['classes', props.output.classes],
      ['quadrants', props.output.quadrants]
    ] as const) {
      if (view && view.length !== rows) {
        throw new Error(`${id} ${name} length must equal the weights row count`);
      }
    }
    const method = resolveGPUMultipleTestingMethod(id, props.multipleTesting);
    const {significant, adjustedPValues, classes} = props.output;
    if (method !== 'none' && !significant && !adjustedPValues && !classes) {
      throw new Error(`${id} multipleTesting requires significant or adjustedPValues`);
    }
    validatePackedUint32View(props.output.status.overflow, `${id} output.status.overflow`);
    validateCapacity(id, 'output.status.overflow length', props.output.status.overflow.length);
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        props.output.exceedances,
        props.output.permutationCounts,
        props.output.pseudoPValues,
        props.output.observed,
        props.output.significant,
        props.output.adjustedPValues,
        props.output.classes,
        props.output.quadrants,
        props.output.islandCount,
        props.output.status.overflow
      ],
      [
        props.weights.offsets,
        props.weights.neighbors,
        props.weights.weights,
        props.values,
        props.secondValues,
        props.mask,
        props.parameters,
        props.weights.status?.overflow
      ]
    );
  }

  /**
   * Packs and validates per-frame settings against this test's `maximumPermutations`; throws
   * when `permutationCount` exceeds it. The kernels never clamp the count.
   */
  getParameterValues(settings: GPUPermutationSettings): Uint32Array {
    return getGPUPermutationParameterValues(settings, {
      maximumPermutations: this.props.maximumPermutations
    });
  }

  /** Describes how the p-values of a run with `settings` were produced, for legends. */
  getInferenceMetadata(settings: GPUPermutationSettings): GPUInferenceMetadata {
    // Validates the seed, the permutation count and the significance level.
    this.getParameterValues(settings);
    const keep = resolveGPUIslandPolicy(this.id, this.props.islands) === 'keep';
    const maximumNeighbors = this.props.maximumNeighbors ?? 32;
    return {
      alternative: resolvePermutationAlternative(this.id, this.props.alternative),
      permutationCount: settings.permutationCount,
      seed: settings.seed,
      multipleTesting: resolveGPUMultipleTestingMethod(this.id, this.props.multipleTesting),
      significanceLevel: settings.significanceLevel ?? 0.05,
      testedRowsDefinition:
        `rows with a nonzero mask and a finite value that have 1 to ${maximumNeighbors} ` +
        `positive-weight neighbors among the analysis rows (islands, rows without such a neighbor, ` +
        (keep ? 'are kept in n but not tested)' : 'are excluded from the analysis)')
    };
  }

  /** Returns the permutation-test nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {weights, parameters, statistic, maximumPermutations} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      weights.offsets,
      weights.neighbors,
      weights.weights,
      weights.status?.overflow,
      props.values,
      props.secondValues,
      props.mask,
      parameters,
      props.output.exceedances,
      props.output.permutationCounts,
      props.output.pseudoPValues,
      props.output.observed,
      props.output.significant,
      props.output.adjustedPValues,
      props.output.classes,
      props.output.quadrants,
      props.output.islandCount,
      props.output.status.overflow
    ]);
    const rows = props.values.length;
    const alternative = resolvePermutationAlternative(id, props.alternative);
    const multiplier = getAlternativeMultiplier(alternative);
    const folded = alternative === 'folded';
    const stopAfter = props.stopAfterExceedances;
    const sequential = stopAfter !== undefined;
    const keepIslands = resolveGPUIslandPolicy(id, props.islands) === 'keep';
    const method = resolveGPUMultipleTestingMethod(id, props.multipleTesting);
    const maximumNeighbors = props.maximumNeighbors ?? 32;
    const spec = getLocalStatisticSpec(statistic, props.joinCase);
    const bivariate = spec.bivariate;
    const inputs = getPermutationInputNodes<Parameters>(graph, {
      ...props,
      islandCount: props.output.islandCount,
      id: `${id}-inputs`,
      operation: OPERATION,
      centerX: spec.centerX,
      centerY: spec.centerY
    });
    const nodes = inputs.nodes;
    const {rowPositions, compactX, compactY, totals} = inputs;
    const effectiveWeights = inputs.weights;
    const observedRaw = createTransientView(graph, `${id}-observed-raw`, 'float32', rows);
    // Bivariate statistics read (x, y) pairs from one interleaved view: the permutation kernel
    // would otherwise exceed the eight storage buffers of a compute stage.
    const pairs = bivariate
      ? createTransientView(graph, `${id}-pairs`, 'float32', 2 * rows)
      : undefined;
    const dataBinding: WGSLKernelBinding = pairs
      ? {name: 'pairs', view: pairs, type: 'f32', access: 'read'}
      : {name: 'compactX', view: compactX, type: 'f32', access: 'read'};
    const constantsWGSL = `const ROWS: u32 = ${rows}u;
const CAPACITY: u32 = ${weights.neighbors.length}u;
const INVALID_POSITION: u32 = ${PERMUTATION_INVALID_POSITION}u;
const MAXIMUM_NEIGHBORS: u32 = ${maximumNeighbors}u;
const NOT_TESTED: u32 = ${GPU_LOCAL_PERMUTATION_NOT_TESTED}u;
${PERMUTATION_FLOAT_WGSL}
// A binary join-count variable: exactly 1 is a join, anything else is 0.
fn isOne(value: f32) -> f32 {
  return select(0.0, 1.0, value == 1.0);
}
// The statistic up to a positive per-row constant; identical for observed and simulated terms.
fn getLocalTerm(fx: f32, fy: f32, lagA: f32, lagB: f32) -> f32 {
  return ${spec.term};
}
// Per-draw terms of the weighted lags, from the focus (fx, fy) and the drawn row (dx, dy).
fn getDrawnA(fx: f32, fy: f32, dx: f32, dy: f32) -> f32 {
  return ${spec.drawnA};
}
fn getDrawnB(fx: f32, fy: f32, dx: f32, dy: f32) -> f32 {
  return ${spec.drawnB ?? '0.0'};
}`;
    // Reads the compacted x and y of a position (y is 0 for univariate statistics).
    const accessWGSL = pairs
      ? `fn getX(position: u32) -> f32 { return pairs[pairsOffset + 2u * position]; }
fn getY(position: u32) -> f32 { return pairs[pairsOffset + 2u * position + 1u]; }`
      : `fn getX(position: u32) -> f32 { return compactX[compactXOffset + position]; }
fn getY(position: u32) -> f32 { return 0.0 * f32(position); }`;
    // Join counts are counts of neighbors: binary weights, as esda.
    const slotWeightWGSL = spec.joinCount ? '1.0' : 'weight';
    // Active slots of row `index` whose neighbor is an analysis row (the shared neighbor definition).
    const neighborLoopWGSL = (action: string) =>
      getGPUSpatialWeightsNeighborLoopWGSL(
        'rowPositions[rowPositionsOffset + neighbor] != INVALID_POSITION',
        action
      );
    const weightsOverflow = weights.status?.overflow;
    // Sequential stopping: the permute kernel already binds eight storage buffers, so it writes the
    // exceedance count and the permutation count into one transient (count at `index`, permutation
    // count at `ROWS + index`) that a following node splits into the two per-row views.
    const permuteResults = sequential
      ? createTransientView(graph, `${id}-permute-results`, 'uint32', 2 * rows)
      : undefined;
    const permutationCounts = sequential
      ? (props.output.permutationCounts ??
        createTransientView(graph, `${id}-permutation-counts`, 'uint32', rows))
      : undefined;

    nodes.push(
      createFillNode<Parameters>(graph, {
        id: `${id}-overflow-clear`,
        operation: OPERATION,
        view: props.output.status.overflow,
        type: 'u32',
        value: '0u',
        componentCount: 1
      })
    );
    if (weightsOverflow) {
      // Truncated weights: the front end analyzed no row; report it.
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-weights-overflow`,
          operation: OPERATION,
          variant: 'weights-overflow',
          bindings: [
            {name: 'weightsOverflow', view: weightsOverflow, type: 'u32', access: 'read'},
            {
              name: 'overflow',
              view: props.output.status.overflow,
              type: 'atomic<u32>',
              access: 'read_write'
            }
          ],
          invocationCount: 1,
          body: `if (weightsOverflow[weightsOverflowOffset] != 0u) {
    atomicMax(&overflow[overflowOffset], 1u);
  }`
        })
      );
    }
    if (pairs) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-pairs`,
          operation: OPERATION,
          variant: 'pairs',
          bindings: [
            {name: 'compactX', view: compactX, type: 'f32', access: 'read'},
            {name: 'compactY', view: compactY, type: 'f32', access: 'read'},
            {name: 'pairs', view: pairs, type: 'f32', access: 'read_write'}
          ],
          invocationCount: rows,
          body: `pairs[pairsOffset + 2u * index] = compactX[compactXOffset + index];
  pairs[pairsOffset + 2u * index + 1u] = compactY[compactYOffset + index];`
        })
      );
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-count-validity`,
        operation: OPERATION,
        variant: 'count-validity',
        bindings: [
          {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
          {
            name: 'overflow',
            view: props.output.status.overflow,
            type: 'atomic<u32>',
            access: 'read_write'
          }
        ],
        invocationCount: 1,
        declarations: getPermutationParameterWGSL(maximumPermutations),
        // The count is never clamped: report an invalid one (permute leaves every row untested).
        body: `if (!isPermutationCountValid()) {
    atomicMax(&overflow[overflowOffset], 1u);
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-observed`,
        operation: OPERATION,
        variant: 'observed',
        bindings: [
          {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
          {name: 'weights', view: effectiveWeights, type: 'f32', access: 'read'},
          {name: 'rowPositions', view: rowPositions, type: 'u32', access: 'read'},
          dataBinding,
          {name: 'observedRaw', view: observedRaw, type: 'f32', access: 'read_write'},
          {
            name: 'overflow',
            view: props.output.status.overflow,
            type: 'atomic<u32>',
            access: 'read_write'
          },
          ...(props.output.quadrants
            ? [
                {
                  name: 'quadrants',
                  view: props.output.quadrants,
                  type: 'u32' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        invocationCount: rows,
        declarations: `${constantsWGSL}
${accessWGSL}`,
        body: `var raw = getQuietNaN(index);
  var quadrant = 0u;
  let position = rowPositions[rowPositionsOffset + index];
  if (position != INVALID_POSITION) {
    let fx = getX(position);
    let fy = getY(position);
    var lagA = 0.0;
    var lagB = 0.0;
    var neighborCount = 0u;
    ${neighborLoopWGSL(`let neighborPosition = rowPositions[rowPositionsOffset + neighbor];
    let dx = getX(neighborPosition);
    let dy = getY(neighborPosition);
    lagA += ${slotWeightWGSL} * getDrawnA(fx, fy, dx, dy);
    ${spec.drawnB ? `lagB += ${slotWeightWGSL} * getDrawnB(fx, fy, dx, dy);` : ''}
    neighborCount++;`)}
    if (neighborCount > MAXIMUM_NEIGHBORS) {
      atomicMax(&overflow[overflowOffset], 1u);
    } else if (neighborCount > 0u || ${keepIslands}) {
      // A kept island has a zero lag: the formula value, never tested.
      raw = getLocalTerm(fx, fy, lagA, lagB);
      // esda Moran_Local_BV.q (PySAL scheme): 1 HH, 2 LH, 3 LL, 4 HL in (x_i, lag of y).
      quadrant = select(select(3u, 4u, fx > 0.0), select(2u, 1u, fx > 0.0), lagA > 0.0);
    }
  }
  ${props.output.quadrants ? 'quadrants[quadrantsOffset + index] = quadrant;' : ''}
  observedRaw[observedRawOffset + index] = raw;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-permute`,
        operation: OPERATION,
        variant: 'permute',
        bindings: [
          {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
          {name: 'weights', view: effectiveWeights, type: 'f32', access: 'read'},
          {name: 'rowPositions', view: rowPositions, type: 'u32', access: 'read'},
          dataBinding,
          {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
          {name: 'observedRaw', view: observedRaw, type: 'f32', access: 'read'},
          {
            name: 'exceedances',
            view: permuteResults ?? props.output.exceedances,
            type: 'u32',
            access: 'read_write'
          }
        ],
        invocationCount: rows,
        declarations: `${constantsWGSL}
${accessWGSL}
${getPermutationParameterWGSL(maximumPermutations)}
${PERMUTATION_RANDOM_WGSL}
${sequential ? `const STOP_AFTER_EXCEEDANCES: u32 = ${stopAfter}u;\n` : ''}const RANDOM_TAG: u32 = ${LOCAL_PERMUTATION_RANDOM_TAG}u;

// Value at a virtual shuffle position: a recorded swap, or the position itself.
fn readSwap(
  swapPositions: ptr<function, array<u32, ${maximumNeighbors}>>,
  swapValues: ptr<function, array<u32, ${maximumNeighbors}>>,
  swapCount: u32,
  position: u32
) -> u32 {
  for (var entry = 0u; entry < swapCount; entry++) {
    if ((*swapPositions)[entry] == position) {
      return (*swapValues)[entry];
    }
  }
  return position;
}`,
        body: `let observed = observedRaw[observedRawOffset + index];
  // The permutation count is never clamped: an invalid one leaves every row untested.
  // Join counts test only rows with at least one join (esda sets the other p-values to NaN).
  if (!isFiniteFloat(observed) || !isPermutationCountValid()${spec.joinCount ? ' || observed == 0.0' : ''}) {
    exceedances[exceedancesOffset + index] = NOT_TESTED;${sequential ? '\n    exceedances[exceedancesOffset + ROWS + index] = NOT_TESTED;' : ''}
    return;
  }
  let position = rowPositions[rowPositionsOffset + index];
  let others = rowPositions[rowPositionsOffset + ROWS] - 1u;
  let fx = getX(position);
  let fy = getY(position);
  let permutations = readValidPermutationCount();${sequential ? '\n  var permutationsRun = permutations;' : ''}
  let key = readSeedKey();
  var greater = 0u;
  var lesser = 0u;
  var folded = 0u;
  var total = 0.0;
  var mean = 0.0;
  // The row's active neighbor weights in slot order, read once: a permutation only reassigns
  // values to slots, so the per-draw work is the random draw and one value gather, not four
  // CSR gathers (offsets, neighbor, neighbor position, weight) per slot per permutation.
  var slotWeights: array<f32, ${maximumNeighbors}>;
  var neighborTotal = 0u;
  ${neighborLoopWGSL(`if (neighborTotal < MAXIMUM_NEIGHBORS) {
      slotWeights[neighborTotal] = ${slotWeightWGSL};
      neighborTotal++;
    }`)}
  // Islands (kept) and rows that cannot draw k distinct others are not tested.
  if (neighborTotal == 0u || neighborTotal > others) {
    exceedances[exceedancesOffset + index] = NOT_TESTED;${sequential ? '\n    exceedances[exceedancesOffset + ROWS + index] = NOT_TESTED;' : ''}
    return;
  }
  // 'folded' needs the simulated mean first: sweep 0 sums, sweep 1 counts (same deterministic stream).
  for (var sweep = 0u; sweep < ${folded ? 2 : 1}u; sweep++) {
  for (var permutation = 0u; permutation < permutations; permutation++) {
    var stream = createPhiloxStream(key, index, permutation, RANDOM_TAG);
    var swapPositions: array<u32, ${maximumNeighbors}>;
    var swapValues: array<u32, ${maximumNeighbors}>;
    var swapCount = 0u;
    var lagA = 0.0;
    var lagB = 0.0;
    for (var drawn = 0u; drawn < neighborTotal; drawn++) {
      // Partial Fisher-Yates over the other analysis rows' virtual positions [0, others).
      let pick = drawn + nextPhiloxBelow(&stream, others - drawn);
      let chosen = readSwap(&swapPositions, &swapValues, swapCount, pick);
      if (pick != drawn) {
        let displaced = readSwap(&swapPositions, &swapValues, swapCount, drawn);
        var entry = 0u;
        loop {
          if (entry == swapCount || swapPositions[entry] == pick) {
            break;
          }
          entry++;
        }
        swapPositions[entry] = pick;
        swapValues[entry] = displaced;
        swapCount = max(swapCount, entry + 1u);
      }
      let drawnPosition = chosen + select(0u, 1u, chosen >= position);
      let dx = getX(drawnPosition);
      let dy = getY(drawnPosition);
      lagA += slotWeights[drawn] * getDrawnA(fx, fy, dx, dy);
      ${spec.drawnB ? 'lagB += slotWeights[drawn] * getDrawnB(fx, fy, dx, dy);' : ''}
    }
    let simulated = getLocalTerm(fx, fy, lagA, lagB);
    ${
      folded
        ? `if (sweep == 0u) {
      total += simulated;
    } else {
      folded += select(0u, 1u, abs(simulated - mean) >= abs(observed - mean));
    }`
        : `greater += select(0u, 1u, simulated >= observed);
    lesser += select(0u, 1u, simulated <= observed);${
      sequential
        ? `
    // Besag-Clifford: stop at the first permutation whose exceedance count reaches the target.
    if (${getExceedanceCountWGSL(alternative)} >= STOP_AFTER_EXCEEDANCES) {
      permutationsRun = permutation + 1u;
      break;
    }`
        : ''
    }`
    }
  }
  mean = total / f32(permutations);
  }
  exceedances[exceedancesOffset + index] = ${getExceedanceCountWGSL(alternative)};${sequential ? '\n  exceedances[exceedancesOffset + ROWS + index] = permutationsRun;' : ''}`
      })
    );

    if (permuteResults && permutationCounts) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-split-counts`,
          operation: OPERATION,
          variant: 'split-counts',
          bindings: [
            {name: 'permuteResults', view: permuteResults, type: 'u32', access: 'read'},
            {
              name: 'exceedances',
              view: props.output.exceedances,
              type: 'u32',
              access: 'read_write'
            },
            {name: 'permutationCounts', view: permutationCounts, type: 'u32', access: 'read_write'}
          ],
          invocationCount: rows,
          declarations: `const ROWS: u32 = ${rows}u;`,
          body: `exceedances[exceedancesOffset + index] = permuteResults[permuteResultsOffset + index];
  permutationCounts[permutationCountsOffset + index] = permuteResults[permuteResultsOffset + ROWS + index];`
        })
      );
    }

    // The classification needs the scaled local Geary values even when the caller wants none.
    const observedView =
      props.output.observed ??
      (props.output.classes
        ? createTransientView(graph, `${id}-observed-geary`, 'float32', rows)
        : undefined);
    if (observedView) {
      const count = 'totals[totalsOffset]';
      const sumSquaresX = `totals[totalsOffset + ${PERMUTATION_TOTAL.sumSquares}u]`;
      const sumSquaresY = `totals[totalsOffset + ${PERMUTATION_TOTAL.sumSquaresY}u]`;
      const observedWGSL = {
        localMoran: `(${count} - 1.0) * raw / ${sumSquaresX}`,
        localGeary: `raw * ${count} / ${sumSquaresX}`,
        localG: 'raw / (totals[totalsOffset + 1u] - compactX[compactXOffset + position])',
        localGStar: 'raw / totals[totalsOffset + 1u]',
        localBivariateMoran: `(${count} - 1.0) * raw / sqrt(${sumSquaresX} * ${sumSquaresY})`,
        localLeeL: `${count} * raw / sqrt(${sumSquaresX} * ${sumSquaresY})`,
        localJoinCount: 'raw',
        localBivariateJoinCount: 'raw'
      }[statistic];
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-observed-statistic`,
          operation: OPERATION,
          variant: 'observed-statistic',
          bindings: [
            {name: 'observedRaw', view: observedRaw, type: 'f32', access: 'read'},
            {name: 'rowPositions', view: rowPositions, type: 'u32', access: 'read'},
            {name: 'compactX', view: compactX, type: 'f32', access: 'read'},
            {name: 'totals', view: totals, type: 'f32', access: 'read'},
            {name: 'observed', view: observedView, type: 'f32', access: 'read_write'}
          ],
          invocationCount: rows,
          declarations: constantsWGSL,
          body: `let raw = observedRaw[observedRawOffset + index];
  let position = min(rowPositions[rowPositionsOffset + index], ROWS - 1u);
  let scaled = ${observedWGSL};
  // Canonical +0: a zero lag times a negative deviation is -0. Tested on the bits because a
  // floating-point compare may be folded by the shader compiler, which treats -0 as 0.
  let scaledBits = bitcast<u32>(scaled);
  let canonical = bitcast<f32>(select(scaledBits, 0u, (scaledBits & 0x7fffffffu) == 0u));
  observed[observedOffset + index] = select(getQuietNaN(index), canonical, isFiniteFloat(raw));`
        })
      );
    }

    const wantsAdjustment = method !== 'none' || props.output.adjustedPValues !== undefined;
    const classifyBindings: WGSLKernelBinding[] = [
      {name: 'exceedances', view: props.output.exceedances, type: 'u32', access: 'read'},
      {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
      {name: 'pseudoPValues', view: props.output.pseudoPValues, type: 'f32', access: 'read_write'}
    ];
    if (permutationCounts) {
      classifyBindings.push({
        name: 'permutationCounts',
        view: permutationCounts,
        type: 'u32',
        access: 'read'
      });
    }
    // A stopped row (exceedance count equal to the target) has p = min(m h, L) / L; the others keep
    // min(m (M + 1), P + 1) / (P + 1). Both are f32 division of exact small integers.
    const numeratorWGSL = sequential
      ? `select(min(${multiplier}u * (count + 1u), readValidPermutationCount() + 1u), min(${multiplier}u * STOP_AFTER_EXCEEDANCES, permutationCounts[permutationCountsOffset + index]), count == STOP_AFTER_EXCEEDANCES)`
      : `min(${multiplier}u * (count + 1u), readValidPermutationCount() + 1u)`;
    const denominatorWGSL = sequential
      ? 'select(readValidPermutationCount() + 1u, permutationCounts[permutationCountsOffset + index], count == STOP_AFTER_EXCEEDANCES)'
      : 'readValidPermutationCount() + 1u';
    // The exact integer comparison of the unadjusted path: only correctly rounded f32 products.
    const exactSignificantWGSL = `significant[significantOffset + index] = select(0u, 1u, tested && f32(${numeratorWGSL}) <= readSignificanceLevel() * f32(${denominatorWGSL}));`;
    if (props.output.significant && method === 'none') {
      classifyBindings.push({
        name: 'significant',
        view: props.output.significant,
        type: 'u32',
        access: 'read_write'
      });
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-classify`,
        operation: OPERATION,
        variant: 'classify',
        bindings: classifyBindings,
        invocationCount: rows,
        declarations: `${getPermutationParameterWGSL(maximumPermutations)}
const NOT_TESTED: u32 = ${GPU_LOCAL_PERMUTATION_NOT_TESTED}u;${sequential ? `\nconst STOP_AFTER_EXCEEDANCES: u32 = ${stopAfter}u;` : ''}`,
        body: `let count = exceedances[exceedancesOffset + index];
  let tested = count != NOT_TESTED;
  let pValue = f32(${numeratorWGSL}) / f32(${denominatorWGSL});
  pseudoPValues[pseudoPValuesOffset + index] = select(bitcast<f32>(0x7fc00000u | (index & 0u)), pValue, tested);
  ${props.output.significant && method === 'none' ? exactSignificantWGSL : ''}`
      })
    );

    let adjustedForClasses: GraphDataView<'float32'> | undefined;
    if (wantsAdjustment) {
      const adjustedPValues =
        props.output.adjustedPValues ??
        createTransientView(graph, `${id}-adjusted-p`, 'float32', rows);
      adjustedForClasses = method === 'none' ? undefined : adjustedPValues;
      nodes.push(
        ...new GPUMultipleTestingAdjustment({
          id: `${id}-multiple-testing`,
          pValues: props.output.pseudoPValues,
          method,
          output: {adjustedPValues}
        }).getCommandNodes(graph)
      );
      if (props.output.significant) {
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-significant`,
            operation: OPERATION,
            variant: 'significant',
            bindings: [
              {name: 'adjustedPValues', view: adjustedPValues, type: 'f32', access: 'read'},
              {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
              {
                name: 'significant',
                view: props.output.significant,
                type: 'u32',
                access: 'read_write'
              }
            ],
            invocationCount: rows,
            declarations: getPermutationParameterWGSL(maximumPermutations),
            // Adjusted p-values are NaN where a row was not tested, and NaN compares false.
            body: `let adjusted = adjustedPValues[adjustedPValuesOffset + index];
  significant[significantOffset + index] = select(0u, 1u, adjusted <= readSignificanceLevel());`
          })
        );
      }
    }
    if (props.output.classes && observedView) {
      nodes.push(
        ...this.getGearyClassificationNodes(
          graph,
          observedView,
          adjustedForClasses,
          compactX,
          rowPositions,
          permutationCounts
        )
      );
    }
    return nodes;
  }

  /**
   * Local Geary classification (esda `Geary_Local` labels): the mean of the scaled local values over
   * the analysis rows, then one class per row.
   */
  private getGearyClassificationNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    observedView: GraphDataView<'float32'>,
    adjustedPValues: GraphDataView<'float32'> | undefined,
    compactX: GraphDataView<'float32'>,
    rowPositions: GraphDataView<'uint32'>,
    permutationCounts: GraphDataView<'uint32'> | undefined
  ): GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const rows = props.values.length;
    const classes = props.output.classes!;
    const multiplier = getAlternativeMultiplier(
      resolvePermutationAlternative(id, props.alternative)
    );
    const matrix = createTransientView(graph, `${id}-geary-mean-matrix`, 'float32', 2 * rows);
    const sums = createTransientView(graph, `${id}-geary-mean-sums`, 'float32', 2);
    const stopAfter = props.stopAfterExceedances;
    // Sequential stopping: a stopped row (count equal to the target) is compared as min(m h, L) / L.
    const unadjustedNumeratorWGSL =
      stopAfter !== undefined
        ? `select(min(${multiplier}u * (count + 1u), readValidPermutationCount() + 1u), min(${multiplier}u * ${stopAfter}u, permutationCounts[permutationCountsOffset + index]), count == ${stopAfter}u)`
        : `min(${multiplier}u * (count + 1u), readValidPermutationCount() + 1u)`;
    const unadjustedDenominatorWGSL =
      stopAfter !== undefined
        ? `select(readValidPermutationCount() + 1u, permutationCounts[permutationCountsOffset + index], count == ${stopAfter}u)`
        : 'readValidPermutationCount() + 1u';
    const significantWGSL = adjustedPValues
      ? 'adjustedPValues[adjustedPValuesOffset + index] <= readSignificanceLevel()'
      : `count != NOT_TESTED && f32(${unadjustedNumeratorWGSL}) <= readSignificanceLevel() * f32(${unadjustedDenominatorWGSL})`;
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-geary-mean-inputs`,
        operation: OPERATION,
        variant: 'geary-mean-inputs',
        bindings: [
          {name: 'observedView', view: observedView, type: 'f32', access: 'read'},
          {name: 'matrix', view: matrix, type: 'f32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations: `const ROWS: u32 = ${rows}u;
${PERMUTATION_FLOAT_WGSL}`,
        // Column 0: the local value of every row with one; column 1: how many such rows.
        body: `let value = observedView[observedViewOffset + index];
  let finite = isFiniteFloat(value);
  matrix[matrixOffset + index] = select(0.0, value, finite);
  matrix[matrixOffset + ROWS + index] = select(0.0, 1.0, finite);`
      }),
      ...getColumnSumNodes<Parameters>(graph, {
        id: `${id}-geary-mean`,
        operation: OPERATION,
        matrix,
        rows,
        blockCount: Math.ceil(rows / BLOCK_ROWS),
        first: 0,
        count: 2,
        totals: sums
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-geary-classes`,
        operation: OPERATION,
        variant: adjustedPValues ? 'geary-classes-adjusted' : 'geary-classes',
        bindings: [
          {name: 'observedView', view: observedView, type: 'f32', access: 'read'},
          {name: 'compactX', view: compactX, type: 'f32', access: 'read'},
          {name: 'rowPositions', view: rowPositions, type: 'u32', access: 'read'},
          {name: 'sums', view: sums, type: 'f32', access: 'read'},
          {name: 'parameters', view: props.parameters, type: 'u32', access: 'read'},
          adjustedPValues
            ? {name: 'adjustedPValues', view: adjustedPValues, type: 'f32', access: 'read'}
            : {name: 'exceedances', view: props.output.exceedances, type: 'u32', access: 'read'},
          ...(!adjustedPValues && permutationCounts
            ? [
                {
                  name: 'permutationCounts',
                  view: permutationCounts,
                  type: 'u32' as const,
                  access: 'read' as const
                }
              ]
            : []),
          {name: 'classes', view: classes, type: 'u32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations: `${getPermutationParameterWGSL(props.maximumPermutations)}
const INVALID_POSITION: u32 = ${PERMUTATION_INVALID_POSITION}u;
const NOT_TESTED: u32 = ${GPU_LOCAL_PERMUTATION_NOT_TESTED}u;
${PERMUTATION_FLOAT_WGSL}`,
        // Codes of GPU_LOCAL_GEARY_CLASS. NaN p-values (rows not tested) compare false and stay 0.
        body: `let value = observedView[observedViewOffset + index];
  let position = rowPositions[rowPositionsOffset + index];
  let mean = sums[sumsOffset] / sums[sumsOffset + 1u];
  ${adjustedPValues ? '' : 'let count = exceedances[exceedancesOffset + index];'}
  var code = 0u;
  if (isFiniteFloat(value) && position != INVALID_POSITION && ${adjustedPValues ? 'isFiniteFloat(adjustedPValues[adjustedPValuesOffset + index]) && ' : 'count != NOT_TESTED && '}isFiniteFloat(mean)) {
    if (!(${significantWGSL})) {
      code = ${GPU_LOCAL_GEARY_CLASS.notSignificant}u;
    } else if (value < mean) {
      let focus = compactX[compactXOffset + position];
      code = select(select(0u, ${GPU_LOCAL_GEARY_CLASS.lowLow}u, focus < 0.0), ${GPU_LOCAL_GEARY_CLASS.highHigh}u, focus > 0.0);
    } else if (value > mean) {
      code = ${GPU_LOCAL_GEARY_CLASS.negativeAssociation}u;
    }
  }
  classes[classesOffset + index] = code;`
      })
    ];
  }
}
