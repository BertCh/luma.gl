// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import type {GPUInferenceAlternative, GPUInferenceMetadata} from '../contracts/index';
import {
  getGPUSpatialWeightsNeighborLoopWGSL,
  resolveGPUIslandPolicy,
  type GPUIslandPolicy,
  type GPUSpatialWeights
} from '../spatial-weights/spatial-weights';
import {
  getPermutationInputNodes,
  getPermutationParameterWGSL,
  PERMUTATION_FLOAT_WGSL,
  PERMUTATION_INVALID_POSITION,
  PERMUTATION_TOTAL,
  validatePermutationInputs
} from './permutation-inference-kernels';
import {PERMUTATION_RANDOM_WGSL} from './permutation-random';
import {PERMUTATION_SHUFFLE_WGSL} from './permutation-shuffle';
import {
  getAlternativeMultiplier,
  getExceedanceCountWGSL,
  resolvePermutationAlternative
} from './permutation-alternative';
import {
  getGPUPermutationParameterValues,
  type GPUPermutationParameters
} from './permutation-parameters';

const OPERATION = 'GPUGlobalPermutationTest';

/** Threads of the single finalize workgroup. */
const FINALIZE_WORKGROUP_SIZE = 256;

/** Rows per workgroup block of the per-permutation pair sums. */
const PAIR_BLOCK_ROWS = 2048;
const PAIR_WORKGROUP_SIZE = 256;

/**
 * A global statistic tested by {@link GPUGlobalPermutationTest}. `'leesL'` is Lee's spatial
 * Pearson `L_xy` (esda `Spatial_Pearson`, element `[0, 1]`).
 */
export type GPUGlobalPermutationStatistic =
  | 'moran'
  | 'geary'
  | 'getisOrdG'
  | 'bivariateMoran'
  | 'leesL';

/** Fields of the `results` view of {@link GPUGlobalPermutationTest}. */
export const GPU_GLOBAL_PERMUTATION_RESULT = {
  /** Observed statistic. */
  observed: 0,
  /** esda `p_sim`: `min(m (M + 1), P + 1) / (P + 1)` for the chosen `alternative` (`m` is 2 for two-sided, else 1). */
  pseudoPValue: 1,
  /** Exceedance count `M` of the chosen `alternative` (an exact integer; folded for 'directed'). */
  exceedances: 2,
  /** Permutation count `P` used; NaN (and every p-value NaN) when the requested count is outside `[1, maximumPermutations]`. */
  permutations: 3,
  /** Mean of the simulated statistics (esda `EI_sim`). */
  simulatedMean: 4,
  /** Population standard deviation of the simulated statistics (esda `seI_sim`). */
  simulatedStandardDeviation: 5,
  /** `(observed - simulatedMean) / simulatedStandardDeviation` (esda `z_sim`). */
  zSimulated: 6,
  /** One-tailed normal p-value of `|zSimulated|` (esda `p_z_sim`). */
  pZSimulated: 7,
  /** Smallest simulated statistic, the histogram's lower edge. */
  minimum: 8,
  /** Largest simulated statistic, the histogram's upper edge. */
  maximum: 9,
  /** Number of islands among the included rows (included rows without a positive-weight included neighbor). */
  islandCount: 10,
  /** Number `n` of analysis rows used (islands excluded under `islands: 'exclude'`). */
  analysisCount: 11,
  /** Number of float32 values in `results`. */
  length: 12
} as const;

/**
 * Properties for {@link GPUGlobalPermutationTest}.
 *
 * Per-frame (no rebuild or recompile): the contents of `weights`, `values`, `secondValues`, `mask`
 * and `parameters` (seed and permutation count). Compile-time: the row count, slot capacity,
 * `statistic`, `maximumPermutations`, `histogramBins` and which optional views exist.
 */
export type GPUGlobalPermutationTestProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'global-permutation-test'`. */
  id?: string;
  /** Square spatial weights. */
  weights: GPUSpatialWeights;
  /** Analysis values `x`, one per row. */
  values: GraphDataView<'float32'>;
  /**
   * Second variable `y` of bivariate Moran's I (permuted, `x` stays) or of Lee's L (`x` and `y` are
   * permuted together, as esda `Spatial_Pearson`). Required for `'bivariateMoran'` and `'leesL'`.
   */
  secondValues?: GraphDataView<'float32'>;
  /** Optional row selection: nonzero includes the row. */
  mask?: GraphDataView<'uint32'>;
  /** Global statistic. Compile-time. */
  statistic: GPUGlobalPermutationStatistic;
  /**
   * Tail of the pseudo p-value (see {@link GPUInferenceAlternative}). Compile-time. Defaults to
   * `'two-sided'`. `'directed'` is esda's legacy default and is anti-conservative.
   */
  alternative?: GPUInferenceAlternative;
  /**
   * Treatment of islands (included rows with no positive-weight included neighbor). Compile-time.
   * `'exclude'` (default): not counted in `n`, the mean, the variance or `S0`. `'keep'` (esda):
   * islands stay in `n` and the moments. Row-standardized weights are re-standardized over each
   * row's active slots either way; see {@link GPUGlobalPermutationTest}.
   */
  islands?: GPUIslandPolicy;
  /**
   * Per-frame uint32 parameters of at least `GPU_PERMUTATION_PARAMETER_LENGTH` elements written with
   * `getGPUPermutationParameterValues`.
   */
  parameters: GraphDataView<'uint32'>;
  /** Upper bound of the per-frame permutation count, at most 2^20. Compile-time; sizes dispatches. */
  maximumPermutations: number;
  /**
   * Caller-owned results, `GPU_GLOBAL_PERMUTATION_RESULT.length` float32 values, including the
   * island and analysis row counts.
   */
  results: GraphDataView<'float32'>;
  /**
   * Optional caller-owned reference distribution: the statistic of permutation `p` at index
   * `p - 1`, quiet NaN past `P`. Length `maximumPermutations`.
   */
  referenceDistribution?: GraphDataView<'float32'>;
  /** Optional caller-owned histogram of the reference distribution over `[minimum, maximum]`. */
  histogram?: GraphDataView<'uint32'>;
};

/**
 * Global permutation inference (esda `p_sim`, `z_sim`) for Moran's I, Geary's C, Getis-Ord General
 * G, bivariate Moran's I and Lee's L, with the reference distribution and its histogram.
 *
 * Definitions over the `n` analysis rows, with the statistic definitions of
 * `GPUGlobalSpatialStatistics`:
 * - A row is *included* when its mask is nonzero and its values are finite. An *active slot* has
 *   a valid neighbor other than the row, a positive weight and an included neighbor; an *island*
 *   is an included row without an active slot. Under `islands: 'exclude'` (default) islands are
 *   not analysis rows (they are not foci or neighbors and are not in `n`, the mean or `S0`);
 *   under `'keep'` they stay in `n` and the moments, as esda does. Excluded rows are neither foci
 *   nor neighbors. There is no cascade: a row whose only neighbors were islands stays in `n` and
 *   adds 0 to `S0`.
 * - Row-standardized weights (`weights.info.transform === 'row'`) are re-standardized over each
 *   row's active slots, in the pair sums and in `S0`, so masked or missing neighbors drop out
 *   instead of shrinking the row sum below 1. Other transforms are used as given.
 * - Permutation `p` (1 to `P`) relabels the analysis values by a keyed pseudorandom bijection of
 *   their compacted positions: a swap-or-not shuffle (Hoang, Morris and Rogaway 2012) with
 *   `6 ceil(log2 n) + 6` rounds whose keys come from Philox 4x32-10 on the seed and `p`: no sort
 *   per permutation and no `P x n` memory. The earlier four-round Feistel network was measurably
 *   non-uniform on small domains (see the documentation for the evidence). Bivariate Moran keeps
 *   `x` and permutes `y`, as esda's `Moran_BV`. Lee's L permutes the pair `(x, y)` of each row, as esda's `Spatial_Pearson`; its
 *   denominator is `1'W'W1` (`sum_i (sum_j w_ij)^2`).
 * - Each statistic is a positive constant times a pair sum `sum_i f(v_i, sum_j w_ij g(v_j))`. The
 *   observed value is the identity permutation run through the same kernel, so `larger =
 *   #{sim >= observed}` and `l = #{sim <= observed}` compare identically computed f32 pair sums;
 *   the `alternative` (default `'two-sided'`, see {@link GPUInferenceAlternative}) defines the
 *   exceedance count `M` and `p = min(m (M + 1), P + 1) / (P + 1)`.
 * - The permutation count is never clamped: a count outside `[1, maximumPermutations]` writes NaN
 *   for `permutations`, every p-value and every simulated summary (use `getParameterValues` to
 *   validate on the CPU). Truncated weights (`weights.status.overflow`) analyze no row, so the
 *   results are NaN.
 * - Pair sums are reduced in a fixed shape (one workgroup per permutation and row block, a
 *   shared-memory tree, then a fixed sequential sum over blocks). No float atomics: results are
 *   bitwise reproducible for a seed.
 *
 * Cost is `(P + 1)` pair sums, i.e. `(P + 1) * nnz` weighted terms, each evaluating the Feistel
 * shuffle for the neighbor.
 */
export class GPUGlobalPermutationTest implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUGlobalPermutationTestProps;

  constructor(props: GPUGlobalPermutationTestProps) {
    const id = props.id ?? 'global-permutation-test';
    this.id = id;
    this.props = props;
    validatePermutationInputs({...props, id, operation: OPERATION});
    if (!['moran', 'geary', 'getisOrdG', 'bivariateMoran', 'leesL'].includes(props.statistic)) {
      throw new Error(`${id} unknown statistic ${props.statistic}`);
    }
    resolvePermutationAlternative(id, props.alternative);
    resolveGPUIslandPolicy(id, props.islands);
    if (
      (props.statistic === 'bivariateMoran' || props.statistic === 'leesL') &&
      !props.secondValues
    ) {
      throw new Error(`${id} ${props.statistic} requires secondValues`);
    }
    validatePackedView(props.results, ['float32'], `${id} results`);
    if (props.results.length < GPU_GLOBAL_PERMUTATION_RESULT.length) {
      throw new Error(
        `${id} results must hold ${GPU_GLOBAL_PERMUTATION_RESULT.length} float32 values`
      );
    }
    if (props.referenceDistribution) {
      validatePackedView(props.referenceDistribution, ['float32'], `${id} referenceDistribution`);
      if (props.referenceDistribution.length !== props.maximumPermutations) {
        throw new Error(`${id} referenceDistribution length must equal maximumPermutations`);
      }
    }
    if (props.histogram) {
      validatePackedUint32View(props.histogram, `${id} histogram`);
      if (props.histogram.length < 1) {
        throw new Error(`${id} histogram must hold at least one bin`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.results, props.referenceDistribution, props.histogram],
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
   * Packs and validates per-frame parameters against this test's `maximumPermutations`; throws
   * when `permutations` exceeds it. The kernels never clamp the count.
   */
  getParameterValues(parameters: GPUPermutationParameters): Uint32Array {
    return getGPUPermutationParameterValues(parameters, {
      maximumPermutations: this.props.maximumPermutations
    });
  }

  /** Describes how the p-value of a run with `parameters` was produced, for legends. */
  getInferenceMetadata(parameters: GPUPermutationParameters): GPUInferenceMetadata {
    // Validates the seed and the permutation count.
    this.getParameterValues(parameters);
    const keep = resolveGPUIslandPolicy(this.id, this.props.islands) === 'keep';
    return {
      alternative: resolvePermutationAlternative(this.id, this.props.alternative),
      permutationCount: parameters.permutations,
      seed: parameters.seed,
      multipleTesting: 'none',
      testedRowsDefinition:
        'one global statistic over the analysis rows (nonzero mask, finite values; islands, ' +
        (keep
          ? 'rows without a positive-weight neighbor, are kept in n)'
          : 'rows without a positive-weight neighbor, are excluded)')
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
      props.values,
      props.secondValues,
      props.mask,
      parameters,
      weights.status?.overflow,
      props.results,
      props.referenceDistribution,
      props.histogram
    ]);
    const rows = props.values.length;
    const lee = statistic === 'leesL';
    const bivariate = statistic === 'bivariateMoran' || lee;
    const alternative = resolvePermutationAlternative(id, props.alternative);
    const multiplier = getAlternativeMultiplier(alternative);
    const inputs = getPermutationInputNodes<Parameters>(graph, {
      ...props,
      secondValues: bivariate ? props.secondValues : undefined,
      islands: props.islands,
      id: `${id}-inputs`,
      operation: OPERATION,
      centerX: statistic !== 'getisOrdG'
    });
    const nodes = inputs.nodes;
    const {rowPositions, compactX, compactY, totals, islandCount} = inputs;
    const effectiveWeights = inputs.weights;
    const blockCount = Math.ceil(rows / PAIR_BLOCK_ROWS);
    const sumCount = maximumPermutations + 1;
    const partials = createTransientView(graph, `${id}-partials`, 'float32', sumCount * blockCount);
    const pairSums = createTransientView(graph, `${id}-pair-sums`, 'float32', sumCount);

    // Neighbor value g(v_j) and row term f(v_i, lag).
    const neighborValue = bivariate ? 'compactY' : 'compactX';
    const termWGSL = lee
      ? `let permutedNeighbor = getPermutedPosition(neighborPosition, permutation, count);
        lag += weight * compactX[compactXOffset + permutedNeighbor];
        lagY += weight * compactY[compactYOffset + permutedNeighbor];`
      : statistic === 'geary'
        ? `let difference = focus - neighborValue;
        lag += weight * difference * difference;`
        : 'lag += weight * neighborValue;';
    const rowWGSL = lee
      ? 'partialSum += lag * lagY;'
      : statistic === 'geary'
        ? 'partialSum += lag;'
        : 'partialSum += focus * lag;';
    const pairBindings: WGSLKernelBinding[] = [
      {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
      {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
      {name: 'weights', view: effectiveWeights, type: 'f32', access: 'read'},
      {name: 'rowPositions', view: rowPositions, type: 'u32', access: 'read'},
      {name: 'compactX', view: compactX, type: 'f32', access: 'read'},
      ...(bivariate
        ? [{name: 'compactY', view: compactY, type: 'f32' as const, access: 'read' as const}]
        : []),
      {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
      {name: 'partials', view: partials, type: 'f32', access: 'read_write'}
    ];
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-pairs`,
        operation: OPERATION,
        variant: `pairs-${statistic}`,
        bindings: pairBindings,
        workgroupSize: PAIR_WORKGROUP_SIZE,
        invocationCount: sumCount * blockCount * PAIR_WORKGROUP_SIZE,
        guardIndex: false,
        declarations: `const ROWS: u32 = ${rows}u;
const CAPACITY: u32 = ${weights.neighbors.length}u;
const INVALID_POSITION: u32 = ${PERMUTATION_INVALID_POSITION}u;
const BLOCK_ROWS: u32 = ${PAIR_BLOCK_ROWS}u;
const BLOCK_COUNT: u32 = ${blockCount}u;
const WORKGROUP_SIZE: u32 = ${PAIR_WORKGROUP_SIZE}u;
var<workgroup> partialSums: array<f32, ${PAIR_WORKGROUP_SIZE}>;
${getPermutationParameterWGSL(maximumPermutations)}
${PERMUTATION_RANDOM_WGSL}
${PERMUTATION_SHUFFLE_WGSL}
fn getPermutedPosition(position: u32, permutation: u32, count: u32) -> u32 {
  if (permutation == 0u) {
    return position;
  }
  return getSwapOrNotPermutationIndex(position, count);
}`,
        // No early return: every invocation of a workgroup must reach the barriers. A workgroup
        // handles one permutation, so its shuffle rounds are drawn once into workgroup memory.
        body: `let workgroup = index / WORKGROUP_SIZE;
  let permutation = workgroup / BLOCK_COUNT;
  let block = workgroup % BLOCK_COUNT;
  let count = rowPositions[rowPositionsOffset + ROWS];
  let runsPermutation = index < INVOCATION_COUNT && permutation <= readValidPermutationCount();
  if (runsPermutation && permutation > 0u) {
    loadSwapOrNotRounds(readSeedKey(), permutation, count, localInvocationIndex, WORKGROUP_SIZE);
  }
  workgroupBarrier();
  var partialSum = 0.0;
  if (runsPermutation) {
    let rowEnd = min((block + 1u) * BLOCK_ROWS, ROWS);
    for (var row = block * BLOCK_ROWS + localInvocationIndex; row < rowEnd; row += WORKGROUP_SIZE) {
      let position = rowPositions[rowPositionsOffset + row];
      if (position == INVALID_POSITION) {
        continue;
      }
      // Lee's L has no focus term: it is the product of the two lags.
      let focus = ${
        lee
          ? '0.0'
          : `compactX[compactXOffset + ${
              bivariate ? 'position' : 'getPermutedPosition(position, permutation, count)'
            }]`
      };
      var lag = 0.0;
      var lagY = 0.0;
      ${getGPUSpatialWeightsNeighborLoopWGSL(
        'rowPositions[rowPositionsOffset + neighbor] != INVALID_POSITION',
        `let neighborPosition = rowPositions[rowPositionsOffset + neighbor];
        let neighborValue = ${neighborValue}[${neighborValue}Offset +
          getPermutedPosition(neighborPosition, permutation, count)];
        ${termWGSL}`,
        'row'
      )}
      ${rowWGSL}
    }
  }
  partialSums[localInvocationIndex] = partialSum;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partialSums[localInvocationIndex] += partialSums[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (index < INVOCATION_COUNT && localInvocationIndex == 0u) {
    partials[partialsOffset + permutation * BLOCK_COUNT + block] = partialSums[0];
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-pair-sums`,
        operation: OPERATION,
        variant: 'pair-sums',
        bindings: [
          {name: 'partials', view: partials, type: 'f32', access: 'read'},
          {name: 'pairSums', view: pairSums, type: 'f32', access: 'read_write'}
        ],
        invocationCount: sumCount,
        declarations: `const BLOCK_COUNT: u32 = ${blockCount}u;`,
        body: `var sum = 0.0;
  for (var block = 0u; block < BLOCK_COUNT; block++) {
    sum += partials[partialsOffset + index * BLOCK_COUNT + block];
  }
  pairSums[pairSumsOffset + index] = sum;`
      })
    );

    const T = PERMUTATION_TOTAL;
    const scaleWGSL = {
      moran: `count / (totals[totalsOffset + ${T.s0}u] * totals[totalsOffset + ${T.sumSquares}u])`,
      geary: `(count - 1.0) / (2.0 * totals[totalsOffset + ${T.s0}u] * totals[totalsOffset + ${T.sumSquares}u])`,
      getisOrdG: `1.0 / (count * (count - 1.0) * totals[totalsOffset + ${T.mean}u] * totals[totalsOffset + ${T.mean}u] - totals[totalsOffset + ${T.sumSquares}u])`,
      bivariateMoran: `count / (totals[totalsOffset + ${T.s0}u] * sqrt(totals[totalsOffset + ${T.sumSquares}u] * totals[totalsOffset + ${T.sumSquaresY}u]))`,
      // L = z'(W'W)z / 1'(W'W)1 with z standardized by the population deviation: n / sqrt(SS_x SS_y) / S0sq.
      leesL: `count / (totals[totalsOffset + ${T.s0Squared}u] * sqrt(totals[totalsOffset + ${T.sumSquares}u] * totals[totalsOffset + ${T.sumSquaresY}u]))`
    }[statistic];
    const R = GPU_GLOBAL_PERMUTATION_RESULT;
    const bins = props.histogram?.length ?? 0;
    const finalizeBindings: WGSLKernelBinding[] = [
      {name: 'pairSums', view: pairSums, type: 'f32', access: 'read'},
      {name: 'totals', view: totals, type: 'f32', access: 'read'},
      {name: 'rowPositions', view: rowPositions, type: 'u32', access: 'read'},
      {name: 'islandCount', view: islandCount, type: 'u32', access: 'read'},
      {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
      {name: 'results', view: props.results, type: 'f32', access: 'read_write'}
    ];
    if (props.referenceDistribution) {
      finalizeBindings.push({
        name: 'referenceDistribution',
        view: props.referenceDistribution,
        type: 'f32',
        access: 'read_write'
      });
    }
    if (props.histogram) {
      finalizeBindings.push({
        name: 'histogram',
        view: props.histogram,
        type: 'atomic<u32>',
        access: 'read_write'
      });
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-finalize`,
        operation: OPERATION,
        variant: 'finalize',
        bindings: finalizeBindings,
        // One workgroup: the passes over the P pair sums are strided across its 256 threads and
        // combined by fixed binary trees (integer counts, minimum and maximum are exact in any
        // order), instead of one invocation walking P up to 2^20 values several times.
        workgroupSize: FINALIZE_WORKGROUP_SIZE,
        invocationCount: FINALIZE_WORKGROUP_SIZE,
        guardIndex: false,
        declarations: `${getPermutationParameterWGSL(maximumPermutations)}
${PERMUTATION_FLOAT_WGSL}
const ROWS: u32 = ${rows}u;
const BINS: u32 = ${bins}u;
const WORKGROUP_SIZE: u32 = ${FINALIZE_WORKGROUP_SIZE}u;
var<workgroup> scratchFloat: array<f32, ${FINALIZE_WORKGROUP_SIZE}>;
var<workgroup> scratchCount: array<u32, ${FINALIZE_WORKGROUP_SIZE}>;

// Each reduction ends with a barrier so the scratch can be reused; all threads get the result.
fn reduceFloat(local: u32, value: f32, mode: u32) -> f32 {
  scratchFloat[local] = value;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (local < stride) {
      let other = scratchFloat[local + stride];
      scratchFloat[local] = select(select(scratchFloat[local] + other, max(scratchFloat[local], other), mode == 2u), min(scratchFloat[local], other), mode == 1u);
    }
    workgroupBarrier();
  }
  let result = scratchFloat[0];
  workgroupBarrier();
  return result;
}

fn reduceCount(local: u32, value: u32) -> u32 {
  scratchCount[local] = value;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (local < stride) {
      scratchCount[local] += scratchCount[local + stride];
    }
    workgroupBarrier();
  }
  let result = scratchCount[0];
  workgroupBarrier();
  return result;
}`,
        body: `let local = localInvocationIndex;
  // An invalid requested count is never clamped: no permutation is summed and the outputs are NaN.
  let permutations = readValidPermutationCount();
  let count = totals[totalsOffset + ${T.count}u];
  let scale = ${scaleWGSL};
  let observedSum = pairSums[pairSumsOffset];
  let observed = scale * observedSum;
  var greaterPart = 0u;
  var lesserPart = 0u;
  var totalPart = 0.0;
  var minimumPart = 3.0e38;
  var maximumPart = -3.0e38;
  for (var permutation = 1u + local; permutation <= permutations; permutation += WORKGROUP_SIZE) {
    let pairSum = pairSums[pairSumsOffset + permutation];
    let simulated = scale * pairSum;
    greaterPart += select(0u, 1u, pairSum >= observedSum);
    lesserPart += select(0u, 1u, pairSum <= observedSum);
    totalPart += simulated;
    minimumPart = min(minimumPart, simulated);
    maximumPart = max(maximumPart, simulated);
    ${props.referenceDistribution ? 'referenceDistribution[referenceDistributionOffset + permutation - 1u] = simulated;' : ''}
  }
  ${
    props.referenceDistribution
      ? `for (var permutation = permutations + 1u + local; permutation <= MAXIMUM_PERMUTATIONS; permutation += WORKGROUP_SIZE) {
    referenceDistribution[referenceDistributionOffset + permutation - 1u] = getQuietNaN(permutation);
  }`
      : ''
  }
  let greater = reduceCount(local, greaterPart);
  let lesser = reduceCount(local, lesserPart);
  let total = reduceFloat(local, totalPart, 0u);
  let minimum = reduceFloat(local, minimumPart, 1u);
  let maximum = reduceFloat(local, maximumPart, 2u);
  let mean = select(getQuietNaN(2u), total / f32(permutations), permutations > 0u);
  var squaresPart = 0.0;
  for (var permutation = 1u + local; permutation <= permutations; permutation += WORKGROUP_SIZE) {
    let deviation = scale * pairSums[pairSumsOffset + permutation] - mean;
    squaresPart += deviation * deviation;
  }
  let squares = reduceFloat(local, squaresPart, 0u);
  let standardDeviation = select(getQuietNaN(3u), sqrt(squares / f32(permutations)), permutations > 0u);
  ${
    alternative === 'folded'
      ? `var foldedPart = 0u;
  let observedDistance = abs(observed - mean);
  for (var permutation = 1u + local; permutation <= permutations; permutation += WORKGROUP_SIZE) {
    foldedPart += select(0u, 1u, abs(scale * pairSums[pairSumsOffset + permutation] - mean) >= observedDistance);
  }
  let folded = reduceCount(local, foldedPart);`
      : ''
  }
  ${
    props.histogram
      ? `for (var bin = local; bin < BINS; bin += WORKGROUP_SIZE) {
    atomicStore(&histogram[histogramOffset + bin], 0u);
  }
  storageBarrier();
  workgroupBarrier();
  let range = maximum - minimum;
  for (var permutation = 1u + local; permutation <= permutations; permutation += WORKGROUP_SIZE) {
    let simulated = scale * pairSums[pairSumsOffset + permutation];
    var bin = 0u;
    if (range > 0.0 && isFiniteFloat(range)) {
      bin = min(u32(max(floor((simulated - minimum) / range * f32(BINS)), 0.0)), BINS - 1u);
    }
    atomicAdd(&histogram[histogramOffset + bin], 1u);
  }`
      : ''
  }
  if (local == 0u) {
    let larger = ${getExceedanceCountWGSL(alternative)};
    let zSimulated = (observed - mean) / standardDeviation;
    let validZ = standardDeviation > 0.0 && isFiniteFloat(zSimulated);
    // Tested: a valid permutation count and an observed statistic that exists (n >= 2, not truncated).
    let tested = permutations > 0u && isFiniteFloat(observed);
    results[resultsOffset + ${R.observed}u] = select(getQuietNaN(0u), observed, isFiniteFloat(observed));
    results[resultsOffset + ${R.pseudoPValue}u] = select(getQuietNaN(1u), f32(min(${multiplier}u * (larger + 1u), permutations + 1u)) / f32(permutations + 1u), tested);
    results[resultsOffset + ${R.exceedances}u] = select(getQuietNaN(1u), f32(larger), tested);
    results[resultsOffset + ${R.permutations}u] = select(getQuietNaN(1u), f32(permutations), permutations > 0u);
    results[resultsOffset + ${R.simulatedMean}u] = mean;
    results[resultsOffset + ${R.simulatedStandardDeviation}u] = standardDeviation;
    results[resultsOffset + ${R.zSimulated}u] = select(getQuietNaN(1u), zSimulated, validZ && tested);
    results[resultsOffset + ${R.pZSimulated}u] = select(getQuietNaN(1u), 0.5 * getTwoSidedPValue(zSimulated), validZ && tested);
    results[resultsOffset + ${R.minimum}u] = select(getQuietNaN(4u), minimum, permutations > 0u);
    results[resultsOffset + ${R.maximum}u] = select(getQuietNaN(4u), maximum, permutations > 0u);
    results[resultsOffset + ${R.islandCount}u] = f32(islandCount[islandCountOffset]);
    results[resultsOffset + ${R.analysisCount}u] = f32(rowPositions[rowPositionsOffset + ROWS]);
  }`
      })
    );
    return nodes;
  }
}
