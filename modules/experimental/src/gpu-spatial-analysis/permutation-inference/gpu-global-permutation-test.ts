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
import {
  createWGSLDispatchGate,
  createWGSLKernelNode,
  type WGSLDispatchGate,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import type {
  GPUInferenceAlternative,
  GPUInferenceMetadata
} from '../../analysis-core/contracts/index';
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
import {getPermutationShuffleWGSL} from './permutation-shuffle';
import {PHILOX_PURPOSES} from '../../utils/philox-random';
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
import type {GPUFloat32OrUint32Values} from '../spatial-weights/float32-values';

const OPERATION = 'GPUGlobalPermutationTest';

/** Threads of the single finalize workgroup. */
const FINALIZE_WORKGROUP_SIZE = 256;

/** Rows per workgroup block of the per-permutation pair sums. */
const PAIR_BLOCK_ROWS = 2048;
const PAIR_WORKGROUP_SIZE = 256;

/** Workgroup (and chunk count) of the in-order scan that finds the stopping index of a batch. */
const SCAN_WORKGROUP_SIZE = 256;

/** Row-block workgroups a sequential-stopping batch aims to occupy the GPU with. */
const SEQUENTIAL_BATCH_TARGET_WORKGROUPS = 128;

/** Most batches a sequential-stopping graph unrolls; larger `maximumPermutations` grow the batch. */
const SEQUENTIAL_MAXIMUM_BATCHES = 64;

/** Words of the sequential-stopping state view (see {@link GPUGlobalPermutationTest}). */
const SEQUENTIAL_STATE = {
  /** Words 0 to 3 copy the per-frame parameters, so kernels read one binding. */
  base: 4,
  greater: 5,
  lesser: 6,
  /** Stopping index `L`, 0 while no stop happened. */
  stoppedAt: 7,
  /** 1 once the run stopped or all permutations ran. */
  done: 8,
  length: 12
} as const;

/**
 * Permutations per batch of a sequential-stopping {@link GPUGlobalPermutationTest}
 * (`stopAfterExceedances`): enough row-block workgroups per batch to occupy the GPU (about 128),
 * at most 64 batches, and never more than `maximumPermutations`. A run computes at most one batch
 * of permutations past its stopping index.
 *
 * @param rows Row count of the test.
 * @param maximumPermutations Compile-time `maximumPermutations` of the test.
 */
export function getGPUGlobalPermutationBatchSize(
  rows: number,
  maximumPermutations: number
): number {
  const blockCount = Math.ceil(rows / PAIR_BLOCK_ROWS);
  return Math.min(
    maximumPermutations,
    Math.max(
      Math.ceil(SEQUENTIAL_BATCH_TARGET_WORKGROUPS / blockCount),
      Math.ceil(maximumPermutations / SEQUENTIAL_MAXIMUM_BATCHES),
      1
    )
  );
}

/**
 * Largest `(maximumPermutations + 1) * ceil(rows / 2048)` of {@link GPUGlobalPermutationTest}: the
 * pair-sum kernel runs one 256-thread workgroup per permutation and row block, so its dispatch of
 * `count * 256` invocations must stay below `2^32`, `floor((2^32 - 1) / 256)`. The partial-sum
 * buffer (`count` float32) then fits one 128 MiB binding. At the largest `maximumPermutations` of
 * 2^20 this allows 15 row blocks (30720 rows). Permutations are not chunked: each batch would
 * need its own sums, shuffle rounds and dispatch, and a constant that depends on the row count
 * keeps the p-value honest about what is computed.
 */
export const GPU_GLOBAL_PERMUTATION_TEST_MAXIMUM_PAIR_BLOCK_COUNT = Math.floor(
  (2 ** 32 - 1) / PAIR_WORKGROUP_SIZE
);

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

/** Fields of the `output.summary` view of {@link GPUGlobalPermutationTest}. */
export const GPU_GLOBAL_PERMUTATION_SUMMARY = {
  /** Observed statistic. */
  observed: 0,
  /** esda `p_sim`: `min(m (M + 1), P + 1) / (P + 1)` for the chosen `alternative` (`m` is 2 for two-sided, else 1). */
  pseudoPValue: 1,
  /** Exceedance count `M` of the chosen `alternative` (an exact integer; folded for 'directed'). */
  exceedances: 2,
  /**
   * Permutation count used: `P`, or the stopping index `L` when `stopAfterExceedances` stopped the run;
   * NaN (and every p-value NaN) when the requested count is outside `[1, maximumPermutations]`.
   */
  permutationCount: 3,
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
  /** Number of float32 values in `output.summary`. */
  length: 12
} as const;

/** Caller-owned outputs of {@link GPUGlobalPermutationTest}. */
export type GPUGlobalPermutationTestOutput = {
  /**
   * Summary row, `GPU_GLOBAL_PERMUTATION_SUMMARY.length` float32 values laid out by
   * `GPU_GLOBAL_PERMUTATION_SUMMARY`, including the island and analysis row counts.
   */
  summary: GraphDataView<'float32'>;
  /**
   * Optional caller-owned reference distribution: the statistic of permutation `p` at index
   * `p - 1`, quiet NaN past `P`. Length `maximumPermutations`.
   */
  referenceDistribution?: GraphDataView<'float32'>;
  /** Optional caller-owned histogram of the reference distribution over `[minimum, maximum]`. */
  histogram?: GraphDataView<'uint32'>;
};

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
  /** Analysis values `x`, one per row, `float32` or `uint32` (converted to f32, exact below 2^24). */
  values: GPUFloat32OrUint32Values;
  /**
   * Second variable `y` of bivariate Moran's I (permuted, `x` stays) or of Lee's L (`x` and `y` are
   * permuted together, as esda `Spatial_Pearson`). Required for `'bivariateMoran'` and `'leesL'`.
   * `float32` or `uint32`.
   */
  secondValues?: GPUFloat32OrUint32Values;
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
  /**
   * Upper bound of the per-frame permutation count, at most 2^20. Compile-time; sizes dispatches.
   * `(maximumPermutations + 1) * ceil(rows / 2048)` is at most
   * {@link GPU_GLOBAL_PERMUTATION_TEST_MAXIMUM_PAIR_BLOCK_COUNT}.
   */
  maximumPermutations: number;
  /**
   * Opt-in Besag-Clifford sequential stopping (Besag and Clifford 1991, Biometrika 78(2)): an
   * integer `h` in `[1, maximumPermutations]`. The permutations are consumed in index order and the
   * run stops at the first `L` whose exceedance count (the count the p-value uses, for the chosen
   * `alternative`) reaches `h`; then `pseudoPValue` is `min(1, m h / L)` with `m` the alternative's
   * multiplier (2 for `'two-sided'`, else 1), `exceedances` is `h` and the `permutationCount` field
   * of `output.summary` is `L`. The reference distribution, its summary statistics, `zSimulated`
   * and the histogram use the first `L` permutations only (`referenceDistribution` is NaN past
   * `L`). A run that never reaches `h` keeps the standard `min(m (M + 1), P + 1) / (P + 1)`, so with
   * `h > P` the result equals the unstopped one bit for bit. `L` does not depend on the batch
   * size: each batch is scanned in permutation order.
   *
   * With it set, the permutations run in fixed batches (see {@link getGPUGlobalPermutationBatchSize}:
   * enough row-block workgroups to occupy the GPU, at most 64 batches) after one pass
   * for the observed statistic; a GPU state word disarms the remaining batches once the run has
   * stopped, so at most one batch of permutations past `L` is computed. Compile-time; omitted
   * (default) runs the single-dispatch path unchanged. Not supported with `alternative:
   * 'folded'`, whose statistic needs the mean of the full run first.
   */
  stopAfterExceedances?: number;
  /** Caller-owned outputs. */
  output: GPUGlobalPermutationTestOutput;
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
 * - Row-standardized weights (`weights.information.transform === 'row'`) are re-standardized over each
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
 * - With `stopAfterExceedances` (Besag-Clifford sequential stopping) the permutations run in
 *   batches and stop once the exceedance count reaches the target; see the prop.
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
    resolveStopAfterExceedances(
      id,
      props.stopAfterExceedances,
      resolvePermutationAlternative(id, props.alternative),
      props.maximumPermutations
    );
    resolveGPUIslandPolicy(id, props.islands);
    if (
      (props.statistic === 'bivariateMoran' || props.statistic === 'leesL') &&
      !props.secondValues
    ) {
      throw new Error(`${id} ${props.statistic} requires secondValues`);
    }
    validateMaximumCount(
      id,
      'permutations x row blocks',
      (props.maximumPermutations + 1) * Math.ceil(props.values.length / PAIR_BLOCK_ROWS),
      GPU_GLOBAL_PERMUTATION_TEST_MAXIMUM_PAIR_BLOCK_COUNT
    );
    const {output} = props;
    validatePackedView(output.summary, ['float32'], `${id} output.summary`);
    validateCapacity(
      id,
      'output.summary length',
      output.summary.length,
      GPU_GLOBAL_PERMUTATION_SUMMARY.length
    );
    if (output.referenceDistribution) {
      validatePackedView(
        output.referenceDistribution,
        ['float32'],
        `${id} output.referenceDistribution`
      );
      if (output.referenceDistribution.length !== props.maximumPermutations) {
        throw new Error(`${id} referenceDistribution length must equal maximumPermutations`);
      }
    }
    if (output.histogram) {
      validatePackedUint32View(output.histogram, `${id} output.histogram`);
      validateCapacity(id, 'output.histogram length', output.histogram.length);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.output.summary, props.output.referenceDistribution, props.output.histogram],
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

  /** Describes how the p-value of a run with `settings` was produced, for legends. */
  getInferenceMetadata(settings: GPUPermutationSettings): GPUInferenceMetadata {
    // Validates the seed and the permutation count.
    this.getParameterValues(settings);
    const keep = resolveGPUIslandPolicy(this.id, this.props.islands) === 'keep';
    return {
      alternative: resolvePermutationAlternative(this.id, this.props.alternative),
      permutationCount: settings.permutationCount,
      seed: settings.seed,
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
      props.output.summary,
      props.output.referenceDistribution,
      props.output.histogram
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
    const stopAfter = props.stopAfterExceedances;
    const sequential = stopAfter !== undefined;
    // Sequential stopping reads the per-frame parameters from a transient copy that also holds the
    // batch cursor, because the pair kernel of a bivariate statistic already binds eight buffers.
    const batchState = sequential
      ? createTransientView(graph, `${id}-batch-state`, 'uint32', SEQUENTIAL_STATE.length)
      : undefined;
    const parametersView = batchState ?? parameters;
    const pairBindings: WGSLKernelBinding[] = [
      {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
      {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
      {name: 'weights', view: effectiveWeights, type: 'f32', access: 'read'},
      {name: 'rowPositions', view: rowPositions, type: 'u32', access: 'read'},
      {name: 'compactX', view: compactX, type: 'f32', access: 'read'},
      ...(bivariate
        ? [{name: 'compactY', view: compactY, type: 'f32' as const, access: 'read' as const}]
        : []),
      {name: 'parameters', view: parametersView, type: 'u32', access: 'read'},
      {name: 'partials', view: partials, type: 'f32', access: 'read_write'}
    ];
    // One pair-sum kernel: `permutationWGSL` is the workgroup's permutation index (0 is the
    // identity, which yields the observed statistic); sequential batches read the batch base from
    // the state word, never from the unrolled batch index (library rule 12).
    const createPairsNode = (options: {
      id: string;
      variant: string;
      invocationCount: number;
      permutationWGSL: string;
      gate?: ReturnType<WGSLDispatchGate<Parameters>['getGate']>;
    }) =>
      createWGSLKernelNode<Parameters>(graph, {
        id: options.id,
        operation: OPERATION,
        variant: options.variant,
        bindings: pairBindings,
        workgroupSize: PAIR_WORKGROUP_SIZE,
        invocationCount: options.invocationCount,
        guardIndex: false,
        ...options.gate,
        declarations: `const ROWS: u32 = ${rows}u;
const CAPACITY: u32 = ${weights.neighbors.length}u;
const INVALID_POSITION: u32 = ${PERMUTATION_INVALID_POSITION}u;
const BLOCK_ROWS: u32 = ${PAIR_BLOCK_ROWS}u;
const BLOCK_COUNT: u32 = ${blockCount}u;
const WORKGROUP_SIZE: u32 = ${PAIR_WORKGROUP_SIZE}u;
var<workgroup> partialSums: array<f32, ${PAIR_WORKGROUP_SIZE}>;
${getPermutationParameterWGSL(maximumPermutations)}
${PERMUTATION_RANDOM_WGSL}
${getPermutationShuffleWGSL(PHILOX_PURPOSES.permutationShuffle.id)}
fn getPermutedPosition(position: u32, permutation: u32, count: u32) -> u32 {
  if (permutation == 0u) {
    return position;
  }
  return getSwapOrNotPermutationIndex(position, count);
}`,
        // No early return: every invocation of a workgroup must reach the barriers. A workgroup
        // handles one permutation, so its shuffle rounds are drawn once into workgroup memory.
        body: `let workgroup = index / WORKGROUP_SIZE;
  let permutation = ${options.permutationWGSL};
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
  if (index < INVOCATION_COUNT && localInvocationIndex == 0u${sequential ? ' && permutation <= MAXIMUM_PERMUTATIONS' : ''}) {
    partials[partialsOffset + permutation * BLOCK_COUNT + block] = partialSums[0];
  }`
      });
    const createPairSumsNode = (options: {
      id: string;
      variant: string;
      invocationCount: number;
      /** WGSL of the permutation index of invocation `index`, with a guard `valid`. */
      permutationWGSL?: string;
      gate?: ReturnType<WGSLDispatchGate<Parameters>['getGate']>;
    }) =>
      createWGSLKernelNode<Parameters>(graph, {
        id: options.id,
        operation: OPERATION,
        variant: options.variant,
        bindings: [
          {name: 'partials', view: partials, type: 'f32', access: 'read'},
          {name: 'pairSums', view: pairSums, type: 'f32', access: 'read_write'},
          ...(options.permutationWGSL
            ? [
                {
                  name: 'parameters',
                  view: parametersView,
                  type: 'u32' as const,
                  access: 'read' as const
                }
              ]
            : [])
        ],
        invocationCount: options.invocationCount,
        ...options.gate,
        declarations: `const BLOCK_COUNT: u32 = ${blockCount}u;${
          options.permutationWGSL ? `\n${getPermutationParameterWGSL(maximumPermutations)}` : ''
        }`,
        body: options.permutationWGSL
          ? `let permutation = ${options.permutationWGSL};
  if (permutation > readValidPermutationCount()) {
    return;
  }
  var sum = 0.0;
  for (var block = 0u; block < BLOCK_COUNT; block++) {
    sum += partials[partialsOffset + permutation * BLOCK_COUNT + block];
  }
  pairSums[pairSumsOffset + permutation] = sum;`
          : `var sum = 0.0;
  for (var block = 0u; block < BLOCK_COUNT; block++) {
    sum += partials[partialsOffset + index * BLOCK_COUNT + block];
  }
  pairSums[pairSumsOffset + index] = sum;`
      });
    if (!sequential || !batchState) {
      nodes.push(
        createPairsNode({
          id: `${id}-pairs`,
          variant: `pairs-${statistic}`,
          invocationCount: sumCount * blockCount * PAIR_WORKGROUP_SIZE,
          permutationWGSL: 'workgroup / BLOCK_COUNT'
        }),
        createPairSumsNode({id: `${id}-pair-sums`, variant: 'pair-sums', invocationCount: sumCount})
      );
    } else {
      nodes.push(
        ...this.getSequentialBatchNodes(graph, {
          batchState,
          createPairsNode,
          createPairSumsNode,
          pairSums,
          blockCount,
          alternative,
          stopAfter
        })
      );
    }

    const T = PERMUTATION_TOTAL;
    const scaleWGSL = {
      moran: `count / (totals[totalsOffset + ${T.s0}u] * totals[totalsOffset + ${T.sumSquares}u])`,
      geary: `(count - 1.0) / (2.0 * totals[totalsOffset + ${T.s0}u] * totals[totalsOffset + ${T.sumSquares}u])`,
      getisOrdG: `1.0 / (count * (count - 1.0) * totals[totalsOffset + ${T.mean}u] * totals[totalsOffset + ${T.mean}u] - totals[totalsOffset + ${T.sumSquares}u])`,
      bivariateMoran: `count / (totals[totalsOffset + ${T.s0}u] * sqrt(totals[totalsOffset + ${T.sumSquares}u] * totals[totalsOffset + ${T.sumSquaresY}u]))`,
      // L = z'(W'W)z / 1'(W'W)1 with z standardized by the population deviation: n / sqrt(SS_x SS_y) / S0sq.
      leesL: `count / (totals[totalsOffset + ${T.s0Squared}u] * sqrt(totals[totalsOffset + ${T.sumSquares}u] * totals[totalsOffset + ${T.sumSquaresY}u]))`
    }[statistic];
    const R = GPU_GLOBAL_PERMUTATION_SUMMARY;
    const bins = props.output.histogram?.length ?? 0;
    const finalizeBindings: WGSLKernelBinding[] = [
      {name: 'pairSums', view: pairSums, type: 'f32', access: 'read'},
      {name: 'totals', view: totals, type: 'f32', access: 'read'},
      {name: 'rowPositions', view: rowPositions, type: 'u32', access: 'read'},
      {name: 'islandCount', view: islandCount, type: 'u32', access: 'read'},
      {name: 'parameters', view: parametersView, type: 'u32', access: 'read'},
      {name: 'results', view: props.output.summary, type: 'f32', access: 'read_write'}
    ];
    if (props.output.referenceDistribution) {
      finalizeBindings.push({
        name: 'referenceDistribution',
        view: props.output.referenceDistribution,
        type: 'f32',
        access: 'read_write'
      });
    }
    if (props.output.histogram) {
      finalizeBindings.push({
        name: 'histogram',
        view: props.output.histogram,
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
  ${
    sequential
      ? `// A stopped run used the first L permutations; the others the requested count.
  let stoppedAt = parameters[parametersOffset + ${SEQUENTIAL_STATE.stoppedAt}u];
  let permutations = select(readValidPermutationCount(), stoppedAt, stoppedAt != 0u);`
      : 'let permutations = readValidPermutationCount();'
  }
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
    ${props.output.referenceDistribution ? 'referenceDistribution[referenceDistributionOffset + permutation - 1u] = simulated;' : ''}
  }
  ${
    props.output.referenceDistribution
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
    props.output.histogram
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
    results[resultsOffset + ${R.pseudoPValue}u] = select(getQuietNaN(1u), ${
      sequential
        ? `select(f32(min(${multiplier}u * (larger + 1u), permutations + 1u)) / f32(permutations + 1u), f32(min(${multiplier}u * ${stopAfter}u, permutations)) / f32(permutations), stoppedAt != 0u)`
        : `f32(min(${multiplier}u * (larger + 1u), permutations + 1u)) / f32(permutations + 1u)`
    }, tested);
    results[resultsOffset + ${R.exceedances}u] = select(getQuietNaN(1u), f32(larger), tested);
    results[resultsOffset + ${R.permutationCount}u] = select(getQuietNaN(1u), f32(permutations), permutations > 0u);
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

  /**
   * Sequential-stopping nodes: a state initializer, the observed pair sum, then fixed-size batches
   * of permutations. Each batch is a gated pair-sum dispatch, a gated reduction over row blocks and
   * an ungated one-workgroup scan that finds the stopping index in permutation order and disarms the
   * remaining batches. The batch base is a GPU state word, so the WGSL of every batch is identical.
   */
  private getSequentialBatchNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    context: {
      batchState: GraphDataView<'uint32'>;
      createPairsNode: (options: {
        id: string;
        variant: string;
        invocationCount: number;
        permutationWGSL: string;
        gate?: ReturnType<WGSLDispatchGate<Parameters>['getGate']>;
      }) => GPUCommandNode<Parameters>;
      createPairSumsNode: (options: {
        id: string;
        variant: string;
        invocationCount: number;
        permutationWGSL?: string;
        gate?: ReturnType<WGSLDispatchGate<Parameters>['getGate']>;
      }) => GPUCommandNode<Parameters>;
      pairSums: GraphDataView<'float32'>;
      blockCount: number;
      alternative: ReturnType<typeof resolvePermutationAlternative>;
      stopAfter: number;
    }
  ): GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {batchState, pairSums, blockCount, alternative, stopAfter} = context;
    const {maximumPermutations, parameters} = props;
    const rows = props.values.length;
    const batchSize = getGPUGlobalPermutationBatchSize(rows, maximumPermutations);
    const batchCount = Math.ceil(maximumPermutations / batchSize);
    const chunkLength = Math.ceil(batchSize / SCAN_WORKGROUP_SIZE);
    const S = SEQUENTIAL_STATE;
    const pairInvocations = batchSize * blockCount * PAIR_WORKGROUP_SIZE;
    const gate = createWGSLDispatchGate<Parameters>(graph, {
      id: `${id}-batch-dispatch`,
      operation: OPERATION,
      shapes: [[pairInvocations, PAIR_WORKGROUP_SIZE], [batchSize]]
    });
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-sequential-init`,
        operation: OPERATION,
        variant: 'sequential-init',
        bindings: [
          {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
          {name: 'state', view: batchState, type: 'u32', access: 'read_write'},
          gate.binding
        ],
        invocationCount: 1,
        declarations: getPermutationParameterWGSL(maximumPermutations),
        // Copies the parameters, clears the cursor and the counts, and arms the batch dispatches
        // unless the requested count is invalid (then nothing runs and the outputs are NaN).
        body: `for (var word = 0u; word < 4u; word++) {
    state[stateOffset + word] = parameters[parametersOffset + word];
  }
  for (var word = ${S.base}u; word < ${S.length}u; word++) {
    state[stateOffset + word] = 0u;
  }
  if (isPermutationCountValid()) {
    ${gate.armWGSL}
  } else {
    state[stateOffset + ${S.done}u] = 1u;
    ${gate.disarmWGSL}
  }`
      }),
      context.createPairsNode({
        id: `${id}-pairs-observed`,
        variant: `pairs-${props.statistic}-observed`,
        invocationCount: blockCount * PAIR_WORKGROUP_SIZE,
        permutationWGSL: '0u'
      }),
      context.createPairSumsNode({
        id: `${id}-pair-sums-observed`,
        variant: 'pair-sums',
        invocationCount: 1
      })
    ];
    const scanDeclarations = `const BATCH_SIZE: u32 = ${batchSize}u;
const CHUNK_LENGTH: u32 = ${chunkLength}u;
const STOP_AFTER_EXCEEDANCES: u32 = ${stopAfter}u;
const WORKGROUP_SIZE: u32 = ${SCAN_WORKGROUP_SIZE}u;
var<workgroup> chunkGreater: array<u32, ${SCAN_WORKGROUP_SIZE}>;
var<workgroup> chunkLesser: array<u32, ${SCAN_WORKGROUP_SIZE}>;
fn getExceedanceCount(greater: u32, lesser: u32) -> u32 {
  return ${getExceedanceCountWGSL(alternative)};
}`;
    for (let batch = 0; batch < batchCount; batch++) {
      const pairsGate = gate.getGate(0);
      const sumsGate = gate.getGate(1);
      nodes.push(
        context.createPairsNode({
          id: `${id}-batch-${batch}-pairs`,
          variant: `pairs-${props.statistic}-batch`,
          invocationCount: pairInvocations,
          permutationWGSL: `parameters[parametersOffset + ${S.base}u] + 1u + workgroup / BLOCK_COUNT`,
          gate: {
            ...pairsGate,
            condition: {...pairsGate.condition, id: `${id}-batch-${batch}-pairs-gate`}
          }
        }),
        context.createPairSumsNode({
          id: `${id}-batch-${batch}-pair-sums`,
          variant: 'pair-sums-batch',
          invocationCount: batchSize,
          permutationWGSL: `parameters[parametersOffset + ${S.base}u] + 1u + index`,
          gate: {
            ...sumsGate,
            condition: {...sumsGate.condition, id: `${id}-batch-${batch}-sums-gate`}
          }
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-batch-${batch}-scan`,
          operation: OPERATION,
          variant: 'sequential-scan',
          bindings: [
            {name: 'pairSums', view: pairSums, type: 'f32', access: 'read'},
            {name: 'state', view: batchState, type: 'u32', access: 'read_write'},
            gate.binding
          ],
          workgroupSize: SCAN_WORKGROUP_SIZE,
          invocationCount: SCAN_WORKGROUP_SIZE,
          guardIndex: false,
          declarations: scanDeclarations,
          // Each thread counts the tail flags of one contiguous chunk of the batch; thread 0 then
          // walks the chunks in permutation order and, inside the first chunk that reaches the
          // target, the permutations in order, so the stopping index does not depend on the batch.
          body: `let local = localInvocationIndex;
  let count = state[stateOffset + 2u];
  let base = state[stateOffset + ${S.base}u];
  let isRunning = state[stateOffset + ${S.done}u] == 0u;
  let observedSum = pairSums[pairSumsOffset];
  var greaterPart = 0u;
  var lesserPart = 0u;
  if (isRunning) {
    for (var item = 0u; item < CHUNK_LENGTH; item++) {
      let offset = local * CHUNK_LENGTH + item;
      let permutation = base + 1u + offset;
      if (offset < BATCH_SIZE && permutation <= count) {
        let pairSum = pairSums[pairSumsOffset + permutation];
        greaterPart += select(0u, 1u, pairSum >= observedSum);
        lesserPart += select(0u, 1u, pairSum <= observedSum);
      }
    }
  }
  chunkGreater[local] = greaterPart;
  chunkLesser[local] = lesserPart;
  workgroupBarrier();
  if (local == 0u && isRunning) {
    var greater = state[stateOffset + ${S.greater}u];
    var lesser = state[stateOffset + ${S.lesser}u];
    var stoppedAt = 0u;
    for (var chunk = 0u; chunk < WORKGROUP_SIZE && stoppedAt == 0u; chunk++) {
      let greaterAfter = greater + chunkGreater[chunk];
      let lesserAfter = lesser + chunkLesser[chunk];
      if (getExceedanceCount(greaterAfter, lesserAfter) >= STOP_AFTER_EXCEEDANCES) {
        for (var item = 0u; item < CHUNK_LENGTH; item++) {
          let offset = chunk * CHUNK_LENGTH + item;
          let permutation = base + 1u + offset;
          if (offset < BATCH_SIZE && permutation <= count) {
            let pairSum = pairSums[pairSumsOffset + permutation];
            greater += select(0u, 1u, pairSum >= observedSum);
            lesser += select(0u, 1u, pairSum <= observedSum);
            if (getExceedanceCount(greater, lesser) >= STOP_AFTER_EXCEEDANCES) {
              stoppedAt = permutation;
              break;
            }
          }
        }
      } else {
        greater = greaterAfter;
        lesser = lesserAfter;
      }
    }
    let nextBase = min(base + BATCH_SIZE, count);
    state[stateOffset + ${S.greater}u] = greater;
    state[stateOffset + ${S.lesser}u] = lesser;
    state[stateOffset + ${S.base}u] = nextBase;
    if (stoppedAt != 0u || nextBase >= count) {
      state[stateOffset + ${S.stoppedAt}u] = stoppedAt;
      state[stateOffset + ${S.done}u] = 1u;
      ${gate.disarmWGSL}
    }
  }`
        })
      );
    }
    return nodes;
  }
}
