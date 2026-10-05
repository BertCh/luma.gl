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
import type {GPUMapGraphRecipe} from '../map-graph-types';
import {createMapGraphKernelNode, type MapGraphKernelBinding} from '../map-graph-kernels';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../map-graph-utils';
import type {GPUSpatialWeights} from '../neighbor-search/spatial-weights';
import {
  getPermutationInputNodes,
  getPermutationParameterWGSL,
  PERMUTATION_FLOAT_WGSL,
  PERMUTATION_INVALID_POSITION,
  PERMUTATION_TOTAL,
  validatePermutationInputs
} from './permutation-inference-kernels';
import {PERMUTATION_RANDOM_WGSL} from './permutation-random';

const OPERATION = 'GPUGlobalPermutationTest';

/** Rows per workgroup block of the per-permutation pair sums. */
const PAIR_BLOCK_ROWS = 2048;
const PAIR_WORKGROUP_SIZE = 256;

/** A global statistic tested by {@link GPUGlobalPermutationTest}. */
export type GPUGlobalPermutationStatistic = 'moran' | 'geary' | 'getisOrdG' | 'bivariateMoran';

/** Fields of the `results` view of {@link GPUGlobalPermutationTest}. */
export const GPU_GLOBAL_PERMUTATION_RESULT = {
  /** Observed statistic. */
  observed: 0,
  /** esda `p_sim`: `(larger + 1) / (P + 1)` with the folded exceedance count. */
  pseudoPValue: 1,
  /** Folded exceedance count `larger` (an exact integer). */
  exceedances: 2,
  /** Permutation count `P` used. */
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
  /** Second variable `y` (permuted) of bivariate Moran's I. Required for `'bivariateMoran'`. */
  secondValues?: GraphDataView<'float32'>;
  /** Optional row selection: nonzero includes the row. */
  mask?: GraphDataView<'uint32'>;
  /** Global statistic. Compile-time. */
  statistic: GPUGlobalPermutationStatistic;
  /**
   * Per-frame uint32 parameters of at least `GPU_PERMUTATION_PARAMETER_LENGTH` elements written with
   * `getGPUPermutationParameterValues`.
   */
  parameters: GraphDataView<'uint32'>;
  /** Upper bound of the per-frame permutation count, at most 2^20. Compile-time; sizes dispatches. */
  maximumPermutations: number;
  /** Caller-owned results, `GPU_GLOBAL_PERMUTATION_RESULT.length` float32 values. */
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
 * G and bivariate Moran's I, with the reference distribution and its histogram.
 *
 * Definitions over the `n` included rows (mask nonzero, finite values; excluded rows are neither
 * foci nor neighbors), with the statistic definitions of `GPUGlobalSpatialStatistics`:
 * - Permutation `p` (1 to `P`) relabels the included values by a keyed pseudorandom bijection of
 *   their compacted positions (a four-round Feistel network keyed by Philox 4x32-10 on the seed and
 *   `p`, with cycle walking): no sort per permutation and no `P x n` memory. Bivariate Moran keeps
 *   `x` and permutes `y`, as esda's `Moran_BV`.
 * - Each statistic is a positive constant times a pair sum `sum_i f(v_i, sum_j w_ij g(v_j))`. The
 *   observed value is the identity permutation run through the same kernel, so `larger =
 *   #{sim >= observed}` compares identically computed f32 pair sums; it is folded to `P - larger`
 *   when that is smaller, and `p_sim = (larger + 1) / (P + 1)`.
 * - Pair sums are reduced in a fixed shape (one workgroup per permutation and row block, a
 *   shared-memory tree, then a fixed sequential sum over blocks). No float atomics: results are
 *   bitwise reproducible for a seed.
 *
 * Cost is `(P + 1)` pair sums, i.e. `(P + 1) * nnz` weighted terms, each evaluating the Feistel
 * bijection for the neighbor.
 */
export class GPUGlobalPermutationTest implements GPUMapGraphRecipe {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Stable recipe name. */
  readonly recipe = 'global-permutation-test';
  /** Validated properties. */
  readonly props: GPUGlobalPermutationTestProps;

  constructor(props: GPUGlobalPermutationTestProps) {
    const id = props.id ?? 'global-permutation-test';
    this.id = id;
    this.props = props;
    validatePermutationInputs({...props, id, operation: OPERATION});
    if (!['moran', 'geary', 'getisOrdG', 'bivariateMoran'].includes(props.statistic)) {
      throw new Error(`${id} unknown statistic ${props.statistic}`);
    }
    if (props.statistic === 'bivariateMoran' && !props.secondValues) {
      throw new Error(`${id} bivariateMoran requires secondValues`);
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
        props.parameters
      ]
    );
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
      props.results,
      props.referenceDistribution,
      props.histogram
    ]);
    const rows = props.values.length;
    const bivariate = statistic === 'bivariateMoran';
    const inputs = getPermutationInputNodes<Parameters>(graph, {
      ...props,
      secondValues: bivariate ? props.secondValues : undefined,
      id: `${id}-inputs`,
      operation: OPERATION,
      centerX: statistic !== 'getisOrdG'
    });
    const nodes = inputs.nodes;
    const {rowPositions, compactX, compactY, totals} = inputs;
    const blockCount = Math.ceil(rows / PAIR_BLOCK_ROWS);
    const sumCount = maximumPermutations + 1;
    const partials = createTransientView(graph, `${id}-partials`, 'float32', sumCount * blockCount);
    const pairSums = createTransientView(graph, `${id}-pair-sums`, 'float32', sumCount);

    // Neighbor value g(v_j) and row term f(v_i, lag).
    const neighborValue = bivariate ? 'compactY' : 'compactX';
    const termWGSL =
      statistic === 'geary'
        ? `let difference = focus - neighborValue;
        lag += weight * difference * difference;`
        : 'lag += weight * neighborValue;';
    const rowWGSL = statistic === 'geary' ? 'partialSum += lag;' : 'partialSum += focus * lag;';
    const pairBindings: MapGraphKernelBinding[] = [
      {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
      {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
      {name: 'weights', view: weights.weights, type: 'f32', access: 'read'},
      {name: 'rowPositions', view: rowPositions, type: 'u32', access: 'read'},
      {name: 'compactX', view: compactX, type: 'f32', access: 'read'},
      ...(bivariate
        ? [{name: 'compactY', view: compactY, type: 'f32' as const, access: 'read' as const}]
        : []),
      {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
      {name: 'partials', view: partials, type: 'f32', access: 'read_write'}
    ];
    nodes.push(
      createMapGraphKernelNode<Parameters>(graph, {
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
fn getPermutedPosition(position: u32, permutation: u32, count: u32, halfBits: u32, roundKeys: vec4<u32>) -> u32 {
  if (permutation == 0u) {
    return position;
  }
  return getFeistelPermutationIndex(position, count, halfBits, roundKeys);
}`,
        // No early return: every invocation of a workgroup must reach the barriers.
        body: `let workgroup = index / WORKGROUP_SIZE;
  let permutation = workgroup / BLOCK_COUNT;
  let block = workgroup % BLOCK_COUNT;
  var partialSum = 0.0;
  if (index < INVOCATION_COUNT && permutation <= readPermutationCount()) {
    let count = rowPositions[rowPositionsOffset + ROWS];
    let halfBits = getFeistelHalfBits(count);
    let roundKeys = getFeistelRoundKeys(readSeedKey(), permutation);
    let rowEnd = min((block + 1u) * BLOCK_ROWS, ROWS);
    for (var row = block * BLOCK_ROWS + localInvocationIndex; row < rowEnd; row += WORKGROUP_SIZE) {
      let position = rowPositions[rowPositionsOffset + row];
      if (position == INVALID_POSITION) {
        continue;
      }
      let focus = compactX[compactXOffset + ${
        bivariate
          ? 'position'
          : 'getPermutedPosition(position, permutation, count, halfBits, roundKeys)'
      }];
      var lag = 0.0;
      let begin = min(offsets[offsetsOffset + row], CAPACITY);
      let end = min(offsets[offsetsOffset + row + 1u], CAPACITY);
      for (var slot = begin; slot < end; slot++) {
        let neighbor = neighbors[neighborsOffset + slot];
        if (neighbor >= ROWS || neighbor == row) {
          continue;
        }
        let neighborPosition = rowPositions[rowPositionsOffset + neighbor];
        if (neighborPosition == INVALID_POSITION) {
          continue;
        }
        let weight = weights[weightsOffset + slot];
        let neighborValue = ${neighborValue}[${neighborValue}Offset +
          getPermutedPosition(neighborPosition, permutation, count, halfBits, roundKeys)];
        ${termWGSL}
      }
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
      createMapGraphKernelNode<Parameters>(graph, {
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
      bivariateMoran: `count / (totals[totalsOffset + ${T.s0}u] * sqrt(totals[totalsOffset + ${T.sumSquares}u] * totals[totalsOffset + ${T.sumSquaresY}u]))`
    }[statistic];
    const R = GPU_GLOBAL_PERMUTATION_RESULT;
    const bins = props.histogram?.length ?? 0;
    const finalizeBindings: MapGraphKernelBinding[] = [
      {name: 'pairSums', view: pairSums, type: 'f32', access: 'read'},
      {name: 'totals', view: totals, type: 'f32', access: 'read'},
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
        type: 'u32',
        access: 'read_write'
      });
    }
    nodes.push(
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-finalize`,
        operation: OPERATION,
        variant: 'finalize',
        bindings: finalizeBindings,
        invocationCount: 1,
        declarations: `${getPermutationParameterWGSL(maximumPermutations)}
${PERMUTATION_FLOAT_WGSL}
const BINS: u32 = ${bins}u;`,
        body: `let permutations = readPermutationCount();
  let count = totals[totalsOffset + ${T.count}u];
  let scale = ${scaleWGSL};
  let observedSum = pairSums[pairSumsOffset];
  let observed = scale * observedSum;
  var larger = 0u;
  var total = 0.0;
  var minimum = 3.0e38;
  var maximum = -3.0e38;
  for (var permutation = 1u; permutation <= permutations; permutation++) {
    let pairSum = pairSums[pairSumsOffset + permutation];
    let simulated = scale * pairSum;
    larger += select(0u, 1u, pairSum >= observedSum);
    total += simulated;
    minimum = min(minimum, simulated);
    maximum = max(maximum, simulated);
    ${props.referenceDistribution ? 'referenceDistribution[referenceDistributionOffset + permutation - 1u] = simulated;' : ''}
  }
  ${
    props.referenceDistribution
      ? `for (var permutation = permutations + 1u; permutation <= MAXIMUM_PERMUTATIONS; permutation++) {
    referenceDistribution[referenceDistributionOffset + permutation - 1u] = getQuietNaN(permutation);
  }`
      : ''
  }
  let mean = total / f32(permutations);
  var squares = 0.0;
  for (var permutation = 1u; permutation <= permutations; permutation++) {
    let deviation = scale * pairSums[pairSumsOffset + permutation] - mean;
    squares += deviation * deviation;
  }
  let standardDeviation = sqrt(squares / f32(permutations));
  if (permutations - larger < larger) {
    larger = permutations - larger;
  }
  let zSimulated = (observed - mean) / standardDeviation;
  let validZ = standardDeviation > 0.0 && isFiniteFloat(zSimulated);
  results[resultsOffset + ${R.observed}u] = select(getQuietNaN(0u), observed, isFiniteFloat(observed));
  results[resultsOffset + ${R.pseudoPValue}u] = f32(larger + 1u) / f32(permutations + 1u);
  results[resultsOffset + ${R.exceedances}u] = f32(larger);
  results[resultsOffset + ${R.permutations}u] = f32(permutations);
  results[resultsOffset + ${R.simulatedMean}u] = mean;
  results[resultsOffset + ${R.simulatedStandardDeviation}u] = standardDeviation;
  results[resultsOffset + ${R.zSimulated}u] = select(getQuietNaN(1u), zSimulated, validZ);
  results[resultsOffset + ${R.pZSimulated}u] = select(getQuietNaN(1u), 0.5 * getTwoSidedPValue(zSimulated), validZ);
  results[resultsOffset + ${R.minimum}u] = minimum;
  results[resultsOffset + ${R.maximum}u] = maximum;
  results[resultsOffset + 10u] = 0.0;
  results[resultsOffset + 11u] = 0.0;
  ${
    props.histogram
      ? `for (var bin = 0u; bin < BINS; bin++) {
    histogram[histogramOffset + bin] = 0u;
  }
  let range = maximum - minimum;
  for (var permutation = 1u; permutation <= permutations; permutation++) {
    let simulated = scale * pairSums[pairSumsOffset + permutation];
    var bin = 0u;
    if (range > 0.0 && isFiniteFloat(range)) {
      bin = min(u32(max(floor((simulated - minimum) / range * f32(BINS)), 0.0)), BINS - 1u);
    }
    histogram[histogramOffset + bin] = histogram[histogramOffset + bin] + 1u;
  }`
      : ''
  }`
      })
    );
    return nodes;
  }
}
