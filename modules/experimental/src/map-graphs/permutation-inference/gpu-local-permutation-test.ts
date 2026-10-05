// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUSort,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUMapGraphRecipe} from '../map-graph-types';
import {
  createMapGraphFillNode,
  createMapGraphKernelNode,
  type MapGraphKernelBinding
} from '../map-graph-kernels';
import {getMapGraphSortKeyBits} from '../map-graph-sorted-sums';
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
  validatePermutationInputs
} from './permutation-inference-kernels';
import {PERMUTATION_RANDOM_WGSL} from './permutation-random';

const OPERATION = 'GPULocalPermutationTest';

/** Philox counter word `c` that tags local conditional-permutation draws. @internal */
export const LOCAL_PERMUTATION_RANDOM_TAG = 0x10ca1;

/** `exceedances` value of a row that was not tested. */
export const GPU_LOCAL_PERMUTATION_NOT_TESTED = 0xffffffff;

/** Largest supported compile-time `maximumNeighbors`. */
export const GPU_LOCAL_PERMUTATION_MAXIMUM_NEIGHBORS = 64;

/** A local statistic tested by {@link GPULocalPermutationTest}. */
export type GPULocalPermutationStatistic = 'localMoran' | 'localG' | 'localGStar';

/**
 * Properties for {@link GPULocalPermutationTest}.
 *
 * Per-frame (no rebuild or recompile): the contents of `weights`, `values`, `mask` and
 * `parameters` (seed, permutation count, significance level). Compile-time: the row count, slot
 * capacity, `statistic`, `maximumPermutations`, `maximumNeighbors`, `falseDiscoveryRate` and which
 * optional views exist.
 */
export type GPULocalPermutationTestProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'local-permutation-test'`. */
  id?: string;
  /** Square spatial weights; each row's slot order assigns the drawn values to weights. */
  weights: GPUSpatialWeights;
  /** Analysis values, one per row. Rows with a non-finite value are excluded. */
  values: GraphDataView<'float32'>;
  /** Optional row selection: nonzero includes the row. */
  mask?: GraphDataView<'uint32'>;
  /** Local statistic. Compile-time. */
  statistic: GPULocalPermutationStatistic;
  /**
   * Per-frame uint32 parameters of at least `GPU_PERMUTATION_PARAMETER_LENGTH` elements written with
   * `getGPUPermutationParameterValues`.
   */
  parameters: GraphDataView<'uint32'>;
  /** Upper bound of the per-frame permutation count, at most 2^20. Compile-time. */
  maximumPermutations: number;
  /**
   * Largest neighbor count a tested row may have, in `[1, 64]`. Compile-time. Rows with more
   * included neighbors are not tested and set `overflow`. Defaults to 32.
   */
  maximumNeighbors?: number;
  /** Caller-owned esda-folded exceedance count per row, or `GPU_LOCAL_PERMUTATION_NOT_TESTED`. */
  exceedances: GraphDataView<'uint32'>;
  /** Caller-owned pseudo p-value `(exceedances + 1) / (P + 1)` per row, quiet NaN if not tested. */
  pseudoPValues: GraphDataView<'float32'>;
  /** Optional caller-owned observed local statistic per row (esda scaling), NaN if not tested. */
  observed?: GraphDataView<'float32'>;
  /** Optional caller-owned mask: 1 when the pseudo p-value is significant, else 0. */
  significant?: GraphDataView<'uint32'>;
  /**
   * Compile-time. When true, `significant` uses Benjamini-Hochberg false discovery rate control at
   * the per-frame significance level over the tested rows. Adds one integer sort.
   */
  falseDiscoveryRate?: boolean;
  /** Caller-owned one-row flag: 1 when an included row has more than `maximumNeighbors` neighbors. */
  overflow: GraphDataView<'uint32'>;
};

/**
 * Conditional-permutation (pseudo p-value) inference for local Moran's I and local Getis-Ord
 * G / G*, matching PySAL esda's `Moran_Local` and `G_Local` conditional randomization.
 *
 * Definitions, over the `n` included rows (mask nonzero, finite value):
 * - Neighbors of row `i` are its CSR slots whose neighbor is included and is not `i`; `k_i` is
 *   their count. Rows with `k_i = 0` (islands) or `k_i > maximumNeighbors` are not tested.
 * - Statistics (weights used as given): local Moran `I_i = (n-1) c_i sum_s w_s c_s / sum c^2` with
 *   `c = x - mean`; `G_i = sum_s w_s x_s / (sum x - x_i)`; `G*_i = (x_i + sum_s w_s x_s) / sum x`,
 *   that is, a self weight of 1 (esda's `star=True` with binary weights).
 * - Each permutation draws an ordered sample of `k_i` distinct rows uniformly from the other
 *   `n - 1` included rows (a partial Fisher-Yates shuffle of their compacted positions) and
 *   assigns them to the row's slots in slot order, as esda does. Draws come from Philox 4x32-10
 *   streams keyed by the seed with counter `(block, row, permutation, tag)`, so they do not
 *   depend on the dispatch shape or on other rows.
 * - `larger = #{simulated >= observed}`, folded to `P - larger` when that is smaller, and
 *   `p = (larger + 1) / (P + 1)`: esda's `p_sim`. The observed and simulated terms use the same
 *   f32 expression, and positive constant factors are dropped, so the counts are exact integers,
 *   bitwise reproducible for a seed.
 * - `significant`: `p <= level`, or Benjamini-Hochberg step-up at `level` over the tested rows,
 *   both evaluated as cross-multiplied f32 products of integers (no division), so they are exact
 *   up to one rounding of the level.
 *
 * Cost is `P * sum_i k_i` draws. Each row is one invocation looping over the permutations.
 */
export class GPULocalPermutationTest implements GPUMapGraphRecipe {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Stable recipe name. */
  readonly recipe = 'local-permutation-test';
  /** Validated properties. */
  readonly props: GPULocalPermutationTestProps;

  constructor(props: GPULocalPermutationTestProps) {
    const id = props.id ?? 'local-permutation-test';
    this.id = id;
    this.props = props;
    const rows = validatePermutationInputs({...props, id, operation: OPERATION});
    if (!['localMoran', 'localG', 'localGStar'].includes(props.statistic)) {
      throw new Error(`${id} unknown statistic ${props.statistic}`);
    }
    const maximumNeighbors = props.maximumNeighbors ?? 32;
    if (
      !Number.isInteger(maximumNeighbors) ||
      maximumNeighbors < 1 ||
      maximumNeighbors > GPU_LOCAL_PERMUTATION_MAXIMUM_NEIGHBORS
    ) {
      throw new Error(
        `${id} maximumNeighbors must be an integer in [1, ${GPU_LOCAL_PERMUTATION_MAXIMUM_NEIGHBORS}]`
      );
    }
    validatePackedUint32View(props.exceedances, `${id} exceedances`);
    validatePackedView(props.pseudoPValues, ['float32'], `${id} pseudoPValues`);
    if (props.observed) {
      validatePackedView(props.observed, ['float32'], `${id} observed`);
    }
    if (props.significant) {
      validatePackedUint32View(props.significant, `${id} significant`);
    }
    for (const [name, view] of [
      ['exceedances', props.exceedances],
      ['pseudoPValues', props.pseudoPValues],
      ['observed', props.observed],
      ['significant', props.significant]
    ] as const) {
      if (view && view.length !== rows) {
        throw new Error(`${id} ${name} length must equal the weights row count`);
      }
    }
    if (props.falseDiscoveryRate && !props.significant) {
      throw new Error(`${id} falseDiscoveryRate requires significant`);
    }
    validatePackedUint32View(props.overflow, `${id} overflow`);
    if (props.overflow.length < 1) {
      throw new Error(`${id} overflow must hold one uint32`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.exceedances, props.pseudoPValues, props.observed, props.significant, props.overflow],
      [
        props.weights.offsets,
        props.weights.neighbors,
        props.weights.weights,
        props.values,
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
      props.mask,
      parameters,
      props.exceedances,
      props.pseudoPValues,
      props.observed,
      props.significant,
      props.overflow
    ]);
    const rows = props.values.length;
    const maximumNeighbors = props.maximumNeighbors ?? 32;
    const inputs = getPermutationInputNodes<Parameters>(graph, {
      ...props,
      id: `${id}-inputs`,
      operation: OPERATION,
      centerX: statistic === 'localMoran'
    });
    const nodes = inputs.nodes;
    const {rowPositions, compactX, totals} = inputs;
    const observedRaw = createTransientView(graph, `${id}-observed-raw`, 'float32', rows);
    const constantsWGSL = `const ROWS: u32 = ${rows}u;
const CAPACITY: u32 = ${weights.neighbors.length}u;
const INVALID_POSITION: u32 = ${PERMUTATION_INVALID_POSITION}u;
const MAXIMUM_NEIGHBORS: u32 = ${maximumNeighbors}u;
const NOT_TESTED: u32 = ${GPU_LOCAL_PERMUTATION_NOT_TESTED}u;
${PERMUTATION_FLOAT_WGSL}
// The statistic up to a positive per-row constant; identical for observed and simulated terms.
fn getLocalTerm(focus: f32, lag: f32) -> f32 {
  ${statistic === 'localMoran' ? 'return focus * lag;' : statistic === 'localG' ? 'return lag;' : 'return lag + focus;'}
}`;
    const neighborLoopWGSL = (
      action: string
    ) => `let begin = min(offsets[offsetsOffset + index], CAPACITY);
    let end = min(offsets[offsetsOffset + index + 1u], CAPACITY);
    for (var slot = begin; slot < end; slot++) {
      let neighbor = neighbors[neighborsOffset + slot];
      if (neighbor >= ROWS || neighbor == index) {
        continue;
      }
      let neighborPosition = rowPositions[rowPositionsOffset + neighbor];
      if (neighborPosition == INVALID_POSITION) {
        continue;
      }
      ${action}
    }`;

    nodes.push(
      createMapGraphFillNode<Parameters>(graph, {
        id: `${id}-overflow-clear`,
        operation: OPERATION,
        view: props.overflow,
        type: 'u32',
        value: '0u',
        componentCount: 1
      }),
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-observed`,
        operation: OPERATION,
        variant: 'observed',
        bindings: [
          {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
          {name: 'weights', view: weights.weights, type: 'f32', access: 'read'},
          {name: 'rowPositions', view: rowPositions, type: 'u32', access: 'read'},
          {name: 'compactX', view: compactX, type: 'f32', access: 'read'},
          {name: 'observedRaw', view: observedRaw, type: 'f32', access: 'read_write'},
          {name: 'overflow', view: props.overflow, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations: constantsWGSL,
        body: `var raw = getQuietNaN(index);
  let position = rowPositions[rowPositionsOffset + index];
  if (position != INVALID_POSITION) {
    var lag = 0.0;
    var neighborCount = 0u;
    ${neighborLoopWGSL(`lag += weights[weightsOffset + slot] * compactX[compactXOffset + neighborPosition];
      neighborCount++;`)}
    if (neighborCount > MAXIMUM_NEIGHBORS) {
      atomicMax(&overflow[overflowOffset], 1u);
    } else if (neighborCount > 0u) {
      raw = getLocalTerm(compactX[compactXOffset + position], lag);
    }
  }
  observedRaw[observedRawOffset + index] = raw;`
      }),
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-permute`,
        operation: OPERATION,
        variant: 'permute',
        bindings: [
          {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
          {name: 'weights', view: weights.weights, type: 'f32', access: 'read'},
          {name: 'rowPositions', view: rowPositions, type: 'u32', access: 'read'},
          {name: 'compactX', view: compactX, type: 'f32', access: 'read'},
          {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
          {name: 'observedRaw', view: observedRaw, type: 'f32', access: 'read'},
          {name: 'exceedances', view: props.exceedances, type: 'u32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations: `${constantsWGSL}
${getPermutationParameterWGSL(maximumPermutations)}
${PERMUTATION_RANDOM_WGSL}
const RANDOM_TAG: u32 = ${LOCAL_PERMUTATION_RANDOM_TAG}u;

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
  if (!isFiniteFloat(observed)) {
    exceedances[exceedancesOffset + index] = NOT_TESTED;
    return;
  }
  let position = rowPositions[rowPositionsOffset + index];
  let others = rowPositions[rowPositionsOffset + ROWS] - 1u;
  let focus = compactX[compactXOffset + position];
  let permutations = readPermutationCount();
  let key = readSeedKey();
  var larger = 0u;
  for (var permutation = 0u; permutation < permutations; permutation++) {
    var stream = createPhiloxStream(key, index, permutation, RANDOM_TAG);
    var swapPositions: array<u32, ${maximumNeighbors}>;
    var swapValues: array<u32, ${maximumNeighbors}>;
    var swapCount = 0u;
    var drawn = 0u;
    var lag = 0.0;
    ${neighborLoopWGSL(`// Partial Fisher-Yates over the other included rows' virtual positions [0, others).
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
      drawn++;
      let drawnPosition = chosen + select(0u, 1u, chosen >= position);
      lag += weights[weightsOffset + slot] * compactX[compactXOffset + drawnPosition];`)}
    larger += select(0u, 1u, getLocalTerm(focus, lag) >= observed);
  }
  if (permutations - larger < larger) {
    larger = permutations - larger;
  }
  exceedances[exceedancesOffset + index] = larger;`
      })
    );

    if (props.observed) {
      const observedWGSL =
        statistic === 'localMoran'
          ? '(totals[totalsOffset] - 1.0) * raw / totals[totalsOffset + 3u]'
          : statistic === 'localG'
            ? 'raw / (totals[totalsOffset + 1u] - compactX[compactXOffset + position])'
            : 'raw / totals[totalsOffset + 1u]';
      nodes.push(
        createMapGraphKernelNode<Parameters>(graph, {
          id: `${id}-observed-statistic`,
          operation: OPERATION,
          variant: 'observed-statistic',
          bindings: [
            {name: 'observedRaw', view: observedRaw, type: 'f32', access: 'read'},
            {name: 'rowPositions', view: rowPositions, type: 'u32', access: 'read'},
            {name: 'compactX', view: compactX, type: 'f32', access: 'read'},
            {name: 'totals', view: totals, type: 'f32', access: 'read'},
            {name: 'observed', view: props.observed, type: 'f32', access: 'read_write'}
          ],
          invocationCount: rows,
          declarations: constantsWGSL,
          body: `let raw = observedRaw[observedRawOffset + index];
  let position = min(rowPositions[rowPositionsOffset + index], ROWS - 1u);
  observed[observedOffset + index] = select(getQuietNaN(index), ${observedWGSL}, isFiniteFloat(raw));`
        })
      );
    }

    let ranks: GraphDataView<'uint32'> | undefined;
    let counters: GraphDataView<'uint32'> | undefined;
    if (props.falseDiscoveryRate) {
      const notTestedKey = maximumPermutations + 1;
      const keys = createTransientView(graph, `${id}-fdr-keys`, 'uint32', rows);
      const rowIds = createTransientView(graph, `${id}-fdr-row-ids`, 'uint32', rows);
      const sortedKeys = createTransientView(graph, `${id}-fdr-sorted-keys`, 'uint32', rows);
      const sortedRows = createTransientView(graph, `${id}-fdr-sorted-rows`, 'uint32', rows);
      ranks = createTransientView(graph, `${id}-fdr-ranks`, 'uint32', rows);
      counters = createTransientView(graph, `${id}-fdr-counters`, 'uint32', 2);
      nodes.push(
        createMapGraphKernelNode<Parameters>(graph, {
          id: `${id}-fdr-keys`,
          operation: OPERATION,
          variant: 'fdr-keys',
          bindings: [
            {name: 'exceedances', view: props.exceedances, type: 'u32', access: 'read'},
            {name: 'keys', view: keys, type: 'u32', access: 'read_write'},
            {name: 'rowIds', view: rowIds, type: 'u32', access: 'read_write'}
          ],
          invocationCount: rows,
          declarations: `const NOT_TESTED: u32 = ${GPU_LOCAL_PERMUTATION_NOT_TESTED}u;
const NOT_TESTED_KEY: u32 = ${notTestedKey}u;`,
          // The pseudo p-value is monotone in the count, so sorting counts sorts p ascending.
          body: `let count = exceedances[exceedancesOffset + index];
  keys[keysOffset + index] = select(count, NOT_TESTED_KEY, count == NOT_TESTED);
  rowIds[rowIdsOffset + index] = index;`
        }),
        ...new GPUSort({
          id: `${id}-fdr-sort`,
          keys,
          values: rowIds,
          outputKeys: sortedKeys,
          outputValues: sortedRows,
          keyBits: getMapGraphSortKeyBits(notTestedKey)
        }).getCommandNodes(graph),
        createMapGraphFillNode<Parameters>(graph, {
          id: `${id}-fdr-clear`,
          operation: OPERATION,
          view: counters,
          type: 'u32',
          value: '0u'
        }),
        createMapGraphKernelNode<Parameters>(graph, {
          id: `${id}-fdr-ranks`,
          operation: OPERATION,
          variant: 'fdr-ranks',
          bindings: [
            {name: 'sortedKeys', view: sortedKeys, type: 'u32', access: 'read'},
            {name: 'sortedRows', view: sortedRows, type: 'u32', access: 'read'},
            {name: 'ranks', view: ranks, type: 'u32', access: 'read_write'},
            {name: 'counters', view: counters, type: 'atomic<u32>', access: 'read_write'}
          ],
          invocationCount: rows,
          declarations: `const NOT_TESTED_KEY: u32 = ${notTestedKey}u;`,
          body: `let tested = sortedKeys[sortedKeysOffset + index] != NOT_TESTED_KEY;
  ranks[ranksOffset + sortedRows[sortedRowsOffset + index]] = select(0u, index + 1u, tested);
  if (tested) {
    atomicMax(&counters[countersOffset], index + 1u);
  }`
        }),
        createMapGraphKernelNode<Parameters>(graph, {
          id: `${id}-fdr-threshold`,
          operation: OPERATION,
          variant: 'fdr-threshold',
          bindings: [
            {name: 'sortedKeys', view: sortedKeys, type: 'u32', access: 'read'},
            {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
            {name: 'counters', view: counters, type: 'atomic<u32>', access: 'read_write'}
          ],
          invocationCount: rows,
          declarations: getPermutationParameterWGSL(maximumPermutations),
          body: `let tested = atomicLoad(&counters[countersOffset]);
  let rank = index + 1u;
  if (rank <= tested) {
    // p_(k) <= k level / m with p = (count + 1) / (P + 1), cross-multiplied so that only
    // correctly rounded f32 products are compared.
    let count = sortedKeys[sortedKeysOffset + index];
    if (f32(count + 1u) * f32(tested) <= f32(rank) * readSignificanceLevel() * f32(readPermutationCount() + 1u)) {
      atomicMax(&counters[countersOffset + 1u], rank);
    }
  }`
        })
      );
    }

    const classifyBindings: MapGraphKernelBinding[] = [
      {name: 'exceedances', view: props.exceedances, type: 'u32', access: 'read'},
      {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
      {name: 'pseudoPValues', view: props.pseudoPValues, type: 'f32', access: 'read_write'}
    ];
    if (props.significant) {
      classifyBindings.push({
        name: 'significant',
        view: props.significant,
        type: 'u32',
        access: 'read_write'
      });
    }
    if (ranks && counters) {
      classifyBindings.push(
        {name: 'ranks', view: ranks, type: 'u32', access: 'read'},
        {name: 'counters', view: counters, type: 'u32', access: 'read'}
      );
    }
    const significantWGSL = ranks
      ? `let rank = ranks[ranksOffset + index];
  significant[significantOffset + index] = select(0u, 1u, tested && rank > 0u && rank <= counters[countersOffset + 1u]);`
      : 'significant[significantOffset + index] = select(0u, 1u, tested && f32(count + 1u) <= readSignificanceLevel() * f32(readPermutationCount() + 1u));';
    nodes.push(
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-classify`,
        operation: OPERATION,
        variant: ranks ? 'classify-fdr' : 'classify',
        bindings: classifyBindings,
        invocationCount: rows,
        declarations: `${getPermutationParameterWGSL(maximumPermutations)}
const NOT_TESTED: u32 = ${GPU_LOCAL_PERMUTATION_NOT_TESTED}u;`,
        body: `let count = exceedances[exceedancesOffset + index];
  let tested = count != NOT_TESTED;
  let pValue = f32(count + 1u) / f32(readPermutationCount() + 1u);
  pseudoPValues[pseudoPValuesOffset + index] = select(bitcast<f32>(0x7fc00000u | (index & 0u)), pValue, tested);
  ${props.significant ? significantWGSL : ''}`
      })
    );
    return nodes;
  }
}
