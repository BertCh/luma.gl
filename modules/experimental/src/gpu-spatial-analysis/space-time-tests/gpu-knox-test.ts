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
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {
  type GPUSpatialWeights,
  validateGPUSpatialWeights
} from '../spatial-weights/spatial-weights';
import {
  getSpaceTimeBlocks,
  getSpaceTimeCommonWGSL,
  getSpaceTimeSummaryBodyWGSL,
  SPACE_TIME_MAXIMUM_PERMUTATIONS
} from './space-time-kernels';
import {
  GPU_SPACE_TIME_PARAMETER_LENGTH,
  GPU_SPACE_TIME_SUMMARY_LENGTH
} from './space-time-parameters';

const OPERATION = 'GPUKnoxTest';

/**
 * Properties for {@link GPUKnoxTest}.
 *
 * Per-frame (no recompile): the contents of `times`, `pairs` and `parameters` (seed, permutation
 * count, time threshold). Compile-time: the row count, `maximumPermutations` and the pair
 * capacity.
 */
export type GPUKnoxTestProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'knox-test'`. */
  id?: string;
  /**
   * Pairs close in space, as a self-join {@link GPUSpatialWeights}: row `i` lists the rows `j`
   * within the spatial threshold, for example a radius-mode `GPUNeighborSearch` (any weight
   * function; only the neighbor IDs are read). Each unordered pair is taken from the lower row, as
   * the entry `j > i`, so a symmetric list counts every pair once and an entry whose mirror is
   * missing still counts. Rows are the events.
   */
  pairs: GPUSpatialWeights;
  /** Event time per row, finite float32 (NaN is not supported). */
  times: GraphDataView<'float32'>;
  /**
   * Per-frame parameters: `GPU_SPACE_TIME_PARAMETER_LENGTH` uint32 words written with
   * `getGPUSpaceTimeParameterValues` (seed, permutations, `timeThreshold`).
   */
  parameters: GraphDataView<'uint32'>;
  /** Compile-time upper bound on the per-frame permutation count, 1 to 65536. */
  maximumPermutations: number;
  /**
   * Caller-owned statistics, `maximumPermutations + 1` rows: element 0 is the observed Knox count,
   * elements `1..P` the counts under permuted times (rows past `P` are 0).
   */
  statistics: GraphDataView<'uint32'>;
  /** Caller-owned `GPU_SPACE_TIME_SUMMARY_LENGTH` float32 words, see `GPU_SPACE_TIME_SUMMARY`. */
  summary: GraphDataView<'float32'>;
};

/**
 * Knox space-time interaction test (pointpats `Knox`): the number `X` of event pairs that are close
 * in space (the `pairs` list) and also close in time (`|t_i - t_j| <= timeThreshold`), with the
 * classic expectation and a Monte Carlo reference distribution.
 *
 * Reference distribution (the modified Knox test, Baker 1996, a Monte Carlo test with no Poisson
 * assumption): `P` random permutations of the event times over the event locations, each a keyed
 * Feistel bijection of `[0, n)`, leave the spatial structure fixed, so
 * `E[X] = S T / (n (n - 1) / 2)` for `S` spatial pairs and `T` time-close pairs. The pseudo
 * p-value is `(1 + #{X_perm >= X}) / (P + 1)`. The classic Poisson p-value for `X` with mean `E` is
 * `getKnoxPoissonPValue(observed, expected)` on the CPU.
 *
 * `X` for every permutation is an exact integer: partial counts per (permutation, row block) are
 * summed in fixed order, so results are deterministic and equal a CPU Feistel oracle exactly. `T` is
 * counted exactly in O(n log n) from sorted times (`GPUSort` plus a binary search). The seed makes
 * the whole result a pure function of the inputs. Pair enumeration is the caller's, normally
 * `GPUNeighborSearch` in radius mode (a `GPUGridIndex` lattice); changing the time threshold needs
 * no new pairs.
 *
 * Open: the Kulldorff space-time scan statistic.
 */
export class GPUKnoxTest implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUKnoxTestProps;
  /** Number of events. */
  readonly rows: number;

  constructor(props: GPUKnoxTestProps) {
    const id = props.id ?? 'knox-test';
    this.id = id;
    this.props = props;
    this.rows = validateGPUSpatialWeights(id, props.pairs, 'pairs');
    validatePackedView(props.times, ['float32'], `${id} times`);
    validatePackedUint32View(props.parameters, `${id} parameters`);
    validatePackedUint32View(props.statistics, `${id} statistics`);
    validatePackedView(props.summary, ['float32'], `${id} summary`);
    if (this.rows < 2) {
      throw new Error(`${id} needs at least two rows`);
    }
    if (props.times.length !== this.rows) {
      throw new Error(`${id} times length must equal the pairs row count`);
    }
    if (props.parameters.length < GPU_SPACE_TIME_PARAMETER_LENGTH) {
      throw new Error(`${id} parameters must hold ${GPU_SPACE_TIME_PARAMETER_LENGTH} words`);
    }
    if (
      !Number.isInteger(props.maximumPermutations) ||
      props.maximumPermutations < 1 ||
      props.maximumPermutations > SPACE_TIME_MAXIMUM_PERMUTATIONS
    ) {
      throw new Error(
        `${id} maximumPermutations must be an integer in [1, ${SPACE_TIME_MAXIMUM_PERMUTATIONS}]`
      );
    }
    if (props.statistics.length < props.maximumPermutations + 1) {
      throw new Error(`${id} statistics must hold maximumPermutations + 1 rows`);
    }
    if (props.summary.length < GPU_SPACE_TIME_SUMMARY_LENGTH) {
      throw new Error(`${id} summary must hold ${GPU_SPACE_TIME_SUMMARY_LENGTH} rows`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.statistics, props.summary],
      [
        props.times,
        props.parameters,
        props.pairs.offsets,
        props.pairs.neighbors,
        props.pairs.weights
      ]
    );
    if (props.statistics.buffer === props.summary.buffer) {
      throw new Error(`${id} outputs must use separate buffers`);
    }
  }

  /**
   * Returns the nodes in order: `keys`, the time sort, `clear-close`, `count-close`, `partials`,
   * `reduce` and `summary`.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, rows} = this;
    const {pairs, times, parameters, statistics, summary, maximumPermutations} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      pairs.offsets,
      pairs.neighbors,
      pairs.weights,
      times,
      parameters,
      statistics,
      summary
    ]);
    const [blocks, rowsPerBlock] = getSpaceTimeBlocks(rows);
    const slotCount = maximumPermutations + 1;
    const common = getSpaceTimeCommonWGSL(rows, maximumPermutations, blocks, rowsPerBlock);
    const u32 = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', length);
    const sortKeys = u32('sort-keys', rows);
    const sortIndices = u32('sort-indices', rows);
    const sortedKeys = u32('sorted-keys', rows);
    const sortedIndices = u32('sorted-indices', rows);
    const timeClose = u32('time-close', 2);
    const partials = u32('partials', slotCount * blocks);
    const pairPartials = u32('pair-partials', blocks);
    const orderedKeyWGSL = /* wgsl */ `
fn getOrderedKey(value: f32) -> u32 {
  let bits = bitcast<u32>(value);
  return select(bits | 0x80000000u, ~bits, (bits >> 31u) == 1u);
}
fn getKeyValue(key: u32) -> f32 {
  return bitcast<f32>(select(key ^ 0x80000000u, ~key, (key >> 31u) == 0u));
}`;
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-keys`,
        operation: OPERATION,
        variant: 'keys',
        bindings: [
          {name: 'times', view: times, type: 'f32', access: 'read'},
          {name: 'sortKeys', view: sortKeys, type: 'u32', access: 'read_write'},
          {name: 'sortIndices', view: sortIndices, type: 'u32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations: orderedKeyWGSL,
        body: `sortKeys[sortKeysOffset + index] = getOrderedKey(times[timesOffset + index]);
  sortIndices[sortIndicesOffset + index] = index;`
      }),
      ...new GPUSort({
        id: `${id}-sort`,
        keys: sortKeys,
        values: sortIndices,
        outputKeys: sortedKeys,
        outputValues: sortedIndices
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-clear-close`,
        operation: OPERATION,
        variant: 'clear-close',
        bindings: [{name: 'timeClose', view: timeClose, type: 'u32', access: 'read_write'}],
        invocationCount: 2,
        body: 'timeClose[timeCloseOffset + index] = 0u;'
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-count-close`,
        operation: OPERATION,
        variant: 'count-close',
        bindings: [
          {name: 'sortedKeys', view: sortedKeys, type: 'u32', access: 'read'},
          {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
          {name: 'timeClose', view: timeClose, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations: `${orderedKeyWGSL}
const ROWS: u32 = ${rows}u;`,
        // Sorted ascending, so rows after `index` within the threshold form a prefix.
        body: `let threshold = bitcast<f32>(parameters[parametersOffset + 3u]);
  let time = getKeyValue(sortedKeys[sortedKeysOffset + index]);
  var low = index + 1u;
  var high = ROWS;
  while (low < high) {
    let middle = low + (high - low) / 2u;
    if (abs(getKeyValue(sortedKeys[sortedKeysOffset + middle]) - time) <= threshold) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  let close = low - (index + 1u);
  if (close > 0u) {
    let previous = atomicAdd(&timeClose[timeCloseOffset], close);
    if (previous + close < previous) {
      atomicAdd(&timeClose[timeCloseOffset + 1u], 1u);
    }
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-partials`,
        operation: OPERATION,
        variant: 'partials',
        bindings: [
          {name: 'offsets', view: pairs.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: pairs.neighbors, type: 'u32', access: 'read'},
          {name: 'times', view: times, type: 'f32', access: 'read'},
          {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
          {name: 'partials', view: partials, type: 'u32', access: 'read_write'},
          {name: 'pairPartials', view: pairPartials, type: 'u32', access: 'read_write'}
        ],
        invocationCount: slotCount * blocks,
        declarations: common,
        body: `let slot = index / BLOCKS;
  let block = index - slot * BLOCKS;
  let permutations = readPermutationCount();
  var count = 0u;
  var pairCount = 0u;
  if (slot <= permutations) {
    let keys = getFeistelRoundKeys(readSeedKey(), select(0u, slot - 1u, slot > 0u));
    let halfBits = getFeistelHalfBits(ROWS);
    let threshold = readTimeThreshold();
    let firstRow = block * ROWS_PER_BLOCK;
    let endRow = min(firstRow + ROWS_PER_BLOCK, ROWS);
    for (var row = firstRow; row < endRow; row++) {
      let time = times[timesOffset + getPermutedRow(row, slot, keys, halfBits)];
      for (var entry = offsets[offsetsOffset + row]; entry < offsets[offsetsOffset + row + 1u]; entry++) {
        let other = neighbors[neighborsOffset + entry];
        if (other > row && other < ROWS) {
          pairCount++;
          let otherTime = times[timesOffset + getPermutedRow(other, slot, keys, halfBits)];
          if (abs(time - otherTime) <= threshold) {
            count++;
          }
        }
      }
    }
  }
  partials[partialsOffset + index] = count;
  if (slot == 0u) {
    pairPartials[pairPartialsOffset + block] = pairCount;
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-reduce`,
        operation: OPERATION,
        variant: 'reduce',
        bindings: [
          {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
          {name: 'partials', view: partials, type: 'u32', access: 'read'},
          {name: 'statistics', view: statistics, type: 'u32', access: 'read_write'}
        ],
        invocationCount: statistics.length,
        declarations: common,
        body: `var total = 0u;
  if (index <= readPermutationCount()) {
    for (var block = 0u; block < BLOCKS; block++) {
      total += partials[partialsOffset + index * BLOCKS + block];
    }
  }
  statistics[statisticsOffset + index] = total;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-summary`,
        operation: OPERATION,
        variant: 'summary',
        bindings: [
          {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
          {name: 'statistics', view: statistics, type: 'u32', access: 'read'},
          {name: 'pairPartials', view: pairPartials, type: 'u32', access: 'read'},
          {name: 'timeClose', view: timeClose, type: 'u32', access: 'read'},
          {name: 'summary', view: summary, type: 'f32', access: 'read_write'}
        ],
        invocationCount: 1,
        declarations: common,
        body: `var pairTotal = 0u;
  for (var block = 0u; block < BLOCKS; block++) {
    pairTotal += pairPartials[pairPartialsOffset + block];
  }
  let observedPairs = f32(pairTotal);
  let timeClosePairs = f32(timeClose[timeCloseOffset + 1u]) * 4294967296.0 + f32(timeClose[timeCloseOffset]);
  let allPairs = f32(ROWS) * f32(ROWS - 1u) * 0.5;
  let expected = observedPairs * timeClosePairs / allPairs;
  ${getSpaceTimeSummaryBodyWGSL('u32')}`
      })
    ];
    return nodes;
  }
}
