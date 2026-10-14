// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {PHILOX_PURPOSES} from '../../utils/philox-random';
import {
  createTransientView,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {
  type GPUSpatialWeights,
  validateGPUSpatialWeights
} from '../spatial-weights/spatial-weights';
import {
  createSpaceTimePairMaskNode,
  getSpaceTimeBlocks,
  getSpaceTimeCommonWGSL,
  getSpaceTimeGroupsPerSlot,
  getSpaceTimeShuffleWGSL,
  getSpaceTimeSummaryBodyWGSL,
  SPACE_TIME_BATCH_PROLOGUE_WGSL,
  SPACE_TIME_MAXIMUM_PERMUTATIONS,
  SPACE_TIME_MAXIMUM_ROWS,
  SPACE_TIME_SLOT_PROLOGUE_WGSL,
  SPACE_TIME_WORKGROUP_SIZE
} from './space-time-kernels';
import {
  getGPUSpaceTimeInferenceMetadata,
  GPU_SPACE_TIME_PARAMETER_LENGTH,
  GPU_SPACE_TIME_SUMMARY_LENGTH,
  type GPUSpaceTimeSettings
} from './space-time-parameters';
import {
  createSpaceTimeStateView,
  createSpaceTimeStoppingNodes,
  getSpaceTimeStoppingBatchSize,
  type SpaceTimeBatchNodeProps,
  validateSpaceTimeStopping
} from './space-time-stopping';
import type {GPUInferenceMetadata} from '../../analysis-core/contracts/index';
import {DECODE_ORDERED_KEY_WGSL, GET_ORDERED_KEY_WGSL} from '../../utils/wgsl-common';

const OPERATION = 'GPUKnoxTest';

/** Caller-owned outputs of {@link GPUKnoxTest}. */
export type GPUKnoxTestOutput = {
  /**
   * Caller-owned statistics, `maximumPermutations + 1` rows: element 0 is the observed Knox count,
   * elements `1..P` the counts under permuted times (rows past `P` are 0).
   */
  statistics: GraphDataView<'uint32'>;
  /** Caller-owned `GPU_SPACE_TIME_SUMMARY_LENGTH` float32 words, see `GPU_SPACE_TIME_SUMMARY`. */
  summary: GraphDataView<'float32'>;
  /**
   * One uint32 word: the number of permutations used, `L` when `stopAfterExceedances` stopped the
   * run and the permutation count `P` otherwise. Required with `stopAfterExceedances`, rejected
   * without it.
   */
  permutationCount?: GraphDataView<'uint32'>;
};

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
   * function; only the neighbor IDs are read). Every unordered pair that appears in either
   * direction is counted exactly once: a symmetric list takes the pair from its lower row, and a
   * pair listed only from the higher row (a k-nearest-neighbor list, or a radius list with
   * overflow-truncated rows) is taken from that row. Duplicate entries and self entries are
   * ignored. Lists need not be sorted. Rows are the events.
   */
  pairs: GPUSpatialWeights;
  /**
   * Event time per row, finite float32 (NaN is not supported), measured from a nearby origin in
   * the unit of `timeThreshold`. Raw epoch seconds do not fit float32 (the spacing is 128 s near
   * 1.7e9), so subtract an origin on the CPU with `getGPUSpaceTimeRelativeTimes`.
   */
  times: GraphDataView<'float32'>;
  /**
   * Per-frame parameters: `GPU_SPACE_TIME_PARAMETER_LENGTH` uint32 words written with
   * `getGPUSpaceTimeParameterValues` (seed, permutation count, `timeThreshold`).
   */
  parameters: GraphDataView<'uint32'>;
  /** Compile-time upper bound on the per-frame permutation count, 1 to 65536. */
  maximumPermutations: number;
  /**
   * Compile-time Besag-Clifford sequential stopping (Besag and Clifford 1991): an integer `h` in
   * `[1, maximumPermutations]`. Omitted (the default) runs all `P` permutations, unchanged. With
   * `h`, the run stops at the first permutation `L` whose running count of permuted counts `>=`
   * the observed count equals `h`, and `pseudoPGreater = min(1, h / L)`; when fewer than `h`
   * exceed in all `P`, it is the usual `(greaterCount + 1) / (P + 1)`. Permutations are evaluated
   * in batches of 64 or more (the batch size grows so there are at most 256 batches), each batch
   * scanned in permutation order on the GPU, so `L` and the p-value do not depend on the batch
   * size, and a stopped run skips the remaining batches. The first `L` permutations equal those
   * of the full run. `statistics` rows past `L` are 0 and the `summary` mean, variance, counts and
   * z-score cover the first `L` permutations; `pseudoPLesser` and `pseudoPTwoSided` keep their
   * fixed-`L` formulas and are not sequentially valid. Requires `output.permutationCount`, which
   * receives `L`.
   */
  stopAfterExceedances?: number;
  /** Caller-owned outputs. */
  output: GPUKnoxTestOutput;
};

/**
 * Knox space-time interaction test (pointpats `Knox`): the number `X` of event pairs that are close
 * in space (the `pairs` list) and also close in time (`|t_i - t_j| <= timeThreshold`), with the
 * classic expectation and a Monte Carlo reference distribution.
 *
 * Reference distribution (the modified Knox test, Baker 1996, a Monte Carlo test with no Poisson
 * assumption): `P` random permutations of the event times over the event locations, each a keyed
 * swap-or-not shuffle (Hoang-Morris-Rogaway) of `[0, n)`, leave the spatial structure fixed, so
 * `E[X] = S T / (n (n - 1) / 2)` for `S` spatial pairs and `T` time-close pairs. The pseudo
 * p-value is `(1 + #{X_perm >= X}) / (P + 1)`. The classic Poisson p-value for `X` with mean `E` is
 * `getKnoxPoissonPValue(observed, expected)` on the CPU.
 *
 * `X` for every permutation is an exact integer: partial counts per (permutation, row block) are
 * summed in fixed order, so results are deterministic and equal the CPU swap-or-not oracle (`fillShufflePermutation`) exactly. `T` is
 * counted exactly in O(n log n) from sorted times (`GPUSort` plus a binary search). The seed makes
 * the whole result a pure function of the inputs. Pair enumeration is the caller's, normally
 * `GPUNeighborSearch` in radius mode (a `GPUGridIndex` lattice); changing the time threshold needs
 * no new pairs.
 *
 * Alternative: Knox's hypothesis is one-sided, an excess of pairs that are close in both space and
 * time, so the Knox p-value is `pseudoPGreater` in the `summary`. `pseudoPTwoSided` and
 * `pseudoPLesser` are reported for completeness (a deficit is space-time repulsion, rarely the
 * question). {@link GPUKnoxTest.getInferenceMetadata} states this on the CPU.
 *
 * Poisson caveat: `getKnoxPoissonPValue` assumes the classic Poisson variance. The true variance of
 * the Knox count is larger when events share locations or times (Mantel-Kulldorff variance
 * inflation), so the Poisson p-value is anti-conservative; the Monte Carlo p-value
 * (`pseudoPGreater`) is preferred. Ties are not handled by the Poisson helper.
 *
 * Breaking change: the permutations used to be a Feistel bijection, which is measurably non-uniform
 * on small domains. The swap-or-not shuffle is uniform, so the permuted counts (and the p-value for
 * a given seed) differ from earlier versions. Permutation `s >= 1` is draw `s` of the seed. Fewer
 * than 2^24 events are supported.
 *
 * Time units: pass times relative to an origin (`getGPUSpaceTimeRelativeTimes`); `timeThreshold` is
 * in the same units. float32 times resolve about `span * 2^-24`.
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
    validatePackedView(props.parameters, ['uint32'], `${id} parameters`);
    validatePackedView(props.output.statistics, ['uint32'], `${id} output.statistics`);
    validatePackedView(props.output.summary, ['float32'], `${id} output.summary`);
    if (this.rows < 2) {
      throw new Error(`${id} needs at least two rows`);
    }
    if (this.rows > SPACE_TIME_MAXIMUM_ROWS) {
      // The swap-or-not shuffle has round capacity for domains below 2^24 rows.
      throw new Error(`${id} supports at most ${SPACE_TIME_MAXIMUM_ROWS} rows`);
    }
    if (props.times.length !== this.rows) {
      throw new Error(`${id} times needs one row per pairs row`);
    }
    if (props.parameters.length < GPU_SPACE_TIME_PARAMETER_LENGTH) {
      throw new Error(`${id} parameters needs ${GPU_SPACE_TIME_PARAMETER_LENGTH} words`);
    }
    validateCapacity(
      id,
      'maximumPermutations',
      props.maximumPermutations,
      1,
      SPACE_TIME_MAXIMUM_PERMUTATIONS
    );
    if (props.output.statistics.length < props.maximumPermutations + 1) {
      throw new Error(`${id} output.statistics needs maximumPermutations + 1 rows`);
    }
    validateSpaceTimeStopping(
      id,
      props.stopAfterExceedances,
      props.maximumPermutations,
      props.output.permutationCount
    );
    if (props.output.summary.length < GPU_SPACE_TIME_SUMMARY_LENGTH) {
      throw new Error(`${id} output.summary needs ${GPU_SPACE_TIME_SUMMARY_LENGTH} rows`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.output.statistics, props.output.summary, props.output.permutationCount],
      [
        props.times,
        props.parameters,
        props.pairs.offsets,
        props.pairs.neighbors,
        props.pairs.weights
      ]
    );
    if (
      props.output.statistics.buffer === props.output.summary.buffer ||
      (props.output.permutationCount &&
        (props.output.permutationCount.buffer === props.output.statistics.buffer ||
          props.output.permutationCount.buffer === props.output.summary.buffer))
    ) {
      throw new Error(`${id} outputs must use separate buffers`);
    }
  }

  /**
   * CPU-side description of how the `output.summary` p-values are produced, for building a
   * legend: alternative `'greater'`, Monte Carlo, no multiple-testing correction.
   *
   * @param settings The per-frame settings packed into `parameters` (validated as
   * `getGPUSpaceTimeParameterValues` does; the count is clamped to `maximumPermutations`).
   */
  getInferenceMetadata(settings: GPUSpaceTimeSettings): GPUInferenceMetadata {
    return getGPUSpaceTimeInferenceMetadata('knox', settings, this.props.maximumPermutations);
  }

  /**
   * Returns the nodes in order: `keys`, the time sort, `clear-close`, `count-close`, `pair-mask`, `partials`,
   * `reduce` and `summary`.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, rows} = this;
    const {pairs, times, parameters, maximumPermutations} = props;
    const {statistics, summary} = props.output;
    validateGraphViewsBelongToGraph(id, graph, [
      pairs.offsets,
      pairs.neighbors,
      pairs.weights,
      times,
      parameters,
      statistics,
      summary,
      ...(props.output.permutationCount ? [props.output.permutationCount] : [])
    ]);
    const [blocks, rowsPerBlock] = getSpaceTimeBlocks(rows);
    const slotCount = maximumPermutations + 1;
    const groupsPerSlot = getSpaceTimeGroupsPerSlot(blocks);
    const common = getSpaceTimeCommonWGSL(rows, maximumPermutations, blocks, rowsPerBlock);
    const u32 = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', length);
    const stopAfterExceedances = props.stopAfterExceedances;
    const stopping = stopAfterExceedances !== undefined;
    // Stopping evaluates the slots in batches, so the (slot, block) table holds one batch.
    const batchSize = getSpaceTimeStoppingBatchSize(slotCount);
    const tableSlots = stopping ? batchSize : slotCount;
    const state = stopping ? createSpaceTimeStateView(graph, id) : undefined;
    const stateBindings = state
      ? [{name: 'state', view: state, type: 'u32' as const, access: 'read' as const}]
      : [];
    const batchDeclarations = stopping ? `\nconst BATCH_SIZE: u32 = ${batchSize}u;` : '';
    const sortKeys = u32('sort-keys', rows);
    const sortIndices = u32('sort-indices', rows);
    const sortedKeys = u32('sorted-keys', rows);
    const sortedIndices = u32('sorted-indices', rows);
    const timeClose = u32('time-close', 2);
    const partials = u32('partials', tableSlots * blocks);
    const pairPartials = u32('pair-partials', blocks);
    const pairMask = u32('pair-mask', pairs.neighbors.length);
    const orderedKeyWGSL = /* wgsl */ `
${GET_ORDERED_KEY_WGSL}${DECODE_ORDERED_KEY_WGSL}`;
    const createPartialsNode = (batch?: SpaceTimeBatchNodeProps<Parameters>) =>
      createWGSLKernelNode<Parameters>(graph, {
        id: batch?.id ?? `${id}-partials`,
        operation: OPERATION,
        variant: 'partials',
        bindings: [
          {name: 'offsets', view: pairs.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: pairs.neighbors, type: 'u32', access: 'read'},
          {name: 'pairMask', view: pairMask, type: 'u32', access: 'read'},
          {name: 'times', view: times, type: 'f32', access: 'read'},
          {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
          ...stateBindings,
          {name: 'partials', view: partials, type: 'u32', access: 'read_write'},
          {name: 'pairPartials', view: pairPartials, type: 'u32', access: 'read_write'}
        ],
        condition: batch?.condition,
        extraResources: batch?.extraResources,
        workgroupSize: SPACE_TIME_WORKGROUP_SIZE,
        invocationCount: tableSlots * groupsPerSlot * SPACE_TIME_WORKGROUP_SIZE,
        guardIndex: false,
        declarations: `${common}
${getSpaceTimeShuffleWGSL(blocks, PHILOX_PURPOSES.knoxTestShuffle.id)}
const SLOT_COUNT: u32 = ${tableSlots}u;${batchDeclarations}`,
        // A workgroup handles one permutation, so its shuffle rounds are drawn once into workgroup
        // memory; each invocation owns one row block. Rows are interleaved across blocks
        // (row = block + k * BLOCKS) to even out skewed row lengths. No early return: every
        // invocation must reach the barrier. With stopping, the slot is the batch cursor (a GPU
        // word) times the batch size plus the workgroup's slot in the batch.
        body: `${stopping ? SPACE_TIME_BATCH_PROLOGUE_WGSL : SPACE_TIME_SLOT_PROLOGUE_WGSL}
  var count = 0u;
  var pairCount = 0u;
  if (runsSlot) {
    let threshold = readTimeThreshold();
    for (var row = block; row < ROWS; row += BLOCKS) {
      let time = times[timesOffset + getPermutedRow(row, slot)];
      for (var entry = offsets[offsetsOffset + row]; entry < offsets[offsetsOffset + row + 1u]; entry++) {
        let other = neighbors[neighborsOffset + entry];
        if (pairMask[pairMaskOffset + entry] != 0u) {
          pairCount++;
          let otherTime = times[timesOffset + getPermutedRow(other, slot)];
          if (abs(time - otherTime) <= threshold) {
            count++;
          }
        }
      }
    }
  }
  if (block < BLOCKS) {
    partials[partialsOffset + block * SLOT_COUNT + ${stopping ? 'localSlot' : 'slot'}] = count;
  }
  if (slot == 0u && block < BLOCKS) {
    pairPartials[pairPartialsOffset + block] = pairCount;
  }`
      });
    const createReduceNode = (batch?: SpaceTimeBatchNodeProps<Parameters>) =>
      createWGSLKernelNode<Parameters>(graph, {
        id: batch?.id ?? `${id}-reduce`,
        operation: OPERATION,
        variant: 'reduce',
        bindings: [
          {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
          ...stateBindings,
          {name: 'partials', view: partials, type: 'u32', access: 'read'},
          {name: 'statistics', view: statistics, type: 'u32', access: 'read_write'}
        ],
        condition: batch?.condition,
        extraResources: batch?.extraResources,
        // Stopping reduces the slots of one batch; the rest of the kernel is the full-run one.
        ...(stopping
          ? {workgroupSize: 64, invocationCount: batchSize}
          : {invocationCount: statistics.length}),
        declarations: `${common}
const SLOT_COUNT: u32 = ${tableSlots}u;${batchDeclarations}`,
        // Adjacent invocations read adjacent slots of one block: coalesced.
        body: stopping
          ? `let slot = state[stateOffset] * BATCH_SIZE + index;
  if (slot <= readPermutationCount()) {
    var total = 0u;
    for (var block = 0u; block < BLOCKS; block++) {
      total += partials[partialsOffset + block * SLOT_COUNT + index];
    }
    statistics[statisticsOffset + slot] = total;
  }`
          : `var total = 0u;
  if (index <= readPermutationCount()) {
    for (var block = 0u; block < BLOCKS; block++) {
      total += partials[partialsOffset + block * SLOT_COUNT + index];
    }
  }
  statistics[statisticsOffset + index] = total;`
      });
    const stoppingNodes =
      stopping && state
        ? createSpaceTimeStoppingNodes<Parameters>(graph, {
            id,
            operation: OPERATION,
            stopAfterExceedances,
            slotCount,
            groupsPerSlot,
            workgroupSize: SPACE_TIME_WORKGROUP_SIZE,
            statisticType: 'u32',
            parameters,
            statistics,
            state,
            common,
            createPartialsNode,
            createReduceNode
          })
        : [];
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
        workgroupSize: 256,
        guardIndex: false,
        // Each workgroup sums its lanes' counts into a workgroup 64-bit (low word, carry count)
        // pair, then lane 0 adds that pair to the global 64-bit total: one global atomic pair per
        // workgroup instead of one per row. Totals stay exact.
        declarations: `${orderedKeyWGSL}
const ROWS: u32 = ${rows}u;
var<workgroup> groupLow: atomic<u32>;
var<workgroup> groupCarry: atomic<u32>;`,
        // Sorted ascending, so rows after `index` within the threshold form a prefix. No early
        // return: every invocation must reach the barrier.
        body: `if (index < ROWS) {
    let threshold = bitcast<f32>(parameters[parametersOffset + 3u]);
    let time = decodeOrderedKey(sortedKeys[sortedKeysOffset + index]);
    var low = index + 1u;
    var high = ROWS;
    while (low < high) {
      let middle = low + (high - low) / 2u;
      if (abs(decodeOrderedKey(sortedKeys[sortedKeysOffset + middle]) - time) <= threshold) {
        low = middle + 1u;
      } else {
        high = middle;
      }
    }
    let close = low - (index + 1u);
    if (close > 0u) {
      let previous = atomicAdd(&groupLow, close);
      if (previous + close < previous) {
        atomicAdd(&groupCarry, 1u);
      }
    }
  }
  workgroupBarrier();
  if (localInvocationIndex == 0u) {
    let groupTotal = atomicLoad(&groupLow);
    var carries = atomicLoad(&groupCarry);
    if (groupTotal > 0u) {
      let previous = atomicAdd(&timeClose[timeCloseOffset], groupTotal);
      if (previous + groupTotal < previous) {
        carries++;
      }
    }
    if (carries > 0u) {
      atomicAdd(&timeClose[timeCloseOffset + 1u], carries);
    }
  }`
      }),
      createSpaceTimePairMaskNode<Parameters>(graph, {
        id,
        operation: OPERATION,
        rows,
        offsets: pairs.offsets,
        neighbors: pairs.neighbors,
        pairMask
      }),
      ...(stopping ? stoppingNodes : [createPartialsNode(), createReduceNode()]),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-summary`,
        operation: OPERATION,
        variant: 'summary',
        bindings: [
          {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
          {name: 'statistics', view: statistics, type: 'u32', access: 'read'},
          {name: 'pairPartials', view: pairPartials, type: 'u32', access: 'read'},
          {name: 'timeClose', view: timeClose, type: 'u32', access: 'read'},
          {name: 'summary', view: summary, type: 'f32', access: 'read_write'},
          ...stateBindings,
          ...(props.output.permutationCount
            ? [
                {
                  name: 'permutationsUsed',
                  view: props.output.permutationCount,
                  type: 'u32' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
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
  ${getSpaceTimeSummaryBodyWGSL('u32', stopAfterExceedances)}`
      })
    ];
    return nodes;
  }
}
