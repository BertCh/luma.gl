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
  createSpaceTimePairMaskNode,
  getSpaceTimeBlocks,
  getSpaceTimeCommonWGSL,
  getSpaceTimeGroupsPerSlot,
  getSpaceTimeShuffleWGSL,
  getSpaceTimeSummaryBodyWGSL,
  SPACE_TIME_MAXIMUM_PERMUTATIONS,
  SPACE_TIME_MAXIMUM_ROWS,
  SPACE_TIME_SLOT_PROLOGUE_WGSL,
  SPACE_TIME_WORKGROUP_SIZE
} from './space-time-kernels';
import {
  getGPUSpaceTimeInferenceMetadata,
  GPU_SPACE_TIME_PARAMETER_LENGTH,
  GPU_SPACE_TIME_SUMMARY_LENGTH,
  type GPUSpaceTimeParameters
} from './space-time-parameters';
import type {GPUInferenceMetadata} from '../contracts/index';

const OPERATION = 'GPUMantelTest';
/**
 * Float words per (permutation, block) partial: `sum y, sum y^2, sum x y`. The spatial moments
 * `sum x` and `sum x^2` do not depend on the permutation, so they are computed once per block.
 */
const PARTIAL_STRIDE = 3;
/** Largest block count of the observed-moment pass, which has no per-permutation parallelism. */
const MAXIMUM_PAIR_STAT_BLOCKS = 4096;

/**
 * Properties for {@link GPUMantelTest}.
 *
 * Per-frame (no recompile): the contents of `times`, `pairs` and `parameters` (seed, permutation
 * count). Compile-time: the row count, `maximumPermutations` and the pair capacity.
 */
export type GPUMantelTestProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'mantel-test'`. */
  id?: string;
  /**
   * Pairs and their spatial distance, as a self-join {@link GPUSpatialWeights} with `distances`:
   * the Mantel statistic runs over every unordered pair that appears in either direction, counted
   * exactly once (taken from the lower row when both directions are listed, otherwise from the row
   * that lists it; duplicates and self entries are ignored), with `distances` as the spatial
   * distance. A radius-mode `GPUNeighborSearch` gives the distance band; a band wide enough to
   * list every pair gives the full Mantel test.
   */
  pairs: GPUSpatialWeights;
  /**
   * Event time per row, finite float32 (NaN is not supported), measured from a nearby origin: raw
   * epoch seconds do not fit float32 (128 s spacing near 1.7e9), so subtract an origin on the CPU
   * with `getGPUSpaceTimeRelativeTimes`.
   */
  times: GraphDataView<'float32'>;
  /** Per-frame parameters: `GPU_SPACE_TIME_PARAMETER_LENGTH` uint32 words (`getGPUSpaceTimeParameterValues`). */
  parameters: GraphDataView<'uint32'>;
  /** Compile-time upper bound on the per-frame permutation count, 1 to 65536. */
  maximumPermutations: number;
  /**
   * Caller-owned statistics, `maximumPermutations + 1` rows: element 0 is the observed correlation
   * `r`, elements `1..P` the correlation under permuted times (rows past `P` are 0).
   */
  statistics: GraphDataView<'float32'>;
  /** Caller-owned `GPU_SPACE_TIME_SUMMARY_LENGTH` float32 words, see `GPU_SPACE_TIME_SUMMARY`. */
  summary: GraphDataView<'float32'>;
};

/**
 * Mantel test of space-time interaction (pointpats `Mantel`): the Pearson correlation `r` between
 * spatial distances `d_ij` and absolute time differences `|t_i - t_j|` over the listed pairs
 * `i < j`, with a Monte Carlo reference from `P` swap-or-not permutations of the times.
 *
 * Breaking change: the permutations used to be a Feistel bijection, which is measurably non-uniform
 * on small domains; the swap-or-not shuffle (Hoang-Morris-Rogaway) is uniform, so the permuted
 * correlations for a given seed differ from earlier versions. Permutation `s >= 1` is draw `s` of
 * the seed. Fewer than 2^24 events are supported.
 *
 * `r = 0` when either side has no variance. The pair list is a restriction of the full distance
 * matrices when a band is used: the statistic is the correlation over the listed pairs only (pairs
 * farther apart than the band are truncated away, so `r` is a distance-band Mantel statistic and
 * differs from the full Mantel `r`; report the band with the result), and a full-matrix Mantel test
 * needs a list of every pair (O(n^2) slots). Spatial and time
 * transformations (log, reciprocal) are not applied: transform `distances` and `times` first if
 * wanted, since `|t_i - t_j|` is taken on the supplied times.
 *
 * Numerics: partial sums per (permutation, row block) are f32, centred on the observed mean
 * spatial distance and time difference (computed first in fixed order) to avoid cancellation, and
 * reduced in fixed order, so results are deterministic and match a double-precision CPU oracle to
 * about 1e-4 in `r` for up to 10^6 pairs. A permuted `r` within f32 rounding of the observed `r`
 * can fall on either side of the exceedance count.
 *
 * Alternative: the space-time interaction hypothesis is a positive correlation (events near in
 * space are near in time), the upper-tail convention of pointpats `mantel`, so
 * `getInferenceMetadata` states `'greater'` and `pseudoPGreater` is the p-value. `pseudoPTwoSided` is also in the `summary`.
 *
 * Time units: pass times relative to an origin (`getGPUSpaceTimeRelativeTimes`); float32 times
 * resolve about `span * 2^-24`.
 */
export class GPUMantelTest implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUMantelTestProps;
  /** Number of events. */
  readonly rows: number;
  constructor(props: GPUMantelTestProps) {
    const id = props.id ?? 'mantel-test';
    this.id = id;
    this.props = props;
    this.rows = validateGPUSpatialWeights(id, props.pairs, 'pairs');
    if (!props.pairs.distances) {
      throw new Error(`${id} pairs.distances is required`);
    }
    validatePackedView(props.times, ['float32'], `${id} times`);
    validatePackedUint32View(props.parameters, `${id} parameters`);
    validatePackedView(props.statistics, ['float32'], `${id} statistics`);
    validatePackedView(props.summary, ['float32'], `${id} summary`);
    if (this.rows < 2) {
      throw new Error(`${id} needs at least two rows`);
    }
    if (this.rows > SPACE_TIME_MAXIMUM_ROWS) {
      // The swap-or-not shuffle has round capacity for domains below 2^24 rows.
      throw new Error(`${id} supports at most ${SPACE_TIME_MAXIMUM_ROWS} rows`);
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
        props.pairs.weights,
        props.pairs.distances
      ]
    );
    if (props.statistics.buffer === props.summary.buffer) {
      throw new Error(`${id} outputs must use separate buffers`);
    }
  }

  /**
   * CPU-side description of how the `summary` p-values are produced, for building a legend.
   * Alternative `'greater'`, Monte Carlo, no multiple-testing correction. Pass the per-frame
   * `parameters` for the actual permutation count and seed; without them the count is
   * `maximumPermutations`.
   */
  getInferenceMetadata(parameters?: GPUSpaceTimeParameters): GPUInferenceMetadata {
    return getGPUSpaceTimeInferenceMetadata(
      'mantel',
      parameters?.permutations ?? this.props.maximumPermutations,
      parameters?.seed
    );
  }

  /** Returns `pair-mask`, `pair-stats`, `shift`, `partials`, `reduce` and `summary` nodes in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, rows} = this;
    const {pairs, times, parameters, statistics, summary, maximumPermutations} = props;
    const distances = pairs.distances!;
    validateGraphViewsBelongToGraph(id, graph, [
      pairs.offsets,
      pairs.neighbors,
      pairs.weights,
      distances,
      times,
      parameters,
      statistics,
      summary
    ]);
    const [blocks, rowsPerBlock] = getSpaceTimeBlocks(rows);
    const slotCount = maximumPermutations + 1;
    const groupsPerSlot = getSpaceTimeGroupsPerSlot(blocks);
    const common = getSpaceTimeCommonWGSL(rows, maximumPermutations, blocks, rowsPerBlock);
    // pair-stats is one pass over the pairs with no permutation axis, so it takes more blocks
    // than the (permutation, block) kernels to fill the device.
    const pairStatBlocks = Math.min(MAXIMUM_PAIR_STAT_BLOCKS, rows);
    const pairCounts = createTransientView(graph, `${id}-pair-counts`, 'uint32', pairStatBlocks);
    const pairSums = createTransientView(graph, `${id}-pair-sums`, 'float32', pairStatBlocks * 2);
    const pairCount = createTransientView(graph, `${id}-pair-count`, 'uint32', 1);
    const pairMask = createTransientView(
      graph,
      `${id}-pair-mask`,
      'uint32',
      pairs.neighbors.length
    );
    const shift = createTransientView(graph, `${id}-shift`, 'float32', 2);
    const partials = createTransientView(
      graph,
      `${id}-partials`,
      'float32',
      // The permutation-invariant spatial moments (2 words per block) follow the partials, which
      // keeps the partials kernel at 8 storage bindings.
      slotCount * blocks * PARTIAL_STRIDE + blocks * 2
    );
    const pairBindings = [
      {name: 'offsets', view: pairs.offsets, type: 'u32' as const, access: 'read' as const},
      {name: 'neighbors', view: pairs.neighbors, type: 'u32' as const, access: 'read' as const},
      {name: 'pairMask', view: pairMask, type: 'u32' as const, access: 'read' as const},
      {name: 'distances', view: distances, type: 'f32' as const, access: 'read' as const},
      {name: 'times', view: times, type: 'f32' as const, access: 'read' as const}
    ];
    return [
      createSpaceTimePairMaskNode<Parameters>(graph, {
        id,
        operation: OPERATION,
        rows,
        offsets: pairs.offsets,
        neighbors: pairs.neighbors,
        pairMask
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-pair-stats`,
        operation: OPERATION,
        variant: 'pair-stats',
        bindings: [
          ...pairBindings,
          {name: 'pairCounts', view: pairCounts, type: 'u32', access: 'read_write'},
          {name: 'pairSums', view: pairSums, type: 'f32', access: 'read_write'}
        ],
        invocationCount: pairStatBlocks,
        declarations: `const ROWS: u32 = ${rows}u;
const PAIR_STAT_BLOCKS: u32 = ${pairStatBlocks}u;`,
        // Rows interleave across blocks, so neighboring invocations read neighboring CSR rows.
        body: `var count = 0u;
  var sumDistance = 0.0;
  var sumTime = 0.0;
  for (var row = index; row < ROWS; row += PAIR_STAT_BLOCKS) {
    for (var entry = offsets[offsetsOffset + row]; entry < offsets[offsetsOffset + row + 1u]; entry++) {
      let other = neighbors[neighborsOffset + entry];
      if (pairMask[pairMaskOffset + entry] != 0u) {
        count++;
        sumDistance += distances[distancesOffset + entry];
        sumTime += abs(times[timesOffset + row] - times[timesOffset + other]);
      }
    }
  }
  pairCounts[pairCountsOffset + index] = count;
  pairSums[pairSumsOffset + 2u * index] = sumDistance;
  pairSums[pairSumsOffset + 2u * index + 1u] = sumTime;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-shift`,
        operation: OPERATION,
        variant: 'shift',
        bindings: [
          {name: 'pairCounts', view: pairCounts, type: 'u32', access: 'read'},
          {name: 'pairSums', view: pairSums, type: 'f32', access: 'read'},
          {name: 'pairCount', view: pairCount, type: 'u32', access: 'read_write'},
          {name: 'shift', view: shift, type: 'f32', access: 'read_write'}
        ],
        invocationCount: 1,
        declarations: `const BLOCKS: u32 = ${pairStatBlocks}u;`,
        body: `var count = 0u;
  var sumDistance = 0.0;
  var sumTime = 0.0;
  for (var block = 0u; block < BLOCKS; block++) {
    count += pairCounts[pairCountsOffset + block];
    sumDistance += pairSums[pairSumsOffset + 2u * block];
    sumTime += pairSums[pairSumsOffset + 2u * block + 1u];
  }
  pairCount[pairCountOffset] = count;
  let divisor = max(f32(count), 1.0);
  shift[shiftOffset] = sumDistance / divisor;
  shift[shiftOffset + 1u] = sumTime / divisor;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-partials`,
        operation: OPERATION,
        variant: 'partials',
        bindings: [
          ...pairBindings,
          {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
          {name: 'shift', view: shift, type: 'f32', access: 'read'},
          {name: 'partials', view: partials, type: 'f32', access: 'read_write'}
        ],
        workgroupSize: SPACE_TIME_WORKGROUP_SIZE,
        invocationCount: slotCount * groupsPerSlot * SPACE_TIME_WORKGROUP_SIZE,
        guardIndex: false,
        declarations: `${common}
${getSpaceTimeShuffleWGSL(blocks)}
const SLOT_COUNT: u32 = ${slotCount}u;
const MOMENTS_BASE: u32 = partialsOffset + ${slotCount * blocks * PARTIAL_STRIDE}u;`,
        // One permutation per workgroup with interleaved rows (see the Knox partials). Slot 0 also
        // writes the permutation-invariant moments.
        body: `${SPACE_TIME_SLOT_PROLOGUE_WGSL}
  var sumX = 0.0;
  var sumXX = 0.0;
  var sumY = 0.0;
  var sumYY = 0.0;
  var sumXY = 0.0;
  if (runsSlot) {
    let shiftDistance = shift[shiftOffset];
    let shiftTime = shift[shiftOffset + 1u];
    for (var row = block; row < ROWS; row += BLOCKS) {
      let time = times[timesOffset + getPermutedRow(row, slot)];
      for (var entry = offsets[offsetsOffset + row]; entry < offsets[offsetsOffset + row + 1u]; entry++) {
        let other = neighbors[neighborsOffset + entry];
        if (pairMask[pairMaskOffset + entry] != 0u) {
          let x = distances[distancesOffset + entry] - shiftDistance;
          let y = abs(time - times[timesOffset + getPermutedRow(other, slot)]) - shiftTime;
          if (slot == 0u) {
            sumX += x;
            sumXX += x * x;
          }
          sumY += y;
          sumYY += y * y;
          sumXY += x * y;
        }
      }
    }
  }
  if (block < BLOCKS) {
    let base = partialsOffset + (block * SLOT_COUNT + slot) * ${PARTIAL_STRIDE}u;
    partials[base] = sumY;
    partials[base + 1u] = sumYY;
    partials[base + 2u] = sumXY;
  }
  if (slot == 0u && block < BLOCKS) {
    partials[MOMENTS_BASE + 2u * block] = sumX;
    partials[MOMENTS_BASE + 2u * block + 1u] = sumXX;
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-reduce`,
        operation: OPERATION,
        variant: 'reduce',
        bindings: [
          {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
          {name: 'partials', view: partials, type: 'f32', access: 'read'},
          {name: 'pairCount', view: pairCount, type: 'u32', access: 'read'},
          {name: 'statistics', view: statistics, type: 'f32', access: 'read_write'}
        ],
        invocationCount: statistics.length,
        declarations: `${common}
const SLOT_COUNT: u32 = ${slotCount}u;
const MOMENTS_BASE: u32 = partialsOffset + ${slotCount * blocks * PARTIAL_STRIDE}u;`,
        body: `var result = 0.0;
  if (index <= readPermutationCount()) {
    var sumX = 0.0;
    var sumXX = 0.0;
    var sumY = 0.0;
    var sumYY = 0.0;
    var sumXY = 0.0;
    for (var block = 0u; block < BLOCKS; block++) {
      let base = partialsOffset + (block * SLOT_COUNT + index) * ${PARTIAL_STRIDE}u;
      sumX += partials[MOMENTS_BASE + 2u * block];
      sumXX += partials[MOMENTS_BASE + 2u * block + 1u];
      sumY += partials[base];
      sumYY += partials[base + 1u];
      sumXY += partials[base + 2u];
    }
    let count = f32(pairCount[pairCountOffset]);
    let covariance = sumXY - sumX * sumY / count;
    let varianceX = sumXX - sumX * sumX / count;
    let varianceY = sumYY - sumY * sumY / count;
    if (count > 1.0 && varianceX > 0.0 && varianceY > 0.0) {
      result = clamp(covariance / sqrt(varianceX * varianceY), -1.0, 1.0);
    }
  }
  statistics[statisticsOffset + index] = result;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-summary`,
        operation: OPERATION,
        variant: 'summary',
        bindings: [
          {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
          {name: 'statistics', view: statistics, type: 'f32', access: 'read'},
          {name: 'pairCount', view: pairCount, type: 'u32', access: 'read'},
          {name: 'summary', view: summary, type: 'f32', access: 'read_write'}
        ],
        invocationCount: 1,
        declarations: common,
        body: `let observedPairs = f32(pairCount[pairCountOffset]);
  let timeClosePairs = 0.0;
  let expected = 0.0;
  ${getSpaceTimeSummaryBodyWGSL('f32')}`
      })
    ];
  }
}
