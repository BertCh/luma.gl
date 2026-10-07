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
  getSpaceTimeBlocks,
  getSpaceTimeCommonWGSL,
  getSpaceTimeSummaryBodyWGSL,
  SPACE_TIME_MAXIMUM_PERMUTATIONS
} from './space-time-kernels';
import {
  GPU_SPACE_TIME_PARAMETER_LENGTH,
  GPU_SPACE_TIME_SUMMARY_LENGTH
} from './space-time-parameters';

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
   * the Mantel statistic runs over the entries `j > i` of every row, with `distances` as the
   * spatial distance matrix. A radius-mode `GPUNeighborSearch` gives the distance band; a band
   * wide enough to list every pair gives the full Mantel test.
   */
  pairs: GPUSpatialWeights;
  /** Event time per row, finite float32 (NaN is not supported). */
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
 * `i < j`, with a Monte Carlo reference from `P` Feistel permutations of the times.
 *
 * `r = 0` when either side has no variance. The pair list is a restriction of the full distance
 * matrices when a band is used: the statistic is the correlation over the listed pairs only,
 * and a full-matrix Mantel test needs a list of every pair (O(n^2) slots). Spatial and time
 * transformations (log, reciprocal) are not applied: transform `distances` and `times` first if
 * wanted, since `|t_i - t_j|` is taken on the supplied times.
 *
 * Numerics: partial sums per (permutation, row block) are f32, centred on the observed mean
 * spatial distance and time difference (computed first in fixed order) to avoid cancellation, and
 * reduced in fixed order, so results are deterministic and match a double-precision CPU oracle to
 * about 1e-4 in `r` for up to 10^6 pairs. A permuted `r` within f32 rounding of the observed `r`
 * can fall on either side of the exceedance count.
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

  /** Returns `pair-stats`, `shift`, `partials`, `reduce` and `summary` nodes in order. */
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
    const common = getSpaceTimeCommonWGSL(rows, maximumPermutations, blocks, rowsPerBlock);
    // pair-stats is one pass over the pairs with no permutation axis, so it takes more blocks
    // than the (permutation, block) kernels to fill the device.
    const pairStatBlocks = Math.min(MAXIMUM_PAIR_STAT_BLOCKS, rows);
    const pairCounts = createTransientView(graph, `${id}-pair-counts`, 'uint32', pairStatBlocks);
    const pairSums = createTransientView(graph, `${id}-pair-sums`, 'float32', pairStatBlocks * 2);
    const spatialMoments = createTransientView(
      graph,
      `${id}-spatial-moments`,
      'float32',
      blocks * 2
    );
    const pairCount = createTransientView(graph, `${id}-pair-count`, 'uint32', 1);
    const shift = createTransientView(graph, `${id}-shift`, 'float32', 2);
    const partials = createTransientView(
      graph,
      `${id}-partials`,
      'float32',
      slotCount * blocks * PARTIAL_STRIDE
    );
    const pairBindings = [
      {name: 'offsets', view: pairs.offsets, type: 'u32' as const, access: 'read' as const},
      {name: 'neighbors', view: pairs.neighbors, type: 'u32' as const, access: 'read' as const},
      {name: 'distances', view: distances, type: 'f32' as const, access: 'read' as const},
      {name: 'times', view: times, type: 'f32' as const, access: 'read' as const}
    ];
    return [
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
      if (other > row && other < ROWS) {
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
          {name: 'partials', view: partials, type: 'f32', access: 'read_write'},
          {name: 'spatialMoments', view: spatialMoments, type: 'f32', access: 'read_write'}
        ],
        invocationCount: slotCount * blocks,
        declarations: `${common}
const SLOT_COUNT: u32 = ${slotCount}u;`,
        // Slot-fastest with interleaved rows (see the Knox partials): a warp shares its block, so
        // each CSR entry is a broadcast load. Slot 0 also writes the permutation-invariant moments.
        body: `let block = index / SLOT_COUNT;
  let slot = index - block * SLOT_COUNT;
  let permutations = readPermutationCount();
  var sumX = 0.0;
  var sumXX = 0.0;
  var sumY = 0.0;
  var sumYY = 0.0;
  var sumXY = 0.0;
  if (slot <= permutations) {
    let keys = getFeistelRoundKeys(readSeedKey(), select(0u, slot - 1u, slot > 0u));
    let halfBits = getFeistelHalfBits(ROWS);
    let shiftDistance = shift[shiftOffset];
    let shiftTime = shift[shiftOffset + 1u];
    for (var row = block; row < ROWS; row += BLOCKS) {
      let time = times[timesOffset + getPermutedRow(row, slot, keys, halfBits)];
      for (var entry = offsets[offsetsOffset + row]; entry < offsets[offsetsOffset + row + 1u]; entry++) {
        let other = neighbors[neighborsOffset + entry];
        if (other > row && other < ROWS) {
          let x = distances[distancesOffset + entry] - shiftDistance;
          let y = abs(time - times[timesOffset + getPermutedRow(other, slot, keys, halfBits)]) - shiftTime;
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
  let base = partialsOffset + (block * SLOT_COUNT + slot) * ${PARTIAL_STRIDE}u;
  partials[base] = sumY;
  partials[base + 1u] = sumYY;
  partials[base + 2u] = sumXY;
  if (slot == 0u) {
    spatialMoments[spatialMomentsOffset + 2u * block] = sumX;
    spatialMoments[spatialMomentsOffset + 2u * block + 1u] = sumXX;
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-reduce`,
        operation: OPERATION,
        variant: 'reduce',
        bindings: [
          {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
          {name: 'partials', view: partials, type: 'f32', access: 'read'},
          {name: 'spatialMoments', view: spatialMoments, type: 'f32', access: 'read'},
          {name: 'pairCount', view: pairCount, type: 'u32', access: 'read'},
          {name: 'statistics', view: statistics, type: 'f32', access: 'read_write'}
        ],
        invocationCount: statistics.length,
        declarations: `${common}
const SLOT_COUNT: u32 = ${slotCount}u;`,
        body: `var result = 0.0;
  if (index <= readPermutationCount()) {
    var sumX = 0.0;
    var sumXX = 0.0;
    var sumY = 0.0;
    var sumYY = 0.0;
    var sumXY = 0.0;
    for (var block = 0u; block < BLOCKS; block++) {
      let base = partialsOffset + (block * SLOT_COUNT + index) * ${PARTIAL_STRIDE}u;
      sumX += spatialMoments[spatialMomentsOffset + 2u * block];
      sumXX += spatialMoments[spatialMomentsOffset + 2u * block + 1u];
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
