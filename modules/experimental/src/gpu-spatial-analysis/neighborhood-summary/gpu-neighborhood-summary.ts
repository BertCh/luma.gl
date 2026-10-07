// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  createFillNode,
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {type GPUSpatialWeights, validateGPUSpatialWeights} from '../spatial-weights/index';
import {SPATIAL_AUTOCORRELATION_FLOAT_WGSL} from '../spatial-autocorrelation/spatial-autocorrelation-kernels';

const OPERATION = 'GPUNeighborhoodSummary';

/** Largest supported compile-time `maximumNeighbors` (members per row for the median). */
export const GPU_NEIGHBORHOOD_SUMMARY_MAXIMUM_NEIGHBORS = 64;

/** `modes` value of a row with no members. */
export const GPU_NEIGHBORHOOD_SUMMARY_NO_MODE = 0xffffffff;

/** A numeric statistic of {@link GPUNeighborhoodSummary}; one `output` column each. */
export type GPUNeighborhoodSummaryStatistic =
  | 'count'
  | 'weightSum'
  | 'sum'
  | 'mean'
  | 'min'
  | 'max'
  | 'standardDeviation'
  | 'median';

const STATISTICS: readonly GPUNeighborhoodSummaryStatistic[] = [
  'count',
  'weightSum',
  'sum',
  'mean',
  'min',
  'max',
  'standardDeviation',
  'median'
];

/**
 * Properties for {@link GPUNeighborhoodSummary}.
 *
 * Per-frame (no rebuild or recompile): the contents of `weights`, `values`, `categories` and
 * `mask`. Compile-time: the row count, slot capacity, `statistics`, `includeFocal`, `focalWeight`,
 * `maximumNeighbors` and which optional views exist.
 */
export type GPUNeighborhoodSummaryProps = {
  /** Prefix for generated node IDs. Defaults to `'neighborhood-summary'`. */
  id?: string;
  /** Square spatial weights whose neighbor IDs index the value rows. */
  weights: GPUSpatialWeights;
  /** Numeric values, one per row. Required when `statistics` is not empty. */
  values?: GraphDataView<'float32'>;
  /** Categorical uint32 column, one per row. Required when `modes` or `entropy` is given. */
  categories?: GraphDataView<'uint32'>;
  /** Optional row selection: nonzero includes the row as a focus and as a neighbor. */
  mask?: GraphDataView<'uint32'>;
  /**
   * Adds the focal row itself to its own neighborhood (momepy's `include_self`, a self weight on
   * the diagonal). The focal member comes first, with weight `focalWeight`. Compile-time.
   */
  includeFocal?: boolean;
  /** Weight of the focal member when `includeFocal` is set. Finite and non-negative; defaults to 1. */
  focalWeight?: number;
  /** Numeric statistics to write, in `output` column order. No duplicates. Compile-time. */
  statistics?: readonly GPUNeighborhoodSummaryStatistic[];
  /**
   * Caller-owned row-major table of `rows * statistics.length` float32 values; row `i` column `c`
   * holds `statistics[c]` at index `i * statistics.length + c`.
   */
  output?: GraphDataView<'float32'>;
  /**
   * Compile-time capacity of the median buffer, in `[1, 64]`. Members beyond it set `overflow`
   * and the row's median is quiet NaN. It also sizes the private member cache of the categorical
   * `modes` and `entropy` (larger rows stay exact but take a slower scan). Defaults to 32.
   */
  maximumNeighbors?: number;
  /** Caller-owned one-row flag: 1 when a row with the median requested exceeded `maximumNeighbors`. */
  overflow?: GraphDataView<'uint32'>;
  /** Optional caller-owned weighted mode category per row, `GPU_NEIGHBORHOOD_SUMMARY_NO_MODE` if empty. */
  modes?: GraphDataView<'uint32'>;
  /** Optional caller-owned Shannon entropy (natural log) of the weighted category shares per row. */
  entropy?: GraphDataView<'float32'>;
};

/**
 * Per-row summaries of a neighborhood over a {@link GPUSpatialWeights} CSR: the generalized spatial
 * lag family (momepy `describe`, libpysal `lag_categorical`).
 *
 * Members of row `i` are, in this fixed order: the focal row (with `includeFocal`, weight
 * `focalWeight`), then the CSR slots of the row in slot order, skipping a listed self slot, slots
 * outside `[0, rows)`, neighbors with a zero mask and, for the numeric statistics, neighbors with
 * a non-finite value. A row with a zero mask is not summarized: `count` is 0 and every other
 * column is quiet NaN (`modes` holds `GPU_NEIGHBORHOOD_SUMMARY_NO_MODE`). A focal row's own value
 * is only used with `includeFocal`. With `k` members, weights `w_s` and values `x_s`:
 * - `count` is `k`; `weightSum` is `W = sum w_s`; `sum` is `sum w_s x_s` (the lag).
 * - `mean` is `sum / W` (NaN when `W = 0`).
 * - `min` and `max` are unweighted over the members (NaN when `k = 0`).
 * - `standardDeviation` is the weighted population deviation `sqrt(sum w_s (x_s - mean)^2 / W)`,
 *   computed in a second pass about the mean (NaN when `W = 0`).
 * - `median` is the unweighted median (mean of the two middle values for even `k`) for `k` up to
 *   `maximumNeighbors`; larger rows are not summarized and set `overflow`.
 * - `modes` is the category with the largest weighted frequency `sum_{s: c_s = c} w_s`; ties go
 *   to the lowest category. `entropy` is `-sum p_c ln p_c` with `p_c = f_c / W` over the categories
 *   in first-occurrence order (0 for one category, NaN when `W = 0`). Category cost is `O(k^2)` per
 *   row with no member limit.
 *
 * Every sum runs sequentially in member order in f32, so results are deterministic and match a
 * CPU sum in the same order up to fused multiply-add rounding.
 *
 * The numeric statistics are one kernel and the categorical outputs another, so each stays under
 * the 8 storage-buffer limit.
 */
export class GPUNeighborhoodSummary implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUNeighborhoodSummaryProps;

  constructor(props: GPUNeighborhoodSummaryProps) {
    const id = props.id ?? 'neighborhood-summary';
    this.id = id;
    this.props = props;
    const rows = validateGPUSpatialWeights(id, props.weights);
    const statistics = props.statistics ?? [];
    for (const statistic of statistics) {
      if (!STATISTICS.includes(statistic)) {
        throw new Error(`${id} unknown statistic ${statistic}`);
      }
    }
    if (new Set(statistics).size !== statistics.length) {
      throw new Error(`${id} statistics must not repeat`);
    }
    const categorical = Boolean(props.modes || props.entropy);
    if (statistics.length === 0 && !categorical) {
      throw new Error(`${id} requests no statistic`);
    }
    if (statistics.length > 0) {
      if (!props.values || !props.output) {
        throw new Error(`${id} statistics require values and output`);
      }
      validatePackedView(props.values, ['float32'], `${id} values`);
      validatePackedView(props.output, ['float32'], `${id} output`);
      if (props.values.length !== rows) {
        throw new Error(`${id} values length must equal the weights row count`);
      }
      if (props.output.length !== rows * statistics.length) {
        throw new Error(`${id} output length must equal rows * statistics.length`);
      }
    } else if (props.output || props.values) {
      throw new Error(`${id} values and output require statistics`);
    }
    if (statistics.includes('median')) {
      if (!props.overflow) {
        throw new Error(`${id} median requires overflow`);
      }
      validatePackedUint32View(props.overflow, `${id} overflow`);
      if (props.overflow.length < 1) {
        throw new Error(`${id} overflow must hold one uint32`);
      }
    }
    const maximumNeighbors = props.maximumNeighbors ?? 32;
    if (
      !Number.isInteger(maximumNeighbors) ||
      maximumNeighbors < 1 ||
      maximumNeighbors > GPU_NEIGHBORHOOD_SUMMARY_MAXIMUM_NEIGHBORS
    ) {
      throw new Error(
        `${id} maximumNeighbors must be an integer in [1, ${GPU_NEIGHBORHOOD_SUMMARY_MAXIMUM_NEIGHBORS}]`
      );
    }
    if (categorical) {
      if (!props.categories) {
        throw new Error(`${id} modes and entropy require categories`);
      }
      validatePackedUint32View(props.categories, `${id} categories`);
      if (props.categories.length !== rows) {
        throw new Error(`${id} categories length must equal the weights row count`);
      }
    }
    if (props.modes) {
      validatePackedUint32View(props.modes, `${id} modes`);
      if (props.modes.length !== rows) {
        throw new Error(`${id} modes length must equal the weights row count`);
      }
    }
    if (props.entropy) {
      validatePackedView(props.entropy, ['float32'], `${id} entropy`);
      if (props.entropy.length !== rows) {
        throw new Error(`${id} entropy length must equal the weights row count`);
      }
    }
    if (props.mask) {
      validatePackedUint32View(props.mask, `${id} mask`);
      if (props.mask.length !== rows) {
        throw new Error(`${id} mask length must equal the weights row count`);
      }
    }
    const focalWeight = props.focalWeight ?? 1;
    if (!Number.isFinite(focalWeight) || focalWeight < 0) {
      throw new Error(`${id} focalWeight must be finite and non-negative`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.output, props.overflow, props.modes, props.entropy],
      [
        props.weights.offsets,
        props.weights.neighbors,
        props.weights.weights,
        props.values,
        props.categories,
        props.mask
      ]
    );
  }

  /** Returns the summary nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {weights, mask} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      weights.offsets,
      weights.neighbors,
      weights.weights,
      props.values,
      props.categories,
      mask,
      props.output,
      props.overflow,
      props.modes,
      props.entropy
    ]);
    const rows = weights.offsets.length - 1;
    const statistics = props.statistics ?? [];
    const includeFocal = Boolean(props.includeFocal);
    const maximumNeighbors = props.maximumNeighbors ?? 32;
    const constants = `const ROWS: u32 = ${rows}u;
const CAPACITY: u32 = ${weights.neighbors.length}u;
const FOCAL_COUNT: u32 = ${includeFocal ? 1 : 0}u;
const FOCAL_WEIGHT: f32 = ${getWGSLFloatLiteral(props.focalWeight ?? 1)};
${SPATIAL_AUTOCORRELATION_FLOAT_WGSL}
struct Member {
  row: u32,
  weight: f32,
  valid: bool
}
fn getMemberRange(row: u32) -> vec2<u32> {
  let begin = min(offsets[offsetsOffset + row], CAPACITY);
  let end = min(offsets[offsetsOffset + row + 1u], CAPACITY);
  return vec2<u32>(begin, max(end, begin));
}`;
    const maskedWGSL = (rowExpression: string) =>
      mask ? `mask[maskOffset + ${rowExpression}] != 0u` : 'true';
    // Member `visit` of a row: the focal row first (if requested), then CSR slots in order.
    const getMemberWGSL = (validExpression: (row: string) => string) => `
fn getMember(row: u32, begin: u32, visit: u32) -> Member {
  var member = Member(row, FOCAL_WEIGHT, false);
  if (visit >= FOCAL_COUNT) {
    let slot = begin + visit - FOCAL_COUNT;
    member.row = neighbors[neighborsOffset + slot];
    member.weight = weights[weightsOffset + slot];
    if (member.row >= ROWS || member.row == row) {
      return member;
    }
  }
  member.valid = ${maskedWGSL('member.row')} && ${validExpression('member.row')};
  return member;
}`;
    const nodes: GPUCommandNode<Parameters>[] = [];

    if (statistics.length > 0) {
      const {values, output} = props;
      const columns = statistics.length;
      const wants = (statistic: GPUNeighborhoodSummaryStatistic) => statistics.includes(statistic);
      const needsMean = wants('mean') || wants('standardDeviation');
      const bindings: WGSLKernelBinding[] = [
        {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
        {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
        {name: 'weights', view: weights.weights, type: 'f32', access: 'read'},
        {name: 'values', view: values!, type: 'f32', access: 'read'},
        ...(mask
          ? [{name: 'mask', view: mask, type: 'u32' as const, access: 'read' as const}]
          : []),
        {name: 'output', view: output!, type: 'f32', access: 'read_write'}
      ];
      if (wants('median')) {
        bindings.push({
          name: 'overflow',
          view: props.overflow!,
          type: 'atomic<u32>',
          access: 'read_write'
        });
        nodes.push(
          createFillNode<Parameters>(graph, {
            id: `${id}-overflow-clear`,
            operation: OPERATION,
            view: props.overflow!,
            type: 'u32',
            value: '0u',
            componentCount: 1
          })
        );
      }
      const columnWGSL = (statistic: GPUNeighborhoodSummaryStatistic) => {
        switch (statistic) {
          case 'count':
            return 'f32(count)';
          case 'weightSum':
            return 'weightSum';
          case 'sum':
            return 'weightedSum';
          case 'mean':
            return 'select(nan, mean, weightSum > 0.0)';
          case 'min':
            return 'select(nan, minimum, count > 0u)';
          case 'max':
            return 'select(nan, maximum, count > 0u)';
          case 'standardDeviation':
            return 'select(nan, sqrt(squares / weightSum), weightSum > 0.0)';
          default:
            return 'median';
        }
      };
      // Rows with a zero mask: count 0 and NaN elsewhere.
      const excludedWGSL = statistics
        .map(
          (statistic, column) =>
            `output[outputOffset + index * ${columns}u + ${column}u] = ${statistic === 'count' ? '0.0' : 'nan'};`
        )
        .join('\n    ');
      const resultWGSL = statistics
        .map(
          (statistic, column) =>
            `output[outputOffset + index * ${columns}u + ${column}u] = ${columnWGSL(statistic)};`
        )
        .join('\n    ');
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-numeric`,
          operation: OPERATION,
          variant: statistics.join('-'),
          bindings,
          invocationCount: rows,
          declarations: `${constants}
const MAXIMUM_NEIGHBORS: u32 = ${maximumNeighbors}u;
${getMemberWGSL(row => `isFiniteFloat(values[valuesOffset + ${row}])`)}`,
          body: `let nan = getQuietNaN(index);
  if (!(${maskedWGSL('index')})) {
    ${excludedWGSL}
    return;
  }
  let range = getMemberRange(index);
  let visits = range.y - range.x + FOCAL_COUNT;
  var count = 0u;
  var weightSum = 0.0;
  var weightedSum = 0.0;
  var minimum = 3.0e38;
  var maximum = -3.0e38;
  ${wants('median') ? `var sorted: array<f32, ${maximumNeighbors}>;` : ''}
  for (var visit = 0u; visit < visits; visit++) {
    let member = getMember(index, range.x, visit);
    if (!member.valid) {
      continue;
    }
    let value = values[valuesOffset + member.row];
    weightSum += member.weight;
    weightedSum += member.weight * value;
    minimum = min(minimum, value);
    maximum = max(maximum, value);
    ${
      wants('median')
        ? `if (count < MAXIMUM_NEIGHBORS) {
      var position = count;
      loop {
        if (position == 0u || sorted[position - 1u] <= value) {
          break;
        }
        sorted[position] = sorted[position - 1u];
        position--;
      }
      sorted[position] = value;
    }`
        : ''
    }
    count++;
  }
  ${needsMean ? 'let mean = weightedSum / weightSum;' : ''}
  var squares = 0.0;
  ${
    wants('standardDeviation')
      ? `for (var visit = 0u; visit < visits; visit++) {
    let member = getMember(index, range.x, visit);
    if (member.valid) {
      let deviation = values[valuesOffset + member.row] - mean;
      squares += member.weight * deviation * deviation;
    }
  }`
      : ''
  }
  ${
    wants('median')
      ? `var median = nan;
  if (count > MAXIMUM_NEIGHBORS) {
    atomicMax(&overflow[overflowOffset], 1u);
  } else if (count > 0u) {
    let middle = count / 2u;
    median = select(0.5 * (sorted[middle - 1u] + sorted[middle]), sorted[middle], (count & 1u) == 1u);
  }`
      : ''
  }
  ${resultWGSL}`
        })
      );
    }

    if (props.modes || props.entropy) {
      const bindings: WGSLKernelBinding[] = [
        {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
        {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
        {name: 'weights', view: weights.weights, type: 'f32', access: 'read'},
        {name: 'categories', view: props.categories!, type: 'u32', access: 'read'},
        ...(mask
          ? [{name: 'mask', view: mask, type: 'u32' as const, access: 'read' as const}]
          : []),
        ...(props.modes
          ? [
              {
                name: 'modes',
                view: props.modes,
                type: 'u32' as const,
                access: 'read_write' as const
              }
            ]
          : []),
        ...(props.entropy
          ? [
              {
                name: 'entropy',
                view: props.entropy,
                type: 'f32' as const,
                access: 'read_write' as const
              }
            ]
          : [])
      ];
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-categorical`,
          operation: OPERATION,
          variant: `categorical${props.modes ? '-mode' : ''}${props.entropy ? '-entropy' : ''}`,
          bindings,
          invocationCount: rows,
          declarations: `${constants}
const NO_MODE: u32 = ${GPU_NEIGHBORHOOD_SUMMARY_NO_MODE}u;
const MAXIMUM_NEIGHBORS: u32 = ${maximumNeighbors}u;
${getMemberWGSL(() => 'true')}`,
          body: `let nan = getQuietNaN(index);
  if (!(${maskedWGSL('index')})) {
    ${props.modes ? 'modes[modesOffset + index] = NO_MODE;' : ''}
    ${props.entropy ? 'entropy[entropyOffset + index] = nan;' : ''}
    return;
  }
  let range = getMemberRange(index);
  let visits = range.y - range.x + FOCAL_COUNT;
  // Decode the members once. Up to MAXIMUM_NEIGHBORS of them are kept in private arrays so the
  // quadratic frequency scan below reads registers instead of re-decoding every member (four
  // global loads) for every pair; larger rows take the unbounded decoding scan.
  var memberCategories: array<u32, ${maximumNeighbors}>;
  var memberWeights: array<f32, ${maximumNeighbors}>;
  var memberCount = 0u;
  var weightSum = 0.0;
  for (var visit = 0u; visit < visits; visit++) {
    let member = getMember(index, range.x, visit);
    if (member.valid) {
      weightSum += member.weight;
      if (memberCount < MAXIMUM_NEIGHBORS) {
        memberCategories[memberCount] = categories[categoriesOffset + member.row];
        memberWeights[memberCount] = member.weight;
      }
      memberCount++;
    }
  }
  var bestCategory = NO_MODE;
  var bestFrequency = -1.0;
  var entropyTotal = 0.0;
  if (memberCount <= MAXIMUM_NEIGHBORS) {
    for (var member = 0u; member < memberCount; member++) {
      let category = memberCategories[member];
      // Frequency of this category, summed in member order, at its first occurrence only.
      var frequency = 0.0;
      var firstOccurrence = true;
      for (var other = 0u; other < memberCount; other++) {
        if (memberCategories[other] == category) {
          if (other < member) {
            firstOccurrence = false;
            break;
          }
          frequency += memberWeights[other];
        }
      }
      if (!firstOccurrence) {
        continue;
      }
      if (frequency > bestFrequency || (frequency == bestFrequency && category < bestCategory)) {
        bestFrequency = frequency;
        bestCategory = category;
      }
      if (frequency > 0.0) {
        let share = frequency / weightSum;
        entropyTotal -= share * log(share);
      }
    }
  } else {
    for (var visit = 0u; visit < visits; visit++) {
      let member = getMember(index, range.x, visit);
      if (!member.valid) {
        continue;
      }
      let category = categories[categoriesOffset + member.row];
      // Frequency of this category, summed in member order, at its first occurrence only.
      var frequency = 0.0;
      var firstOccurrence = true;
      for (var other = 0u; other < visits; other++) {
        let candidate = getMember(index, range.x, other);
        if (candidate.valid && categories[categoriesOffset + candidate.row] == category) {
          if (other < visit) {
            firstOccurrence = false;
            break;
          }
          frequency += candidate.weight;
        }
      }
      if (!firstOccurrence) {
        continue;
      }
      if (frequency > bestFrequency || (frequency == bestFrequency && category < bestCategory)) {
        bestFrequency = frequency;
        bestCategory = category;
      }
      if (frequency > 0.0) {
        let share = frequency / weightSum;
        entropyTotal -= share * log(share);
      }
    }
  }
  ${props.modes ? 'modes[modesOffset + index] = bestCategory;' : ''}
  ${props.entropy ? 'entropy[entropyOffset + index] = select(nan, entropyTotal, weightSum > 0.0);' : ''}`
        })
      );
    }
    return nodes;
  }
}
