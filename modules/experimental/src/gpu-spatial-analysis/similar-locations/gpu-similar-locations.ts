// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUSort,
  GraphVectorView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {
  getGPUSimilarLocationsParameterLength,
  GPU_SIMILAR_LOCATIONS_CONTROL_LENGTH
} from './similar-locations-parameters';

const OPERATION = 'GPUSimilarLocations';
/** Lanes of the column-reduction workgroups (one workgroup per attribute). */
const REDUCTION_LANES = 256;
/**
 * Row count from which `'rank'` standardization sorts each column (O(n log n)) instead of the
 * all-pairs count (O(n^2) per column). Below it the single all-pairs dispatch per column is
 * cheaper than a radix sort per column.
 */
const RANK_SORT_MINIMUM_ROWS = 4096;

/** WGSL tree reductions of one workgroup. Every lane of the workgroup must call them together. */
const REDUCTION_WGSL = /* wgsl */ `
const LANES: u32 = ${REDUCTION_LANES}u;
var<workgroup> floatScratch: array<f32, ${REDUCTION_LANES}>;
var<workgroup> countScratch: array<u32, ${REDUCTION_LANES}>;
fn reduceSum(value: f32, lane: u32) -> f32 {
  workgroupBarrier();
  floatScratch[lane] = value;
  workgroupBarrier();
  for (var stride = LANES / 2u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      floatScratch[lane] += floatScratch[lane + stride];
    }
    workgroupBarrier();
  }
  return floatScratch[0];
}
fn reduceCount(value: u32, lane: u32) -> u32 {
  workgroupBarrier();
  countScratch[lane] = value;
  workgroupBarrier();
  for (var stride = LANES / 2u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      countScratch[lane] += countScratch[lane + stride];
    }
    workgroupBarrier();
  }
  return countScratch[0];
}`;

/** Order-preserving u32 key of a finite f32 (-0 and +0 share a key); WGSL for the rank sort. */
const RANK_KEY_WGSL = /* wgsl */ `
const INVALID_KEY: u32 = 0xffffffffu;
fn getOrderedKey(value: f32) -> u32 {
  // Integer test: a float compare of -0 against 0 can be folded away by the shader compiler.
  let raw = bitcast<u32>(value);
  let bits = select(raw, 0u, (raw & 0x7fffffffu) == 0u);
  return select(bits | 0x80000000u, ~bits, (bits >> 31u) == 1u);
}`;

/**
 * How attribute columns are standardized before distances are measured:
 * - `'zscore'`: `(x - mean) / standard deviation` over the valid rows (population deviation); a
 *   constant column contributes zero.
 * - `'rank'`: average-tie percentile rank in `[0, 1]`, `(rank - 1) / (n - 1)`. Below 4096 rows
 *   the ranking is an all-pairs count per column; from 4096 rows each column is radix sorted and
 *   the counts come from binary searches (O(n log n)). Both give the same ranks.
 */
export type GPUSimilarLocationsStandardization = 'zscore' | 'rank';

/** Caller-owned outputs of {@link GPUSimilarLocations}. */
export type GPUSimilarLocationsOutput = {
  /**
   * Similarity rank per row, uint32. 0 is the most similar (or, with `direction: 'least'`, the
   * least similar). Rows that cannot be ranked hold `GPU_SIMILAR_LOCATIONS_NO_RANK`.
   */
  ranks: GraphDataView<'uint32'>;
  /** Weighted Euclidean distance to the reference in standardized space; NaN when unranked. */
  distances: GraphDataView<'float32'>;
  /**
   * Row IDs of the best `min(resultCount, eligible rows)` rows in rank order, at least
   * `maximumResultCount` uint32 rows; unused rows hold `GPU_SIMILAR_LOCATIONS_NO_RANK`.
   */
  topIds: GraphDataView<'uint32'>;
  /** Single uint32 row: the number of valid entries of `topIds`. */
  count: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUSimilarLocations}.
 *
 * Per-frame (no recompile): the contents of `parameters`, `selection`, `mask` and `attributes`.
 * Topology (needs a new graph): `rowCount`, `attributeCount`, `standardization` and
 * `maximumResultCount`, and whether `mask` is present.
 */
export type GPUSimilarLocationsProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'similar-locations'`. */
  id?: string;
  /**
   * Row-major packed float32 attribute matrix, `rowCount * attributeCount` elements. A row with
   * any non-finite attribute is invalid and is neither ranked nor used in the statistics.
   */
  attributes: GraphDataView<'float32'>;
  /** Number of attribute columns, at least 1. */
  attributeCount: number;
  /** Optional packed `uint32` row mask; zero makes the row invalid. */
  mask?: GraphDataView<'uint32'>;
  /**
   * Packed `uint32` reference selection, one per row; non-zero rows form the reference. The
   * reference vector is the mean of their standardized attributes, so a single selected row
   * compares everything to that row. With no valid selected row every row is unranked.
   */
  selection: GraphDataView<'uint32'>;
  /** Per-frame float32 view written with `getGPUSimilarLocationsParameterValues`. */
  parameters: GraphDataView<'float32'>;
  /** Standardization of the attribute columns. Defaults to `'zscore'`. */
  standardization?: GPUSimilarLocationsStandardization;
  /** Compile-time capacity of `output.topIds`, at most `rowCount`. Defaults to `min(rowCount, 32)`. */
  maximumResultCount?: number;
  /** Caller-owned outputs. */
  output: GPUSimilarLocationsOutput;
};

/**
 * Feature-space nearest neighbors: ranks every row by its weighted Euclidean distance in
 * standardized attribute space to a reference row, or to the mean of several reference rows
 * (the CARTO `FIND_SIMILAR_LOCATIONS` and ArcGIS Similarity Search model).
 *
 * The graph validates rows, standardizes the columns (z-score or rank), reduces the reference
 * vector, measures distances and orders them with the stable `GPUSort` on the f32 bit pattern of
 * the distance, so ties keep the lowest row ID. No float atomics are used and results are
 * deterministic. Rows are ranked only when valid, unmasked and not excluded as references.
 *
 * Matches a float64 CPU oracle to f32 rounding; rows whose distances differ by less than that
 * rounding may swap ranks against the oracle.
 */
export class GPUSimilarLocations implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSimilarLocationsProps;
  /** Number of rows. */
  readonly rowCount: number;
  /** Capacity of `output.topIds`. */
  readonly maximumResultCount: number;

  constructor(props: GPUSimilarLocationsProps) {
    this.id = props.id ?? 'similar-locations';
    this.props = props;
    const {id} = this;
    if (!Number.isSafeInteger(props.attributeCount) || props.attributeCount < 1) {
      throw new Error(`${id} attributeCount must be a positive integer`);
    }
    for (const [name, view] of [
      ['attributes', props.attributes],
      ['mask', props.mask],
      ['selection', props.selection],
      ['parameters', props.parameters],
      ['output.ranks', props.output.ranks],
      ['output.distances', props.output.distances],
      ['output.topIds', props.output.topIds],
      ['output.count', props.output.count]
    ] as const) {
      if ((view as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
      }
    }
    validatePackedView(props.attributes, ['float32'], `${id} attributes`);
    if (props.attributes.length % props.attributeCount !== 0) {
      throw new Error(`${id} attributes length must be a multiple of attributeCount`);
    }
    this.rowCount = props.attributes.length / props.attributeCount;
    if (this.rowCount < 1) {
      throw new Error(`${id} needs at least one row`);
    }
    if (props.standardization && !['zscore', 'rank'].includes(props.standardization)) {
      throw new Error(`${id} standardization must be 'zscore' or 'rank'`);
    }
    this.maximumResultCount = props.maximumResultCount ?? Math.min(this.rowCount, 32);
    if (
      !Number.isSafeInteger(this.maximumResultCount) ||
      this.maximumResultCount < 1 ||
      this.maximumResultCount > this.rowCount
    ) {
      throw new Error(`${id} maximumResultCount must be an integer in [1, rowCount]`);
    }
    if (props.mask) {
      validatePackedUint32View(props.mask, `${id} mask`);
      if (props.mask.length !== this.rowCount) {
        throw new Error(`${id} mask length must equal the row count`);
      }
    }
    validatePackedUint32View(props.selection, `${id} selection`);
    if (props.selection.length !== this.rowCount) {
      throw new Error(`${id} selection length must equal the row count`);
    }
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    const parameterLength = getGPUSimilarLocationsParameterLength(props.attributeCount);
    if (props.parameters.length < parameterLength) {
      throw new Error(`${id} parameters must hold ${parameterLength} float32 values`);
    }
    validatePackedUint32View(props.output.ranks, `${id} output.ranks`);
    validatePackedView(props.output.distances, ['float32'], `${id} output.distances`);
    validatePackedUint32View(props.output.topIds, `${id} output.topIds`);
    validatePackedUint32View(props.output.count, `${id} output.count`);
    if (
      props.output.ranks.length < this.rowCount ||
      props.output.distances.length < this.rowCount
    ) {
      throw new Error(`${id} output.ranks and output.distances must hold one row per input row`);
    }
    if (props.output.topIds.length < this.maximumResultCount) {
      throw new Error(`${id} output.topIds must hold maximumResultCount rows`);
    }
    if (props.output.count.length < 1) {
      throw new Error(`${id} output.count must hold one row`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.output.ranks, props.output.distances, props.output.topIds, props.output.count],
      [props.attributes, props.mask, props.selection, props.parameters]
    );
  }

  /** Returns validation, standardization, reference, distance, sort and rank nodes, in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, rowCount, maximumResultCount} = this;
    const {output, attributeCount} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.attributes,
      props.mask,
      props.selection,
      props.parameters,
      output.ranks,
      output.distances,
      output.topIds,
      output.count
    ]);
    const isRank = props.standardization === 'rank';
    const matrixLength = rowCount * attributeCount;
    const validity = createTransientView(graph, `${id}-validity`, 'uint32', rowCount);
    const standardized = createTransientView(graph, `${id}-standardized`, 'float32', matrixLength);
    const reference = createTransientView(graph, `${id}-reference`, 'float32', attributeCount);
    const keys = createTransientView(graph, `${id}-keys`, 'uint32', rowCount);
    const rowIds = createTransientView(graph, `${id}-row-ids`, 'uint32', rowCount);
    const sortedKeys = createTransientView(graph, `${id}-sorted-keys`, 'uint32', rowCount);
    const sortedIds = createTransientView(graph, `${id}-sorted-ids`, 'uint32', rowCount);
    const declarations = `const ROW_COUNT: u32 = ${rowCount}u;
const ATTRIBUTE_COUNT: u32 = ${attributeCount}u;
const MAX_RESULTS: u32 = ${maximumResultCount}u;
const NO_RANK: u32 = 0xffffffffu;
fn isFiniteValue(value: f32) -> bool { return (bitcast<u32>(value) & 0x7fffffffu) < 0x7f800000u; }
fn getNaN() -> f32 {
  var bits = 0x7fc00000u;
  return bitcast<f32>(bits);
}`;
    const nodes: GPUCommandNode<Parameters>[] = [];
    const kernel = (
      name: string,
      bindings: WGSLKernelBinding[],
      invocationCount: number,
      body: string,
      options: {reduction?: boolean; declarations?: string} = {}
    ) =>
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-${name}`,
          operation: OPERATION,
          variant: name,
          bindings,
          // A reduction kernel is one workgroup per output; every lane reaches the barriers.
          invocationCount: options.reduction ? invocationCount * REDUCTION_LANES : invocationCount,
          workgroupSize: options.reduction ? REDUCTION_LANES : 64,
          guardIndex: options.reduction ? false : undefined,
          declarations: `${declarations}\n${options.reduction ? REDUCTION_WGSL : ''}\n${options.declarations ?? ''}`,
          body
        })
      );
    const read = (name: string, view: GraphDataView, type: 'f32' | 'u32'): WGSLKernelBinding => ({
      name,
      view,
      type,
      access: 'read'
    });
    const write = (name: string, view: GraphDataView, type: 'f32' | 'u32'): WGSLKernelBinding => ({
      name,
      view,
      type,
      access: 'read_write'
    });

    kernel(
      'validate',
      [
        read('attributes', props.attributes, 'f32'),
        ...(props.mask ? [read('rowMask', props.mask, 'u32')] : []),
        write('validity', validity, 'u32')
      ],
      rowCount,
      `var isValid = true;
  ${props.mask ? 'isValid = rowMask[rowMaskOffset + index] != 0u;' : ''}
  for (var channel = 0u; channel < ATTRIBUTE_COUNT; channel++) {
    isValid = isValid && isFiniteValue(attributes[attributesOffset + index * ATTRIBUTE_COUNT + channel]);
  }
  validity[validityOffset + index] = select(0u, 1u, isValid);`
    );

    if (isRank && rowCount < RANK_SORT_MINIMUM_ROWS) {
      kernel(
        'standardize-rank',
        [
          read('attributes', props.attributes, 'f32'),
          read('validity', validity, 'u32'),
          write('standardized', standardized, 'f32')
        ],
        matrixLength,
        `let row = index / ATTRIBUTE_COUNT;
  let channel = index % ATTRIBUTE_COUNT;
  if (validity[validityOffset + row] == 0u) {
    standardized[standardizedOffset + index] = getNaN();
    return;
  }
  let value = attributes[attributesOffset + index];
  var less = 0u;
  var equal = 0u;
  var valid = 0u;
  for (var other = 0u; other < ROW_COUNT; other++) {
    if (validity[validityOffset + other] == 0u) {
      continue;
    }
    valid += 1u;
    let otherValue = attributes[attributesOffset + other * ATTRIBUTE_COUNT + channel];
    less += select(0u, 1u, otherValue < value);
    equal += select(0u, 1u, otherValue == value);
  }
  let denominator = max(f32(valid) - 1.0, 1.0);
  standardized[standardizedOffset + index] = (f32(less) + 0.5 * (f32(equal) - 1.0)) / denominator;`
      );
    } else if (isRank) {
      // Average-tie percentile ranks from one stable sort per column: the exact less-than and
      // equal counts are two binary searches on the sorted keys, the same integers (so the same
      // floats) as the all-pairs count.
      const rankKeys = createTransientView(graph, `${id}-rank-keys`, 'uint32', rowCount);
      const rankRowIds = createTransientView(graph, `${id}-rank-row-ids`, 'uint32', rowCount);
      const rankSortedKeys = createTransientView(
        graph,
        `${id}-rank-sorted-keys`,
        'uint32',
        rowCount
      );
      const rankSortedIds = createTransientView(graph, `${id}-rank-sorted-ids`, 'uint32', rowCount);
      for (let channel = 0; channel < attributeCount; channel++) {
        kernel(
          `rank-keys-${channel}`,
          [
            read('attributes', props.attributes, 'f32'),
            read('validity', validity, 'u32'),
            write('rankKeys', rankKeys, 'u32'),
            write('rankRowIds', rankRowIds, 'u32')
          ],
          rowCount,
          `let value = attributes[attributesOffset + index * ATTRIBUTE_COUNT + ${channel}u];
  rankKeys[rankKeysOffset + index] =
    select(INVALID_KEY, getOrderedKey(value), validity[validityOffset + index] != 0u);
  rankRowIds[rankRowIdsOffset + index] = index;`,
          {declarations: RANK_KEY_WGSL}
        );
        nodes.push(
          ...new GPUSort({
            id: `${id}-rank-sort-${channel}`,
            keys: rankKeys,
            values: rankRowIds,
            outputKeys: rankSortedKeys,
            outputValues: rankSortedIds,
            keyBits: 32
          }).getCommandNodes(graph)
        );
        kernel(
          `rank-assign-${channel}`,
          [
            read('sortedKeys', rankSortedKeys, 'u32'),
            read('sortedIds', rankSortedIds, 'u32'),
            write('standardized', standardized, 'f32')
          ],
          rowCount,
          `let key = sortedKeys[sortedKeysOffset + index];
  let row = sortedIds[sortedIdsOffset + index];
  if (key == INVALID_KEY) {
    standardized[standardizedOffset + row * ATTRIBUTE_COUNT + ${channel}u] = getNaN();
    return;
  }
  let below = lowerBound(key);
  let notAbove = lowerBound(key + 1u);
  let valid = lowerBound(INVALID_KEY);
  let denominator = max(f32(valid) - 1.0, 1.0);
  standardized[standardizedOffset + row * ATTRIBUTE_COUNT + ${channel}u] =
    (f32(below) + 0.5 * (f32(notAbove - below) - 1.0)) / denominator;`,
          {
            declarations: `${RANK_KEY_WGSL}
fn lowerBound(bound: u32) -> u32 {
  var low = 0u;
  var high = ROW_COUNT;
  while (low < high) {
    let middle = low + (high - low) / 2u;
    if (sortedKeys[sortedKeysOffset + middle] < bound) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  return low;
}`
          }
        );
      }
    } else {
      const stats = createTransientView(graph, `${id}-stats`, 'float32', 2 * attributeCount);
      kernel(
        'column-statistics',
        [
          read('attributes', props.attributes, 'f32'),
          read('validity', validity, 'u32'),
          write('stats', stats, 'f32')
        ],
        attributeCount,
        `let column = index / LANES;
  let lane = localInvocationIndex;
  var validPartial = 0u;
  var sumPartial = 0.0;
  for (var row = lane; row < ROW_COUNT; row += LANES) {
    if (validity[validityOffset + row] != 0u) {
      validPartial += 1u;
      sumPartial += attributes[attributesOffset + row * ATTRIBUTE_COUNT + column];
    }
  }
  let valid = reduceCount(validPartial, lane);
  let mean = select(0.0, reduceSum(sumPartial, lane) / f32(valid), valid > 0u);
  var squaresPartial = 0.0;
  for (var row = lane; row < ROW_COUNT; row += LANES) {
    if (validity[validityOffset + row] != 0u) {
      let delta = attributes[attributesOffset + row * ATTRIBUTE_COUNT + column] - mean;
      squaresPartial += delta * delta;
    }
  }
  let squares = reduceSum(squaresPartial, lane);
  if (lane == 0u) {
    let deviation = select(0.0, sqrt(squares / f32(valid)), valid > 0u);
    stats[statsOffset + 2u * column] = mean;
    stats[statsOffset + 2u * column + 1u] = select(0.0, 1.0 / deviation, deviation > 0.0);
  }`,
        {reduction: true}
      );
      kernel(
        'standardize-zscore',
        [
          read('attributes', props.attributes, 'f32'),
          read('validity', validity, 'u32'),
          read('stats', stats, 'f32'),
          write('standardized', standardized, 'f32')
        ],
        matrixLength,
        `let row = index / ATTRIBUTE_COUNT;
  let channel = index % ATTRIBUTE_COUNT;
  if (validity[validityOffset + row] == 0u) {
    standardized[standardizedOffset + index] = getNaN();
    return;
  }
  standardized[standardizedOffset + index] =
    (attributes[attributesOffset + index] - stats[statsOffset + 2u * channel]) *
    stats[statsOffset + 2u * channel + 1u];`
      );
    }

    kernel(
      'reference',
      [
        read('standardized', standardized, 'f32'),
        read('validity', validity, 'u32'),
        read('selection', props.selection, 'u32'),
        write('reference', reference, 'f32')
      ],
      attributeCount,
      `let column = index / LANES;
  let lane = localInvocationIndex;
  var countPartial = 0u;
  var sumPartial = 0.0;
  for (var row = lane; row < ROW_COUNT; row += LANES) {
    if (validity[validityOffset + row] != 0u && selection[selectionOffset + row] != 0u) {
      countPartial += 1u;
      sumPartial += standardized[standardizedOffset + row * ATTRIBUTE_COUNT + column];
    }
  }
  let count = reduceCount(countPartial, lane);
  let sum = reduceSum(sumPartial, lane);
  if (lane == 0u) {
    reference[referenceOffset + column] = select(getNaN(), sum / f32(count), count > 0u);
  }`,
      {reduction: true}
    );

    kernel(
      'distance',
      [
        read('standardized', standardized, 'f32'),
        read('validity', validity, 'u32'),
        read('selection', props.selection, 'u32'),
        read('reference', reference, 'f32'),
        read('params', props.parameters, 'f32'),
        write('keys', keys, 'u32'),
        write('rowIds', rowIds, 'u32'),
        write('distances', output.distances, 'f32')
      ],
      rowCount,
      `rowIds[rowIdsOffset + index] = index;
  let isLeast = params[paramsOffset + 1u] > 0.5;
  let excludeReference = params[paramsOffset + 2u] > 0.5;
  var eligible = validity[validityOffset + index] != 0u &&
    !(excludeReference && selection[selectionOffset + index] != 0u);
  var distanceSquared = 0.0;
  for (var channel = 0u; channel < ATTRIBUTE_COUNT; channel++) {
    let weight = params[paramsOffset + ${GPU_SIMILAR_LOCATIONS_CONTROL_LENGTH}u + channel];
    let delta = standardized[standardizedOffset + index * ATTRIBUTE_COUNT + channel] -
      reference[referenceOffset + channel];
    distanceSquared += weight * delta * delta;
  }
  eligible = eligible && isFiniteValue(distanceSquared);
  let distance = sqrt(distanceSquared);
  // Non-negative f32 bit patterns order like the values; the sentinel sorts after every key.
  let bits = bitcast<u32>(distance);
  keys[keysOffset + index] = select(NO_RANK, select(bits, 0xfffffffeu - bits, isLeast), eligible);
  distances[distancesOffset + index] = select(getNaN(), distance, eligible);`
    );

    nodes.push(
      ...new GPUSort({
        id: `${id}-sort`,
        keys,
        values: rowIds,
        outputKeys: sortedKeys,
        outputValues: sortedIds,
        keyBits: 32
      }).getCommandNodes(graph)
    );

    kernel(
      'rank',
      [
        read('sortedKeys', sortedKeys, 'u32'),
        read('sortedIds', sortedIds, 'u32'),
        read('params', props.parameters, 'f32'),
        write('ranks', output.ranks, 'u32'),
        write('topIds', output.topIds, 'u32'),
        write('resultCount', output.count, 'u32')
      ],
      rowCount,
      `let key = sortedKeys[sortedKeysOffset + index];
  let isEligible = key != NO_RANK;
  let row = sortedIds[sortedIdsOffset + index];
  ranks[ranksOffset + row] = select(NO_RANK, index, isEligible);
  let requested = params[paramsOffset];
  let limit = min(u32(clamp(select(0.0, requested, isFiniteValue(requested)), 0.0, f32(MAX_RESULTS))), MAX_RESULTS);
  // Eligible rows sort first, so the boundary row owns the eligible count.
  let isLast = index + 1u == ROW_COUNT || sortedKeys[sortedKeysOffset + index + 1u] == NO_RANK;
  if (isEligible && isLast) {
    resultCount[resultCountOffset] = min(index + 1u, limit);
  }
  if (index == 0u && !isEligible) {
    resultCount[resultCountOffset] = 0u;
  }
  if (index < MAX_RESULTS) {
    topIds[topIdsOffset + index] = select(NO_RANK, row, isEligible && index < limit);
  }`
    );
    return nodes;
  }
}
