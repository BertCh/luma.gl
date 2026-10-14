// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createLoadBalancedSegmentSumNodes} from '../../utils/load-balanced-for-each';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {
  getStorageBindingElementLimit,
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {
  type GPUSpatialWeights,
  validateGPUSpatialWeights
} from '../spatial-weights/spatial-weights';
import {GET_NAN_WGSL} from '../../utils/wgsl-common';

const OPERATION = 'GPURegionPartitionEvaluation';
/** Rows or labels reduced by one thread of a summary tile pass. */
const SUMMARY_TILE = 256;

/** Largest supported attribute column count. */
export const GPU_REGION_PARTITION_EVALUATION_MAXIMUM_COLUMNS = 32;

/**
 * Largest `labels` row count of {@link GPURegionPartitionEvaluation} on `device`: the
 * contiguity-link pass writes two uint32 link counts (8 bytes) per row in one storage binding, so
 * 16777216 on a default device (128 MiB bindings).
 *
 * @param device Device whose `maxStorageBufferBindingSize` applies.
 */
export function getGPURegionPartitionEvaluationMaximumRowCount(device: Device): number {
  return getStorageBindingElementLimit(device, 8);
}

/**
 * Largest `labelCapacity` of {@link GPURegionPartitionEvaluation} on `device`: the label offsets
 * hold `labelCapacity + 1` uint32 values in one storage binding, so 33554431 on a default device
 * (128 MiB bindings). The per-region column sums also bound `labelCapacity * columnCount` by
 * {@link getGPURegionPartitionEvaluationMaximumRegionValueCount}.
 *
 * @param device Device whose `maxStorageBufferBindingSize` applies.
 */
export function getGPURegionPartitionEvaluationMaximumLabelCapacity(device: Device): number {
  return getStorageBindingElementLimit(device, 4, 1);
}

/**
 * Largest `labelCapacity * columnCount` of {@link GPURegionPartitionEvaluation} on `device`: one
 * float32 region sum per label and column in one storage binding, so 33554432 on a default device
 * (128 MiB bindings).
 *
 * @param device Device whose `maxStorageBufferBindingSize` applies.
 */
export function getGPURegionPartitionEvaluationMaximumRegionValueCount(device: Device): number {
  return getStorageBindingElementLimit(device, 4);
}

/** Layout of the `output.summary` words written by {@link GPURegionPartitionEvaluation}. */
export const GPU_REGION_PARTITION_EVALUATION_LAYOUT = {
  /** Number of non-empty regions. */
  regionCount: 0,
  /** Sum over regions of the within-region sum of squared deviations from the region mean. */
  withinSumOfSquares: 1,
  /** `totalSumOfSquares - withinSumOfSquares`: the sum of squares explained by the partition. */
  betweenSumOfSquares: 2,
  /** Sum of squared deviations from the global mean over the labelled rows. */
  totalSumOfSquares: 3,
  /** Smallest non-empty region size in rows. */
  minimumSize: 4,
  /** Largest region size in rows. */
  maximumSize: 5,
  /**
   * Fraction of weights links (between two labelled rows) whose endpoints lie in different
   * regions: 0 for a partition with no boundary, higher for ragged regions. 0 without `weights`.
   */
  crossLinkFraction: 6,
  /** Rows whose label is `0xffffffff` or at least `labelCapacity`, left out of every statistic. */
  ignoredRows: 7,
  /**
   * Pseudo-F (Calinski-Harabasz) statistic
   * `(betweenSumOfSquares / (k - 1)) / (withinSumOfSquares / (n - k))` with `k` the region count
   * and `n` the labelled rows, the `sklearn.metrics.calinski_harabasz_score` convention (1 when
   * `withinSumOfSquares` is 0). NaN when `k < 2` or `n <= k`, where it is undefined.
   * Larger is a better separated partition; it compares partitions of the same table.
   */
  pseudoF: 8,
  /** Total word count. */
  length: 9
} as const;

/**
 * Properties for {@link GPURegionPartitionEvaluation}.
 *
 * Compile-time: row count, `columnCount`, `labelCapacity` and which outputs exist. Per-frame: the
 * contents of `values`, `labels` and `weights`.
 */
export type GPURegionPartitionEvaluationProps = {
  /** Prefix for generated node IDs. Defaults to `'region-partition-evaluation'`. */
  id?: string;
  /** Attribute table, row-major with `columnCount` columns (`rows * columnCount` entries). */
  values: GraphDataView<'float32'>;
  /** Number of attribute columns. Defaults to 1; at most 32. */
  columnCount?: number;
  /**
   * Region label per row. Any label column works: `GPUSkaterRegions`, a clustering, a
   * classification or an administrative code. Labels at or above `labelCapacity` and
   * `0xffffffff` (unlabelled) are ignored.
   */
  labels: GraphDataView<'uint32'>;
  /**
   * Number of distinct label values `[0, labelCapacity)`. Defaults to the row count. At most
   * {@link getGPURegionPartitionEvaluationMaximumLabelCapacity}, and `labelCapacity * columnCount`
   * at most {@link getGPURegionPartitionEvaluationMaximumRegionValueCount}.
   */
  labelCapacity?: number;
  /** Optional weights for `crossLinkFraction` (a compactness proxy on the contiguity graph). */
  weights?: GPUSpatialWeights;
  /** Caller-owned outputs. */
  output: GPURegionPartitionEvaluationOutput;
};

/** Caller-owned outputs of {@link GPURegionPartitionEvaluation}. */
export type GPURegionPartitionEvaluationOutput = {
  /**
   * Summary words, see {@link GPU_REGION_PARTITION_EVALUATION_LAYOUT} (`length` float32
   * entries).
   */
  summary: GraphDataView<'float32'>;
  /** Optional row count per label (`labelCapacity` entries). */
  regionSizes?: GraphDataView<'uint32'>;
  /** Optional within-region sum of squared deviations per label (`labelCapacity` entries). */
  regionWithinSumOfSquares?: GraphDataView<'float32'>;
};

/**
 * Evaluates a region partition on any label column: within-region and between-region sum of
 * squares, the pseudo-F (Calinski-Harabasz) statistic, region sizes and a contiguity-graph
 * boundary measure. `GPUMaxPRegions`, `GPUAZPRegions` and `GPUWardRegions` report the same
 * within-region sum of squares as their `objective`, so pass the table they used (the standardized
 * one unless they ran with `standardize: false`) to compare.
 *
 * Rows are grouped by label with a stable radix sort (ascending row inside a label), so every
 * label is a segment of the sorted rows. Its column sums, then its squared deviations about its
 * mean, are load-balanced segment sums (merge-path partitioning over the label segments), and the
 * contiguity link counts are load-balanced over the weights rows, so a huge region or a hub row
 * spreads over many workgroups instead of serialising one thread. The summary is a tiled
 * reduction of 256 rows or labels per thread merged in tile order. Every association depends
 * only on the labels and offsets, so the result is bitwise reproducible and matches a sequential
 * CPU evaluation within f32 rounding. Work is `O(rows + links)` plus the sort, independent of
 * `labelCapacity` and of the region size distribution.
 */
export class GPURegionPartitionEvaluation implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPURegionPartitionEvaluationProps;

  constructor(props: GPURegionPartitionEvaluationProps) {
    const id = props.id ?? 'region-partition-evaluation';
    this.id = id;
    this.props = props;
    const columnCount = props.columnCount ?? 1;
    const {output} = props;
    validateCapacity(
      id,
      'columnCount',
      columnCount,
      1,
      GPU_REGION_PARTITION_EVALUATION_MAXIMUM_COLUMNS
    );
    validatePackedView(props.values, ['float32'], `${id} values`);
    validatePackedView(props.labels, ['uint32'], `${id} labels`);
    const rows = props.labels.length;
    if (rows < 1 || props.values.length !== rows * columnCount) {
      throw new Error(`${id} values length must equal labels.length * columnCount`);
    }
    const labelCapacity = props.labelCapacity ?? rows;
    validateCapacity(id, 'labelCapacity', labelCapacity);
    if (props.weights && validateGPUSpatialWeights(id, props.weights) !== rows) {
      throw new Error(`${id} weights row count must equal labels.length`);
    }
    validatePackedView(output.summary, ['float32'], `${id} output.summary`);
    if (output.summary.length < GPU_REGION_PARTITION_EVALUATION_LAYOUT.length) {
      throw new Error(
        `${id} output.summary must hold ${GPU_REGION_PARTITION_EVALUATION_LAYOUT.length} words`
      );
    }
    if (output.regionSizes) {
      validatePackedView(output.regionSizes, ['uint32'], `${id} output.regionSizes`);
      if (output.regionSizes.length < labelCapacity) {
        throw new Error(`${id} output.regionSizes must hold labelCapacity entries`);
      }
    }
    if (output.regionWithinSumOfSquares) {
      validatePackedView(
        output.regionWithinSumOfSquares,
        ['float32'],
        `${id} output.regionWithinSumOfSquares`
      );
      if (output.regionWithinSumOfSquares.length < labelCapacity) {
        throw new Error(`${id} output.regionWithinSumOfSquares must hold labelCapacity entries`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [output.summary, output.regionSizes, output.regionWithinSumOfSquares],
      [props.values, props.labels, props.weights?.offsets, props.weights?.neighbors]
    );
  }

  /** Returns the per-label, per-link and summary nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {values, labels, weights, output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      values,
      labels,
      output.summary,
      output.regionSizes,
      output.regionWithinSumOfSquares,
      weights?.offsets,
      weights?.neighbors
    ]);
    const rows = labels.length;
    const columnCount = props.columnCount ?? 1;
    const labelCapacity = props.labelCapacity ?? rows;
    validateMaximumCount(
      id,
      'labels rows',
      rows,
      getGPURegionPartitionEvaluationMaximumRowCount(graph.device)
    );
    validateCapacity(
      id,
      'labelCapacity',
      labelCapacity,
      1,
      getGPURegionPartitionEvaluationMaximumLabelCapacity(graph.device)
    );
    validateMaximumCount(
      id,
      'labelCapacity x columnCount',
      labelCapacity * columnCount,
      getGPURegionPartitionEvaluationMaximumRegionValueCount(graph.device)
    );
    const sizes =
      output.regionSizes ?? createTransientView(graph, `${id}-sizes`, 'uint32', labelCapacity);
    const within =
      output.regionWithinSumOfSquares ??
      createTransientView(graph, `${id}-within`, 'float32', labelCapacity);
    // Interleaved (total, cross) link counts per row.
    const links = weights
      ? createTransientView(graph, `${id}-link-counts`, 'uint32', 2 * rows)
      : undefined;
    const labelOffsets = createTransientView(
      graph,
      `${id}-label-offsets`,
      'uint32',
      labelCapacity + 1
    );
    const regionSums = createTransientView(
      graph,
      `${id}-region-column-sums`,
      'float32',
      labelCapacity * columnCount
    );
    const regionDeviations = createTransientView(
      graph,
      `${id}-region-column-deviations`,
      'float32',
      labelCapacity * columnCount
    );
    const declarations = `const ROWS: u32 = ${rows}u;
const COLUMNS: u32 = ${columnCount}u;
const LABELS: u32 = ${labelCapacity}u;`;
    // Rows grouped by label with a stable sort (ascending row inside a label), so each label's
    // thread reads only its own rows, in the same order as a scan of all rows would, instead of
    // every label scanning every row (`labelCapacity * rows`, quadratic with the default capacity).
    const sortKeys = createTransientView(graph, `${id}-sort-keys`, 'uint32', rows);
    const rowIds = createTransientView(graph, `${id}-row-ids`, 'uint32', rows);
    const sortedKeys = createTransientView(graph, `${id}-sorted-keys`, 'uint32', rows);
    const sortedRows = createTransientView(graph, `${id}-sorted-rows`, 'uint32', rows);
    // Key `labelCapacity` collects the ignored rows (out of range or unlabelled).
    const keyBits = Math.max(1, Math.ceil(Math.log2(labelCapacity + 1)));
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-sort-keys`,
        operation: OPERATION,
        variant: 'sort-keys',
        bindings: [
          {name: 'labels', view: labels, type: 'u32', access: 'read'},
          {name: 'sortKeys', view: sortKeys, type: 'u32', access: 'read_write'},
          {name: 'rowIds', view: rowIds, type: 'u32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations,
        body: `let label = labels[labelsOffset + index];
  sortKeys[sortKeysOffset + index] = select(LABELS, label, label < LABELS);
  rowIds[rowIdsOffset + index] = index;`
      }),
      ...new GPUSort({
        id: `${id}-sort`,
        keys: sortKeys,
        values: rowIds,
        outputKeys: sortedKeys,
        outputValues: sortedRows,
        keyBits
      }).getCommandNodes(graph),
      // Label segments of the sorted rows: offsets[label] is the first position with key >= label,
      // offsets[LABELS] the first ignored row, so the labels form CSR rows over the sorted rows.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-label-offsets`,
        operation: OPERATION,
        variant: 'label-offsets',
        bindings: [
          {name: 'sortedKeys', view: sortedKeys, type: 'u32', access: 'read'},
          {name: 'labelOffsets', view: labelOffsets, type: 'u32', access: 'read_write'},
          {name: 'sizes', view: sizes, type: 'u32', access: 'read_write'}
        ],
        invocationCount: labelCapacity,
        declarations: `${declarations}
fn getLowerBound(key: u32) -> u32 {
  var low = 0u;
  var high = ROWS;
  while (low < high) {
    let middle = low + (high - low) / 2u;
    if (sortedKeys[sortedKeysOffset + middle] < key) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  return low;
}`,
        body: `let first = getLowerBound(index);
  let end = getLowerBound(index + 1u);
  labelOffsets[labelOffsetsOffset + index] = first;
  sizes[sizesOffset + index] = end - first;
  if (index + 1u == LABELS) {
    labelOffsets[labelOffsetsOffset + LABELS] = end;
  }`
      }),
      // Per-label column sums, then squared deviations about the label means, as load-balanced
      // segment sums over the label segments (algorithms roadmap decision 3), so one huge region
      // spreads over many workgroups instead of serialising one thread.
      ...createLoadBalancedSegmentSumNodes<Parameters>(graph, {
        id: `${id}-region-sums`,
        operation: OPERATION,
        offsets: labelOffsets,
        itemCapacity: rows,
        laneCount: columnCount,
        accumulatorType: 'f32',
        output: regionSums,
        bindings: [
          {name: 'values', view: values, type: 'f32', access: 'read'},
          {name: 'sortedRows', view: sortedRows, type: 'u32', access: 'read'}
        ],
        declarations: `${declarations}
fn getLoadBalancedContribution(label: u32, item: u32, column: u32) -> f32 {
  return values[valuesOffset + sortedRows[sortedRowsOffset + item] * COLUMNS + column];
}`
      }),
      ...createLoadBalancedSegmentSumNodes<Parameters>(graph, {
        id: `${id}-region-deviations`,
        operation: OPERATION,
        offsets: labelOffsets,
        itemCapacity: rows,
        laneCount: columnCount,
        accumulatorType: 'f32',
        output: regionDeviations,
        bindings: [
          {name: 'values', view: values, type: 'f32', access: 'read'},
          {name: 'sortedRows', view: sortedRows, type: 'u32', access: 'read'},
          {name: 'regionSums', view: regionSums, type: 'f32', access: 'read'},
          {name: 'sizes', view: sizes, type: 'u32', access: 'read'}
        ],
        declarations: `${declarations}
fn getLoadBalancedContribution(label: u32, item: u32, column: u32) -> f32 {
  // A visited label has at least one row, so the count is positive.
  let mean = regionSums[regionSumsOffset + label * COLUMNS + column] / f32(sizes[sizesOffset + label]);
  let delta = values[valuesOffset + sortedRows[sortedRowsOffset + item] * COLUMNS + column] - mean;
  return delta * delta;
}`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-region-within`,
        operation: OPERATION,
        variant: 'region-within',
        bindings: [
          {name: 'regionDeviations', view: regionDeviations, type: 'f32', access: 'read'},
          {name: 'within', view: within, type: 'f32', access: 'read_write'}
        ],
        invocationCount: labelCapacity,
        declarations,
        body: `var ssd = 0.0;
  for (var column = 0u; column < COLUMNS; column++) {
    ssd += regionDeviations[regionDeviationsOffset + index * COLUMNS + column];
  }
  within[withinOffset + index] = ssd;`
      })
    ];
    if (weights && links) {
      // Per-row (total, cross) link counts, load balanced over the weights rows (hub rows).
      nodes.push(
        ...createLoadBalancedSegmentSumNodes<Parameters>(graph, {
          id: `${id}-links`,
          operation: OPERATION,
          offsets: weights.offsets,
          itemCapacity: weights.neighbors.length,
          accumulatorType: 'vec2<u32>',
          output: links,
          bindings: [
            {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
            {name: 'labels', view: labels, type: 'u32', access: 'read'}
          ],
          declarations: `${declarations}
fn getLoadBalancedContribution(row: u32, item: u32, lane: u32) -> vec2<u32> {
  let own = labels[labelsOffset + row];
  let other = labels[labelsOffset + neighbors[neighborsOffset + item]];
  if (own >= LABELS || other >= LABELS) {
    return vec2<u32>();
  }
  return vec2<u32>(1u, select(0u, 1u, other != own));
}`
        })
      );
    }
    // The summary is a tiled reduction (fixed tile order, no atomics) rather than one thread
    // walking every row and label: tiles of rows and of labels in parallel, the column means,
    // the deviations about those means, and a short merge over the tiles.
    const rowTileCount = Math.ceil(rows / SUMMARY_TILE);
    const labelTileCount = Math.ceil(labelCapacity / SUMMARY_TILE);
    const rowCounts = createTransientView(graph, `${id}-row-counts`, 'uint32', rowTileCount * 4);
    const rowSums = createTransientView(
      graph,
      `${id}-row-sums`,
      'float32',
      rowTileCount * columnCount
    );
    const labelCounts = createTransientView(
      graph,
      `${id}-label-counts`,
      'uint32',
      labelTileCount * 3
    );
    const labelWithin = createTransientView(graph, `${id}-label-within`, 'float32', labelTileCount);
    const means = createTransientView(graph, `${id}-means`, 'float32', columnCount);
    const deviations = createTransientView(
      graph,
      `${id}-deviations`,
      'float32',
      rowTileCount * columnCount
    );
    const tileDeclarations = `${declarations}
const SUMMARY_TILE: u32 = ${SUMMARY_TILE}u;
const ROW_TILES: u32 = ${rowTileCount}u;
const LABEL_TILES: u32 = ${labelTileCount}u;`;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-row-tiles`,
        operation: OPERATION,
        variant: 'row-tiles',
        bindings: [
          {name: 'values', view: values, type: 'f32', access: 'read'},
          {name: 'labels', view: labels, type: 'u32', access: 'read'},
          ...(links
            ? [{name: 'links', view: links, type: 'u32' as const, access: 'read' as const}]
            : []),
          {name: 'rowCounts', view: rowCounts, type: 'u32', access: 'read_write'},
          {name: 'rowSums', view: rowSums, type: 'f32', access: 'read_write'}
        ],
        invocationCount: rowTileCount,
        declarations: tileDeclarations,
        body: `var labelled = 0u;
  var ignored = 0u;
  var cross = 0u;
  var total = 0u;
  var sums: array<f32, ${columnCount}>;
  let firstRow = index * SUMMARY_TILE;
  for (var row = firstRow; row < min(firstRow + SUMMARY_TILE, ROWS); row++) {
    if (labels[labelsOffset + row] < LABELS) {
      labelled += 1u;
      ${links ? 'total += links[linksOffset + 2u * row]; cross += links[linksOffset + 2u * row + 1u];' : ''}
      for (var column = 0u; column < COLUMNS; column++) {
        sums[column] += values[valuesOffset + row * COLUMNS + column];
      }
    } else {
      ignored += 1u;
    }
  }
  rowCounts[rowCountsOffset + 4u * index] = labelled;
  rowCounts[rowCountsOffset + 4u * index + 1u] = ignored;
  rowCounts[rowCountsOffset + 4u * index + 2u] = cross;
  rowCounts[rowCountsOffset + 4u * index + 3u] = total;
  for (var column = 0u; column < COLUMNS; column++) {
    rowSums[rowSumsOffset + index * COLUMNS + column] = sums[column];
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-label-tiles`,
        operation: OPERATION,
        variant: 'label-tiles',
        bindings: [
          {name: 'sizes', view: sizes, type: 'u32', access: 'read'},
          {name: 'within', view: within, type: 'f32', access: 'read'},
          {name: 'labelCounts', view: labelCounts, type: 'u32', access: 'read_write'},
          {name: 'labelWithin', view: labelWithin, type: 'f32', access: 'read_write'}
        ],
        invocationCount: labelTileCount,
        declarations: tileDeclarations,
        body: `var regionCount = 0u;
  var withinSsd = 0.0;
  var minimumSize = 0xffffffffu;
  var maximumSize = 0u;
  let firstLabel = index * SUMMARY_TILE;
  for (var label = firstLabel; label < min(firstLabel + SUMMARY_TILE, LABELS); label++) {
    let size = sizes[sizesOffset + label];
    if (size > 0u) {
      regionCount += 1u;
      withinSsd += within[withinOffset + label];
      minimumSize = min(minimumSize, size);
      maximumSize = max(maximumSize, size);
    }
  }
  labelCounts[labelCountsOffset + 3u * index] = regionCount;
  labelCounts[labelCountsOffset + 3u * index + 1u] = minimumSize;
  labelCounts[labelCountsOffset + 3u * index + 2u] = maximumSize;
  labelWithin[labelWithinOffset + index] = withinSsd;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-means`,
        operation: OPERATION,
        variant: 'means',
        bindings: [
          {name: 'rowCounts', view: rowCounts, type: 'u32', access: 'read'},
          {name: 'rowSums', view: rowSums, type: 'f32', access: 'read'},
          {name: 'means', view: means, type: 'f32', access: 'read_write'}
        ],
        invocationCount: columnCount,
        declarations: tileDeclarations,
        body: `var labelled = 0u;
  var sum = 0.0;
  for (var tile = 0u; tile < ROW_TILES; tile++) {
    labelled += rowCounts[rowCountsOffset + 4u * tile];
    sum += rowSums[rowSumsOffset + tile * COLUMNS + index];
  }
  means[meansOffset + index] = sum / max(f32(labelled), 1.0);`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-deviation-tiles`,
        operation: OPERATION,
        variant: 'deviation-tiles',
        bindings: [
          {name: 'values', view: values, type: 'f32', access: 'read'},
          {name: 'labels', view: labels, type: 'u32', access: 'read'},
          {name: 'means', view: means, type: 'f32', access: 'read'},
          {name: 'deviations', view: deviations, type: 'f32', access: 'read_write'}
        ],
        invocationCount: rowTileCount,
        declarations: tileDeclarations,
        body: `var squares: array<f32, ${columnCount}>;
  let firstRow = index * SUMMARY_TILE;
  for (var row = firstRow; row < min(firstRow + SUMMARY_TILE, ROWS); row++) {
    if (labels[labelsOffset + row] < LABELS) {
      for (var column = 0u; column < COLUMNS; column++) {
        let delta = values[valuesOffset + row * COLUMNS + column] - means[meansOffset + column];
        squares[column] += delta * delta;
      }
    }
  }
  for (var column = 0u; column < COLUMNS; column++) {
    deviations[deviationsOffset + index * COLUMNS + column] = squares[column];
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-summary`,
        operation: OPERATION,
        variant: 'summary',
        bindings: [
          {name: 'rowCounts', view: rowCounts, type: 'u32', access: 'read'},
          {name: 'labelCounts', view: labelCounts, type: 'u32', access: 'read'},
          {name: 'labelWithin', view: labelWithin, type: 'f32', access: 'read'},
          {name: 'deviations', view: deviations, type: 'f32', access: 'read'},
          {name: 'summary', view: output.summary, type: 'f32', access: 'read_write'}
        ],
        invocationCount: 1,
        declarations: `${tileDeclarations}
${GET_NAN_WGSL}`,
        body: `var regionCount = 0u;
  var withinSsd = 0.0;
  var minimumSize = 0xffffffffu;
  var maximumSize = 0u;
  for (var tile = 0u; tile < LABEL_TILES; tile++) {
    regionCount += labelCounts[labelCountsOffset + 3u * tile];
    withinSsd += labelWithin[labelWithinOffset + tile];
    minimumSize = min(minimumSize, labelCounts[labelCountsOffset + 3u * tile + 1u]);
    maximumSize = max(maximumSize, labelCounts[labelCountsOffset + 3u * tile + 2u]);
  }
  var ignored = 0u;
  var cross = 0u;
  var total = 0u;
  for (var tile = 0u; tile < ROW_TILES; tile++) {
    ignored += rowCounts[rowCountsOffset + 4u * tile + 1u];
    cross += rowCounts[rowCountsOffset + 4u * tile + 2u];
    total += rowCounts[rowCountsOffset + 4u * tile + 3u];
  }
  var totalSsd = 0.0;
  for (var column = 0u; column < COLUMNS; column++) {
    for (var tile = 0u; tile < ROW_TILES; tile++) {
      totalSsd += deviations[deviationsOffset + tile * COLUMNS + column];
    }
  }
  summary[summaryOffset + 0u] = f32(regionCount);
  summary[summaryOffset + 1u] = withinSsd;
  summary[summaryOffset + 2u] = max(totalSsd - withinSsd, 0.0);
  summary[summaryOffset + 3u] = totalSsd;
  summary[summaryOffset + 4u] = f32(select(minimumSize, 0u, regionCount == 0u));
  summary[summaryOffset + 5u] = f32(maximumSize);
  summary[summaryOffset + 6u] = select(0.0, f32(cross) / f32(total), total > 0u);
  summary[summaryOffset + 7u] = f32(ignored);
  // Calinski-Harabasz: undefined below two regions or without more rows than regions.
  let labelledRows = f32(ROWS - ignored);
  let regions = f32(regionCount);
  let between = max(totalSsd - withinSsd, 0.0);
  var pseudoF = getNaN();
  if (regionCount >= 2u && labelledRows > regions) {
    pseudoF = select(between * (labelledRows - regions) / (withinSsd * (regions - 1.0)), 1.0, withinSsd == 0.0);
  }
  summary[summaryOffset + 8u] = pseudoF;`
      })
    );
    return nodes;
  }
}
