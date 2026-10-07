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
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {
  type GPUSpatialWeights,
  validateGPUSpatialWeights
} from '../spatial-weights/spatial-weights';

const OPERATION = 'GPURegionPartitionEvaluation';
/** Rows or labels reduced by one thread of a summary tile pass. */
const SUMMARY_TILE = 256;

/** Largest supported attribute column count. */
export const GPU_REGION_PARTITION_EVALUATION_MAXIMUM_COLUMNS = 32;

/** Layout of the `summary` words written by {@link GPURegionPartitionEvaluation}. */
export const GPU_REGION_PARTITION_EVALUATION_LAYOUT = {
  /** Number of non-empty regions. */
  regionCount: 0,
  /** Sum over regions of the within-region sum of squared deviations from the region mean. */
  withinSsd: 1,
  /** `totalSsd - withinSsd`: the sum of squares explained by the partition. */
  betweenSsd: 2,
  /** Sum of squared deviations from the global mean over the labelled rows. */
  totalSsd: 3,
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
  /** Total word count. */
  length: 8
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
  /** Number of distinct label values `[0, labelCapacity)`. Defaults to the row count. */
  labelCapacity?: number;
  /** Optional weights for `crossLinkFraction` (a compactness proxy on the contiguity graph). */
  weights?: GPUSpatialWeights;
  /**
   * Caller-owned summary words, see {@link GPU_REGION_PARTITION_EVALUATION_LAYOUT}
   * (`length` float32 entries).
   */
  summary: GraphDataView<'float32'>;
  /** Optional caller-owned row count per label (`labelCapacity` entries). */
  regionSizes?: GraphDataView<'uint32'>;
  /** Optional caller-owned within-region SSD per label (`labelCapacity` entries). */
  regionWithinSsd?: GraphDataView<'float32'>;
};

/**
 * Evaluates a region partition on any label column: within-region and between-region sum of
 * squares, region sizes and a contiguity-graph boundary measure.
 *
 * Rows are grouped by label with a stable radix sort (ascending row inside a label). Each label is
 * then reduced by one thread over its own rows (mean, then deviations). The summary is a tiled
 * reduction of 256 rows or labels per thread merged in tile order, so the result is deterministic
 * and matches a sequential CPU evaluation within f32 rounding. Work is `O(rows)` plus the sort, independent of `labelCapacity`;
 * a few huge regions still serialise on one thread each.
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
    if (
      !Number.isSafeInteger(columnCount) ||
      columnCount < 1 ||
      columnCount > GPU_REGION_PARTITION_EVALUATION_MAXIMUM_COLUMNS
    ) {
      throw new Error(
        `${id} columnCount must be an integer in [1, ${GPU_REGION_PARTITION_EVALUATION_MAXIMUM_COLUMNS}]`
      );
    }
    validatePackedView(props.values, ['float32'], `${id} values`);
    validatePackedUint32View(props.labels, `${id} labels`);
    const rows = props.labels.length;
    if (rows < 1 || props.values.length !== rows * columnCount) {
      throw new Error(`${id} values length must equal labels.length * columnCount`);
    }
    const labelCapacity = props.labelCapacity ?? rows;
    if (!Number.isSafeInteger(labelCapacity) || labelCapacity < 1) {
      throw new Error(`${id} labelCapacity must be a positive integer`);
    }
    if (props.weights && validateGPUSpatialWeights(id, props.weights) !== rows) {
      throw new Error(`${id} weights row count must equal labels.length`);
    }
    validatePackedView(props.summary, ['float32'], `${id} summary`);
    if (props.summary.length < GPU_REGION_PARTITION_EVALUATION_LAYOUT.length) {
      throw new Error(
        `${id} summary must hold ${GPU_REGION_PARTITION_EVALUATION_LAYOUT.length} words`
      );
    }
    if (props.regionSizes) {
      validatePackedUint32View(props.regionSizes, `${id} regionSizes`);
      if (props.regionSizes.length < labelCapacity) {
        throw new Error(`${id} regionSizes must hold labelCapacity entries`);
      }
    }
    if (props.regionWithinSsd) {
      validatePackedView(props.regionWithinSsd, ['float32'], `${id} regionWithinSsd`);
      if (props.regionWithinSsd.length < labelCapacity) {
        throw new Error(`${id} regionWithinSsd must hold labelCapacity entries`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.summary, props.regionSizes, props.regionWithinSsd],
      [props.values, props.labels, props.weights?.offsets, props.weights?.neighbors]
    );
  }

  /** Returns the per-label, per-link and summary nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {values, labels, weights} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      values,
      labels,
      props.summary,
      props.regionSizes,
      props.regionWithinSsd,
      weights?.offsets,
      weights?.neighbors
    ]);
    const rows = labels.length;
    const columnCount = props.columnCount ?? 1;
    const labelCapacity = props.labelCapacity ?? rows;
    const sizes =
      props.regionSizes ?? createTransientView(graph, `${id}-sizes`, 'uint32', labelCapacity);
    const within =
      props.regionWithinSsd ?? createTransientView(graph, `${id}-within`, 'float32', labelCapacity);
    const crossLinks = weights
      ? createTransientView(graph, `${id}-cross-links`, 'uint32', rows)
      : undefined;
    const totalLinks = weights
      ? createTransientView(graph, `${id}-total-links`, 'uint32', rows)
      : undefined;
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
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-regions`,
        operation: OPERATION,
        variant: 'regions',
        bindings: [
          {name: 'values', view: values, type: 'f32', access: 'read'},
          {name: 'sortedKeys', view: sortedKeys, type: 'u32', access: 'read'},
          {name: 'sortedRows', view: sortedRows, type: 'u32', access: 'read'},
          {name: 'sizes', view: sizes, type: 'u32', access: 'read_write'},
          {name: 'within', view: within, type: 'f32', access: 'read_write'}
        ],
        invocationCount: labelCapacity,
        declarations,
        body: `// Sorted range of this label: first position with key >= label, and with key > label.
  var low = 0u;
  var high = ROWS;
  while (low < high) {
    let middle = low + (high - low) / 2u;
    if (sortedKeys[sortedKeysOffset + middle] < index) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  let first = low;
  high = ROWS;
  while (low < high) {
    let middle = low + (high - low) / 2u;
    if (sortedKeys[sortedKeysOffset + middle] <= index) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  let end = low;
  let count = end - first;
  var sums: array<f32, ${columnCount}>;
  for (var slot = first; slot < end; slot++) {
    let row = sortedRows[sortedRowsOffset + slot];
    for (var column = 0u; column < COLUMNS; column++) {
      sums[column] += values[valuesOffset + row * COLUMNS + column];
    }
  }
  var ssd = 0.0;
  for (var slot = first; slot < end; slot++) {
    let row = sortedRows[sortedRowsOffset + slot];
    for (var column = 0u; column < COLUMNS; column++) {
      let delta = values[valuesOffset + row * COLUMNS + column] - sums[column] / f32(count);
      ssd += delta * delta;
    }
  }
  sizes[sizesOffset + index] = count;
  within[withinOffset + index] = ssd;`
      })
    ];
    if (weights && crossLinks && totalLinks) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-links`,
          operation: OPERATION,
          variant: 'links',
          bindings: [
            {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
            {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
            {name: 'labels', view: labels, type: 'u32', access: 'read'},
            {name: 'crossLinks', view: crossLinks, type: 'u32', access: 'read_write'},
            {name: 'totalLinks', view: totalLinks, type: 'u32', access: 'read_write'}
          ],
          invocationCount: rows,
          declarations,
          body: `var cross = 0u;
  var total = 0u;
  let own = labels[labelsOffset + index];
  if (own < LABELS) {
    for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
      let other = labels[labelsOffset + neighbors[neighborsOffset + slot]];
      if (other < LABELS) {
        total += 1u;
        cross += select(0u, 1u, other != own);
      }
    }
  }
  crossLinks[crossLinksOffset + index] = cross;
  totalLinks[totalLinksOffset + index] = total;`
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
          ...(crossLinks && totalLinks
            ? [
                {
                  name: 'crossLinks',
                  view: crossLinks,
                  type: 'u32' as const,
                  access: 'read' as const
                },
                {
                  name: 'totalLinks',
                  view: totalLinks,
                  type: 'u32' as const,
                  access: 'read' as const
                }
              ]
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
      ${crossLinks ? 'cross += crossLinks[crossLinksOffset + row]; total += totalLinks[totalLinksOffset + row];' : ''}
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
          {name: 'summary', view: props.summary, type: 'f32', access: 'read_write'}
        ],
        invocationCount: 1,
        declarations: tileDeclarations,
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
  summary[summaryOffset + 7u] = f32(ignored);`
      })
    );
    return nodes;
  }
}
