// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  GPUSort,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {getSortKeyBits} from '../../utils/sorted-segment-sums';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {
  getOffsetRangeSearchSource,
  GPU_POLYGON_RASTERIZATION_EXTENT_LENGTH,
  GPU_POLYGON_RASTERIZATION_NO_ZONE,
  POLYGON_RASTER_EXTENT_WGSL
} from './polygon-rasterization-parameters';

const OPERATION = 'GPUPolygonRasterization';
const NO_EDGE = '0xffffffffu';

/**
 * Properties for {@link GPUPolygonRasterization}.
 *
 * Per-frame (no recompile): the contents of `extent` and of every polygon buffer. Topology (needs
 * a new graph): `width`, `height`, `crossingCapacity`, every view length, and whether `boundary`
 * and `crossingCount` are present.
 */
export type GPUPolygonRasterizationProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'polygon-rasterization'`. */
  id?: string;
  /** Raster width in cells. Compile-time. */
  width: number;
  /** Raster height in cells. Compile-time. */
  height: number;
  /**
   * Per-frame raster placement `[originX, originY, cellWidth, cellHeight]` (float32, at least 4
   * rows), for example a `GPUParameterBuffer` filled with
   * `getGPUPolygonRasterizationExtentValues`. Row 0 is the row with the smallest y.
   */
  extent: GraphDataView<'float32'>;
  /** Flattened polygon vertices (GeoArrow layout, as `GPUPointInPolygonJoin`). */
  polygonPositions: GraphDataView<'float32x2'>;
  /** Feature-to-polygon offsets with `featureCount + 1` entries, first 0. */
  featureOffsets: GraphDataView<'uint32'>;
  /** Polygon-to-ring offsets with a terminal entry. Ring 0 is the shell; others are holes. */
  polygonOffsets: GraphDataView<'uint32'>;
  /** Ring-to-vertex offsets with a terminal entry. Rings close implicitly. */
  ringOffsets: GraphDataView<'uint32'>;
  /**
   * Maximum (edge, raster row) crossings per encoding. A ring crosses each row its interior covers
   * at least twice, so a safe size is about `2 * (rows spanned) * (rings per row)` summed over
   * polygons; read `crossingCount` to size it. Compile-time.
   */
  crossingCapacity: number;
  /**
   * Output: packed row-major zone raster, `width * height` rows, rewritten on every encoding. Each
   * cell holds the smallest feature row whose polygon contains the cell center, or
   * `GPU_POLYGON_RASTERIZATION_NO_ZONE`.
   */
  zones: GraphDataView<'uint32'>;
  /**
   * Optional output: `width * height` flags, 1 where any polygon edge touches the closed cell
   * (conservatively), otherwise 0. Cells without the flag are entirely inside or outside every
   * polygon, so points in them join exactly like their cell center.
   */
  boundary?: GraphDataView<'uint32'>;
  /**
   * One-row flag: 1 when the crossings exceed `crossingCapacity`. The zone raster is then left at
   * `GPU_POLYGON_RASTERIZATION_NO_ZONE` everywhere rather than partially filled; `boundary` is
   * still complete.
   */
  overflow: GraphDataView<'uint32'>;
  /** Optional one-row unclamped crossing count, for sizing `crossingCapacity`. */
  crossingCount?: GraphDataView<'uint32'>;
};

/**
 * Scan-converts polygons into a dense zone-ID raster on the GPU, for raster joins
 * (`GPURasterJoin`) and raster zonal statistics (`GPURasterZonalStatistics`).
 *
 * Semantics:
 * - A cell is covered by a polygon when its center is inside under the even-odd rule over the
 *   polygon's rings (holes subtract; this matches `GPUPairwisePointInPolygon`). Polygons of one
 *   feature are unioned. When features overlap, the smallest feature row wins (`atomicMin`), as in
 *   `GPUPointInPolygonJoin`.
 * - Centers exactly on an edge follow a half-open rule: an edge spans rows with
 *   `min(y0, y1) <= centerY < max(y0, y1)`, and a cell is inside a span when
 *   `leftCrossing <= centerX < rightCrossing`. Two polygons that share an edge therefore never both
 *   claim a cell on it, and no cell between them is lost.
 * - Polygons smaller than a cell cover the cells whose centers they contain, possibly none. Use
 *   the `boundary` flags to find the points they may still contain.
 *
 * Algorithm (compute only, deterministic):
 * 1. One invocation per vertex (edge to the next ring vertex) finds its ring, polygon and feature
 *    by binary search over the offsets, and counts the raster rows whose center line it crosses.
 * 2. An exclusive scan of the counts gives each edge a slot range; edges emit one
 *    `(polygon, row, column)` record per crossed row, where `column` is the first cell whose center
 *    is at or right of the crossing.
 * 3. A stable radix sort orders records by polygon, row and column (one sort when the key fits 32
 *    bits, otherwise two stable passes), and a segmented scan ranks records within each
 *    `(polygon, row)`.
 * 4. Each even-ranked record pairs with the next one into a half-open span of columns, and every
 *    cell of the span takes `atomicMin(zone, featureRow)`.
 * 5. Optionally, one invocation per edge marks every cell the closed edge segment touches.
 *
 * Cost per encoding: O(V log R) for the edge pass (V vertices, R rings), O(C) emission for C
 * crossings, two or so radix sorts over `crossingCapacity` keys, one segmented scan over
 * `crossingCapacity`, and O(covered cells) atomics in the fill, where one invocation fills one
 * span (up to `width` cells). The boundary pass costs O(cells touched by edges + rows spanned).
 * Every sort and scan always runs over the full capacity, so oversizing it costs time.
 *
 * Precision: coordinates and the extent are float32. Use extent-local or tile-local coordinates.
 * A non-finite vertex drops its two edges, which can corrupt only the rows of its own polygon.
 */
export class GPUPolygonRasterization implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUPolygonRasterizationProps;
  /** Number of features, `featureOffsets.length - 1`. */
  readonly featureCount: number;
  /** Number of polygons, `polygonOffsets.length - 1`. */
  readonly polygonCount: number;
  /** Number of rings, `ringOffsets.length - 1`. */
  readonly ringCount: number;
  /** Whether records sort with one combined 32-bit key (otherwise two stable passes). */
  readonly singleSort: boolean;

  constructor(props: GPUPolygonRasterizationProps) {
    this.id = props.id ?? 'polygon-rasterization';
    this.props = props;
    const {id} = this;
    const {width, height} = props;
    for (const [name, value] of [
      ['width', width],
      ['height', height]
    ] as const) {
      if (!Number.isSafeInteger(value) || value < 1) {
        throw new Error(`${id} ${name} must be a positive integer`);
      }
    }
    if (height * (width + 1) > 0xffffffff) {
      throw new Error(`${id} height * (width + 1) must fit in uint32`);
    }
    validatePackedView(props.extent, ['float32'], `${id} extent`);
    if (props.extent.length < GPU_POLYGON_RASTERIZATION_EXTENT_LENGTH) {
      throw new Error(
        `${id} extent must contain ${GPU_POLYGON_RASTERIZATION_EXTENT_LENGTH} float32 rows`
      );
    }
    validatePackedView(props.polygonPositions, ['float32x2'], `${id} polygonPositions`);
    if (props.polygonPositions.length < 1) {
      throw new Error(`${id} polygonPositions must not be empty`);
    }
    for (const [name, view] of [
      ['featureOffsets', props.featureOffsets],
      ['polygonOffsets', props.polygonOffsets],
      ['ringOffsets', props.ringOffsets]
    ] as const) {
      validatePackedUint32View(view, `${id} ${name}`);
      if (view.length < 2) {
        throw new Error(`${id} ${name} requires at least one range and a terminal entry`);
      }
    }
    this.featureCount = props.featureOffsets.length - 1;
    this.polygonCount = props.polygonOffsets.length - 1;
    this.ringCount = props.ringOffsets.length - 1;
    if (this.featureCount >= GPU_POLYGON_RASTERIZATION_NO_ZONE) {
      throw new Error(`${id} feature count must be below the no-zone sentinel`);
    }
    if (
      !Number.isSafeInteger(props.crossingCapacity) ||
      props.crossingCapacity < 2 ||
      props.crossingCapacity >= 0xffffffff
    ) {
      throw new Error(`${id} crossingCapacity must be an integer of at least 2`);
    }
    const cellCount = width * height;
    for (const [name, view, length] of [
      ['zones', props.zones, cellCount],
      ['boundary', props.boundary, cellCount],
      ['overflow', props.overflow, 1],
      ['crossingCount', props.crossingCount, 1]
    ] as const) {
      if (!view) {
        continue;
      }
      validatePackedUint32View(view, `${id} ${name}`);
      if (view.length !== length) {
        throw new Error(`${id} ${name} must contain ${length} uint32 rows`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.zones, props.boundary, props.overflow, props.crossingCount],
      [
        props.extent,
        props.polygonPositions,
        props.featureOffsets,
        props.polygonOffsets,
        props.ringOffsets
      ]
    );
    this.singleSort =
      getSortKeyBits(this.polygonCount) + getSortKeyBits(cellCount + height - 1) <=
      32;
  }

  /** Returns reset, edge, scan, emit, sort, rank, fill, and optional boundary nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, polygonCount, ringCount, featureCount, singleSort} = this;
    const {width, height, extent, polygonPositions, zones, boundary, overflow} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      extent,
      polygonPositions,
      props.featureOffsets,
      props.polygonOffsets,
      props.ringOffsets,
      zones,
      boundary,
      overflow,
      props.crossingCount
    ]);
    const vertexCount = polygonPositions.length;
    const capacity = props.crossingCapacity;
    const cellCount = width * height;
    const rowStride = width + 1;
    // Largest valid row-column key is height * (width + 1) - 1; the next value marks unused slots.
    const rowColumnLimit = height * rowStride;
    const rowColumnBits = getSortKeyBits(rowColumnLimit - 1);
    const declarations = `const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;
const ROW_STRIDE: u32 = ${rowStride}u;
const CAPACITY: u32 = ${capacity}u;
const POLYGON_COUNT: u32 = ${polygonCount}u;
const SINGLE_SORT: bool = ${singleSort};
// Only the single-sort key packs polygons above the row-column bits.
const ROW_COLUMN_BITS: u32 = ${singleSort ? rowColumnBits : 0}u;
const ROW_COLUMN_LIMIT: u32 = ${rowColumnLimit}u;`;
    const nodes: GPUCommandNode<Parameters>[] = [];

    // Records: `primary` / `secondary` are the first sort's keys and values. With one sort the
    // key is `polygon << ROW_COLUMN_BITS | rowColumn` and the value is `rowColumn`; with two sorts
    // the first sort orders `rowColumn` keys carrying `polygon` values.
    const primary = createTransientView(graph, `${id}-primary`, 'uint32', capacity);
    const secondary = createTransientView(graph, `${id}-secondary`, 'uint32', capacity);
    const invalidPrimary = `${singleSort ? polygonCount * 2 ** rowColumnBits : rowColumnLimit}u`;
    const invalidSecondary = singleSort ? '0u' : `${polygonCount}u`;
    const resetBindings: WGSLKernelBinding[] = [
      {name: 'zonesOut', view: zones, type: 'u32', access: 'read_write'},
      {name: 'primary', view: primary, type: 'u32', access: 'read_write'},
      {name: 'secondary', view: secondary, type: 'u32', access: 'read_write'}
    ];
    if (boundary) {
      resetBindings.push({name: 'boundaryOut', view: boundary, type: 'u32', access: 'read_write'});
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-reset`,
        operation: OPERATION,
        variant: 'reset',
        bindings: resetBindings,
        invocationCount: Math.max(cellCount, capacity),
        declarations: `const CELL_COUNT: u32 = ${cellCount}u;
const CAPACITY: u32 = ${capacity}u;`,
        body: `if (index < CELL_COUNT) {
    zonesOut[zonesOutOffset + index] = ${GPU_POLYGON_RASTERIZATION_NO_ZONE}u;
    ${boundary ? 'boundaryOut[boundaryOutOffset + index] = 0u;' : ''}
  }
  if (index < CAPACITY) {
    primary[primaryOffset + index] = ${invalidPrimary};
    secondary[secondaryOffset + index] = ${invalidSecondary};
  }`
      })
    );

    // 1. Per edge: ring, next vertex, polygon, and crossed row count.
    const edgeCounts = createTransientView(graph, `${id}-edge-counts`, 'uint32', vertexCount);
    const edgeNext = createTransientView(graph, `${id}-edge-next`, 'uint32', vertexCount);
    const edgePolygons = createTransientView(graph, `${id}-edge-polygons`, 'uint32', vertexCount);
    const rowRangeSource = `${POLYGON_RASTER_EXTENT_WGSL}
fn getRowThreshold(y: f32, rasterExtent: RasterExtent) -> u32 {
  // Smallest row whose center line is at or above y.
  let row = ceil((y - rasterExtent.origin.y) / rasterExtent.cellSize.y - 0.5);
  return u32(clamp(row, 0.0, f32(HEIGHT)));
}
fn loadPosition(vertex: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + vertex * 2u], positions[positionsOffset + vertex * 2u + 1u]);
}
fn isFinitePosition(position: vec2<f32>) -> bool {
  return isFiniteValue(position.x) && isFiniteValue(position.y);
}`;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-edges`,
        operation: OPERATION,
        variant: 'edges',
        bindings: [
          {name: 'positions', view: polygonPositions, type: 'f32', access: 'read'},
          {name: 'ringOffsets', view: props.ringOffsets, type: 'u32', access: 'read'},
          {name: 'polygonOffsets', view: props.polygonOffsets, type: 'u32', access: 'read'},
          {name: 'featureOffsets', view: props.featureOffsets, type: 'u32', access: 'read'},
          {name: 'extent', view: extent, type: 'f32', access: 'read'},
          {name: 'edgeCounts', view: edgeCounts, type: 'u32', access: 'read_write'},
          {name: 'edgeNext', view: edgeNext, type: 'u32', access: 'read_write'},
          {name: 'edgePolygons', view: edgePolygons, type: 'u32', access: 'read_write'}
        ],
        invocationCount: vertexCount,
        declarations: `${declarations}
${rowRangeSource}
${getOffsetRangeSearchSource('findRing', 'ringOffsets', ringCount)}
${getOffsetRangeSearchSource('findPolygon', 'polygonOffsets', polygonCount)}
${getOffsetRangeSearchSource('findFeature', 'featureOffsets', featureCount)}`,
        body: `var next = ${NO_EDGE};
  var polygon = ${NO_EDGE};
  var count = 0u;
  let ring = findRing(index);
  if (ring != ${NO_EDGE}) {
    polygon = findPolygon(ring);
  }
  if (polygon != ${NO_EDGE} && findFeature(polygon) != ${NO_EDGE}) {
    let ringStart = ringOffsets[ringOffsetsOffset + ring];
    let ringEnd = ringOffsets[ringOffsetsOffset + ring + 1u];
    next = select(index + 1u, ringStart, index + 1u >= ringEnd);
    let start = loadPosition(index);
    let end = loadPosition(next);
    let rasterExtent = getRasterExtent();
    if (rasterExtent.valid && isFinitePosition(start) && isFinitePosition(end)) {
      let low = getRowThreshold(min(start.y, end.y), rasterExtent);
      let high = getRowThreshold(max(start.y, end.y), rasterExtent);
      count = high - min(low, high);
    } else {
      next = ${NO_EDGE};
    }
  } else {
    polygon = ${NO_EDGE};
  }
  edgeCounts[edgeCountsOffset + index] = count;
  edgeNext[edgeNextOffset + index] = next;
  edgePolygons[edgePolygonsOffset + index] = polygon;`
      })
    );

    // 2. Slot ranges, total, overflow, and emission.
    const edgeOffsets = createTransientView(graph, `${id}-edge-offsets`, 'uint32', vertexCount);
    nodes.push(
      ...new GPUScan({
        id: `${id}-edge-scan`,
        input: edgeCounts,
        output: edgeOffsets,
        mode: 'exclusive'
      }).getCommandNodes(graph)
    );
    const total = createTransientView(graph, `${id}-total`, 'uint32', 1);
    const totalBindings: WGSLKernelBinding[] = [
      {name: 'edgeCounts', view: edgeCounts, type: 'u32', access: 'read'},
      {name: 'edgeOffsets', view: edgeOffsets, type: 'u32', access: 'read'},
      {name: 'totalOut', view: total, type: 'u32', access: 'read_write'},
      {name: 'overflowOut', view: overflow, type: 'u32', access: 'read_write'}
    ];
    if (props.crossingCount) {
      totalBindings.push({
        name: 'crossingCountOut',
        view: props.crossingCount,
        type: 'u32',
        access: 'read_write'
      });
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-total`,
        operation: OPERATION,
        variant: 'total',
        bindings: totalBindings,
        invocationCount: 1,
        declarations: `const LAST_EDGE: u32 = ${vertexCount - 1}u;
const CAPACITY: u32 = ${capacity}u;`,
        body: `let crossings = edgeOffsets[edgeOffsetsOffset + LAST_EDGE] + edgeCounts[edgeCountsOffset + LAST_EDGE];
  totalOut[totalOutOffset] = crossings;
  overflowOut[overflowOutOffset] = select(0u, 1u, crossings > CAPACITY);
  ${props.crossingCount ? 'crossingCountOut[crossingCountOutOffset] = crossings;' : ''}`
      })
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-emit`,
        operation: OPERATION,
        variant: 'emit',
        bindings: [
          {name: 'positions', view: polygonPositions, type: 'f32', access: 'read'},
          {name: 'extent', view: extent, type: 'f32', access: 'read'},
          {name: 'edgeCounts', view: edgeCounts, type: 'u32', access: 'read'},
          {name: 'edgeNext', view: edgeNext, type: 'u32', access: 'read'},
          {name: 'edgePolygons', view: edgePolygons, type: 'u32', access: 'read'},
          {name: 'edgeOffsets', view: edgeOffsets, type: 'u32', access: 'read'},
          {name: 'primary', view: primary, type: 'u32', access: 'read_write'},
          {name: 'secondary', view: secondary, type: 'u32', access: 'read_write'}
        ],
        invocationCount: vertexCount,
        declarations: `${declarations}
${rowRangeSource}`,
        body: `let count = edgeCounts[edgeCountsOffset + index];
  if (count == 0u) {
    return;
  }
  let next = edgeNext[edgeNextOffset + index];
  let polygon = edgePolygons[edgePolygonsOffset + index];
  let rasterExtent = getRasterExtent();
  let first = loadPosition(index);
  let second = loadPosition(next);
  // Orient bottom to top so both traversal directions compute bitwise identical crossings.
  let lower = select(second, first, first.y < second.y);
  let upper = select(first, second, first.y < second.y);
  let firstRow = getRowThreshold(lower.y, rasterExtent);
  let base = edgeOffsets[edgeOffsetsOffset + index];
  for (var step = 0u; step < count; step++) {
    let slot = base + step;
    if (slot >= CAPACITY) {
      break;
    }
    let row = firstRow + step;
    let centerY = rasterExtent.origin.y + (f32(row) + 0.5) * rasterExtent.cellSize.y;
    let t = clamp((centerY - lower.y) / (upper.y - lower.y), 0.0, 1.0);
    let crossingX = lower.x + (upper.x - lower.x) * t;
    // First column whose center is at or right of the crossing.
    let column = u32(clamp(ceil((crossingX - rasterExtent.origin.x) / rasterExtent.cellSize.x - 0.5), 0.0, f32(WIDTH)));
    let rowColumn = row * ROW_STRIDE + column;
    if (SINGLE_SORT) {
      primary[primaryOffset + slot] = (polygon << ROW_COLUMN_BITS) | rowColumn;
      secondary[secondaryOffset + slot] = rowColumn;
    } else {
      primary[primaryOffset + slot] = rowColumn;
      secondary[secondaryOffset + slot] = polygon;
    }
  }`
      })
    );

    // 3. Sort records by (polygon, row, column) and rank them within each (polygon, row).
    const sortedKeys = createTransientView(graph, `${id}-sorted-keys`, 'uint32', capacity);
    const sortedValues = createTransientView(graph, `${id}-sorted-values`, 'uint32', capacity);
    let recordKeys: GraphDataView<'uint32'>;
    let recordRowColumns: GraphDataView<'uint32'>;
    if (singleSort) {
      nodes.push(
        ...new GPUSort({
          id: `${id}-sort`,
          keys: primary,
          values: secondary,
          outputKeys: sortedKeys,
          outputValues: sortedValues,
          keyBits: getSortKeyBits(polygonCount) + rowColumnBits
        }).getCommandNodes(graph)
      );
      recordKeys = sortedKeys;
      recordRowColumns = sortedValues;
    } else {
      const polygonKeys = createTransientView(graph, `${id}-polygon-keys`, 'uint32', capacity);
      const rowColumns = createTransientView(graph, `${id}-row-columns`, 'uint32', capacity);
      nodes.push(
        ...new GPUSort({
          id: `${id}-sort-row-column`,
          keys: primary,
          values: secondary,
          outputKeys: sortedKeys,
          outputValues: sortedValues,
          keyBits: getSortKeyBits(rowColumnLimit)
        }).getCommandNodes(graph),
        // Stable: equal polygons keep their row-column order.
        ...new GPUSort({
          id: `${id}-sort-polygon`,
          keys: sortedValues,
          values: sortedKeys,
          outputKeys: polygonKeys,
          outputValues: rowColumns,
          keyBits: getSortKeyBits(polygonCount)
        }).getCommandNodes(graph)
      );
      recordKeys = polygonKeys;
      recordRowColumns = rowColumns;
    }
    // Record i is valid when its polygon is below POLYGON_COUNT; its segment is (polygon, row).
    const recordSource = `fn getRecordPolygon(record: u32) -> u32 {
  let key = recordKeys[recordKeysOffset + record];
  return select(key, key >> ROW_COLUMN_BITS, SINGLE_SORT);
}
fn getRecordRowColumn(record: u32) -> u32 {
  return recordRowColumns[recordRowColumnsOffset + record];
}`;
    const segmentFlags = createTransientView(graph, `${id}-segment-flags`, 'uint32', capacity);
    const ones = createTransientView(graph, `${id}-ones`, 'uint32', capacity);
    const ranks = createTransientView(graph, `${id}-ranks`, 'uint32', capacity);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-segments`,
        operation: OPERATION,
        variant: 'segments',
        bindings: [
          {name: 'recordKeys', view: recordKeys, type: 'u32', access: 'read'},
          {name: 'recordRowColumns', view: recordRowColumns, type: 'u32', access: 'read'},
          {name: 'segmentFlags', view: segmentFlags, type: 'u32', access: 'read_write'},
          {name: 'ones', view: ones, type: 'u32', access: 'read_write'}
        ],
        invocationCount: capacity,
        declarations: `${declarations}
${recordSource}`,
        body: `var isHead = index == 0u;
  if (!isHead) {
    isHead = getRecordPolygon(index) != getRecordPolygon(index - 1u) ||
      getRecordRowColumn(index) / ROW_STRIDE != getRecordRowColumn(index - 1u) / ROW_STRIDE;
  }
  segmentFlags[segmentFlagsOffset + index] = select(0u, 1u, isHead);
  ones[onesOffset + index] = 1u;`
      })
    );
    nodes.push(
      ...new GPUScan({
        id: `${id}-rank-scan`,
        input: ones,
        output: ranks,
        mode: 'exclusive',
        segmentFlags
      }).getCommandNodes(graph)
    );

    // 4. Fill spans between even-ranked crossings and their successors.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-fill`,
        operation: OPERATION,
        variant: 'fill',
        bindings: [
          {name: 'recordKeys', view: recordKeys, type: 'u32', access: 'read'},
          {name: 'recordRowColumns', view: recordRowColumns, type: 'u32', access: 'read'},
          {name: 'ranks', view: ranks, type: 'u32', access: 'read'},
          {name: 'featureOffsets', view: props.featureOffsets, type: 'u32', access: 'read'},
          {name: 'total', view: total, type: 'u32', access: 'read'},
          {name: 'zonesOut', view: zones, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: capacity - 1,
        declarations: `${declarations}
${recordSource}
${getOffsetRangeSearchSource('findFeature', 'featureOffsets', featureCount)}`,
        body: `if (total[totalOffset] > CAPACITY || (ranks[ranksOffset + index] & 1u) != 0u) {
    return;
  }
  let polygon = getRecordPolygon(index);
  if (polygon >= POLYGON_COUNT || getRecordPolygon(index + 1u) != polygon) {
    return;
  }
  let start = getRecordRowColumn(index);
  let end = getRecordRowColumn(index + 1u);
  let row = start / ROW_STRIDE;
  if (end / ROW_STRIDE != row) {
    return;
  }
  let feature = findFeature(polygon);
  let rowBase = row * WIDTH;
  for (var column = start % ROW_STRIDE; column < end % ROW_STRIDE; column++) {
    atomicMin(&zonesOut[zonesOutOffset + rowBase + column], feature);
  }`
      })
    );

    // 5. Conservative boundary cells: every cell the closed edge segment touches.
    if (boundary) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-boundary`,
          operation: OPERATION,
          variant: 'boundary',
          bindings: [
            {name: 'positions', view: polygonPositions, type: 'f32', access: 'read'},
            {name: 'extent', view: extent, type: 'f32', access: 'read'},
            {name: 'edgeNext', view: edgeNext, type: 'u32', access: 'read'},
            {name: 'boundaryOut', view: boundary, type: 'atomic<u32>', access: 'read_write'}
          ],
          invocationCount: vertexCount,
          declarations: `${declarations}
${POLYGON_RASTER_EXTENT_WGSL}
// Cell padding that keeps the marking conservative under float32 rounding: a fixed part plus
// about 8 ulps of the largest magnitude involved, in cell units.
const BASE_PADDING: f32 = 1e-3;
const RELATIVE_PADDING: f32 = 1e-6;
fn loadPosition(vertex: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + vertex * 2u], positions[positionsOffset + vertex * 2u + 1u]);
}
fn clampCell(value: f32, limit: u32) -> i32 {
  return i32(clamp(floor(value), -1.0, f32(limit)));
}`,
          body: `let next = edgeNext[edgeNextOffset + index];
  if (next == ${NO_EDGE}) {
    return;
  }
  let rasterExtent = getRasterExtent();
  let startPosition = loadPosition(index);
  let endPosition = loadPosition(next);
  // Edge in cell units.
  let start = (startPosition - rasterExtent.origin) / rasterExtent.cellSize;
  let end = (endPosition - rasterExtent.origin) / rasterExtent.cellSize;
  let magnitude = (max(abs(startPosition), abs(endPosition)) + abs(rasterExtent.origin)) / rasterExtent.cellSize;
  let padding = BASE_PADDING + RELATIVE_PADDING * max(magnitude.x, magnitude.y);
  let lower = select(end, start, start.y < end.y);
  let upper = select(start, end, start.y < end.y);
  let firstRow = max(clampCell(lower.y - padding, HEIGHT), 0);
  let lastRow = min(clampCell(upper.y + padding, HEIGHT), i32(HEIGHT) - 1);
  for (var row = firstRow; row <= lastRow; row++) {
    // x extent of the segment inside the padded row slab.
    var slabX = vec2<f32>(lower.x, upper.x);
    let deltaY = upper.y - lower.y;
    if (deltaY > 0.0) {
      let t0 = clamp((f32(row) - padding - lower.y) / deltaY, 0.0, 1.0);
      let t1 = clamp((f32(row + 1) + padding - lower.y) / deltaY, 0.0, 1.0);
      slabX = vec2<f32>(lower.x + (upper.x - lower.x) * t0, lower.x + (upper.x - lower.x) * t1);
    }
    let firstColumn = max(clampCell(min(slabX.x, slabX.y) - padding, WIDTH), 0);
    let lastColumn = min(clampCell(max(slabX.x, slabX.y) + padding, WIDTH), i32(WIDTH) - 1);
    for (var column = firstColumn; column <= lastColumn; column++) {
      atomicStore(&boundaryOut[boundaryOutOffset + u32(row) * WIDTH + u32(column)], 1u);
    }
  }`
        })
      );
    }
    return nodes;
  }
}
