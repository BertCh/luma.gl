// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUPointSpatialFilter,
  GraphVectorView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, getWGSLFloatLiteral} from '../../utils/wgsl-kernel-nodes';
import type {GPUFloat32Positions} from '../../utils/gpu-contributor-types';
import {GPURegionMask} from './gpu-region-mask';
import type {
  GPURegionPolygon,
  GPURegionRadius,
  GPURegionRectangle,
  GPURegionStatisticsGridIndex
} from './region-statistics-types';

const OPERATION = 'GPURegionStatistics';

/** Views produced by {@link getGridCandidateNodes}, all of length `capacity`. @internal */
export type GridCandidates<Parameters> = {
  /** Query, gather, and exact-predicate nodes in execution order. */
  nodes: GPUCommandNode<Parameters>[];
  /** Candidate slot count. */
  capacity: number;
  /** Candidate source rows, valid for slots below `min(count, capacity)`. */
  rows: GraphDataView<'uint32'>;
  /** One-row unclamped candidate count. */
  count: GraphDataView<'uint32'>;
  /** Candidate positions, quiet NaN in unused slots. */
  positions: GraphDataView<'float32x2'>;
  /** Candidate values, NaN in unused slots. Present only with source values. */
  values?: GraphDataView<'float32'>;
  /** Exact predicate result per candidate slot, 0 in unused slots. */
  mask: GraphDataView<'uint32'>;
  /** One-row flag, 1 when the candidate set may be incomplete. Written by `statusNode`. */
  candidateFlag: GraphDataView<'uint32'>;
  /** Writes `counts[2]` and `candidateFlag`. Run it after `counts` is cleared. */
  getStatusNode: (counts: GraphDataView<'uint32'>) => GPUCommandNode<Parameters>;
};

/**
 * Gathers candidate rows from a grid index and evaluates the same exact predicate primitive as the
 * brute-force path on the compacted candidate positions.
 *
 * @internal
 */
export function getGridCandidateNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    selection: GPURegionRectangle | GPURegionRadius | GPURegionPolygon;
    positions: GPUFloat32Positions;
    values?: GraphDataView<'float32'>;
    spatialIndex: GPURegionStatisticsGridIndex;
    rowCount: number;
    regionOverflow: GraphDataView<'uint32'>;
  }
): GridCandidates<Parameters> {
  const {id, selection, spatialIndex, rowCount, regionOverflow} = props;
  const positions = props.positions as GraphDataView<'float32x2'>;
  const capacity = Math.min(spatialIndex.candidateCapacity, rowCount);
  const nodes: GPUCommandNode<Parameters>[] = [];

  // Step 1: query view.
  let query: GraphDataView<'float32'>;
  let kind: 'bounds' | 'radius';
  if (selection.kind === 'rectangle') {
    query = selection.bounds;
    kind = 'bounds';
  } else if (selection.kind === 'radius') {
    query = selection.circle;
    kind = 'radius';
  } else {
    query = createTransientView(graph, `${id}-polygon-bounds`, 'float32', 4);
    kind = 'bounds';
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-polygon-bounds-kernel`,
        operation: OPERATION,
        variant: 'polygon-bounds',
        bindings: [
          {name: 'vertices', view: selection.vertices, type: 'f32', access: 'read'},
          {name: 'vertexCount', view: selection.vertexCount, type: 'u32', access: 'read'},
          {name: 'bounds', view: query, type: 'f32', access: 'read_write'}
        ],
        invocationCount: 1,
        declarations: `const VERTEX_CAPACITY: u32 = ${selection.vertices.length}u;
fn isFiniteValue(value: f32) -> bool { return value == value && abs(value) <= 3.402823466e+38; }`,
        // GPURegionMask feeds non-finite vertices straight into its crossing test, whose result
        // is then not reliably bounded by the finite vertices. A lasso with any non-finite active
        // vertex therefore queries the whole finite range, so the candidate set (and the exact
        // predicate on it) matches brute force, or the capacity overflow is flagged. Fewer than
        // three active vertices write NaN bounds, which select no cell, matching brute force.
        body: `let activeCount = min(vertexCount[vertexCountOffset], VERTEX_CAPACITY);
  var minimum = vec2f(3.402823466e+38);
  var maximum = vec2f(-3.402823466e+38);
  var allFinite = true;
  for (var vertexIndex = 0u; vertexIndex < activeCount; vertexIndex++) {
    let vertex = vec2f(vertices[verticesOffset + vertexIndex * 2u], vertices[verticesOffset + vertexIndex * 2u + 1u]);
    if (isFiniteValue(vertex.x) && isFiniteValue(vertex.y)) {
      minimum = min(minimum, vertex);
      maximum = max(maximum, vertex);
    } else {
      allFinite = false;
    }
  }
  if (!allFinite) {
    minimum = vec2f(-3.402823466e+38);
    maximum = vec2f(3.402823466e+38);
  }
  // A runtime expression keeps the quiet-NaN bit pattern out of const evaluation.
  let quietNaN = bitcast<f32>(0x7fc00000u | (index & 0u));
  let valid = activeCount >= 3u;
  bounds[boundsOffset] = select(quietNaN, minimum.x, valid);
  bounds[boundsOffset + 1u] = select(quietNaN, minimum.y, valid);
  bounds[boundsOffset + 2u] = select(quietNaN, maximum.x, valid);
  bounds[boundsOffset + 3u] = select(quietNaN, maximum.y, valid);`
      })
    );
  }

  // Step 2: candidate rows from cell ranges. Cells [cx0..cx1] of one grid row are contiguous in
  // `cellOffsets`, so their objects form one contiguous range of `objectIds`. The candidate set is
  // the union of at most `height` such ranges, enumerated through a prefix sum over grid rows.
  const {index} = spatialIndex;
  const [width, height] = index.gridSize;
  const [domainMinX, domainMinY, domainMaxX, domainMaxY] = index.bounds;
  const rows = createTransientView(graph, `${id}-candidate-rows`, 'uint32', capacity);
  const count = createTransientView(graph, `${id}-candidate-count`, 'uint32', 1);
  const queryOverflow = createTransientView(graph, `${id}-candidate-overflow`, 'uint32', 1);
  const rangeStart = createTransientView(graph, `${id}-range-start`, 'uint32', height);
  const rangePrefix = createTransientView(graph, `${id}-range-prefix`, 'uint32', height + 1);
  const isRadius = kind === 'radius';
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-cell-ranges`,
      operation: OPERATION,
      variant: 'grid-cell-ranges',
      bindings: [
        {
          name: 'cellOffsets',
          view: index.cellOffsets as GraphDataView<'uint32'>,
          type: 'u32',
          access: 'read'
        },
        {name: 'indexOverflow', view: index.overflow, type: 'u32', access: 'read'},
        {name: 'query', view: query, type: 'f32', access: 'read'},
        {name: 'rangeStart', view: rangeStart, type: 'u32', access: 'read_write'},
        {name: 'rangePrefix', view: rangePrefix, type: 'u32', access: 'read_write'},
        {name: 'candidateCount', view: count, type: 'u32', access: 'read_write'},
        {name: 'candidateOverflow', view: queryOverflow, type: 'u32', access: 'read_write'}
      ],
      invocationCount: 1,
      declarations: `const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;
const CAPACITY: u32 = ${capacity}u;
const DOMAIN_MIN: vec2f = vec2f(${getWGSLFloatLiteral(domainMinX)}, ${getWGSLFloatLiteral(domainMinY)});
const DOMAIN_MAX: vec2f = vec2f(${getWGSLFloatLiteral(domainMaxX)}, ${getWGSLFloatLiteral(domainMaxY)});
fn isFiniteValue(value: f32) -> bool { return value == value && abs(value) <= 3.402823466e+38; }
// Same cell mapping as GPUGridIndex.
fn getCoordinate(value: f32, minimum: f32, maximum: f32, size: u32) -> u32 {
  if (maximum == minimum || value == minimum) { return 0u; }
  if (value == maximum) { return size - 1u; }
  if (minimum < 0.0 && maximum > 0.0) {
    let scale = max(abs(minimum), abs(maximum));
    let scaledValue = value / scale;
    let scaledMinimum = minimum / scale;
    let scaledMaximum = maximum / scale;
    return min(
      u32((scaledValue - scaledMinimum) / (scaledMaximum - scaledMinimum) * f32(size)),
      size - 1u
    );
  }
  return min(u32((value - minimum) / (maximum - minimum) * f32(size)), size - 1u);
}`,
      // Invalid, empty, or non-overlapping queries produce zero ranges. The cell rectangle is
      // widened by one cell per side to absorb f32 rounding differences from the binning pass; it
      // only adds candidates, which the exact predicate rejects.
      body: `${
        isRadius
          ? `let center = vec2f(query[queryOffset], query[queryOffset + 1u]);
  let radius = query[queryOffset + 2u];
  let valid = isFiniteValue(center.x) && isFiniteValue(center.y) && isFiniteValue(radius) && radius >= 0.0;
  let queryMin = center - vec2f(radius);
  let queryMax = center + vec2f(radius);`
          : `let queryMin = vec2f(query[queryOffset], query[queryOffset + 1u]);
  let queryMax = vec2f(query[queryOffset + 2u], query[queryOffset + 3u]);
  let valid = isFiniteValue(queryMin.x) && isFiniteValue(queryMin.y) &&
    isFiniteValue(queryMax.x) && isFiniteValue(queryMax.y) && all(queryMin <= queryMax);`
      }
  let overlaps = valid && all(queryMax >= DOMAIN_MIN) && all(queryMin <= DOMAIN_MAX);
  let clampedMin = max(queryMin, DOMAIN_MIN);
  let clampedMax = min(queryMax, DOMAIN_MAX);
  var columnLow = 0u;
  var columnHigh = 0u;
  var rowLow = 0u;
  var rowHigh = 0u;
  if (overlaps) {
    columnLow = getCoordinate(clampedMin.x, DOMAIN_MIN.x, DOMAIN_MAX.x, WIDTH);
    columnHigh = getCoordinate(clampedMax.x, DOMAIN_MIN.x, DOMAIN_MAX.x, WIDTH);
    rowLow = getCoordinate(clampedMin.y, DOMAIN_MIN.y, DOMAIN_MAX.y, HEIGHT);
    rowHigh = getCoordinate(clampedMax.y, DOMAIN_MIN.y, DOMAIN_MAX.y, HEIGHT);
    columnLow = columnLow - min(columnLow, 1u);
    rowLow = rowLow - min(rowLow, 1u);
    columnHigh = min(columnHigh + 1u, WIDTH - 1u);
    rowHigh = min(rowHigh + 1u, HEIGHT - 1u);
  }
  var total = 0u;
  for (var row = 0u; row < HEIGHT; row++) {
    var start = 0u;
    var length = 0u;
    if (overlaps && row >= rowLow && row <= rowHigh) {
      start = cellOffsets[cellOffsetsOffset + row * WIDTH + columnLow];
      let end = cellOffsets[cellOffsetsOffset + row * WIDTH + columnHigh + 1u];
      length = select(0u, end - start, end > start);
    }
    rangeStart[rangeStartOffset + row] = start;
    rangePrefix[rangePrefixOffset + row] = total;
    total += length;
  }
  rangePrefix[rangePrefixOffset + HEIGHT] = total;
  candidateCount[candidateCountOffset] = total;
  candidateOverflow[candidateOverflowOffset] = select(0u, 1u, total > CAPACITY || indexOverflow[indexOverflowOffset] != 0u);`
    })
  );
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-cell-gather`,
      operation: OPERATION,
      variant: 'grid-cell-gather',
      bindings: [
        {
          name: 'objectIds',
          view: index.objectIds as GraphDataView<'uint32'>,
          type: 'u32',
          access: 'read'
        },
        {name: 'rangeStart', view: rangeStart, type: 'u32', access: 'read'},
        {name: 'rangePrefix', view: rangePrefix, type: 'u32', access: 'read'},
        {name: 'candidateCount', view: count, type: 'u32', access: 'read'},
        {name: 'candidateRows', view: rows, type: 'u32', access: 'read_write'}
      ],
      invocationCount: capacity,
      declarations: `const HEIGHT: u32 = ${height}u;
const CAPACITY: u32 = ${capacity}u;
const OBJECT_COUNT: u32 = ${index.objectIds.length}u;`,
      // Slot k belongs to the grid row r with rangePrefix[r] <= k < rangePrefix[r + 1].
      body: `let storedCount = min(candidateCount[candidateCountOffset], CAPACITY);
  var row = 0xffffffffu;
  if (index < storedCount) {
    var low = 0u;
    var high = HEIGHT + 1u;
    loop {
      if (low >= high) { break; }
      let middle = low + (high - low) / 2u;
      if (rangePrefix[rangePrefixOffset + middle] <= index) { low = middle + 1u; } else { high = middle; }
    }
    let gridRow = low - 1u;
    let objectIndex = rangeStart[rangeStartOffset + gridRow] + index - rangePrefix[rangePrefixOffset + gridRow];
    if (objectIndex < OBJECT_COUNT) { row = objectIds[objectIdsOffset + objectIndex]; }
  }
  candidateRows[candidateRowsOffset + index] = row;`
    })
  );

  // Step 3: gather.
  const candidatePositions = createTransientView(
    graph,
    `${id}-candidate-positions`,
    'float32x2',
    capacity
  );
  const candidateValues = props.values
    ? createTransientView(graph, `${id}-candidate-values`, 'float32', capacity)
    : undefined;
  const gatherBindings = [
    {name: 'positions', view: positions, type: 'f32', access: 'read'},
    ...(props.values
      ? [{name: 'values', view: props.values, type: 'f32', access: 'read'} as const]
      : []),
    {name: 'candidateRows', view: rows, type: 'u32', access: 'read'},
    {name: 'candidateCount', view: count, type: 'u32', access: 'read'},
    {name: 'candidatePositions', view: candidatePositions, type: 'f32', access: 'read_write'},
    ...(candidateValues
      ? [
          {
            name: 'candidateValues',
            view: candidateValues,
            type: 'f32',
            access: 'read_write'
          } as const
        ]
      : [])
  ] as const;
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-gather`,
      operation: OPERATION,
      variant: 'grid-gather',
      bindings: gatherBindings,
      invocationCount: capacity,
      declarations: `const ROW_COUNT: u32 = ${rowCount}u;
const CAPACITY: u32 = ${capacity}u;`,
      body: `let storedCount = min(candidateCount[candidateCountOffset], CAPACITY);
  // A runtime expression keeps the quiet-NaN bit pattern out of const evaluation.
  let quietNaN = bitcast<f32>(0x7fc00000u | (index & 0u));
  var position = vec2f(quietNaN);
  var value = quietNaN;
  if (index < storedCount) {
    let row = candidateRows[candidateRowsOffset + index];
    if (row < ROW_COUNT) {
      position = vec2f(positions[positionsOffset + row * 2u], positions[positionsOffset + row * 2u + 1u]);
      ${props.values ? 'value = values[valuesOffset + row];' : ''}
    }
  }
  candidatePositions[candidatePositionsOffset + index * 2u] = position.x;
  candidatePositions[candidatePositionsOffset + index * 2u + 1u] = position.y;
  ${candidateValues ? 'candidateValues[candidateValuesOffset + index] = value;' : ''}`
    })
  );

  // Step 4: the brute-force predicate primitive, over the candidate slots only.
  const mask = createTransientView(graph, `${id}-candidate-mask`, 'uint32', capacity);
  if (selection.kind === 'polygon') {
    nodes.push(
      ...new GPURegionMask({
        id: `${id}-region`,
        positions: candidatePositions,
        region: selection,
        outputMask: mask,
        overflow: regionOverflow
      }).getCommandNodes(graph)
    );
  } else {
    nodes.push(
      ...new GPUPointSpatialFilter({
        id: `${id}-filter`,
        positions: candidatePositions,
        kind,
        query,
        outputMask: mask,
        overflow: regionOverflow
      }).getCommandNodes(graph)
    );
  }

  const candidateFlag = createTransientView(graph, `${id}-candidate-flag`, 'uint32', 1);
  return {
    nodes,
    capacity,
    rows,
    count,
    positions: candidatePositions,
    values: candidateValues,
    mask,
    candidateFlag,
    getStatusNode: counts =>
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-candidate-status`,
        operation: OPERATION,
        variant: 'grid-status',
        bindings: [
          {name: 'candidateCount', view: count, type: 'u32', access: 'read'},
          {name: 'candidateOverflow', view: queryOverflow, type: 'u32', access: 'read'},
          {name: 'indexCount', view: spatialIndex.index.count, type: 'u32', access: 'read'},
          {name: 'counts', view: counts, type: 'u32', access: 'read_write'},
          {name: 'candidateFlag', view: candidateFlag, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 1,
        declarations: `const ROW_COUNT: u32 = ${rowCount}u;
const CAPACITY: u32 = ${capacity}u;`,
        body: `let truncated = candidateOverflow[candidateOverflowOffset] != 0u ||
    candidateCount[candidateCountOffset] > CAPACITY ||
    indexCount[indexCountOffset] < ROW_COUNT;
  counts[countsOffset + 2u] = select(0u, 1u, truncated);
  candidateFlag[candidateFlagOffset] = select(0u, 1u, truncated);`
      })
  };
}

/** Builds the node that scatters the candidate mask back to a source-aligned mask. @internal */
export function getGridMaskScatterNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    candidates: GridCandidates<Parameters>;
    mask: GraphDataView<'uint32'>;
    rowCount: number;
  }
): GPUCommandNode<Parameters> {
  const {candidates} = props;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'grid-scatter',
    bindings: [
      {name: 'candidateRows', view: candidates.rows, type: 'u32', access: 'read'},
      {name: 'candidateCount', view: candidates.count, type: 'u32', access: 'read'},
      {name: 'candidateMask', view: candidates.mask, type: 'u32', access: 'read'},
      {name: 'mask', view: props.mask, type: 'u32', access: 'read_write'}
    ],
    invocationCount: candidates.capacity,
    declarations: `const ROW_COUNT: u32 = ${props.rowCount}u;
const CAPACITY: u32 = ${candidates.capacity}u;`,
    // Each candidate row appears once in the index, so no two invocations write the same word.
    body: `let storedCount = min(candidateCount[candidateCountOffset], CAPACITY);
  if (index < storedCount && candidateMask[candidateMaskOffset + index] != 0u) {
    let row = candidateRows[candidateRowsOffset + index];
    if (row < ROW_COUNT) { mask[maskOffset + row] = 1u; }
  }`
  });
}

/** Throws unless every spatial-index restriction holds. @internal */
export function validateGridIndexProps(
  id: string,
  props: {
    selection: {kind: string; screenTransform?: unknown};
    positions?: unknown;
    values?: unknown;
    sourceIds?: unknown;
    spatialIndex: GPURegionStatisticsGridIndex;
  }
): void {
  const {selection, spatialIndex} = props;
  if (
    selection.kind !== 'rectangle' &&
    selection.kind !== 'radius' &&
    selection.kind !== 'polygon'
  ) {
    throw new Error(`${id} spatialIndex supports rectangle, radius, and polygon selections`);
  }
  if (selection.screenTransform) {
    throw new Error(`${id} spatialIndex supports only world-space selections (no screenTransform)`);
  }
  for (const [name, view] of [
    ['positions', props.positions],
    ['values', props.values],
    ['sourceIds', props.sourceIds]
  ] as const) {
    if (view instanceof GraphVectorView) {
      throw new Error(`${id} spatialIndex requires packed (non-vector) ${name}`);
    }
  }
  if (spatialIndex.kind !== 'grid') {
    throw new Error(`${id} spatialIndex.kind must be 'grid'`);
  }
  const {index, candidateCapacity} = spatialIndex;
  if (index.cellOffsets instanceof GraphVectorView || index.objectIds instanceof GraphVectorView) {
    throw new Error(
      `${id} spatialIndex.index requires packed (non-vector) cellOffsets and objectIds`
    );
  }
  if (index.gridSize.length !== 2 || index.bounds.length !== 4) {
    throw new Error(`${id} spatialIndex.index must be two-dimensional`);
  }
  if (!Number.isSafeInteger(candidateCapacity) || candidateCapacity < 1) {
    throw new Error(`${id} spatialIndex.candidateCapacity must be a positive integer`);
  }
}
