// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {createMapGraphKernelNode, type MapGraphKernelBinding} from '../map-graph-kernels';
import {GEODESIC_WGSL} from '../geometry-measures/geodesic-wgsl';
import type {GPULineCoordinateSystem} from './line-segmentize-types';

/**
 * Words per output piece in the internal piece column:
 * `[path, startMeasure, endMeasure, flags, firstInteriorRow, interiorEndRow, vertexCount, pathStart,
 * pathEnd, 0]`.
 *
 * @internal
 */
export const LINE_PIECE_STRIDE = 10;

/** Piece flag: copy the path's rows verbatim (zero-length paths). @internal */
const PIECE_COPY = 1;

/** How `createPieceCountNode` splits each path. @internal */
export type LinePieceMode = 'chunk' | 'substring';

/**
 * Per-input-path piece count: chunks `min(ceil(length / chunkLength), maximumPieces)` (one piece
 * for zero-length paths or a non-positive chunk length), substrings one piece. Empty paths have
 * no pieces.
 *
 * @internal
 */
export function createPieceCountNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    mode: LinePieceMode;
    pathOffsets: GraphDataView<'uint32'>;
    rowMeasures: GraphDataView<'float32'>;
    parameters: GraphDataView<'float32'>;
    rowCount: number;
    maximumPieces: number;
    pieceCounts: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createMapGraphKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'piece-count',
    bindings: [
      {name: 'pathOffsets', view: props.pathOffsets, type: 'u32', access: 'read'},
      {name: 'rowMeasures', view: props.rowMeasures, type: 'f32', access: 'read'},
      {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
      {name: 'pieceCounts', view: props.pieceCounts, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.pathOffsets.length - 1,
    declarations: `const ROW_COUNT: u32 = ${props.rowCount}u;
const MAXIMUM_PIECES: u32 = ${props.maximumPieces}u;
const IS_CHUNK: bool = ${props.mode === 'chunk'};`,
    body: /* wgsl */ `let start = min(pathOffsets[pathOffsetsOffset + index], ROW_COUNT);
  let end = min(max(pathOffsets[pathOffsetsOffset + index + 1u], start), ROW_COUNT);
  var count = 0u;
  if (end > start) {
    count = 1u;
    let pathLength = rowMeasures[rowMeasuresOffset + end - 1u];
    let chunkLength = parameters[parametersOffset];
    if (IS_CHUNK && pathLength > 0.0 && chunkLength > 0.0) {
      count = u32(min(ceil(pathLength / chunkLength), f32(MAXIMUM_PIECES)));
      count = clamp(count, 1u, MAXIMUM_PIECES);
    }
  }
  pieceCounts[pieceCountsOffset + index] = count;`
  });
}

/**
 * Per output piece slot: finds the source path by binary search over the scanned piece starts,
 * resolves the piece's measure range, the interior vertex rows strictly inside it (two binary
 * searches over the cumulative measures) and its vertex count. Slot 0 also writes the unclamped
 * piece total.
 *
 * @internal
 */
export function createPieceRangeNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    mode: LinePieceMode;
    pathOffsets: GraphDataView<'uint32'>;
    rowMeasures: GraphDataView<'float32'>;
    parameters: GraphDataView<'float32'>;
    pieceCounts: GraphDataView<'uint32'>;
    pieceStarts: GraphDataView<'uint32'>;
    rowCount: number;
    pieces: GraphDataView<'uint32'>;
    vertexCounts: GraphDataView<'uint32'>;
    pieceTotal: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const pathCount = props.pathOffsets.length - 1;
  return createMapGraphKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'piece-range',
    bindings: [
      {name: 'pathOffsets', view: props.pathOffsets, type: 'u32', access: 'read'},
      {name: 'rowMeasures', view: props.rowMeasures, type: 'f32', access: 'read'},
      {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
      {name: 'pieceCounts', view: props.pieceCounts, type: 'u32', access: 'read'},
      {name: 'pieceStarts', view: props.pieceStarts, type: 'u32', access: 'read'},
      {name: 'pieces', view: props.pieces, type: 'u32', access: 'read_write'},
      {name: 'vertexCounts', view: props.vertexCounts, type: 'u32', access: 'read_write'},
      {name: 'pieceTotal', view: props.pieceTotal, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.vertexCounts.length,
    declarations: `const PATH_COUNT: u32 = ${pathCount}u;
const ROW_COUNT: u32 = ${props.rowCount}u;
const STRIDE: u32 = ${LINE_PIECE_STRIDE}u;
const IS_CHUNK: bool = ${props.mode === 'chunk'};
const PIECE_COPY: u32 = ${PIECE_COPY}u;

fn getMeasure(row: u32) -> f32 {
  return rowMeasures[rowMeasuresOffset + row];
}

// First row in [low, high) whose measure is greater than (strict) or at least (!strict) the value.
fn findRow(low: u32, high: u32, value: f32, strict: bool) -> u32 {
  var first = low;
  var last = high;
  while (first < last) {
    let middle = (first + last) / 2u;
    let measure = getMeasure(middle);
    if ((strict && measure <= value) || (!strict && measure < value)) {
      first = middle + 1u;
    } else {
      last = middle;
    }
  }
  return first;
}`,
    body: /* wgsl */ `let total = pieceStarts[pieceStartsOffset + PATH_COUNT - 1u] + pieceCounts[pieceCountsOffset + PATH_COUNT - 1u];
  if (index == 0u) {
    pieceTotal[pieceTotalOffset] = total;
  }
  let base = piecesOffset + index * STRIDE;
  if (index >= total) {
    pieces[base] = 0xffffffffu;
    pieces[base + 6u] = 0u;
    vertexCounts[vertexCountsOffset + index] = 0u;
    return;
  }
  // Largest path whose first piece is at most index (paths without pieces are skipped).
  var low = 0u;
  var high = PATH_COUNT;
  while (high - low > 1u) {
    let middle = (low + high) / 2u;
    if (pieceStarts[pieceStartsOffset + middle] <= index) {
      low = middle;
    } else {
      high = middle;
    }
  }
  let path = low;
  let local = index - pieceStarts[pieceStartsOffset + path];
  let pieceCount = pieceCounts[pieceCountsOffset + path];
  let start = min(pathOffsets[pathOffsetsOffset + path], ROW_COUNT);
  let end = min(max(pathOffsets[pathOffsetsOffset + path + 1u], start), ROW_COUNT);
  let pathLength = getMeasure(end - 1u);
  var startMeasure = 0.0;
  var endMeasure = pathLength;
  var flags = 0u;
  var firstInterior = start;
  var interiorEnd = end;
  var vertexCount = end - start;
  if (!(pathLength > 0.0)) {
    flags = PIECE_COPY;
  } else {
    if (IS_CHUNK) {
      let chunkLength = parameters[parametersOffset];
      if (chunkLength > 0.0) {
        startMeasure = f32(local) * chunkLength;
        if (local + 1u < pieceCount) {
          endMeasure = f32(local + 1u) * chunkLength;
        }
      }
    } else {
      startMeasure = clamp(parameters[parametersOffset + 1u], 0.0, pathLength);
      endMeasure = clamp(parameters[parametersOffset + 2u], 0.0, pathLength);
    }
    if (startMeasure > endMeasure) {
      vertexCount = 0u;
      firstInterior = start;
      interiorEnd = start;
    } else {
      firstInterior = findRow(start, end, startMeasure, true);
      interiorEnd = max(findRow(start, end, endMeasure, false), firstInterior);
      vertexCount = 2u + interiorEnd - firstInterior;
    }
  }
  pieces[base] = path;
  pieces[base + 1u] = bitcast<u32>(startMeasure);
  pieces[base + 2u] = bitcast<u32>(endMeasure);
  pieces[base + 3u] = flags;
  pieces[base + 4u] = firstInterior;
  pieces[base + 5u] = interiorEnd;
  pieces[base + 6u] = vertexCount;
  pieces[base + 7u] = start;
  pieces[base + 8u] = end;
  vertexCounts[vertexCountsOffset + index] = vertexCount;`
  });
}

/**
 * Emits every piece: an interpolated start vertex, the interior input vertices, and an
 * interpolated end vertex (or the path's rows verbatim for zero-length paths).
 *
 * @internal
 */
export function createPieceEmitNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    coordinateSystem: GPULineCoordinateSystem;
    positions: GraphDataView<'float32x2'>;
    rowMeasures: GraphDataView<'float32'>;
    rowShifts?: GraphDataView<'float32'>;
    pieces: GraphDataView<'uint32'>;
    vertexStarts: GraphDataView<'uint32'>;
    outputPositions: GraphDataView<'float32x2'>;
    outputMeasures?: GraphDataView<'float32'>;
    outputSourceRows?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const spherical = props.coordinateSystem === 'spherical';
  const bindings: MapGraphKernelBinding[] = [
    {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
    {name: 'rowMeasures', view: props.rowMeasures, type: 'f32', access: 'read'},
    {name: 'pieces', view: props.pieces, type: 'u32', access: 'read'},
    {name: 'vertexStarts', view: props.vertexStarts, type: 'u32', access: 'read'},
    {name: 'outputPositions', view: props.outputPositions, type: 'f32', access: 'read_write'}
  ];
  if (spherical && props.rowShifts) {
    bindings.push({name: 'rowShifts', view: props.rowShifts, type: 'f32', access: 'read'});
  }
  if (props.outputMeasures) {
    bindings.push({
      name: 'outputMeasures',
      view: props.outputMeasures,
      type: 'f32',
      access: 'read_write'
    });
  }
  if (props.outputSourceRows) {
    bindings.push({
      name: 'outputSourceRows',
      view: props.outputSourceRows,
      type: 'u32',
      access: 'read_write'
    });
  }
  return createMapGraphKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'piece-emit',
    bindings,
    invocationCount: props.vertexStarts.length,
    declarations: `const CAPACITY: u32 = ${props.outputPositions.length}u;
const STRIDE: u32 = ${LINE_PIECE_STRIDE}u;
const PIECE_COPY: u32 = ${PIECE_COPY}u;
${spherical ? GEODESIC_WGSL : ''}

fn getMeasure(row: u32) -> f32 {
  return rowMeasures[rowMeasuresOffset + row];
}

// Position of an input row, with spherical longitudes unwrapped along the path.
fn getVertex(row: u32) -> vec2<f32> {
  var vertex = vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
  ${spherical && props.rowShifts ? 'vertex.x = vertex.x + rowShifts[rowShiftsOffset + row];' : ''}
  return vertex;
}

fn interpolate(segmentRow: u32, measure: f32) -> vec2<f32> {
  let start = getVertex(segmentRow);
  let end = getVertex(segmentRow + 1u);
  let segmentMeasure = getMeasure(segmentRow + 1u) - getMeasure(segmentRow);
  var fraction = 0.0;
  if (segmentMeasure > 0.0) {
    fraction = clamp((measure - getMeasure(segmentRow)) / segmentMeasure, 0.0, 1.0);
  }
  ${
    spherical
      ? 'return geodesicInterpolate(start, end, geodesicCentralAngle(start, end), fraction);'
      : 'return mix(start, end, fraction);'
  }
}

fn writeVertex(outputRow: u32, vertex: vec2<f32>, measure: f32, sourceRow: u32) {
  if (outputRow >= CAPACITY) {
    return;
  }
  outputPositions[outputPositionsOffset + 2u * outputRow] = vertex.x;
  outputPositions[outputPositionsOffset + 2u * outputRow + 1u] = vertex.y;
  ${props.outputMeasures ? 'outputMeasures[outputMeasuresOffset + outputRow] = measure;' : ''}
  ${props.outputSourceRows ? 'outputSourceRows[outputSourceRowsOffset + outputRow] = sourceRow;' : ''}
}`,
    body: /* wgsl */ `let base = piecesOffset + index * STRIDE;
  let vertexCount = pieces[base + 6u];
  if (vertexCount == 0u) {
    return;
  }
  let outputStart = vertexStarts[vertexStartsOffset + index];
  if (outputStart >= CAPACITY) {
    return;
  }
  let flags = pieces[base + 3u];
  let firstInterior = pieces[base + 4u];
  let interiorEnd = pieces[base + 5u];
  if ((flags & PIECE_COPY) != 0u) {
    for (var row = firstInterior; row < interiorEnd; row++) {
      writeVertex(outputStart + row - firstInterior, getVertex(row), getMeasure(row), row);
    }
    return;
  }
  let pathStart = pieces[base + 7u];
  let pathEnd = pieces[base + 8u];
  let startMeasure = bitcast<f32>(pieces[base + 1u]);
  let endMeasure = bitcast<f32>(pieces[base + 2u]);
  // A measure lies on the segment ending at the first row past it, clamped to the path's
  // segments (the path has at least two rows because its length is positive).
  let startSegment = clamp(firstInterior, pathStart + 1u, pathEnd - 1u) - 1u;
  let endSegment = clamp(interiorEnd, pathStart + 1u, pathEnd - 1u) - 1u;
  writeVertex(outputStart, interpolate(startSegment, startMeasure), startMeasure, startSegment);
  for (var row = firstInterior; row < interiorEnd; row++) {
    writeVertex(outputStart + 1u + row - firstInterior, getVertex(row), getMeasure(row), row);
  }
  writeVertex(
    outputStart + 1u + interiorEnd - firstInterior,
    interpolate(endSegment, endMeasure),
    endMeasure,
    endSegment
  );`
  });
}

/**
 * Publishes piece path offsets (clamped), the vertex count and overflow (vertex or path capacity),
 * the optional total and clamped path count, and the optional source-path column.
 *
 * @internal
 */
export function createPiecePublishNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    pieceTotal: GraphDataView<'uint32'>;
    pieces: GraphDataView<'uint32'>;
    vertexStarts: GraphDataView<'uint32'>;
    capacity: number;
    outputPathOffsets: GraphDataView<'uint32'>;
    count: GraphDataView<'uint32'>;
    overflow: GraphDataView<'uint32'>;
    totalCount?: GraphDataView<'uint32'>;
    pathCountOutput?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const pathCapacity = props.vertexStarts.length;
  const bindings: MapGraphKernelBinding[] = [
    {name: 'pieceTotal', view: props.pieceTotal, type: 'u32', access: 'read'},
    {name: 'pieces', view: props.pieces, type: 'u32', access: 'read'},
    {name: 'vertexStarts', view: props.vertexStarts, type: 'u32', access: 'read'},
    {name: 'outputPathOffsets', view: props.outputPathOffsets, type: 'u32', access: 'read_write'},
    {name: 'countOut', view: props.count, type: 'u32', access: 'read_write'},
    {name: 'overflowOut', view: props.overflow, type: 'u32', access: 'read_write'}
  ];
  if (props.totalCount) {
    bindings.push({name: 'totalOut', view: props.totalCount, type: 'u32', access: 'read_write'});
  }
  if (props.pathCountOutput) {
    bindings.push({
      name: 'pathCountOut',
      view: props.pathCountOutput,
      type: 'u32',
      access: 'read_write'
    });
  }
  return createMapGraphKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'piece-publish',
    bindings,
    invocationCount: pathCapacity + 1,
    declarations: `const PATH_CAPACITY: u32 = ${pathCapacity}u;
const CAPACITY: u32 = ${props.capacity}u;
const STRIDE: u32 = ${LINE_PIECE_STRIDE}u;`,
    body: /* wgsl */ `let lastBase = piecesOffset + (PATH_CAPACITY - 1u) * STRIDE;
  let total = vertexStarts[vertexStartsOffset + PATH_CAPACITY - 1u] + pieces[lastBase + 6u];
  var start = total;
  if (index < PATH_CAPACITY) {
    start = vertexStarts[vertexStartsOffset + index];
  }
  outputPathOffsets[outputPathOffsetsOffset + index] = min(start, CAPACITY);
  if (index == 0u) {
    let pieceTotal = pieceTotal[pieceTotalOffset];
    countOut[countOutOffset] = min(total, CAPACITY);
    overflowOut[overflowOutOffset] = select(0u, 1u, total > CAPACITY || pieceTotal > PATH_CAPACITY);
    ${props.totalCount ? 'totalOut[totalOutOffset] = total;' : ''}
    ${props.pathCountOutput ? 'pathCountOut[pathCountOutOffset] = min(pieceTotal, PATH_CAPACITY);' : ''}
  }`
  });
}

/**
 * Writes `sourcePaths[slot]` from the piece column (`0xffffffff` past the piece count).
 *
 * @internal
 */
export function createPieceSourcePathsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    pieces: GraphDataView<'uint32'>;
    sourcePaths: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createMapGraphKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'piece-source-paths',
    bindings: [
      {name: 'pieces', view: props.pieces, type: 'u32', access: 'read'},
      {name: 'sourcePaths', view: props.sourcePaths, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.sourcePaths.length,
    declarations: `const STRIDE: u32 = ${LINE_PIECE_STRIDE}u;`,
    body: 'sourcePaths[sourcePathsOffset + index] = pieces[piecesOffset + index * STRIDE];'
  });
}
