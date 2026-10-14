// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {GEODESIC_WGSL} from '../geometry-measures/geodesic-wgsl';
import type {GPULineCoordinateSystem} from './line-segmentize-types';
import {BALANCED_SEARCH_WORKGROUP_SIZE, getBalancedSearchSource} from './balanced-search';

/**
 * Words per output piece in the internal piece column:
 * `[line, startMeasure, endMeasure, flags, firstInteriorRow, interiorEndRow, vertexCount, lineStart,
 * lineEnd, 0]`.
 *
 * @internal
 */
export const LINE_PIECE_STRIDE = 10;

/** Piece flag: copy the line's rows verbatim (zero-length lines). @internal */
const PIECE_COPY = 1;

/** How `createPieceCountNode` splits each line. @internal */
export type LinePieceMode = 'chunk' | 'substring';

/**
 * Per-input-line piece count: chunks `min(ceil(length / chunkLength), maximumPieces)` (one piece
 * for zero-length lines or a non-positive chunk length), substrings one piece. Empty lines have
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
    lineOffsets: GraphDataView<'uint32'>;
    rowMeasures: GraphDataView<'float32'>;
    parameters: GraphDataView<'float32'>;
    rowCount: number;
    maximumPieces: number;
    pieceCounts: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'piece-count',
    bindings: [
      {name: 'lineOffsets', view: props.lineOffsets, type: 'u32', access: 'read'},
      {name: 'rowMeasures', view: props.rowMeasures, type: 'f32', access: 'read'},
      {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
      {name: 'pieceCounts', view: props.pieceCounts, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.lineOffsets.length - 1,
    declarations: `const ROW_COUNT: u32 = ${props.rowCount}u;
const MAXIMUM_PIECES: u32 = ${props.maximumPieces}u;
const IS_CHUNK: bool = ${props.mode === 'chunk'};`,
    body: /* wgsl */ `let start = min(lineOffsets[lineOffsetsOffset + index], ROW_COUNT);
  let end = min(max(lineOffsets[lineOffsetsOffset + index + 1u], start), ROW_COUNT);
  var count = 0u;
  if (end > start) {
    count = 1u;
    let lineLength = rowMeasures[rowMeasuresOffset + end - 1u];
    let chunkLength = parameters[parametersOffset];
    if (IS_CHUNK && lineLength > 0.0 && chunkLength > 0.0) {
      count = u32(min(ceil(lineLength / chunkLength), f32(MAXIMUM_PIECES)));
      count = clamp(count, 1u, MAXIMUM_PIECES);
    }
  }
  pieceCounts[pieceCountsOffset + index] = count;`
  });
}

/**
 * Per output piece slot: finds the source line by binary search over the scanned piece starts,
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
    lineOffsets: GraphDataView<'uint32'>;
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
  const lineCount = props.lineOffsets.length - 1;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'piece-range',
    bindings: [
      {name: 'lineOffsets', view: props.lineOffsets, type: 'u32', access: 'read'},
      {name: 'rowMeasures', view: props.rowMeasures, type: 'f32', access: 'read'},
      {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
      {name: 'pieceCounts', view: props.pieceCounts, type: 'u32', access: 'read'},
      {name: 'pieceStarts', view: props.pieceStarts, type: 'u32', access: 'read'},
      {name: 'pieces', view: props.pieces, type: 'u32', access: 'read_write'},
      {name: 'vertexCounts', view: props.vertexCounts, type: 'u32', access: 'read_write'},
      {name: 'pieceTotal', view: props.pieceTotal, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.vertexCounts.length,
    declarations: `const LINE_COUNT: u32 = ${lineCount}u;
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
    body: /* wgsl */ `let total = pieceStarts[pieceStartsOffset + LINE_COUNT - 1u] + pieceCounts[pieceCountsOffset + LINE_COUNT - 1u];
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
  // Largest line whose first piece is at most index (lines without pieces are skipped).
  var low = 0u;
  var high = LINE_COUNT;
  while (high - low > 1u) {
    let middle = (low + high) / 2u;
    if (pieceStarts[pieceStartsOffset + middle] <= index) {
      low = middle;
    } else {
      high = middle;
    }
  }
  let line = low;
  let local = index - pieceStarts[pieceStartsOffset + line];
  let pieceCount = pieceCounts[pieceCountsOffset + line];
  let start = min(lineOffsets[lineOffsetsOffset + line], ROW_COUNT);
  let end = min(max(lineOffsets[lineOffsetsOffset + line + 1u], start), ROW_COUNT);
  let lineLength = getMeasure(end - 1u);
  var startMeasure = 0.0;
  var endMeasure = lineLength;
  var flags = 0u;
  var firstInterior = start;
  var interiorEnd = end;
  var vertexCount = end - start;
  if (!(lineLength > 0.0)) {
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
      startMeasure = clamp(parameters[parametersOffset + 1u], 0.0, lineLength);
      endMeasure = clamp(parameters[parametersOffset + 2u], 0.0, lineLength);
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
  pieces[base] = line;
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
 * interpolated end vertex (or the line's rows verbatim for zero-length lines).
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
  const bindings: WGSLKernelBinding[] = [
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
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'piece-emit',
    bindings,
    // One invocation per output vertex (not per piece), so a piece spanning a long line copies in
    // parallel instead of serializing one lane (see balanced-search).
    invocationCount: props.outputPositions.length,
    workgroupSize: BALANCED_SEARCH_WORKGROUP_SIZE,
    guardIndex: false,
    declarations: `const CAPACITY: u32 = ${props.outputPositions.length}u;
const STRIDE: u32 = ${LINE_PIECE_STRIDE}u;
const PIECE_COPY: u32 = ${PIECE_COPY}u;
${spherical ? GEODESIC_WGSL : ''}
${getBalancedSearchSource(props.vertexStarts.length)}
fn balancedStart(piece: u32) -> u32 { return vertexStarts[vertexStartsOffset + piece]; }
fn balancedCount(piece: u32) -> u32 { return pieces[piecesOffset + piece * STRIDE + 6u]; }

fn getMeasure(row: u32) -> f32 {
  return rowMeasures[rowMeasuresOffset + row];
}

// Position of an input row, with spherical longitudes unwrapped along the line.
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
    body: /* wgsl */ `let piece = findBalancedOwner(index, workgroupIndex * BALANCED_LANES, localInvocationIndex);
  if (piece == BALANCED_NONE) {
    return;
  }
  let base = piecesOffset + piece * STRIDE;
  let local = index - vertexStarts[vertexStartsOffset + piece];
  let flags = pieces[base + 3u];
  let firstInterior = pieces[base + 4u];
  let interiorEnd = pieces[base + 5u];
  if ((flags & PIECE_COPY) != 0u) {
    let row = firstInterior + local;
    writeVertex(index, getVertex(row), getMeasure(row), row);
    return;
  }
  let lineStart = pieces[base + 7u];
  let lineEnd = pieces[base + 8u];
  let startMeasure = bitcast<f32>(pieces[base + 1u]);
  let endMeasure = bitcast<f32>(pieces[base + 2u]);
  // A measure lies on the segment ending at the first row past it, clamped to the line's
  // segments (the line has at least two rows because its length is positive).
  let startSegment = clamp(firstInterior, lineStart + 1u, lineEnd - 1u) - 1u;
  let endSegment = clamp(interiorEnd, lineStart + 1u, lineEnd - 1u) - 1u;
  if (local == 0u) {
    writeVertex(index, interpolate(startSegment, startMeasure), startMeasure, startSegment);
  } else if (local <= interiorEnd - firstInterior) {
    let row = firstInterior + local - 1u;
    writeVertex(index, getVertex(row), getMeasure(row), row);
  } else {
    writeVertex(index, interpolate(endSegment, endMeasure), endMeasure, endSegment);
  }`
  });
}

/**
 * Publishes piece line offsets (clamped), the vertex count and overflow (vertex or line capacity),
 * the optional total and clamped line count, and the optional source-line-index column.
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
    outputLineOffsets: GraphDataView<'uint32'>;
    count: GraphDataView<'uint32'>;
    overflow: GraphDataView<'uint32'>;
    requiredCount?: GraphDataView<'uint32'>;
    lineCountOutput?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const lineCapacity = props.vertexStarts.length;
  const bindings: WGSLKernelBinding[] = [
    {name: 'pieceTotal', view: props.pieceTotal, type: 'u32', access: 'read'},
    {name: 'pieces', view: props.pieces, type: 'u32', access: 'read'},
    {name: 'vertexStarts', view: props.vertexStarts, type: 'u32', access: 'read'},
    {name: 'outputLineOffsets', view: props.outputLineOffsets, type: 'u32', access: 'read_write'},
    {name: 'countOut', view: props.count, type: 'u32', access: 'read_write'},
    {name: 'overflowOut', view: props.overflow, type: 'u32', access: 'read_write'}
  ];
  if (props.requiredCount) {
    bindings.push({name: 'totalOut', view: props.requiredCount, type: 'u32', access: 'read_write'});
  }
  if (props.lineCountOutput) {
    bindings.push({
      name: 'lineCountOut',
      view: props.lineCountOutput,
      type: 'u32',
      access: 'read_write'
    });
  }
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'piece-publish',
    bindings,
    invocationCount: lineCapacity + 1,
    declarations: `const LINE_CAPACITY: u32 = ${lineCapacity}u;
const CAPACITY: u32 = ${props.capacity}u;
const STRIDE: u32 = ${LINE_PIECE_STRIDE}u;`,
    body: /* wgsl */ `let lastBase = piecesOffset + (LINE_CAPACITY - 1u) * STRIDE;
  let total = vertexStarts[vertexStartsOffset + LINE_CAPACITY - 1u] + pieces[lastBase + 6u];
  var start = total;
  if (index < LINE_CAPACITY) {
    start = vertexStarts[vertexStartsOffset + index];
  }
  outputLineOffsets[outputLineOffsetsOffset + index] = min(start, CAPACITY);
  if (index == 0u) {
    let pieceTotal = pieceTotal[pieceTotalOffset];
    countOut[countOutOffset] = min(total, CAPACITY);
    overflowOut[overflowOutOffset] = select(0u, 1u, total > CAPACITY || pieceTotal > LINE_CAPACITY);
    ${props.requiredCount ? 'totalOut[totalOutOffset] = total;' : ''}
    ${props.lineCountOutput ? 'lineCountOut[lineCountOutOffset] = min(pieceTotal, LINE_CAPACITY);' : ''}
  }`
  });
}

/**
 * Writes `sourceLineIndices[slot]` from the piece column (`0xffffffff` past the piece count).
 *
 * @internal
 */
export function createPieceSourceLineIndicesNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    pieces: GraphDataView<'uint32'>;
    sourceLineIndices: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'piece-source-line-indexs',
    bindings: [
      {name: 'pieces', view: props.pieces, type: 'u32', access: 'read'},
      {name: 'sourceLineIndices', view: props.sourceLineIndices, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.sourceLineIndices.length,
    declarations: `const STRIDE: u32 = ${LINE_PIECE_STRIDE}u;`,
    body: 'sourceLineIndices[sourceLineIndicesOffset + index] = pieces[piecesOffset + index * STRIDE];'
  });
}
