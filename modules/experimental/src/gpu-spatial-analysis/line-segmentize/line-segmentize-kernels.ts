// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import {LON_LAT_RING_WGSL} from '../../analysis-core/lon-lat-ring-wgsl';
import type {GPUStatusScalar} from '../../analysis-core/contracts/index';
import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {GEODESIC_WGSL} from '../geometry-measures/geodesic-wgsl';
import type {GPULineCoordinateSystem} from './line-segmentize-types';
import {BALANCED_SEARCH_WORKGROUP_SIZE, getBalancedSearchSource} from './balanced-search';

/**
 * WGSL `findLine(row) -> u32`: upper-bound binary search over the bound `lineOffsets`
 * (`LINE_COUNT + 1` rows). Returns the line owning `row`, or `NO_LINE` when the row lies outside
 * every line. Empty lines are skipped naturally.
 *
 * @internal
 */
export const FIND_LINE_WGSL = /* wgsl */ `
const NO_LINE: u32 = 0xffffffffu;

fn findLine(row: u32) -> u32 {
  var low = 0u;
  var high = LINE_COUNT + 1u;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (lineOffsets[lineOffsetsOffset + middle] <= row) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  if (low == 0u || low > LINE_COUNT) {
    return NO_LINE;
  }
  return low - 1u;
}
`;

/**
 * WGSL segment helpers over the bound `positions` (f32 pairs) for one coordinate system:
 * `getPosition(row)`, `getSegmentLength(a, b)` in output units (planar units or radius units).
 *
 * @internal
 */
export function getLineGeometrySource(
  coordinateSystem: GPULineCoordinateSystem,
  radius: number
): string {
  const lengthSource =
    coordinateSystem === 'spherical'
      ? `return geodesicCentralAngle(a, b) * RADIUS;`
      : `return length(b - a);`;
  return /* wgsl */ `
${coordinateSystem === 'spherical' ? GEODESIC_WGSL : ''}
const RADIUS: f32 = ${getWGSLFloatLiteral(radius)};

fn getPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}

fn getSegmentLength(a: vec2<f32>, b: vec2<f32>) -> f32 {
  ${lengthSource}
}
`;
}

/** Properties for {@link createLinePrefixNode}. @internal */
export type LinePrefixNodeProps = {
  /** Preserve degree-plane pole edges and use spherical angular extents for polygon preparation. */
  polygonEdges?: boolean;
  id: string;
  operation: string;
  positions: GraphDataView<'float32x2'>;
  lineOffsets: GraphDataView<'uint32'>;
  coordinateSystem: GPULineCoordinateSystem;
  radius: number;
  /** Per-row measure of the row's vertex from its line start. Rows outside lines are untouched. */
  rowMeasures?: GraphDataView<'float32'>;
  /**
   * Per-row longitude shift (a multiple of 360) that makes longitudes continuous along the line.
   * Spherical only.
   */
  rowShifts?: GraphDataView<'float32'>;
  /** Integer prefix turns, used instead of rowShifts by wrapped-output consumers. */
  rowTurns?: GraphDataView<'sint32'>;
  /** Rings close implicitly; the prefix still measures only up to each input vertex. */
  closedLines?: boolean;
};

/** Lanes of the workgroup that scans one line in {@link createLinePrefixNode}. */
const PREFIX_LANES = 64;

/**
 * Builds the per-line prefix: cumulative measures and longitude unwrapping shifts.
 *
 * Block hybrid: workgroup `w` owns lines `[PREFIX_LANES * w, PREFIX_LANES * w + PREFIX_LANES)`.
 * Phase 1: every lane walks its own line serially when it has at most `PREFIX_LANES` rows, with a
 * Neumaier-compensated f32 sum in row order (deterministic, and bit-identical to the former
 * one-thread-per-line kernel); longer lines are queued. Phase 2: the workgroup visits the queued
 * lines of its block in ascending order and scans each cooperatively in tiles of `PREFIX_LANES`
 * segments: each tile is a Hillis-Steele inclusive scan in workgroup memory, and the running total
 * carries between tiles in compensated form. That cuts the dependent chain of a line of `n` rows
 * from `n` global read-modify-write steps to `n / 64` tiles of 6 scan rounds, so one very long line
 * no longer stalls the stage, while many short lines still cost one thread each (not one
 * workgroup each). Shifts are sums of multiples of 360, exact in f32 either way; long-line
 * measures differ from the serial sum only by f32 rounding (about `log2(64)` roundings per tile).
 *
 * @internal
 */
export function createLinePrefixNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: LinePrefixNodeProps
): GPUCommandNode<Parameters> {
  // Select one prefix representation; integer turns never accumulate unwrapped float32 positions.
  assert(!(props.rowTurns && props.rowShifts));
  const shiftView = props.rowTurns ?? props.rowShifts;
  const shiftType = props.rowTurns ? 'i32' : 'f32';
  const shiftZero = props.rowTurns ? '0i' : '0.0';
  const rawShiftStep = props.rowTurns
    ? '-i32(round((current.x - previous.x) / 360.0))'
    : '-360.0 * round((current.x - previous.x) / 360.0)';
  const shiftStep = props.polygonEdges
    ? `select(${rawShiftStep}, ${shiftZero}, lonLatIsPoleEdge(previous, current))`
    : rawShiftStep;
  const shiftCarry = props.rowTurns ? 'turnCarry[0]' : 'prefixCarry[2]';
  const lineCount = props.lineOffsets.length - 1;
  const bindings: WGSLKernelBinding[] = [
    {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
    {
      name: 'lineOffsets',
      view: props.lineOffsets,
      type: 'u32',
      access: 'read'
    }
  ];
  if (props.rowMeasures) {
    bindings.push({
      name: 'rowMeasures',
      view: props.rowMeasures,
      type: 'f32',
      access: 'read_write'
    });
  }
  if (shiftView) {
    bindings.push({
      name: 'rowShifts',
      view: shiftView,
      type: shiftType,
      access: 'read_write'
    });
  }
  const {rowMeasures} = props;
  const rowShifts = shiftView;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'line-prefix',
    bindings,
    invocationCount: Math.ceil(lineCount / PREFIX_LANES) * PREFIX_LANES,
    workgroupSize: PREFIX_LANES,
    guardIndex: false,
    declarations: `const LINE_COUNT: u32 = ${lineCount}u;
const ROW_COUNT: u32 = ${props.positions.length}u;
const PREFIX_LANES: u32 = ${PREFIX_LANES}u;
var<workgroup> largeBits: array<atomic<u32>, 2>;
var<workgroup> largeBitsCopy: array<u32, 2>;
var<workgroup> largeStart: array<u32, ${PREFIX_LANES}>;
var<workgroup> largeEnd: array<u32, ${PREFIX_LANES}>;
var<workgroup> prefixCarry: array<f32, 3>;
var<workgroup> tileLengths: array<f32, ${PREFIX_LANES}>;
var<workgroup> tileShifts: array<${shiftType}, ${PREFIX_LANES}>;
${props.rowTurns ? 'var<workgroup> turnCarry: array<i32, 1>;' : ''}
${getLineGeometrySource(props.coordinateSystem, props.radius)}
${props.polygonEdges ? LON_LAT_RING_WGSL : ''}`,
    body: /* wgsl */ `let lane = localInvocationIndex;
  let lineBase = workgroupIndex * PREFIX_LANES;
  let ownLine = lineBase + lane;
  // Phase 1: this lane's own line, serially when small; larger lines are queued for phase 2.
  if (ownLine < LINE_COUNT) {
    let first = min(lineOffsets[lineOffsetsOffset + ownLine], ROW_COUNT);
    let last = min(max(lineOffsets[lineOffsetsOffset + ownLine + 1u], first), ROW_COUNT);
    if (last > first) {
      if (last - first <= PREFIX_LANES) {
        var sum = 0.0;
        var compensation = 0.0;
        var shift = ${shiftZero};
        var previous = getPosition(first);
        ${rowMeasures ? 'rowMeasures[rowMeasuresOffset + first] = 0.0;' : ''}
        ${rowShifts ? `rowShifts[rowShiftsOffset + first] = ${shiftZero};` : ''}
        for (var row = first + 1u; row < last; row++) {
          let current = getPosition(row);
          ${
            rowMeasures
              ? `let segmentLength = getSegmentLength(previous, current);
          let nextSum = sum + segmentLength;
          if (abs(sum) >= abs(segmentLength)) {
            compensation = compensation + ((sum - nextSum) + segmentLength);
          } else {
            compensation = compensation + ((segmentLength - nextSum) + sum);
          }
          sum = nextSum;
          rowMeasures[rowMeasuresOffset + row] = sum + compensation;`
              : ''
          }
          ${
            rowShifts
              ? `shift = shift + ${shiftStep};
          rowShifts[rowShiftsOffset + row] = shift;`
              : ''
          }
          previous = current;
        }
      } else {
        largeStart[lane] = first;
        largeEnd[lane] = last;
        atomicOr(&largeBits[lane / 32u], 1u << (lane % 32u));
      }
    }
  }
  workgroupBarrier();
  if (lane == 0u) {
    largeBitsCopy[0] = atomicLoad(&largeBits[0]);
    largeBitsCopy[1] = atomicLoad(&largeBits[1]);
  }
  // Phase 2: every barrier below is in workgroup-uniform control flow (the queue is read through
  // workgroupUniformLoad), and the queued lines are visited in ascending order.
  for (var word = 0u; word < 2u; word++) {
    var pending = workgroupUniformLoad(&largeBitsCopy[word]);
    while (pending != 0u) {
      let slot = word * 32u + firstTrailingBit(pending);
      pending = pending & (pending - 1u);
      let lineStart = workgroupUniformLoad(&largeStart[slot]);
      let lineEnd = workgroupUniformLoad(&largeEnd[slot]);
      if (lane == 0u) {
        ${rowMeasures ? 'rowMeasures[rowMeasuresOffset + lineStart] = 0.0;' : ''}
        ${rowShifts ? `rowShifts[rowShiftsOffset + lineStart] = ${shiftZero};` : ''}
        prefixCarry[0] = 0.0;
        prefixCarry[1] = 0.0;
        ${shiftCarry} = ${shiftZero};
      }
      for (var tileStart = lineStart + 1u; tileStart < lineEnd; tileStart += PREFIX_LANES) {
        let row = tileStart + lane;
        var segmentLength = 0.0;
        var segmentShift = ${shiftZero};
        if (row < lineEnd) {
          let previous = getPosition(row - 1u);
          let current = getPosition(row);
          ${rowMeasures ? 'segmentLength = getSegmentLength(previous, current);' : ''}
          ${rowShifts ? `segmentShift = ${shiftStep};` : ''}
        }
        tileLengths[lane] = segmentLength;
        tileShifts[lane] = segmentShift;
        workgroupBarrier();
        for (var offset = 1u; offset < PREFIX_LANES; offset = offset << 1u) {
          var addedLength = 0.0;
          var addedShift = ${shiftZero};
          if (lane >= offset) {
            addedLength = tileLengths[lane - offset];
            addedShift = tileShifts[lane - offset];
          }
          workgroupBarrier();
          tileLengths[lane] += addedLength;
          tileShifts[lane] += addedShift;
          workgroupBarrier();
        }
        if (row < lineEnd) {
          ${rowMeasures ? 'rowMeasures[rowMeasuresOffset + row] = (prefixCarry[0] + tileLengths[lane]) + prefixCarry[1];' : ''}
          ${rowShifts ? `rowShifts[rowShiftsOffset + row] = ${shiftCarry} + tileShifts[lane];` : ''}
        }
        workgroupBarrier();
        if (lane == 0u) {
          // Neumaier-compensated carry of the tile total into the running measure.
          let total = tileLengths[PREFIX_LANES - 1u];
          let sum = prefixCarry[0];
          let nextSum = sum + total;
          if (abs(sum) >= abs(total)) {
            prefixCarry[1] += (sum - nextSum) + total;
          } else {
            prefixCarry[1] += (total - nextSum) + sum;
          }
          prefixCarry[0] = nextSum;
          ${shiftCarry} += tileShifts[PREFIX_LANES - 1u];
        }
        workgroupBarrier();
      }
    }
  }`
  });
}

/** Properties for {@link createSegmentizeCountNode}. @internal */
export type SegmentizeCountNodeProps = {
  /** Preserve degree-plane pole edges and use spherical angular extents for polygon preparation. */
  polygonEdges?: boolean;
  id: string;
  operation: string;
  positions: GraphDataView<'float32x2'>;
  lineOffsets: GraphDataView<'uint32'>;
  parameters: GraphDataView<'float32'>;
  coordinateSystem: GPULineCoordinateSystem;
  radius: number;
  maximumPieceCount: number;
  /** Include the implicit closing edge and discard a bit-exact repeated closing row. */
  closedLines?: boolean;
  /** Degree-step uses angular extent rather than a metric length. Default length. */
  pieceCountModel?: 'length' | 'degree-step';
  /** Selects linear degree steps or great-circle central angles. Defaults from coordinateSystem. */
  interpolation?: 'planar' | 'linear-degrees' | 'great-circle';
  /** Index of the maximum length or degree step in parameters. Default 0. */
  parameterIndex?: number;
  /** Atomically set to 1 for clamped edges. The caller must clear this before each encoding. */
  approximation?: GPUStatusScalar;
  /** Per-row output vertex count. */
  counts: GraphDataView<'uint32'>;
};

/**
 * Per-row output vertex count of `GPULineSegmentize`: `0` outside lines, `1` for the last row of a
 * line, otherwise `clamp(ceil(length / maximumSegmentLength), 1, maximumPieceCount)`.
 *
 * @internal
 */
export function createSegmentizeCountNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: SegmentizeCountNodeProps
): GPUCommandNode<Parameters> {
  const parameterIndex = props.parameterIndex ?? 0;
  // The parameter index belongs to the caller's packed layout.
  assert(
    Number.isInteger(parameterIndex) &&
      parameterIndex >= 0 &&
      parameterIndex < props.parameters.length
  );
  const greatCircle =
    (props.interpolation ??
      (props.coordinateSystem === 'spherical' ? 'great-circle' : 'planar')) === 'great-circle';
  const lineCount = props.lineOffsets.length - 1;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'count',
    bindings: [
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      {
        name: 'lineOffsets',
        view: props.lineOffsets,
        type: 'u32',
        access: 'read'
      },
      {
        name: 'parameters',
        view: props.parameters,
        type: 'f32',
        access: 'read'
      },
      {name: 'counts', view: props.counts, type: 'u32', access: 'read_write'},
      ...(props.approximation
        ? [
            {
              name: 'approximation',
              view: props.approximation,
              type: 'atomic<u32>' as const,
              access: 'read_write' as const
            }
          ]
        : [])
    ],
    invocationCount: props.positions.length,
    declarations: `const LINE_COUNT: u32 = ${lineCount}u;
const MAXIMUM_PIECES: u32 = ${props.maximumPieceCount}u;
${FIND_LINE_WGSL}
${getLineGeometrySource(greatCircle || props.polygonEdges ? 'spherical' : props.coordinateSystem, props.radius)}

fn getPieceExtent(start: vec2<f32>, end: vec2<f32>) -> f32 {
  ${
    props.pieceCountModel === 'degree-step'
      ? `let rawLongitude = end.x - start.x;
  let longitudeStep = rawLongitude - 360.0 * round(rawLongitude / 360.0);
  return max(abs(longitudeStep), ${greatCircle || props.polygonEdges ? 'geodesicCentralAngle(start, end) * GEODESIC_RADIANS_TO_DEGREES' : 'abs(end.y - start.y)'});`
      : 'return getSegmentLength(start, end);'
  }
}

fn getPieceCount(segmentLength: f32, maximumLength: f32) -> u32 {
  if (!(maximumLength > 0.0) || !(segmentLength > 0.0)) {
    return 1u;
  }
  let requestedPieces = ceil(segmentLength / maximumLength);
  ${
    props.approximation
      ? `if (requestedPieces > f32(MAXIMUM_PIECES)) {
    atomicStore(&approximation[approximationOffset], 1u);
  }`
      : ''
  }
  let pieces = min(requestedPieces, f32(MAXIMUM_PIECES));
  return clamp(u32(pieces), 1u, MAXIMUM_PIECES);
}`,
    body: /* wgsl */ `let line = findLine(index);
  var count = 0u;
  if (line != NO_LINE) {
    let first = lineOffsets[lineOffsetsOffset + line];
    let last = min(lineOffsets[lineOffsetsOffset + line + 1u], ${props.positions.length}u);
    ${
      props.closedLines
        ? `if (last > first + 1u && index + 1u == last &&
        all(bitcast<vec2<u32>>(getPosition(index)) == bitcast<vec2<u32>>(getPosition(first)))) {
      counts[countsOffset + index] = 0u;
      return;
    }`
        : ''
    }
    if (index + 1u >= last ${props.closedLines ? '&& last <= first + 1u' : ''}) {
      count = 1u;
    } else {
      count = getPieceCount(
        getPieceExtent(getPosition(index), getPosition(${props.closedLines ? 'select(index + 1u, first, index + 1u >= last)' : 'index + 1u'})),
        parameters[parametersOffset + ${parameterIndex}u]
      );
    }
  }
  counts[countsOffset + index] = count;`
  });
}

/** Properties for {@link createSegmentizeEmitNode}. @internal */
export type SegmentizeEmitNodeProps = {
  /** Preserve degree-plane pole edges and use spherical angular extents for polygon preparation. */
  polygonEdges?: boolean;
  id: string;
  operation: string;
  positions: GraphDataView<'float32x2'>;
  coordinateSystem: GPULineCoordinateSystem;
  radius: number;
  counts: GraphDataView<'uint32'>;
  starts: GraphDataView<'uint32'>;
  rowMeasures?: GraphDataView<'float32'>;
  rowShifts?: GraphDataView<'float32'>;
  /** Integer prefix from createLinePrefixNode; preferred for wrapped output. */
  rowTurns?: GraphDataView<'sint32'>;
  /** Include the implicit closing edge. Requires lineOffsets. */
  closedLines?: boolean;
  /** Input line boundaries, required only for closedLines. */
  lineOffsets?: GraphDataView<'uint32'>;
  /** Defaults to planar or great-circle according to coordinateSystem. */
  interpolation?: 'planar' | 'linear-degrees' | 'great-circle';
  /** Per-output-vertex integer turns. Implies wrapped output longitudes. */
  outputTurns?: GraphDataView<'sint32'>;
  /** Keep output longitude in [-180, 180]; kept canonical vertices retain their exact bits. */
  wrapOutputLongitudes?: boolean;
  outputPositions: GraphDataView<'float32x2'>;
  outputSourceRows?: GraphDataView<'uint32'>;
  outputMeasures?: GraphDataView<'float32'>;
};

/**
 * Emits each row's own vertex and its interior points (planar lerp or great-circle slerp, evenly
 * spaced) at the row's scanned start. Writes past the vertex capacity are dropped.
 *
 * @internal
 */
export function createSegmentizeEmitNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: SegmentizeEmitNodeProps
): GPUCommandNode<Parameters> {
  // Closed edge lookup needs the owning line; turn output has one element per position.
  assert(!props.closedLines || props.lineOffsets);
  assert(!props.outputTurns || props.outputTurns.length >= props.outputPositions.length);
  const interpolation =
    props.interpolation ?? (props.coordinateSystem === 'spherical' ? 'great-circle' : 'planar');
  const spherical = interpolation === 'great-circle';
  const geographic = interpolation !== 'planar';
  const wrapOutput = Boolean(props.wrapOutputLongitudes || props.outputTurns);
  const shiftView = props.rowTurns ?? props.rowShifts;
  const bindings: WGSLKernelBinding[] = [
    {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
    {name: 'counts', view: props.counts, type: 'u32', access: 'read'},
    {name: 'starts', view: props.starts, type: 'u32', access: 'read'},
    {
      name: 'outputPositions',
      view: props.outputPositions,
      type: 'f32',
      access: 'read_write'
    }
  ];
  if (props.closedLines && props.lineOffsets) {
    bindings.push({name: 'lineOffsets', view: props.lineOffsets, type: 'u32', access: 'read'});
  }
  if (props.outputTurns) {
    bindings.push({
      name: 'outputTurns',
      view: props.outputTurns,
      type: 'i32',
      access: 'read_write'
    });
  }
  if (geographic && shiftView) {
    bindings.push({
      name: 'rowShifts',
      view: shiftView,
      type: props.rowTurns ? 'i32' : 'f32',
      access: 'read'
    });
  }
  if (props.outputMeasures && props.rowMeasures) {
    bindings.push(
      {
        name: 'rowMeasures',
        view: props.rowMeasures,
        type: 'f32',
        access: 'read'
      },
      {
        name: 'outputMeasures',
        view: props.outputMeasures,
        type: 'f32',
        access: 'read_write'
      }
    );
  }
  if (props.outputSourceRows) {
    bindings.push({
      name: 'outputSourceRows',
      view: props.outputSourceRows,
      type: 'u32',
      access: 'read_write'
    });
  }
  const writeMeasures = Boolean(props.outputMeasures && props.rowMeasures);
  const defaultInterpolate = spherical
    ? 'geodesicInterpolate(start, end, angle, fraction)'
    : interpolation === 'linear-degrees'
      ? 'vec2<f32>(start.x + (end.x - start.x - 360.0 * round((end.x - start.x) / 360.0)) * fraction, mix(start.y, end.y, fraction))'
      : 'mix(start, end, fraction)';
  const interpolate = props.polygonEdges
    ? `interpolatePolygonEdge(start, end, fraction)`
    : defaultInterpolate;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'emit',
    bindings,
    // One invocation per output vertex (not per input row), so a segment cut into 1024 pieces does
    // not serialize one lane while its neighbors finish after a single write.
    invocationCount: props.outputPositions.length,
    workgroupSize: BALANCED_SEARCH_WORKGROUP_SIZE,
    guardIndex: false,
    declarations: `const CAPACITY: u32 = ${props.outputPositions.length}u;
${getLineGeometrySource(spherical ? 'spherical' : props.coordinateSystem, props.radius)}
${
  props.polygonEdges
    ? `${LON_LAT_RING_WGSL}
fn interpolatePolygonEdge(start: vec2<f32>, end: vec2<f32>, fraction: f32) -> vec2<f32> {
  if (lonLatIsPoleEdge(start, end)) { return mix(start, end, fraction); }
  ${spherical ? 'let angle = geodesicCentralAngle(start, end);' : ''}
  return ${defaultInterpolate};
}`
    : ''
}
${
  props.closedLines
    ? `const LINE_COUNT: u32 = ${props.lineOffsets!.length - 1}u;
${FIND_LINE_WGSL}`
    : ''
}
${getBalancedSearchSource(props.positions.length)}
fn balancedStart(row: u32) -> u32 { return starts[startsOffset + row]; }
fn balancedCount(row: u32) -> u32 { return counts[countsOffset + row]; }

fn writeVertex(outputRow: u32, vertex: vec2<f32>, measure: f32, sourceRow: u32, prefixTurn: i32) {
  if (outputRow >= CAPACITY) {
    return;
  }
  var longitude = vertex.x;
  var turn = prefixTurn;
  ${
    wrapOutput
      ? `// Preserve canonical input endpoints exactly, including signed zero and both seam signs.
  if (longitude < -180.0 || longitude > 180.0) {
    let localTurn = i32(floor((longitude + 180.0) / 360.0));
    longitude -= 360.0 * f32(localTurn);
    turn += localTurn;
  }`
      : ''
  }
  outputPositions[outputPositionsOffset + 2u * outputRow] = longitude;
  ${props.outputTurns ? 'outputTurns[outputTurnsOffset + outputRow] = turn;' : ''}
  outputPositions[outputPositionsOffset + 2u * outputRow + 1u] = vertex.y;
  ${writeMeasures ? 'outputMeasures[outputMeasuresOffset + outputRow] = measure;' : ''}
  ${props.outputSourceRows ? 'outputSourceRows[outputSourceRowsOffset + outputRow] = sourceRow;' : ''}
}`,
    body: /* wgsl */ `let row = findBalancedOwner(index, workgroupIndex * BALANCED_LANES, localInvocationIndex);
  if (row == BALANCED_NONE) {
    return;
  }
  let count = counts[countsOffset + row];
  let piece = index - starts[startsOffset + row];
  var start = getPosition(row);
  let prefixTurn = ${geographic && shiftView ? (props.rowTurns ? 'rowShifts[rowShiftsOffset + row]' : 'i32(rowShifts[rowShiftsOffset + row] / 360.0)') : '0i'};
  ${!wrapOutput && geographic && shiftView ? (props.rowTurns ? 'start.x += 360.0 * f32(prefixTurn);' : 'start.x = start.x + rowShifts[rowShiftsOffset + row];') : ''}
  let measure = ${writeMeasures ? 'rowMeasures[rowMeasuresOffset + row]' : '0.0'};
  if (piece == 0u) {
    writeVertex(index, start, measure, row, prefixTurn);
    return;
  }
  ${
    props.closedLines
      ? `let line = findLine(row);
  let first = lineOffsets[lineOffsetsOffset + line];
  let last = min(lineOffsets[lineOffsetsOffset + line + 1u], ${props.positions.length}u);
  let next = select(row + 1u, first, row + 1u >= last);`
      : 'let next = row + 1u;'
  }
  let end = getPosition(next);
  let segmentLength = getSegmentLength(start, end);
  ${spherical ? 'let angle = geodesicCentralAngle(start, end);' : ''}
  let fraction = f32(piece) / f32(count);
  writeVertex(index, ${interpolate}, measure + segmentLength * fraction, row, prefixTurn);`
  });
}

/** Emits source rows separately when closed-ring metadata fills the eight-binding budget. @internal */
export function createSegmentizeSourceRowsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    counts: GraphDataView<'uint32'>;
    starts: GraphDataView<'uint32'>;
    outputSourceRows: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'source-rows',
    bindings: [
      {name: 'counts', view: props.counts, type: 'u32', access: 'read'},
      {name: 'starts', view: props.starts, type: 'u32', access: 'read'},
      {name: 'outputSourceRows', view: props.outputSourceRows, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.outputSourceRows.length,
    workgroupSize: BALANCED_SEARCH_WORKGROUP_SIZE,
    guardIndex: false,
    declarations: `${getBalancedSearchSource(props.counts.length)}
fn balancedStart(row: u32) -> u32 { return starts[startsOffset + row]; }
fn balancedCount(row: u32) -> u32 { return counts[countsOffset + row]; }`,
    body: `let row = findBalancedOwner(index, workgroupIndex * BALANCED_LANES, localInvocationIndex);
  if (index < ${props.outputSourceRows.length}u && row != BALANCED_NONE) {
    outputSourceRows[outputSourceRowsOffset + index] = row;
  }`
  });
}

/** Properties for {@link createArcCountNode}. @internal */
export type ArcCountNodeProps = {
  id: string;
  operation: string;
  sources: GraphDataView<'float32x2'>;
  targets: GraphDataView<'float32x2'>;
  parameters: GraphDataView<'float32'>;
  radius: number;
  maximumSegmentCount: number;
  /** Per-pair output vertex count (`segments + 1`). */
  counts: GraphDataView<'uint32'>;
};

const ARC_GEOMETRY_WGSL = /* wgsl */ `
fn getSource(row: u32) -> vec2<f32> {
  return vec2<f32>(sources[sourcesOffset + 2u * row], sources[sourcesOffset + 2u * row + 1u]);
}

fn getTarget(row: u32) -> vec2<f32> {
  return vec2<f32>(targets[targetsOffset + 2u * row], targets[targetsOffset + 2u * row + 1u]);
}
`;

/**
 * Per-pair vertex count of `GPUGreatCircleArcs`: `segments + 1` with
 * `segments = clamp(max(ceil(distance / maximumSegmentLength), minimumSegments), 1, maximumSegmentCount)`.
 *
 * @internal
 */
export function createArcCountNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: ArcCountNodeProps
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'count',
    bindings: [
      {name: 'sources', view: props.sources, type: 'f32', access: 'read'},
      {name: 'targets', view: props.targets, type: 'f32', access: 'read'},
      {
        name: 'parameters',
        view: props.parameters,
        type: 'f32',
        access: 'read'
      },
      {name: 'counts', view: props.counts, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.sources.length,
    declarations: `${GEODESIC_WGSL}
const RADIUS: f32 = ${getWGSLFloatLiteral(props.radius)};
const MAXIMUM_SEGMENTS: u32 = ${props.maximumSegmentCount}u;
${ARC_GEOMETRY_WGSL}`,
    body: /* wgsl */ `let distance = geodesicCentralAngle(getSource(index), getTarget(index)) * RADIUS;
  let maximumLength = parameters[parametersOffset];
  let minimumSegments = parameters[parametersOffset + 1u];
  var segments = 1.0;
  if (maximumLength > 0.0 && distance > 0.0) {
    segments = ceil(min(distance / maximumLength, f32(MAXIMUM_SEGMENTS)));
  }
  if (minimumSegments > segments) {
    segments = min(round(minimumSegments), f32(MAXIMUM_SEGMENTS));
  }
  counts[countsOffset + index] = clamp(u32(segments), 1u, MAXIMUM_SEGMENTS) + 1u;`
  });
}

/** Properties for {@link createArcEmitNode}. @internal */
export type ArcEmitNodeProps = {
  id: string;
  operation: string;
  sources: GraphDataView<'float32x2'>;
  targets: GraphDataView<'float32x2'>;
  radius: number;
  counts: GraphDataView<'uint32'>;
  starts: GraphDataView<'uint32'>;
  outputPositions: GraphDataView<'float32x2'>;
  outputSourceRows?: GraphDataView<'uint32'>;
  outputMeasures?: GraphDataView<'float32'>;
};

/**
 * Emits each pair's great-circle arc, slerped evenly in angle, with longitudes continuous from the
 * source. The first and last vertices are the source and the unwrapped target exactly.
 *
 * @internal
 */
export function createArcEmitNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: ArcEmitNodeProps
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {name: 'sources', view: props.sources, type: 'f32', access: 'read'},
    {name: 'targets', view: props.targets, type: 'f32', access: 'read'},
    {name: 'counts', view: props.counts, type: 'u32', access: 'read'},
    {name: 'starts', view: props.starts, type: 'u32', access: 'read'},
    {
      name: 'outputPositions',
      view: props.outputPositions,
      type: 'f32',
      access: 'read_write'
    }
  ];
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
    variant: 'emit',
    bindings,
    // One invocation per output vertex, so long arcs do not serialize a lane (see balanced-search).
    invocationCount: props.outputPositions.length,
    workgroupSize: BALANCED_SEARCH_WORKGROUP_SIZE,
    guardIndex: false,
    declarations: `${GEODESIC_WGSL}
const RADIUS: f32 = ${getWGSLFloatLiteral(props.radius)};
const CAPACITY: u32 = ${props.outputPositions.length}u;
${ARC_GEOMETRY_WGSL}
${getBalancedSearchSource(props.sources.length)}
fn balancedStart(row: u32) -> u32 { return starts[startsOffset + row]; }
fn balancedCount(row: u32) -> u32 { return counts[countsOffset + row]; }

fn writeVertex(outputRow: u32, vertex: vec2<f32>, measure: f32, sourceRow: u32) {
  if (outputRow >= CAPACITY) {
    return;
  }
  outputPositions[outputPositionsOffset + 2u * outputRow] = vertex.x;
  outputPositions[outputPositionsOffset + 2u * outputRow + 1u] = vertex.y;
  ${props.outputMeasures ? 'outputMeasures[outputMeasuresOffset + outputRow] = measure;' : ''}
  ${props.outputSourceRows ? 'outputSourceRows[outputSourceRowsOffset + outputRow] = sourceRow;' : ''}
}`,
    body: /* wgsl */ `let pair = findBalancedOwner(index, workgroupIndex * BALANCED_LANES, localInvocationIndex);
  if (pair == BALANCED_NONE) {
    return;
  }
  let segments = counts[countsOffset + pair] - 1u;
  let vertex = index - starts[startsOffset + pair];
  let source = getSource(pair);
  if (vertex == 0u) {
    writeVertex(index, source, 0.0, pair);
    return;
  }
  var destination = getTarget(pair);
  destination.x = source.x + geodesicWrapLongitudeDelta(destination.x - source.x);
  let angle = geodesicCentralAngle(source, destination);
  if (vertex == segments) {
    writeVertex(index, destination, angle * RADIUS, pair);
    return;
  }
  let fraction = f32(vertex) / f32(segments);
  writeVertex(
    index,
    geodesicInterpolate(source, destination, angle, fraction),
    angle * RADIUS * fraction,
    pair
  );`
  });
}

/** Properties for {@link createLineOffsetsPublishNode}. @internal */
export type LineOffsetsPublishNodeProps = {
  id: string;
  operation: string;
  /**
   * Input line row offsets (`lineCount + 1`). Omit when every line is one input row (pairs), so
   * line `p` starts at row `p`.
   */
  lineOffsets?: GraphDataView<'uint32'>;
  lineCount: number;
  rowCount: number;
  counts: GraphDataView<'uint32'>;
  starts: GraphDataView<'uint32'>;
  outputLineOffsets: GraphDataView<'uint32'>;
  capacity: number;
  count: GraphDataView<'uint32'>;
  overflow: GraphDataView<'uint32'>;
  requiredCount?: GraphDataView<'uint32'>;
  lineCountOutput?: GraphDataView<'uint32'>;
};

/**
 * Writes clamped output line offsets from the scanned row starts, plus the clamped vertex count,
 * the overflow flag, and the optional unclamped total and line count. Every word is rewritten.
 *
 * @internal
 */
export function createLineOffsetsPublishNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: LineOffsetsPublishNodeProps
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [];
  if (props.lineOffsets) {
    bindings.push({
      name: 'lineOffsets',
      view: props.lineOffsets,
      type: 'u32',
      access: 'read'
    });
  }
  bindings.push(
    {name: 'counts', view: props.counts, type: 'u32', access: 'read'},
    {name: 'starts', view: props.starts, type: 'u32', access: 'read'},
    {
      name: 'outputLineOffsets',
      view: props.outputLineOffsets,
      type: 'u32',
      access: 'read_write'
    },
    {name: 'countOut', view: props.count, type: 'u32', access: 'read_write'},
    {
      name: 'overflowOut',
      view: props.overflow,
      type: 'u32',
      access: 'read_write'
    }
  );
  if (props.requiredCount) {
    bindings.push({
      name: 'totalOut',
      view: props.requiredCount,
      type: 'u32',
      access: 'read_write'
    });
  }
  if (props.lineCountOutput) {
    bindings.push({
      name: 'lineCountOut',
      view: props.lineCountOutput,
      type: 'u32',
      access: 'read_write'
    });
  }
  const lineRow = props.lineOffsets
    ? 'min(lineOffsets[lineOffsetsOffset + index], ROW_COUNT)'
    : 'min(index, ROW_COUNT)';
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'publish',
    bindings,
    invocationCount: props.lineCount + 1,
    declarations: `const ROW_COUNT: u32 = ${props.rowCount}u;
const CAPACITY: u32 = ${props.capacity}u;
const LINE_COUNT: u32 = ${props.lineCount}u;`,
    body: /* wgsl */ `let total = starts[startsOffset + ROW_COUNT - 1u] + counts[countsOffset + ROW_COUNT - 1u];
  let row = ${lineRow};
  var start = total;
  if (row < ROW_COUNT) {
    start = starts[startsOffset + row];
  }
  outputLineOffsets[outputLineOffsetsOffset + index] = min(start, CAPACITY);
  if (index == 0u) {
    countOut[countOutOffset] = min(total, CAPACITY);
    overflowOut[overflowOutOffset] = select(0u, 1u, total > CAPACITY);
    ${props.requiredCount ? 'totalOut[totalOutOffset] = total;' : ''}
    ${props.lineCountOutput ? 'lineCountOut[lineCountOutOffset] = LINE_COUNT;' : ''}
  }`
  });
}

/**
 * Writes the identity source-line-index column (`sourceLineIndices[p] = p`) for contributors that emit one output
 * line per input line.
 *
 * @internal
 */
export function createIdentitySourceLineIndicesNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    sourceLineIndices: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'source-line-indexs',
    bindings: [
      {
        name: 'sourceLineIndices',
        view: props.sourceLineIndices,
        type: 'u32',
        access: 'read_write'
      }
    ],
    invocationCount: props.sourceLineIndices.length,
    body: 'sourceLineIndices[sourceLineIndicesOffset + index] = index;'
  });
}
