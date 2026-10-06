// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {GEODESIC_WGSL} from '../geometry-measures/geodesic-wgsl';
import {FIND_PATH_WGSL} from '../line-segmentize/line-segmentize-kernels';

/** Static description shared by the walk kernels. @internal */
export type LineDensityGridConstants = {
  positions: GraphDataView<'float32x2'>;
  pathOffsets: GraphDataView<'uint32'>;
  columns: number;
  rows: number;
  coordinateSystem: 'planar' | 'spherical';
  radius: number;
  capacity: number;
};

/**
 * WGSL `walkSegment(row, slot) -> u32`: clips one segment to the grid (Liang-Barsky in grid
 * units), then walks the cells it crosses with an Amanatides-Woo traversal that uses integer step
 * counts, so pieces always telescope from the clipped start to the clipped end and their lengths
 * sum to the clipped length. Returns the piece count; with `emit` writes `keys[slot + k]` (cell
 * index) and `contributions[slot + k]` (piece length) for `slot + k < CAPACITY`.
 *
 * Counting and emitting share this text, so counts always match emitted pieces.
 *
 * @internal
 */
export function getWalkSegmentSource(constants: LineDensityGridConstants, emit: boolean): string {
  const spherical = constants.coordinateSystem === 'spherical';
  return /* wgsl */ `
const PATH_COUNT: u32 = ${constants.pathOffsets.length - 1}u;
const ROW_COUNT: u32 = ${constants.positions.length}u;
const COLUMNS: i32 = ${constants.columns};
const ROWS: i32 = ${constants.rows};
const CAPACITY: u32 = ${constants.capacity}u;
const RADIUS: f32 = ${getWGSLFloatLiteral(constants.radius)};
const INFINITE: f32 = 3.0e38;
${spherical ? GEODESIC_WGSL : ''}
${FIND_PATH_WGSL}

fn getPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}

fn getPieceLength(a: vec2<f32>, b: vec2<f32>, t0: f32, t1: f32) -> f32 {
  ${
    spherical
      ? 'return geodesicCentralAngle(mix(a, b, t0), mix(a, b, t1)) * RADIUS;'
      : 'return (t1 - t0) * length(b - a);'
  }
}

fn walkSegment(row: u32, slot: u32) -> u32 {
  let path = findPath(row);
  if (path == NO_PATH) {
    return 0u;
  }
  if (row + 1u >= min(pathOffsets[pathOffsetsOffset + path + 1u], ROW_COUNT)) {
    return 0u;
  }
  let origin = vec2<f32>(parameters[parametersOffset], parameters[parametersOffset + 1u]);
  let cellSize = vec2<f32>(parameters[parametersOffset + 2u], parameters[parametersOffset + 3u]);
  let a = getPosition(row);
  let b = getPosition(row + 1u);
  let ua = (a - origin) / cellSize;
  let delta = (b - origin) / cellSize - ua;
  // Liang-Barsky against [0, COLUMNS] x [0, ROWS].
  var t0 = 0.0;
  var t1 = 1.0;
  var p = array<f32, 4>(-delta.x, delta.x, -delta.y, delta.y);
  var q = array<f32, 4>(ua.x, f32(COLUMNS) - ua.x, ua.y, f32(ROWS) - ua.y);
  for (var side = 0u; side < 4u; side++) {
    if (p[side] == 0.0) {
      if (q[side] < 0.0) {
        return 0u;
      }
    } else {
      let r = q[side] / p[side];
      if (p[side] < 0.0) {
        if (r > t1) {
          return 0u;
        }
        t0 = max(t0, r);
      } else {
        if (r < t0) {
          return 0u;
        }
        t1 = min(t1, r);
      }
    }
  }
  if (t1 <= t0) {
    return 0u;
  }
  let start = ua + delta * t0;
  let end = ua + delta * t1;
  var cx = clamp(i32(floor(start.x)), 0, COLUMNS - 1);
  var cy = clamp(i32(floor(start.y)), 0, ROWS - 1);
  let endX = clamp(i32(floor(end.x)), 0, COLUMNS - 1);
  let endY = clamp(i32(floor(end.y)), 0, ROWS - 1);
  var stepsX = u32(abs(endX - cx));
  var stepsY = u32(abs(endY - cy));
  let pieceCount = stepsX + stepsY + 1u;
  let stepX = select(-1, 1, delta.x > 0.0);
  let stepY = select(-1, 1, delta.y > 0.0);
  var tMaxX = INFINITE;
  var tMaxY = INFINITE;
  if (stepsX > 0u) {
    tMaxX = t0 + (select(f32(cx), f32(cx + 1), delta.x > 0.0) - start.x) / delta.x;
  }
  if (stepsY > 0u) {
    tMaxY = t0 + (select(f32(cy), f32(cy + 1), delta.y > 0.0) - start.y) / delta.y;
  }
  let tDeltaX = select(INFINITE, 1.0 / abs(delta.x), delta.x != 0.0);
  let tDeltaY = select(INFINITE, 1.0 / abs(delta.y), delta.y != 0.0);
  var tCurrent = t0;
  for (var piece = 0u; piece < pieceCount; piece++) {
    let moveX = stepsX > 0u && (stepsY == 0u || tMaxX <= tMaxY);
    var tNext = t1;
    if (piece + 1u < pieceCount) {
      tNext = clamp(select(tMaxY, tMaxX, moveX), tCurrent, t1);
    }
    ${
      emit
        ? `if (slot + piece < CAPACITY) {
      keys[keysOffset + slot + piece] = u32(cy * COLUMNS + cx);
      contributions[contributionsOffset + slot + piece] = getPieceLength(a, b, tCurrent, tNext);
    }`
        : ''
    }
    if (moveX) {
      cx += stepX;
      stepsX -= 1u;
      tMaxX += tDeltaX;
    } else {
      cy += stepY;
      stepsY -= 1u;
      tMaxY += tDeltaY;
    }
    tCurrent = tNext;
  }
  return pieceCount;
}
`;
}

/** Builds the per-segment piece count node. @internal */
export function createWalkCountNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    constants: LineDensityGridConstants;
    parameters: GraphDataView<'float32'>;
    counts: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'count',
    bindings: [
      {name: 'positions', view: props.constants.positions, type: 'f32', access: 'read'},
      {name: 'pathOffsets', view: props.constants.pathOffsets, type: 'u32', access: 'read'},
      {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
      {name: 'counts', view: props.counts, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.constants.positions.length,
    declarations: getWalkSegmentSource(props.constants, false),
    body: 'counts[countsOffset + index] = walkSegment(index, 0u);'
  });
}

/** Builds the per-segment piece emit node. @internal */
export function createWalkEmitNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    constants: LineDensityGridConstants;
    parameters: GraphDataView<'float32'>;
    starts: GraphDataView<'uint32'>;
    keys: GraphDataView<'uint32'>;
    contributions: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'emit',
    bindings: [
      {name: 'positions', view: props.constants.positions, type: 'f32', access: 'read'},
      {name: 'pathOffsets', view: props.constants.pathOffsets, type: 'u32', access: 'read'},
      {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
      {name: 'starts', view: props.starts, type: 'u32', access: 'read'},
      {name: 'keys', view: props.keys, type: 'u32', access: 'read_write'},
      {name: 'contributions', view: props.contributions, type: 'f32', access: 'read_write'}
    ],
    invocationCount: props.constants.positions.length,
    declarations: getWalkSegmentSource(props.constants, true),
    body: 'walkSegment(index, starts[startsOffset + index]);'
  });
}

/** Counts records per cell with integer atomics (order independent). @internal */
export function createCellCountNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    cellCount: number;
    keys: GraphDataView<'uint32'>;
    cellCounts: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'cell-counts',
    bindings: [
      {name: 'keys', view: props.keys, type: 'u32', access: 'read'},
      {name: 'cellCounts', view: props.cellCounts, type: 'atomic<u32>', access: 'read_write'}
    ],
    invocationCount: props.keys.length,
    declarations: `const CELL_COUNT: u32 = ${props.cellCount}u;`,
    body: `let key = keys[keysOffset + index];
  if (key < CELL_COUNT) {
    atomicAdd(&cellCounts[cellCountsOffset + key], 1u);
  }`
  });
}

/** Writes the unclamped record total and the overflow flag. @internal */
export function createRecordPublishNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    capacity: number;
    starts: GraphDataView<'uint32'>;
    counts: GraphDataView<'uint32'>;
    overflow: GraphDataView<'uint32'>;
    totalRecords?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {name: 'starts', view: props.starts, type: 'u32', access: 'read'},
    {name: 'counts', view: props.counts, type: 'u32', access: 'read'},
    {name: 'overflowOut', view: props.overflow, type: 'u32', access: 'read_write'}
  ];
  if (props.totalRecords) {
    bindings.push({name: 'totalOut', view: props.totalRecords, type: 'u32', access: 'read_write'});
  }
  const last = props.starts.length - 1;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'publish',
    bindings,
    invocationCount: 1,
    body: `let total = starts[startsOffset + ${last}u] + counts[countsOffset + ${last}u];
  overflowOut[overflowOutOffset] = select(0u, 1u, total > ${props.capacity}u);
  ${props.totalRecords ? 'totalOut[totalOutOffset] = total;' : ''}`
  });
}

/** Divides cell lengths by cell areas. @internal */
export function createDensityNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    columns: number;
    rows: number;
    coordinateSystem: 'planar' | 'spherical';
    radius: number;
    parameters: GraphDataView<'float32'>;
    lengths: GraphDataView<'float32'>;
    densities: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'density',
    bindings: [
      {name: 'lengths', view: props.lengths, type: 'f32', access: 'read'},
      {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
      {name: 'densities', view: props.densities, type: 'f32', access: 'read_write'}
    ],
    invocationCount: props.columns * props.rows,
    declarations: `const COLUMNS: u32 = ${props.columns}u;
const RADIUS: f32 = ${getWGSLFloatLiteral(props.radius)};
const DEGREES_TO_RADIANS: f32 = 0.017453292519943295;`,
    body: `let row = index / COLUMNS;
  let width = parameters[parametersOffset + 2u];
  let height = parameters[parametersOffset + 3u];
  ${
    props.coordinateSystem === 'spherical'
      ? `let south = parameters[parametersOffset + 1u] + f32(row) * height;
  // Exact area of a longitude/latitude cell on the sphere.
  let area = RADIUS * RADIUS * width * DEGREES_TO_RADIANS *
    (sin((south + height) * DEGREES_TO_RADIANS) - sin(south * DEGREES_TO_RADIANS));`
      : 'let area = width * height;'
  }
  densities[densitiesOffset + index] = lengths[lengthsOffset + index] / area;`
  });
}
