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
import {NEAREST_BOX_LOWER_ANGLE_WGSL} from '../spatial-join/nearest-wgsl';
import {FIND_LINE_WGSL} from '../line-segmentize/line-segmentize-kernels';

/** Words per point in the internal `GPULinearReferencing` result column. @internal */
export const LINEAR_REFERENCING_RESULT_STRIDE = 9;

/** Floats per row of the internal spherical segment table. @internal */
export const SPHERICAL_SEGMENT_TABLE_STRIDE = 16;

/** Words per event in the internal `GPULineLocate` result column. @internal */
export const LINE_LOCATE_RESULT_STRIDE = 6;

/** `GPULineLocate` per-event status codes. */
export const GPU_LINE_LOCATE_STATUS = {
  /** The measure lies on the line. */
  ok: 0,
  /** The measure was outside `[0, length]` and was clamped to the nearest end. */
  clamped: 1,
  /** The line index is out of range or the line is empty; outputs are NaN. */
  invalid: 2
} as const;

const POSITION_WGSL = /* wgsl */ `
const NAN_BITS: u32 = 0x7fc00000u;
fn getPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}
`;

/**
 * Writes one segment per vertex row: `[row, row + 1]` when both rows belong to the same line,
 * otherwise a NaN segment, which the nearest-feature join gives empty bounds and never matches.
 *
 * @internal
 */
export function createLineSegmentsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    positions: GraphDataView<'float32x2'>;
    lineOffsets: GraphDataView<'uint32'>;
    segmentStarts: GraphDataView<'float32x2'>;
    segmentEnds: GraphDataView<'float32x2'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'line-segments',
    bindings: [
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      {name: 'lineOffsets', view: props.lineOffsets, type: 'u32', access: 'read'},
      {name: 'segmentStarts', view: props.segmentStarts, type: 'f32', access: 'read_write'},
      {name: 'segmentEnds', view: props.segmentEnds, type: 'f32', access: 'read_write'}
    ],
    invocationCount: props.positions.length,
    declarations: `const LINE_COUNT: u32 = ${props.lineOffsets.length - 1}u;
${FIND_LINE_WGSL}
${POSITION_WGSL}`,
    body: /* wgsl */ `let line = findLine(index);
  // A runtime operand keeps the NaN out of const-expression evaluation, where it is an error.
  var start = vec2<f32>(bitcast<f32>(0x7fc00000u | (index & 0u)));
  var end = start;
  if (line != NO_LINE && index + 1u < lineOffsets[lineOffsetsOffset + line + 1u]) {
    start = getPosition(index);
    end = getPosition(index + 1u);
  }
  segmentStarts[segmentStartsOffset + 2u * index] = start.x;
  segmentStarts[segmentStartsOffset + 2u * index + 1u] = start.y;
  segmentEnds[segmentEndsOffset + 2u * index] = end.x;
  segmentEnds[segmentEndsOffset + 2u * index + 1u] = end.y;`
  });
}

/**
 * Projects each point onto its nearest segment and writes the packed result
 * `[lineIndex, segmentRow, fraction, footX, footY, distance, measure, side, normalizedMeasure]`
 * (indices and side as u32/i32 bits).
 *
 * @internal
 */
export function createPointProjectionNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    points: GraphDataView<'float32x2'>;
    positions: GraphDataView<'float32x2'>;
    lineOffsets: GraphDataView<'uint32'>;
    nearestSegments: GraphDataView<'uint32'>;
    vertexMeasures: GraphDataView<'float32'>;
    results: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'project',
    bindings: [
      {name: 'points', view: props.points, type: 'f32', access: 'read'},
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      {name: 'lineOffsets', view: props.lineOffsets, type: 'u32', access: 'read'},
      {name: 'nearestSegments', view: props.nearestSegments, type: 'u32', access: 'read'},
      {name: 'vertexMeasures', view: props.vertexMeasures, type: 'f32', access: 'read'},
      {name: 'results', view: props.results, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.points.length,
    declarations: `const LINE_COUNT: u32 = ${props.lineOffsets.length - 1}u;
const ROW_COUNT: u32 = ${props.positions.length}u;
const STRIDE: u32 = ${LINEAR_REFERENCING_RESULT_STRIDE}u;
${FIND_LINE_WGSL}
${POSITION_WGSL}`,
    body: /* wgsl */ `let base = resultsOffset + index * STRIDE;
  let segment = nearestSegments[nearestSegmentsOffset + index];
  // Unmatched points carry the join's 0xffffffff sentinel; it also fails this bound.
  if (segment >= ROW_COUNT - 1u) {
    results[base] = NO_LINE;
    results[base + 1u] = NO_LINE;
    results[base + 2u] = NAN_BITS;
    results[base + 3u] = NAN_BITS;
    results[base + 4u] = NAN_BITS;
    results[base + 5u] = bitcast<u32>(-1.0);
    results[base + 6u] = NAN_BITS;
    results[base + 7u] = 0u;
    results[base + 8u] = NAN_BITS;
    return;
  }
  let point = vec2<f32>(points[pointsOffset + 2u * index], points[pointsOffset + 2u * index + 1u]);
  let start = getPosition(segment);
  let direction = getPosition(segment + 1u) - start;
  let relative = point - start;
  let lengthSquared = dot(direction, direction);
  var fraction = 0.0;
  if (lengthSquared > 0.0) {
    fraction = clamp(dot(relative, direction) / lengthSquared, 0.0, 1.0);
  }
  let footOffset = direction * fraction;
  let distance = length(relative - footOffset);
  let cross = direction.x * relative.y - direction.y * relative.x;
  var side = 0i;
  if (distance > 0.0 && cross > 0.0) {
    side = 1i;
  } else if (distance > 0.0 && cross < 0.0) {
    side = -1i;
  }
  let line = findLine(segment);
  let lineLength = vertexMeasures[vertexMeasuresOffset + lineOffsets[lineOffsetsOffset + line + 1u] - 1u];
  let measure = vertexMeasures[vertexMeasuresOffset + segment] + fraction * sqrt(lengthSquared);
  results[base] = line;
  results[base + 1u] = segment - lineOffsets[lineOffsetsOffset + line];
  results[base + 2u] = bitcast<u32>(fraction);
  results[base + 3u] = bitcast<u32>(start.x + footOffset.x);
  results[base + 4u] = bitcast<u32>(start.y + footOffset.y);
  results[base + 5u] = bitcast<u32>(distance);
  results[base + 6u] = bitcast<u32>(measure);
  results[base + 7u] = bitcast<u32>(side);
  // shapely line_locate_point(normalized=True); zero-length lines report 0.
  results[base + 8u] = bitcast<u32>(select(0.0, measure / lineLength, lineLength > 0.0));`
  });
}

/**
 * Builds the spherical segment table: per vertex row 12 floats
 * `[A(3), tangent(3), normal(3), arcAngle, kind, 0, 0]` where `A` is the unit vector of the row,
 * `tangent` the unit direction of the great-circle arc to the next row at `A`, `normal = A x tangent`
 * (pointing left of the direction of travel), and `kind` is 0 for rows that start no segment, 1 for
 * arcs and 2 for zero-length segments. The tangent comes from the degree-difference bearing, so
 * meter-scale arcs stay well conditioned. Also zeroes the status words (the arc BVH packing node
 * then publishes its leaf overflow; there is no candidate list, so `candidateCount` stays 0).
 *
 * @internal
 */
export function createSphericalSegmentTableNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    positions: GraphDataView<'float32x2'>;
    lineOffsets: GraphDataView<'uint32'>;
    table: GraphDataView<'float32'>;
    overflow: GraphDataView<'uint32'>;
    candidateCount?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
    {name: 'lineOffsets', view: props.lineOffsets, type: 'u32', access: 'read'},
    {name: 'table', view: props.table, type: 'f32', access: 'read_write'},
    {name: 'overflow', view: props.overflow, type: 'u32', access: 'read_write'}
  ];
  if (props.candidateCount) {
    bindings.push({
      name: 'candidateCount',
      view: props.candidateCount,
      type: 'u32',
      access: 'read_write'
    });
  }
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'spherical-table',
    bindings,
    invocationCount: props.positions.length,
    declarations: `const LINE_COUNT: u32 = ${props.lineOffsets.length - 1}u;
const TABLE_STRIDE: u32 = ${SPHERICAL_SEGMENT_TABLE_STRIDE}u;
${GEODESIC_WGSL}
${FIND_LINE_WGSL}
${POSITION_WGSL}`,
    body: /* wgsl */ `if (index == 0u) {
    overflow[overflowOffset] = 0u;
    ${props.candidateCount ? 'candidateCount[candidateCountOffset] = 0u;' : ''}
  }
  let base = tableOffset + index * TABLE_STRIDE;
  for (var word = 0u; word < TABLE_STRIDE; word++) {
    table[base + word] = 0.0;
  }
  // Bounding cap half-angle of an invalid row: negative, so projection scans skip it.
  table[base + 15u] = -1.0;
  let line = findLine(index);
  if (line == NO_LINE || index + 1u >= lineOffsets[lineOffsetsOffset + line + 1u]) {
    return;
  }
  let start = getPosition(index);
  let end = getPosition(index + 1u);
  let angle = geodesicCentralAngle(start, end);
  table[base + 11u] = 2.0;
  // A degenerate segment is a point: its cap is the start vector with half-angle 0.
  let startVector = geodesicUnitVector(start);
  table[base + 12u] = startVector.x;
  table[base + 13u] = startVector.y;
  table[base + 14u] = startVector.z;
  table[base + 15u] = 0.0;
  if (!(angle > 0.0)) {
    return;
  }
  let bearing = geodesicInitialBearingDegrees(start, end) * GEODESIC_DEGREES_TO_RADIANS;
  let lambda = start.x * GEODESIC_DEGREES_TO_RADIANS;
  let phi = start.y * GEODESIC_DEGREES_TO_RADIANS;
  let sinPhi = sin(phi);
  let cosPhi = geodesicCosLatitude(start.y);
  let up = vec3<f32>(cosPhi * cos(lambda), cosPhi * sin(lambda), sinPhi);
  let east = vec3<f32>(-sin(lambda), cos(lambda), 0.0);
  let north = vec3<f32>(-sinPhi * cos(lambda), -sinPhi * sin(lambda), cosPhi);
  let tangent = cos(bearing) * north + sin(bearing) * east;
  let normal = cross(up, tangent);
  table[base] = up.x;
  table[base + 1u] = up.y;
  table[base + 2u] = up.z;
  table[base + 3u] = tangent.x;
  table[base + 4u] = tangent.y;
  table[base + 5u] = tangent.z;
  table[base + 6u] = normal.x;
  table[base + 7u] = normal.y;
  table[base + 8u] = normal.z;
  table[base + 9u] = angle;
  table[base + 11u] = 1.0;
  // Bounding cap: the arc midpoint and half the arc angle.
  let center = cos(0.5 * angle) * up + sin(0.5 * angle) * tangent;
  table[base + 12u] = center.x;
  table[base + 13u] = center.y;
  table[base + 14u] = center.z;
  table[base + 15u] = 0.5 * angle;`
  });
}

/** Words per node of the packed spherical arc BVH: minimum xy, maximum xy (f32 bits), leaf row. @internal */
export const SPHERICAL_NODE_WORDS = 5;

/**
 * Writes a conservative longitude/latitude box of every great-circle arc of the segment table, for
 * the arc BVH. The latitude range covers the endpoints and the great circle's highest and lowest
 * point when it lies on the arc (the arc is `cos(s) up + sin(s) tangent`, `s` in `[0, arcAngle]`,
 * so `z` peaks at `s = atan2(tangent.z, up.z)` with value `sqrt(up.z^2 + tangent.z^2)`). The
 * longitude range is the endpoints' when the arc stays on one side of the antimeridian and away
 * from the poles (longitude is then monotone along it); an arc whose short longitude path crosses
 * +-180, spans 179 degrees or more of longitude, or comes within 0.1 degrees of a pole gets every
 * longitude. Boxes are padded by 1e-4 degrees for f32 rounding. Rows that start no segment get an
 * empty box (minimum above maximum), which the BVH and the traversal ignore. @internal
 */
export function createSphericalArcBoundsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    positions: GraphDataView<'float32x2'>;
    table: GraphDataView<'float32'>;
    arcMinima: GraphDataView<'float32x2'>;
    arcMaxima: GraphDataView<'float32x2'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'spherical-arc-bounds',
    bindings: [
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      {name: 'table', view: props.table, type: 'f32', access: 'read'},
      {name: 'arcMinima', view: props.arcMinima, type: 'f32', access: 'read_write'},
      {name: 'arcMaxima', view: props.arcMaxima, type: 'f32', access: 'read_write'}
    ],
    invocationCount: props.positions.length,
    declarations: `const TABLE_STRIDE: u32 = ${SPHERICAL_SEGMENT_TABLE_STRIDE}u;
const BOX_PADDING: f32 = 1.0e-4;
const FLOAT32_MAXIMUM: f32 = 3.402823e38;
const PI: f32 = 3.14159265358979;
const DEGREES_PER_RADIAN: f32 = 57.2957795130823;
${POSITION_WGSL}`,
    body: /* wgsl */ `let base = tableOffset + index * TABLE_STRIDE;
  let kind = u32(table[base + 11u]);
  var minimum = vec2<f32>(FLOAT32_MAXIMUM);
  var maximum = vec2<f32>(-FLOAT32_MAXIMUM);
  if (kind != 0u) {
    let start = getPosition(index);
    var end = start;
    var isFullLongitude = false;
    if (kind == 1u) {
      end = getPosition(index + 1u);
    }
    minimum = min(start, end);
    maximum = max(start, end);
    if (kind == 1u) {
      let upZ = table[base + 2u];
      let tangentZ = table[base + 5u];
      let arcAngle = table[base + 9u];
      let reach = asin(min(sqrt(upZ * upZ + tangentZ * tangentZ), 1.0)) * DEGREES_PER_RADIAN;
      let peak = atan2(tangentZ, upZ);
      let trough = select(peak + PI, peak - PI, peak > 0.0);
      if (peak >= 0.0 && peak <= arcAngle) {
        maximum.y = max(maximum.y, reach);
      }
      if (trough >= 0.0 && trough <= arcAngle) {
        minimum.y = min(minimum.y, -reach);
      }
      isFullLongitude = abs(end.x - start.x) >= 179.0;
    }
    minimum = minimum - vec2<f32>(BOX_PADDING);
    maximum = maximum + vec2<f32>(BOX_PADDING);
    if (isFullLongitude || maximum.y >= 89.9 || minimum.y <= -89.9) {
      minimum.x = -180.0;
      maximum.x = 180.0;
    }
    minimum.y = max(minimum.y, -90.0);
    maximum.y = min(maximum.y, 90.0);
  }
  arcMinima[arcMinimaOffset + index * 2u] = minimum.x;
  arcMinima[arcMinimaOffset + index * 2u + 1u] = minimum.y;
  arcMaxima[arcMaximaOffset + index * 2u] = maximum.x;
  arcMaxima[arcMaximaOffset + index * 2u + 1u] = maximum.y;`
  });
}

/**
 * Packs the arc BVH into one table of {@link SPHERICAL_NODE_WORDS} words per node (box bits and the
 * leaf's segment row, `0xffffffff` for internal nodes), so the projection kernel binds one buffer,
 * and publishes the BVH leaf overflow (`leafCapacity` below the row count) to `overflow`. @internal
 */
export function createSphericalNodeDataNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    /** BVH node minima, two components (longitude, latitude) per node. */
    nodeMinima: GraphDataView;
    /** BVH node maxima, two components per node. */
    nodeMaxima: GraphDataView;
    leafIds: GraphDataView<'uint32'>;
    bvhOverflow: GraphDataView<'uint32'>;
    internalNodeCount: number;
    nodeCount: number;
    nodeData: GraphDataView<'uint32'>;
    overflow: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'spherical-node-data',
    bindings: [
      {name: 'nodeMinima', view: props.nodeMinima, type: 'f32', access: 'read'},
      {name: 'nodeMaxima', view: props.nodeMaxima, type: 'f32', access: 'read'},
      {name: 'leafIds', view: props.leafIds, type: 'u32', access: 'read'},
      {name: 'bvhOverflow', view: props.bvhOverflow, type: 'u32', access: 'read'},
      {name: 'nodeData', view: props.nodeData, type: 'u32', access: 'read_write'},
      {name: 'overflow', view: props.overflow, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.nodeCount,
    declarations: `const INTERNAL_NODE_COUNT: u32 = ${props.internalNodeCount}u;
const NODE_WORDS: u32 = ${SPHERICAL_NODE_WORDS}u;`,
    body: `if (index == 0u) {
    overflow[overflowOffset] = select(0u, 1u, bvhOverflow[bvhOverflowOffset] != 0u);
  }
  let base = nodeDataOffset + index * NODE_WORDS;
  nodeData[base] = bitcast<u32>(nodeMinima[nodeMinimaOffset + index * 2u]);
  nodeData[base + 1u] = bitcast<u32>(nodeMinima[nodeMinimaOffset + index * 2u + 1u]);
  nodeData[base + 2u] = bitcast<u32>(nodeMaxima[nodeMaximaOffset + index * 2u]);
  nodeData[base + 3u] = bitcast<u32>(nodeMaxima[nodeMaximaOffset + index * 2u + 1u]);
  nodeData[base + 4u] = select(0xffffffffu, leafIds[leafIdsOffset + index - INTERNAL_NODE_COUNT], index >= INTERNAL_NODE_COUNT);`
  });
}

/**
 * Spherical projection: each point walks the arc BVH (packed by {@link createSphericalNodeDataNode})
 * with an explicit stack, nearer child first, pruning every node whose conservative great-circle
 * lower bound (`boxLowerAngle` of its longitude/latitude box) exceeds the best angle so far or the
 * search radius. At the leaves it evaluates the exact arc distance and keeps the smallest central
 * angle, ties to the lowest segment row (so the earliest segment and line) whatever the traversal
 * order. It keeps the match when it lies within `radius` meters and writes the same packed result
 * as {@link createPointProjectionNode} with distances and measures in sphere-radius units.
 *
 * @internal
 */
export function createSphericalProjectionNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    points: GraphDataView<'float32x2'>;
    positions: GraphDataView<'float32x2'>;
    lineOffsets: GraphDataView<'uint32'>;
    table: GraphDataView<'float32'>;
    vertexMeasures: GraphDataView<'float32'>;
    radius: GraphDataView<'float32'>;
    sphereRadius: number;
    /** Packed arc BVH from {@link createSphericalNodeDataNode}. */
    nodeData: GraphDataView<'uint32'>;
    /** Internal node count of the BVH; leaves follow at this index. */
    internalNodeCount: number;
    /** Levels of the BVH, which bound the traversal stack. */
    levelCount: number;
    results: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'spherical-project',
    bindings: [
      {name: 'points', view: props.points, type: 'f32', access: 'read'},
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      {name: 'lineOffsets', view: props.lineOffsets, type: 'u32', access: 'read'},
      {name: 'table', view: props.table, type: 'f32', access: 'read'},
      {name: 'vertexMeasures', view: props.vertexMeasures, type: 'f32', access: 'read'},
      {name: 'radius', view: props.radius, type: 'f32', access: 'read'},
      {name: 'nodeData', view: props.nodeData, type: 'u32', access: 'read'},
      {name: 'results', view: props.results, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.points.length,
    declarations: `const LINE_COUNT: u32 = ${props.lineOffsets.length - 1}u;
const ROW_COUNT: u32 = ${props.positions.length}u;
const STRIDE: u32 = ${LINEAR_REFERENCING_RESULT_STRIDE}u;
const TABLE_STRIDE: u32 = ${SPHERICAL_SEGMENT_TABLE_STRIDE}u;
const SPHERE_RADIUS: f32 = ${getWGSLFloatLiteral(props.sphereRadius)};
const INTERNAL_NODE_COUNT: u32 = ${props.internalNodeCount}u;
const NODE_WORDS: u32 = ${SPHERICAL_NODE_WORDS}u;
const STACK_SIZE: u32 = ${2 * props.levelCount}u;
// Relative and absolute (radians, about 0.6 m) slack on the pruning bound, so f32 rounding in the
// box bound and in the arc evaluation never drops the best or a tied arc.
const BOUND_SLACK: f32 = 1.000001;
const BOUND_ABSOLUTE_SLACK: f32 = 1.0e-7;
const FLOAT32_MAXIMUM: f32 = 3.402823e38;
${GEODESIC_WGSL}
${NEAREST_BOX_LOWER_ANGLE_WGSL}
${FIND_LINE_WGSL}
${POSITION_WGSL}

// Lower bound of the central angle from the point to any arc under the node, or FLOAT32_MAXIMUM
// for an empty node.
fn nodeLowerAngle(node: u32, point: vec2<f32>) -> f32 {
  let base = nodeDataOffset + node * NODE_WORDS;
  let nodeMinimum = vec2<f32>(bitcast<f32>(nodeData[base]), bitcast<f32>(nodeData[base + 1u]));
  let nodeMaximum = vec2<f32>(bitcast<f32>(nodeData[base + 2u]), bitcast<f32>(nodeData[base + 3u]));
  if (nodeMinimum.x > nodeMaximum.x || nodeMinimum.y > nodeMaximum.y) { return FLOAT32_MAXIMUM; }
  return boxLowerAngle(nodeMinimum, nodeMaximum, point, point);
}

fn getTableVector(base: u32, word: u32) -> vec3<f32> {
  return vec3<f32>(table[base + word], table[base + word + 1u], table[base + word + 2u]);
}

struct SegmentHit {
  angle: f32,
  along: f32,
  /** 0 interior of the arc, 1 start endpoint, 2 end endpoint. */
  endpoint: u32,
  normalDot: f32
}

// Angle from the point to segment row of the table (a valid row: kind 1 or 2).
fn evaluateSegment(row: u32, point: vec2<f32>, pointVector: vec3<f32>) -> SegmentHit {
  let tableBase = tableOffset + row * TABLE_STRIDE;
  let kind = u32(table[tableBase + 11u]);
  let start = getPosition(row);
  var angle = 0.0;
  var along = 0.0;
  var endpoint = 1u;
  var normalDot = 0.0;
  if (kind == 1u) {
    let up = getTableVector(tableBase, 0u);
    let tangent = getTableVector(tableBase, 3u);
    let normal = getTableVector(tableBase, 6u);
    let arcAngle = table[tableBase + 9u];
    let startDot = dot(pointVector, up);
    let tangentDot = dot(pointVector, tangent);
    normalDot = dot(pointVector, normal);
    let planeLength = sqrt(startDot * startDot + tangentDot * tangentDot);
    // Angle of the point's projection on the great circle, measured from the segment start.
    var projected = 0.0;
    if (planeLength > 0.0) {
      projected = atan2(tangentDot, startDot);
    }
    if (projected >= 0.0 && projected <= arcAngle) {
      angle = atan2(abs(normalDot), planeLength);
      along = projected;
      endpoint = 0u;
    }
  }
  if (endpoint != 0u) {
    let startAngle = geodesicCentralAngle(point, start);
    angle = startAngle;
    if (kind == 1u) {
      let endAngle = geodesicCentralAngle(point, getPosition(row + 1u));
      if (endAngle < startAngle) {
        angle = endAngle;
        endpoint = 2u;
      }
    }
  }
  return SegmentHit(angle, along, endpoint, normalDot);
}`,
    body: /* wgsl */ `let base = resultsOffset + index * STRIDE;
  let point = vec2<f32>(points[pointsOffset + 2u * index], points[pointsOffset + 2u * index + 1u]);
  let pointVector = geodesicUnitVector(point);
  var bestAngle = FLOAT32_MAXIMUM;
  var bestRow = NO_LINE;
  var bestKind = 0u;
  var bestAlong = 0.0;
  var bestNormalDot = 0.0;
  // The radius only tightens the bound; the result check below still applies it exactly.
  let radiusLimit = radius[radiusOffset];
  var radiusAngle = FLOAT32_MAXIMUM;
  if ((bitcast<u32>(radiusLimit) & 0x7f800000u) != 0x7f800000u && radiusLimit >= 0.0) {
    radiusAngle = radiusLimit / SPHERE_RADIUS;
  }
  let isPointFinite = (bitcast<u32>(point.x) & 0x7f800000u) != 0x7f800000u &&
    (bitcast<u32>(point.y) & 0x7f800000u) != 0x7f800000u;
  var stackNode: array<u32, STACK_SIZE>;
  var stackLower: array<f32, STACK_SIZE>;
  var top = 0u;
  let rootLower = nodeLowerAngle(0u, point);
  if (isPointFinite && rootLower < FLOAT32_MAXIMUM) {
    stackNode[0] = 0u;
    stackLower[0] = rootLower;
    top = 1u;
  }
  loop {
    if (top == 0u) { break; }
    top = top - 1u;
    let node = stackNode[top];
    let bound = min(bestAngle, radiusAngle);
    let limit = select(bound, bound * BOUND_SLACK + BOUND_ABSOLUTE_SLACK, bound < 1.0e37);
    if (stackLower[top] > limit) { continue; }
    if (node < INTERNAL_NODE_COUNT) {
      let first = node * 2u + 1u;
      let second = first + 1u;
      let firstLower = nodeLowerAngle(first, point);
      let secondLower = nodeLowerAngle(second, point);
      let firstUsable = firstLower < FLOAT32_MAXIMUM && firstLower <= limit;
      let secondUsable = secondLower < FLOAT32_MAXIMUM && secondLower <= limit;
      // Push the farther child first so the nearer one is popped first.
      if (firstUsable && secondUsable && top + 2u <= STACK_SIZE) {
        if (firstLower <= secondLower) {
          stackNode[top] = second; stackLower[top] = secondLower;
          stackNode[top + 1u] = first; stackLower[top + 1u] = firstLower;
        } else {
          stackNode[top] = first; stackLower[top] = firstLower;
          stackNode[top + 1u] = second; stackLower[top + 1u] = secondLower;
        }
        top = top + 2u;
      } else if (firstUsable && top + 1u <= STACK_SIZE) {
        stackNode[top] = first; stackLower[top] = firstLower;
        top = top + 1u;
      } else if (secondUsable && top + 1u <= STACK_SIZE) {
        stackNode[top] = second; stackLower[top] = secondLower;
        top = top + 1u;
      }
      continue;
    }
    let row = nodeData[nodeDataOffset + node * NODE_WORDS + 4u];
    if (row + 1u >= ROW_COUNT) { continue; }
    let halfAngle = table[tableOffset + row * TABLE_STRIDE + 15u];
    if (halfAngle < 0.0) { continue; }
    let hit = evaluateSegment(row, point, pointVector);
    // Lexicographic (angle, row): the traversal order is spatial, the tie rule is row order.
    if (hit.angle < bestAngle || (hit.angle == bestAngle && row < bestRow)) {
      bestAngle = hit.angle;
      bestRow = row;
      bestKind = hit.endpoint;
      bestAlong = hit.along;
      bestNormalDot = hit.normalDot;
    }
  }
  let radiusMeters = radius[radiusOffset];
  let radiusUsable = (bitcast<u32>(radiusMeters) & 0x7f800000u) != 0x7f800000u && radiusMeters >= 0.0;
  if (bestRow == NO_LINE || !radiusUsable || bestAngle * SPHERE_RADIUS > radiusMeters) {
    results[base] = NO_LINE;
    results[base + 1u] = NO_LINE;
    results[base + 2u] = NAN_BITS;
    results[base + 3u] = NAN_BITS;
    results[base + 4u] = NAN_BITS;
    results[base + 5u] = bitcast<u32>(-1.0);
    results[base + 6u] = NAN_BITS;
    results[base + 7u] = 0u;
    results[base + 8u] = NAN_BITS;
    return;
  }
  let tableBase = tableOffset + bestRow * TABLE_STRIDE;
  let arcAngle = table[tableBase + 9u];
  let start = getPosition(bestRow);
  let end = getPosition(bestRow + 1u);
  let isArc = table[tableBase + 11u] == 1.0;
  var fraction = 0.0;
  var foot = start;
  var measure = vertexMeasures[vertexMeasuresOffset + bestRow];
  if (bestKind == 0u) {
    fraction = clamp(bestAlong / arcAngle, 0.0, 1.0);
    foot = geodesicInterpolate(start, end, arcAngle, fraction);
    measure = measure + bestAlong * SPHERE_RADIUS;
  } else if (bestKind == 2u) {
    fraction = 1.0;
    foot = end;
    measure = vertexMeasures[vertexMeasuresOffset + bestRow + 1u];
  }
  var side = 0i;
  if (isArc && bestAngle > 0.0 && bestNormalDot > 0.0) {
    side = 1i;
  } else if (isArc && bestAngle > 0.0 && bestNormalDot < 0.0) {
    side = -1i;
  }
  let line = findLine(bestRow);
  let lineLength = vertexMeasures[vertexMeasuresOffset + lineOffsets[lineOffsetsOffset + line + 1u] - 1u];
  results[base] = line;
  results[base + 1u] = bestRow - lineOffsets[lineOffsetsOffset + line];
  results[base + 2u] = bitcast<u32>(fraction);
  results[base + 3u] = bitcast<u32>(foot.x);
  results[base + 4u] = bitcast<u32>(foot.y);
  results[base + 5u] = bitcast<u32>(bestAngle * SPHERE_RADIUS);
  results[base + 6u] = bitcast<u32>(measure);
  results[base + 7u] = bitcast<u32>(side);
  results[base + 8u] = bitcast<u32>(select(0.0, measure / lineLength, lineLength > 0.0));`
  });
}

/** Output columns of the projection scatter. @internal */
export type PointProjectionColumns = {
  lineIndices?: GraphDataView<'uint32'>;
  segmentIndices?: GraphDataView<'uint32'>;
  fractions?: GraphDataView<'float32'>;
  footPoints?: GraphDataView<'float32x2'>;
  distances?: GraphDataView<'float32'>;
  measures?: GraphDataView<'float32'>;
  normalizedMeasures?: GraphDataView<'float32'>;
  sides?: GraphDataView<'sint32'>;
  signedOffsets?: GraphDataView<'float32'>;
};

type ScatterColumn = {
  name: string;
  view: GraphDataView | undefined;
  type: 'f32' | 'u32' | 'i32';
  statement: string;
};

/**
 * Builds up to two scatter nodes copying a packed result column into requested outputs (at most
 * seven outputs per node, within the eight storage bindings).
 *
 * @internal
 */
export function createPackedScatterNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    rowCount: number;
    stride: number;
    results: GraphDataView<'uint32'>;
    columns: readonly ScatterColumn[];
  }
): GPUCommandNode<Parameters>[] {
  const columns = props.columns.filter(column => column.view);
  const nodes: GPUCommandNode<Parameters>[] = [];
  for (let first = 0; first < columns.length; first += 7) {
    const group = columns.slice(first, first + 7);
    const bindings: WGSLKernelBinding[] = [
      {name: 'results', view: props.results, type: 'u32', access: 'read'},
      ...group.map(column => ({
        name: column.name,
        view: column.view as GraphDataView,
        type: column.type,
        access: 'read_write' as const
      }))
    ];
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: first === 0 ? `${props.id}` : `${props.id}-${first / 7}`,
        operation: props.operation,
        variant: 'scatter',
        bindings,
        invocationCount: props.rowCount,
        declarations: `const STRIDE: u32 = ${props.stride}u;`,
        body: `let base = resultsOffset + index * STRIDE;
  ${group.map(column => column.statement).join('\n  ')}`
      })
    );
  }
  return nodes;
}

/** Scatter columns of `GPULinearReferencing`. @internal */
export function getPointProjectionScatterColumns(columns: PointProjectionColumns): ScatterColumn[] {
  return [
    {
      name: 'lineIndices',
      view: columns.lineIndices,
      type: 'u32',
      statement: 'lineIndices[lineIndicesOffset + index] = results[base];'
    },
    {
      name: 'segmentIndices',
      view: columns.segmentIndices,
      type: 'u32',
      statement: 'segmentIndices[segmentIndicesOffset + index] = results[base + 1u];'
    },
    {
      name: 'fractions',
      view: columns.fractions,
      type: 'f32',
      statement: 'fractions[fractionsOffset + index] = bitcast<f32>(results[base + 2u]);'
    },
    {
      name: 'footPoints',
      view: columns.footPoints,
      type: 'f32',
      statement: `footPoints[footPointsOffset + 2u * index] = bitcast<f32>(results[base + 3u]);
  footPoints[footPointsOffset + 2u * index + 1u] = bitcast<f32>(results[base + 4u]);`
    },
    {
      name: 'distances',
      view: columns.distances,
      type: 'f32',
      statement: 'distances[distancesOffset + index] = bitcast<f32>(results[base + 5u]);'
    },
    {
      name: 'measures',
      view: columns.measures,
      type: 'f32',
      statement: 'measures[measuresOffset + index] = bitcast<f32>(results[base + 6u]);'
    },
    {
      name: 'normalizedMeasures',
      view: columns.normalizedMeasures,
      type: 'f32',
      statement:
        'normalizedMeasures[normalizedMeasuresOffset + index] = bitcast<f32>(results[base + 8u]);'
    },
    {
      name: 'sides',
      view: columns.sides,
      type: 'i32',
      statement: 'sides[sidesOffset + index] = bitcast<i32>(results[base + 7u]);'
    },
    {
      name: 'signedOffsets',
      view: columns.signedOffsets,
      type: 'f32',
      // Unmatched points copy the NaN fraction instead of multiplying the -1 distance sentinel.
      statement: `signedOffsets[signedOffsetsOffset + index] = select(
    f32(bitcast<i32>(results[base + 7u])) * bitcast<f32>(results[base + 5u]),
    bitcast<f32>(results[base + 2u]),
    results[base] == 0xffffffffu
  );`
    }
  ];
}

/**
 * Locates each event on its line by measure and writes the packed result
 * `[x, y, segmentIndex, tangentX, tangentY, status]` (index and status as u32 bits).
 *
 * @internal
 */
export function createLineLocateNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    positions: GraphDataView<'float32x2'>;
    lineOffsets: GraphDataView<'uint32'>;
    vertexMeasures: GraphDataView<'float32'>;
    eventLineIndices: GraphDataView<'uint32'>;
    eventMeasures: GraphDataView<'float32'>;
    eventOffsets?: GraphDataView<'float32'>;
    parameters?: GraphDataView<'float32'>;
    measureMode: 'distance' | 'fraction';
    coordinateSystem?: 'planar' | 'spherical';
    sphereRadius?: number;
    results: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const spherical = props.coordinateSystem === 'spherical';
  const bindings: WGSLKernelBinding[] = [
    {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
    {name: 'lineOffsets', view: props.lineOffsets, type: 'u32', access: 'read'},
    {name: 'vertexMeasures', view: props.vertexMeasures, type: 'f32', access: 'read'},
    {name: 'eventLineIndices', view: props.eventLineIndices, type: 'u32', access: 'read'},
    {name: 'eventMeasures', view: props.eventMeasures, type: 'f32', access: 'read'},
    {name: 'results', view: props.results, type: 'u32', access: 'read_write'}
  ];
  if (props.eventOffsets) {
    bindings.push({name: 'eventOffsets', view: props.eventOffsets, type: 'f32', access: 'read'});
  }
  if (props.parameters) {
    bindings.push({name: 'parameters', view: props.parameters, type: 'f32', access: 'read'});
  }
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'locate',
    bindings,
    invocationCount: props.eventLineIndices.length,
    declarations: `const LINE_COUNT: u32 = ${props.lineOffsets.length - 1}u;
const ROW_COUNT: u32 = ${props.positions.length}u;
const STRIDE: u32 = ${LINE_LOCATE_RESULT_STRIDE}u;
const IS_FRACTION: bool = ${props.measureMode === 'fraction'};
const STATUS_OK: u32 = ${GPU_LINE_LOCATE_STATUS.ok}u;
const STATUS_CLAMPED: u32 = ${GPU_LINE_LOCATE_STATUS.clamped}u;
const STATUS_INVALID: u32 = ${GPU_LINE_LOCATE_STATUS.invalid}u;
const SPHERE_RADIUS: f32 = ${getWGSLFloatLiteral(props.sphereRadius ?? 1)};
${spherical ? GEODESIC_WGSL : ''}
${POSITION_WGSL}

fn getMeasure(row: u32) -> f32 {
  return vertexMeasures[vertexMeasuresOffset + row];
}`,
    body: /* wgsl */ `let base = resultsOffset + index * STRIDE;
  let line = eventLineIndices[eventLineIndicesOffset + index];
  var lineStart = 0u;
  var lineEnd = 0u;
  if (line < LINE_COUNT) {
    lineStart = min(lineOffsets[lineOffsetsOffset + line], ROW_COUNT);
    lineEnd = min(max(lineOffsets[lineOffsetsOffset + line + 1u], lineStart), ROW_COUNT);
  }
  if (lineEnd <= lineStart) {
    results[base] = NAN_BITS;
    results[base + 1u] = NAN_BITS;
    results[base + 2u] = 0xffffffffu;
    results[base + 3u] = NAN_BITS;
    results[base + 4u] = NAN_BITS;
    results[base + 5u] = STATUS_INVALID;
    return;
  }
  let total = getMeasure(lineEnd - 1u);
  var measure = eventMeasures[eventMeasuresOffset + index];
  ${props.parameters ? 'measure = measure * parameters[parametersOffset] + parameters[parametersOffset + 1u];' : ''}
  if (IS_FRACTION) {
    measure = measure * total;
  }
  var status = STATUS_OK;
  if (measure < 0.0) {
    measure = 0.0;
    status = STATUS_CLAMPED;
  } else if (measure > total) {
    measure = total;
    status = STATUS_CLAMPED;
  }
  var position = getPosition(lineStart);
  var tangent = vec2<f32>(0.0);
  var segment = 0u;
  if (lineEnd - lineStart >= 2u) {
    // Upper bound: the first row in (lineStart, lineEnd) whose measure exceeds the target.
    var low = lineStart + 1u;
    var high = lineEnd;
    while (low < high) {
      let middle = (low + high) / 2u;
      if (getMeasure(middle) <= measure) {
        low = middle + 1u;
      } else {
        high = middle;
      }
    }
    let segmentRow = min(low, lineEnd - 1u) - 1u;
    let start = getPosition(segmentRow);
    let end = getPosition(segmentRow + 1u);
    let segmentMeasure = getMeasure(segmentRow + 1u) - getMeasure(segmentRow);
    var fraction = 0.0;
    if (segmentMeasure > 0.0) {
      fraction = clamp((measure - getMeasure(segmentRow)) / segmentMeasure, 0.0, 1.0);
    }
    ${
      spherical
        ? `let angle = geodesicCentralAngle(start, end);
    if (fraction <= 0.0) {
      position = start;
    } else if (fraction >= 1.0) {
      position = vec2<f32>(start.x + geodesicWrapLongitudeDelta(end.x - start.x), end.y);
    } else {
      position = geodesicInterpolate(start, end, angle, fraction);
    }
    if (angle > 0.0) {
      // Compass direction of the arc at the located point. The tangent is rotated along the
      // great circle from the start vector (not derived from the f32-rounded position), then
      // expressed in the east/north basis of the located point.
      let startLambda = start.x * GEODESIC_DEGREES_TO_RADIANS;
      let startPhi = start.y * GEODESIC_DEGREES_TO_RADIANS;
      let startBearing = geodesicInitialBearingDegrees(start, end) * GEODESIC_DEGREES_TO_RADIANS;
      let startUp = geodesicUnitVector(start);
      let startEast = vec3<f32>(-sin(startLambda), cos(startLambda), 0.0);
      let startNorth = cross(startUp, startEast);
      let startTangent = cos(startBearing) * startNorth + sin(startBearing) * startEast;
      let along = fraction * angle;
      let locatedUp = cos(along) * startUp + sin(along) * startTangent;
      let locatedTangent = cos(along) * startTangent - sin(along) * startUp;
      let eastLength = length(locatedUp.xy);
      var locatedEast = vec3<f32>(0.0, 1.0, 0.0);
      if (eastLength > 0.0) {
        locatedEast = vec3<f32>(-locatedUp.y, locatedUp.x, 0.0) / eastLength;
      }
      let locatedNorth = cross(locatedUp, locatedEast);
      tangent = vec2<f32>(dot(locatedTangent, locatedEast), dot(locatedTangent, locatedNorth));
      let bearing = atan2(tangent.x, tangent.y) * GEODESIC_RADIANS_TO_DEGREES;
      ${
        props.eventOffsets
          ? `if (eventOffsets[eventOffsetsOffset + index] != 0.0) {
        position = geodesicDestination(
          position,
          bearing - 90.0,
          eventOffsets[eventOffsetsOffset + index] / SPHERE_RADIUS
        );
      }`
          : ''
      }
    }`
        : `let direction = end - start;
    position = start + direction * fraction;
    let directionLength = length(direction);
    if (directionLength > 0.0) {
      tangent = direction / directionLength;
    }`
    }
    segment = segmentRow - lineStart;
  }
  ${
    props.eventOffsets && !spherical
      ? 'position = position + vec2<f32>(-tangent.y, tangent.x) * eventOffsets[eventOffsetsOffset + index];'
      : ''
  }
  results[base] = bitcast<u32>(position.x);
  results[base + 1u] = bitcast<u32>(position.y);
  results[base + 2u] = segment;
  results[base + 3u] = bitcast<u32>(tangent.x);
  results[base + 4u] = bitcast<u32>(tangent.y);
  results[base + 5u] = status;`
  });
}
