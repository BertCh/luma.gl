// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import type {GPUSegmentGeometry} from './segment-intersection-types';
import {SEGMENT_PREDICATES_WGSL, SEGMENT_TABLE_STRIDE} from './segment-intersection-wgsl';

/** Maximum forward distance searched for the next non-degenerate segment of a ring. */
const MAXIMUM_SUCCESSOR_SCAN = 1024;

/** Returns the number of segment slots (one per vertex) of `geometry`. @internal */
export function getSegmentSlotCount(geometry: GPUSegmentGeometry): number {
  return geometry.positions.length;
}

/**
 * Builds the segment table of `geometry`: one row of {@link SEGMENT_TABLE_STRIDE} `u32` words per
 * vertex, so a segment ID is its start-vertex index.
 *
 * Words 0 to 3 are the endpoint bit patterns, word 4 the ring (or linestring) row, word 5 the
 * successor segment (the next non-degenerate segment of the same ring, wrapping for closed
 * rings, or `SEGMENT_NONE`) and word 6 the feature row. Skipped segments have ring `SEGMENT_NONE`.
 * Optional `minima` and `maxima` receive segment bounds, inverted for skipped segments.
 *
 * @internal
 */
export function createSegmentTableNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    geometry: GPUSegmentGeometry;
    table: GraphDataView<'uint32'>;
    minima?: GraphDataView<'float32x2'>;
    maxima?: GraphDataView<'float32x2'>;
  }
): GPUCommandNode<Parameters> {
  const {geometry} = props;
  const isPolygons = geometry.kind === 'polygons';
  const bindings: WGSLKernelBinding[] = [
    {name: 'positions', view: geometry.positions, type: 'f32', access: 'read'}
  ];
  let declarations: string;
  if (isPolygons) {
    bindings.push(
      {name: 'ringOffsets', view: geometry.ringOffsets, type: 'u32', access: 'read'},
      {name: 'polygonOffsets', view: geometry.polygonOffsets, type: 'u32', access: 'read'},
      {name: 'featureOffsets', view: geometry.featureOffsets, type: 'u32', access: 'read'}
    );
    declarations = `
const RING_COUNT: u32 = ${geometry.ringOffsets.length - 1}u;
const POLYGON_COUNT: u32 = ${geometry.polygonOffsets.length - 1}u;
const FEATURE_COUNT: u32 = ${geometry.featureOffsets.length - 1}u;
const MINIMUM_RING_VERTICES: u32 = 3u;
fn lastAtMost(offsetsKind: u32, count: u32, value: u32) -> u32 {
  if (count == 0u) { return SEGMENT_NONE; }
  var low = 0u;
  var high = count;
  if (readOffset(offsetsKind, 0u) > value) { return SEGMENT_NONE; }
  while (high - low > 1u) {
    let middle = (low + high) / 2u;
    if (readOffset(offsetsKind, middle) <= value) { low = middle; } else { high = middle; }
  }
  return low;
}
fn readOffset(offsetsKind: u32, row: u32) -> u32 {
  if (offsetsKind == 0u) { return ringOffsets[ringOffsetsOffset + row]; }
  if (offsetsKind == 1u) { return polygonOffsets[polygonOffsetsOffset + row]; }
  return featureOffsets[featureOffsetsOffset + row];
}
fn ringStartOf(ring: u32) -> u32 { return ringOffsets[ringOffsetsOffset + ring]; }
fn ringEndOf(ring: u32) -> u32 { return min(ringOffsets[ringOffsetsOffset + ring + 1u], VERTEX_COUNT); }
fn featureOfRing(ring: u32) -> u32 {
  if (ring >= polygonOffsets[polygonOffsetsOffset + POLYGON_COUNT]) { return SEGMENT_NONE; }
  let polygon = lastAtMost(1u, POLYGON_COUNT, ring);
  if (polygon == SEGMENT_NONE) { return SEGMENT_NONE; }
  if (polygon >= featureOffsets[featureOffsetsOffset + FEATURE_COUNT]) { return SEGMENT_NONE; }
  return lastAtMost(2u, FEATURE_COUNT, polygon);
}
fn ringOfVertex(vertex: u32) -> u32 {
  let ring = lastAtMost(0u, RING_COUNT, vertex);
  if (ring == SEGMENT_NONE || vertex >= ringEndOf(ring)) { return SEGMENT_NONE; }
  return ring;
}`;
  } else {
    bindings.push({name: 'lineOffsets', view: geometry.lineOffsets, type: 'u32', access: 'read'});
    declarations = `
const RING_COUNT: u32 = ${geometry.lineOffsets.length - 1}u;
const MINIMUM_RING_VERTICES: u32 = 2u;
fn readOffset(offsetsKind: u32, row: u32) -> u32 { return lineOffsets[lineOffsetsOffset + row]; }
fn lastAtMost(offsetsKind: u32, count: u32, value: u32) -> u32 {
  if (count == 0u) { return SEGMENT_NONE; }
  var low = 0u;
  var high = count;
  if (readOffset(offsetsKind, 0u) > value) { return SEGMENT_NONE; }
  while (high - low > 1u) {
    let middle = (low + high) / 2u;
    if (readOffset(offsetsKind, middle) <= value) { low = middle; } else { high = middle; }
  }
  return low;
}
fn ringStartOf(ring: u32) -> u32 { return lineOffsets[lineOffsetsOffset + ring]; }
fn ringEndOf(ring: u32) -> u32 { return min(lineOffsets[lineOffsetsOffset + ring + 1u], VERTEX_COUNT); }
fn featureOfRing(ring: u32) -> u32 { return ring; }
fn ringOfVertex(vertex: u32) -> u32 {
  let ring = lastAtMost(0u, RING_COUNT, vertex);
  if (ring == SEGMENT_NONE || vertex >= ringEndOf(ring)) { return SEGMENT_NONE; }
  return ring;
}`;
  }
  bindings.push({name: 'table', view: props.table, type: 'u32', access: 'read_write'});
  if (props.minima && props.maxima) {
    bindings.push(
      {name: 'segmentMinima', view: props.minima, type: 'f32', access: 'read_write'},
      {name: 'segmentMaxima', view: props.maxima, type: 'f32', access: 'read_write'}
    );
  }
  const writesBounds = Boolean(props.minima && props.maxima);
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: `segment-table-${geometry.kind}`,
    bindings,
    invocationCount: getSegmentSlotCount(geometry),
    declarations: `${SEGMENT_PREDICATES_WGSL}
const VERTEX_COUNT: u32 = ${geometry.positions.length}u;
const CYCLIC: bool = ${isPolygons};
const MAXIMUM_SUCCESSOR_SCAN: u32 = ${MAXIMUM_SUCCESSOR_SCAN}u;
const FLOAT32_MAXIMUM: f32 = 3.402823466e+38;
${declarations}
fn vertexAt(vertex: u32) -> vec2f {
  return vec2f(positions[positionsOffset + vertex * 2u], positions[positionsOffset + vertex * 2u + 1u]);
}
fn finiteValue(value: f32) -> bool { return value == value && abs(value) <= FLOAT32_MAXIMUM; }
fn finitePoint(point: vec2f) -> bool { return finiteValue(point.x) && finiteValue(point.y); }
// The vertex that ends the segment starting at vertex, or SEGMENT_NONE when there is none.
fn nextVertex(vertex: u32, ringStart: u32, ringEnd: u32, closedLine: bool) -> u32 {
  if (vertex + 1u < ringEnd) { return vertex + 1u; }
  if (CYCLIC || closedLine) { return ringStart; }
  return SEGMENT_NONE;
}
fn segmentIsUsable(vertex: u32, ringStart: u32, ringEnd: u32, closedLine: bool) -> bool {
  let next = nextVertex(vertex, ringStart, ringEnd, closedLine);
  if (next == SEGMENT_NONE) { return false; }
  // A closed linestring's last vertex repeats its first and owns no segment.
  if (!CYCLIC && vertex + 1u == ringEnd) { return false; }
  let start = vertexAt(vertex);
  let end = vertexAt(next);
  return finitePoint(start) && finitePoint(end) && (start.x != end.x || start.y != end.y);
}`,
    body: `var row0 = 0u; var row1 = 0u; var row2 = 0u; var row3 = 0u;
  var rowRing = SEGMENT_NONE;
  var rowSuccessor = SEGMENT_NONE;
  var rowFeature = SEGMENT_NONE;
  var minimum = vec2f(FLOAT32_MAXIMUM);
  var maximum = vec2f(-FLOAT32_MAXIMUM);
  let ring = ringOfVertex(index);
  if (ring != SEGMENT_NONE) {
    let ringStart = ringStartOf(ring);
    let ringEnd = ringEndOf(ring);
    let feature = featureOfRing(ring);
    var closedLine = false;
    if (!CYCLIC && ringEnd - ringStart >= 3u) {
      let first = vertexAt(ringStart);
      let last = vertexAt(ringEnd - 1u);
      closedLine = first.x == last.x && first.y == last.y;
    }
    if (ringEnd - ringStart >= MINIMUM_RING_VERTICES && feature != SEGMENT_NONE &&
        segmentIsUsable(index, ringStart, ringEnd, closedLine)) {
      let start = vertexAt(index);
      let end = vertexAt(nextVertex(index, ringStart, ringEnd, closedLine));
      row0 = bitcast<u32>(start.x); row1 = bitcast<u32>(start.y);
      row2 = bitcast<u32>(end.x); row3 = bitcast<u32>(end.y);
      rowRing = ring;
      rowFeature = feature;
      minimum = min(start, end);
      maximum = max(start, end);
      var candidate = index;
      for (var step = 0u; step < min(ringEnd - ringStart, MAXIMUM_SUCCESSOR_SCAN); step++) {
        let following = nextVertex(candidate, ringStart, ringEnd, closedLine);
        if (following == SEGMENT_NONE) { break; }
        candidate = following;
        if (segmentIsUsable(candidate, ringStart, ringEnd, closedLine)) {
          rowSuccessor = candidate;
          break;
        }
      }
    }
  }
  let base = tableOffset + index * ${SEGMENT_TABLE_STRIDE}u;
  table[base] = row0; table[base + 1u] = row1; table[base + 2u] = row2; table[base + 3u] = row3;
  table[base + 4u] = rowRing; table[base + 5u] = rowSuccessor; table[base + 6u] = rowFeature;
  table[base + 7u] = 0u;
  ${
    writesBounds
      ? `segmentMinima[segmentMinimaOffset + index * 2u] = minimum.x;
  segmentMinima[segmentMinimaOffset + index * 2u + 1u] = minimum.y;
  segmentMaxima[segmentMaximaOffset + index * 2u] = maximum.x;
  segmentMaxima[segmentMaximaOffset + index * 2u + 1u] = maximum.y;`
      : ''
  }`
  });
}
