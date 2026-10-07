// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import type {WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import type {GPUSpatialJoinGeometry} from '../spatial-join/index';

/** Returns the graph views of a geometry in binding order: positions, then every offsets view. @internal */
export function getPredicateGeometryViews(geometry: GPUSpatialJoinGeometry): GraphDataView[] {
  if (geometry.kind === 'points') {
    return [geometry.positions];
  }
  if (geometry.kind === 'lines') {
    return [geometry.positions, geometry.lineOffsets];
  }
  return [
    geometry.positions,
    geometry.featureOffsets,
    geometry.polygonOffsets,
    geometry.ringOffsets
  ];
}

/** Returns the number of features of a geometry. @internal */
export function getPredicateFeatureCount(geometry: GPUSpatialJoinGeometry): number {
  if (geometry.kind === 'points') {
    return geometry.positions.length;
  }
  if (geometry.kind === 'lines') {
    return geometry.lineOffsets.length - 1;
  }
  return geometry.featureOffsets.length - 1;
}

/**
 * Returns the read-only storage bindings of a geometry, named `<prefix>Positions` (unless
 * `includePositions` is false) and, by kind,
 * `<prefix>LineOffsets` or `<prefix>FeatureOffsets`, `<prefix>PolygonOffsets`, `<prefix>RingOffsets`.
 * @internal
 */
export function getPredicateGeometryBindings(
  prefix: string,
  geometry: GPUSpatialJoinGeometry,
  includePositions: boolean = true
): WGSLKernelBinding[] {
  const bindings: WGSLKernelBinding[] = includePositions
    ? [{name: `${prefix}Positions`, view: geometry.positions, type: 'f32', access: 'read'}]
    : [];
  if (geometry.kind === 'lines') {
    bindings.push({
      name: `${prefix}LineOffsets`,
      view: geometry.lineOffsets,
      type: 'u32',
      access: 'read'
    });
  } else if (geometry.kind === 'polygons') {
    bindings.push(
      {name: `${prefix}FeatureOffsets`, view: geometry.featureOffsets, type: 'u32', access: 'read'},
      {name: `${prefix}PolygonOffsets`, view: geometry.polygonOffsets, type: 'u32', access: 'read'},
      {name: `${prefix}RingOffsets`, view: geometry.ringOffsets, type: 'u32', access: 'read'}
    );
  }
  return bindings;
}

/** Emits a WGSL function returning the last row in `[0, count)` whose offset is at most `value`. */
function getLastAtMostWGSL(functionName: string, arrayName: string): string {
  return `fn ${functionName}(count: u32, value: u32) -> u32 {
  if (count == 0u || ${arrayName}[${arrayName}Offset] > value) { return 0u; }
  var low = 0u;
  var high = count;
  while (high - low > 1u) {
    let middle = (low + high) / 2u;
    if (${arrayName}[${arrayName}Offset + middle] <= value) { low = middle; } else { high = middle; }
  }
  return low;
}`;
}

/**
 * Returns WGSL accessors for the bindings of {@link getPredicateGeometryBindings}:
 * `<prefix>Vertex(v) -> vec2f`, `<prefix>VertexStart(f) -> u32` (valid for `f` up to and including
 * the feature count, so `VertexStart(f + 1)` is the end), `<prefix>FeatureOfVertex(v) -> u32`, and
 * the constants `<prefix>_FEATURE_COUNT` and `<prefix>_VERTEX_COUNT`.
 * @internal
 */
export function getPredicateGeometryWGSL(
  prefix: string,
  geometry: GPUSpatialJoinGeometry,
  includePositions: boolean = true
): string {
  const vertexCount = geometry.positions.length;
  const featureCount = getPredicateFeatureCount(geometry);
  const common = `const ${prefix}_FEATURE_COUNT: u32 = ${featureCount}u;
const ${prefix}_VERTEX_COUNT: u32 = ${vertexCount}u;
${
  includePositions
    ? `fn ${prefix}Vertex(vertex: u32) -> vec2f {
  return vec2f(${prefix}Positions[${prefix}PositionsOffset + vertex * 2u], ${prefix}Positions[${prefix}PositionsOffset + vertex * 2u + 1u]);
}`
    : ''
}`;
  if (geometry.kind === 'points') {
    return `${common}
fn ${prefix}VertexStart(feature: u32) -> u32 { return min(feature, ${prefix}_VERTEX_COUNT); }
fn ${prefix}FeatureOfVertex(vertex: u32) -> u32 { return vertex; }`;
  }
  if (geometry.kind === 'lines') {
    return `${common}
fn ${prefix}VertexStart(feature: u32) -> u32 {
  return min(${prefix}LineOffsets[${prefix}LineOffsetsOffset + feature], ${prefix}_VERTEX_COUNT);
}
${getLastAtMostWGSL(`${prefix}LastLine`, `${prefix}LineOffsets`)}
fn ${prefix}FeatureOfVertex(vertex: u32) -> u32 { return ${prefix}LastLine(${prefix}_FEATURE_COUNT, vertex); }`;
  }
  return `${common}
const ${prefix}_POLYGON_COUNT: u32 = ${geometry.polygonOffsets.length - 1}u;
const ${prefix}_RING_COUNT: u32 = ${geometry.ringOffsets.length - 1}u;
fn ${prefix}VertexStart(feature: u32) -> u32 {
  let polygon = ${prefix}FeatureOffsets[${prefix}FeatureOffsetsOffset + feature];
  let ring = ${prefix}PolygonOffsets[${prefix}PolygonOffsetsOffset + polygon];
  return min(${prefix}RingOffsets[${prefix}RingOffsetsOffset + ring], ${prefix}_VERTEX_COUNT);
}
${getLastAtMostWGSL(`${prefix}LastRing`, `${prefix}RingOffsets`)}
${getLastAtMostWGSL(`${prefix}LastPolygon`, `${prefix}PolygonOffsets`)}
${getLastAtMostWGSL(`${prefix}LastFeature`, `${prefix}FeatureOffsets`)}
fn ${prefix}FeatureOfVertex(vertex: u32) -> u32 {
  let ring = ${prefix}LastRing(${prefix}_RING_COUNT, vertex);
  let polygon = ${prefix}LastPolygon(${prefix}_POLYGON_COUNT, ring);
  return ${prefix}LastFeature(${prefix}_FEATURE_COUNT, polygon);
}`;
}
