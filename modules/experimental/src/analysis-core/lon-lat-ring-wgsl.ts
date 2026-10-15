// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {validateGPUStatusPort, type GPULonLatPolygonStatusPort} from './contracts/index';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../utils/wgsl-kernel-nodes';

/**
 * Binding-free longitude/latitude ring walk. Non-pole edges take the short way; pole edges
 * retain their raw step because the degree-plane pole line closes an already capped ring.
 * Great-circle minor arcs through a pole are ambiguous in v1, as are antipodal endpoints.
 * @internal
 */
export const LON_LAT_RING_WGSL = /* wgsl */ `
const LON_LAT_POLE_LATITUDE_TOLERANCE: f32 = 1e-6;
const LON_LAT_AMBIGUOUS_STEP_TOLERANCE: f32 = 1e-4;
const LON_LAT_POLE_WINDING_LIMIT: f32 = 180.0;
const LON_LAT_RADIANS_PER_DEGREE: f32 = 0.0174532925199433;

fn lonLatWrapLongitudeStep(rawStep: f32) -> f32 {
  return rawStep - 360.0 * round(rawStep / 360.0);
}

fn lonLatIsPoleEdge(a: vec2<f32>, b: vec2<f32>) -> bool {
  return abs(a.y) >= 90.0 - LON_LAT_POLE_LATITUDE_TOLERANCE &&
    abs(b.y) >= 90.0 - LON_LAT_POLE_LATITUDE_TOLERANCE && a.y * b.y > 0.0;
}

fn lonLatRingStep(a: vec2<f32>, b: vec2<f32>) -> f32 {
  let rawStep = b.x - a.x;
  if (lonLatIsPoleEdge(a, b)) { return rawStep; }
  return lonLatWrapLongitudeStep(rawStep);
}

fn lonLatIsCrossingEdge(a: vec2<f32>, b: vec2<f32>) -> bool {
  return !lonLatIsPoleEdge(a, b) && abs(b.x - a.x) > 180.0;
}

fn lonLatIsAmbiguousLinearEdge(a: vec2<f32>, b: vec2<f32>) -> bool {
  return !lonLatIsPoleEdge(a, b) &&
    abs(lonLatWrapLongitudeStep(b.x - a.x)) >= 180.0 - LON_LAT_AMBIGUOUS_STEP_TOLERANCE;
}

fn lonLatIsAmbiguousGreatCircleEdge(a: vec2<f32>, b: vec2<f32>) -> bool {
  if (lonLatIsPoleEdge(a, b)) { return false; }
  // Rotate the first longitude to zero. Colatitudes and the supplementary longitude keep
  // exact pole/180-degree inputs exact, avoiding sin(pi)'s f32 residual at the v1 threshold.
  let longitude = lonLatWrapLongitudeStep(b.x - a.x);
  let longitudeRadians = longitude * LON_LAT_RADIANS_PER_DEGREE;
  let longitudeSine = sign(longitude) * sin(min(abs(longitude), 180.0 - abs(longitude)) * LON_LAT_RADIANS_PER_DEGREE);
  let firstCosine = sin((90.0 - abs(a.y)) * LON_LAT_RADIANS_PER_DEGREE);
  let secondCosine = sin((90.0 - abs(b.y)) * LON_LAT_RADIANS_PER_DEGREE);
  let first = vec3<f32>(firstCosine, 0.0, sin(a.y * LON_LAT_RADIANS_PER_DEGREE));
  let second = vec3<f32>(secondCosine * cos(longitudeRadians), secondCosine * longitudeSine, sin(b.y * LON_LAT_RADIANS_PER_DEGREE));
  if (length(first + second) <= 1e-6) { return true; }
  if (abs(a.y) >= 90.0 - LON_LAT_POLE_LATITUDE_TOLERANCE ||
      abs(b.y) >= 90.0 - LON_LAT_POLE_LATITUDE_TOLERANCE) { return true; }
  let normal = cross(first, second);
  let normalLength = length(normal);
  if (normalLength == 0.0) { return false; }
  let unitNormal = normal / normalLength;
  if (abs(unitNormal.z) > sin(LON_LAT_POLE_LATITUDE_TOLERANCE * LON_LAT_RADIANS_PER_DEGREE)) { return false; }
  let north = normalize(vec3<f32>(0.0, 0.0, 1.0) - unitNormal.z * unitNormal);
  for (var hemisphere = 0u; hemisphere < 2u; hemisphere++) {
    let pole = select(north, -north, hemisphere == 1u);
    if (dot(cross(first, pole), normal) >= 0.0 && dot(cross(pole, second), normal) >= 0.0) { return true; }
  }
  return false;
}

fn lonLatIsPoleWinding(winding: f32) -> bool {
  return abs(winding) > LON_LAT_POLE_WINDING_LIMIT;
}
`;

/** Returns the shared binding-free ring walk source. @internal */
export function getLonLatRingWGSL(): string {
  return LON_LAT_RING_WGSL;
}

/** Geometry and optional counters for a consumer's ring check. @internal */
export type LonLatPolygonCheckNodeProps = {
  /** Command node identifier. */
  id: string;
  /** Consumer operation name for graph diagnostics. */
  operation: string;
  /** Longitude/latitude pairs in degrees. */
  positions: GraphDataView<'float32x2'>;
  /** R + 1 vertex offsets, with implicit ring closure. */
  ringOffsets: GraphDataView<'uint32'>;
  /** Caller-cleared counters; only supplied fields are bound. */
  status: GPULonLatPolygonStatusPort;
};

/** Whether a lon/lat consumer has requested either ring diagnostic. @internal */
export function hasLonLatPolygonStatus(status?: GPULonLatPolygonStatusPort): boolean {
  return Boolean(status?.antimeridianUncertainCount || status?.poleUncertainCount);
}

/**
 * Counts crossing and pole-winding rings with one GPU invocation per ring. Repeated closing
 * vertices are omitted. Counters are additive and must be cleared by the consumer each run.
 * An empty status port produces a harmless node with no status bindings.
 * @internal
 */
export function createLonLatPolygonCheckNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: LonLatPolygonCheckNodeProps
): GPUCommandNode<Parameters> {
  validateGPUStatusPort(props.operation, props.status);
  const bindings: WGSLKernelBinding[] = [
    {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
    {name: 'ringOffsets', view: props.ringOffsets, type: 'u32', access: 'read'}
  ];
  for (const name of ['antimeridianUncertainCount', 'poleUncertainCount'] as const) {
    const view = props.status[name];
    if (view) {
      bindings.push({name, view, type: 'atomic<u32>', access: 'read_write'});
    }
  }
  return createWGSLKernelNode(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'lon-lat-polygon-check',
    bindings,
    invocationCount: Math.max(0, props.ringOffsets.length - 1),
    declarations: `${LON_LAT_RING_WGSL}
fn lonLatCheckPosition(vertex: u32) -> vec2<f32> {
  let positionOffset = positionsOffset + 2u * vertex;
  return vec2<f32>(positions[positionOffset], positions[positionOffset + 1u]);
}`,
    body: `let begin = ringOffsets[ringOffsetsOffset + index];
  var end = ringOffsets[ringOffsetsOffset + index + 1u];
  if (end <= begin + 2u) { return; }
  let first = lonLatCheckPosition(begin);
  if (all(first == lonLatCheckPosition(end - 1u))) { end--; }
  if (end <= begin + 2u) { return; }
  // Find three distinct positions without quadratic work or per-ring scratch storage.
  var second = first;
  var hasSecond = false;
  var hasThird = false;
  var winding = 0.0;
  var crossing = false;
  for (var vertex = begin; vertex < end; vertex++) {
    let position = lonLatCheckPosition(vertex);
    if (any(position != first)) {
      if (!hasSecond) { second = position; hasSecond = true; }
      else if (any(position != second)) { hasThird = true; }
    }
    var next = first;
    if (vertex + 1u < end) { next = lonLatCheckPosition(vertex + 1u); }
    winding += lonLatRingStep(position, next);
    crossing = crossing || lonLatIsCrossingEdge(position, next);
  }
  if (!hasThird) { return; }
  ${props.status.antimeridianUncertainCount ? 'if (crossing) { atomicAdd(&antimeridianUncertainCount[antimeridianUncertainCountOffset], 1u); }' : ''}
  ${props.status.poleUncertainCount ? 'if (lonLatIsPoleWinding(winding)) { atomicAdd(&poleUncertainCount[poleUncertainCountOffset], 1u); }' : ''}`
  });
}
