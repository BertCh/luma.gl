// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {EXACT_ORIENTATION_WGSL} from './exact-orientation-wgsl';
import {GPU_SEGMENT_INTERSECTION_KIND, GPU_SEGMENT_NONE} from './segment-intersection-types';

/** Number of `u32` words per segment-table row. @internal */
export const SEGMENT_TABLE_STRIDE = 8;

/**
 * WGSL helpers shared by the segment kernels: segment classification on top of the exact
 * orientation predicate from `exact-orientation-wgsl.ts`.
 *
 * @internal
 */
export const SEGMENT_PREDICATES_WGSL = /* wgsl */ `
const SEGMENT_NONE: u32 = ${GPU_SEGMENT_NONE}u;
const KIND_NONE: u32 = 0u;
const KIND_PROPER: u32 = ${GPU_SEGMENT_INTERSECTION_KIND.proper}u;
const KIND_TOUCH: u32 = ${GPU_SEGMENT_INTERSECTION_KIND.touch}u;
const KIND_COLLINEAR_TOUCH: u32 = ${GPU_SEGMENT_INTERSECTION_KIND.collinearTouch}u;
const KIND_OVERLAP: u32 = ${GPU_SEGMENT_INTERSECTION_KIND.overlap}u;
const KIND_UNCERTAIN: u32 = ${GPU_SEGMENT_INTERSECTION_KIND.uncertain}u;
${EXACT_ORIENTATION_WGSL}

struct SegmentHit {
  kind: u32,
  point: vec2f,
  endPoint: vec2f,
}

fn axisValue(point: vec2f, useX: bool) -> f32 { return select(point.y, point.x, useX); }

// Picks the point among the four endpoints whose axis coordinate equals value.
fn pointAtAxis(a: vec2f, b: vec2f, c: vec2f, d: vec2f, useX: bool, value: f32) -> vec2f {
  if (axisValue(a, useX) == value) { return a; }
  if (axisValue(b, useX) == value) { return b; }
  if (axisValue(c, useX) == value) { return c; }
  return d;
}

// Classifies segments a-b and c-d, both non-degenerate. withGeometry also fills the points.
fn classifySegments(a: vec2f, b: vec2f, c: vec2f, d: vec2f, withGeometry: bool) -> SegmentHit {
  var hit = SegmentHit(KIND_NONE, a, a);
  if (max(a.x, b.x) < min(c.x, d.x) || max(c.x, d.x) < min(a.x, b.x) ||
      max(a.y, b.y) < min(c.y, d.y) || max(c.y, d.y) < min(a.y, b.y)) {
    return hit;
  }
  let o1 = orientSign(a, b, c);
  let o2 = orientSign(a, b, d);
  if (o1 == 2 || o2 == 2) { hit.kind = KIND_UNCERTAIN; return hit; }
  if (o1 != 0 && o1 == o2) { return hit; }
  let o3 = orientSign(c, d, a);
  let o4 = orientSign(c, d, b);
  if (o3 == 2 || o4 == 2) { hit.kind = KIND_UNCERTAIN; return hit; }
  if (o3 != 0 && o3 == o4) { return hit; }
  if (o1 == 0 && o2 == 0) {
    // All four orientations vanish: collinear. Compare along the axis the line is not parallel to.
    let useX = a.x != b.x;
    let low = max(min(axisValue(a, useX), axisValue(b, useX)), min(axisValue(c, useX), axisValue(d, useX)));
    let high = min(max(axisValue(a, useX), axisValue(b, useX)), max(axisValue(c, useX), axisValue(d, useX)));
    if (low > high) { return hit; }
    hit.kind = select(KIND_OVERLAP, KIND_COLLINEAR_TOUCH, low == high);
    hit.point = pointAtAxis(a, b, c, d, useX, low);
    hit.endPoint = pointAtAxis(a, b, c, d, useX, high);
    return hit;
  }
  if (o1 != 0 && o2 != 0 && o3 != 0 && o4 != 0) {
    hit.kind = KIND_PROPER;
    if (withGeometry) {
      let r = b - a;
      let s = d - c;
      let denominator = r.x * s.y - r.y * s.x;
      let t = clamp(((c.x - a.x) * s.y - (c.y - a.y) * s.x) / denominator, 0.0, 1.0);
      let crossing = a + r * t;
      hit.point = clamp(crossing, max(min(a, b), min(c, d)), min(max(a, b), max(c, d)));
      hit.endPoint = hit.point;
    }
    return hit;
  }
  hit.kind = KIND_TOUCH;
  if (o1 == 0) { hit.point = c; } else if (o2 == 0) { hit.point = d; } else if (o3 == 0) { hit.point = a; } else { hit.point = b; }
  hit.endPoint = hit.point;
  return hit;
}

`;

/**
 * WGSL accessors for one segment table, bound as `${prefix}Table`. Rows hold the segment's two
 * endpoints as `f32` bit patterns, then its ring, successor segment and feature.
 *
 * @internal
 */
export function getSegmentTableAccessorsWGSL(prefix: string): string {
  const table = `${prefix}Table`;
  const base = (row: string) => `${table}Offset + ${row} * ${SEGMENT_TABLE_STRIDE}u`;
  return /* wgsl */ `
fn ${prefix}Ring(row: u32) -> u32 { return ${table}[${base('row')} + 4u]; }
fn ${prefix}Valid(row: u32) -> bool { return ${prefix}Ring(row) != SEGMENT_NONE; }
fn ${prefix}Successor(row: u32) -> u32 { return ${table}[${base('row')} + 5u]; }
fn ${prefix}Feature(row: u32) -> u32 { return ${table}[${base('row')} + 6u]; }
fn ${prefix}Start(row: u32) -> vec2f {
  return vec2f(bitcast<f32>(${table}[${base('row')}]), bitcast<f32>(${table}[${base('row')} + 1u]));
}
fn ${prefix}End(row: u32) -> vec2f {
  return vec2f(bitcast<f32>(${table}[${base('row')} + 2u]), bitcast<f32>(${table}[${base('row')} + 3u]));
}`;
}
