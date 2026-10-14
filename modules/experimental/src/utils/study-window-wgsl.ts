// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  POLYGON_EDGE_SLAB_PARAMS_STRIDE,
  getPolygonEdgeSlabQueryWGSL,
  getPolygonEdgeSlabLoopWGSL,
  type PolygonEdgeSlabs,
  type PolygonEdgeSlabStorage
} from './polygon-edge-slabs';
import type {WGSLKernelBinding} from './wgsl-kernel-nodes';

/**
 * Word layout of a polygon study window's one-binding `packed` view (`GPUPolygonStudyWindowViews.packed`).
 * Every field is a word offset from the start of the view; floats are stored as their bits.
 * @internal
 */
export type StudyWindowPackedLayout = {
  /** `GPU_STUDY_WINDOW_SUMMARY_LENGTH` (8) floats laid out by `GPU_STUDY_WINDOW_SUMMARY`. */
  readonly summary: number;
  /** `4 * vertexCount` floats: oriented edge `[ax, ay, bx, by]` of the ring edge starting at each vertex. */
  readonly edges: number;
  /** `POLYGON_EDGE_SLAB_PARAMS_STRIDE` words: the edge-slab params of the single owner. */
  readonly slabParams: number;
  /** `slabOffsetCount` words: the edge-slab offsets. */
  readonly slabOffsets: number;
  /** `2 * slabEntryCount` words: the edge-slab entry pairs. */
  readonly slabEntries: number;
  /** One word: the number of sampler cells. */
  readonly sampleCellCount: number;
  /** `sampleCellCapacity` words: sampler cell ids, bit 31 set for boundary cells. */
  readonly sampleCells: number;
  /** Total word count of the packed view. */
  readonly length: number;
};

/** Element counts the packed layout is derived from. @internal */
export type StudyWindowPackedSizes = {
  /** Vertex count of the window (rows of the `edges` view divided by 4). */
  vertexCount: number;
  /** `slabs.slabOffsets.length`. */
  slabOffsetCount: number;
  /** `slabs.slabEntries.length` (pairs). */
  slabEntryCount: number;
  /** `sampleCells.length`. */
  sampleCellCapacity: number;
};

/** Number of floats in the study window summary (mirrors `GPU_STUDY_WINDOW_SUMMARY_LENGTH`). */
const SUMMARY_LENGTH = 8;

/**
 * Computes the packed layout of a polygon study window. Producer and consumers derive it from the
 * same view lengths, so it is never stored. @internal
 */
export function getStudyWindowPackedLayout(sizes: StudyWindowPackedSizes): StudyWindowPackedLayout {
  const summary = 0;
  const edges = summary + SUMMARY_LENGTH;
  const slabParams = edges + 4 * sizes.vertexCount;
  const slabOffsets = slabParams + POLYGON_EDGE_SLAB_PARAMS_STRIDE;
  const slabEntries = slabOffsets + sizes.slabOffsetCount;
  const sampleCellCount = slabEntries + 2 * sizes.slabEntryCount;
  const sampleCells = sampleCellCount + 1;
  const length = sampleCells + sizes.sampleCellCapacity;
  return {
    summary,
    edges,
    slabParams,
    slabOffsets,
    slabEntries,
    sampleCellCount,
    sampleCells,
    length
  };
}

/** Bit 31 of a sampler cell id: the cell touches the boundary and needs a point-in-window test. @internal */
export const STUDY_WINDOW_BOUNDARY_CELL_BIT = 0x80000000;

/**
 * WGSL of `getStudyWindowSampleCell(index: u32) -> vec4f`: `(originX, originY, sizeX, sizeY)` of the
 * row-major cell `index` (bit 31 ignored) of the sampling grid laid over the window's bounding box.
 * The preparation writes its rasterization placement with this same function and samplers draw
 * points with it, so both agree bit for bit.
 * @param sampleGrid Columns and rows of the sampling grid (baked as constants).
 * @param readSummary Name of a consumer WGSL function `(slot: u32) -> f32` reading summary slots 0-3.
 * @internal
 */
export function getStudyWindowSampleCellWGSL(
  sampleGrid: readonly [number, number],
  readSummary: string
): string {
  return /* wgsl */ `
const STUDY_WINDOW_SAMPLE_COLUMNS: u32 = ${sampleGrid[0]}u;
const STUDY_WINDOW_SAMPLE_ROWS: u32 = ${sampleGrid[1]}u;

fn getStudyWindowSampleCell(index: u32) -> vec4f {
  let cell = index & 0x7fffffffu;
  let minimum = vec2f(${readSummary}(0u), ${readSummary}(1u));
  let maximum = vec2f(${readSummary}(2u), ${readSummary}(3u));
  let size = (maximum - minimum) / vec2f(f32(STUDY_WINDOW_SAMPLE_COLUMNS), f32(STUDY_WINDOW_SAMPLE_ROWS));
  let column = cell % STUDY_WINDOW_SAMPLE_COLUMNS;
  let row = cell / STUDY_WINDOW_SAMPLE_COLUMNS;
  return vec4f(minimum + size * vec2f(f32(column), f32(row)), size);
}
`;
}

/** Radially symmetric KDE kernels; mirrors GPUKernelDensityKernel without importing an entry. */
export type StudyWindowKernel = 'gaussian' | 'quartic' | 'epanechnikov';

/** Compile-time study-window declarations. @internal */
export type StudyWindowWGSLOptions = {
  /** Packed layout derived from the consumer's views. */
  layout: StudyWindowPackedLayout;
  /** Source vertex count of the window's edge-slab index. */
  vertexCount: number;
  /** Slab entry capacity, excluding the walked region. */
  entryCapacity: number;
  /** Emit getStudyWindowKernelMass for this kernel when present. */
  kernel?: StudyWindowKernel;
  /** Emit the shared sample-cell placement function when present. */
  sampleGrid?: readonly [number, number];
};

/** Single storage binding for all study-window queries. @internal */
export function getStudyWindowBinding(views: {packed: GraphDataView<'uint32'>}): WGSLKernelBinding {
  return {name: 'studyWindow', view: views.packed, type: 'u32', access: 'read'};
}

/** Derives the layout and index sizes from the actual consumer views. @internal */
export function getStudyWindowWGSLOptions(
  views: {
    edges: GraphDataView<'float32'>;
    slabs: PolygonEdgeSlabs;
    sampleCells: GraphDataView<'uint32'>;
    sampleGrid: readonly [number, number];
  },
  kernel?: StudyWindowKernel
): StudyWindowWGSLOptions {
  const vertexCount = views.edges.length / 4;
  return {
    layout: getStudyWindowPackedLayout({
      vertexCount,
      slabOffsetCount: views.slabs.slabOffsets.length,
      slabEntryCount: views.slabs.slabEntries.length,
      sampleCellCapacity: views.sampleCells.length
    }),
    vertexCount,
    entryCapacity: views.slabs.entryCapacity,
    sampleGrid: views.sampleGrid,
    kernel
  };
}

/**
 * Module-scope queries in absolute operator-frame coordinates. Requires getStudyWindowBinding,
 * EXACT_ORIENTATION_WGSL (which already declares isFiniteBits) and EXACT_GEOMETRY_WGSL, each
 * declared once; do not add IS_FINITE_BITS_WGSL again.
 * The consumer must call isInsideStudyWindow from ONE call site: its crossing is the module's
 * sole exact-orientation path. Fan queries use floating-point metric arithmetic, not topology.
 * The fan baseline is one for interior centres; boundary centres use their angular limit,
 * and exterior centres zero. This also makes the metric functions useful outside the window.
 * @internal
 */
export function getStudyWindowWGSL(options: StudyWindowWGSLOptions): string {
  const {layout} = options;
  const storage: PolygonEdgeSlabStorage = {
    params: {array: 'studyWindow', offset: `(studyWindowOffset + ${layout.slabParams}u)`},
    offsets: {array: 'studyWindow', offset: `(studyWindowOffset + ${layout.slabOffsets}u)`},
    entries: {array: 'studyWindow', offset: `(studyWindowOffset + ${layout.slabEntries}u)`}
  };
  const loop = (lowY: string, highY: string, body: string, dedupe = false) =>
    getPolygonEdgeSlabLoopWGSL({
      owner: '0u',
      lowY,
      highY,
      storage,
      ...(dedupe
        ? {dedupe: {edgeLowY: 'min(studyWindowEdge(edgeStart).y, studyWindowEdge(edgeStart).w)'}}
        : {}),
      body: `let edge = studyWindowEdge(edgeStart);
    if (!studyWindowFinitePoint(edge.xy) || !studyWindowFinitePoint(edge.zw)) { continue; }
${body}`
    });
  return /* wgsl */ `
${getPolygonEdgeSlabQueryWGSL(options, storage)}
const STUDY_WINDOW_TWO_PI: f32 = 6.283185307179586;
fn studyWindowSummaryValue(slot: u32) -> f32 {
  return bitcast<f32>(studyWindow[studyWindowOffset + ${layout.summary}u + slot]);
}
fn studyWindowEdge(vertex: u32) -> vec4f {
  let base = studyWindowOffset + ${layout.edges}u + 4u * vertex;
  return vec4f(bitcast<f32>(studyWindow[base]), bitcast<f32>(studyWindow[base + 1u]),
    bitcast<f32>(studyWindow[base + 2u]), bitcast<f32>(studyWindow[base + 3u]));
}
fn isStudyWindowUsable() -> bool {
  let area = studyWindowSummaryValue(4u);
  let flags = studyWindowSummaryValue(7u);
  return isFiniteBits(area) && area > 0.0 && isFiniteBits(flags) && (u32(flags) & 2u) == 0u;
}
fn studyWindowFinitePoint(point: vec2f) -> bool {
  return isFiniteBits(point.x) && isFiniteBits(point.y);
}
fn studyWindowCross(first: vec2f, second: vec2f) -> f32 {
  return first.x * second.y - first.y * second.x;
}
fn studyWindowSegmentDistance(point: vec2f, edge: vec4f) -> f32 {
  let direction = edge.zw - edge.xy;
  let squaredLength = dot(direction, direction);
  if (!(squaredLength > 0.0)) { return length(point - edge.xy); }
  let fraction = clamp(dot(point - edge.xy, direction) / squaredLength, 0.0, 1.0);
  return length(point - edge.xy - fraction * direction);
}
fn isInsideStudyWindow(point: vec2f) -> bool {
  if (!studyWindowFinitePoint(point) || !isStudyWindowUsable()) { return false; }
  if (point.x < studyWindowSummaryValue(0u) || point.y < studyWindowSummaryValue(1u) ||
      point.x > studyWindowSummaryValue(2u) || point.y > studyWindowSummaryValue(3u)) { return false; }
  let probe = ExactProbe(point, point, point, EXACT_PROBE_POINT);
  var parity = 0i;
  ${loop('point.y', 'point.y', 'parity ^= abs(exactProbeCrossing(probe, edge.xy, edge.zw).x);')}
  return (parity & 1i) != 0i;
}
fn getStudyWindowBoundaryDistance(point: vec2f, limit: f32) -> f32 {
  // Runtime bits: WGSL rejects non-finite constant expressions.
  var infinityBits = 0x7f800000u;
  var distance = bitcast<f32>(infinityBits);
  if (!studyWindowFinitePoint(point) || !(limit >= 0.0)) { return distance; }
  ${loop(
    'point.y - limit',
    'point.y + limit',
    `
    if (max(edge.x, edge.z) < point.x - limit || min(edge.x, edge.z) > point.x + limit) { continue; }
    let candidate = studyWindowSegmentDistance(point, edge);
    if (candidate <= limit) { distance = min(distance, candidate); }`
  )}
  return distance;
}
// Metric fan baseline. The ray test is only used for a floating-point integral, never membership.
// At a boundary point the full signed angular sum supplies the interior angular limit; zero-area
// fans contribute zero. The slower full walk is restricted to this exceptional case.
fn studyWindowFanBase(centre: vec2f) -> f32 {
  var parity = false;
  var boundary = false;
  ${loop(
    'centre.y',
    'centre.y',
    `
    let start = edge.xy - centre;
    let end = edge.zw - centre;
    let cross = studyWindowCross(start, end);
    if (cross == 0.0 && dot(start, end) <= 0.0 && any(edge.xy != edge.zw)) { boundary = true; }
    if ((edge.y > centre.y) != (edge.w > centre.y)) {
      if ((cross > 0.0) == (edge.w > edge.y)) { parity = !parity; }
    }`
  )}
  if (!boundary) { return select(0.0, 1.0, parity); }
  var angle = 0.0;
  for (var vertex = 0u; vertex < EDGE_SLAB_VERTEX_COUNT; vertex++) {
    let edge = studyWindowEdge(vertex);
    if (!studyWindowFinitePoint(edge.xy) || !studyWindowFinitePoint(edge.zw)) { continue; }
    let start = edge.xy - centre;
    let end = edge.zw - centre;
    let cross = studyWindowCross(start, end);
    if (isFiniteBits(cross) && cross != 0.0) { angle += atan2(cross, dot(start, end)); }
  }
  return clamp(angle / STUDY_WINDOW_TWO_PI, 0.0, 1.0);
}
// (line distance, signed along-edge start, signed along-edge end, fan orientation).
fn studyWindowFan(edge: vec4f, centre: vec2f) -> vec4f {
  let direction = edge.zw - edge.xy;
  let edgeLength = length(direction);
  if (!(edgeLength > 0.0)) { return vec4f(0.0); }
  let start = edge.xy - centre;
  let cross = studyWindowCross(start, direction);
  let along = dot(start, direction / edgeLength);
  return vec4f(abs(cross) / edgeLength, along, along + edgeLength, sign(cross));
}
fn getStudyWindowCircleFraction(centre: vec2f, radius: f32) -> f32 {
  if (!studyWindowFinitePoint(centre) || !(radius > 0.0) || !isFiniteBits(radius)) { return 0.0; }
  var correction = 0.0;
  ${loop(
    'centre.y - radius',
    'centre.y + radius',
    `
    let fan = studyWindowFan(edge, centre);
    if (!(fan.x > 0.0 && fan.x < radius)) { continue; }
    let ratio = fan.x / radius;
    let angle = atan2(sqrt(max(0.0, (1.0 - ratio) * (1.0 + ratio))), ratio);
    let low = max(atan2(fan.y, fan.x), -angle);
    let high = min(atan2(fan.z, fan.x), angle);
    correction += fan.w * max(0.0, high - low);`,
    true
  )}
  return clamp(studyWindowFanBase(centre) - correction / STUDY_WINDOW_TWO_PI, 0.0, 1.0);
}
${options.kernel ? getKernelIntegralWGSL(options.kernel) : ''}
${
  options.kernel
    ? /* wgsl */ `
fn getStudyWindowKernelMass(centre: vec2f, bandwidth: f32) -> f32 {
  if (!studyWindowFinitePoint(centre) || !(bandwidth > 0.0) || !isFiniteBits(bandwidth)) { return 0.0; }
  let radius = ${options.kernel === 'gaussian' ? '5.0' : '1.0'} * bandwidth;
  var correction = 0.0;
  ${loop(
    'centre.y - radius',
    'centre.y + radius',
    `
    let fan = studyWindowFan(edge, centre);
    if (!(fan.x > 0.0 && fan.x < radius)) { continue; }
    correction += fan.w * studyWindowKernelIntegral(fan.x / bandwidth, fan.y / bandwidth, fan.z / bandwidth);`,
    true
  )}
  return clamp(studyWindowFanBase(centre) - correction / STUDY_WINDOW_TWO_PI, 0.0, 1.0);
}`
    : ''
}
${options.sampleGrid ? getStudyWindowSampleCellWGSL(options.sampleGrid, 'studyWindowSummaryValue') : ''}
`;
}

function getKernelIntegralWGSL(kernel: StudyWindowKernel): string {
  if (kernel !== 'gaussian') {
    return /* wgsl */ `
fn studyWindowKernelPolynomial(value: f32, squaredRatio: f32) -> f32 {
  ${
    kernel === 'epanechnikov'
      ? 'return (-2.0 + squaredRatio) * value + value * value * value / 3.0;'
      : `return (-3.0 + 3.0 * squaredRatio - squaredRatio * squaredRatio) * value +
    (1.0 - 2.0 * squaredRatio / 3.0) * value * value * value - value * value * value * value * value / 5.0;`
  }
}
fn studyWindowKernelIntegral(ratio: f32, start: f32, end: f32) -> f32 {
  let squaredRatio = ratio * ratio;
  let extent = sqrt(max(0.0, 1.0 - squaredRatio));
  let low = max(start, -extent);
  let high = min(end, extent);
  if (!(high > low)) { return 0.0; }
  // x = tan(t), evaluated as ratio*x = along/h to avoid huge powers near pi/2.
  return atan2(high, ratio) - atan2(low, ratio) + ratio *
    (studyWindowKernelPolynomial(high, squaredRatio) - studyWindowKernelPolynomial(low, squaredRatio));
}
`;
  }
  return /* wgsl */ `
// 16-point Gauss-Legendre on half-bandwidth panels, split additionally at x=p/h.
// Integrate the complement in x=(p/h)*tan(t): subtraction removes the angular boundary layer
// at tiny p/h. Only rho/h <= 5 is retained (Gaussian tail <= exp(-12.5)).
fn studyWindowGaussianCorrection(ratio: f32, low: f32, high: f32) -> f32 {
  var nodes = array<f32, 8>(0.09501250983763744, 0.2816035507792589, 0.4580167776572274,
    0.6178762444026438, 0.755404408355003, 0.8656312023878318, 0.9445750230732326, 0.9894009349916499);
  var weights = array<f32, 8>(0.1894506104550685, 0.1826034150449236, 0.16915651939500254,
    0.14959598881657673, 0.12462897125553387, 0.09515851168249278, 0.06225352393864789, 0.027152459411754096);
  var total = 0.0;
  for (var panel = 0u; panel < 10u; panel++) {
    let begin = max(low, f32(panel) * 0.5);
    let end = min(high, f32(panel + 1u) * 0.5);
    if (!(end > begin)) { continue; }
    let split = clamp(ratio, begin, end);
    for (var part = 0u; part < 2u; part++) {
      let left = select(begin, split, part == 1u);
      let right = select(split, end, part == 1u);
      if (!(right > left)) { continue; }
      let middle = (left + right) * 0.5;
      let half = (right - left) * 0.5;
      for (var node = 0u; node < 8u; node++) {
        for (var side = 0u; side < 2u; side++) {
          let value = middle + select(-1.0, 1.0, side == 1u) * half * nodes[node];
          let squaredValue = value * value;
          total += half * weights[node] * (1.0 - exp(-squaredValue * 0.5)) * ratio / (ratio * ratio + squaredValue);
        }
      }
    }
  }
  return total;
}
fn studyWindowKernelIntegral(ratio: f32, start: f32, end: f32) -> f32 {
  let extent = sqrt(max(0.0, 25.0 - ratio * ratio));
  let low = max(start, -extent);
  let high = min(end, extent);
  if (!(high > low)) { return 0.0; }
  let correction = studyWindowGaussianCorrection(ratio, max(0.0, low), max(0.0, high)) +
    studyWindowGaussianCorrection(ratio, max(0.0, -high), max(0.0, -low));
  return exp(-ratio * ratio * 0.5) * (atan2(high, ratio) - atan2(low, ratio) - correction);
}
`;
}
