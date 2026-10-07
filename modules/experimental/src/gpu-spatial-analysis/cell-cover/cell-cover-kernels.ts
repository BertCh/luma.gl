// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {dggs} from '@luma.gl/shadertools';
import {CELL_KEY_WGSL, QUADBIN_TILE_WGSL, getCellKeyLayout} from '../cell-aggregation/cell-keys';
import {H3_INDEX_WGSL} from '../cell-indexing/h3-index-wgsl';
import {H3_BOUNDARY_WGSL} from '../cell-indexing/h3-boundary-wgsl';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {
  CELL_COVER_H3_LATTICE_SPACING_DEGREES,
  CELL_COVER_H3_MAXIMUM_LATITUDE
} from './cell-cover-h3-constants';

const OPERATION = 'GPUCellCover';

/** Containment modes in shader constant order. @internal */
export const CELL_COVER_MODE_CODES = {center: 0, full: 1, intersects: 2} as const;

/**
 * Bit of a candidate's high key word that carries the core flag between the test and write
 * kernels. Quadbin and H3 keys leave bit 63 reserved (zero), so it never collides with the key.
 * @internal
 */
export const CELL_COVER_CORE_BIT = 0x80000000;

/** Absolute slack in degrees added around a cell before the core test (f32 tile edge error). @internal */
export const CELL_COVER_CORE_MARGIN_DEGREES = 2e-4;

/**
 * Number of `uint32` words per feature in the candidate range table. @internal
 *
 * Words 0 to 4 hold the candidate lattice (family specific). With edge slabs, words 5 to 10 hold
 * the slab count (0 when the feature is not slabbed), the first slab's index, the slab origin and
 * scale (f32 bits) and the feature's first and last vertex.
 */
export const CELL_COVER_RANGE_STRIDE = 12;

/** Features with fewer candidates than this skip the edge slabs (the build would cost more). @internal */
export const CELL_COVER_MINIMUM_SLAB_CANDIDATES = 16;

/** Most slabs of one feature. @internal */
export const CELL_COVER_MAXIMUM_SLABS = 4096;

/** Slab and entry capacities of the optional edge-slab index. @internal */
export type CellCoverEdgeSlabs = {
  /** Slab slots over all features; the slab count of a feature is at most its vertex count / 4. */
  slabCapacity: number;
  /** Edge-slab entries (one per edge and slab it spans) over all features. */
  entryCapacity: number;
};

/** Common description of a cover kernel family. @internal */
export type CellCoverKernelContext = {
  id: string;
  family: 'quadbin' | 'h3';
  resolution: number;
  containment: keyof typeof CELL_COVER_MODE_CODES;
  /** Whether the test kernel also decides the core flag (bit 31 of the candidate's high word). */
  computeCore: boolean;
  featureCount: number;
  candidateCapacity: number;
  /** Capacities of the edge-slab index, or `undefined` to test every edge of the feature. */
  edgeSlabs?: CellCoverEdgeSlabs;
  polygonPositions: GraphDataView<'float32x2'>;
  featureOffsets: GraphDataView<'uint32'>;
  polygonOffsets: GraphDataView<'uint32'>;
  ringOffsets: GraphDataView<'uint32'>;
};

function getPolygonBindings(context: CellCoverKernelContext): WGSLKernelBinding[] {
  return [
    {name: 'polygonPositions', view: context.polygonPositions, type: 'f32', access: 'read'},
    {name: 'featureOffsets', view: context.featureOffsets, type: 'u32', access: 'read'},
    {name: 'polygonOffsets', view: context.polygonOffsets, type: 'u32', access: 'read'},
    {name: 'ringOffsets', view: context.ringOffsets, type: 'u32', access: 'read'}
  ];
}

/**
 * WGSL polygon helpers over the bindings `polygonPositions` (f32 pairs), `featureOffsets`,
 * `polygonOffsets` and `ringOffsets`. Predicates use plain f32 operations in a fixed order so the
 * CPU oracle can mirror them with `Math.fround`.
 *
 * - `coverGetFeatureInfo(feature)`: bounds `(minX, minY, maxX, maxY)` over finite vertices (`minX > maxX` when
 *   none), vertex count, first and last vertex and whether any vertex is non-finite.
 * - `coverContains(feature, p)`: even-odd containment over every ring of every polygon.
 * - `coverEdgesHitRect(feature, lo, hi)`: whether any ring edge meets the OPEN rectangle `(lo, hi)`.
 */
const COMMON_WGSL = /* wgsl */ `
fn coverGetVertex(vertex: u32) -> vec2f {
  let rangeBase = polygonPositionsOffset + 2u * vertex;
  return vec2f(polygonPositions[rangeBase], polygonPositions[rangeBase + 1u]);
}

fn coverIsFinite(p: vec2f) -> bool {
  return p.x == p.x && p.y == p.y && abs(p.x) < 1e30 && abs(p.y) < 1e30;
}

/** Whether the closed segment a-b meets the open rectangle (lo, hi). */
fn coverSegmentHitsRect(a: vec2f, b: vec2f, lo: vec2f, hi: vec2f) -> bool {
  var lower = -1e30;
  var upper = 1e30;
  let d = b - a;
  if (d.x == 0.0) {
    if (!(lo.x < a.x && a.x < hi.x)) { return false; }
  } else {
    let t1 = (lo.x - a.x) / d.x;
    let t2 = (hi.x - a.x) / d.x;
    lower = max(lower, min(t1, t2));
    upper = min(upper, max(t1, t2));
  }
  if (d.y == 0.0) {
    if (!(lo.y < a.y && a.y < hi.y)) { return false; }
  } else {
    let t1 = (lo.y - a.y) / d.y;
    let t2 = (hi.y - a.y) / d.y;
    lower = max(lower, min(t1, t2));
    upper = min(upper, max(t1, t2));
  }
  return lower < upper && lower < 1.0 && upper > 0.0;
}
`;

const POLYGON_WGSL = /* wgsl */ `${COMMON_WGSL}
struct CoverFeatureInfo {
  bounds: vec4f,
  vertexCount: u32,
  vertexBegin: u32,
  vertexEnd: u32,
  hasNonFinite: bool
}

fn coverGetFeatureInfo(feature: u32) -> CoverFeatureInfo {
  var info = CoverFeatureInfo(vec4f(1e30, 1e30, -1e30, -1e30), 0u, 0u, 0u, false);
  var isFirstRing = true;
  let firstPolygon = featureOffsets[featureOffsetsOffset + feature];
  let endPolygon = featureOffsets[featureOffsetsOffset + feature + 1u];
  for (var polygon = firstPolygon; polygon < endPolygon; polygon++) {
    let firstRing = polygonOffsets[polygonOffsetsOffset + polygon];
    let endRing = polygonOffsets[polygonOffsetsOffset + polygon + 1u];
    for (var ring = firstRing; ring < endRing; ring++) {
      let firstVertex = ringOffsets[ringOffsetsOffset + ring];
      let endVertex = ringOffsets[ringOffsetsOffset + ring + 1u];
      if (isFirstRing) {
        info.vertexBegin = firstVertex;
        isFirstRing = false;
      }
      info.vertexEnd = endVertex;
      if (endVertex > firstVertex) {
        info.vertexCount += endVertex - firstVertex;
      }
      for (var vertex = firstVertex; vertex < endVertex; vertex++) {
        let p = coverGetVertex(vertex);
        if (coverIsFinite(p)) {
          info.bounds = vec4f(min(info.bounds.xy, p), max(info.bounds.zw, p));
        } else {
          info.hasNonFinite = true;
        }
      }
    }
  }
  return info;
}

fn coverContains(feature: u32, p: vec2f) -> bool {
  var inside = false;
  let firstPolygon = featureOffsets[featureOffsetsOffset + feature];
  let endPolygon = featureOffsets[featureOffsetsOffset + feature + 1u];
  for (var polygon = firstPolygon; polygon < endPolygon; polygon++) {
    let firstRing = polygonOffsets[polygonOffsetsOffset + polygon];
    let endRing = polygonOffsets[polygonOffsetsOffset + polygon + 1u];
    for (var ring = firstRing; ring < endRing; ring++) {
      let firstVertex = ringOffsets[ringOffsetsOffset + ring];
      let endVertex = ringOffsets[ringOffsetsOffset + ring + 1u];
      for (var vertex = firstVertex; vertex < endVertex; vertex++) {
        let a = coverGetVertex(vertex);
        let b = coverGetVertex(select(vertex + 1u, firstVertex, vertex + 1u == endVertex));
        if ((a.y > p.y) != (b.y > p.y)) {
          if (p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x) {
            inside = !inside;
          }
        }
      }
    }
  }
  return inside;
}

fn coverEdgesHitRect(feature: u32, lo: vec2f, hi: vec2f) -> bool {
  let firstPolygon = featureOffsets[featureOffsetsOffset + feature];
  let endPolygon = featureOffsets[featureOffsetsOffset + feature + 1u];
  for (var polygon = firstPolygon; polygon < endPolygon; polygon++) {
    let firstRing = polygonOffsets[polygonOffsetsOffset + polygon];
    let endRing = polygonOffsets[polygonOffsetsOffset + polygon + 1u];
    for (var ring = firstRing; ring < endRing; ring++) {
      let firstVertex = ringOffsets[ringOffsetsOffset + ring];
      let endVertex = ringOffsets[ringOffsetsOffset + ring + 1u];
      for (var vertex = firstVertex; vertex < endVertex; vertex++) {
        let a = coverGetVertex(vertex);
        let b = coverGetVertex(select(vertex + 1u, firstVertex, vertex + 1u == endVertex));
        if (coverSegmentHitsRect(a, b, lo, hi)) { return true; }
      }
    }
  }
  return false;
}
`;

/**
 * Edge-slab helpers. Each slabbed feature splits its y-range into `n` equal slabs; an edge is listed
 * in every slab from `slabOf(min y)` to `slabOf(max y)`. `slabOf` is a monotone f32 expression
 * (subtract, multiply, floor, clamp), so for `a.y <= p.y <= b.y` the slab of `p.y` lies between the
 * slabs of the two endpoints: every edge that straddles or touches a query height is listed in the
 * query's slab, and the predicates then see exactly the edges the brute-force walk would have
 * accepted. Needs `ranges` and `RANGE_STRIDE`.
 */
const SLAB_COMMON_WGSL = /* wgsl */ `
const NO_VERTEX: u32 = 0xffffffffu;
const SLAB_COUNT_WORD: u32 = 5u;
const SLAB_BASE_WORD: u32 = 6u;
const SLAB_ORIGIN_WORD: u32 = 7u;
const SLAB_SCALE_WORD: u32 = 8u;
const VERTEX_BEGIN_WORD: u32 = 9u;
const VERTEX_END_WORD: u32 = 10u;

fn coverSlabOf(y: f32, slabCount: u32, origin: f32, scale: f32) -> u32 {
  if (slabCount <= 1u) { return 0u; }
  return u32(clamp(floor((y - origin) * scale), 0.0, f32(slabCount - 1u)));
}
`;

/**
 * Edge-slab versions of `coverContains` and `coverEdgesHitRect` with the signatures of the
 * brute-force ones. A feature without slabs (too few candidates or vertices, non-finite vertices,
 * or entries beyond the entry capacity) walks its vertex range `[vertexBegin, vertexEnd)` through
 * `nextVertex`, the same edges as the ring walk. Needs `polygonPositions`, `ranges`, `slabOffsets`,
 * `slabEntries` and `nextVertex`.
 */
function getSlabQueryWGSL(entryCapacity: number): string {
  return /* wgsl */ `${COMMON_WGSL}${SLAB_COMMON_WGSL}
const ENTRY_CAPACITY: u32 = ${entryCapacity}u;

fn coverIsSlabbed(rangeBase: u32) -> bool {
  let slabCount = ranges[rangeBase + SLAB_COUNT_WORD];
  return slabCount != 0u &&
    slabOffsets[slabOffsetsOffset + ranges[rangeBase + SLAB_BASE_WORD] + slabCount] <= ENTRY_CAPACITY;
}

fn coverSlabOfHeight(rangeBase: u32, y: f32) -> u32 {
  return coverSlabOf(
    y,
    ranges[rangeBase + SLAB_COUNT_WORD],
    bitcast<f32>(ranges[rangeBase + SLAB_ORIGIN_WORD]),
    bitcast<f32>(ranges[rangeBase + SLAB_SCALE_WORD])
  );
}

fn coverCrosses(a: vec2f, b: vec2f, p: vec2f) -> bool {
  if ((a.y > p.y) != (b.y > p.y)) {
    return p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x;
  }
  return false;
}

fn coverContains(feature: u32, p: vec2f) -> bool {
  let rangeBase = rangesOffset + feature * RANGE_STRIDE;
  var inside = false;
  if (coverIsSlabbed(rangeBase)) {
    let slabIndex = ranges[rangeBase + SLAB_BASE_WORD] + coverSlabOfHeight(rangeBase, p.y);
    let end = slabOffsets[slabOffsetsOffset + slabIndex + 1u];
    for (var entry = slabOffsets[slabOffsetsOffset + slabIndex]; entry < end; entry++) {
      let a = coverGetVertex(slabEntries[slabEntriesOffset + 2u * entry]);
      let b = coverGetVertex(slabEntries[slabEntriesOffset + 2u * entry + 1u]);
      if (coverCrosses(a, b, p)) {
        inside = !inside;
      }
    }
  } else {
    for (var vertex = ranges[rangeBase + VERTEX_BEGIN_WORD]; vertex < ranges[rangeBase + VERTEX_END_WORD]; vertex++) {
      let next = nextVertex[nextVertexOffset + vertex];
      if (next != NO_VERTEX && coverCrosses(coverGetVertex(vertex), coverGetVertex(next), p)) {
        inside = !inside;
      }
    }
  }
  return inside;
}

fn coverEdgesHitRect(feature: u32, lo: vec2f, hi: vec2f) -> bool {
  let rangeBase = rangesOffset + feature * RANGE_STRIDE;
  if (coverIsSlabbed(rangeBase)) {
    let base = ranges[rangeBase + SLAB_BASE_WORD];
    let lastSlab = coverSlabOfHeight(rangeBase, hi.y);
    for (var slab = coverSlabOfHeight(rangeBase, lo.y); slab <= lastSlab; slab++) {
      let end = slabOffsets[slabOffsetsOffset + base + slab + 1u];
      for (var entry = slabOffsets[slabOffsetsOffset + base + slab]; entry < end; entry++) {
        let a = coverGetVertex(slabEntries[slabEntriesOffset + 2u * entry]);
        let b = coverGetVertex(slabEntries[slabEntriesOffset + 2u * entry + 1u]);
        if (coverSegmentHitsRect(a, b, lo, hi)) { return true; }
      }
    }
  } else {
    for (var vertex = ranges[rangeBase + VERTEX_BEGIN_WORD]; vertex < ranges[rangeBase + VERTEX_END_WORD]; vertex++) {
      let next = nextVertex[nextVertexOffset + vertex];
      if (next != NO_VERTEX && coverSegmentHitsRect(coverGetVertex(vertex), coverGetVertex(next), lo, hi)) {
        return true;
      }
    }
  }
  return false;
}
`;
}

/** Feature lookup by candidate index over the exclusive scan `starts` (length featureCount + 1). */
function getFindFeatureWGSL(featureCount: number): string {
  return /* wgsl */ `
const FEATURE_COUNT: u32 = ${featureCount}u;

/** Feature whose candidate range [starts[f], starts[f + 1]) contains \`candidate\`. */
fn coverFindFeature(candidate: u32) -> u32 {
  var low = 0u;
  var high = FEATURE_COUNT;
  while (low < high) {
    let middle = (low + high) >> 1u;
    if (starts[startsOffset + middle + 1u] > candidate) {
      high = middle;
    } else {
      low = middle + 1u;
    }
  }
  return min(low, FEATURE_COUNT - 1u);
}
`;
}

function getQuadbinConstantsWGSL(context: CellCoverKernelContext): string {
  const resolution = context.resolution;
  return /* wgsl */ `
const TILE_WIDTH: f32 = ${getWGSLFloatLiteral(45 * 2 ** (3 - resolution))};
const ROW_SCALE: f32 = ${getWGSLFloatLiteral(2 ** (1 - resolution))};
const PI: f32 = 3.14159265358979;
`;
}

/**
 * Tail of the count kernel with edge slabs: picks the slab count (about one slab per four edges,
 * capped, and none for features with few candidates, few edges, non-finite vertices or no height),
 * and records the slab lattice and the feature's vertex range for the later passes.
 */
const SLAB_PARAMETERS_BODY = `
  var slabCount = 0u;
  var slabScale = 0.0;
  if (count >= ${CELL_COVER_MINIMUM_SLAB_CANDIDATES}u && !info.hasNonFinite && info.vertexCount >= 8u) {
    let wanted = min(info.vertexCount / 4u, ${CELL_COVER_MAXIMUM_SLABS}u);
    let scale = f32(wanted) / (bounds.w - bounds.y);
    if (scale > 0.0 && scale < 1e30) {
      slabCount = wanted;
      slabScale = scale;
    }
  }
  ranges[rangeBase + SLAB_COUNT_WORD] = slabCount;
  ranges[rangeBase + SLAB_ORIGIN_WORD] = bitcast<u32>(bounds.y);
  ranges[rangeBase + SLAB_SCALE_WORD] = bitcast<u32>(slabScale);
  ranges[rangeBase + VERTEX_BEGIN_WORD] = info.vertexBegin;
  ranges[rangeBase + VERTEX_END_WORD] = info.vertexEnd;
  slabNumbers[slabNumbersOffset + index] = slabCount;`;

/** Candidate counting node: one thread per feature (plus one trailing zero count). @internal */
export function createCoverCountNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  context: CellCoverKernelContext,
  counts: GraphDataView<'uint32'>,
  ranges: GraphDataView<'uint32'>,
  slabNumbers?: GraphDataView<'uint32'>
): GPUCommandNode<Parameters> {
  const {resolution, candidateCapacity, family} = context;
  const capacityPlusOne = candidateCapacity + 1;
  let declarations = `${POLYGON_WGSL}
const CAPACITY_PLUS_ONE: u32 = ${capacityPlusOne}u;
const RANGE_STRIDE: u32 = ${CELL_COVER_RANGE_STRIDE}u;
fn coverClampCount(width: u32, height: u32) -> u32 {
  if (width == 0u || height == 0u) { return 0u; }
  if (width > CAPACITY_PLUS_ONE || height > CAPACITY_PLUS_ONE) { return CAPACITY_PLUS_ONE; }
  if (width > CAPACITY_PLUS_ONE / height) { return CAPACITY_PLUS_ONE; }
  return width * height;
}
`;
  let body: string;
  if (family === 'quadbin') {
    declarations += `${CELL_KEY_WGSL}${QUADBIN_TILE_WGSL}const RESOLUTION: u32 = ${resolution}u;`;
    body = `if (index == FEATURE_COUNT) { counts[countsOffset + index] = 0u;${slabNumbers ? ' slabNumbers[slabNumbersOffset + index] = 0u;' : ''} return; }
  let info = coverGetFeatureInfo(index);
  let bounds = info.bounds;
  let rangeBase = rangesOffset + index * RANGE_STRIDE;
  var count = 0u;
  if (bounds.x <= bounds.z && bounds.y <= bounds.w) {
    let x0 = quadbinGetTileX(bounds.x, RESOLUTION);
    let x1 = quadbinGetTileX(bounds.z, RESOLUTION);
    let y0 = quadbinGetTileY(bounds.w, RESOLUTION);
    let y1 = quadbinGetTileY(bounds.y, RESOLUTION);
    if (x1 >= x0 && y1 >= y0) {
      ranges[rangeBase] = x0;
      ranges[rangeBase + 1u] = y0;
      ranges[rangeBase + 2u] = x1 - x0 + 1u;
      ranges[rangeBase + 3u] = y1 - y0 + 1u;
      count = coverClampCount(x1 - x0 + 1u, y1 - y0 + 1u);
    }
  }
  counts[countsOffset + index] = count;${slabNumbers ? SLAB_PARAMETERS_BODY : ''}`;
  } else {
    const spacing = CELL_COVER_H3_LATTICE_SPACING_DEGREES[resolution];
    declarations += `const LATTICE_SPACING: f32 = ${getWGSLFloatLiteral(spacing)};
const MAXIMUM_LATITUDE: f32 = ${getWGSLFloatLiteral(CELL_COVER_H3_MAXIMUM_LATITUDE)};
const DEGREES_TO_RADIANS: f32 = 0.017453292519943295;
fn coverToCount(value: f32) -> u32 {
  return u32(clamp(value, 1.0, 4.0e9));
}`;
    body = `if (index == FEATURE_COUNT) { counts[countsOffset + index] = 0u;${slabNumbers ? ' slabNumbers[slabNumbersOffset + index] = 0u;' : ''} return; }
  let info = coverGetFeatureInfo(index);
  let bounds = info.bounds;
  let rangeBase = rangesOffset + index * RANGE_STRIDE;
  var count = 0u;
  let originLat = max(bounds.y - LATTICE_SPACING, -MAXIMUM_LATITUDE);
  let endLat = min(bounds.w + LATTICE_SPACING, MAXIMUM_LATITUDE);
  if (bounds.x <= bounds.z && bounds.y <= bounds.w && originLat <= endLat) {
    let spacingLng = LATTICE_SPACING / cos(max(abs(originLat), abs(endLat)) * DEGREES_TO_RADIANS);
    let originLng = bounds.x - spacingLng;
    let width = coverToCount(ceil((bounds.z + spacingLng - originLng) / spacingLng) + 1.0);
    let height = coverToCount(ceil((endLat - originLat) / LATTICE_SPACING) + 1.0);
    ranges[rangeBase] = width;
    ranges[rangeBase + 1u] = height;
    ranges[rangeBase + 2u] = bitcast<u32>(originLng);
    ranges[rangeBase + 3u] = bitcast<u32>(originLat);
    ranges[rangeBase + 4u] = bitcast<u32>(spacingLng);
    count = coverClampCount(width, height);
  }
  counts[countsOffset + index] = count;${slabNumbers ? SLAB_PARAMETERS_BODY : ''}`;
  }
  declarations += `const FEATURE_COUNT: u32 = ${context.featureCount}u;`;
  if (slabNumbers) {
    declarations += SLAB_COMMON_WGSL;
  }
  return createWGSLKernelNode<Parameters>(graph, {
    id: `${context.id}-count`,
    operation: OPERATION,
    variant: slabNumbers ? 'count-slabs' : 'count',
    bindings: [
      ...getPolygonBindings(context),
      {name: 'counts', view: counts, type: 'u32', access: 'read_write'},
      {name: 'ranges', view: ranges, type: 'u32', access: 'read_write'},
      ...(slabNumbers
        ? [{name: 'slabNumbers', view: slabNumbers, type: 'u32', access: 'read_write'} as const]
        : [])
    ],
    invocationCount: context.featureCount + 1,
    declarations,
    body
  });
}

/**
 * Candidate test node: one thread per candidate slot. Writes the accept flag and the candidate's
 * cell key (little-endian words) into `candidateCells`. @internal
 */
export function createCoverTestNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  context: CellCoverKernelContext,
  views: {
    ranges: GraphDataView<'uint32'>;
    starts: GraphDataView<'uint32'>;
    flags: GraphDataView<'uint32'>;
    candidateCells: GraphDataView<'uint32x2'>;
    /** Edge-slab index views, present exactly when `context.edgeSlabs` is. */
    slabOffsets?: GraphDataView<'uint32'>;
    slabEntries?: GraphDataView<'uint32x2'>;
    nextVertex?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const {family, resolution, candidateCapacity, edgeSlabs} = context;
  const mode = CELL_COVER_MODE_CODES[context.containment];
  const slabBindings: WGSLKernelBinding[] =
    edgeSlabs && views.slabOffsets && views.slabEntries && views.nextVertex
      ? [
          {name: 'slabOffsets', view: views.slabOffsets, type: 'u32', access: 'read'},
          {name: 'slabEntries', view: views.slabEntries, type: 'u32', access: 'read'},
          {name: 'nextVertex', view: views.nextVertex, type: 'u32', access: 'read'}
        ]
      : [];
  let declarations = `${edgeSlabs ? getSlabQueryWGSL(edgeSlabs.entryCapacity) : POLYGON_WGSL}${getFindFeatureWGSL(context.featureCount)}
const CANDIDATE_CAPACITY: u32 = ${candidateCapacity}u;
const RANGE_STRIDE: u32 = ${CELL_COVER_RANGE_STRIDE}u;
const MODE: u32 = ${mode}u;
const RESOLUTION: u32 = ${resolution}u;
const CORE_BIT: u32 = ${CELL_COVER_CORE_BIT}u;
const CORE_MARGIN: f32 = ${getWGSLFloatLiteral(CELL_COVER_CORE_MARGIN_DEGREES)};
`;
  let evaluation: string;
  if (family === 'quadbin') {
    const layout = getCellKeyLayout('quadbin', resolution);
    declarations += `${getQuadbinConstantsWGSL(context)}${CELL_KEY_WGSL}${QUADBIN_TILE_WGSL}
fn coverTileLatitude(row: f32) -> f32 {
  let t = PI * (1.0 - row * ROW_SCALE);
  return (2.0 * atan(exp(t)) - PI * 0.5) * 57.29577951308232;
}
`;
    evaluation = `let width = ranges[rangeBase + 2u];
  let x = ranges[rangeBase] + candidateLocal % width;
  let y = ranges[rangeBase + 1u] + candidateLocal / width;
  let lng0 = f32(x) * TILE_WIDTH - 180.0;
  let lng1 = f32(x + 1u) * TILE_WIDTH - 180.0;
  let latNorth = coverTileLatitude(f32(y));
  let latSouth = coverTileLatitude(f32(y + 1u));
  let centre = vec2f((f32(x) + 0.5) * TILE_WIDTH - 180.0, coverTileLatitude(f32(y) + 0.5));
  var accepted = false;
  if (MODE == 0u) {
    accepted = coverContains(feature, centre);
  } else {
    let hit = coverEdgesHitRect(feature, vec2f(lng0, latSouth), vec2f(lng1, latNorth));
    if (MODE == 1u) {
      accepted = !hit && coverContains(feature, centre);
    } else {
      accepted = hit || coverContains(feature, centre);
    }
  }
  var key = vec2u(0u);
  if (accepted) {
    key = cellGetKey(quadbinGetCompactKey(x, y), ${layout.headerHigh}u, RESOLUTION, ${layout.lowBit}u);
    ${
      context.computeCore
        ? `// Core: the centre is inside and no polygon edge enters the cell grown by a safety margin.
    let slack = vec2f(CORE_MARGIN + 0.001 * (lng1 - lng0));
    if (coverContains(feature, centre) && !coverEdgesHitRect(feature, vec2f(lng0, latSouth) - slack, vec2f(lng1, latNorth) + slack)) {
      key.x = key.x | CORE_BIT;
    }`
        : ''
    }
  }`;
  } else {
    declarations += `${dggs.source}\n${H3_INDEX_WGSL}\n${context.computeCore ? H3_BOUNDARY_WGSL : ''}`;
    evaluation = `let width = ranges[rangeBase];
  let originLng = bitcast<f32>(ranges[rangeBase + 2u]);
  let originLat = bitcast<f32>(ranges[rangeBase + 3u]);
  let spacingLng = bitcast<f32>(ranges[rangeBase + 4u]);
  let spacingLat = ${getWGSLFloatLiteral(CELL_COVER_H3_LATTICE_SPACING_DEGREES[resolution])};
  let column = candidateLocal % width;
  let row = candidateLocal / width;
  let point = vec2f(originLng + f32(column) * spacingLng, originLat + f32(row) * spacingLat);
  var key = vec2u(0u);
  let cell = cellIndexH3FromLngLat(point, RESOLUTION);
  if (cell.x != 0u || cell.y != 0u) {
    let centre = dggs_h3_get_center_lnglat(cell);
    // Shift the center by whole turns next to the lattice, then round it to the nearest lattice
    // index with one floating point expression of the center alone, so exactly one lattice point
    // claims each cell (no tie ambiguity between neighbouring points).
    let middleLng = originLng + 0.5 * f32(width) * spacingLng;
    let centreLng = centre.x + 360.0 * round((middleLng - centre.x) / 360.0);
    let nearestColumn = floor((centreLng - originLng) / spacingLng + 0.5);
    let nearestRow = floor((centre.y - originLat) / spacingLat + 0.5);
    if (nearestColumn == f32(column) && nearestRow == f32(row) && coverContains(feature, centre)) {
      key = cell;
      ${
        context.computeCore
          ? `// Core: no polygon edge enters the bounding box of the cell boundary, grown by a margin.
      let boundary = cellIndexH3GetBoundary(cell);
      var lo = boundary.points[0];
      var hi = boundary.points[0];
      for (var vertex = 1u; vertex < boundary.count; vertex++) {
        lo = min(lo, boundary.points[vertex]);
        hi = max(hi, boundary.points[vertex]);
      }
      let slack = vec2f(CORE_MARGIN) + 0.001 * (hi - lo);
      if (boundary.count > 0u && hi.x - lo.x < 90.0 && !coverEdgesHitRect(feature, lo - slack, hi + slack)) {
        key.x = key.x | CORE_BIT;
      }`
          : ''
      }
    }
  }
  let accepted = key.x != 0u || key.y != 0u;`;
  }
  return createWGSLKernelNode<Parameters>(graph, {
    id: `${context.id}-test`,
    operation: OPERATION,
    variant: edgeSlabs ? 'test-slabs' : 'test',
    bindings: [
      ...(edgeSlabs
        ? [
            {
              name: 'polygonPositions',
              view: context.polygonPositions,
              type: 'f32',
              access: 'read'
            } as WGSLKernelBinding
          ]
        : getPolygonBindings(context)),
      {name: 'ranges', view: views.ranges, type: 'u32', access: 'read'},
      {name: 'starts', view: views.starts, type: 'u32', access: 'read'},
      {name: 'flags', view: views.flags, type: 'u32', access: 'read_write'},
      {name: 'candidateCells', view: views.candidateCells, type: 'u32', access: 'read_write'},
      ...slabBindings
    ],
    invocationCount: candidateCapacity,
    declarations,
    body: `let activeCount = min(starts[startsOffset + FEATURE_COUNT], CANDIDATE_CAPACITY);
  if (index >= activeCount) {
    flags[flagsOffset + index] = 0u;
    return;
  }
  let feature = coverFindFeature(index);
  let candidateLocal = index - starts[startsOffset + feature];
  let rangeBase = rangesOffset + feature * RANGE_STRIDE;
  ${evaluation}
  flags[flagsOffset + index] = select(0u, 1u, accepted);
  candidateCells[candidateCellsOffset + 2u * index] = key.y;
  candidateCells[candidateCellsOffset + 2u * index + 1u] = key.x;`
  });
}

/**
 * Writes each slabbed feature's first slab index into the range table and drops the slabs of
 * features that no longer fit the slab capacity (they fall back to the vertex-range walk).
 * @internal
 */
export function createCoverSlabBasesNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  context: CellCoverKernelContext,
  views: {ranges: GraphDataView<'uint32'>; slabStarts: GraphDataView<'uint32'>}
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: `${context.id}-slab-bases`,
    operation: OPERATION,
    variant: 'slab-bases',
    bindings: [
      {name: 'ranges', view: views.ranges, type: 'u32', access: 'read_write'},
      {name: 'slabStarts', view: views.slabStarts, type: 'u32', access: 'read'}
    ],
    invocationCount: context.featureCount,
    declarations: `const RANGE_STRIDE: u32 = ${CELL_COVER_RANGE_STRIDE}u;
const SLAB_CAPACITY: u32 = ${context.edgeSlabs!.slabCapacity}u;${SLAB_COMMON_WGSL}`,
    body: `let rangeBase = rangesOffset + index * RANGE_STRIDE;
  let base = slabStarts[slabStartsOffset + index];
  if (ranges[rangeBase + SLAB_COUNT_WORD] != 0u && base + ranges[rangeBase + SLAB_COUNT_WORD] > SLAB_CAPACITY) {
    ranges[rangeBase + SLAB_COUNT_WORD] = 0u;
  }
  ranges[rangeBase + SLAB_BASE_WORD] = base;`
  });
}

/**
 * One thread per ring: the feature that owns the ring (`featureOffsets` over `polygonOffsets`
 * over `ringOffsets`, by binary search), or `0xffffffff` for rings outside every feature.
 * @internal
 */
export function createCoverRingFeatureNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  context: CellCoverKernelContext,
  ringFeature: GraphDataView<'uint32'>
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: `${context.id}-ring-feature`,
    operation: OPERATION,
    variant: 'ring-feature',
    bindings: [
      {name: 'featureOffsets', view: context.featureOffsets, type: 'u32', access: 'read'},
      {name: 'polygonOffsets', view: context.polygonOffsets, type: 'u32', access: 'read'},
      {name: 'ringFeature', view: ringFeature, type: 'u32', access: 'read_write'}
    ],
    invocationCount: context.ringOffsets.length - 1,
    declarations: `const FEATURE_COUNT: u32 = ${context.featureCount}u;
const POLYGON_COUNT: u32 = ${context.polygonOffsets.length - 1}u;
const NO_FEATURE: u32 = 0xffffffffu;`,
    body: `var polygon = 0u;
  var high = POLYGON_COUNT;
  while (polygon + 1u < high) {
    let middle = (polygon + high) / 2u;
    if (polygonOffsets[polygonOffsetsOffset + middle] <= index) {
      polygon = middle;
    } else {
      high = middle;
    }
  }
  var feature = NO_FEATURE;
  if (POLYGON_COUNT > 0u && index >= polygonOffsets[polygonOffsetsOffset + polygon] &&
      index < polygonOffsets[polygonOffsetsOffset + polygon + 1u]) {
    var low = 0u;
    high = FEATURE_COUNT;
    while (low + 1u < high) {
      let middle = (low + high) / 2u;
      if (featureOffsets[featureOffsetsOffset + middle] <= polygon) {
        low = middle;
      } else {
        high = middle;
      }
    }
    if (polygon >= featureOffsets[featureOffsetsOffset + low] &&
        polygon < featureOffsets[featureOffsetsOffset + low + 1u]) {
      feature = low;
    }
  }
  ringFeature[ringFeatureOffset + index] = feature;`
  });
}

/**
 * Vertex-parallel edge-slab construction. With `phase: 'count'` every edge of a slabbed feature
 * adds one to each slab it spans (and the kernel writes `nextVertex` and `vertexFeature`); with
 * `phase: 'fill'` each edge claims a position in every slab it spans through a per-slab cursor and
 * stores its two vertices. Entries within a slab are unordered, which does not matter: every
 * predicate over them is a parity or an any-hit test. @internal
 */
export function createCoverSlabEdgesNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  context: CellCoverKernelContext,
  phase: 'count' | 'fill',
  views: {
    ranges: GraphDataView<'uint32'>;
    nextVertex: GraphDataView<'uint32'>;
    vertexFeature: GraphDataView<'uint32'>;
    /** `'count'`: ring owner per ring. */
    ringFeature?: GraphDataView<'uint32'>;
    /** Per-slab counters (`'count'`) or cursors (`'fill'`), zero on entry. */
    slabCounters: GraphDataView<'uint32'>;
    /** `'fill'`: exclusive scan of the final per-slab counts. */
    slabOffsets?: GraphDataView<'uint32'>;
    /** `'fill'`: (first vertex, second vertex) per entry. */
    slabEntries?: GraphDataView<'uint32x2'>;
  }
): GPUCommandNode<Parameters> {
  const isCount = phase === 'count';
  const edgeSlabs = context.edgeSlabs!;
  const declarations = `${COMMON_WGSL}${SLAB_COMMON_WGSL}
const RANGE_STRIDE: u32 = ${CELL_COVER_RANGE_STRIDE}u;
const RING_COUNT: u32 = ${context.ringOffsets.length - 1}u;
const ENTRY_CAPACITY: u32 = ${edgeSlabs.entryCapacity}u;

/** Slab range [first, last] of the edge a-b and the feature's first slab index. */
fn coverGetEdgeSlabs(rangeBase: u32, a: vec2f, b: vec2f) -> vec2u {
  let slabCount = ranges[rangeBase + SLAB_COUNT_WORD];
  let origin = bitcast<f32>(ranges[rangeBase + SLAB_ORIGIN_WORD]);
  let scale = bitcast<f32>(ranges[rangeBase + SLAB_SCALE_WORD]);
  return vec2u(
    coverSlabOf(min(a.y, b.y), slabCount, origin, scale),
    coverSlabOf(max(a.y, b.y), slabCount, origin, scale)
  );
}`;
  const body = isCount
    ? `var ring = 0u;
  var high = RING_COUNT;
  while (ring + 1u < high) {
    let middle = (ring + high) / 2u;
    if (ringOffsets[ringOffsetsOffset + middle] <= index) {
      ring = middle;
    } else {
      high = middle;
    }
  }
  let ringBegin = ringOffsets[ringOffsetsOffset + ring];
  let ringEnd = ringOffsets[ringOffsetsOffset + ring + 1u];
  var feature = NO_VERTEX;
  var next = NO_VERTEX;
  if (index >= ringBegin && index < ringEnd) {
    feature = ringFeature[ringFeatureOffset + ring];
    if (feature != NO_VERTEX) {
      next = select(index + 1u, ringBegin, index + 1u == ringEnd);
    }
  }
  nextVertex[nextVertexOffset + index] = next;
  vertexFeature[vertexFeatureOffset + index] = select(NO_VERTEX, feature, next != NO_VERTEX);
  if (next == NO_VERTEX) { return; }
  let rangeBase = rangesOffset + feature * RANGE_STRIDE;
  if (ranges[rangeBase + SLAB_COUNT_WORD] == 0u) { return; }
  let slabs = coverGetEdgeSlabs(rangeBase, coverGetVertex(index), coverGetVertex(next));
  let base = ranges[rangeBase + SLAB_BASE_WORD];
  for (var slab = slabs.x; slab <= slabs.y; slab++) {
    atomicAdd(&slabCounters[slabCountersOffset + base + slab], 1u);
  }`
    : `let feature = vertexFeature[vertexFeatureOffset + index];
  if (feature == NO_VERTEX) { return; }
  let rangeBase = rangesOffset + feature * RANGE_STRIDE;
  if (ranges[rangeBase + SLAB_COUNT_WORD] == 0u) { return; }
  let next = nextVertex[nextVertexOffset + index];
  let slabs = coverGetEdgeSlabs(rangeBase, coverGetVertex(index), coverGetVertex(next));
  let base = ranges[rangeBase + SLAB_BASE_WORD];
  for (var slab = slabs.x; slab <= slabs.y; slab++) {
    let slabIndex = base + slab;
    let position = slabOffsets[slabOffsetsOffset + slabIndex] +
      atomicAdd(&slabCounters[slabCountersOffset + slabIndex], 1u);
    if (position < ENTRY_CAPACITY) {
      slabEntries[slabEntriesOffset + 2u * position] = index;
      slabEntries[slabEntriesOffset + 2u * position + 1u] = next;
    }
  }`;
  const bindings: WGSLKernelBinding[] = [
    {name: 'polygonPositions', view: context.polygonPositions, type: 'f32', access: 'read'},
    {name: 'ranges', view: views.ranges, type: 'u32', access: 'read'}
  ];
  if (isCount) {
    bindings.push(
      {name: 'ringOffsets', view: context.ringOffsets, type: 'u32', access: 'read'},
      {name: 'ringFeature', view: views.ringFeature!, type: 'u32', access: 'read'},
      {name: 'nextVertex', view: views.nextVertex, type: 'u32', access: 'read_write'},
      {name: 'vertexFeature', view: views.vertexFeature, type: 'u32', access: 'read_write'}
    );
  } else {
    bindings.push(
      {name: 'vertexFeature', view: views.vertexFeature, type: 'u32', access: 'read'},
      {name: 'nextVertex', view: views.nextVertex, type: 'u32', access: 'read'},
      {name: 'slabOffsets', view: views.slabOffsets!, type: 'u32', access: 'read'},
      {name: 'slabEntries', view: views.slabEntries!, type: 'u32', access: 'read_write'}
    );
  }
  bindings.push({
    name: 'slabCounters',
    view: views.slabCounters,
    type: 'atomic<u32>',
    access: 'read_write'
  });
  return createWGSLKernelNode<Parameters>(graph, {
    id: `${context.id}-slab-edges-${phase}`,
    operation: OPERATION,
    variant: `slab-edges-${phase}`,
    bindings,
    invocationCount: context.polygonPositions.length,
    declarations,
    body
  });
}

/** Writes accepted candidates at their scanned slots. @internal */
export function createCoverWriteNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  context: CellCoverKernelContext,
  views: {
    starts: GraphDataView<'uint32'>;
    accepted: GraphDataView<'uint32'>;
    candidateCells: GraphDataView<'uint32x2'>;
    featureIds?: GraphDataView<'uint32'>;
    outputFeatureIds: GraphDataView<'uint32'>;
    outputCells: GraphDataView<'uint32x2'>;
    outputCore?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {name: 'starts', view: views.starts, type: 'u32', access: 'read'},
    {name: 'accepted', view: views.accepted, type: 'u32', access: 'read'},
    {name: 'candidateCells', view: views.candidateCells, type: 'u32', access: 'read'}
  ];
  if (views.featureIds) {
    bindings.push({name: 'featureIds', view: views.featureIds, type: 'u32', access: 'read'});
  }
  bindings.push(
    {name: 'outputFeatureIds', view: views.outputFeatureIds, type: 'u32', access: 'read_write'},
    {name: 'outputCells', view: views.outputCells, type: 'u32', access: 'read_write'}
  );
  if (views.outputCore) {
    bindings.push({name: 'outputCore', view: views.outputCore, type: 'u32', access: 'read_write'});
  }
  return createWGSLKernelNode<Parameters>(graph, {
    id: `${context.id}-write`,
    operation: OPERATION,
    variant: 'write',
    bindings,
    invocationCount: context.candidateCapacity,
    declarations: `${getFindFeatureWGSL(context.featureCount)}
const OUTPUT_CAPACITY: u32 = ${views.outputFeatureIds.length}u;
const CORE_BIT: u32 = ${CELL_COVER_CORE_BIT}u;`,
    body: `var beforeCount = 0u;
  if (index > 0u) { beforeCount = accepted[acceptedOffset + index - 1u]; }
  let afterCount = accepted[acceptedOffset + index];
  if (afterCount == beforeCount) { return; }
  let slot = afterCount - 1u;
  if (slot >= OUTPUT_CAPACITY) { return; }
  let feature = coverFindFeature(index);
  outputFeatureIds[outputFeatureIdsOffset + slot] = ${
    views.featureIds ? 'featureIds[featureIdsOffset + feature]' : 'feature'
  };
  outputCells[outputCellsOffset + 2u * slot] = candidateCells[candidateCellsOffset + 2u * index];
  let highWord = candidateCells[candidateCellsOffset + 2u * index + 1u];
  outputCells[outputCellsOffset + 2u * slot + 1u] = highWord & ~CORE_BIT;
  ${views.outputCore ? 'outputCore[outputCoreOffset + slot] = select(0u, 1u, (highWord & CORE_BIT) != 0u);' : ''}`
  });
}

/** Writes the unclamped accepted total and the candidate overflow flag. @internal */
export function createCoverFinalizeNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  context: CellCoverKernelContext,
  views: {
    starts: GraphDataView<'uint32'>;
    accepted: GraphDataView<'uint32'>;
    total: GraphDataView<'uint32'>;
    candidateOverflow: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: `${context.id}-finalize`,
    operation: OPERATION,
    variant: 'finalize',
    bindings: [
      {name: 'starts', view: views.starts, type: 'u32', access: 'read'},
      {name: 'accepted', view: views.accepted, type: 'u32', access: 'read'},
      {name: 'total', view: views.total, type: 'u32', access: 'read_write'},
      {name: 'candidateOverflow', view: views.candidateOverflow, type: 'u32', access: 'read_write'}
    ],
    invocationCount: 1,
    declarations: `const FEATURE_COUNT: u32 = ${context.featureCount}u;
const CANDIDATE_CAPACITY: u32 = ${context.candidateCapacity}u;`,
    body: `total[totalOffset] = accepted[acceptedOffset + CANDIDATE_CAPACITY - 1u];
  candidateOverflow[candidateOverflowOffset] =
    select(0u, 1u, starts[startsOffset + FEATURE_COUNT] > CANDIDATE_CAPACITY);`
  });
}
