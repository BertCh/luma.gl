// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {LON_LAT_RING_WGSL} from '../../analysis-core/lon-lat-ring-wgsl';
import {dggs} from '@luma.gl/shadertools';
import {EXACT_GEOMETRY_WGSL, getExactRingWindingWGSL} from '../../utils/exact-geometry-wgsl';
import {EXACT_ORIENTATION_WGSL} from '../../utils/exact-orientation-wgsl';
import {CELL_KEY_WGSL, QUADBIN_TILE_WGSL, getCellKeyLayout} from '../cell-aggregation/cell-keys';
import {H3_INDEX_WGSL} from '../cell-indexing/h3-index-wgsl';
import {H3_BOUNDARY_WGSL} from '../cell-indexing/h3-boundary-wgsl';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {
  getPolygonEdgeSlabBindings,
  getPolygonEdgeSlabLoopWGSL,
  getPolygonEdgeSlabQueryWGSL,
  type PolygonEdgeSlabs
} from '../spatial-join/polygon-edge-slabs';
import {
  CELL_COVER_H3_LATTICE_SPACING_DEGREES,
  CELL_COVER_H3_MAXIMUM_LATITUDE
} from './cell-cover-h3-constants';
import {
  CELL_COVER_SCANLINE_RANGE_STRIDE,
  getCellCoverScanlineDecodeWGSL,
  type CellCoverScanlineSettings
} from './cell-cover-scanline';

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
 * Words 0 to 4 hold the candidate lattice (family specific).
 */
export const CELL_COVER_RANGE_STRIDE = 5;

/** Features with fewer candidates than this skip the edge slabs (the build would cost more). @internal */
export const CELL_COVER_MINIMUM_SLAB_CANDIDATES = 16;

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
  /** The built edge-slab index (per-feature owners), or `undefined` to test every edge of the feature. */
  edgeSlabs?: PolygonEdgeSlabs;
  /** Scanline enumeration settings, or `undefined` when only the bounding-box enumeration is built. */
  scanline?: CellCoverScanlineSettings;
  positions: GraphDataView<'float32x2'>;
  featureOffsets: GraphDataView<'uint32'>;
  polygonOffsets: GraphDataView<'uint32'>;
  ringOffsets: GraphDataView<'uint32'>;
};

/** Words per feature of the range table: the lattice, plus flags and run words with the scanline stage. */
function getRangeStride(context: CellCoverKernelContext): number {
  return context.scanline ? CELL_COVER_SCANLINE_RANGE_STRIDE : CELL_COVER_RANGE_STRIDE;
}

function getPolygonBindings(context: CellCoverKernelContext): WGSLKernelBinding[] {
  return [
    {name: 'positions', view: context.positions, type: 'f32', access: 'read'},
    {name: 'featureOffsets', view: context.featureOffsets, type: 'u32', access: 'read'},
    {name: 'polygonOffsets', view: context.polygonOffsets, type: 'u32', access: 'read'},
    {name: 'ringOffsets', view: context.ringOffsets, type: 'u32', access: 'read'}
  ];
}

/**
 * WGSL polygon helpers over the bindings `positions` (f32 pairs), `featureOffsets`,
 * `polygonOffsets` and `ringOffsets`. Containment is exact: the shared point-in-ring rule of
 * `utils/exact-geometry-wgsl.ts` (exact orientation signs, Simulation-of-Simplicity ties, so a cell
 * centre exactly on an edge is decided consistently). The rectangle test uses plain f32 operations
 * in a fixed order so the CPU oracle can mirror it with `Math.fround`. A kernel calls
 * `coverContains` from one site (the exact helpers inline a large integer fallback).
 *
 * - `coverGetFeatureInfo(feature)`: bounds `(minX, minY, maxX, maxY)` over finite vertices (`minX > maxX` when
 *   none), vertex count, first and last vertex and whether any vertex is non-finite.
 * - `coverContains(feature, p)`: even-odd containment over every ring of every polygon.
 * - `coverEdgesHitRect(feature, lo, hi)`: whether any ring edge meets the OPEN rectangle `(lo, hi)`.
 */
const COMMON_WGSL = /* wgsl */ `${EXACT_ORIENTATION_WGSL}
${EXACT_GEOMETRY_WGSL}
fn coverGetVertex(vertex: u32) -> vec2f {
  let rangeBase = positionsOffset + 2u * vertex;
  return vec2f(positions[rangeBase], positions[rangeBase + 1u]);
}

fn coverIsFinite(p: vec2f) -> bool {
  return isFiniteBits(p.x) && isFiniteBits(p.y) && abs(p.x) < 1e30 && abs(p.y) < 1e30;
}

/** Whether the closed segment a-b meets the open rectangle (lo, hi). */
fn coverSegmentHitsRect(a: vec2f, b: vec2f, lo: vec2f, hi: vec2f) -> bool {
  if (!coverIsFinite(a) || !coverIsFinite(b)) { return false; }
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
${LON_LAT_RING_WGSL}
struct CoverFeatureInfo {
  bounds: vec4f,
  vertexCount: u32,
  vertexBegin: u32,
  vertexEnd: u32,
  hasNonFinite: bool,
  hasCrossingEdge: bool
}

fn coverGetFeatureInfo(feature: u32) -> CoverFeatureInfo {
  var info = CoverFeatureInfo(vec4f(1e30, 1e30, -1e30, -1e30), 0u, 0u, 0u, false, false);
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
        let next = coverGetVertex(select(vertex + 1u, firstVertex, vertex + 1u == endVertex));
        info.hasCrossingEdge = info.hasCrossingEdge || lonLatIsCrossingEdge(p, next);
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

${getExactRingWindingWGSL('coverGetRingWinding', 'coverGetVertex')}
// Even-odd containment of the SoS-perturbed point p over every ring of the feature.
fn coverContains(feature: u32, p: vec2f) -> bool {
  let probe = ExactProbe(p, p, p, EXACT_PROBE_POINT);
  var winding = 0;
  let firstPolygon = featureOffsets[featureOffsetsOffset + feature];
  let endPolygon = featureOffsets[featureOffsetsOffset + feature + 1u];
  for (var polygon = firstPolygon; polygon < endPolygon; polygon++) {
    let firstRing = polygonOffsets[polygonOffsetsOffset + polygon];
    let endRing = polygonOffsets[polygonOffsetsOffset + polygon + 1u];
    for (var ring = firstRing; ring < endRing; ring++) {
      let firstVertex = ringOffsets[ringOffsetsOffset + ring];
      let endVertex = ringOffsets[ringOffsetsOffset + ring + 1u];
      // Rings of two vertices or fewer enclose nothing (the helper returns 0 for them).
      winding += coverGetRingWinding(probe, firstVertex, endVertex).x;
    }
  }
  return (winding & 1) != 0;
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
 * Edge-slab versions of `coverContains` and `coverEdgesHitRect` with the signatures of the
 * brute-force ones, over the polygon edge-slab index (`spatial-join/polygon-edge-slabs.ts`, owner =
 * feature). A feature without slabs (too few candidates or vertices, non-finite vertices, or
 * entries beyond the entry capacity) walks all its edges, the same edges as the ring walk. Needs
 * `positions` and the three edge-slab bindings.
 */
function getSlabQueryWGSL(edgeSlabs: PolygonEdgeSlabs): string {
  return /* wgsl */ `${COMMON_WGSL}${getPolygonEdgeSlabQueryWGSL(edgeSlabs)}
// Even-odd containment of the SoS-perturbed point p. Slabbed and walked features share one edge
// loop, so the exact crossing helper has a single call site.
fn coverContains(feature: u32, p: vec2f) -> bool {
  let probe = ExactProbe(p, p, p, EXACT_PROBE_POINT);
  var winding = 0;
  ${getPolygonEdgeSlabLoopWGSL({
    owner: 'feature',
    lowY: 'p.y',
    highY: 'p.y',
    getVertex: 'coverGetVertex',
    body: 'winding += exactProbeCrossing(probe, a, b).x;'
  })}
  return (winding & 1) != 0;
}

fn coverEdgesHitRect(feature: u32, lo: vec2f, hi: vec2f) -> bool {
  ${getPolygonEdgeSlabLoopWGSL({
    owner: 'feature',
    lowY: 'lo.y',
    highY: 'hi.y',
    getVertex: 'coverGetVertex',
    body: 'if (coverSegmentHitsRect(a, b, lo, hi)) { return true; }'
  })}
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

/** Per-feature status counter updates over `bounds`; defines `hasCrossingEdge`. */
function getCoverStatusSource(
  family: CellCoverKernelContext['family'],
  status: {invalidCount?: GraphDataView<'uint32'>; uncertainCount?: GraphDataView<'uint32'>}
): string {
  // Unsplit crossing edges get no candidates; split full-width features remain valid.
  // A feature reaching the latitudes the lattice clamps to loses polar cells: counted uncertain.
  const invalidStatement = status.invalidCount
    ? 'if (hasCrossingEdge) { atomicAdd(&invalidCount[invalidCountOffset], 1u); }'
    : '';
  const polarLatitude = family === 'h3' ? CELL_COVER_H3_MAXIMUM_LATITUDE : 85.051129;
  const uncertainStatement = status.uncertainCount
    ? `if (!hasCrossingEdge && bounds.x <= bounds.z && (bounds.w > ${getWGSLFloatLiteral(polarLatitude)} || bounds.y < -${getWGSLFloatLiteral(polarLatitude)})) { atomicAdd(&uncertainCount[uncertainCountOffset], 1u); }`
    : '';
  return `let hasCrossingEdge = info.hasCrossingEdge;
  ${invalidStatement}
  ${uncertainStatement}`;
}

function getCoverStatusBindings(status: {
  invalidCount?: GraphDataView<'uint32'>;
  uncertainCount?: GraphDataView<'uint32'>;
}): WGSLKernelBinding[] {
  return (['invalidCount', 'uncertainCount'] as const).flatMap(name => {
    const view = status[name];
    return view ? [{name, view, type: 'atomic<u32>', access: 'read_write'} as const] : [];
  });
}

/** Candidate counting node: one thread per feature (plus one trailing zero count). @internal */
export function createCoverCountNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  context: CellCoverKernelContext,
  counts: GraphDataView<'uint32'>,
  ranges: GraphDataView<'uint32'>,
  status: {
    invalidCount?: GraphDataView<'uint32'>;
    uncertainCount?: GraphDataView<'uint32'>;
  } = {}
): GPUCommandNode<Parameters> {
  const {resolution, candidateCapacity, family} = context;
  const crossingSource = getCoverStatusSource(family, status);
  const capacityPlusOne = candidateCapacity + 1;
  // Flag word 5 of the scanline range table: bit 0 marks a non-finite vertex (bounding box only).
  const nonFiniteSource = context.scanline
    ? 'ranges[rangeBase + 5u] = select(0u, 1u, info.hasNonFinite);'
    : '';
  let declarations = `${POLYGON_WGSL}
const CAPACITY_PLUS_ONE: u32 = ${capacityPlusOne}u;
const RANGE_STRIDE: u32 = ${getRangeStride(context)}u;
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
    body = `if (index == FEATURE_COUNT) { counts[countsOffset + index] = 0u; return; }
  let info = coverGetFeatureInfo(index);
  let bounds = info.bounds;
  let rangeBase = rangesOffset + index * RANGE_STRIDE;
  var count = 0u;
  ${nonFiniteSource}
  ${crossingSource}
  if (!hasCrossingEdge && bounds.x <= bounds.z && bounds.y <= bounds.w) {
    // Point indexing wraps +180 to column zero; bounds must retain the easternmost column.
    let lastColumn = (1u << RESOLUTION) - 1u;
    let x0 = select(quadbinGetTileX(bounds.x, RESOLUTION), lastColumn, bounds.x >= 180.0);
    let x1 = select(quadbinGetTileX(bounds.z, RESOLUTION), lastColumn, bounds.z >= 180.0);
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
  counts[countsOffset + index] = count;`;
  } else {
    const spacing = CELL_COVER_H3_LATTICE_SPACING_DEGREES[resolution];
    declarations += `const LATTICE_SPACING: f32 = ${getWGSLFloatLiteral(spacing)};
const MAXIMUM_LATITUDE: f32 = ${getWGSLFloatLiteral(CELL_COVER_H3_MAXIMUM_LATITUDE)};
const DEGREES_TO_RADIANS: f32 = 0.017453292519943295;
fn coverToCount(value: f32) -> u32 {
  return u32(clamp(value, 1.0, 4.0e9));
}`;
    body = `if (index == FEATURE_COUNT) { counts[countsOffset + index] = 0u; return; }
  let info = coverGetFeatureInfo(index);
  let bounds = info.bounds;
  let rangeBase = rangesOffset + index * RANGE_STRIDE;
  var count = 0u;
  ${nonFiniteSource}
  let originLat = max(bounds.y - LATTICE_SPACING, -MAXIMUM_LATITUDE);
  let endLat = min(bounds.w + LATTICE_SPACING, MAXIMUM_LATITUDE);
  ${crossingSource}
  if (!hasCrossingEdge && bounds.x <= bounds.z && bounds.y <= bounds.w && originLat <= endLat) {
    // Longitude spacing from the most equatorward latitude of the band, where a degree of longitude
    // is longest: a spacing taken from the poleward edge is too coarse for the rows nearer the
    // equator and loses cells there.
    let equatorwardLat = select(min(abs(originLat), abs(endLat)), 0.0, originLat <= 0.0 && endLat >= 0.0);
    let spacingLng = LATTICE_SPACING / cos(equatorwardLat * DEGREES_TO_RADIANS);
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
  counts[countsOffset + index] = count;`;
  }
  declarations += `const FEATURE_COUNT: u32 = ${context.featureCount}u;`;
  return createWGSLKernelNode<Parameters>(graph, {
    id: `${context.id}-count`,
    operation: OPERATION,
    variant: 'count',
    bindings: [
      ...getPolygonBindings(context),
      {name: 'counts', view: counts, type: 'u32', access: 'read_write'},
      {name: 'ranges', view: ranges, type: 'u32', access: 'read_write'},
      ...getCoverStatusBindings(status)
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
  }
): GPUCommandNode<Parameters> {
  const {family, resolution, candidateCapacity, edgeSlabs} = context;
  const mode = CELL_COVER_MODE_CODES[context.containment];
  // Lattice `column` and `row` of the candidate: bounding-box features decode the row-major index,
  // scanline features look up their run (see cell-cover-scanline.ts).
  const decodeSource = context.scanline
    ? getCellCoverScanlineDecodeWGSL(context.featureCount, 'width')
    : `let column = candidateLocal % width;
  let row = candidateLocal / width;`;
  const slabBindings: WGSLKernelBinding[] = edgeSlabs ? getPolygonEdgeSlabBindings(edgeSlabs) : [];
  let declarations = `${edgeSlabs ? getSlabQueryWGSL(edgeSlabs) : POLYGON_WGSL}${getFindFeatureWGSL(context.featureCount)}
const CANDIDATE_CAPACITY: u32 = ${candidateCapacity}u;
const RANGE_STRIDE: u32 = ${getRangeStride(context)}u;
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
  ${decodeSource}
  let x = ranges[rangeBase] + column;
  let y = ranges[rangeBase + 1u] + row;
  let lng0 = f32(x) * TILE_WIDTH - 180.0;
  let lng1 = f32(x + 1u) * TILE_WIDTH - 180.0;
  let latNorth = coverTileLatitude(f32(y));
  let latSouth = coverTileLatitude(f32(y + 1u));
  let centre = vec2f((f32(x) + 0.5) * TILE_WIDTH - 180.0, coverTileLatitude(f32(y) + 0.5));
  // One call site for the exact containment test (see COMMON_WGSL).
  let centreInside = coverContains(feature, centre);
  var accepted = false;
  if (MODE == 0u) {
    accepted = centreInside;
  } else {
    let hit = coverEdgesHitRect(feature, vec2f(lng0, latSouth), vec2f(lng1, latNorth));
    if (MODE == 1u) {
      accepted = !hit && centreInside;
    } else {
      accepted = hit || centreInside;
    }
  }
  var key = vec2u(0u);
  if (accepted) {
    key = cellGetKey(quadbinGetCompactKey(x, y), ${layout.headerHigh}u, RESOLUTION, ${layout.lowBit}u);
    ${
      context.computeCore
        ? `// Core: the centre is inside and no polygon edge enters the cell grown by a safety margin.
    let slack = vec2f(CORE_MARGIN + 0.001 * (lng1 - lng0));
    if (centreInside && !coverEdgesHitRect(feature, vec2f(lng0, latSouth) - slack, vec2f(lng1, latNorth) + slack)) {
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
  ${decodeSource}
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
              name: 'positions',
              view: context.positions,
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

/** Writes the unclamped accepted total, the candidate overflow flag and the optional candidate count. @internal */
export function createCoverFinalizeNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  context: CellCoverKernelContext,
  views: {
    starts: GraphDataView<'uint32'>;
    accepted: GraphDataView<'uint32'>;
    total: GraphDataView<'uint32'>;
    candidateOverflow: GraphDataView<'uint32'>;
    candidateCount?: GraphDataView<'uint32'>;
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
      {name: 'candidateOverflow', view: views.candidateOverflow, type: 'u32', access: 'read_write'},
      ...(views.candidateCount
        ? [
            {
              name: 'candidateCount',
              view: views.candidateCount,
              type: 'u32',
              access: 'read_write'
            } as WGSLKernelBinding
          ]
        : [])
    ],
    invocationCount: 1,
    declarations: `const FEATURE_COUNT: u32 = ${context.featureCount}u;
const CANDIDATE_CAPACITY: u32 = ${context.candidateCapacity}u;`,
    body: `total[totalOffset] = accepted[acceptedOffset + CANDIDATE_CAPACITY - 1u];
  candidateOverflow[candidateOverflowOffset] =
    select(0u, 1u, starts[startsOffset + FEATURE_COUNT] > CANDIDATE_CAPACITY);
  ${views.candidateCount ? 'candidateCount[candidateCountOffset] = starts[startsOffset + FEATURE_COUNT];' : ''}`
  });
}
