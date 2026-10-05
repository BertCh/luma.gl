// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {dggs} from '@luma.gl/shadertools';
import {CELL_KEY_WGSL, QUADBIN_TILE_WGSL, getCellKeyLayout} from '../cell-aggregation/cell-keys';
import {H3_INDEX_WGSL} from '../cell-indexing/h3-index-wgsl';
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

/** Number of `uint32` words per feature in the candidate range table. @internal */
export const CELL_COVER_RANGE_STRIDE = 8;

/** Common description of a cover kernel family. @internal */
export type CellCoverKernelContext = {
  id: string;
  family: 'quadbin' | 'h3';
  resolution: number;
  containment: keyof typeof CELL_COVER_MODE_CODES;
  featureCount: number;
  candidateCapacity: number;
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
 * - `coverGetFeatureBounds(feature)`: `(minX, minY, maxX, maxY)` over finite vertices, `minX > maxX` when none.
 * - `coverContains(feature, p)`: even-odd containment over every ring of every polygon.
 * - `coverEdgesHitRect(feature, lo, hi)`: whether any ring edge meets the OPEN rectangle `(lo, hi)`.
 */
const POLYGON_WGSL = /* wgsl */ `
fn coverGetVertex(vertex: u32) -> vec2f {
  let rangeBase = polygonPositionsOffset + 2u * vertex;
  return vec2f(polygonPositions[rangeBase], polygonPositions[rangeBase + 1u]);
}

fn coverIsFinite(p: vec2f) -> bool {
  return p.x == p.x && p.y == p.y && abs(p.x) < 1e30 && abs(p.y) < 1e30;
}

fn coverGetFeatureBounds(feature: u32) -> vec4f {
  var bounds = vec4f(1e30, 1e30, -1e30, -1e30);
  let firstPolygon = featureOffsets[featureOffsetsOffset + feature];
  let endPolygon = featureOffsets[featureOffsetsOffset + feature + 1u];
  for (var polygon = firstPolygon; polygon < endPolygon; polygon++) {
    let firstRing = polygonOffsets[polygonOffsetsOffset + polygon];
    let endRing = polygonOffsets[polygonOffsetsOffset + polygon + 1u];
    for (var ring = firstRing; ring < endRing; ring++) {
      let firstVertex = ringOffsets[ringOffsetsOffset + ring];
      let endVertex = ringOffsets[ringOffsetsOffset + ring + 1u];
      for (var vertex = firstVertex; vertex < endVertex; vertex++) {
        let p = coverGetVertex(vertex);
        if (coverIsFinite(p)) {
          bounds = vec4f(min(bounds.xy, p), max(bounds.zw, p));
        }
      }
    }
  }
  return bounds;
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

/** Candidate counting node: one thread per feature (plus one trailing zero count). @internal */
export function createCoverCountNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  context: CellCoverKernelContext,
  counts: GraphDataView<'uint32'>,
  ranges: GraphDataView<'uint32'>
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
    body = `if (index == FEATURE_COUNT) { counts[countsOffset + index] = 0u; return; }
  let bounds = coverGetFeatureBounds(index);
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
  let bounds = coverGetFeatureBounds(index);
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
      {name: 'ranges', view: ranges, type: 'u32', access: 'read_write'}
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
  const {family, resolution, candidateCapacity} = context;
  const mode = CELL_COVER_MODE_CODES[context.containment];
  let declarations = `${POLYGON_WGSL}${getFindFeatureWGSL(context.featureCount)}
const CANDIDATE_CAPACITY: u32 = ${candidateCapacity}u;
const RANGE_STRIDE: u32 = ${CELL_COVER_RANGE_STRIDE}u;
const MODE: u32 = ${mode}u;
const RESOLUTION: u32 = ${resolution}u;
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
  }`;
  } else {
    declarations += `${dggs.source}\n${H3_INDEX_WGSL}`;
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
    }
  }
  let accepted = key.x != 0u || key.y != 0u;`;
  }
  return createWGSLKernelNode<Parameters>(graph, {
    id: `${context.id}-test`,
    operation: OPERATION,
    variant: 'test',
    bindings: [
      ...getPolygonBindings(context),
      {name: 'ranges', view: views.ranges, type: 'u32', access: 'read'},
      {name: 'starts', view: views.starts, type: 'u32', access: 'read'},
      {name: 'flags', view: views.flags, type: 'u32', access: 'read_write'},
      {name: 'candidateCells', view: views.candidateCells, type: 'u32', access: 'read_write'}
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
  return createWGSLKernelNode<Parameters>(graph, {
    id: `${context.id}-write`,
    operation: OPERATION,
    variant: 'write',
    bindings,
    invocationCount: context.candidateCapacity,
    declarations: `${getFindFeatureWGSL(context.featureCount)}
const OUTPUT_CAPACITY: u32 = ${views.outputFeatureIds.length}u;`,
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
  outputCells[outputCellsOffset + 2u * slot + 1u] = candidateCells[candidateCellsOffset + 2u * index + 1u];`
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
