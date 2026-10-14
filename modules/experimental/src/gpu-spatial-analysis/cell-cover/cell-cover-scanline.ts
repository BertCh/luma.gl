// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {CELL_KEY_WGSL, QUADBIN_TILE_WGSL} from '../cell-aggregation/cell-keys';
import {createStableLexicographicIndexSortNodes} from '../../utils/stable-lexicographic-index-sort';
import {getSortKeyBits} from '../../utils/sorted-segment-sums';
import {
  createFillNode,
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {CELL_COVER_H3_LATTICE_SPACING_DEGREES} from './cell-cover-h3-constants';
import type {CellCoverKernelContext} from './cell-cover-kernels';

/**
 * Boundary-trace plus scanline-fill candidate enumeration of the cell cover (internal).
 *
 * The bounding-box enumeration tests every cell of a feature's bounding box. The candidate lattice
 * of both families is axis aligned in degree space (Quadbin tile rows are latitude bands, H3 sample
 * rows are constant-latitude lines), so per lattice row the cells that can be accepted lie in the
 * x intervals where the polygon meets the row. This stage computes those intervals and the
 * existing exact test then runs on the cells inside them only.
 *
 * Why the accepted cells are bitwise the cells of the bounding-box enumeration (the output is a
 * subset of a superset). Fix a feature and a lattice row with centre line `y` (the Quadbin tile
 * centre latitude, or the H3 sample row latitude) and band `B` (the tile latitude band for Quadbin,
 * `y +- (spacing / 2)` for H3, both grown by `Y_SLACK`). A candidate cell is accepted only if
 *
 *   (i)  its test point (Quadbin tile centre, H3 cell centre) is inside the polygon, or, for
 *        Quadbin `'intersects'`, an edge meets the open tile rectangle.
 *
 * The test point has `y` within the band and `x` within `M` of the candidate's lattice x (`M` is
 * zero for Quadbin, where the tile column contains the centre, and about half a column spacing for
 * H3, whose accepted cell centre rounds to the lattice point). Two cases cover (i):
 *
 *   (a) some polygon edge meets the candidate's rectangle `R` (x within `M`, y in the band). The
 *       boundary trace emits, per edge and per row whose band the edge meets, the interval
 *       `[xLow, xHigh]` of the part of the edge inside the band, grown by an f32 slack. The edge
 *       point inside `R` has x in this interval, so the candidate column is inside the interval
 *       grown by `M`. This also covers every case the exact test decides by tie rules (a test
 *       point exactly on an edge lies on the edge, hence in its interval).
 *   (b) no edge meets `R`. `R` is connected, so it lies wholly inside or outside the polygon; for
 *       it to contain an inside test point it lies inside, and then so does its lattice point
 *       `(x, y)` on the centre line. The scanline fill collects, per edge crossing the centre line
 *       (half-open on y: `(a.y <= y) != (b.y <= y)`, so a vertex on the line counts once), the
 *       crossing x. Sorted by x and paired (even-odd over all rings, as the exact test), the pairs
 *       are the interior intervals of the line; `x` is inside one of them.
 *
 * Both interval kinds become events (open and close of a span, a toggle for a crossing), sorted by
 * (row, x). A gap between consecutive events of a row is kept when a span covers it or the number
 * of crossings left of it is odd. The kept gaps are grown by the slack, converted to column
 * ranges with one extra column on each side, and made disjoint by dropping columns already owned
 * by the previous kept gap of the row (column ranges are monotone in x, so one neighbour
 * suffices). The enumeration therefore visits each cell of the union once, in the row-major order
 * of the bounding-box enumeration.
 *
 * The f32 slack. A crossing x is `a.x + (b.x - a.x) * t` with `t = (y - a.y) / (b.y - a.y)`; each
 * operation rounds with relative error 2^-24, subtraction results included (each difference is
 * rounded relative to its own magnitude), so the error is at most `4 * 2^-24 * |b.x - a.x|` plus
 * half an ulp of `a.x` (at most `2^-17` for |x| <= 180). The slack `4e-5 + 1e-6 * |b.x - a.x|` is
 * at least 2.6 times that bound. Order statistics are 1-Lipschitz in the sup norm, so sorting
 * crossings with errors of at most that bound moves every pairing endpoint by at most the bound.
 * Lattice coordinates (`originLng + column * spacing`, tile edges from f32 `atan`/`exp`) differ
 * between kernels by a few ulps, covered by the extra column and by `Y_SLACK` (2e-4 degrees, more
 * than ten times the f32 latitude error of the tile functions; for H3 the row margin also adds
 * `3e-7` of a spacing per row for the f32 quotient that rounds a centre to a row).
 *
 * Capacities and fallback. Edge-row slots, events and runs are bounded by `slotCapacity`
 * (`3 * slotCapacity` events and runs). A feature whose slots would not fit, whose row count
 * exceeds it, that has a non-finite vertex or whose scanline enumeration would hold 2^32 or more
 * candidates keeps the bounding-box enumeration (the latter overflows `candidateCapacity` either
 * way). The fallback gives the same cells, so the stage never changes results.
 *
 * @internal
 */

/** Words per feature in the range table with the scanline stage. */
export const CELL_COVER_SCANLINE_RANGE_STRIDE = 8;

/** Words per candidate run record, stored after the per-feature words of the range table. */
export const CELL_COVER_SCANLINE_RUN_STRIDE = 8;

/** Largest `scanlineSlotCapacity`: `3 * slots` runs of 8 words must fit one storage binding. */
export const CELL_COVER_SCANLINE_MAXIMUM_SLOT_CAPACITY = 2 ** 20;

/** `candidateCapacity` from which `candidateEnumeration: 'auto'` builds the scanline stage. */
export const CELL_COVER_SCANLINE_AUTO_MINIMUM_CANDIDATE_CAPACITY = 65536;

/** The scanline enumeration wins when bounding-box cells exceed this times its work estimate. */
export const CELL_COVER_SCANLINE_COST_RATIO = 4;

/** Absolute x slack (degrees) of every boundary span and kept gap, see the file comment. */
export const CELL_COVER_SCANLINE_X_SLACK_DEGREES = 4e-5;

/** Relative x slack (times the edge's x extent) of every boundary span. */
export const CELL_COVER_SCANLINE_X_SLACK_RELATIVE = 1e-6;

/** Quadbin latitude slack (degrees) added to every tile band. */
export const CELL_COVER_SCANLINE_QUADBIN_Y_SLACK_DEGREES = 2e-4;

/** H3 absolute latitude margin (degrees) beyond half a row spacing. */
export const CELL_COVER_SCANLINE_H3_Y_SLACK_DEGREES = 3e-5;

/** H3 absolute longitude margin (degrees) beyond half a column spacing. */
export const CELL_COVER_SCANLINE_H3_X_SLACK_DEGREES = 6e-5;

/** Compile-time settings of the scanline stage. @internal */
export type CellCoverScanlineSettings = {
  /** Edge-row slot capacity; events and runs are three times as many. */
  slotCapacity: number;
  /** Whether every eligible feature uses the scanline enumeration (`'scanline'`), not only when cheaper. */
  force: boolean;
};

/** Number of `uint32` words of the range table: per-feature words, then the run records. @internal */
export function getCellCoverRangesLength(
  featureCount: number,
  settings: CellCoverScanlineSettings | undefined,
  baseStride: number
): number {
  const features = Math.max(featureCount, 1);
  return settings
    ? features * CELL_COVER_SCANLINE_RANGE_STRIDE +
        3 * settings.slotCapacity * CELL_COVER_SCANLINE_RUN_STRIDE
    : features * baseStride;
}

const NONE = '0xffffffffu';

/** Offset-array search: the largest `i` in `[0, rangeCount)` with `offsets[i] <= value`, else NONE. */
function getRangeSearchWGSL(name: string, binding: string, rangeCount: number): string {
  return /* wgsl */ `
fn ${name}(value: u32) -> u32 {
  var low = 0u;
  var high = ${rangeCount}u;
  while (low < high) {
    let middle = (low + high) >> 1u;
    if (${binding}[${binding}Offset + middle] <= value) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  if (low == 0u) { return ${NONE}; }
  let range = low - 1u;
  if (value >= ${binding}[${binding}Offset + range + 1u]) { return ${NONE}; }
  return range;
}`;
}

/**
 * Family-specific lattice helpers over the binding `ranges` and a feature's word base `rb`
 * (offset included). They define the rows and columns the candidate decode of the test kernel uses.
 *
 * - `coverRowCount`, `coverColumnCount(rb)`: lattice size.
 * - `coverRowLine(rb, row)`: latitude of the row's test line, computed like the test kernel.
 * - `coverRowBand(rb, row)`: `(low, high)` latitudes of the row's band with slack.
 * - `coverEdgeRows(rb, lowY, highY)`: inclusive row range whose band can meet `[lowY, highY]`
 *   (one spare row on each side), clamped to the lattice; empty when `x > y`.
 * - `coverColumnRange(rb, lowX, highX)`: inclusive column range that can hold a candidate for the
 *   x interval, one spare column on each side, clamped; empty when `x > y`.
 */
function getLatticeWGSL(context: CellCoverKernelContext): string {
  const common = /* wgsl */ `
const X_SLACK_BASE: f32 = ${getWGSLFloatLiteral(CELL_COVER_SCANLINE_X_SLACK_DEGREES)};
const X_SLACK_RELATIVE: f32 = ${getWGSLFloatLiteral(CELL_COVER_SCANLINE_X_SLACK_RELATIVE)};
fn coverFloorToInt(value: f32) -> i32 { return i32(clamp(floor(value), -2.0e9, 2.0e9)); }
fn coverCeilToInt(value: f32) -> i32 { return i32(clamp(ceil(value), -2.0e9, 2.0e9)); }
`;
  if (context.family === 'quadbin') {
    const resolution = context.resolution;
    return /* wgsl */ `${common}${CELL_KEY_WGSL}${QUADBIN_TILE_WGSL}
const TILE_WIDTH: f32 = ${getWGSLFloatLiteral(45 * 2 ** (3 - resolution))};
const ROW_SCALE: f32 = ${getWGSLFloatLiteral(2 ** (1 - resolution))};
const PI: f32 = 3.14159265358979;
const RESOLUTION: u32 = ${resolution}u;
const Y_SLACK: f32 = ${getWGSLFloatLiteral(CELL_COVER_SCANLINE_QUADBIN_Y_SLACK_DEGREES)};
fn coverTileLatitude(row: f32) -> f32 {
  let t = PI * (1.0 - row * ROW_SCALE);
  return (2.0 * atan(exp(t)) - PI * 0.5) * 57.29577951308232;
}
fn coverRowCount(rb: u32) -> u32 { return ranges[rb + 3u]; }
fn coverColumnCount(rb: u32) -> u32 { return ranges[rb + 2u]; }
fn coverRowLine(rb: u32, row: u32) -> f32 {
  return coverTileLatitude(f32(ranges[rb + 1u] + row) + 0.5);
}
fn coverRowBand(rb: u32, row: u32) -> vec2f {
  let y = ranges[rb + 1u] + row;
  return vec2f(coverTileLatitude(f32(y + 1u)) - Y_SLACK, coverTileLatitude(f32(y)) + Y_SLACK);
}
fn coverEdgeRows(rb: u32, lowY: f32, highY: f32) -> vec2i {
  let origin = i32(ranges[rb + 1u]);
  let first = i32(quadbinGetTileY(highY + Y_SLACK, RESOLUTION)) - origin - 1;
  let last = i32(quadbinGetTileY(lowY - Y_SLACK, RESOLUTION)) - origin + 1;
  return vec2i(max(first, 0), min(last, i32(ranges[rb + 3u]) - 1));
}
fn coverColumnRange(rb: u32, lowX: f32, highX: f32) -> vec2i {
  let origin = i32(ranges[rb]);
  let first = coverFloorToInt((lowX - X_SLACK_BASE + 180.0) / TILE_WIDTH) - 1 - origin;
  let last = coverFloorToInt((highX + X_SLACK_BASE + 180.0) / TILE_WIDTH) + 1 - origin;
  return vec2i(max(first, 0), min(last, i32(ranges[rb + 2u]) - 1));
}
`;
  }
  const spacing = CELL_COVER_H3_LATTICE_SPACING_DEGREES[context.resolution];
  return /* wgsl */ `${common}
const LATTICE_SPACING: f32 = ${getWGSLFloatLiteral(spacing)};
const Y_SLACK: f32 = ${getWGSLFloatLiteral(CELL_COVER_SCANLINE_H3_Y_SLACK_DEGREES)};
const X_MARGIN_SLACK: f32 = ${getWGSLFloatLiteral(CELL_COVER_SCANLINE_H3_X_SLACK_DEGREES)};
fn coverRowCount(rb: u32) -> u32 { return ranges[rb + 1u]; }
fn coverColumnCount(rb: u32) -> u32 { return ranges[rb]; }
fn coverOriginLng(rb: u32) -> f32 { return bitcast<f32>(ranges[rb + 2u]); }
fn coverOriginLat(rb: u32) -> f32 { return bitcast<f32>(ranges[rb + 3u]); }
fn coverSpacingLng(rb: u32) -> f32 { return bitcast<f32>(ranges[rb + 4u]); }
fn coverRowLine(rb: u32, row: u32) -> f32 {
  return coverOriginLat(rb) + f32(row) * LATTICE_SPACING;
}
// An accepted cell's centre rounds to its lattice row: its latitude is within half a spacing of the
// row line, plus the f32 error of the rounding quotient (3e-7 of a spacing per row) and of the line.
fn coverRowHalfBand(rb: u32) -> f32 {
  return LATTICE_SPACING * (0.501 + 3.0e-7 * f32(ranges[rb + 1u])) + Y_SLACK;
}
fn coverRowBand(rb: u32, row: u32) -> vec2f {
  let line = coverRowLine(rb, row);
  let half = coverRowHalfBand(rb);
  return vec2f(line - half, line + half);
}
fn coverEdgeRows(rb: u32, lowY: f32, highY: f32) -> vec2i {
  let origin = coverOriginLat(rb);
  let half = coverRowHalfBand(rb);
  let first = coverFloorToInt((lowY - half - origin) / LATTICE_SPACING) - 1;
  let last = coverCeilToInt((highY + half - origin) / LATTICE_SPACING) + 1;
  return vec2i(max(first, 0), min(last, i32(ranges[rb + 1u]) - 1));
}
fn coverColumnMargin(rb: u32) -> f32 {
  return coverSpacingLng(rb) * (0.501 + 3.0e-7 * f32(ranges[rb])) + X_MARGIN_SLACK;
}
fn coverColumnRange(rb: u32, lowX: f32, highX: f32) -> vec2i {
  let origin = coverOriginLng(rb);
  let spacing = coverSpacingLng(rb);
  let margin = coverColumnMargin(rb) + X_SLACK_BASE;
  let first = coverFloorToInt((lowX - margin - origin) / spacing) - 1;
  let last = coverCeilToInt((highX + margin - origin) / spacing) + 1;
  return vec2i(max(first, 0), min(last, i32(ranges[rb]) - 1));
}
`;
}

/** Order-preserving u32 keys of finite f32 values, for the event sort. */
const SORTABLE_KEY_WGSL = /* wgsl */ `
fn coverSortableKey(value: f32) -> u32 {
  let bits = bitcast<u32>(value);
  return select(bits | 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}

fn coverFromSortableKey(key: u32) -> f32 {
  return bitcast<f32>(select(~key, key & 0x7fffffffu, (key & 0x80000000u) != 0u));
}
`;

/** Geometry helpers shared by the scanline kernels: vertices, edge clip, crossing, sortable keys. */
const GEOMETRY_WGSL = /* wgsl */ `${SORTABLE_KEY_WGSL}
fn coverGetVertex(vertex: u32) -> vec2f {
  let base = positionsOffset + 2u * vertex;
  return vec2f(positions[base], positions[base + 1u]);
}

/** (valid, xLow, xHigh): x extent of the part of edge a-b inside the latitude band, with slack. */
fn coverClipEdge(a: vec2f, b: vec2f, band: vec2f) -> vec3f {
  let aIsLower = a.y <= b.y;
  let lower = select(b, a, aIsLower);
  let upper = select(a, b, aIsLower);
  if ((upper.y < band.x) || (lower.y > band.y)) {
    return vec3f(0.0, 0.0, 0.0);
  }
  var x0 = lower.x;
  var x1 = upper.x;
  let dy = upper.y - lower.y;
  if (dy > 0.0) {
    let dx = upper.x - lower.x;
    x0 = lower.x + dx * clamp((band.x - lower.y) / dy, 0.0, 1.0);
    x1 = lower.x + dx * clamp((band.y - lower.y) / dy, 0.0, 1.0);
  }
  let slack = X_SLACK_BASE + X_SLACK_RELATIVE * abs(upper.x - lower.x);
  return vec3f(1.0, min(x0, x1) - slack, max(x0, x1) + slack);
}

/** x where edge a-b crosses latitude line; only called when (a.y <= line) != (b.y <= line). */
fn coverGetCrossing(a: vec2f, b: vec2f, line: f32) -> f32 {
  let aIsLower = a.y <= b.y;
  let lower = select(b, a, aIsLower);
  let upper = select(a, b, aIsLower);
  let t = clamp((line - lower.y) / (upper.y - lower.y), 0.0, 1.0);
  return lower.x + (upper.x - lower.x) * t;
}

`;

/** Views the scanline stage reads and rewrites. @internal */
export type CellCoverScanlineViews = {
  /** Per-feature candidate counts; scanline features get their enumerated count. */
  counts: GraphDataView<'uint32'>;
  /** Range table of {@link getCellCoverRangesLength} words. */
  ranges: GraphDataView<'uint32'>;
};

/**
 * Returns the scanline stage nodes: edge/row discovery, per-feature choice, event emission, sort,
 * fill, column ranges, run table and final per-feature counts. Runs after the count kernel and
 * before the candidate scan. @internal
 */
export function getCellCoverScanlineNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  context: CellCoverKernelContext & {scanline: CellCoverScanlineSettings},
  views: CellCoverScanlineViews
): GPUCommandNode<Parameters>[] {
  const {id, featureCount, candidateCapacity, positions} = context;
  const {slotCapacity, force} = context.scanline;
  const {counts, ranges} = views;
  const operation = 'GPUCellCover';
  const vertexCount = positions.length;
  const eventCount = 3 * slotCapacity;
  const stride = CELL_COVER_SCANLINE_RANGE_STRIDE;
  const runBase = `rangesOffset + ${featureCount}u * ${stride}u`;
  const nodes: GPUCommandNode<Parameters>[] = [];
  const lattice = getLatticeWGSL(context);
  const make = (name: string) => `${id}-scanline-${name}`;
  const view = (name: string, length: number) =>
    createTransientView(graph, make(name), 'uint32', Math.max(length, 1));
  const featureConstants = `const FEATURE_COUNT: u32 = ${featureCount}u;
const VERTEX_COUNT: u32 = ${vertexCount}u;
const STRIDE: u32 = ${stride}u;
const SLOT_CAPACITY: u32 = ${slotCapacity}u;
const INVALID_ROW: u32 = ${slotCapacity}u;`;
  const polygonBindings: WGSLKernelBinding[] = [
    {name: 'positions', view: positions, type: 'f32', access: 'read'}
  ];

  // 1. Edge discovery: feature and successor vertex of every eligible edge.
  const edgeFeature = view('edge-feature', vertexCount + 1);
  const edgeNext = view('edge-next', vertexCount + 1);
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: make('edges'),
      operation,
      variant: 'scanline-edges',
      bindings: [
        {name: 'ringOffsets', view: context.ringOffsets, type: 'u32', access: 'read'},
        {name: 'polygonOffsets', view: context.polygonOffsets, type: 'u32', access: 'read'},
        {name: 'featureOffsets', view: context.featureOffsets, type: 'u32', access: 'read'},
        {name: 'ranges', view: ranges, type: 'u32', access: 'read'},
        {name: 'counts', view: counts, type: 'u32', access: 'read'},
        {name: 'edgeFeature', view: edgeFeature, type: 'u32', access: 'read_write'},
        {name: 'edgeNext', view: edgeNext, type: 'u32', access: 'read_write'}
      ],
      invocationCount: vertexCount + 1,
      declarations: `${featureConstants}
${getRangeSearchWGSL('coverFindRing', 'ringOffsets', context.ringOffsets.length - 1)}
${getRangeSearchWGSL('coverFindPolygon', 'polygonOffsets', context.polygonOffsets.length - 1)}
${getRangeSearchWGSL('coverFindFeatureOfPolygon', 'featureOffsets', featureCount)}`,
      body: `var feature = ${NONE};
  var next = ${NONE};
  if (index < VERTEX_COUNT) {
    let ring = coverFindRing(index);
    if (ring != ${NONE}) {
      let polygon = coverFindPolygon(ring);
      if (polygon != ${NONE}) {
        let candidate = coverFindFeatureOfPolygon(polygon);
        if (candidate != ${NONE}) {
          let rb = rangesOffset + candidate * STRIDE;
          // Non-finite vertices (flag bit 0) or no lattice: the feature keeps the bounding box.
          if (counts[countsOffset + candidate] > 0u && (ranges[rb + 5u] & 1u) == 0u) {
            let ringStart = ringOffsets[ringOffsetsOffset + ring];
            let ringEnd = ringOffsets[ringOffsetsOffset + ring + 1u];
            feature = candidate;
            next = select(index + 1u, ringStart, (index + 1u) >= ringEnd);
          }
        }
      }
    }
  }
  edgeFeature[edgeFeatureOffset + index] = feature;
  edgeNext[edgeNextOffset + index] = next;`
    })
  );

  // 2. Rows touched by every edge, and the per-feature total as a 64-bit (low, high) pair.
  const edgeSlots = view('edge-slots', vertexCount + 1);
  const demand = view('demand', 2 * featureCount);
  nodes.push(
    createFillNode<Parameters>(graph, {
      id: make('clear-demand'),
      operation,
      view: demand,
      type: 'u32',
      value: '0u'
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: make('rows'),
      operation,
      variant: 'scanline-rows',
      bindings: [
        ...polygonBindings,
        {name: 'ranges', view: ranges, type: 'u32', access: 'read'},
        {name: 'edgeFeature', view: edgeFeature, type: 'u32', access: 'read'},
        {name: 'edgeNext', view: edgeNext, type: 'u32', access: 'read'},
        {name: 'edgeSlots', view: edgeSlots, type: 'u32', access: 'read_write'},
        {name: 'demand', view: demand, type: 'atomic<u32>', access: 'read_write'}
      ],
      invocationCount: vertexCount + 1,
      declarations: `${featureConstants}
${lattice}${GEOMETRY_WGSL}`,
      body: `var count = 0u;
  let feature = edgeFeature[edgeFeatureOffset + index];
  if (feature != ${NONE}) {
    let a = coverGetVertex(index);
    let b = coverGetVertex(edgeNext[edgeNextOffset + index]);
    let rows = coverEdgeRows(rangesOffset + feature * STRIDE, min(a.y, b.y), max(a.y, b.y));
    if (rows.x <= rows.y) {
      count = u32(rows.y - rows.x + 1);
      // Exact 64-bit total: a wrapped low word carries one into the high word.
      let before = atomicAdd(&demand[demandOffset + 2u * feature], count);
      if (before + count < before) {
        atomicAdd(&demand[demandOffset + 2u * feature + 1u], 1u);
      }
    }
  }
  edgeSlots[edgeSlotsOffset + index] = count;`
    })
  );

  // 3. Per-feature choice. cost = slots + rows (zero: bounding box), capped at capacity + 1.
  const cost = view('cost', featureCount + 1);
  const costStarts = view('cost-starts', featureCount + 1);
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: make('choose'),
      operation,
      variant: 'scanline-choose',
      bindings: [
        {name: 'counts', view: counts, type: 'u32', access: 'read'},
        {name: 'ranges', view: ranges, type: 'u32', access: 'read'},
        {name: 'demand', view: demand, type: 'u32', access: 'read'},
        {name: 'cost', view: cost, type: 'u32', access: 'read_write'}
      ],
      invocationCount: featureCount + 1,
      declarations: `${featureConstants}
const FORCE: bool = ${force};
const COST_RATIO: f32 = ${getWGSLFloatLiteral(CELL_COVER_SCANLINE_COST_RATIO)};
${lattice}`,
      body: `var featureCost = 0u;
  if (index < FEATURE_COUNT) {
    let rb = rangesOffset + index * STRIDE;
    if (counts[countsOffset + index] > 0u && (ranges[rb + 5u] & 1u) == 0u) {
      let rows = coverRowCount(rb);
      let slots = select(demand[demandOffset + 2u * index], 0xffffffffu, (demand[demandOffset + 2u * index + 1u] != 0u));
      let cells = f32(rows) * f32(coverColumnCount(rb));
      // Work of the scanline enumeration (events and runs per row, one slot per edge row) against
      // the cells of the bounding box. The row count and the slots must also fit the stage.
      let cheaper = cells > COST_RATIO * (2.0 * f32(rows) + f32(slots));
      if ((FORCE || cheaper) && rows <= SLOT_CAPACITY && slots <= SLOT_CAPACITY) {
        featureCost = min(slots + rows, SLOT_CAPACITY + 1u);
      }
    }
  }
  cost[costOffset + index] = featureCost;`
    }),
    ...new GPUScan({
      id: make('cost-scan'),
      input: cost,
      output: costStarts,
      mode: 'exclusive'
    }).getCommandNodes(graph)
  );

  // 4. Selection: a feature whose cost fits after the earlier candidates takes the scanline path.
  // `countCarry` is the high word of its candidate count (the low word is `counts`).
  const countCarry = view('count-carry', featureCount);
  // Flag word 5: bit 0 non-finite vertex (count kernel), bit 1 scanline. Words 6 and 7 receive the
  // feature's first run and run count.
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: make('select'),
      operation,
      variant: 'scanline-select',
      bindings: [
        {name: 'counts', view: counts, type: 'u32', access: 'read_write'},
        {name: 'ranges', view: ranges, type: 'u32', access: 'read_write'},
        {name: 'cost', view: cost, type: 'u32', access: 'read'},
        {name: 'costStarts', view: costStarts, type: 'u32', access: 'read'},
        {name: 'countCarry', view: countCarry, type: 'u32', access: 'read_write'}
      ],
      invocationCount: featureCount,
      declarations: featureConstants,
      body: `let rb = rangesOffset + index * STRIDE;
  countCarry[countCarryOffset + index] = 0u;
  let featureCost = cost[costOffset + index];
  // The sum of capped costs stays below 2^32 because (featureCount + 1) * (capacity + 1) does.
  let selected = featureCost > 0u && (costStarts[costStartsOffset + index] + featureCost) <= SLOT_CAPACITY;
  ranges[rb + 5u] = (ranges[rb + 5u] & 1u) | select(0u, 2u, selected);
  ranges[rb + 6u] = 0u;
  ranges[rb + 7u] = 0u;
  if (selected) {
    counts[countsOffset + index] = 0u;
  }`
    })
  );

  // 5. Slots of selected features only, then offsets.
  const edgeOffsets = view('edge-offsets', vertexCount + 1);
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: make('slots'),
      operation,
      variant: 'scanline-slots',
      bindings: [
        {name: 'ranges', view: ranges, type: 'u32', access: 'read'},
        {name: 'edgeFeature', view: edgeFeature, type: 'u32', access: 'read'},
        {name: 'edgeSlots', view: edgeSlots, type: 'u32', access: 'read_write'}
      ],
      invocationCount: vertexCount + 1,
      declarations: featureConstants,
      body: `let feature = edgeFeature[edgeFeatureOffset + index];
  if (feature == ${NONE} || (ranges[rangesOffset + feature * STRIDE + 5u] & 2u) == 0u) {
    edgeSlots[edgeSlotsOffset + index] = 0u;
  }`
    }),
    ...new GPUScan({
      id: make('edge-scan'),
      input: edgeSlots,
      output: edgeOffsets,
      mode: 'exclusive'
    }).getCommandNodes(graph)
  );

  // 6. Events: three per slot (span open, span close, crossing), invalid ones in row INVALID_ROW.
  const eventRow = view('event-row', eventCount);
  const eventPosition = view('event-position', eventCount);
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: make('events'),
      operation,
      variant: 'scanline-events',
      bindings: [
        ...polygonBindings,
        {name: 'ranges', view: ranges, type: 'u32', access: 'read'},
        {name: 'edgeFeature', view: edgeFeature, type: 'u32', access: 'read'},
        {name: 'edgeNext', view: edgeNext, type: 'u32', access: 'read'},
        {name: 'edgeOffsets', view: edgeOffsets, type: 'u32', access: 'read'},
        {name: 'costStarts', view: costStarts, type: 'u32', access: 'read'},
        {name: 'eventRow', view: eventRow, type: 'u32', access: 'read_write'},
        {name: 'eventPosition', view: eventPosition, type: 'u32', access: 'read_write'}
      ],
      invocationCount: slotCapacity,
      declarations: `${featureConstants}
${lattice}${GEOMETRY_WGSL}`,
      body: `var rowOpen = INVALID_ROW;
  var rowClose = INVALID_ROW;
  var rowCross = INVALID_ROW;
  var keyOpen = 0u;
  var keyClose = 0u;
  var keyCross = 0u;
  if (index < edgeOffsets[edgeOffsetsOffset + VERTEX_COUNT]) {
    // The edge owning slot index: the last edge whose first slot is at most index.
    var low = 0u;
    var high = VERTEX_COUNT;
    while (low < high) {
      let middle = (low + high) >> 1u;
      if (edgeOffsets[edgeOffsetsOffset + middle] <= index) {
        low = middle + 1u;
      } else {
        high = middle;
      }
    }
    let edge = low - 1u;
    let feature = edgeFeature[edgeFeatureOffset + edge];
    if (feature != ${NONE}) {
      let rb = rangesOffset + feature * STRIDE;
      let a = coverGetVertex(edge);
      let b = coverGetVertex(edgeNext[edgeNextOffset + edge]);
      let rows = coverEdgeRows(rb, min(a.y, b.y), max(a.y, b.y));
      let row = rows.x + i32(index - edgeOffsets[edgeOffsetsOffset + edge]);
      if (row <= rows.y) {
        let rowKey = costStarts[costStartsOffset + feature] + u32(row);
        let clip = coverClipEdge(a, b, coverRowBand(rb, u32(row)));
        if (clip.x > 0.0) {
          rowOpen = rowKey;
          rowClose = rowKey;
          keyOpen = coverSortableKey(clip.y);
          keyClose = coverSortableKey(clip.z);
        }
        let line = coverRowLine(rb, u32(row));
        if ((a.y <= line) != (b.y <= line)) {
          rowCross = rowKey;
          keyCross = coverSortableKey(coverGetCrossing(a, b, line));
        }
      }
    }
  }
  let base = 3u * index;
  eventRow[eventRowOffset + base] = rowOpen;
  eventRow[eventRowOffset + base + 1u] = rowClose;
  eventRow[eventRowOffset + base + 2u] = rowCross;
  eventPosition[eventPositionOffset + base] = keyOpen;
  eventPosition[eventPositionOffset + base + 1u] = keyClose;
  eventPosition[eventPositionOffset + base + 2u] = keyCross;`
    })
  );

  // 7. Sort events by (row, position), stable.
  const eventIds = view('event-ids', eventCount);
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: make('iota'),
      operation,
      variant: 'scanline-iota',
      bindings: [{name: 'eventIds', view: eventIds, type: 'u32', access: 'read_write'}],
      invocationCount: eventCount,
      body: 'eventIds[eventIdsOffset + index] = index;'
    })
  );
  const sort = createStableLexicographicIndexSortNodes<Parameters>(graph, {
    id: make('sort'),
    operation,
    indices: eventIds,
    keys: [{view: eventRow, keyBits: getSortKeyBits(slotCapacity)}, {view: eventPosition}]
  });
  nodes.push(...sort.nodes);
  const sortedIndices = sort.sortedIndices;
  const sortedRows = sort.sortedPrimaryKeys;

  // 8. Depth (spans) and crossing parity prefix sums per row.
  const depthDelta = view('depth-delta', eventCount);
  const crossingFlag = view('crossing-flag', eventCount);
  const segmentFlags = view('segment-flags', eventCount);
  const depthSum = view('depth-sum', eventCount);
  const crossingSum = view('crossing-sum', eventCount);
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: make('flags'),
      operation,
      variant: 'scanline-flags',
      bindings: [
        {name: 'sortedIndices', view: sortedIndices, type: 'u32', access: 'read'},
        {name: 'sortedRows', view: sortedRows, type: 'u32', access: 'read'},
        {name: 'depthDelta', view: depthDelta, type: 'u32', access: 'read_write'},
        {name: 'crossingFlag', view: crossingFlag, type: 'u32', access: 'read_write'},
        {name: 'segmentFlags', view: segmentFlags, type: 'u32', access: 'read_write'}
      ],
      invocationCount: eventCount,
      declarations: featureConstants,
      body: `let row = sortedRows[sortedRowsOffset + index];
  let valid = row < INVALID_ROW;
  // Event e is slot e / 3: kind 0 opens a span, 1 closes it, 2 is a crossing.
  let kind = sortedIndices[sortedIndicesOffset + index] % 3u;
  var head = index == 0u;
  if (!head) {
    head = row != sortedRows[sortedRowsOffset + index - 1u];
  }
  segmentFlags[segmentFlagsOffset + index] = select(0u, 1u, head);
  depthDelta[depthDeltaOffset + index] = select(0u, select(0u, select(0xffffffffu, 1u, (kind == 0u)), (kind < 2u)), valid);
  crossingFlag[crossingFlagOffset + index] = select(0u, 1u, (valid && kind == 2u));`
    }),
    ...new GPUScan({
      id: make('depth-scan'),
      input: depthDelta,
      output: depthSum,
      mode: 'inclusive',
      segmentFlags
    }).getCommandNodes(graph),
    ...new GPUScan({
      id: make('crossing-scan'),
      input: crossingFlag,
      output: crossingSum,
      mode: 'inclusive',
      segmentFlags
    }).getCommandNodes(graph)
  );

  // 9. Kept gaps: between consecutive events of a row, covered by a span or inside the polygon.
  // `covered` is rewritten in place with the final keep flag once the column range is known.
  const covered = view('covered', eventCount);
  const gapColumns = view('gap-columns', 2 * eventCount);
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: make('cover-state'),
      operation,
      variant: 'scanline-cover-state',
      bindings: [
        {name: 'sortedRows', view: sortedRows, type: 'u32', access: 'read'},
        {name: 'depthSum', view: depthSum, type: 'u32', access: 'read'},
        {name: 'crossingSum', view: crossingSum, type: 'u32', access: 'read'},
        {name: 'covered', view: covered, type: 'u32', access: 'read_write'}
      ],
      invocationCount: eventCount,
      declarations: `${featureConstants}
const EVENT_COUNT: u32 = ${eventCount}u;`,
      body: `let row = sortedRows[sortedRowsOffset + index];
  var keep = false;
  // Depth is never negative after all events of one position, so nonzero means a span covers.
  if (row < INVALID_ROW && (index + 1u) < EVENT_COUNT && sortedRows[sortedRowsOffset + index + 1u] == row) {
    keep = depthSum[depthSumOffset + index] != 0u || (crossingSum[crossingSumOffset + index] & 1u) != 0u;
  }
  covered[coveredOffset + index] = select(0u, 1u, keep);`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: make('gaps'),
      operation,
      variant: 'scanline-gaps',
      bindings: [
        {name: 'sortedIndices', view: sortedIndices, type: 'u32', access: 'read'},
        {name: 'sortedRows', view: sortedRows, type: 'u32', access: 'read'},
        {name: 'eventPosition', view: eventPosition, type: 'u32', access: 'read'},
        {name: 'ranges', view: ranges, type: 'u32', access: 'read'},
        {name: 'costStarts', view: costStarts, type: 'u32', access: 'read'},
        {name: 'covered', view: covered, type: 'u32', access: 'read_write'},
        {name: 'gapColumns', view: gapColumns, type: 'u32', access: 'read_write'}
      ],
      invocationCount: eventCount,
      declarations: `${featureConstants}
${lattice}${SORTABLE_KEY_WGSL}
${getRangeSearchWGSL('coverFindCostFeature', 'costStarts', featureCount)}`,
      body: `if (covered[coveredOffset + index] == 0u) {
    return;
  }
  let lowKey = eventPosition[eventPositionOffset + sortedIndices[sortedIndicesOffset + index]];
  let highKey = eventPosition[eventPositionOffset + sortedIndices[sortedIndicesOffset + index + 1u]];
  var keep = 0u;
  // Zero-length gaps add no cell the neighbouring gaps and spans do not already contain.
  if (lowKey < highKey) {
    let feature = coverFindCostFeature(sortedRows[sortedRowsOffset + index]);
    let columns = coverColumnRange(rangesOffset + feature * STRIDE, coverFromSortableKey(lowKey), coverFromSortableKey(highKey));
    if (columns.x <= columns.y) {
      keep = 1u;
      gapColumns[gapColumnsOffset + 2u * index] = u32(columns.x);
      gapColumns[gapColumnsOffset + 2u * index + 1u] = u32(columns.y);
    }
  }
  covered[coveredOffset + index] = keep;`
    })
  );

  // 10. Compact the kept gaps into run records after the per-feature words of the range table.
  const gapSlots = view('gap-slots', eventCount);
  nodes.push(
    ...new GPUScan({
      id: make('gap-scan'),
      input: covered,
      output: gapSlots,
      mode: 'exclusive'
    }).getCommandNodes(graph),
    createWGSLKernelNode<Parameters>(graph, {
      id: make('compact'),
      operation,
      variant: 'scanline-compact',
      bindings: [
        {name: 'covered', view: covered, type: 'u32', access: 'read'},
        {name: 'gapSlots', view: gapSlots, type: 'u32', access: 'read'},
        {name: 'sortedRows', view: sortedRows, type: 'u32', access: 'read'},
        {name: 'gapColumns', view: gapColumns, type: 'u32', access: 'read'},
        {name: 'ranges', view: ranges, type: 'u32', access: 'read_write'}
      ],
      invocationCount: eventCount,
      declarations: featureConstants,
      body: `if (covered[coveredOffset + index] == 0u) {
    return;
  }
  let run = ${runBase} + gapSlots[gapSlotsOffset + index] * ${CELL_COVER_SCANLINE_RUN_STRIDE}u;
  ranges[run] = sortedRows[sortedRowsOffset + index];
  ranges[run + 1u] = gapColumns[gapColumnsOffset + 2u * index];
  ranges[run + 2u] = gapColumns[gapColumnsOffset + 2u * index + 1u];`
    })
  );

  // 11. Make runs disjoint (a column range monotone in x only overlaps the previous run of its row),
  // find each run's feature and local row, and scan run lengths per feature.
  const runLengths = view('run-lengths', eventCount);
  const runOffsets = view('run-offsets', eventCount);
  const runFlags = view('run-flags', eventCount);
  const runTotal = `(gapSlots[gapSlotsOffset + ${eventCount - 1}u] + covered[coveredOffset + ${eventCount - 1}u])`;
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: make('runs'),
      operation,
      variant: 'scanline-runs',
      bindings: [
        {name: 'ranges', view: ranges, type: 'u32', access: 'read_write'},
        {name: 'costStarts', view: costStarts, type: 'u32', access: 'read'},
        {name: 'covered', view: covered, type: 'u32', access: 'read'},
        {name: 'gapSlots', view: gapSlots, type: 'u32', access: 'read'},
        {name: 'runLengths', view: runLengths, type: 'u32', access: 'read_write'},
        {name: 'runFlags', view: runFlags, type: 'u32', access: 'read_write'}
      ],
      invocationCount: eventCount,
      declarations: `${featureConstants}
${getRangeSearchWGSL('coverFindCostFeature', 'costStarts', featureCount)}`,
      body: `var runLength = 0u;
  var isHead = false;
  if (index < ${runTotal}) {
    let run = ${runBase} + index * ${CELL_COVER_SCANLINE_RUN_STRIDE}u;
    let rowKey = ranges[run];
    let columnLow = ranges[run + 1u];
    let columnHigh = ranges[run + 2u];
    let feature = coverFindCostFeature(rowKey);
    var start = columnLow;
    isHead = true;
    if (index > 0u) {
      let previous = run - ${CELL_COVER_SCANLINE_RUN_STRIDE}u;
      let previousRow = ranges[previous];
      if (previousRow == rowKey) {
        start = max(start, ranges[previous + 2u] + 1u);
      }
      isHead = coverFindCostFeature(previousRow) != feature;
    }
    runLength = select(0u, columnHigh + 1u - start, (columnHigh >= start));
    ranges[run + 3u] = feature;
    ranges[run + 5u] = start;
    ranges[run + 6u] = rowKey - costStarts[costStartsOffset + feature];
  }
  runLengths[runLengthsOffset + index] = runLength;
  runFlags[runFlagsOffset + index] = select(0u, 1u, isHead);`
    }),
    ...new GPUScan({
      id: make('run-scan'),
      input: runLengths,
      output: runOffsets,
      mode: 'exclusive',
      segmentFlags: runFlags
    }).getCommandNodes(graph),
    createWGSLKernelNode<Parameters>(graph, {
      id: make('finish'),
      operation,
      variant: 'scanline-finish',
      bindings: [
        {name: 'ranges', view: ranges, type: 'atomic<u32>', access: 'read_write'},
        {name: 'runOffsets', view: runOffsets, type: 'u32', access: 'read'},
        {name: 'runLengths', view: runLengths, type: 'u32', access: 'read'},
        {name: 'covered', view: covered, type: 'u32', access: 'read'},
        {name: 'gapSlots', view: gapSlots, type: 'u32', access: 'read'},
        {name: 'counts', view: counts, type: 'atomic<u32>', access: 'read_write'},
        {name: 'countCarry', view: countCarry, type: 'atomic<u32>', access: 'read_write'}
      ],
      invocationCount: eventCount,
      declarations: featureConstants,
      body: `if (index >= ${runTotal}) {
    return;
  }
  let run = ${runBase} + index * ${CELL_COVER_SCANLINE_RUN_STRIDE}u;
  let feature = atomicLoad(&ranges[run + 3u]);
  atomicStore(&ranges[run + 4u], runOffsets[runOffsetsOffset + index]);
  let featureBase = rangesOffset + feature * STRIDE;
  if (index == 0u || atomicLoad(&ranges[run - ${CELL_COVER_SCANLINE_RUN_STRIDE}u + 3u]) != feature) {
    atomicStore(&ranges[featureBase + 6u], index);
  }
  atomicAdd(&ranges[featureBase + 7u], 1u);
  // Exact 64-bit candidate count: a wrapped low word carries one into the high word.
  let runLength = runLengths[runLengthsOffset + index];
  let before = atomicAdd(&counts[countsOffset + feature], runLength);
  if (before + runLength < before) {
    atomicAdd(&countCarry[countCarryOffset + feature], 1u);
  }`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: make('clamp'),
      operation,
      variant: 'scanline-clamp',
      bindings: [
        {name: 'ranges', view: ranges, type: 'u32', access: 'read_write'},
        {name: 'counts', view: counts, type: 'u32', access: 'read_write'},
        {name: 'countCarry', view: countCarry, type: 'u32', access: 'read'}
      ],
      invocationCount: featureCount,
      declarations: `${featureConstants}
const CAPACITY_PLUS_ONE: u32 = ${candidateCapacity + 1}u;`,
      body: `let flagIndex = rangesOffset + index * STRIDE + 5u;
  if ((ranges[flagIndex] & 2u) != 0u) {
    counts[countsOffset + index] = min(counts[countsOffset + index], CAPACITY_PLUS_ONE);
    // 2^32 or more candidates: far beyond any capacity, and the 32-bit run offsets of the feature
    // wrapped. Its count saturates (candidate overflow) and it falls back to the bounding box.
    if (countCarry[countCarryOffset + index] != 0u) {
      counts[countsOffset + index] = CAPACITY_PLUS_ONE;
      ranges[flagIndex] = ranges[flagIndex] & 1u;
    }
  }`
    })
  );
  return nodes;
}

/**
 * WGSL that decodes a candidate of a scanline feature into its lattice `column` and `row`
 * (relative to the lattice origin) from `candidateLocal`. Needs `ranges`, `rangeBase`,
 * `candidateLocal` in scope and defines `column` and `row`; for other features the bounding-box
 * decode `lattice width` applies. @internal
 */
export function getCellCoverScanlineDecodeWGSL(featureCount: number, width: string): string {
  const runBase = `rangesOffset + ${featureCount}u * ${CELL_COVER_SCANLINE_RANGE_STRIDE}u`;
  return `var column = candidateLocal % ${width};
  var row = candidateLocal / ${width};
  if ((ranges[rangeBase + 5u] & 2u) != 0u) {
    // Runs of this feature are [first, first + runCount); offsets are local to the feature, and
    // the run holding the candidate is the last one that starts at or before it.
    let firstRun = ranges[rangeBase + 6u];
    var low = firstRun;
    var high = firstRun + ranges[rangeBase + 7u];
    while (low < high) {
      let middle = (low + high) >> 1u;
      if (ranges[${runBase} + middle * ${CELL_COVER_SCANLINE_RUN_STRIDE}u + 4u] <= candidateLocal) {
        low = middle + 1u;
      } else {
        high = middle;
      }
    }
    let run = ${runBase} + (low - 1u) * ${CELL_COVER_SCANLINE_RUN_STRIDE}u;
    column = ranges[run + 5u] + (candidateLocal - ranges[run + 4u]);
    row = ranges[run + 6u];
  }`;
}
