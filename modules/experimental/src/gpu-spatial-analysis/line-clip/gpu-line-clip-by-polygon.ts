// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {validateGraphViewsBelongToGraph} from '../../utils/gpu-contributor-utils';
import {createStableLexicographicIndexSortNodes} from '../../utils/stable-lexicographic-index-sort';
import {GPUSegmentIntersection} from '../segment-intersection/index';
import {GPUPointInPolygonJoin, GPU_SPATIAL_JOIN_NO_FEATURE} from '../spatial-join/index';
import type {GPUSpatialJoinLines, GPUSpatialJoinPolygons} from '../spatial-join/index';
import type {GPULineSplitPieces} from '../line-split/index';

const OPERATION = 'GPULineClipByPolygon';

/** Number of `u32` words per crossing-event row. */
const EVENT_STRIDE = 8;

/** Number of `u32` words per sub-piece row. */
const SUB_PIECE_STRIDE = 8;

/** Sub-piece flag bits (word 6 of a sub-piece row). */
const FLAG_BOUNDARY = 1;
const FLAG_KEPT = 2;
const FLAG_VALID = 4;

/**
 * Properties for {@link GPULineClipByPolygon}.
 *
 * Per-frame: the contents of every input buffer. Topology: view lengths, capacities, `mode`,
 * `leafCapacity`, `spatialSort`.
 */
export type GPULineClipByPolygonProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'line-clip-by-polygon'`. */
  id?: string;
  /** Linestrings to clip. */
  lines: GPUSpatialJoinLines;
  /**
   * Clip polygons. Several features (and several polygons per feature) act as their union, as
   * GeoPandas `clip` dissolves its mask. Containment uses even/odd fill per feature, so features
   * must be valid; see {@link GPUSpatialJoinPolygons}.
   */
  polygons: GPUSpatialJoinPolygons;
  /**
   * Compile-time. `'inside'` (default) keeps the parts of every line that lie inside the polygons
   * or on their boundary (GeoPandas `clip(lines, polygons)`, Shapely `intersection`). `'outside'`
   * keeps the parts strictly outside the polygons (Shapely `difference`); a part running along a
   * polygon boundary belongs to the polygons and is removed.
   */
  mode?: 'inside' | 'outside';
  /**
   * Capacity of the internal `GPUSegmentIntersection` pair list, which holds one row per
   * (line segment, polygon segment) intersection. When it overflows, pieces are produced from the
   * sorted prefix of pairs and `pieces.status.overflow` is set.
   */
  intersectionCapacity: number;
  /**
   * Maximum (sub-piece midpoint, polygon feature) bounding-box candidates of the internal
   * `GPUPointInPolygonJoin`. Overflow sets `pieces.status.overflow`.
   */
  candidateCapacity: number;
  /** Output pieces, in the layout of {@link GPULineSplitPieces}. */
  pieces: GPULineSplitPieces;
  /** Passed to `GPUSegmentIntersection`. */
  spatialSort?: boolean;
  /** Passed to `GPUSegmentIntersection` (the BVH over polygon segments). */
  leafCapacity?: number;
  /**
   * Optional one-row count of segment pairs whose intersection could not be certified plus
   * sub-piece midpoints whose containment could not be certified. A nonzero value means the
   * result may be incomplete: uncertain crossings are ignored and uncertain midpoints count as outside.
   */
  uncertainCount?: GraphDataView<'uint32'>;
};

/**
 * Clips linestrings to polygons: GeoPandas `clip(lines, polygons)` and Shapely
 * `intersection(line, polygon)`, or with `mode: 'outside'` Shapely `difference(line, polygon)`.
 *
 * Pipeline: `GPUSegmentIntersection` (two-sided, lines against polygon rings) lists every
 * crossing, touch and collinear overlap. Pairs arrive sorted by line segment, so the crossings of
 * one segment are contiguous; each is ordered along the segment by rank (no sort pass) and
 * deduplicated, which cuts the segment into sub-pieces. A sub-piece lying inside a collinear
 * overlap span is on the boundary; every other sub-piece is classified by
 * `GPUPointInPolygonJoin` on its midpoint, with robust predicates. Boundary sub-pieces count as
 * inside. Kept sub-pieces that continue each other (same line, shared end point) are merged into
 * one piece, so a line that never leaves the polygon comes out whole. Crossing points that fall
 * inside a kept run are not emitted as vertices; original line vertices always are.
 *
 * Output order is deterministic: pieces follow the input line order and, within a line, the
 * traversal order. Pieces are linestrings with at least two vertices; a line that only touches a
 * polygon at a point yields no piece (GeoPandas `keep_geom_type=True`). A line that is clipped to
 * nothing yields no row, and a closed line that is kept whole is not rejoined at its start. Pieces
 * whose end would exceed the capacities are dropped as a suffix and `pieces.status.overflow` is set.
 *
 * Crossings of proper intersections are rounded to f32 once per pair. A sub-piece shorter than
 * a few f32 ulps next to a polygon edge can have its midpoint classified on the wrong side.
 * Nothing is read back.
 */
export class GPULineClipByPolygon implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPULineClipByPolygonProps;
  /** Number of vertices (segment slots) of the input lines. */
  readonly vertexCount: number;
  /** Number of input linestrings. */
  readonly lineCount: number;
  /** Piece capacity. */
  readonly pieceCapacity: number;
  /** Output vertex capacity. */
  readonly vertexCapacity: number;
  /** Whether the parts inside the polygons are kept. */
  readonly keepInside: boolean;

  constructor(props: GPULineClipByPolygonProps) {
    this.id = props.id ?? 'line-clip-by-polygon';
    this.props = props;
    const {id} = this;
    const {lines, polygons, pieces} = props;
    if (lines.kind !== 'lines') {
      throw new Error(`${id} lines must be linestring geometry`);
    }
    if (polygons.kind !== 'polygons') {
      throw new Error(`${id} polygons must be polygon geometry`);
    }
    validatePackedView(lines.positions, ['float32x2'], `${id} lines.positions`);
    validatePackedUint32View(lines.lineOffsets, `${id} lines.lineOffsets`);
    validatePackedView(polygons.positions, ['float32x2'], `${id} polygons.positions`);
    for (const [name, view] of [
      ['featureOffsets', polygons.featureOffsets],
      ['polygonOffsets', polygons.polygonOffsets],
      ['ringOffsets', polygons.ringOffsets]
    ] as const) {
      validatePackedUint32View(view, `${id} polygons.${name}`);
      if (view.length < 1) {
        throw new Error(`${id} polygons.${name} requires a terminal entry`);
      }
    }
    if (lines.lineOffsets.length < 2) {
      throw new Error(`${id} lines.lineOffsets requires at least one line and a terminal entry`);
    }
    if (lines.positions.length < 1 || lines.positions.length >= 0x80000000) {
      throw new Error(`${id} lines.positions must hold between 1 and 2^31 - 1 vertices`);
    }
    if (polygons.positions.length < 1) {
      throw new Error(`${id} polygons.positions must hold at least one vertex`);
    }
    if (props.mode !== undefined && props.mode !== 'inside' && props.mode !== 'outside') {
      throw new Error(`${id} mode must be 'inside' or 'outside'`);
    }
    for (const [name, value] of [
      ['intersectionCapacity', props.intersectionCapacity],
      ['candidateCapacity', props.candidateCapacity]
    ] as const) {
      if (!Number.isSafeInteger(value) || value < 1) {
        throw new Error(`${id} ${name} must be a positive integer`);
      }
    }
    this.keepInside = props.mode !== 'outside';
    this.vertexCount = lines.positions.length;
    this.lineCount = lines.lineOffsets.length - 1;
    validatePackedUint32View(pieces.sourceIds, `${id} pieces.sourceIds`);
    validatePackedUint32View(pieces.geometry.lineOffsets, `${id} pieces.geometry.lineOffsets`);
    validatePackedView(pieces.geometry.positions, ['float32x2'], `${id} pieces.geometry.positions`);
    this.pieceCapacity = pieces.sourceIds.length;
    this.vertexCapacity = pieces.geometry.positions.length;
    if (this.pieceCapacity < 1 || pieces.geometry.lineOffsets.length !== this.pieceCapacity + 1) {
      throw new Error(
        `${id} pieces.geometry.lineOffsets length must be pieces.sourceIds.length + 1`
      );
    }
    if (this.vertexCapacity < 1) {
      throw new Error(`${id} pieces.geometry.positions must be non-empty`);
    }
    for (const [name, view] of [
      ['count', pieces.status.count],
      ['vertexCount', pieces.vertexCount],
      ['overflow', pieces.status.overflow],
      ['requiredCount', pieces.status.requiredCount],
      ['requiredVertexCount', pieces.requiredVertexCount],
      ['uncertainCount', props.uncertainCount]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} ${name}`);
        if (view.length < 1) {
          throw new Error(`${id} ${name} must contain one uint32 row`);
        }
      }
    }
  }

  /** Returns intersection, ordering, classification, merge and output nodes in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, vertexCount, lineCount, pieceCapacity, vertexCapacity, keepInside} = this;
    const {lines, polygons, pieces} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      lines.positions,
      lines.lineOffsets,
      polygons.positions,
      polygons.featureOffsets,
      polygons.polygonOffsets,
      polygons.ringOffsets,
      pieces.sourceIds,
      pieces.geometry.lineOffsets,
      pieces.geometry.positions,
      pieces.status.count,
      pieces.vertexCount,
      pieces.status.overflow,
      pieces.status.requiredCount,
      pieces.requiredVertexCount,
      props.uncertainCount
    ]);
    const nodes: GPUCommandNode<Parameters>[] = [];
    const pairCapacity = props.intersectionCapacity;
    const eventCount = pairCapacity * 2;
    const subPieceBound = vertexCount + eventCount;
    const T = <Format extends 'uint32' | 'float32x2'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, `${id}-${name}`, format, length);

    // 1. Intersections between line segments and polygon segments.
    const pairLeft = T('pair-left', 'uint32', pairCapacity);
    const pairRight = T('pair-right', 'uint32', pairCapacity);
    const pairCount = T('pair-count', 'uint32', 1);
    const pairOverflow = T('pair-overflow', 'uint32', 1);
    const kinds = T('kinds', 'uint32', pairCapacity);
    const points = T('points', 'float32x2', pairCapacity);
    const endPoints = T('end-points', 'float32x2', pairCapacity);
    const segmentUncertain = T('segment-uncertain', 'uint32', 1);
    nodes.push(
      ...new GPUSegmentIntersection({
        id: `${id}-intersection`,
        left: lines,
        right: polygons,
        spatialSort: props.spatialSort,
        leafCapacity: props.leafCapacity,
        pairs: {
          leftIds: pairLeft,
          rightIds: pairRight,
          count: pairCount,
          overflow: pairOverflow
        },
        uncertainCount: segmentUncertain,
        kinds,
        points,
        endPoints
      }).getCommandNodes(graph)
    );

    const lineLookupWGSL = `
const VERTEX_COUNT: u32 = ${vertexCount}u;
const LINE_COUNT: u32 = ${lineCount}u;
fn vertexAt(vertex: u32) -> vec2f {
  return vec2f(positions[positionsOffset + vertex * 2u], positions[positionsOffset + vertex * 2u + 1u]);
}
fn isFiniteValue(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }
// Largest line whose first vertex is at most vertex.
fn lineOfVertex(vertex: u32) -> u32 {
  var low = 0u;
  var high = LINE_COUNT - 1u;
  while (low < high) {
    let middle = (low + high + 1u) / 2u;
    if (lineOffsets[lineOffsetsOffset + middle] <= vertex) { low = middle; } else { high = middle - 1u; }
  }
  return low;
}
// Coordinate of p along the segment direction d, on the dominant axis; ascending along d.
fn keyOf(p: vec2f, d: vec2f) -> f32 {
  if (abs(d.x) >= abs(d.y)) { return p.x * select(-1.0, 1.0, d.x > 0.0); }
  return p.y * select(-1.0, 1.0, d.y > 0.0);
}
fn otherOf(p: vec2f, d: vec2f) -> f32 {
  if (abs(d.x) >= abs(d.y)) { return p.y * select(-1.0, 1.0, d.y >= 0.0); }
  return p.x * select(-1.0, 1.0, d.x >= 0.0);
}`;

    // 2. Per segment slot: validity, line, and the range of its pairs (pairs are sorted by left).
    const segmentTable = T('segment-table', 'uint32', vertexCount * 4);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-segment-ranges`,
        operation: OPERATION,
        variant: 'segment-ranges',
        bindings: [
          {name: 'positions', view: lines.positions, type: 'f32', access: 'read'},
          {name: 'lineOffsets', view: lines.lineOffsets, type: 'u32', access: 'read'},
          {name: 'pairLeft', view: pairLeft, type: 'u32', access: 'read'},
          {name: 'pairCount', view: pairCount, type: 'u32', access: 'read'},
          {name: 'segmentTable', view: segmentTable, type: 'u32', access: 'read_write'}
        ],
        invocationCount: vertexCount,
        declarations: `${lineLookupWGSL}
fn lowerBound(value: u32) -> u32 {
  var low = 0u;
  var high = pairCount[pairCountOffset];
  while (low < high) {
    let middle = (low + high) / 2u;
    if (pairLeft[pairLeftOffset + middle] < value) { low = middle + 1u; } else { high = middle; }
  }
  return low;
}`,
        body: `let line = lineOfVertex(index);
  let lineEnd = lineOffsets[lineOffsetsOffset + line + 1u];
  let a = vertexAt(index);
  var valid = index + 1u < lineEnd && index + 1u < VERTEX_COUNT;
  var b = a;
  if (valid) { b = vertexAt(index + 1u); }
  valid = valid && isFiniteValue(a.x) && isFiniteValue(a.y) && isFiniteValue(b.x) && isFiniteValue(b.y) &&
    (a.x != b.x || a.y != b.y);
  var first = lowerBound(index);
  var last = lowerBound(index + 1u);
  if (!valid) { last = first; }
  let row = segmentTableOffset + index * 4u;
  segmentTable[row] = first;
  segmentTable[row + 1u] = last;
  segmentTable[row + 2u] = select(0u, 1u, valid);
  segmentTable[row + 3u] = line;`
      })
    );

    // 3. Crossing events: up to two per pair (both ends of an overlap).
    const eventTable = T('event-table', 'uint32', eventCount * EVENT_STRIDE);
    const eventSegments = T('event-segments', 'uint32', eventCount);
    const eventOrderKeys = T('event-order-keys', 'uint32', eventCount);
    const eventOtherKeys = T('event-other-keys', 'uint32', eventCount);
    const eventIndices = T('event-indices', 'uint32', eventCount);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-events`,
        operation: OPERATION,
        variant: 'events',
        bindings: [
          {name: 'positions', view: lines.positions, type: 'f32', access: 'read'},
          {name: 'pairLeft', view: pairLeft, type: 'u32', access: 'read'},
          {name: 'pairCount', view: pairCount, type: 'u32', access: 'read'},
          {name: 'kinds', view: kinds, type: 'u32', access: 'read'},
          {name: 'points', view: points, type: 'f32', access: 'read'},
          {name: 'endPoints', view: endPoints, type: 'f32', access: 'read'},
          {name: 'eventTable', view: eventTable, type: 'u32', access: 'read_write'}
        ],
        invocationCount: eventCount,
        declarations: `const VERTEX_COUNT: u32 = ${vertexCount}u;
fn vertexAt(vertex: u32) -> vec2f {
  return vec2f(positions[positionsOffset + vertex * 2u], positions[positionsOffset + vertex * 2u + 1u]);
}
${lineLookupWGSL.slice(lineLookupWGSL.indexOf('// Coordinate of p'))}`,
        body: `let pair = index / 2u;
  let which = index % 2u;
  var flags = 0u;
  var point = vec2f(0.0);
  var key = 0.0;
  var other = 0.0;
  var kind = 0u;
  if (pair < pairCount[pairCountOffset]) { kind = kinds[kindsOffset + pair]; }
  if (kind >= 1u && kind <= 4u) {
    let segment = pairLeft[pairLeftOffset + pair];
    let a = vertexAt(segment);
    let b = vertexAt(segment + 1u);
    let direction = b - a;
    if (which == 0u) {
      point = vec2f(points[pointsOffset + pair * 2u], points[pointsOffset + pair * 2u + 1u]);
    } else {
      point = vec2f(endPoints[endPointsOffset + pair * 2u], endPoints[endPointsOffset + pair * 2u + 1u]);
    }
    key = keyOf(point, direction);
    other = otherOf(point, direction);
    let interior = !(point.x == a.x && point.y == a.y) && !(point.x == b.x && point.y == b.y);
    if (interior && (which == 0u || kind == 4u)) { flags = 1u; }
    if (kind == 4u) { flags = flags | 2u; }
  }
  let row = eventTableOffset + index * ${EVENT_STRIDE}u;
  eventTable[row] = flags;
  eventTable[row + 1u] = 0u;
  eventTable[row + 2u] = 0u;
  eventTable[row + 3u] = bitcast<u32>(point.x);
  eventTable[row + 4u] = bitcast<u32>(point.y);
  eventTable[row + 5u] = bitcast<u32>(key);
  eventTable[row + 6u] = bitcast<u32>(other);
  eventTable[row + 7u] = 0u;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-event-sort-keys`,
        operation: OPERATION,
        variant: 'event-sort-keys',
        bindings: [
          {name: 'pairLeft', view: pairLeft, type: 'u32', access: 'read'},
          {name: 'eventTable', view: eventTable, type: 'u32', access: 'read'},
          {name: 'eventSegments', view: eventSegments, type: 'u32', access: 'read_write'},
          {name: 'eventOrderKeys', view: eventOrderKeys, type: 'u32', access: 'read_write'},
          {name: 'eventOtherKeys', view: eventOtherKeys, type: 'u32', access: 'read_write'},
          {name: 'eventIndices', view: eventIndices, type: 'u32', access: 'read_write'}
        ],
        invocationCount: eventCount,
        declarations: `const VERTEX_COUNT: u32 = ${vertexCount}u;`,
        body: `let row = eventTableOffset + index * ${EVENT_STRIDE}u;
  let valid = (eventTable[row] & 1u) != 0u;
  let keyBits = eventTable[row + 5u];
  let otherBits = eventTable[row + 6u];
  eventSegments[eventSegmentsOffset + index] = select(VERTEX_COUNT, pairLeft[pairLeftOffset + index / 2u], valid);
  eventOrderKeys[eventOrderKeysOffset + index] = select(keyBits ^ 0xffffffffu, keyBits ^ 0x80000000u, (keyBits & 0x80000000u) == 0u);
  eventOtherKeys[eventOtherKeysOffset + index] = select(otherBits ^ 0xffffffffu, otherBits ^ 0x80000000u, (otherBits & 0x80000000u) == 0u);
  eventIndices[eventIndicesOffset + index] = index;`
      })
    );

    // 4. Order, deduplicate and rank the interior events of each segment. Stable tuple passes
    // distribute a dense segment over the radix workgroups instead of making every event scan the
    // full segment range (quadratic work and capacity-wide dependent depth).
    const eventAccessWGSL = `
fn eventPointBits(event: u32) -> vec2u {
  let row = eventTableOffset + event * ${EVENT_STRIDE}u;
  return vec2u(eventTable[row + 3u], eventTable[row + 4u]);
}
fn eventPoint(event: u32) -> vec2f { return bitcast<vec2f>(eventPointBits(event)); }
fn eventKey(event: u32) -> f32 { return bitcast<f32>(eventTable[eventTableOffset + event * ${EVENT_STRIDE}u + 5u]); }
`;
    const eventSort = createStableLexicographicIndexSortNodes(graph, {
      id: `${id}-event-order`,
      operation: OPERATION,
      indices: eventIndices,
      keys: [{view: eventSegments}, {view: eventOrderKeys}, {view: eventOtherKeys}]
    });
    nodes.push(...eventSort.nodes);
    const sortedSegments = eventSort.sortedPrimaryKeys;
    const sortedEvents = eventSort.sortedIndices;
    const uniqueFlags = T('event-unique-flags', 'uint32', eventCount);
    const uniqueRanks = T('event-unique-ranks', 'uint32', eventCount);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-event-unique`,
        operation: OPERATION,
        variant: 'event-unique',
        bindings: [
          {name: 'sortedSegments', view: sortedSegments, type: 'u32', access: 'read'},
          {name: 'sortedEvents', view: sortedEvents, type: 'u32', access: 'read'},
          {name: 'eventTable', view: eventTable, type: 'u32', access: 'read'},
          {name: 'uniqueFlags', view: uniqueFlags, type: 'u32', access: 'read_write'}
        ],
        invocationCount: eventCount,
        declarations: `const VERTEX_COUNT: u32 = ${vertexCount}u;
${eventAccessWGSL}`,
        body: `let segment = sortedSegments[sortedSegmentsOffset + index];
  var unique = 1u;
  if (segment >= VERTEX_COUNT) { unique = 0u; }
  if (unique != 0u && index > 0u && sortedSegments[sortedSegmentsOffset + index - 1u] == segment) {
    let event = sortedEvents[sortedEventsOffset + index];
    let previous = sortedEvents[sortedEventsOffset + index - 1u];
    if (all(eventPointBits(event) == eventPointBits(previous))) { unique = 0u; }
  }
  uniqueFlags[uniqueFlagsOffset + index] = unique;`
      }),
      ...new GPUScan({
        id: `${id}-event-unique-scan`,
        input: uniqueFlags,
        output: uniqueRanks,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-event-rank`,
        operation: OPERATION,
        variant: 'event-rank',
        bindings: [
          {name: 'sortedSegments', view: sortedSegments, type: 'u32', access: 'read'},
          {name: 'sortedEvents', view: sortedEvents, type: 'u32', access: 'read'},
          {name: 'uniqueFlags', view: uniqueFlags, type: 'u32', access: 'read'},
          {name: 'uniqueRanks', view: uniqueRanks, type: 'u32', access: 'read'},
          {name: 'eventTable', view: eventTable, type: 'u32', access: 'read_write'}
        ],
        invocationCount: eventCount,
        declarations: `const EVENT_COUNT: u32 = ${eventCount}u;
fn lowerBoundSegment(value: u32) -> u32 {
  var low = 0u;
  var high = EVENT_COUNT;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (sortedSegments[sortedSegmentsOffset + middle] < value) { low = middle + 1u; } else { high = middle; }
  }
  return low;
}`,
        body: `if (uniqueFlags[uniqueFlagsOffset + index] == 0u) { return; }
  let event = sortedEvents[sortedEventsOffset + index];
  let segment = sortedSegments[sortedSegmentsOffset + index];
  let first = lowerBoundSegment(segment);
  let rank = uniqueRanks[uniqueRanksOffset + index] - uniqueRanks[uniqueRanksOffset + first];
  eventTable[eventTableOffset + event * ${EVENT_STRIDE}u + 1u] = 1u;
  eventTable[eventTableOffset + event * ${EVENT_STRIDE}u + 2u] = rank;
  // Word 7 of the globally compacted unique-event row names the original event. Segment bases
  // turn local ranks into these O(1) lookups without reusing pair-table ranges.
  let globalRank = uniqueRanks[uniqueRanksOffset + index];
  eventTable[eventTableOffset + globalRank * ${EVENT_STRIDE}u + 7u] = event;`
      })
    );

    // 5. Sub-pieces per segment: unique interior events plus one.
    const subCounts = T('sub-counts', 'uint32', vertexCount);
    const segmentUniqueBases = T('segment-unique-bases', 'uint32', vertexCount);
    const subStarts = T('sub-starts', 'uint32', vertexCount);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-sub-counts`,
        operation: OPERATION,
        variant: 'sub-counts',
        bindings: [
          {name: 'segmentTable', view: segmentTable, type: 'u32', access: 'read'},
          {name: 'sortedSegments', view: sortedSegments, type: 'u32', access: 'read'},
          {name: 'uniqueFlags', view: uniqueFlags, type: 'u32', access: 'read'},
          {name: 'uniqueRanks', view: uniqueRanks, type: 'u32', access: 'read'},
          {name: 'segmentUniqueBases', view: segmentUniqueBases, type: 'u32', access: 'read_write'},
          {name: 'subCounts', view: subCounts, type: 'u32', access: 'read_write'}
        ],
        invocationCount: vertexCount,
        declarations: `const EVENT_COUNT: u32 = ${eventCount}u;
fn lowerBoundSegment(value: u32) -> u32 {
  var low = 0u;
  var high = EVENT_COUNT;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (sortedSegments[sortedSegmentsOffset + middle] < value) { low = middle + 1u; } else { high = middle; }
  }
  return low;
}
fn uniquePrefix(position: u32) -> u32 {
  if (position < EVENT_COUNT) { return uniqueRanks[uniqueRanksOffset + position]; }
  return uniqueRanks[uniqueRanksOffset + EVENT_COUNT - 1u] + uniqueFlags[uniqueFlagsOffset + EVENT_COUNT - 1u];
}`,
        body: `let row = segmentTableOffset + index * 4u;
  var count = 0u;
  if (segmentTable[row + 2u] != 0u) {
    let begin = lowerBoundSegment(index);
    let end = lowerBoundSegment(index + 1u);
    segmentUniqueBases[segmentUniqueBasesOffset + index] = uniquePrefix(begin);
    count = 1u + uniquePrefix(end) - uniquePrefix(begin);
  }
  subCounts[subCountsOffset + index] = count;`
      }),
      ...new GPUScan({
        id: `${id}-sub-scan`,
        input: subCounts,
        output: subStarts,
        mode: 'exclusive'
      }).getCommandNodes(graph)
    );

    // 6. Sub-pieces: end points, midpoint, line, segment and the boundary flag.
    const subTable = T('sub-table', 'uint32', subPieceBound * SUB_PIECE_STRIDE);
    const subMidpoints = T('sub-midpoints', 'float32x2', subPieceBound);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-sub-pieces`,
        operation: OPERATION,
        variant: 'sub-pieces',
        bindings: [
          {name: 'positions', view: lines.positions, type: 'f32', access: 'read'},
          {name: 'segmentTable', view: segmentTable, type: 'u32', access: 'read'},
          {name: 'eventTable', view: eventTable, type: 'u32', access: 'read'},
          {name: 'subStarts', view: subStarts, type: 'u32', access: 'read'},
          {name: 'subCounts', view: subCounts, type: 'u32', access: 'read'},
          {name: 'segmentUniqueBases', view: segmentUniqueBases, type: 'u32', access: 'read'},
          {name: 'subTable', view: subTable, type: 'u32', access: 'read_write'},
          {name: 'subMidpoints', view: subMidpoints, type: 'f32', access: 'read_write'}
        ],
        invocationCount: subPieceBound,
        declarations: `const VERTEX_COUNT: u32 = ${vertexCount}u;
fn vertexAt(vertex: u32) -> vec2f {
  return vec2f(positions[positionsOffset + vertex * 2u], positions[positionsOffset + vertex * 2u + 1u]);
}
${lineLookupWGSL.slice(lineLookupWGSL.indexOf('// Coordinate of p'))}
${eventAccessWGSL}
fn pointOfRank(base: u32, rank: u32) -> vec2f {
  return eventPoint(eventTable[eventTableOffset + (base + rank) * ${EVENT_STRIDE}u + 7u]);
}`,
        body: `let total = subStarts[subStartsOffset + VERTEX_COUNT - 1u] + subCounts[subCountsOffset + VERTEX_COUNT - 1u];
  let row = subTableOffset + index * ${SUB_PIECE_STRIDE}u;
  if (index >= total) {
    subTable[row + 6u] = 0u;
    subMidpoints[subMidpointsOffset + index * 2u] = 3.0e38;
    subMidpoints[subMidpointsOffset + index * 2u + 1u] = 3.0e38;
    return;
  }
  // Last segment whose first sub-piece is at most index; zero-count segments precede it.
  var low = 0u;
  var high = VERTEX_COUNT - 1u;
  while (low < high) {
    let middle = (low + high + 1u) / 2u;
    if (subStarts[subStartsOffset + middle] <= index) { low = middle; } else { high = middle - 1u; }
  }
  let segment = low;
  let local = index - subStarts[subStartsOffset + segment];
  let uniqueCount = subCounts[subCountsOffset + segment] - 1u;
  let first = segmentTable[segmentTableOffset + segment * 4u];
  let last = segmentTable[segmentTableOffset + segment * 4u + 1u];
  let uniqueBase = segmentUniqueBases[segmentUniqueBasesOffset + segment];
  let a = vertexAt(segment);
  let b = vertexAt(segment + 1u);
  let direction = b - a;
  var startPoint = a;
  var endPoint = b;
  if (local > 0u) { startPoint = pointOfRank(uniqueBase, local - 1u); }
  if (local < uniqueCount) { endPoint = pointOfRank(uniqueBase, local); }
  let startKey = keyOf(startPoint, direction);
  let endKey = keyOf(endPoint, direction);
  var flags = ${FLAG_VALID}u;
  for (var pair = first; pair < last; pair++) {
    let overlapEvent = pair * 2u;
    if ((eventTable[eventTableOffset + overlapEvent * ${EVENT_STRIDE}u] & 2u) != 0u) {
      let spanLow = min(eventKey(overlapEvent), eventKey(overlapEvent + 1u));
      let spanHigh = max(eventKey(overlapEvent), eventKey(overlapEvent + 1u));
      if (startKey >= spanLow && endKey <= spanHigh) { flags = flags | ${FLAG_BOUNDARY}u; }
    }
  }
  subTable[row] = bitcast<u32>(startPoint.x);
  subTable[row + 1u] = bitcast<u32>(startPoint.y);
  subTable[row + 2u] = bitcast<u32>(endPoint.x);
  subTable[row + 3u] = bitcast<u32>(endPoint.y);
  subTable[row + 4u] = segmentTable[segmentTableOffset + segment * 4u + 3u];
  subTable[row + 5u] = segment;
  subTable[row + 6u] = flags;
  let midpoint = (startPoint + endPoint) * 0.5;
  subMidpoints[subMidpointsOffset + index * 2u] = midpoint.x;
  subMidpoints[subMidpointsOffset + index * 2u + 1u] = midpoint.y;`
      })
    );

    // 7. Containment of every sub-piece midpoint.
    const subFeatures = T('sub-features', 'uint32', subPieceBound);
    const containmentOverflow = T('containment-overflow', 'uint32', 1);
    const containmentUncertain = T('containment-uncertain', 'uint32', 1);
    nodes.push(
      ...new GPUPointInPolygonJoin({
        id: `${id}-containment`,
        points: subMidpoints,
        polygonPositions: polygons.positions,
        featureOffsets: polygons.featureOffsets,
        polygonOffsets: polygons.polygonOffsets,
        ringOffsets: polygons.ringOffsets,
        candidateCapacity: props.candidateCapacity,
        includeBoundary: true,
        pointFeatureIds: subFeatures,
        overflow: containmentOverflow,
        uncertainCount: containmentUncertain
      }).getCommandNodes(graph)
    );

    // 8. Keep decision, then merge runs of continuing kept sub-pieces.
    const subAccessWGSL = `
const SUB_PIECE_BOUND: u32 = ${subPieceBound}u;
fn subFlags(piece: u32) -> u32 { return subTable[subTableOffset + piece * ${SUB_PIECE_STRIDE}u + 6u]; }
fn isKept(piece: u32) -> bool { return piece < SUB_PIECE_BOUND && (subFlags(piece) & ${FLAG_KEPT | FLAG_VALID}u) == ${FLAG_KEPT | FLAG_VALID}u; }
// Whether kept sub-piece piece continues the kept sub-piece before it.
fn continuesPrevious(piece: u32) -> bool {
  if (piece == 0u || !isKept(piece) || !isKept(piece - 1u)) { return false; }
  let row = subTableOffset + piece * ${SUB_PIECE_STRIDE}u;
  let previous = row - ${SUB_PIECE_STRIDE}u;
  return subTable[row + 4u] == subTable[previous + 4u] && subTable[row] == subTable[previous + 2u] &&
    subTable[row + 1u] == subTable[previous + 3u];
}
fn startsRun(piece: u32) -> bool { return isKept(piece) && !continuesPrevious(piece); }
// Whether the end point is a vertex: not when the run goes on along the same segment.
fn emitsEnd(piece: u32) -> bool {
  if (!isKept(piece)) { return false; }
  let sameSegmentNext = continuesPrevious(piece + 1u) &&
    subTable[subTableOffset + (piece + 1u) * ${SUB_PIECE_STRIDE}u + 5u] == subTable[subTableOffset + piece * ${SUB_PIECE_STRIDE}u + 5u];
  return !sameSegmentNext;
}`;
    const runFlags = T('run-flags', 'uint32', subPieceBound);
    const runRanks = T('run-ranks', 'uint32', subPieceBound);
    const vertexCounts = T('vertex-counts', 'uint32', subPieceBound);
    const vertexStarts = T('vertex-starts', 'uint32', subPieceBound);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-classify`,
        operation: OPERATION,
        variant: 'classify',
        bindings: [
          {name: 'subFeatures', view: subFeatures, type: 'u32', access: 'read'},
          {name: 'subTable', view: subTable, type: 'u32', access: 'read_write'}
        ],
        invocationCount: subPieceBound,
        declarations: `const NO_FEATURE: u32 = ${GPU_SPATIAL_JOIN_NO_FEATURE}u;`,
        body: `let row = subTableOffset + index * ${SUB_PIECE_STRIDE}u;
  let flags = subTable[row + 6u];
  if ((flags & ${FLAG_VALID}u) == 0u) { return; }
  let inside = (flags & ${FLAG_BOUNDARY}u) != 0u || subFeatures[subFeaturesOffset + index] != NO_FEATURE;
  if (inside == ${keepInside}) { subTable[row + 6u] = flags | ${FLAG_KEPT}u; }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-run-flags`,
        operation: OPERATION,
        variant: 'run-flags',
        bindings: [
          {name: 'subTable', view: subTable, type: 'u32', access: 'read'},
          {name: 'runFlags', view: runFlags, type: 'u32', access: 'read_write'},
          {name: 'vertexCounts', view: vertexCounts, type: 'u32', access: 'read_write'}
        ],
        invocationCount: subPieceBound,
        declarations: subAccessWGSL,
        body: `let start = select(0u, 1u, startsRun(index));
  runFlags[runFlagsOffset + index] = start;
  vertexCounts[vertexCountsOffset + index] = select(0u, start + select(0u, 1u, emitsEnd(index)), isKept(index));`
      }),
      ...new GPUScan({
        id: `${id}-run-scan`,
        input: runFlags,
        output: runRanks,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      ...new GPUScan({
        id: `${id}-vertex-scan`,
        input: vertexCounts,
        output: vertexStarts,
        mode: 'exclusive'
      }).getCommandNodes(graph)
    );

    // 9. Run table, capacity clamp and outputs.
    const runTable = T('run-table', 'uint32', subPieceBound * 2);
    const flagState = T('flag-state', 'uint32', 2);
    const state = T('state', 'uint32', 8);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-run-table`,
        operation: OPERATION,
        variant: 'run-table',
        bindings: [
          {name: 'subTable', view: subTable, type: 'u32', access: 'read'},
          {name: 'runFlags', view: runFlags, type: 'u32', access: 'read'},
          {name: 'runRanks', view: runRanks, type: 'u32', access: 'read'},
          {name: 'vertexStarts', view: vertexStarts, type: 'u32', access: 'read'},
          {name: 'runTable', view: runTable, type: 'u32', access: 'read_write'}
        ],
        invocationCount: subPieceBound,
        body: `if (runFlags[runFlagsOffset + index] == 0u) { return; }
  let rank = runRanks[runRanksOffset + index];
  runTable[runTableOffset + rank * 2u] = vertexStarts[vertexStartsOffset + index];
  runTable[runTableOffset + rank * 2u + 1u] = subTable[subTableOffset + index * ${SUB_PIECE_STRIDE}u + 4u];`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-flags`,
        operation: OPERATION,
        variant: 'flags',
        bindings: [
          {name: 'pairOverflow', view: pairOverflow, type: 'u32', access: 'read'},
          {name: 'containmentOverflow', view: containmentOverflow, type: 'u32', access: 'read'},
          {name: 'segmentUncertain', view: segmentUncertain, type: 'u32', access: 'read'},
          {name: 'containmentUncertain', view: containmentUncertain, type: 'u32', access: 'read'},
          {name: 'flagState', view: flagState, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 1,
        body: `flagState[flagStateOffset] = select(0u, 1u, pairOverflow[pairOverflowOffset] != 0u || containmentOverflow[containmentOverflowOffset] != 0u);
  flagState[flagStateOffset + 1u] = segmentUncertain[segmentUncertainOffset] + containmentUncertain[containmentUncertainOffset];`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-finalize`,
        operation: OPERATION,
        variant: 'finalize',
        bindings: [
          {name: 'runFlags', view: runFlags, type: 'u32', access: 'read'},
          {name: 'runRanks', view: runRanks, type: 'u32', access: 'read'},
          {name: 'vertexCounts', view: vertexCounts, type: 'u32', access: 'read'},
          {name: 'vertexStarts', view: vertexStarts, type: 'u32', access: 'read'},
          {name: 'runTable', view: runTable, type: 'u32', access: 'read'},
          {name: 'flagState', view: flagState, type: 'u32', access: 'read'},
          {name: 'state', view: state, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 1,
        declarations: `const LAST: u32 = ${subPieceBound - 1}u;
const PIECE_CAPACITY: u32 = ${pieceCapacity}u;
const VERTEX_CAPACITY: u32 = ${vertexCapacity}u;`,
        body: `let totalRuns = runRanks[runRanksOffset + LAST] + runFlags[runFlagsOffset + LAST];
  let totalVertices = vertexStarts[vertexStartsOffset + LAST] + vertexCounts[vertexCountsOffset + LAST];
  var low = 0u;
  var high = min(totalRuns, PIECE_CAPACITY);
  while (low < high) {
    let middle = (low + high + 1u) / 2u;
    // End of run middle - 1 is the start of run middle, or the vertex total after the last run.
    var end = totalVertices;
    if (middle < totalRuns) { end = runTable[runTableOffset + middle * 2u]; }
    if (end <= VERTEX_CAPACITY) { low = middle; } else { high = middle - 1u; }
  }
  var written = totalVertices;
  if (low < totalRuns) { written = runTable[runTableOffset + low * 2u]; }
  state[stateOffset] = low;
  state[stateOffset + 1u] = written;
  state[stateOffset + 2u] = totalRuns;
  state[stateOffset + 3u] = totalVertices;
  state[stateOffset + 4u] = select(0u, 1u, low < totalRuns || flagState[flagStateOffset] != 0u);
  state[stateOffset + 5u] = flagState[flagStateOffset + 1u];`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-write-pieces`,
        operation: OPERATION,
        variant: 'write-pieces',
        bindings: [
          {name: 'state', view: state, type: 'u32', access: 'read'},
          {name: 'runTable', view: runTable, type: 'u32', access: 'read'},
          {name: 'lineIds', view: pieces.sourceIds, type: 'u32', access: 'read_write'},
          {name: 'offsets', view: pieces.geometry.lineOffsets, type: 'u32', access: 'read_write'}
        ],
        invocationCount: pieceCapacity + 1,
        declarations: `const PIECE_CAPACITY: u32 = ${pieceCapacity}u;`,
        body: `if (index < state[stateOffset]) {
    lineIds[lineIdsOffset + index] = runTable[runTableOffset + index * 2u + 1u];
    offsets[offsetsOffset + index] = runTable[runTableOffset + index * 2u];
  } else {
    if (index < PIECE_CAPACITY) { lineIds[lineIdsOffset + index] = 0xffffffffu; }
    offsets[offsetsOffset + index] = state[stateOffset + 1u];
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-write-vertices`,
        operation: OPERATION,
        variant: 'write-vertices',
        bindings: [
          {name: 'state', view: state, type: 'u32', access: 'read'},
          {name: 'subTable', view: subTable, type: 'u32', access: 'read'},
          {name: 'runFlags', view: runFlags, type: 'u32', access: 'read'},
          {name: 'runRanks', view: runRanks, type: 'u32', access: 'read'},
          {name: 'vertexStarts', view: vertexStarts, type: 'u32', access: 'read'},
          {name: 'outPositions', view: pieces.geometry.positions, type: 'f32', access: 'read_write'}
        ],
        invocationCount: subPieceBound,
        declarations: subAccessWGSL,
        body: `if (!isKept(index)) { return; }
  let start = runFlags[runFlagsOffset + index];
  // Rank of the run this sub-piece belongs to: ranks count run starts before the piece.
  let rank = runRanks[runRanksOffset + index] + start - 1u;
  if (rank >= state[stateOffset]) { return; }
  var slot = vertexStarts[vertexStartsOffset + index];
  let row = subTableOffset + index * ${SUB_PIECE_STRIDE}u;
  if (start == 1u) {
    outPositions[outPositionsOffset + slot * 2u] = bitcast<f32>(subTable[row]);
    outPositions[outPositionsOffset + slot * 2u + 1u] = bitcast<f32>(subTable[row + 1u]);
    slot = slot + 1u;
  }
  if (emitsEnd(index)) {
    outPositions[outPositionsOffset + slot * 2u] = bitcast<f32>(subTable[row + 2u]);
    outPositions[outPositionsOffset + slot * 2u + 1u] = bitcast<f32>(subTable[row + 3u]);
  }`
      })
    );

    // 10. Scalars.
    const scalarBindings: WGSLKernelBinding[] = [
      {name: 'state', view: state, type: 'u32', access: 'read'}
    ];
    const scalars: [string, GraphDataView<'uint32'> | undefined, number][] = [
      ['count', pieces.status.count, 0],
      ['vertexCount', pieces.vertexCount, 1],
      ['requiredCount', pieces.status.requiredCount, 2],
      ['requiredVertexCount', pieces.requiredVertexCount, 3],
      ['overflow', pieces.status.overflow, 4],
      ['uncertainCount', props.uncertainCount, 5]
    ];
    for (const [name, view] of scalars) {
      if (view) {
        scalarBindings.push({name, view, type: 'u32', access: 'read_write'});
      }
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-scalars`,
        operation: OPERATION,
        variant: 'scalars',
        bindings: scalarBindings,
        invocationCount: 1,
        body: scalars
          .filter(([, view]) => view)
          .map(([name, , word]) => `${name}[${name}Offset] = state[stateOffset + ${word}u];`)
          .join('\n  ')
      })
    );
    return nodes;
  }
}
