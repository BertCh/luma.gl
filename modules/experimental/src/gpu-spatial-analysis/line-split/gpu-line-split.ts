// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  GPUSort,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {validateGraphViewsBelongToGraph} from '../../utils/gpu-contributor-utils';
import {GPUSegmentIntersection} from '../segment-intersection/index';
import type {GPUSpatialJoinLines} from '../spatial-join/index';

const OPERATION = 'GPULineSplit';

/** Value written to `pieces.lineIds` slots that hold no piece. */
export const GPU_LINE_SPLIT_NONE = 0xffffffff;

/** Number of `u32` words per piece-table row. */
const PIECE_STRIDE = 8;

/** Number of `u32` words per compacted split-table row: slot, phase, point x bits, point y bits. */
const SPLIT_STRIDE = 4;

/**
 * Caller-owned, capacity-bounded output pieces of {@link GPULineSplit}, in GeoArrow linestring
 * layout. Piece `q` is the vertex run `positions[offsets[q] .. offsets[q + 1])`.
 */
export type GPULineSplitPieces = {
  /** Source linestring row of each piece. Capacity is the length; unused slots hold `GPU_LINE_SPLIT_NONE`. */
  lineIds: GraphDataView<'uint32'>;
  /**
   * Piece-to-vertex offsets with `lineIds.length + 1` entries, first 0. Entries after `count`
   * repeat `vertexCount`, so the array is always monotone.
   */
  offsets: GraphDataView<'uint32'>;
  /** Vertices of all pieces, split points included. Capacity is the length. */
  positions: GraphDataView<'float32x2'>;
  /** One-row scalar receiving the number of complete pieces written. */
  count: GraphDataView<'uint32'>;
  /** Optional one-row scalar receiving the number of vertices written. */
  vertexCount?: GraphDataView<'uint32'>;
  /** One-row scalar receiving `1` when any capacity was exceeded (pieces, vertices, intersections). */
  overflow?: GraphDataView<'uint32'>;
  /** Optional one-row scalar receiving the unclamped number of pieces. */
  totalCount?: GraphDataView<'uint32'>;
  /** Optional one-row scalar receiving the unclamped number of vertices. */
  totalVertexCount?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPULineSplit}.
 *
 * Per-frame: the contents of every input buffer. Topology: view lengths, capacities,
 * `leafCapacity`, `spatialSort`.
 */
export type GPULineSplitProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'line-split'`. */
  id?: string;
  /** Linestrings to split at their mutual and self intersections. */
  lines: GPUSpatialJoinLines;
  /**
   * Capacity of the internal `GPUSegmentIntersection` pair list. Splitting needs at most four
   * transient rows per pair. When the intersection overflows, pieces are produced from the sorted
   * prefix of pairs and `pieces.overflow` is set.
   */
  intersectionCapacity: number;
  /** Output pieces. */
  pieces: GPULineSplitPieces;
  /** Passed to `GPUSegmentIntersection`. */
  spatialSort?: boolean;
  /** Passed to `GPUSegmentIntersection`. */
  leafCapacity?: number;
  /** Optional one-row count of segment pairs the intersection predicates could not certify. */
  uncertainCount?: GraphDataView<'uint32'>;
};

function getBitCount(value: number): number {
  return Math.max(1, Math.ceil(Math.log2(value + 1)));
}

/**
 * Splits linestrings at every point where they cross, touch or overlap each other or themselves
 * (turf `lineSplit` against the union of the lines, and the noding step of `lineIntersect`).
 *
 * Pipeline: `GPUSegmentIntersection` in self mode finds the intersecting segment pairs. Every
 * intersection point becomes one split event on each of its two segments. Events are grouped by
 * segment with a stable `GPUSort`, ordered along the segment, and deduplicated. A split exactly at
 * a vertex, or at the first or last vertex of a line, does not create a new vertex; a split in the
 * interior of a segment inserts the intersection point. Splits at the ends of a line are ignored.
 *
 * Output order is deterministic: pieces follow the input line order and, within a line, the
 * traversal order, so concatenating the pieces of line `i` reproduces it. A line with fewer than
 * two vertices yields no piece; an unsplit line yields one piece equal to the input. Overlapping
 * collinear spans split at both ends of the shared span. Pieces whose end would exceed the vertex
 * capacity are dropped as a suffix and `pieces.overflow` is set.
 *
 * Intersection points of proper crossings are rounded to f32 once per pair, so both lines receive
 * the identical point. Nothing is read back.
 */
export class GPULineSplit implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPULineSplitProps;
  /** Number of vertices of the input lines. */
  readonly vertexCount: number;
  /** Number of input linestrings. */
  readonly lineCount: number;
  /** Piece capacity. */
  readonly pieceCapacity: number;
  /** Output vertex capacity. */
  readonly vertexCapacity: number;

  constructor(props: GPULineSplitProps) {
    this.id = props.id ?? 'line-split';
    this.props = props;
    const {id} = this;
    const {lines, pieces} = props;
    if (lines.kind !== 'lines') {
      throw new Error(`${id} lines must be linestring geometry`);
    }
    validatePackedView(lines.positions, ['float32x2'], `${id} lines.positions`);
    validatePackedUint32View(lines.lineOffsets, `${id} lines.lineOffsets`);
    if (lines.lineOffsets.length < 2) {
      throw new Error(`${id} lines.lineOffsets requires at least one line and a terminal entry`);
    }
    if (lines.positions.length < 1 || lines.positions.length >= 0x80000000) {
      throw new Error(`${id} lines.positions must hold between 1 and 2^31 - 1 vertices`);
    }
    if (!Number.isSafeInteger(props.intersectionCapacity) || props.intersectionCapacity < 1) {
      throw new Error(`${id} intersectionCapacity must be a positive integer`);
    }
    this.vertexCount = lines.positions.length;
    this.lineCount = lines.lineOffsets.length - 1;
    validatePackedUint32View(pieces.lineIds, `${id} pieces.lineIds`);
    validatePackedUint32View(pieces.offsets, `${id} pieces.offsets`);
    validatePackedView(pieces.positions, ['float32x2'], `${id} pieces.positions`);
    this.pieceCapacity = pieces.lineIds.length;
    this.vertexCapacity = pieces.positions.length;
    if (this.pieceCapacity < 1 || pieces.offsets.length !== this.pieceCapacity + 1) {
      throw new Error(`${id} pieces.offsets length must be pieces.lineIds.length + 1`);
    }
    if (this.vertexCapacity < 1) {
      throw new Error(`${id} pieces.positions must be non-empty`);
    }
    for (const [name, view] of [
      ['count', pieces.count],
      ['vertexCount', pieces.vertexCount],
      ['overflow', pieces.overflow],
      ['totalCount', pieces.totalCount],
      ['totalVertexCount', pieces.totalVertexCount],
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

  /** Returns intersection, event, sort, piece and output nodes in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, vertexCount, lineCount, pieceCapacity, vertexCapacity} = this;
    const {lines, pieces} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      lines.positions,
      lines.lineOffsets,
      pieces.lineIds,
      pieces.offsets,
      pieces.positions,
      pieces.count,
      pieces.vertexCount,
      pieces.overflow,
      pieces.totalCount,
      pieces.totalVertexCount,
      props.uncertainCount
    ]);
    const nodes: GPUCommandNode<Parameters>[] = [];
    const pairCapacity = props.intersectionCapacity;
    const eventCount = pairCapacity * 4;
    const pieceBound = lineCount + eventCount;
    const T = <Format extends 'uint32' | 'float32x2'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, `${id}-${name}`, format, length);

    // 1. Intersections between segments of the lines (self mode).
    const pairLeft = T('pair-left', 'uint32', pairCapacity);
    const pairRight = T('pair-right', 'uint32', pairCapacity);
    const pairCount = T('pair-count', 'uint32', 1);
    const pairOverflow = T('pair-overflow', 'uint32', 1);
    const kinds = T('kinds', 'uint32', pairCapacity);
    const points = T('points', 'float32x2', pairCapacity);
    const endPoints = T('end-points', 'float32x2', pairCapacity);
    nodes.push(
      ...new GPUSegmentIntersection({
        id: `${id}-intersection`,
        left: lines,
        spatialSort: props.spatialSort,
        leafCapacity: props.leafCapacity,
        pairs: {
          leftIds: pairLeft,
          rightIds: pairRight,
          count: pairCount,
          overflow: pairOverflow
        },
        uncertainCount: props.uncertainCount,
        kinds,
        points,
        endPoints
      }).getCommandNodes(graph)
    );

    // 2. Split events: up to four per pair (both ends of an overlap, on both segments).
    const eventSegments = T('event-segments', 'uint32', eventCount);
    const eventPoints = T('event-points', 'float32x2', eventCount);
    const eventIndices = T('event-indices', 'uint32', eventCount);
    const eventSlots = T('event-slots', 'uint32', eventCount);
    const eventTable = T('event-table', 'uint32', eventCount * 2);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-event-segments`,
        operation: OPERATION,
        variant: 'event-segments',
        bindings: [
          {name: 'pairLeft', view: pairLeft, type: 'u32', access: 'read'},
          {name: 'pairRight', view: pairRight, type: 'u32', access: 'read'},
          {name: 'kinds', view: kinds, type: 'u32', access: 'read'},
          {name: 'pairCount', view: pairCount, type: 'u32', access: 'read'},
          {name: 'eventSegments', view: eventSegments, type: 'u32', access: 'read_write'}
        ],
        invocationCount: eventCount,
        body: `let pair = index / 4u;
  let which = index % 4u;
  let kind = kinds[kindsOffset + pair];
  // Kinds 1 to 4 intersect at a point or span; 5 (uncertain) and 0 produce no split.
  var valid = pair < pairCount[pairCountOffset] && kind >= 1u && kind <= 4u;
  if (which >= 2u && kind != 4u) { valid = false; }
  var segment = pairLeft[pairLeftOffset + pair];
  if ((which & 1u) == 1u) { segment = pairRight[pairRightOffset + pair]; }
  eventSegments[eventSegmentsOffset + index] = select(0xffffffffu, segment, valid);`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-event-points`,
        operation: OPERATION,
        variant: 'event-points',
        bindings: [
          {name: 'eventSegments', view: eventSegments, type: 'u32', access: 'read'},
          {name: 'points', view: points, type: 'f32', access: 'read'},
          {name: 'endPoints', view: endPoints, type: 'f32', access: 'read'},
          {name: 'eventPoints', view: eventPoints, type: 'f32', access: 'read_write'},
          {name: 'eventIndices', view: eventIndices, type: 'u32', access: 'read_write'}
        ],
        invocationCount: eventCount,
        body: `let pair = index / 4u;
  var point = vec2f(points[pointsOffset + pair * 2u], points[pointsOffset + pair * 2u + 1u]);
  if (index % 4u >= 2u) {
    point = vec2f(endPoints[endPointsOffset + pair * 2u], endPoints[endPointsOffset + pair * 2u + 1u]);
  }
  eventPoints[eventPointsOffset + index * 2u] = point.x;
  eventPoints[eventPointsOffset + index * 2u + 1u] = point.y;
  eventIndices[eventIndicesOffset + index] = index;`
      })
    );

    const lineLookupWGSL = `
const VERTEX_COUNT: u32 = ${vertexCount}u;
const LINE_COUNT: u32 = ${lineCount}u;
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
fn vertexAt(vertex: u32) -> vec2f {
  return vec2f(positions[positionsOffset + vertex * 2u], positions[positionsOffset + vertex * 2u + 1u]);
}`;
    // 3. Normalize every event to (slot, phase, key): phase 0 is the vertex slot itself, phase 1 a
    // point inside the segment that starts at slot, ordered by its coordinate along the segment.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-event-slots`,
        operation: OPERATION,
        variant: 'event-slots',
        bindings: [
          {name: 'eventSegments', view: eventSegments, type: 'u32', access: 'read'},
          {name: 'eventPoints', view: eventPoints, type: 'f32', access: 'read'},
          {name: 'positions', view: lines.positions, type: 'f32', access: 'read'},
          {name: 'lineOffsets', view: lines.lineOffsets, type: 'u32', access: 'read'},
          {name: 'eventSlots', view: eventSlots, type: 'u32', access: 'read_write'},
          {name: 'eventTable', view: eventTable, type: 'u32', access: 'read_write'}
        ],
        invocationCount: eventCount,
        declarations: lineLookupWGSL,
        body: `let segment = eventSegments[eventSegmentsOffset + index];
  var slot = VERTEX_COUNT;
  var phase = 0u;
  var key = 0.0;
  if (segment != 0xffffffffu) {
    let line = lineOfVertex(segment);
    let lineStart = lineOffsets[lineOffsetsOffset + line];
    let lineEnd = lineOffsets[lineOffsetsOffset + line + 1u];
    let a = vertexAt(segment);
    let b = vertexAt(segment + 1u);
    let p = vec2f(eventPoints[eventPointsOffset + index * 2u], eventPoints[eventPointsOffset + index * 2u + 1u]);
    if (p.x == a.x && p.y == a.y) {
      if (segment > lineStart) { slot = segment; }
    } else if (p.x == b.x && p.y == b.y) {
      if (segment + 1u < lineEnd - 1u) { slot = segment + 1u; }
    } else {
      slot = segment;
      phase = 1u;
      let d = b - a;
      if (abs(d.x) >= abs(d.y)) { key = p.x * select(-1.0, 1.0, d.x > 0.0); }
      else { key = p.y * select(-1.0, 1.0, d.y > 0.0); }
    }
  }
  eventSlots[eventSlotsOffset + index] = slot;
  eventTable[eventTableOffset + index * 2u] = phase;
  eventTable[eventTableOffset + index * 2u + 1u] = bitcast<u32>(key);`
      })
    );

    // 4. Group events by slot (stable), then order and deduplicate inside every group.
    const sortedSlots = T('sorted-slots', 'uint32', eventCount);
    const sortedEvents = T('sorted-events', 'uint32', eventCount);
    nodes.push(
      ...new GPUSort({
        id: `${id}-sort`,
        keys: eventSlots,
        values: eventIndices,
        outputKeys: sortedSlots,
        outputValues: sortedEvents,
        keyBits: getBitCount(vertexCount)
      }).getCommandNodes(graph)
    );
    const eventOrderWGSL = `
const VERTEX_COUNT: u32 = ${vertexCount}u;
const EVENT_COUNT: u32 = ${eventCount}u;
fn eventPhase(event: u32) -> u32 { return eventTable[eventTableOffset + event * 2u]; }
fn eventKey(event: u32) -> f32 { return bitcast<f32>(eventTable[eventTableOffset + event * 2u + 1u]); }
fn eventLess(first: u32, second: u32) -> bool {
  if (eventPhase(first) != eventPhase(second)) { return eventPhase(first) < eventPhase(second); }
  return eventKey(first) < eventKey(second);
}
fn eventSame(first: u32, second: u32) -> bool {
  return eventPhase(first) == eventPhase(second) && eventKey(first) == eventKey(second);
}`;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-order-groups`,
        operation: OPERATION,
        variant: 'order-groups',
        bindings: [
          {name: 'sortedSlots', view: sortedSlots, type: 'u32', access: 'read'},
          {name: 'sortedEvents', view: sortedEvents, type: 'u32', access: 'read_write'},
          {name: 'eventTable', view: eventTable, type: 'u32', access: 'read'}
        ],
        invocationCount: eventCount,
        declarations: `${eventOrderWGSL}
const INSERTION_SORT_LIMIT: u32 = 16u;
// Total order: the event order, then the event index (what the stable slot sort already gives).
fn eventBefore(first: u32, second: u32) -> bool {
  if (eventLess(first, second)) { return true; }
  if (eventLess(second, first)) { return false; }
  return first < second;
}
fn eventAt(base: u32, position: u32) -> u32 { return sortedEvents[sortedEventsOffset + base + position]; }
fn siftDown(base: u32, rootStart: u32, count: u32) {
  var root = rootStart;
  let value = eventAt(base, root);
  loop {
    var child = 2u * root + 1u;
    if (child >= count) { break; }
    if (child + 1u < count && eventBefore(eventAt(base, child), eventAt(base, child + 1u))) { child = child + 1u; }
    let childValue = eventAt(base, child);
    if (!eventBefore(value, childValue)) { break; }
    sortedEvents[sortedEventsOffset + base + root] = childValue;
    root = child;
  }
  sortedEvents[sortedEventsOffset + base + root] = value;
}`,
        // One thread per slot orders that slot's events. Few events (the usual case) use a stable
        // insertion sort; a segment crossed by many lines uses an in-place heapsort so a slot with
        // k events costs O(k log k) instead of O(k^2).
        body: `let slot = sortedSlots[sortedSlotsOffset + index];
  if (slot >= VERTEX_COUNT) { return; }
  if (index > 0u && sortedSlots[sortedSlotsOffset + index - 1u] == slot) { return; }
  var end = index + 1u;
  while (end < EVENT_COUNT && sortedSlots[sortedSlotsOffset + end] == slot) { end = end + 1u; }
  let count = end - index;
  if (count <= INSERTION_SORT_LIMIT) {
    for (var item = index + 1u; item < end; item++) {
      let value = sortedEvents[sortedEventsOffset + item];
      var hole = item;
      while (hole > index) {
        let previous = sortedEvents[sortedEventsOffset + hole - 1u];
        if (!eventLess(value, previous)) { break; }
        sortedEvents[sortedEventsOffset + hole] = previous;
        hole = hole - 1u;
      }
      sortedEvents[sortedEventsOffset + hole] = value;
    }
    return;
  }
  var start = count / 2u;
  while (start > 0u) {
    start = start - 1u;
    siftDown(index, start, count);
  }
  var last = count;
  while (last > 1u) {
    last = last - 1u;
    let top = eventAt(index, 0u);
    sortedEvents[sortedEventsOffset + index] = eventAt(index, last);
    sortedEvents[sortedEventsOffset + index + last] = top;
    siftDown(index, 0u, last);
  }`
      })
    );
    const uniqueFlags = T('unique-flags', 'uint32', eventCount);
    const uniqueRanks = T('unique-ranks', 'uint32', eventCount);
    const splitTable = T('split-table', 'uint32', eventCount * SPLIT_STRIDE);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-unique-flags`,
        operation: OPERATION,
        variant: 'unique-flags',
        bindings: [
          {name: 'sortedSlots', view: sortedSlots, type: 'u32', access: 'read'},
          {name: 'sortedEvents', view: sortedEvents, type: 'u32', access: 'read'},
          {name: 'eventTable', view: eventTable, type: 'u32', access: 'read'},
          {name: 'uniqueFlags', view: uniqueFlags, type: 'u32', access: 'read_write'}
        ],
        invocationCount: eventCount,
        declarations: eventOrderWGSL,
        body: `let slot = sortedSlots[sortedSlotsOffset + index];
  var flag = 0u;
  if (slot < VERTEX_COUNT) {
    if (index == 0u || sortedSlots[sortedSlotsOffset + index - 1u] != slot) { flag = 1u; }
    else if (!eventSame(sortedEvents[sortedEventsOffset + index], sortedEvents[sortedEventsOffset + index - 1u])) { flag = 1u; }
  }
  uniqueFlags[uniqueFlagsOffset + index] = flag;`
      }),
      ...new GPUScan({
        id: `${id}-unique-scan`,
        input: uniqueFlags,
        output: uniqueRanks,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-compact-splits`,
        operation: OPERATION,
        variant: 'compact-splits',
        bindings: [
          {name: 'sortedSlots', view: sortedSlots, type: 'u32', access: 'read'},
          {name: 'sortedEvents', view: sortedEvents, type: 'u32', access: 'read'},
          {name: 'eventTable', view: eventTable, type: 'u32', access: 'read'},
          {name: 'eventPoints', view: eventPoints, type: 'u32', access: 'read'},
          {name: 'uniqueFlags', view: uniqueFlags, type: 'u32', access: 'read'},
          {name: 'uniqueRanks', view: uniqueRanks, type: 'u32', access: 'read'},
          {name: 'splitTable', view: splitTable, type: 'u32', access: 'read_write'}
        ],
        invocationCount: eventCount,
        declarations: eventOrderWGSL,
        body: `if (uniqueFlags[uniqueFlagsOffset + index] == 0u) { return; }
  let rank = uniqueRanks[uniqueRanksOffset + index];
  let event = sortedEvents[sortedEventsOffset + index];
  let row = splitTableOffset + rank * ${SPLIT_STRIDE}u;
  splitTable[row] = sortedSlots[sortedSlotsOffset + index];
  splitTable[row + 1u] = eventPhase(event);
  splitTable[row + 2u] = eventPoints[eventPointsOffset + event * 2u];
  splitTable[row + 3u] = eventPoints[eventPointsOffset + event * 2u + 1u];`
      })
    );

    // 5. Per line: first split rank and piece count, then pieces.
    const splitBases = T('split-bases', 'uint32', lineCount);
    const pieceCounts = T('piece-counts', 'uint32', lineCount);
    const pieceStarts = T('piece-starts', 'uint32', lineCount);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-line-splits`,
        operation: OPERATION,
        variant: 'line-splits',
        bindings: [
          {name: 'lineOffsets', view: lines.lineOffsets, type: 'u32', access: 'read'},
          {name: 'sortedSlots', view: sortedSlots, type: 'u32', access: 'read'},
          {name: 'uniqueFlags', view: uniqueFlags, type: 'u32', access: 'read'},
          {name: 'uniqueRanks', view: uniqueRanks, type: 'u32', access: 'read'},
          {name: 'splitBases', view: splitBases, type: 'u32', access: 'read_write'},
          {name: 'pieceCounts', view: pieceCounts, type: 'u32', access: 'read_write'}
        ],
        invocationCount: lineCount,
        declarations: `const EVENT_COUNT: u32 = ${eventCount}u;
fn lowerBound(value: u32) -> u32 {
  var low = 0u;
  var high = EVENT_COUNT;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (sortedSlots[sortedSlotsOffset + middle] < value) { low = middle + 1u; } else { high = middle; }
  }
  return low;
}
fn uniqueBefore(position: u32) -> u32 {
  if (position < EVENT_COUNT) { return uniqueRanks[uniqueRanksOffset + position]; }
  return uniqueRanks[uniqueRanksOffset + EVENT_COUNT - 1u] + uniqueFlags[uniqueFlagsOffset + EVENT_COUNT - 1u];
}`,
        body: `let lineStart = lineOffsets[lineOffsetsOffset + index];
  let lineEnd = lineOffsets[lineOffsetsOffset + index + 1u];
  let first = uniqueBefore(lowerBound(lineStart));
  let last = uniqueBefore(lowerBound(lineEnd));
  splitBases[splitBasesOffset + index] = first;
  pieceCounts[pieceCountsOffset + index] = select(0u, last - first + 1u, lineEnd >= lineStart + 2u);`
      }),
      ...new GPUScan({
        id: `${id}-piece-scan`,
        input: pieceCounts,
        output: pieceStarts,
        mode: 'exclusive'
      }).getCommandNodes(graph)
    );

    const pieceTable = T('piece-table', 'uint32', pieceBound * PIECE_STRIDE);
    const pieceVertexCounts = T('piece-vertex-counts', 'uint32', pieceBound);
    const pieceVertexStarts = T('piece-vertex-starts', 'uint32', pieceBound);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-pieces`,
        operation: OPERATION,
        variant: 'pieces',
        bindings: [
          {name: 'lineOffsets', view: lines.lineOffsets, type: 'u32', access: 'read'},
          {name: 'pieceStarts', view: pieceStarts, type: 'u32', access: 'read'},
          {name: 'pieceCounts', view: pieceCounts, type: 'u32', access: 'read'},
          {name: 'splitBases', view: splitBases, type: 'u32', access: 'read'},
          {name: 'splitTable', view: splitTable, type: 'u32', access: 'read'},
          {name: 'pieceTable', view: pieceTable, type: 'u32', access: 'read_write'},
          {name: 'pieceVertexCounts', view: pieceVertexCounts, type: 'u32', access: 'read_write'}
        ],
        invocationCount: pieceBound,
        declarations: `const LINE_COUNT: u32 = ${lineCount}u;`,
        body: `let total = pieceStarts[pieceStartsOffset + LINE_COUNT - 1u] + pieceCounts[pieceCountsOffset + LINE_COUNT - 1u];
  if (index >= total) { pieceVertexCounts[pieceVertexCountsOffset + index] = 0u; return; }
  // Last line whose first piece is at most index.
  var low = 0u;
  var high = LINE_COUNT - 1u;
  while (low < high) {
    let middle = (low + high + 1u) / 2u;
    if (pieceStarts[pieceStartsOffset + middle] <= index) { low = middle; } else { high = middle - 1u; }
  }
  let line = low;
  let k = index - pieceStarts[pieceStartsOffset + line];
  let lastPiece = pieceCounts[pieceCountsOffset + line] - 1u;
  let base = splitBases[splitBasesOffset + line];
  var firstVertex = lineOffsets[lineOffsetsOffset + line];
  var lastVertex = lineOffsets[lineOffsetsOffset + line + 1u] - 1u;
  var prefix = 0u;
  var suffix = 0u;
  var prefixPoint = vec2u(0u);
  var suffixPoint = vec2u(0u);
  if (k > 0u) {
    let row = splitTableOffset + (base + k - 1u) * ${SPLIT_STRIDE}u;
    let slot = splitTable[row];
    let phase = splitTable[row + 1u];
    firstVertex = slot + phase;
    prefix = phase;
    prefixPoint = vec2u(splitTable[row + 2u], splitTable[row + 3u]);
  }
  if (k < lastPiece) {
    let row = splitTableOffset + (base + k) * ${SPLIT_STRIDE}u;
    lastVertex = splitTable[row];
    suffix = splitTable[row + 1u];
    suffixPoint = vec2u(splitTable[row + 2u], splitTable[row + 3u]);
  }
  let row = pieceTableOffset + index * ${PIECE_STRIDE}u;
  pieceTable[row] = line;
  pieceTable[row + 1u] = firstVertex;
  pieceTable[row + 2u] = lastVertex;
  pieceTable[row + 3u] = prefix | (suffix << 1u);
  pieceTable[row + 4u] = prefixPoint.x;
  pieceTable[row + 5u] = prefixPoint.y;
  pieceTable[row + 6u] = suffixPoint.x;
  pieceTable[row + 7u] = suffixPoint.y;
  pieceVertexCounts[pieceVertexCountsOffset + index] = lastVertex - firstVertex + 1u + prefix + suffix;`
      }),
      ...new GPUScan({
        id: `${id}-vertex-scan`,
        input: pieceVertexCounts,
        output: pieceVertexStarts,
        mode: 'exclusive'
      }).getCommandNodes(graph)
    );

    // 6. Capacity clamp: the longest prefix of pieces that fits both capacities.
    const state = T('state', 'uint32', 8);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-finalize`,
        operation: OPERATION,
        variant: 'finalize',
        bindings: [
          {name: 'pieceStarts', view: pieceStarts, type: 'u32', access: 'read'},
          {name: 'pieceCounts', view: pieceCounts, type: 'u32', access: 'read'},
          {name: 'vertexStarts', view: pieceVertexStarts, type: 'u32', access: 'read'},
          {name: 'vertexCounts', view: pieceVertexCounts, type: 'u32', access: 'read'},
          {name: 'pairOverflow', view: pairOverflow, type: 'u32', access: 'read'},
          {name: 'state', view: state, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 1,
        declarations: `const LINE_COUNT: u32 = ${lineCount}u;
const PIECE_CAPACITY: u32 = ${pieceCapacity}u;
const VERTEX_CAPACITY: u32 = ${vertexCapacity}u;
fn pieceEnd(piece: u32) -> u32 { return vertexStarts[vertexStartsOffset + piece] + vertexCounts[vertexCountsOffset + piece]; }`,
        body: `let totalPieces = pieceStarts[pieceStartsOffset + LINE_COUNT - 1u] + pieceCounts[pieceCountsOffset + LINE_COUNT - 1u];
  var totalVertices = 0u;
  if (totalPieces > 0u) { totalVertices = pieceEnd(totalPieces - 1u); }
  var low = 0u;
  var high = min(totalPieces, PIECE_CAPACITY);
  while (low < high) {
    let middle = (low + high + 1u) / 2u;
    if (pieceEnd(middle - 1u) <= VERTEX_CAPACITY) { low = middle; } else { high = middle - 1u; }
  }
  var written = 0u;
  if (low > 0u) { written = pieceEnd(low - 1u); }
  state[stateOffset] = low;
  state[stateOffset + 1u] = written;
  state[stateOffset + 2u] = totalPieces;
  state[stateOffset + 3u] = totalVertices;
  state[stateOffset + 4u] = select(0u, 1u, low < totalPieces || pairOverflow[pairOverflowOffset] != 0u);`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-write-pieces`,
        operation: OPERATION,
        variant: 'write-pieces',
        bindings: [
          {name: 'state', view: state, type: 'u32', access: 'read'},
          {name: 'pieceTable', view: pieceTable, type: 'u32', access: 'read'},
          {name: 'vertexStarts', view: pieceVertexStarts, type: 'u32', access: 'read'},
          {name: 'lineIds', view: pieces.lineIds, type: 'u32', access: 'read_write'},
          {name: 'offsets', view: pieces.offsets, type: 'u32', access: 'read_write'}
        ],
        invocationCount: pieceCapacity + 1,
        declarations: `const PIECE_CAPACITY: u32 = ${pieceCapacity}u;`,
        body: `let kept = state[stateOffset];
  if (index < kept) {
    lineIds[lineIdsOffset + index] = pieceTable[pieceTableOffset + index * ${PIECE_STRIDE}u];
    offsets[offsetsOffset + index] = vertexStarts[vertexStartsOffset + index];
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
          {name: 'pieceTable', view: pieceTable, type: 'u32', access: 'read'},
          {name: 'vertexStarts', view: pieceVertexStarts, type: 'u32', access: 'read'},
          {name: 'positions', view: lines.positions, type: 'f32', access: 'read'},
          {name: 'outPositions', view: pieces.positions, type: 'f32', access: 'read_write'}
        ],
        invocationCount: vertexCapacity,
        body: `if (index >= state[stateOffset + 1u]) { return; }
  // Last kept piece whose first vertex is at most index.
  var low = 0u;
  var high = state[stateOffset] - 1u;
  while (low < high) {
    let middle = (low + high + 1u) / 2u;
    if (vertexStarts[vertexStartsOffset + middle] <= index) { low = middle; } else { high = middle - 1u; }
  }
  let row = pieceTableOffset + low * ${PIECE_STRIDE}u;
  let firstVertex = pieceTable[row + 1u];
  let runLength = pieceTable[row + 2u] - firstVertex + 1u;
  let flags = pieceTable[row + 3u];
  var local = index - vertexStarts[vertexStartsOffset + low];
  var point = vec2f(0.0);
  var resolved = false;
  if ((flags & 1u) != 0u) {
    if (local == 0u) {
      point = vec2f(bitcast<f32>(pieceTable[row + 4u]), bitcast<f32>(pieceTable[row + 5u]));
      resolved = true;
    } else { local = local - 1u; }
  }
  if (!resolved) {
    if (local < runLength) {
      let source = positionsOffset + (firstVertex + local) * 2u;
      point = vec2f(positions[source], positions[source + 1u]);
    } else {
      point = vec2f(bitcast<f32>(pieceTable[row + 6u]), bitcast<f32>(pieceTable[row + 7u]));
    }
  }
  outPositions[outPositionsOffset + index * 2u] = point.x;
  outPositions[outPositionsOffset + index * 2u + 1u] = point.y;`
      })
    );

    // 7. Scalars.
    const scalarBindings: WGSLKernelBinding[] = [
      {name: 'state', view: state, type: 'u32', access: 'read'}
    ];
    const scalars: [string, GraphDataView<'uint32'> | undefined, number][] = [
      ['count', pieces.count, 0],
      ['vertexCount', pieces.vertexCount, 1],
      ['totalCount', pieces.totalCount, 2],
      ['totalVertexCount', pieces.totalVertexCount, 3],
      ['overflow', pieces.overflow, 4]
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
