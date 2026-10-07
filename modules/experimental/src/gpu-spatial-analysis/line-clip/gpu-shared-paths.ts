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
import {GPUSegmentIntersection} from '../segment-intersection/index';
import type {GPUSpatialJoinLines} from '../spatial-join/index';

const OPERATION = 'GPUSharedPaths';

/** Number of `u32` words per span-table row (one row per intersecting segment pair). */
const SPAN_STRIDE = 12;

/** Number of `u32` words per run-table row. */
const RUN_STRIDE = 4;

/** Value written to `runs.leftLineIds` and `runs.rightLineIds` slots that hold no run. */
export const GPU_SHARED_PATHS_NONE = 0xffffffff;

/**
 * Caller-owned, capacity-bounded output runs of {@link GPUSharedPaths}, in GeoArrow linestring
 * layout. Run `q` is the vertex run `positions[offsets[q] .. offsets[q + 1])`, ordered along the
 * left line.
 */
export type GPUSharedPathsRuns = {
  /** Left linestring row of each run. Capacity is the length; unused slots hold `GPU_SHARED_PATHS_NONE`. */
  leftLineIds: GraphDataView<'uint32'>;
  /** Right linestring row of each run; same length as `leftLineIds`. */
  rightLineIds: GraphDataView<'uint32'>;
  /**
   * `1` when the right line runs the same way as the left line along the run (Shapely's forward
   * collection), `0` when it runs the opposite way (the backward collection).
   */
  forward: GraphDataView<'uint32'>;
  /**
   * Run-to-vertex offsets with `leftLineIds.length + 1` entries, first 0. Entries after `count`
   * repeat the written vertex count, so the array is always monotone.
   */
  offsets: GraphDataView<'uint32'>;
  /** Vertices of all runs. Capacity is the length. */
  positions: GraphDataView<'float32x2'>;
  /** One-row scalar receiving the number of complete runs written. */
  count: GraphDataView<'uint32'>;
  /** Optional one-row scalar receiving the number of vertices written. */
  vertexCount?: GraphDataView<'uint32'>;
  /** Optional one-row scalar receiving `1` when any capacity was exceeded. */
  overflow?: GraphDataView<'uint32'>;
  /** Optional one-row scalar receiving the unclamped number of runs. */
  totalCount?: GraphDataView<'uint32'>;
  /** Optional one-row scalar receiving the unclamped number of vertices. */
  totalVertexCount?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUSharedPaths}.
 *
 * Per-frame: the contents of every input buffer. Topology: view lengths, capacities,
 * `leafCapacity`, `spatialSort`.
 */
export type GPUSharedPathsProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'shared-paths'`. */
  id?: string;
  /** Left linestrings. */
  left: GPUSpatialJoinLines;
  /** Right linestrings, indexed by a BVH. */
  right: GPUSpatialJoinLines;
  /**
   * Capacity of the internal `GPUSegmentIntersection` pair list (one row per intersecting segment
   * pair, not only the shared ones). When it overflows, runs are produced from the sorted prefix
   * and `runs.overflow` is set.
   */
  intersectionCapacity: number;
  /** Output runs. */
  runs: GPUSharedPathsRuns;
  /** Passed to `GPUSegmentIntersection`. */
  spatialSort?: boolean;
  /** Passed to `GPUSegmentIntersection`. */
  leafCapacity?: number;
  /** Optional one-row count of segment pairs the intersection predicates could not certify. */
  uncertainCount?: GraphDataView<'uint32'>;
};

/**
 * Finds the paths that two sets of linestrings share (Shapely `shared_paths`, GeoPandas has no
 * direct name), with the direction of the right line relative to the left line.
 *
 * Pipeline: `GPUSegmentIntersection` (two-sided) lists every intersecting segment pair; the
 * collinear overlaps are the shared spans. Every span is oriented along its left segment and
 * tagged forward or backward by the sign of the product of the two segment directions along the
 * dominant axis (exact). Spans of one (left line, right line, direction) combination that meet
 * end to start, on the same left segment or on consecutive ones, are chained into one run, so a
 * road shared by two networks over many segments is one output row. A run from line pair `(i, j)`
 * is a single polyline along line `i`; points where the run crosses a left vertex are kept and
 * points inside a left segment are not.
 *
 * Output order is deterministic: runs are ordered by the first left segment they cover and then
 * by the pair of right segments. A closed left line is cut at its first vertex, so a fully shared
 * ring is one run starting there. Two coincident right segments report a run each. Overlap is
 * decided with exact predicates; a point shared by two lines is not a path and yields no run.
 * Runs beyond the capacities are dropped as a suffix and `runs.overflow` is set. Nothing is read
 * back.
 */
export class GPUSharedPaths implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSharedPathsProps;
  /** Number of left segment slots (vertices). */
  readonly leftVertexCount: number;
  /** Run capacity. */
  readonly runCapacity: number;
  /** Output vertex capacity. */
  readonly vertexCapacity: number;

  constructor(props: GPUSharedPathsProps) {
    this.id = props.id ?? 'shared-paths';
    this.props = props;
    const {id} = this;
    const {runs} = props;
    for (const [name, lines] of [
      ['left', props.left],
      ['right', props.right]
    ] as const) {
      if (lines.kind !== 'lines') {
        throw new Error(`${id} ${name} must be linestring geometry`);
      }
      validatePackedView(lines.positions, ['float32x2'], `${id} ${name}.positions`);
      validatePackedUint32View(lines.lineOffsets, `${id} ${name}.lineOffsets`);
      if (lines.lineOffsets.length < 2) {
        throw new Error(
          `${id} ${name}.lineOffsets requires at least one line and a terminal entry`
        );
      }
      if (lines.positions.length < 1 || lines.positions.length >= 0x80000000) {
        throw new Error(`${id} ${name}.positions must hold between 1 and 2^31 - 1 vertices`);
      }
    }
    if (!Number.isSafeInteger(props.intersectionCapacity) || props.intersectionCapacity < 1) {
      throw new Error(`${id} intersectionCapacity must be a positive integer`);
    }
    this.leftVertexCount = props.left.positions.length;
    for (const [name, view] of [
      ['runs.leftLineIds', runs.leftLineIds],
      ['runs.rightLineIds', runs.rightLineIds],
      ['runs.forward', runs.forward],
      ['runs.offsets', runs.offsets]
    ] as const) {
      validatePackedUint32View(view, `${id} ${name}`);
    }
    validatePackedView(runs.positions, ['float32x2'], `${id} runs.positions`);
    this.runCapacity = runs.leftLineIds.length;
    this.vertexCapacity = runs.positions.length;
    if (
      this.runCapacity < 1 ||
      runs.rightLineIds.length !== this.runCapacity ||
      runs.forward.length !== this.runCapacity
    ) {
      throw new Error(`${id} runs.leftLineIds, rightLineIds and forward need equal nonzero length`);
    }
    if (runs.offsets.length !== this.runCapacity + 1) {
      throw new Error(`${id} runs.offsets length must be runs.leftLineIds.length + 1`);
    }
    if (this.vertexCapacity < 1) {
      throw new Error(`${id} runs.positions must be non-empty`);
    }
    for (const [name, view] of [
      ['count', runs.count],
      ['vertexCount', runs.vertexCount],
      ['overflow', runs.overflow],
      ['totalCount', runs.totalCount],
      ['totalVertexCount', runs.totalVertexCount],
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

  /** Returns intersection, span, link, run and output nodes in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, runCapacity, vertexCapacity} = this;
    const {left, right, runs} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      left.positions,
      left.lineOffsets,
      right.positions,
      right.lineOffsets,
      runs.leftLineIds,
      runs.rightLineIds,
      runs.forward,
      runs.offsets,
      runs.positions,
      runs.count,
      runs.vertexCount,
      runs.overflow,
      runs.totalCount,
      runs.totalVertexCount,
      props.uncertainCount
    ]);
    const nodes: GPUCommandNode<Parameters>[] = [];
    const pairCapacity = props.intersectionCapacity;
    const T = <Format extends 'uint32' | 'float32x2'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, `${id}-${name}`, format, length);

    // 1. Segment pairs between the two sets of lines.
    const pairLeft = T('pair-left', 'uint32', pairCapacity);
    const pairRight = T('pair-right', 'uint32', pairCapacity);
    const pairCount = T('pair-count', 'uint32', 1);
    const pairOverflow = T('pair-overflow', 'uint32', 1);
    const kinds = T('kinds', 'uint32', pairCapacity);
    const points = T('points', 'float32x2', pairCapacity);
    const endPoints = T('end-points', 'float32x2', pairCapacity);
    const leftRings = T('left-rings', 'uint32', pairCapacity);
    const rightRings = T('right-rings', 'uint32', pairCapacity);
    nodes.push(
      ...new GPUSegmentIntersection({
        id: `${id}-intersection`,
        left,
        right,
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
        endPoints,
        leftRings,
        rightRings
      }).getCommandNodes(graph)
    );

    const geometryWGSL = (prefix: string) => `
fn ${prefix}Vertex(vertex: u32) -> vec2f {
  return vec2f(${prefix}Positions[${prefix}PositionsOffset + vertex * 2u], ${prefix}Positions[${prefix}PositionsOffset + vertex * 2u + 1u]);
}`;
    const keyWGSL = `
// Coordinate of p along direction d on the dominant axis; ascending along d.
fn keyOf(p: vec2f, d: vec2f) -> f32 {
  if (abs(d.x) >= abs(d.y)) { return p.x * select(-1.0, 1.0, d.x > 0.0); }
  return p.y * select(-1.0, 1.0, d.y > 0.0);
}
// Component of d on the dominant axis of reference.
fn axisComponent(d: vec2f, reference: vec2f) -> f32 {
  if (abs(reference.x) >= abs(reference.y)) { return d.x; }
  return d.y;
}`;

    // 2. Span table: validity and direction, then oriented end points and lines.
    // Words: 0 flags (1 overlap, 2 forward), 1 next span, 2 has previous, 3..6 start and end bits,
    // 7 left line, 8 right line, 9 left segment.
    const spanTable = T('span-table', 'uint32', pairCapacity * SPAN_STRIDE);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-directions`,
        operation: OPERATION,
        variant: 'directions',
        bindings: [
          {name: 'leftPositions', view: left.positions, type: 'f32', access: 'read'},
          {name: 'rightPositions', view: right.positions, type: 'f32', access: 'read'},
          {name: 'pairLeft', view: pairLeft, type: 'u32', access: 'read'},
          {name: 'pairRight', view: pairRight, type: 'u32', access: 'read'},
          {name: 'pairCount', view: pairCount, type: 'u32', access: 'read'},
          {name: 'kinds', view: kinds, type: 'u32', access: 'read'},
          {name: 'spanTable', view: spanTable, type: 'u32', access: 'read_write'}
        ],
        invocationCount: pairCapacity,
        declarations: `${geometryWGSL('left')}${geometryWGSL('right')}${keyWGSL}`,
        body: `let row = spanTableOffset + index * ${SPAN_STRIDE}u;
  var flags = 0u;
  if (index < pairCount[pairCountOffset] && kinds[kindsOffset + index] == 4u) {
    let leftSegment = pairLeft[pairLeftOffset + index];
    let rightSegment = pairRight[pairRightOffset + index];
    let leftDirection = leftVertex(leftSegment + 1u) - leftVertex(leftSegment);
    let rightDirection = rightVertex(rightSegment + 1u) - rightVertex(rightSegment);
    flags = 1u;
    if ((axisComponent(rightDirection, leftDirection) > 0.0) == (axisComponent(leftDirection, leftDirection) > 0.0)) { flags = flags | 2u; }
    spanTable[row + 9u] = leftSegment;
  }
  spanTable[row] = flags;
  spanTable[row + 1u] = 0xffffffffu;
  spanTable[row + 2u] = 0u;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-spans`,
        operation: OPERATION,
        variant: 'spans',
        bindings: [
          {name: 'leftPositions', view: left.positions, type: 'f32', access: 'read'},
          {name: 'pairLeft', view: pairLeft, type: 'u32', access: 'read'},
          {name: 'points', view: points, type: 'f32', access: 'read'},
          {name: 'endPoints', view: endPoints, type: 'f32', access: 'read'},
          {name: 'leftRings', view: leftRings, type: 'u32', access: 'read'},
          {name: 'rightRings', view: rightRings, type: 'u32', access: 'read'},
          {name: 'spanTable', view: spanTable, type: 'u32', access: 'read_write'}
        ],
        invocationCount: pairCapacity,
        declarations: `${geometryWGSL('left')}${keyWGSL}`,
        body: `let row = spanTableOffset + index * ${SPAN_STRIDE}u;
  if ((spanTable[row] & 1u) == 0u) { return; }
  let segment = pairLeft[pairLeftOffset + index];
  let direction = leftVertex(segment + 1u) - leftVertex(segment);
  var first = vec2f(points[pointsOffset + index * 2u], points[pointsOffset + index * 2u + 1u]);
  var second = vec2f(endPoints[endPointsOffset + index * 2u], endPoints[endPointsOffset + index * 2u + 1u]);
  if (keyOf(second, direction) < keyOf(first, direction)) {
    let swap = first;
    first = second;
    second = swap;
  }
  spanTable[row + 3u] = bitcast<u32>(first.x);
  spanTable[row + 4u] = bitcast<u32>(first.y);
  spanTable[row + 5u] = bitcast<u32>(second.x);
  spanTable[row + 6u] = bitcast<u32>(second.y);
  spanTable[row + 7u] = leftRings[leftRingsOffset + index];
  spanTable[row + 8u] = rightRings[rightRingsOffset + index];`
      })
    );

    // 3. Links: the previous and next span of the same left line, right line and direction.
    const spanAccessWGSL = `
const PAIR_CAPACITY: u32 = ${pairCapacity}u;
const SPAN_NONE: u32 = 0xffffffffu;
fn spanWord(span: u32, word: u32) -> u32 { return spanTable[spanTableOffset + span * ${SPAN_STRIDE}u + word]; }
fn isSpan(span: u32) -> bool { return span < PAIR_CAPACITY && (spanWord(span, 0u) & 1u) != 0u; }
fn spanSegment(span: u32) -> u32 { return spanWord(span, 9u); }
// Whether span second starts where span first ends, on the same line pair and direction.
fn follows(first: u32, second: u32) -> bool {
  return isSpan(first) && isSpan(second) && first != second &&
    spanWord(first, 7u) == spanWord(second, 7u) && spanWord(first, 8u) == spanWord(second, 8u) &&
    (spanWord(first, 0u) & 2u) == (spanWord(second, 0u) & 2u) &&
    spanWord(first, 5u) == spanWord(second, 3u) && spanWord(first, 6u) == spanWord(second, 4u);
}`;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-links`,
        operation: OPERATION,
        variant: 'links',
        bindings: [
          {name: 'pairLeft', view: pairLeft, type: 'u32', access: 'read'},
          {name: 'pairCount', view: pairCount, type: 'u32', access: 'read'},
          {name: 'spanTable', view: spanTable, type: 'u32', access: 'read_write'}
        ],
        invocationCount: pairCapacity,
        declarations: `${spanAccessWGSL}
fn lowerBound(value: u32) -> u32 {
  var low = 0u;
  var high = pairCount[pairCountOffset];
  while (low < high) {
    let middle = (low + high) / 2u;
    if (pairLeft[pairLeftOffset + middle] < value) { low = middle + 1u; } else { high = middle; }
  }
  return low;
}`,
        body: `if (!isSpan(index)) { return; }
  let segment = spanSegment(index);
  var first = 0u;
  if (segment > 0u) { first = lowerBound(segment - 1u); }
  let last = lowerBound(segment + 2u);
  var hasPrevious = 0u;
  var next = SPAN_NONE;
  for (var other = first; other < last; other++) {
    if (follows(other, index) && spanSegment(other) <= segment) { hasPrevious = 1u; }
    if (next == SPAN_NONE && follows(index, other) && spanSegment(other) >= segment) { next = other; }
  }
  spanTable[spanTableOffset + index * ${SPAN_STRIDE}u + 1u] = next;
  spanTable[spanTableOffset + index * ${SPAN_STRIDE}u + 2u] = hasPrevious;`
      })
    );

    // 4. Runs: every span without a previous span starts one; walk its chain to size it.
    const runFlags = T('run-flags', 'uint32', pairCapacity);
    const runRanks = T('run-ranks', 'uint32', pairCapacity);
    const vertexCounts = T('vertex-counts', 'uint32', pairCapacity);
    const vertexStarts = T('vertex-starts', 'uint32', pairCapacity);
    const chainWGSL = `${spanAccessWGSL}
fn startsRun(span: u32) -> bool { return isSpan(span) && spanWord(span, 2u) == 0u; }
// Whether the end of span is a vertex of the run: not when the chain goes on in the same segment.
fn emitsEnd(span: u32) -> bool {
  let next = spanWord(span, 1u);
  return next == SPAN_NONE || spanSegment(next) != spanSegment(span);
}`;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-run-sizes`,
        operation: OPERATION,
        variant: 'run-sizes',
        bindings: [
          {name: 'spanTable', view: spanTable, type: 'u32', access: 'read'},
          {name: 'runFlags', view: runFlags, type: 'u32', access: 'read_write'},
          {name: 'vertexCounts', view: vertexCounts, type: 'u32', access: 'read_write'}
        ],
        invocationCount: pairCapacity,
        declarations: chainWGSL,
        body: `var count = 0u;
  if (startsRun(index)) {
    count = 1u;
    var span = index;
    loop {
      if (emitsEnd(span)) { count = count + 1u; }
      span = spanWord(span, 1u);
      if (span == SPAN_NONE) { break; }
    }
  }
  runFlags[runFlagsOffset + index] = select(0u, 1u, startsRun(index));
  vertexCounts[vertexCountsOffset + index] = count;`
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

    // 5. Run table, capacity clamp and outputs.
    const runTable = T('run-table', 'uint32', pairCapacity * RUN_STRIDE);
    const state = T('state', 'uint32', 8);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-run-table`,
        operation: OPERATION,
        variant: 'run-table',
        bindings: [
          {name: 'spanTable', view: spanTable, type: 'u32', access: 'read'},
          {name: 'runFlags', view: runFlags, type: 'u32', access: 'read'},
          {name: 'runRanks', view: runRanks, type: 'u32', access: 'read'},
          {name: 'vertexStarts', view: vertexStarts, type: 'u32', access: 'read'},
          {name: 'runTable', view: runTable, type: 'u32', access: 'read_write'}
        ],
        invocationCount: pairCapacity,
        body: `if (runFlags[runFlagsOffset + index] == 0u) { return; }
  let row = runTableOffset + runRanks[runRanksOffset + index] * ${RUN_STRIDE}u;
  let source = spanTableOffset + index * ${SPAN_STRIDE}u;
  runTable[row] = vertexStarts[vertexStartsOffset + index];
  runTable[row + 1u] = spanTable[source + 7u];
  runTable[row + 2u] = spanTable[source + 8u];
  runTable[row + 3u] = (spanTable[source] >> 1u) & 1u;`
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
          {name: 'pairOverflow', view: pairOverflow, type: 'u32', access: 'read'},
          {name: 'state', view: state, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 1,
        declarations: `const LAST: u32 = ${pairCapacity - 1}u;
const RUN_CAPACITY: u32 = ${runCapacity}u;
const VERTEX_CAPACITY: u32 = ${vertexCapacity}u;`,
        body: `let totalRuns = runRanks[runRanksOffset + LAST] + runFlags[runFlagsOffset + LAST];
  let totalVertices = vertexStarts[vertexStartsOffset + LAST] + vertexCounts[vertexCountsOffset + LAST];
  var low = 0u;
  var high = min(totalRuns, RUN_CAPACITY);
  while (low < high) {
    let middle = (low + high + 1u) / 2u;
    var end = totalVertices;
    if (middle < totalRuns) { end = runTable[runTableOffset + middle * ${RUN_STRIDE}u]; }
    if (end <= VERTEX_CAPACITY) { low = middle; } else { high = middle - 1u; }
  }
  var written = totalVertices;
  if (low < totalRuns) { written = runTable[runTableOffset + low * ${RUN_STRIDE}u]; }
  state[stateOffset] = low;
  state[stateOffset + 1u] = written;
  state[stateOffset + 2u] = totalRuns;
  state[stateOffset + 3u] = totalVertices;
  state[stateOffset + 4u] = select(0u, 1u, low < totalRuns || pairOverflow[pairOverflowOffset] != 0u);`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-write-runs`,
        operation: OPERATION,
        variant: 'write-runs',
        bindings: [
          {name: 'state', view: state, type: 'u32', access: 'read'},
          {name: 'runTable', view: runTable, type: 'u32', access: 'read'},
          {name: 'leftLineIds', view: runs.leftLineIds, type: 'u32', access: 'read_write'},
          {name: 'rightLineIds', view: runs.rightLineIds, type: 'u32', access: 'read_write'},
          {name: 'forward', view: runs.forward, type: 'u32', access: 'read_write'},
          {name: 'offsets', view: runs.offsets, type: 'u32', access: 'read_write'}
        ],
        invocationCount: runCapacity + 1,
        declarations: `const RUN_CAPACITY: u32 = ${runCapacity}u;`,
        body: `if (index < state[stateOffset]) {
    let row = runTableOffset + index * ${RUN_STRIDE}u;
    leftLineIds[leftLineIdsOffset + index] = runTable[row + 1u];
    rightLineIds[rightLineIdsOffset + index] = runTable[row + 2u];
    forward[forwardOffset + index] = runTable[row + 3u];
    offsets[offsetsOffset + index] = runTable[row];
  } else {
    if (index < RUN_CAPACITY) {
      leftLineIds[leftLineIdsOffset + index] = 0xffffffffu;
      rightLineIds[rightLineIdsOffset + index] = 0xffffffffu;
      forward[forwardOffset + index] = 0u;
    }
    offsets[offsetsOffset + index] = state[stateOffset + 1u];
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-write-vertices`,
        operation: OPERATION,
        variant: 'write-vertices',
        bindings: [
          {name: 'state', view: state, type: 'u32', access: 'read'},
          {name: 'spanTable', view: spanTable, type: 'u32', access: 'read'},
          {name: 'runFlags', view: runFlags, type: 'u32', access: 'read'},
          {name: 'runRanks', view: runRanks, type: 'u32', access: 'read'},
          {name: 'vertexStarts', view: vertexStarts, type: 'u32', access: 'read'},
          {name: 'outPositions', view: runs.positions, type: 'f32', access: 'read_write'}
        ],
        invocationCount: pairCapacity,
        declarations: chainWGSL,
        body: `if (runFlags[runFlagsOffset + index] == 0u || runRanks[runRanksOffset + index] >= state[stateOffset]) { return; }
  var slot = vertexStarts[vertexStartsOffset + index];
  outPositions[outPositionsOffset + slot * 2u] = bitcast<f32>(spanWord(index, 3u));
  outPositions[outPositionsOffset + slot * 2u + 1u] = bitcast<f32>(spanWord(index, 4u));
  slot = slot + 1u;
  var span = index;
  loop {
    if (emitsEnd(span)) {
      outPositions[outPositionsOffset + slot * 2u] = bitcast<f32>(spanWord(span, 5u));
      outPositions[outPositionsOffset + slot * 2u + 1u] = bitcast<f32>(spanWord(span, 6u));
      slot = slot + 1u;
    }
    span = spanWord(span, 1u);
    if (span == SPAN_NONE) { break; }
  }`
      })
    );

    // 6. Scalars.
    const scalarBindings: WGSLKernelBinding[] = [
      {name: 'state', view: state, type: 'u32', access: 'read'}
    ];
    const scalars: [string, GraphDataView<'uint32'> | undefined, number][] = [
      ['count', runs.count, 0],
      ['vertexCount', runs.vertexCount, 1],
      ['totalCount', runs.totalCount, 2],
      ['totalVertexCount', runs.totalVertexCount, 3],
      ['overflow', runs.overflow, 4]
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
