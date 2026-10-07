// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
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
import {SEGMENT_PREDICATES_WGSL} from '../segment-intersection/segment-intersection-wgsl';
import type {
  GPUSpatialJoinLines,
  GPUSpatialJoinPoints,
  GPUSpatialJoinPolygons
} from '../spatial-join/index';
import {
  GPU_GEOMETRY_VALIDITY_BIT,
  type GPUGeometryValidityOrientation,
  type GPUGeometryValidityRingClosure
} from './geometry-validity-types';

const OPERATION = 'GPUGeometryValidity';

/**
 * Lanes per workgroup. A workgroup owns this many consecutive rings (block hybrid): each lane first
 * checks its own ring serially when it has at most this many vertices; the workgroup then visits
 * the larger rings (and holes needing a containment walk) of its block in ascending order and
 * splits each across the lanes with order-independent (bitwise OR, lexicographic minimum with an
 * index tiebreak, minimum edge index, parity) reductions. Results match the serial code exactly,
 * the thread count stays the ring count rounded up to this value, and one huge ring no longer
 * serializes the stage.
 */
const RING_LANES = 64;

/**
 * Properties for {@link GPUGeometryValidity}.
 *
 * Per-frame: the contents of every input buffer. Topology: view lengths, `intersectionCapacity`,
 * `leafCapacity`, `ringClosure`, `orientation`, and which optional views exist.
 */
export type GPUGeometryValidityProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'geometry-validity'`. */
  id?: string;
  /**
   * Polygon or multipolygon features in the GeoArrow layout of `GPUSpatialPredicateJoin`. Give
   * exactly one of `polygons`, `lines` and `points`.
   */
  polygons?: GPUSpatialJoinPolygons;
  /**
   * Linestring features. Valid means finite coordinates and, for a non-empty line, at least two
   * distinct vertices (Shapely `is_valid` for a `LineString`). Self-intersection is allowed.
   */
  lines?: GPUSpatialJoinLines;
  /** Point features. Valid means finite coordinates (Shapely `is_valid` for a `Point`). */
  points?: GPUSpatialJoinPoints;
  /** Caller-owned bitmask per feature, `featureCount` rows. See {@link GPU_GEOMETRY_VALIDITY_BIT}. Lines and points use only `nonFinite` and `tooFewPoints`. */
  mask: GraphDataView<'uint32'>;
  /**
   * Capacity of the internal list of same-feature segment intersections. Valid polygons produce
   * almost none, so it only needs to cover the intersections of the invalid features, plus ring
   * contacts that are allowed (a hole touching its shell at a point). Required for polygons.
   */
  intersectionCapacity?: number;
  /**
   * One-row flag that is 1 when the intersection list or the BVH overflowed. Then
   * `selfIntersection` and `crossingRings` may be missing for some features; every other bit is
   * unaffected. Required for polygons.
   */
  overflow?: GraphDataView<'uint32'>;
  /** Optional one-row unclamped number of same-feature segment intersections found, for sizing `intersectionCapacity`. */
  intersectionCount?: GraphDataView<'uint32'>;
  /** Power-of-two BVH leaf slots over segments. Defaults to the next power of two of the vertex count. */
  leafCapacity?: number;
  /** Ring closure convention. Defaults to `'implicit'`. */
  ringClosure?: GPUGeometryValidityRingClosure;
  /** Ring orientation convention. Defaults to `'counter-clockwise-shell'`. */
  orientation?: GPUGeometryValidityOrientation;
};

/**
 * Computes a per-feature validity bitmask for polygons, multipolygons, linestrings and points: the GPU analog of the
 * checks behind PostGIS `ST_IsValidDetail`, Shapely `explain_validity` and turf `kinks`, as a mask
 * users can filter on. Repair stays on the CPU.
 *
 * The checks are listed in {@link GPU_GEOMETRY_VALIDITY_BIT}. Per-ring checks (non-finite
 * coordinates, closure, short rings, repeated vertices, orientation, holes outside their shell) run
 * one thread per ring. Self-intersection and crossing rings come from a same-feature
 * {@link GPUSegmentIntersection} in self mode, so they are exact. Nothing is read back.
 *
 * Use `mask & GPU_GEOMETRY_VALIDITY_STRUCTURAL_MASK` to ignore the orientation convention.
 *
 * **Lines and points.** One thread per feature checks finite coordinates and, for lines, that a
 * non-empty linestring has two distinct vertices, which is all Shapely `is_valid` and
 * `is_valid_reason` test for those kinds. They need no intersection list, so `overflow` and
 * `intersectionCapacity` are not required.
 *
 * **Not checked.** Interior connectivity (two holes that touch at two points, or a hole that touches
 * its shell at two points, disconnect the interior), a hole nested in another hole, a polygon nested
 * in another polygon of the same multipolygon, and a ring that leaves its shell only through
 * vertices on the boundary. A hole is tested by one representative vertex that is not on the
 * shell boundary, which is exact once `crossingRings` is clear.
 */
export class GPUGeometryValidity implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUGeometryValidityProps;
  /** Number of features, `featureOffsets.length - 1`. */
  readonly featureCount: number;
  /** Number of rings, `ringOffsets.length - 1`. */
  readonly ringCount: number;
  private readonly polygons?: GPUSpatialJoinPolygons;

  constructor(props: GPUGeometryValidityProps) {
    this.id = props.id ?? 'geometry-validity';
    this.props = props;
    const {id} = this;
    const {polygons, lines, points} = props;
    this.polygons = polygons;
    if ([polygons, lines, points].filter(Boolean).length !== 1) {
      throw new Error(`${id} requires exactly one of polygons, lines and points`);
    }
    if (polygons) {
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
      this.featureCount = polygons.featureOffsets.length - 1;
      this.ringCount = polygons.ringOffsets.length - 1;
      if (polygons.positions.length < 1) {
        throw new Error(`${id} polygons must contain at least one vertex`);
      }
      if (!props.overflow || props.intersectionCapacity === undefined) {
        throw new Error(`${id} polygons require overflow and intersectionCapacity`);
      }
    } else if (lines) {
      validatePackedView(lines.positions, ['float32x2'], `${id} lines.positions`);
      validatePackedUint32View(lines.lineOffsets, `${id} lines.lineOffsets`);
      if (lines.lineOffsets.length < 1) {
        throw new Error(`${id} lines.lineOffsets requires a terminal entry`);
      }
      this.featureCount = lines.lineOffsets.length - 1;
      this.ringCount = 0;
    } else {
      const pointGeometry = points as GPUSpatialJoinPoints;
      validatePackedView(pointGeometry.positions, ['float32x2'], `${id} points.positions`);
      this.featureCount = pointGeometry.positions.length;
      this.ringCount = 0;
    }
    if (this.featureCount < 1) {
      throw new Error(`${id} geometry must contain at least one feature`);
    }
    validatePackedUint32View(props.mask, `${id} mask`);
    if (props.mask.length !== this.featureCount) {
      throw new Error(`${id} mask length must equal the feature count`);
    }
    for (const [name, view] of [
      ['overflow', props.overflow],
      ['intersectionCount', props.intersectionCount]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} ${name}`);
        if (view.length < 1) {
          throw new Error(`${id} ${name} must contain one uint32 row`);
        }
      }
    }
    if (
      polygons &&
      (!Number.isSafeInteger(props.intersectionCapacity) ||
        (props.intersectionCapacity as number) < 1)
    ) {
      throw new Error(`${id} intersectionCapacity must be a positive integer`);
    }
    if (props.ringClosure && !['implicit', 'explicit'].includes(props.ringClosure)) {
      throw new Error(`${id} ringClosure must be implicit or explicit`);
    }
    if (
      props.orientation &&
      !['counter-clockwise-shell', 'clockwise-shell', 'ignore'].includes(props.orientation)
    ) {
      throw new Error(
        `${id} orientation must be counter-clockwise-shell, clockwise-shell or ignore`
      );
    }
  }

  private getViews(): (GraphDataView | undefined)[] {
    const {polygons, props} = this;
    const geometryViews: GraphDataView[] = polygons
      ? [polygons.positions, polygons.featureOffsets, polygons.polygonOffsets, polygons.ringOffsets]
      : props.lines
        ? [props.lines.positions, props.lines.lineOffsets]
        : [(props.points as GPUSpatialJoinPoints).positions];
    return [...geometryViews, props.mask, props.overflow, props.intersectionCount];
  }

  /** One thread per line or point feature: finite coordinates and, for lines, two distinct vertices. */
  private getLineOrPointNode<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): GPUCommandNode<Parameters> {
    const {id, props, featureCount} = this;
    const bit = GPU_GEOMETRY_VALIDITY_BIT;
    const {lines} = props;
    const positions = (lines ?? (props.points as GPUSpatialJoinPoints)).positions;
    const bindings: WGSLKernelBinding[] = [
      {name: 'positions', view: positions, type: 'f32', access: 'read'}
    ];
    if (lines) {
      bindings.push({name: 'lineOffsets', view: lines.lineOffsets, type: 'u32', access: 'read'});
    }
    bindings.push({name: 'mask', view: props.mask, type: 'u32', access: 'read_write'});
    return createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-simple-kinds`,
      operation: OPERATION,
      variant: lines ? 'lines' : 'points',
      bindings,
      invocationCount: featureCount,
      declarations: `${SEGMENT_PREDICATES_WGSL}
const VERTEX_COUNT: u32 = ${positions.length}u;
const BIT_NON_FINITE: u32 = ${bit.nonFinite}u;
const BIT_TOO_FEW: u32 = ${bit.tooFewPoints}u;
fn vertexAt(vertex: u32) -> vec2f {
  return vec2f(positions[positionsOffset + vertex * 2u], positions[positionsOffset + vertex * 2u + 1u]);
}`,
      body: lines
        ? `let start = min(lineOffsets[lineOffsetsOffset + index], VERTEX_COUNT);
  let end = min(lineOffsets[lineOffsetsOffset + index + 1u], VERTEX_COUNT);
  var bits = 0u;
  var distinct = false;
  if (end > start) {
    let first = vertexAt(start);
    for (var k = start; k < end; k++) {
      let vertex = vertexAt(k);
      if (!orientIsFinite(vertex.x) || !orientIsFinite(vertex.y)) { bits = bits | BIT_NON_FINITE; }
      if (vertex.x != first.x || vertex.y != first.y) { distinct = true; }
    }
    if (!distinct) { bits = bits | BIT_TOO_FEW; }
  }
  mask[maskOffset + index] = bits;`
        : `let vertex = vertexAt(index);
  mask[maskOffset + index] = select(BIT_NON_FINITE, 0u, orientIsFinite(vertex.x) && orientIsFinite(vertex.y));`
    });
  }

  /** Returns the mask clear, ring checks, same-feature segment intersection and fold nodes in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, polygons, featureCount, ringCount} = this;
    validateGraphViewsBelongToGraph(id, graph, this.getViews());
    if (!polygons) {
      return [this.getLineOrPointNode(graph)];
    }
    const {mask} = props;
    const intersectionCapacity = props.intersectionCapacity as number;
    const orientation = props.orientation ?? 'counter-clockwise-shell';
    const explicitClosure = props.ringClosure === 'explicit';
    const nodes: GPUCommandNode<Parameters>[] = [];

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-clear`,
        operation: OPERATION,
        variant: 'clear',
        bindings: [{name: 'mask', view: mask, type: 'u32', access: 'read_write'}],
        invocationCount: featureCount,
        body: 'mask[maskOffset + index] = 0u;'
      })
    );

    const bit = GPU_GEOMETRY_VALIDITY_BIT;
    // Bounds of every shell that has holes: a hole vertex outside them is outside the shell, which
    // replaces an O(shell) point-in-ring walk with four comparisons. Other rings are never read.
    const ringBounds = createTransientView(
      graph,
      `${id}-ring-bounds`,
      'float32',
      Math.max(ringCount, 1) * 4
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-ring-bounds`,
        operation: OPERATION,
        variant: 'ring-bounds',
        bindings: [
          {name: 'positions', view: polygons.positions, type: 'f32', access: 'read'},
          {name: 'ringOffsets', view: polygons.ringOffsets, type: 'u32', access: 'read'},
          {name: 'polygonOffsets', view: polygons.polygonOffsets, type: 'u32', access: 'read'},
          {name: 'bounds', view: ringBounds, type: 'f32', access: 'read_write'}
        ],
        invocationCount: Math.ceil(ringCount / RING_LANES) * RING_LANES,
        workgroupSize: RING_LANES,
        guardIndex: false,
        declarations: `${SEGMENT_PREDICATES_WGSL}
const VERTEX_COUNT: u32 = ${polygons.positions.length}u;
const POLYGON_COUNT: u32 = ${polygons.polygonOffsets.length - 1}u;
const RING_COUNT: u32 = ${ringCount}u;
const RING_LANES: u32 = ${RING_LANES}u;
const BOUNDS_LARGE: f32 = 3.4e38;
var<workgroup> queueBits: array<atomic<u32>, 2>;
var<workgroup> queueBitsCopy: array<u32, 2>;
var<workgroup> queuedFirst: array<u32, ${RING_LANES}>;
var<workgroup> queuedLast: array<u32, ${RING_LANES}>;
var<workgroup> partialMinimum: array<vec2f, ${RING_LANES}>;
var<workgroup> partialMaximum: array<vec2f, ${RING_LANES}>;
var<workgroup> partialNonFinite: array<u32, ${RING_LANES}>;
fn vertexAt(vertex: u32) -> vec2f {
  return vec2f(positions[positionsOffset + vertex * 2u], positions[positionsOffset + vertex * 2u + 1u]);
}
// Last polygon whose first ring is at most ring, or SEGMENT_NONE.
fn lastPolygonAtMost(ring: u32) -> u32 {
  if (polygonOffsets[polygonOffsetsOffset] > ring) { return SEGMENT_NONE; }
  var low = 0u;
  var high = POLYGON_COUNT;
  while (high - low > 1u) {
    let middle = (low + high) / 2u;
    if (polygonOffsets[polygonOffsetsOffset + middle] <= ring) { low = middle; } else { high = middle; }
  }
  return low;
}
// Non-finite or empty rings get the whole plane, so the prefilter never rejects through them.
fn writeBounds(ring: u32, minimum: vec2f, maximum: vec2f, nonFinite: u32) {
  let unbounded = nonFinite != 0u || minimum.x > maximum.x;
  let low = select(minimum, vec2f(-BOUNDS_LARGE), unbounded);
  let high = select(maximum, vec2f(BOUNDS_LARGE), unbounded);
  bounds[boundsOffset + 4u * ring] = low.x;
  bounds[boundsOffset + 4u * ring + 1u] = low.y;
  bounds[boundsOffset + 4u * ring + 2u] = high.x;
  bounds[boundsOffset + 4u * ring + 3u] = high.y;
}`,
        body: `let lane = localInvocationIndex;
  let ringBase = workgroupIndex * RING_LANES;
  let ownRing = ringBase + lane;
  // Phase 1: the shell of a polygon with holes is bounded serially by its own lane when small,
  // and queued for the cooperative phase otherwise.
  if (ownRing < RING_COUNT && POLYGON_COUNT > 0u) {
    let polygon = lastPolygonAtMost(ownRing);
    if (polygon != SEGMENT_NONE && ownRing < polygonOffsets[polygonOffsetsOffset + POLYGON_COUNT]) {
      let firstRing = polygonOffsets[polygonOffsetsOffset + polygon];
      let nextRing = polygonOffsets[polygonOffsetsOffset + polygon + 1u];
      if (ownRing == firstRing && nextRing - firstRing >= 2u) {
        let first = ringOffsets[ringOffsetsOffset + ownRing];
        let last = min(ringOffsets[ringOffsetsOffset + ownRing + 1u], VERTEX_COUNT);
        if (last <= first || last - first <= RING_LANES) {
          var minimum = vec2f(BOUNDS_LARGE);
          var maximum = vec2f(-BOUNDS_LARGE);
          var nonFinite = 0u;
          for (var vertex = first; vertex < last; vertex++) {
            let point = vertexAt(vertex);
            if (!orientIsFinite(point.x) || !orientIsFinite(point.y)) {
              nonFinite = 1u;
            } else {
              minimum = min(minimum, point);
              maximum = max(maximum, point);
            }
          }
          writeBounds(ownRing, minimum, maximum, nonFinite);
        } else {
          queuedFirst[lane] = first;
          queuedLast[lane] = last;
          atomicOr(&queueBits[lane / 32u], 1u << (lane % 32u));
        }
      }
    }
  }
  workgroupBarrier();
  if (lane == 0u) {
    queueBitsCopy[0] = atomicLoad(&queueBits[0]);
    queueBitsCopy[1] = atomicLoad(&queueBits[1]);
  }
  // Phase 2: queued rings in ascending order; all barriers are in workgroup-uniform control flow.
  for (var word = 0u; word < 2u; word++) {
    var pending = workgroupUniformLoad(&queueBitsCopy[word]);
    while (pending != 0u) {
      let slot = word * 32u + firstTrailingBit(pending);
      pending = pending & (pending - 1u);
      let first = workgroupUniformLoad(&queuedFirst[slot]);
      let last = workgroupUniformLoad(&queuedLast[slot]);
      var minimum = vec2f(BOUNDS_LARGE);
      var maximum = vec2f(-BOUNDS_LARGE);
      var nonFinite = 0u;
      for (var vertex = first + lane; vertex < last; vertex += RING_LANES) {
        let point = vertexAt(vertex);
        if (!orientIsFinite(point.x) || !orientIsFinite(point.y)) {
          nonFinite = 1u;
        } else {
          minimum = min(minimum, point);
          maximum = max(maximum, point);
        }
      }
      partialMinimum[lane] = minimum;
      partialMaximum[lane] = maximum;
      partialNonFinite[lane] = nonFinite;
      workgroupBarrier();
      for (var stride = RING_LANES / 2u; stride > 0u; stride = stride / 2u) {
        if (lane < stride) {
          partialMinimum[lane] = min(partialMinimum[lane], partialMinimum[lane + stride]);
          partialMaximum[lane] = max(partialMaximum[lane], partialMaximum[lane + stride]);
          partialNonFinite[lane] = partialNonFinite[lane] | partialNonFinite[lane + stride];
        }
        workgroupBarrier();
      }
      if (lane == 0u) {
        writeBounds(ringBase + slot, partialMinimum[0], partialMaximum[0], partialNonFinite[0]);
      }
    }
  }`
      })
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-rings`,
        operation: OPERATION,
        variant: 'rings',
        bindings: [
          {name: 'positions', view: polygons.positions, type: 'f32', access: 'read'},
          {name: 'ringOffsets', view: polygons.ringOffsets, type: 'u32', access: 'read'},
          {name: 'polygonOffsets', view: polygons.polygonOffsets, type: 'u32', access: 'read'},
          {name: 'featureOffsets', view: polygons.featureOffsets, type: 'u32', access: 'read'},
          {name: 'ringBounds', view: ringBounds, type: 'f32', access: 'read'},
          {name: 'mask', view: mask, type: 'atomic<u32>', access: 'read_write'}
        ],
        // Block hybrid: one workgroup per RING_LANES consecutive rings.
        invocationCount: Math.ceil(ringCount / RING_LANES) * RING_LANES,
        workgroupSize: RING_LANES,
        guardIndex: false,
        declarations: `${SEGMENT_PREDICATES_WGSL}
const VERTEX_COUNT: u32 = ${polygons.positions.length}u;
const POLYGON_COUNT: u32 = ${polygons.polygonOffsets.length - 1}u;
const FEATURE_COUNT: u32 = ${polygons.featureOffsets.length - 1}u;
const RING_COUNT: u32 = ${ringCount}u;
const RING_LANES: u32 = ${RING_LANES}u;
const EXPLICIT_CLOSURE: bool = ${explicitClosure};
const CHECK_ORIENTATION: bool = ${orientation !== 'ignore'};
const SHELL_COUNTER_CLOCKWISE: bool = ${orientation !== 'clockwise-shell'};
const BIT_NON_FINITE: u32 = ${bit.nonFinite}u;
const BIT_UNCLOSED: u32 = ${bit.unclosedRing}u;
const BIT_SHORT: u32 = ${bit.shortRing}u;
const BIT_REPEATED: u32 = ${bit.repeatedVertex}u;
const BIT_HOLE_OUTSIDE: u32 = ${bit.holeOutsideShell}u;
const BIT_ORIENTATION: u32 = ${bit.badOrientation}u;
const BIT_UNCERTAIN: u32 = ${bit.uncertain}u;
fn vertexAt(vertex: u32) -> vec2f {
  return vec2f(positions[positionsOffset + vertex * 2u], positions[positionsOffset + vertex * 2u + 1u]);
}
fn samePoint(first: vec2f, second: vec2f) -> bool { return first.x == second.x && first.y == second.y; }
fn readOffset(kind: u32, row: u32) -> u32 {
  if (kind == 0u) { return polygonOffsets[polygonOffsetsOffset + row]; }
  return featureOffsets[featureOffsetsOffset + row];
}
// Last row in [0, count) whose offset is at most value, or SEGMENT_NONE.
fn lastAtMost(kind: u32, count: u32, value: u32) -> u32 {
  if (count == 0u || readOffset(kind, 0u) > value) { return SEGMENT_NONE; }
  var low = 0u;
  var high = count;
  while (high - low > 1u) {
    let middle = (low + high) / 2u;
    if (readOffset(kind, middle) <= value) { low = middle; } else { high = middle; }
  }
  return low;
}
fn ringStartOf(ring: u32) -> u32 { return ringOffsets[ringOffsetsOffset + ring]; }
fn ringEndOf(ring: u32) -> u32 { return min(ringOffsets[ringOffsetsOffset + ring + 1u], VERTEX_COUNT); }
// Vertex count without a closing duplicate of the first vertex.
fn effectiveCount(start: u32, end: u32) -> u32 {
  if (end <= start) { return 0u; }
  let count = end - start;
  if (count >= 2u && samePoint(vertexAt(start), vertexAt(end - 1u))) { return count - 1u; }
  return count;
}
const NO_INDEX: u32 = 0xffffffffu;
// Per-ring state of the block, indexed by the ring's lane. Written by the owner lane in phase 1 and
// by lane 0 for cooperatively processed rings; read through workgroupUniformLoad when it steers a barrier.
var<workgroup> ringActive: array<u32, ${RING_LANES}>;
var<workgroup> ringFeature: array<u32, ${RING_LANES}>;
var<workgroup> ringIsHole: array<u32, ${RING_LANES}>;
var<workgroup> ringStart: array<u32, ${RING_LANES}>;
var<workgroup> ringEnd: array<u32, ${RING_LANES}>;
var<workgroup> ringFirstRing: array<u32, ${RING_LANES}>;
var<workgroup> ringLarge: array<u32, ${RING_LANES}>;
var<workgroup> ringBits: array<u32, ${RING_LANES}>;
var<workgroup> ringNeeds: array<u32, ${RING_LANES}>;
var<workgroup> ringEffective: array<u32, ${RING_LANES}>;
var<workgroup> ringShellStart: array<u32, ${RING_LANES}>;
var<workgroup> ringShellCount: array<u32, ${RING_LANES}>;
var<workgroup> ringShellRing: array<u32, ${RING_LANES}>;
var<workgroup> queueBits: array<atomic<u32>, 2>;
var<workgroup> queueBitsCopy: array<u32, 2>;
var<workgroup> sharedPoint: vec2f;
var<workgroup> sharedResult: array<u32, 2>;
var<workgroup> partialFlags: array<u32, ${RING_LANES}>;
var<workgroup> partialExtreme: array<u32, ${RING_LANES}>;
var<workgroup> partialPoint: array<vec2f, ${RING_LANES}>;
var<workgroup> partialEdge: array<u32, ${RING_LANES}>;
var<workgroup> partialKind: array<u32, ${RING_LANES}>;
var<workgroup> partialParity: array<u32, ${RING_LANES}>;
fn isLexicographicallyLess(candidate: vec2f, best: vec2f) -> bool {
  return candidate.x < best.x || (candidate.x == best.x && candidate.y < best.y);
}
// Winding of the ring at its lexicographically smallest vertex (first one on ties): 1
// counter-clockwise, -1 clockwise, 0 undetermined, 2 uncertain. A known extreme (from the
// cooperative search) skips the serial search when it is not NO_INDEX.
fn ringWinding(start: u32, count: u32, knownExtreme: u32) -> i32 {
  var extreme = 0u;
  if (knownExtreme != NO_INDEX) {
    extreme = knownExtreme;
  } else {
    for (var k = 1u; k < count; k++) {
      if (isLexicographicallyLess(vertexAt(start + k), vertexAt(start + extreme))) { extreme = k; }
    }
  }
  let center = vertexAt(start + extreme);
  var previous = extreme;
  var next = extreme;
  var foundPrevious = false;
  var foundNext = false;
  for (var step = 1u; step < count; step++) {
    if (!foundPrevious) {
      previous = (extreme + count - step) % count;
      foundPrevious = !samePoint(vertexAt(start + previous), center);
    }
    if (!foundNext) {
      next = (extreme + step) % count;
      foundNext = !samePoint(vertexAt(start + next), center);
    }
    // Both neighbours are normally the adjacent vertices: stop instead of walking the whole ring.
    if (foundPrevious && foundNext) { break; }
  }
  if (!foundPrevious || !foundNext) { return 0; }
  return orientSign(vertexAt(start + previous), center, vertexAt(start + next));
}
// Folds the ring-local checks of ring slot into ringBits and decides whether the ring is a hole
// whose containment in its shell must be tested (ringNeeds, with the shell in the other slots arrays).
fn finishRing(slot: u32, start: u32, end: u32, isHole: bool, polygonFirstRing: u32, finiteAll: bool, repeated: bool, extreme: u32) {
  let count = select(0u, end - start, end > start);
  var bits = 0u;
  if (!finiteAll) { bits = bits | BIT_NON_FINITE; }
  if (repeated) { bits = bits | BIT_REPEATED; }
  if (EXPLICIT_CLOSURE && count >= 1u && !samePoint(vertexAt(start), vertexAt(end - 1u))) { bits = bits | BIT_UNCLOSED; }
  let effective = effectiveCount(start, end);
  let isShort = effective < 3u;
  if (isShort) { bits = bits | BIT_SHORT; }
  var needsContainment = 0u;
  if (finiteAll && !isShort) {
    if (CHECK_ORIENTATION) {
      let winding = ringWinding(start, effective, extreme);
      if (winding == 2) { bits = bits | BIT_UNCERTAIN; }
      else if (winding != 0) {
        let wantsCounterClockwise = select(SHELL_COUNTER_CLOCKWISE, !SHELL_COUNTER_CLOCKWISE, isHole);
        if ((winding > 0) != wantsCounterClockwise) { bits = bits | BIT_ORIENTATION; }
      }
    }
    if (isHole) {
      let shellStart = ringStartOf(polygonFirstRing);
      let shellCount = effectiveCount(shellStart, ringEndOf(polygonFirstRing));
      if (shellCount >= 3u) {
        needsContainment = 1u;
        ringEffective[slot] = effective;
        ringShellStart[slot] = shellStart;
        ringShellCount[slot] = shellCount;
        ringShellRing[slot] = polygonFirstRing;
      }
    }
  }
  ringBits[slot] = bits;
  ringNeeds[slot] = needsContainment;
}
// Workgroup-cooperative point-in-ring: 1 inside, 0 outside, 2 on the boundary, 3 uncertain, with
// the serial walk's result: the event (boundary or uncertain) at the lowest edge index wins,
// otherwise the crossing parity decides. A point outside the shell's bounds is outside it.
// Must be called from workgroup-uniform control flow with a uniform argument.
fn locateInShell(point: vec2f, shellStart: u32, count: u32, shellRing: u32, lane: u32) -> u32 {
  if (lane == 0u) {
    let base = ringBoundsOffset + 4u * shellRing;
    sharedResult[0] = select(0u, 1u,
      point.x < ringBounds[base] || point.x > ringBounds[base + 2u] ||
      point.y < ringBounds[base + 1u] || point.y > ringBounds[base + 3u]);
  }
  if (workgroupUniformLoad(&sharedResult[0]) == 1u) { return 0u; }
  var eventEdge = NO_INDEX;
  var eventKind = 0u;
  var parity = 0u;
  for (var k = lane; k < count; k += RING_LANES) {
    let a = vertexAt(shellStart + k);
    let b = vertexAt(shellStart + (k + 1u) % count);
    let inBox = point.x >= min(a.x, b.x) && point.x <= max(a.x, b.x) &&
      point.y >= min(a.y, b.y) && point.y <= max(a.y, b.y);
    let straddles = (a.y > point.y) != (b.y > point.y);
    if (!inBox && !straddles) { continue; }
    let side = orientSign(a, b, point);
    if (side == 2) { eventEdge = k; eventKind = 3u; break; }
    if (inBox && side == 0) { eventEdge = k; eventKind = 2u; break; }
    if (straddles && ((b.y > a.y) == (side > 0))) { parity = parity ^ 1u; }
  }
  partialEdge[lane] = eventEdge;
  partialKind[lane] = eventKind;
  partialParity[lane] = parity;
  workgroupBarrier();
  for (var stride = RING_LANES / 2u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      if (partialEdge[lane + stride] < partialEdge[lane]) {
        partialEdge[lane] = partialEdge[lane + stride];
        partialKind[lane] = partialKind[lane + stride];
      }
      partialParity[lane] = partialParity[lane] ^ partialParity[lane + stride];
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    sharedResult[1] = select(partialParity[0], partialKind[0], partialEdge[0] != NO_INDEX);
  }
  return workgroupUniformLoad(&sharedResult[1]);
}`,
        body: `let lane = localInvocationIndex;
  let ring = workgroupIndex * RING_LANES + lane;
  // Phase 1: this lane's own ring. Small rings run the original serial checks; large rings and
  // holes that need a containment walk are queued (bitmask over the block's lanes).
  if (ring < RING_COUNT) {
    let polygon = lastAtMost(0u, POLYGON_COUNT, ring);
    if (polygon != SEGMENT_NONE && ring < polygonOffsets[polygonOffsetsOffset + POLYGON_COUNT] &&
        polygon < featureOffsets[featureOffsetsOffset + FEATURE_COUNT]) {
      let feature = lastAtMost(1u, FEATURE_COUNT, polygon);
      if (feature != SEGMENT_NONE) {
        let start = ringStartOf(ring);
        let end = ringEndOf(ring);
        let isHole = ring != polygonOffsets[polygonOffsetsOffset + polygon];
        let firstRing = polygonOffsets[polygonOffsetsOffset + polygon];
        ringActive[lane] = 1u;
        ringFeature[lane] = feature;
        ringIsHole[lane] = select(0u, 1u, isHole);
        ringStart[lane] = start;
        ringEnd[lane] = end;
        ringFirstRing[lane] = firstRing;
        let count = select(0u, end - start, end > start);
        if (count <= RING_LANES) {
          var finiteAll = true;
          var repeated = false;
          for (var k = 0u; k < count; k++) {
            let vertex = vertexAt(start + k);
            if (!orientIsFinite(vertex.x) || !orientIsFinite(vertex.y)) { finiteAll = false; }
            if (k + 1u < count && samePoint(vertex, vertexAt(start + k + 1u))) { repeated = true; }
          }
          finishRing(lane, start, end, isHole, firstRing, finiteAll, repeated, NO_INDEX);
          if (ringNeeds[lane] == 1u) { atomicOr(&queueBits[lane / 32u], 1u << (lane % 32u)); }
        } else {
          ringLarge[lane] = 1u;
          atomicOr(&queueBits[lane / 32u], 1u << (lane % 32u));
        }
      }
    }
  }
  workgroupBarrier();
  if (lane == 0u) {
    queueBitsCopy[0] = atomicLoad(&queueBits[0]);
    queueBitsCopy[1] = atomicLoad(&queueBits[1]);
  }
  // Phase 2: queued rings in ascending order. Every barrier is in workgroup-uniform control flow:
  // the queue and everything that steers a loop or branch go through workgroupUniformLoad.
  for (var word = 0u; word < 2u; word++) {
    var pending = workgroupUniformLoad(&queueBitsCopy[word]);
    while (pending != 0u) {
      let slot = word * 32u + firstTrailingBit(pending);
      pending = pending & (pending - 1u);
      let start = workgroupUniformLoad(&ringStart[slot]);
      if (workgroupUniformLoad(&ringLarge[slot]) == 1u) {
        // Large ring: lanes scan strided vertices for non-finite coordinates, repeated neighbours
        // and the lexicographic minimum, then a tree combines them.
        let end = workgroupUniformLoad(&ringEnd[slot]);
        let count = select(0u, end - start, end > start);
        let effective = effectiveCount(start, end);
        var flags = 0u;
        var best = NO_INDEX;
        var bestPoint = vec2f(0.0);
        for (var k = lane; k < count; k += RING_LANES) {
          let vertex = vertexAt(start + k);
          if (!orientIsFinite(vertex.x) || !orientIsFinite(vertex.y)) { flags = flags | 1u; }
          if (k + 1u < count && samePoint(vertex, vertexAt(start + k + 1u))) { flags = flags | 2u; }
          if (k < effective && (best == NO_INDEX || isLexicographicallyLess(vertex, bestPoint))) {
            best = k;
            bestPoint = vertex;
          }
        }
        partialFlags[lane] = flags;
        partialExtreme[lane] = best;
        partialPoint[lane] = bestPoint;
        workgroupBarrier();
        for (var stride = RING_LANES / 2u; stride > 0u; stride = stride / 2u) {
          if (lane < stride) {
            partialFlags[lane] = partialFlags[lane] | partialFlags[lane + stride];
            let other = partialExtreme[lane + stride];
            let current = partialExtreme[lane];
            if (other != NO_INDEX && (current == NO_INDEX ||
                isLexicographicallyLess(partialPoint[lane + stride], partialPoint[lane]) ||
                (partialPoint[lane + stride].x == partialPoint[lane].x &&
                 partialPoint[lane + stride].y == partialPoint[lane].y && other < current))) {
              partialExtreme[lane] = other;
              partialPoint[lane] = partialPoint[lane + stride];
            }
          }
          workgroupBarrier();
        }
        if (lane == 0u) {
          finishRing(slot, start, end, ringIsHole[slot] == 1u, ringFirstRing[slot],
            (partialFlags[0] & 1u) == 0u, (partialFlags[0] & 2u) != 0u, partialExtreme[0]);
        }
      }
      // The hole-in-shell test runs on the whole workgroup whatever the hole's size, because the
      // shell may be huge. One hole vertex off the shell boundary decides containment once no ring
      // edges cross.
      if (workgroupUniformLoad(&ringNeeds[slot]) == 1u) {
        let effective = workgroupUniformLoad(&ringEffective[slot]);
        let shellStart = workgroupUniformLoad(&ringShellStart[slot]);
        let shellCount = workgroupUniformLoad(&ringShellCount[slot]);
        let shellRing = workgroupUniformLoad(&ringShellRing[slot]);
        var located = 2u;
        for (var k = 0u; k < effective && located == 2u; k++) {
          if (lane == 0u) { sharedPoint = vertexAt(start + k); }
          let point = workgroupUniformLoad(&sharedPoint);
          located = locateInShell(point, shellStart, shellCount, shellRing, lane);
        }
        if (lane == 0u) {
          if (located == 3u) { ringBits[slot] = ringBits[slot] | BIT_UNCERTAIN; }
          else if (located != 1u) { ringBits[slot] = ringBits[slot] | BIT_HOLE_OUTSIDE; }
        }
      }
    }
  }
  workgroupBarrier();
  if (ringActive[lane] == 1u && ringBits[lane] != 0u) {
    atomicOr(&mask[maskOffset + ringFeature[lane]], ringBits[lane]);
  }`
      })
    );

    // Same-feature segment intersections of the whole set.
    const pairLeft = createTransientView(graph, `${id}-pair-left`, 'uint32', intersectionCapacity);
    const pairRight = createTransientView(
      graph,
      `${id}-pair-right`,
      'uint32',
      intersectionCapacity
    );
    const pairCount = createTransientView(graph, `${id}-pair-count`, 'uint32', 1);
    const kinds = createTransientView(graph, `${id}-kinds`, 'uint32', intersectionCapacity);
    const leftFeatures = createTransientView(
      graph,
      `${id}-left-features`,
      'uint32',
      intersectionCapacity
    );
    const leftRings = createTransientView(
      graph,
      `${id}-left-rings`,
      'uint32',
      intersectionCapacity
    );
    const rightRings = createTransientView(
      graph,
      `${id}-right-rings`,
      'uint32',
      intersectionCapacity
    );
    const intersection = new GPUSegmentIntersection({
      id: `${id}-intersection`,
      left: polygons,
      sameFeatureOnly: true,
      leafCapacity: props.leafCapacity,
      pairs: {
        leftIds: pairLeft,
        rightIds: pairRight,
        count: pairCount,
        overflow: props.overflow as GraphDataView<'uint32'>,
        totalCount: props.intersectionCount
      },
      kinds,
      leftFeatures,
      leftRings,
      rightRings
    });
    nodes.push(...intersection.getCommandNodes(graph));

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-fold`,
        operation: OPERATION,
        variant: 'fold',
        bindings: [
          {name: 'pairCount', view: pairCount, type: 'u32', access: 'read'},
          {name: 'kinds', view: kinds, type: 'u32', access: 'read'},
          {name: 'leftFeatures', view: leftFeatures, type: 'u32', access: 'read'},
          {name: 'leftRings', view: leftRings, type: 'u32', access: 'read'},
          {name: 'rightRings', view: rightRings, type: 'u32', access: 'read'},
          {name: 'mask', view: mask, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: intersectionCapacity,
        declarations: `${SEGMENT_PREDICATES_WGSL}
const BIT_SELF: u32 = ${bit.selfIntersection}u;
const BIT_CROSSING: u32 = ${bit.crossingRings}u;
const BIT_UNCERTAIN: u32 = ${bit.uncertain}u;`,
        body: `if (index >= pairCount[pairCountOffset]) { return; }
  let kind = kinds[kindsOffset + index];
  var bits = 0u;
  if (kind == KIND_UNCERTAIN) {
    bits = BIT_UNCERTAIN;
  } else if (leftRings[leftRingsOffset + index] == rightRings[rightRingsOffset + index]) {
    bits = BIT_SELF;
  } else if (kind == KIND_PROPER || kind == KIND_OVERLAP) {
    bits = BIT_CROSSING;
  }
  if (bits != 0u) { atomicOr(&mask[maskOffset + leftFeatures[leftFeaturesOffset + index]], bits); }`
      })
    );
    return nodes;
  }
}
