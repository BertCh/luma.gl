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
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {validateGraphViewsBelongToGraph} from '../../utils/gpu-contributor-utils';
import {GPUSegmentIntersection} from '../segment-intersection/index';
import {SEGMENT_PREDICATES_WGSL} from '../segment-intersection/segment-intersection-wgsl';
import type {GPUSpatialJoinPolygons} from '../spatial-join/index';
import {
  GPU_GEOMETRY_VALIDITY_BIT,
  type GPUGeometryValidityOrientation,
  type GPUGeometryValidityRingClosure
} from './geometry-validity-types';

const OPERATION = 'GPUGeometryValidity';

/**
 * Properties for {@link GPUGeometryValidity}.
 *
 * Per-frame: the contents of every input buffer. Topology: view lengths, `intersectionCapacity`,
 * `leafCapacity`, `ringClosure`, `orientation`, and which optional views exist.
 */
export type GPUGeometryValidityProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'geometry-validity'`. */
  id?: string;
  /** Polygon or multipolygon features in the GeoArrow layout of `GPUSpatialPredicateJoin`. */
  polygons: GPUSpatialJoinPolygons;
  /** Caller-owned bitmask per feature, `featureCount` rows. See {@link GPU_GEOMETRY_VALIDITY_BIT}. */
  mask: GraphDataView<'uint32'>;
  /**
   * Capacity of the internal list of same-feature segment intersections. Valid polygons produce
   * almost none, so it only needs to cover the intersections of the invalid features, plus ring
   * contacts that are allowed (a hole touching its shell at a point).
   */
  intersectionCapacity: number;
  /**
   * One-row flag that is 1 when the intersection list or the BVH overflowed. Then
   * `selfIntersection` and `crossingRings` may be missing for some features; every other bit is
   * unaffected.
   */
  overflow: GraphDataView<'uint32'>;
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
 * Computes a per-feature validity bitmask for polygons and multipolygons: the GPU analog of the
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
  private readonly polygons: GPUSpatialJoinPolygons;

  constructor(props: GPUGeometryValidityProps) {
    this.id = props.id ?? 'geometry-validity';
    this.props = props;
    const {id} = this;
    const {polygons} = props;
    this.polygons = polygons;
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
    if (this.featureCount < 1) {
      throw new Error(`${id} polygons must contain at least one feature`);
    }
    if (polygons.positions.length < 1) {
      throw new Error(`${id} polygons must contain at least one vertex`);
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
    if (!Number.isSafeInteger(props.intersectionCapacity) || props.intersectionCapacity < 1) {
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
    return [
      polygons.positions,
      polygons.featureOffsets,
      polygons.polygonOffsets,
      polygons.ringOffsets,
      props.mask,
      props.overflow,
      props.intersectionCount
    ];
  }

  /** Returns the mask clear, ring checks, same-feature segment intersection and fold nodes in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, polygons, featureCount, ringCount} = this;
    validateGraphViewsBelongToGraph(id, graph, this.getViews());
    const {mask, intersectionCapacity} = props;
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
          {name: 'mask', view: mask, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: ringCount,
        declarations: `${SEGMENT_PREDICATES_WGSL}
const VERTEX_COUNT: u32 = ${polygons.positions.length}u;
const POLYGON_COUNT: u32 = ${polygons.polygonOffsets.length - 1}u;
const FEATURE_COUNT: u32 = ${polygons.featureOffsets.length - 1}u;
const RING_COUNT: u32 = ${ringCount}u;
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
// Winding of the ring at its lexicographically smallest vertex: 1 counter-clockwise, -1 clockwise,
// 0 undetermined, 2 uncertain.
fn ringWinding(start: u32, count: u32) -> i32 {
  var extreme = 0u;
  for (var k = 1u; k < count; k++) {
    let candidate = vertexAt(start + k);
    let best = vertexAt(start + extreme);
    if (candidate.x < best.x || (candidate.x == best.x && candidate.y < best.y)) { extreme = k; }
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
  }
  if (!foundPrevious || !foundNext) { return 0; }
  return orientSign(vertexAt(start + previous), center, vertexAt(start + next));
}
// 1 inside the ring, 0 outside, 2 on its boundary, 3 uncertain; even/odd with exact orientation.
fn locateInRing(point: vec2f, start: u32, count: u32) -> u32 {
  var inside = false;
  for (var k = 0u; k < count; k++) {
    let a = vertexAt(start + k);
    let b = vertexAt(start + (k + 1u) % count);
    let inBox = point.x >= min(a.x, b.x) && point.x <= max(a.x, b.x) &&
      point.y >= min(a.y, b.y) && point.y <= max(a.y, b.y);
    let straddles = (a.y > point.y) != (b.y > point.y);
    if (!inBox && !straddles) { continue; }
    let side = orientSign(a, b, point);
    if (side == 2) { return 3u; }
    if (inBox && side == 0) { return 2u; }
    if (straddles && ((b.y > a.y) == (side > 0))) { inside = !inside; }
  }
  return select(0u, 1u, inside);
}`,
        body: `let start = ringStartOf(index);
  let end = ringEndOf(index);
  let polygon = lastAtMost(0u, POLYGON_COUNT, index);
  if (polygon == SEGMENT_NONE || index >= polygonOffsets[polygonOffsetsOffset + POLYGON_COUNT]) { return; }
  if (polygon >= featureOffsets[featureOffsetsOffset + FEATURE_COUNT]) { return; }
  let feature = lastAtMost(1u, FEATURE_COUNT, polygon);
  if (feature == SEGMENT_NONE) { return; }
  let isHole = index != polygonOffsets[polygonOffsetsOffset + polygon];
  let count = select(0u, end - start, end > start);
  var bits = 0u;
  var finiteAll = true;
  var repeated = false;
  for (var k = 0u; k < count; k++) {
    let vertex = vertexAt(start + k);
    if (!orientIsFinite(vertex.x) || !orientIsFinite(vertex.y)) { finiteAll = false; }
    if (k + 1u < count && samePoint(vertex, vertexAt(start + k + 1u))) { repeated = true; }
  }
  if (!finiteAll) { bits = bits | BIT_NON_FINITE; }
  if (repeated) { bits = bits | BIT_REPEATED; }
  if (EXPLICIT_CLOSURE && count >= 1u && !samePoint(vertexAt(start), vertexAt(end - 1u))) { bits = bits | BIT_UNCLOSED; }
  let effective = effectiveCount(start, end);
  let isShort = effective < 3u;
  if (isShort) { bits = bits | BIT_SHORT; }
  if (finiteAll && !isShort) {
    if (CHECK_ORIENTATION) {
      let winding = ringWinding(start, effective);
      if (winding == 2) { bits = bits | BIT_UNCERTAIN; }
      else if (winding != 0) {
        let wantsCounterClockwise = select(SHELL_COUNTER_CLOCKWISE, !SHELL_COUNTER_CLOCKWISE, isHole);
        if ((winding > 0) != wantsCounterClockwise) { bits = bits | BIT_ORIENTATION; }
      }
    }
    if (isHole) {
      let shell = polygonOffsets[polygonOffsetsOffset + polygon];
      let shellStart = ringStartOf(shell);
      let shellEnd = ringEndOf(shell);
      let shellCount = effectiveCount(shellStart, shellEnd);
      if (shellCount >= 3u) {
        // One hole vertex off the shell boundary decides containment once no ring edges cross.
        var located = 2u;
        for (var k = 0u; k < effective && located == 2u; k++) {
          located = locateInRing(vertexAt(start + k), shellStart, shellCount);
        }
        if (located == 3u) { bits = bits | BIT_UNCERTAIN; }
        else if (located != 1u) { bits = bits | BIT_HOLE_OUTSIDE; }
      }
    }
  }
  if (bits != 0u) { atomicOr(&mask[maskOffset + feature], bits); }`
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
        overflow: props.overflow,
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
