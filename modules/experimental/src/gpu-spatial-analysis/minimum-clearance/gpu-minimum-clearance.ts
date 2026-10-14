// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {IS_NAN_BITS_WGSL} from '../../utils/wgsl-common';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validatePackedView,
  validatePowerOfTwo
} from '../../utils/gpu-contributor-utils';
import {getSegmentBVHNodes} from '../segment-intersection/segment-bvh';
import {createSegmentTableNode} from '../segment-intersection/segment-table';
import {
  getSegmentTableAccessorsWGSL,
  SEGMENT_TABLE_STRIDE
} from '../segment-intersection/segment-intersection-wgsl';
import type {GPUSegmentGeometry} from '../segment-intersection/segment-intersection-types';
import {getGPUSegmentIntersectionMaximumVertexCount} from '../segment-intersection/segment-intersection-wgsl';
import {GPU_SEGMENT_NONE} from '../segment-intersection/segment-intersection-types';
import {getNextPowerOfTwo} from '../../utils/gpu-contributor-utils';

const OPERATION = 'GPUMinimumClearance';

/** Vertex count from which `spatialSort` defaults to on. */
const SPATIAL_SORT_MINIMUM_SLOTS = 1024;

/** Lanes of the per-feature minimum workgroup. */
const FEATURE_REDUCTION_WORKGROUP_SIZE = 64;

/** Caller-owned per-feature outputs of {@link GPUMinimumClearance}; at least one is required. */
export type GPUMinimumClearanceOutput = {
  /**
   * Caller-owned `float32` clearance per feature, like `shapely.minimum_clearance`. `Infinity` for
   * a feature with no clearance (empty, a single distinct point, or a degenerate linestring).
   */
  clearances?: GraphDataView<'float32'>;
  /**
   * Caller-owned `float32x4` per feature, like `shapely.minimum_clearance_line`: `(x0, y0, x1, y1)`
   * where `(x0, y0)` is the fragile vertex and `(x1, y1)` the nearest point it must not approach (a
   * point on a non-adjacent segment, or another vertex). All four are `NaN` for a feature with no
   * clearance (GEOS returns an empty line).
   */
  lines?: GraphDataView<'float32x4'>;
  /**
   * Optional caller-owned `uint32` per feature: the row of the fragile vertex in the `positions`
   * view, or `0xffffffff` when the feature has no clearance.
   */
  vertexIds?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUMinimumClearance}.
 *
 * Per-frame: the contents of every input buffer. Topology: view lengths, `leafCapacity`,
 * `spatialSort`, and which optional outputs exist.
 */
export type GPUMinimumClearanceProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'minimum-clearance'`. */
  id?: string;
  /**
   * Planar only. Spatial context of `positions`; omitted means native planar coordinates.
   * Longitude/latitude (or any other non-planar) contexts throw at construction: project
   * positions to a planar frame first, for example with `createGPULocalFrame` and
   * `GPULocalFrameProjection`, and pass the frame's context so its `frame` identity is kept.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * Linestring or polygon features in the layouts of `GPUSpatialPredicateJoin`. Polygon rings may
   * be explicitly closed or not; rings with fewer than three vertices are ignored. At most
   * {@link getGPUSegmentIntersectionMaximumVertexCount} vertex rows (4194304 on a default device; the 32-byte segment table
   * per vertex must fit one storage binding).
   */
  geometry: GPUSegmentGeometry;
  /** Caller-owned outputs; at least one column is required. */
  output: GPUMinimumClearanceOutput;
  /** Power-of-two BVH leaf slots over the segments. Defaults to the next power of two of the vertex count. */
  leafCapacity?: number;
  /**
   * Compile-time. Morton-sorts the segments before the BVH build, which keeps traversal short for
   * sources whose vertex order jumps around the plane. The result is identical either way.
   * Defaults to `true` from 1024 vertices on.
   */
  spatialSort?: boolean;
};

/**
 * Computes the minimum clearance of every feature: the GPU analog of `shapely.minimum_clearance`
 * and `shapely.minimum_clearance_line` (GEOS `MinimumClearance`). The clearance is the smallest
 * distance by which one vertex could move to make the feature invalid: the minimum, over every
 * vertex of a feature, of its distance to any other vertex of the feature and to any segment of the
 * feature that does not have the vertex as an endpoint. Repeated points are ignored, and the
 * clearance is `Infinity` when there is nothing to measure.
 *
 * Pipeline: a segment table and a `GPUBVH` over all segments, one nearest-neighbor probe per
 * vertex (branch and bound, nearest child first, restricted to the vertex's own feature; each
 * segment contributes its distance to the vertex, or the distance to its other endpoint when the
 * vertex is one of its endpoints), then a per-feature minimum: one 64-lane workgroup per feature
 * strides over the feature's contiguous vertex range and reduces the lexicographic minimum of
 * (distance bits, vertex row) with a fixed binary tree (no atomics). Non-negative f32 values order
 * like their bit patterns, so the result is deterministic and ties go to the lowest vertex row.
 * Nothing is read back.
 *
 * **Precision.** Distances are f32. Subtract a common origin from large coordinates, as for
 * `GPUGeometryMeasures`; distances far below the coordinate magnitude times 1e-7 are not resolved.
 *
 * **Cost.** The probe is bounded per vertex by the BVH, but the BVH spans all features, so densely
 * overlapping features visit each other's nodes before the vertex's own best distance prunes
 * them. Vertex pairs with equal distance report the closest pair of the lowest vertex row, which
 * may differ from GEOS on exact ties (the distance never differs).
 *
 * **Differences from GEOS.** Polygon rings with fewer than three vertices are ignored and
 * `NaN` coordinates are skipped instead of poisoning the feature.
 *
 * Planar only: `spatialContext` defaults to native planar coordinates and a longitude/latitude
 * context throws at construction. Project first, for example with `createGPULocalFrame` and
 * `GPULocalFrameProjection`, and pass the frame's context so its `frame` identity is kept.
 */
export class GPUMinimumClearance implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUMinimumClearanceProps;
  /** Resolved spatial context (planar). */
  readonly spatialContext: GPUSpatialContext;
  /** Number of features. */
  readonly featureCount: number;
  /** Number of vertices, which is also the number of segment slots. */
  readonly vertexCount: number;
  /** Resolved BVH leaf capacity. */
  readonly leafCapacity: number;
  /** Whether segments are Morton sorted before the BVH build. */
  readonly spatialSort: boolean;

  constructor(props: GPUMinimumClearanceProps) {
    this.id = props.id ?? 'minimum-clearance';
    this.props = props;
    this.spatialContext = resolveGPUSpatialContext(this.id, props.spatialContext);
    const {id} = this;
    const {geometry} = props;
    validatePackedView(geometry.positions, ['float32x2'], `${id} geometry.positions`);
    const offsetViews =
      geometry.kind === 'lines'
        ? [['lineOffsets', geometry.lineOffsets] as const]
        : [
            ['featureOffsets', geometry.featureOffsets] as const,
            ['polygonOffsets', geometry.polygonOffsets] as const,
            ['ringOffsets', geometry.ringOffsets] as const
          ];
    for (const [name, view] of offsetViews) {
      validatePackedView(view, ['uint32'], `${id} geometry.${name}`);
      if (view.length < 1) {
        throw new Error(`${id} geometry.${name} requires a terminal entry`);
      }
    }
    this.vertexCount = geometry.positions.length;
    if (this.vertexCount < 1) {
      throw new Error(`${id} geometry must contain at least one vertex`);
    }
    this.featureCount =
      (geometry.kind === 'lines' ? geometry.lineOffsets.length : geometry.featureOffsets.length) -
      1;
    if (this.featureCount < 1) {
      throw new Error(`${id} geometry must contain at least one feature`);
    }
    const {clearances, lines, vertexIds} = props.output;
    if (!clearances && !lines && !vertexIds) {
      throw new Error(`${id} requires at least one of clearances, lines or vertexIds`);
    }
    for (const [name, view, format] of [
      ['clearances', clearances, 'float32'],
      ['lines', lines, 'float32x4'],
      ['vertexIds', vertexIds, 'uint32']
    ] as const) {
      if (view) {
        validatePackedView(view, [format], `${id} output.${name}`);
        if (view.length !== this.featureCount) {
          throw new Error(`${id} output.${name} length must equal the feature count`);
        }
      }
    }
    this.spatialSort = props.spatialSort ?? this.vertexCount >= SPATIAL_SORT_MINIMUM_SLOTS;
    this.leafCapacity = props.leafCapacity ?? getNextPowerOfTwo(this.vertexCount);
    validatePowerOfTwo(id, 'leafCapacity', this.leafCapacity);
  }

  private getInputViews(): GraphDataView[] {
    const {geometry} = this.props;
    return geometry.kind === 'lines'
      ? [geometry.positions, geometry.lineOffsets]
      : [
          geometry.positions,
          geometry.featureOffsets,
          geometry.polygonOffsets,
          geometry.ringOffsets
        ];
  }

  private getOutputViews(): (GraphDataView | undefined)[] {
    const {clearances, lines, vertexIds} = this.props.output;
    return [clearances, lines, vertexIds];
  }

  /** Returns table, BVH, vertex feature, probe, per-feature minimum and output nodes in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, featureCount, vertexCount, leafCapacity, spatialSort} = this;
    const {geometry} = props;
    const {clearances, lines, vertexIds} = props.output;
    validateGraphViewsBelongToGraph(id, graph, [...this.getInputViews(), ...this.getOutputViews()]);
    // The segment table holds 32 bytes per vertex, as in GPUSegmentIntersection.
    validateMaximumCount(
      id,
      'geometry.positions rows',
      vertexCount,
      getGPUSegmentIntersectionMaximumVertexCount(graph.device)
    );
    validateGraphOutputsDisjointFromInputs(id, this.getOutputViews(), this.getInputViews());
    const nodes: GPUCommandNode<Parameters>[] = [];

    const table = createTransientView(
      graph,
      `${id}-table`,
      'uint32',
      vertexCount * SEGMENT_TABLE_STRIDE
    );
    const minima = createTransientView(graph, `${id}-bounds-minima`, 'float32x2', vertexCount);
    const maxima = createTransientView(graph, `${id}-bounds-maxima`, 'float32x2', vertexCount);
    nodes.push(
      createSegmentTableNode<Parameters>(graph, {
        id: `${id}-table`,
        operation: OPERATION,
        geometry,
        table,
        minima,
        maxima
      })
    );
    const {bvh, nodes: bvhNodes} = getSegmentBVHNodes(graph, {
      id,
      operation: OPERATION,
      minima,
      maxima,
      leafCapacity,
      spatialSort
    });
    nodes.push(...bvhNodes);

    // Feature row of every vertex, or SEGMENT_NONE for vertices that belong to no usable ring.
    const vertexFeatures = createTransientView(
      graph,
      `${id}-vertex-features`,
      'uint32',
      vertexCount
    );
    const isPolygons = geometry.kind === 'polygons';
    const featureBindings: WGSLKernelBinding[] = isPolygons
      ? [
          {name: 'ringOffsets', view: geometry.ringOffsets, type: 'u32', access: 'read'},
          {name: 'polygonOffsets', view: geometry.polygonOffsets, type: 'u32', access: 'read'},
          {name: 'featureOffsets', view: geometry.featureOffsets, type: 'u32', access: 'read'}
        ]
      : [{name: 'lineOffsets', view: geometry.lineOffsets, type: 'u32', access: 'read'}];
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-vertex-features`,
        operation: OPERATION,
        variant: 'vertex-features',
        bindings: [
          ...featureBindings,
          {name: 'vertexFeatures', view: vertexFeatures, type: 'u32', access: 'read_write'}
        ],
        invocationCount: vertexCount,
        declarations: `const SEGMENT_NONE: u32 = ${GPU_SEGMENT_NONE}u;
const VERTEX_COUNT: u32 = ${vertexCount}u;
${
  isPolygons
    ? `const RING_COUNT: u32 = ${geometry.ringOffsets.length - 1}u;
const POLYGON_COUNT: u32 = ${geometry.polygonOffsets.length - 1}u;
const FEATURE_COUNT: u32 = ${geometry.featureOffsets.length - 1}u;
fn readOffset(kind: u32, row: u32) -> u32 {
  if (kind == 0u) { return ringOffsets[ringOffsetsOffset + row]; }
  if (kind == 1u) { return polygonOffsets[polygonOffsetsOffset + row]; }
  return featureOffsets[featureOffsetsOffset + row];
}`
    : `const RING_COUNT: u32 = ${geometry.lineOffsets.length - 1}u;
fn readOffset(kind: u32, row: u32) -> u32 { return lineOffsets[lineOffsetsOffset + row]; }`
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
}`,
        body: `var feature = SEGMENT_NONE;
  let ring = lastAtMost(0u, RING_COUNT, index);
  if (ring != SEGMENT_NONE) {
    let ringStart = readOffset(0u, ring);
    let ringEnd = min(readOffset(0u, ring + 1u), VERTEX_COUNT);
    if (index < ringEnd) {
      ${
        isPolygons
          ? `if (ringEnd - ringStart >= 3u && ring < readOffset(1u, POLYGON_COUNT)) {
        let polygon = lastAtMost(1u, POLYGON_COUNT, ring);
        if (polygon != SEGMENT_NONE && polygon < readOffset(2u, FEATURE_COUNT)) {
          feature = lastAtMost(2u, FEATURE_COUNT, polygon);
        }
      }`
          : 'feature = ring;'
      }
    }
  }
  vertexFeatures[vertexFeaturesOffset + index] = feature;`
      })
    );

    // Nearest partner of every vertex: (distance, closest x, closest y, unused).
    const nearest = createTransientView(graph, `${id}-vertex-nearest`, 'float32x4', vertexCount);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-probe`,
        operation: OPERATION,
        variant: 'probe',
        bindings: [
          {name: 'positions', view: geometry.positions, type: 'f32', access: 'read'},
          {name: 'rightTable', view: table, type: 'u32', access: 'read'},
          {name: 'nodeMinima', view: bvh.nodeMinima, type: 'f32', access: 'read'},
          {name: 'nodeMaxima', view: bvh.nodeMaxima, type: 'f32', access: 'read'},
          {name: 'leafIds', view: bvh.leafIds, type: 'u32', access: 'read'},
          {name: 'vertexFeatures', view: vertexFeatures, type: 'u32', access: 'read'},
          {name: 'nearest', view: nearest, type: 'f32', access: 'read_write'}
        ],
        invocationCount: vertexCount,
        declarations: `const SEGMENT_NONE: u32 = ${GPU_SEGMENT_NONE}u;
${IS_NAN_BITS_WGSL}
const FLOAT32_MAXIMUM: f32 = 3.402823466e+38;
const VERTEX_COUNT: u32 = ${vertexCount}u;
const INTERNAL_NODE_COUNT: u32 = ${bvh.internalNodeCount}u;
${getSegmentTableAccessorsWGSL('right')}
struct Candidate {
  distanceSquared: f32,
  point: vec2f,
}
// Squared distance from point to the segment (a, b) with a != b, and the closest point.
fn segmentCandidate(point: vec2f, a: vec2f, b: vec2f) -> Candidate {
  let edge = b - a;
  let offset = point - a;
  let lengthSquared = dot(edge, edge);
  let along = dot(offset, edge);
  if (along <= 0.0) { return Candidate(dot(offset, offset), a); }
  if (along >= lengthSquared) {
    let toEnd = point - b;
    return Candidate(dot(toEnd, toEnd), b);
  }
  let side = edge.x * offset.y - edge.y * offset.x;
  return Candidate(side * side / lengthSquared, a + edge * (along / lengthSquared));
}
// Candidate of one segment row for a vertex: the segment itself, or its other endpoint when the
// vertex is one of its endpoints.
fn rowCandidate(point: vec2f, row: u32) -> Candidate {
  let start = rightStart(row);
  let end = rightEnd(row);
  if (point.x == start.x && point.y == start.y) {
    let toEnd = point - end;
    return Candidate(dot(toEnd, toEnd), end);
  }
  if (point.x == end.x && point.y == end.y) {
    let toStart = point - start;
    return Candidate(dot(toStart, toStart), start);
  }
  return segmentCandidate(point, start, end);
}
fn boxDistanceSquared(node: u32, point: vec2f) -> f32 {
  let component = node * 2u;
  let minimum = vec2f(nodeMinima[nodeMinimaOffset + component], nodeMinima[nodeMinimaOffset + component + 1u]);
  let maximum = vec2f(nodeMaxima[nodeMaximaOffset + component], nodeMaxima[nodeMaximaOffset + component + 1u]);
  let gap = max(max(minimum - point, point - maximum), vec2f(0.0));
  return dot(gap, gap);
}`,
        body: `let feature = vertexFeatures[vertexFeaturesOffset + index];
  var bestSquared = FLOAT32_MAXIMUM;
  var bestPoint = vec2f(0.0);
  var bestRow = SEGMENT_NONE;
  let point = vec2f(positions[positionsOffset + index * 2u], positions[positionsOffset + index * 2u + 1u]);
  if (feature != SEGMENT_NONE && !isNanBits(point.x) && !isNanBits(point.y)) {
    // Seed the bound with this vertex's own segment and the previous slot (usually the other
    // incident segment). Any valid segment of the same feature is a legitimate candidate, so the
    // result is unchanged; the bound just starts at about the shortest incident edge, which
    // prunes almost the whole tree (including other features' overlapping nodes) up front.
    for (var seed = 0u; seed < 2u; seed++) {
      if (seed == 1u && index == 0u) { break; }
      let seedRow = index - seed;
      if (!rightValid(seedRow) || rightFeature(seedRow) != feature) { continue; }
      let seedCandidate = rowCandidate(point, seedRow);
      if (seedCandidate.distanceSquared < bestSquared ||
          (seedCandidate.distanceSquared == bestSquared && seedRow < bestRow)) {
        bestSquared = seedCandidate.distanceSquared;
        bestPoint = seedCandidate.point;
        bestRow = seedRow;
      }
    }
    // Stack entries carry their box distance, tested when pushed and again when popped.
    var stack: array<u32, 64>;
    var stackLower: array<f32, 64>;
    var depth = 1u;
    stack[0] = 0u;
    stackLower[0] = boxDistanceSquared(0u, point);
    loop {
      if (depth == 0u) { break; }
      depth = depth - 1u;
      let node = stack[depth];
      if (stackLower[depth] > bestSquared) { continue; }
      if (node < INTERNAL_NODE_COUNT) {
        let first = node * 2u + 1u;
        let second = first + 1u;
        let firstLower = boxDistanceSquared(first, point);
        let secondLower = boxDistanceSquared(second, point);
        // Push the farther child first so the nearer one is popped next.
        if (firstLower <= secondLower) {
          if (secondLower <= bestSquared) { stack[depth] = second; stackLower[depth] = secondLower; depth = depth + 1u; }
          if (firstLower <= bestSquared) { stack[depth] = first; stackLower[depth] = firstLower; depth = depth + 1u; }
        } else {
          if (firstLower <= bestSquared) { stack[depth] = first; stackLower[depth] = firstLower; depth = depth + 1u; }
          if (secondLower <= bestSquared) { stack[depth] = second; stackLower[depth] = secondLower; depth = depth + 1u; }
        }
        continue;
      }
      let row = leafIds[leafIdsOffset + node - INTERNAL_NODE_COUNT];
      if (row >= VERTEX_COUNT || !rightValid(row) || rightFeature(row) != feature) { continue; }
      let candidate = rowCandidate(point, row);
      if (candidate.distanceSquared < bestSquared ||
          (candidate.distanceSquared == bestSquared && row < bestRow)) {
        bestSquared = candidate.distanceSquared;
        bestPoint = candidate.point;
        bestRow = row;
      }
    }
  }
  // A runtime zero keeps the bit pattern of infinity from being rejected as a constant.
  let infinity = bitcast<f32>(0x7f800000u | (index & 0u));
  let found = bestRow != SEGMENT_NONE;
  let base = nearestOffset + index * 4u;
  nearest[base] = select(infinity, sqrt(bestSquared), found);
  nearest[base + 1u] = bestPoint.x;
  nearest[base + 2u] = bestPoint.y;
  nearest[base + 3u] = 0.0;`
      })
    );

    // Per feature: [minimum distance bits, lowest vertex row reaching them]. Library rule 14: one
    // workgroup per feature reduces the feature's contiguous vertex range with a fixed binary tree,
    // instead of per-vertex atomics onto two words per feature (contention on large features).
    const featureState = createTransientView(
      graph,
      `${id}-feature-state`,
      'uint32',
      featureCount * 2
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-feature-minimum`,
        operation: OPERATION,
        variant: `feature-minimum-${geometry.kind}`,
        bindings: [
          ...featureBindings,
          {name: 'vertexFeatures', view: vertexFeatures, type: 'u32', access: 'read'},
          {name: 'nearest', view: nearest, type: 'f32', access: 'read'},
          {name: 'featureState', view: featureState, type: 'u32', access: 'read_write'}
        ],
        workgroupSize: FEATURE_REDUCTION_WORKGROUP_SIZE,
        invocationCount: featureCount * FEATURE_REDUCTION_WORKGROUP_SIZE,
        guardIndex: false,
        declarations: `const SEGMENT_NONE: u32 = ${GPU_SEGMENT_NONE}u;
const INFINITY_BITS: u32 = 0x7f800000u;
const VERTEX_COUNT: u32 = ${vertexCount}u;
const LANE_COUNT: u32 = ${FEATURE_REDUCTION_WORKGROUP_SIZE}u;
${
  isPolygons
    ? `const RING_COUNT: u32 = ${geometry.ringOffsets.length - 1}u;
const POLYGON_COUNT: u32 = ${geometry.polygonOffsets.length - 1}u;
// First vertex row of a feature row (the feature count gives the end), clamped like the offsets.
fn getFeatureStart(row: u32) -> u32 {
  let polygon = min(featureOffsets[featureOffsetsOffset + row], POLYGON_COUNT);
  let ring = min(polygonOffsets[polygonOffsetsOffset + polygon], RING_COUNT);
  return min(ringOffsets[ringOffsetsOffset + ring], VERTEX_COUNT);
}`
    : `fn getFeatureStart(row: u32) -> u32 {
  return min(lineOffsets[lineOffsetsOffset + row], VERTEX_COUNT);
}`
}
var<workgroup> laneBits: array<u32, ${FEATURE_REDUCTION_WORKGROUP_SIZE}>;
var<workgroup> laneRows: array<u32, ${FEATURE_REDUCTION_WORKGROUP_SIZE}>;
// Lexicographic (distance bits, vertex row) order: non-negative f32 values order like their bits.
fn isBefore(bits: u32, row: u32, otherBits: u32, otherRow: u32) -> bool {
  return bits < otherBits || (bits == otherBits && row < otherRow);
}`,
        // No early return: every invocation of a workgroup must reach the barriers.
        body: `let feature = index / LANE_COUNT;
  let isInRange = index < INVOCATION_COUNT;
  var bestBits = 0xffffffffu;
  var bestRow = 0xffffffffu;
  if (isInRange) {
    let begin = getFeatureStart(feature);
    let end = getFeatureStart(feature + 1u);
    for (var vertex = begin + localInvocationIndex; vertex < end; vertex += LANE_COUNT) {
      let bits = bitcast<u32>(nearest[nearestOffset + vertex * 4u]);
      if (vertexFeatures[vertexFeaturesOffset + vertex] == feature && bits < INFINITY_BITS &&
          isBefore(bits, vertex, bestBits, bestRow)) {
        bestBits = bits;
        bestRow = vertex;
      }
    }
  }
  laneBits[localInvocationIndex] = bestBits;
  laneRows[localInvocationIndex] = bestRow;
  workgroupBarrier();
  for (var stride = LANE_COUNT / 2u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      let otherBits = laneBits[localInvocationIndex + stride];
      let otherRow = laneRows[localInvocationIndex + stride];
      if (isBefore(otherBits, otherRow, laneBits[localInvocationIndex], laneRows[localInvocationIndex])) {
        laneBits[localInvocationIndex] = otherBits;
        laneRows[localInvocationIndex] = otherRow;
      }
    }
    workgroupBarrier();
  }
  if (isInRange && localInvocationIndex == 0u) {
    featureState[featureStateOffset + feature * 2u] = laneBits[0];
    featureState[featureStateOffset + feature * 2u + 1u] = laneRows[0];
  }`
      })
    );

    const outputBindings: WGSLKernelBinding[] = [
      {name: 'positions', view: geometry.positions, type: 'f32', access: 'read'},
      {name: 'nearest', view: nearest, type: 'f32', access: 'read'},
      {name: 'featureState', view: featureState, type: 'u32', access: 'read'}
    ];
    const writes: string[] = [];
    if (clearances) {
      outputBindings.push({
        name: 'clearances',
        view: clearances,
        type: 'f32',
        access: 'read_write'
      });
      writes.push(
        'clearances[clearancesOffset + index] = select(infinity, nearest[nearestOffset + row * 4u], found);'
      );
    }
    if (lines) {
      outputBindings.push({name: 'lines', view: lines, type: 'f32', access: 'read_write'});
      writes.push(`let lineBase = linesOffset + index * 4u;
  lines[lineBase] = select(notANumber, positions[positionsOffset + row * 2u], found);
  lines[lineBase + 1u] = select(notANumber, positions[positionsOffset + row * 2u + 1u], found);
  lines[lineBase + 2u] = select(notANumber, nearest[nearestOffset + row * 4u + 1u], found);
  lines[lineBase + 3u] = select(notANumber, nearest[nearestOffset + row * 4u + 2u], found);`);
    }
    if (vertexIds) {
      outputBindings.push({name: 'vertexIds', view: vertexIds, type: 'u32', access: 'read_write'});
      writes.push('vertexIds[vertexIdsOffset + index] = rowWord;');
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-write`,
        operation: OPERATION,
        variant: 'write',
        bindings: outputBindings,
        invocationCount: featureCount,
        declarations: 'const SEGMENT_NONE: u32 = 0xffffffffu;',
        body: `let rowWord = featureState[featureStateOffset + index * 2u + 1u];
  let found = rowWord != SEGMENT_NONE;
  let row = select(0u, rowWord, found);
  // Runtime zeros keep the bit patterns of infinity and NaN from being rejected as constants.
  let infinity = bitcast<f32>(0x7f800000u | (index & 0u));
  let notANumber = bitcast<f32>(0x7fc00000u | (index & 0u));
  ${writes.join('\n  ')}`
      })
    );
    return nodes;
  }
}
