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
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {getSegmentBVHNodes} from '../segment-intersection/segment-bvh';
import {createSegmentTableNode} from '../segment-intersection/segment-table';
import {
  getSegmentTableAccessorsWGSL,
  SEGMENT_TABLE_STRIDE
} from '../segment-intersection/segment-intersection-wgsl';
import type {GPUSegmentGeometry} from '../segment-intersection/segment-intersection-types';
import {GPU_SEGMENT_NONE} from '../segment-intersection/segment-intersection-types';

const OPERATION = 'GPUMinimumClearance';

/** Vertex count from which `spatialSort` defaults to on. */
const SPATIAL_SORT_MINIMUM_SLOTS = 1024;

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
   * Linestring or polygon features in the layouts of `GPUSpatialPredicateJoin`. Polygon rings may
   * be explicitly closed or not; rings with fewer than three vertices are ignored.
   */
  geometry: GPUSegmentGeometry;
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
  /** Power-of-two BVH leaf slots over the segments. Defaults to the next power of two of the vertex count. */
  leafCapacity?: number;
  /**
   * Compile-time. Morton-sorts the segments before the BVH build, which keeps traversal short for
   * sources whose vertex order jumps around the plane. The result is identical either way.
   * Defaults to `true` from 1024 vertices on.
   */
  spatialSort?: boolean;
};

function getNextPowerOfTwo(value: number): number {
  let result = 1;
  while (result < value) {
    result *= 2;
  }
  return result;
}

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
 * vertex is one of its endpoints), then a fixed-order per-feature minimum with two integer
 * `atomicMin` passes: the distance bits, then the lowest vertex row reaching them. Non-negative
 * f32 values order like their bit patterns, so the result is deterministic and ties go to the
 * lowest vertex row. Nothing is read back.
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
 */
export class GPUMinimumClearance implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUMinimumClearanceProps;
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
      validatePackedUint32View(view, `${id} geometry.${name}`);
      if (view.length < 1) {
        throw new Error(`${id} geometry.${name} requires a terminal entry`);
      }
    }
    this.vertexCount = geometry.positions.length;
    if (this.vertexCount < 1) {
      throw new Error(`${id} geometry must contain at least one vertex`);
    }
    if (this.vertexCount >= 0x80000000) {
      throw new Error(`${id} geometry has too many vertices`);
    }
    this.featureCount =
      (geometry.kind === 'lines' ? geometry.lineOffsets.length : geometry.featureOffsets.length) -
      1;
    if (this.featureCount < 1) {
      throw new Error(`${id} geometry must contain at least one feature`);
    }
    if (!props.clearances && !props.lines && !props.vertexIds) {
      throw new Error(`${id} requires at least one of clearances, lines or vertexIds`);
    }
    if (props.clearances) {
      validatePackedView(props.clearances, ['float32'], `${id} clearances`);
    }
    if (props.lines) {
      validatePackedView(props.lines, ['float32x4'], `${id} lines`);
    }
    if (props.vertexIds) {
      validatePackedUint32View(props.vertexIds, `${id} vertexIds`);
    }
    for (const [name, view] of [
      ['clearances', props.clearances],
      ['lines', props.lines],
      ['vertexIds', props.vertexIds]
    ] as const) {
      if (view && view.length !== this.featureCount) {
        throw new Error(`${id} ${name} length must equal the feature count`);
      }
    }
    this.spatialSort = props.spatialSort ?? this.vertexCount >= SPATIAL_SORT_MINIMUM_SLOTS;
    this.leafCapacity = props.leafCapacity ?? getNextPowerOfTwo(this.vertexCount);
    if (
      !Number.isSafeInteger(this.leafCapacity) ||
      this.leafCapacity < 1 ||
      (this.leafCapacity & (this.leafCapacity - 1)) !== 0
    ) {
      throw new Error(`${id} leafCapacity must be a positive power of two`);
    }
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
    const {clearances, lines, vertexIds} = this.props;
    return [clearances, lines, vertexIds];
  }

  /** Returns table, BVH, vertex feature, probe, per-feature minimum and output nodes in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, featureCount, vertexCount, leafCapacity, spatialSort} = this;
    const {geometry, clearances, lines, vertexIds} = props;
    validateGraphViewsBelongToGraph(id, graph, [...this.getInputViews(), ...this.getOutputViews()]);
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
  if (feature != SEGMENT_NONE && point.x == point.x && point.y == point.y) {
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

    // Per feature: [minimum distance bits, lowest vertex row reaching them].
    const featureState = createTransientView(
      graph,
      `${id}-feature-state`,
      'uint32',
      featureCount * 2
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-clear`,
        operation: OPERATION,
        variant: 'clear',
        bindings: [{name: 'featureState', view: featureState, type: 'u32', access: 'read_write'}],
        invocationCount: featureCount * 2,
        body: 'featureState[featureStateOffset + index] = 0xffffffffu;'
      })
    );
    const reduceDeclarations = `const SEGMENT_NONE: u32 = ${GPU_SEGMENT_NONE}u;
const INFINITY_BITS: u32 = 0x7f800000u;
fn distanceBits(vertex: u32) -> u32 { return bitcast<u32>(nearest[nearestOffset + vertex * 4u]); }`;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-minimum-distance`,
        operation: OPERATION,
        variant: 'minimum-distance',
        bindings: [
          {name: 'vertexFeatures', view: vertexFeatures, type: 'u32', access: 'read'},
          {name: 'nearest', view: nearest, type: 'f32', access: 'read'},
          {name: 'featureState', view: featureState, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: vertexCount,
        declarations: reduceDeclarations,
        body: `let feature = vertexFeatures[vertexFeaturesOffset + index];
  let bits = distanceBits(index);
  if (feature != SEGMENT_NONE && bits < INFINITY_BITS) {
    atomicMin(&featureState[featureStateOffset + feature * 2u], bits);
  }`
      })
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-minimum-row`,
        operation: OPERATION,
        variant: 'minimum-row',
        bindings: [
          {name: 'vertexFeatures', view: vertexFeatures, type: 'u32', access: 'read'},
          {name: 'nearest', view: nearest, type: 'f32', access: 'read'},
          {name: 'featureState', view: featureState, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: vertexCount,
        declarations: reduceDeclarations,
        body: `let feature = vertexFeatures[vertexFeaturesOffset + index];
  let bits = distanceBits(index);
  if (feature != SEGMENT_NONE && bits < INFINITY_BITS &&
      bits == atomicLoad(&featureState[featureStateOffset + feature * 2u])) {
    atomicMin(&featureState[featureStateOffset + feature * 2u + 1u], index);
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
