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
import {
  createFillNode,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {getSegmentBVHNodes} from '../segment-intersection/segment-bvh';
import {createSegmentTableNode} from '../segment-intersection/segment-table';
import {
  getSegmentTableAccessorsWGSL,
  SEGMENT_PREDICATES_WGSL
} from '../segment-intersection/segment-intersection-wgsl';
import {createRingOrientationNode} from './coverage-topology-kernels';
import {GPU_COVERAGE_VALIDITY_PARAMETER_LENGTH} from './coverage-validity-parameters';

const OPERATION = 'GPUCoverageValidity';
/** Vertex count from which `spatialSort` defaults to on, as `GPUSegmentIntersection`. */
const SPATIAL_SORT_MINIMUM_SLOTS = 1024;

/**
 * Bits of `output.segmentFlags` of {@link GPUCoverageValidity}. A segment is valid when its flags
 * are `0`.
 */
export const GPU_COVERAGE_VALIDITY_FLAG = {
  /**
   * The segment overlaps a segment of another polygon along a span that is not identical (the two
   * polygons disagree about the vertices of a shared boundary), or an endpoint of one lies in the
   * interior of the other (a T-junction without a matching vertex).
   */
  unmatched: 1,
  /** The segment properly crosses a segment of another polygon. */
  crossing: 2,
  /**
   * The segment runs nearly parallel to a segment of another polygon, within `gapWidth` of it, so
   * the two polygons are separated by a narrow gap.
   */
  gap: 4,
  /**
   * The segment lies inside another polygon without touching its boundary, or equals a segment
   * of another polygon whose interior is on the same side (a duplicated or overlapping polygon).
   */
  overlap: 8,
  /** An exact orientation test could not be certified for a candidate pair (non-finite input). */
  uncertain: 16
} as const;

/** Caller-owned outputs of {@link GPUCoverageValidity}. */
export type GPUCoverageValidityOutput = {
  /**
   * `GPU_COVERAGE_VALIDITY_FLAG` bits for the segment that starts at each input vertex (a ring edge
   * `i -> next(i)`, one row per input vertex, so rows align with `positions`). Rows of rings with
   * fewer than three vertices and of degenerate or non-finite segments are `0`.
   */
  segmentFlags: GraphDataView<'uint32'>;
  /** Optional invalid segment count per polygon, `polygonOffsets.length - 1` rows. */
  polygonInvalidCounts?: GraphDataView<'uint32'>;
  /** Optional one-row total of invalid segments. */
  invalidSegmentCount?: GraphDataView<'uint32'>;
  /** Optional one-row flag: 1 when no segment is invalid (Shapely `coverage_is_valid`), else 0. */
  isValid?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUCoverageValidity}.
 *
 * Per-frame (no recompile): the contents of `parameters` (the gap width) and of the input buffers
 * as long as lengths stay the same. Compile-time: view lengths, `spatialSort`, `leafCapacity` and
 * which optional outputs are present.
 */
export type GPUCoverageValidityProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'coverage-validity'`. */
  id?: string;
  /** Flattened ring vertices (GeoArrow layout, as `GPUCoverageSimplification`). Finite coordinates. */
  positions: GraphDataView<'float32x2'>;
  /** Ring-to-vertex offsets with a terminal entry. Rings close implicitly. */
  ringOffsets: GraphDataView<'uint32'>;
  /**
   * Polygon-to-ring offsets with a terminal entry. The first ring of a polygon is its shell, the
   * others are holes. Winding is free: orientation is read from the signed area.
   */
  polygonOffsets: GraphDataView<'uint32'>;
  /**
   * Per-frame packed float32 view of at least {@link GPU_COVERAGE_VALIDITY_PARAMETER_LENGTH}
   * elements written with `getGPUCoverageValidityParameterValues`: the gap width.
   */
  parameters: GraphDataView<'float32'>;
  /**
   * Compile-time. Reorders segments along a Morton curve before the BVH build; identical results,
   * different cost, as `GPUSegmentIntersection.spatialSort`. Defaults to `true` from 1024 vertices.
   */
  spatialSort?: boolean;
  /** Power-of-two BVH leaf slots. Defaults to the next power of two of the vertex count. */
  leafCapacity?: number;
  /** Caller-owned outputs. */
  output: GPUCoverageValidityOutput;
};

function getNextPowerOfTwo(value: number): number {
  let result = 1;
  while (result < value) {
    result *= 2;
  }
  return result;
}

/**
 * Validates a polygon coverage: Shapely `coverage_is_valid` and `coverage_invalid_edges`
 * (GEOS `CoverageValidator`), reported per ring segment.
 *
 * **Definition.** Every polygon is a member of one coverage (the polygon rows of the input). A
 * segment of a polygon is checked against the segments of every *other* polygon whose bounds are
 * within `gapWidth`:
 * - an identical segment (either direction) is a *matched* edge. It is valid unless both polygons
 *   have their interior on the same side of it (duplicate or overlapping polygons): `overlap`. A
 *   matched edge is not tested against any other segment, as in GEOS;
 * - a proper crossing is `crossing`;
 * - a collinear overlap of different segments, or a touch where an endpoint of either segment lies
 *   in the interior of the other (a T-junction without a matching vertex), is `unmatched`;
 * - when `gapWidth > 0`, two segments that neither cross nor touch in the ways above are `gap` when
 *   they run side by side: both endpoints of the shorter lie within `gapWidth` of the longer, and
 *   each projects onto the other over a length greater than `gapWidth`. Every other pair, such as
 *   a short segment next to a long one, a vertex pointing at an edge or edges meeting at a vertex,
 *   is valid;
 * - an otherwise clean, unmatched segment whose midpoint lies inside another polygon (an island or
 *   spike inside a neighbor) is `overlap`, found with a winding number over the other polygons'
 *   segments.
 *
 * Segments sharing only a vertex are valid. Segments of the same polygon are never compared with
 * each other: self-intersections within one polygon are the business of `GPUGeometryValidity`.
 * Orientation is normalized from the signed area, so mixed clockwise and counter-clockwise rings
 * are fine.
 *
 * **Parity with GEOS.** The gap rule was fitted against `shapely.coverage_invalid_edges` (GEOS
 * `CoverageValidator`), whose exact rule is not documented: the invalid segment sets agree on the
 * curated scenes of the specs and on 158 of 160 random jittered-triangulation cases at gap widths
 * from 0 to 0.75 (all 40 at gap width 0). The two differing cases are a segment next to a vertex of
 * an overlapping neighbor within `gapWidth`, which GEOS flags and this contributor does not. The
 * interior test uses the f32 midpoint of a segment, so a matched edge lying inside a *third*
 * polygon is not flagged (that polygon's own segments are).
 *
 * **Cost.** One BVH built over all segments; each segment probes it with its bounds inflated by
 * `gapWidth`. Output is one `uint32` row per vertex: nothing is capacity-bounded, nothing
 * overflows. Counts use integer atomics, so results are deterministic.
 */
export class GPUCoverageValidity implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUCoverageValidityProps;
  /** Resolved BVH leaf capacity. */
  readonly leafCapacity: number;
  /** Whether segments are Morton sorted before the BVH build. */
  readonly spatialSort: boolean;

  constructor(props: GPUCoverageValidityProps) {
    const id = props.id ?? 'coverage-validity';
    this.id = id;
    this.props = props;
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    if (props.positions.length < 1 || props.positions.length >= 2 ** 30) {
      throw new Error(`${id} positions must hold between 1 and 2^30 - 1 vertices`);
    }
    validatePackedUint32View(props.ringOffsets, `${id} ringOffsets`);
    validatePackedUint32View(props.polygonOffsets, `${id} polygonOffsets`);
    if (props.ringOffsets.length < 2) {
      throw new Error(`${id} ringOffsets must hold at least two entries`);
    }
    if (props.polygonOffsets.length < 2) {
      throw new Error(`${id} polygonOffsets must hold at least two entries`);
    }
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_COVERAGE_VALIDITY_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_COVERAGE_VALIDITY_PARAMETER_LENGTH} float32 values`
      );
    }
    this.spatialSort = props.spatialSort ?? props.positions.length >= SPATIAL_SORT_MINIMUM_SLOTS;
    this.leafCapacity = props.leafCapacity ?? getNextPowerOfTwo(props.positions.length);
    if (
      !Number.isSafeInteger(this.leafCapacity) ||
      this.leafCapacity < props.positions.length ||
      (this.leafCapacity & (this.leafCapacity - 1)) !== 0
    ) {
      throw new Error(`${id} leafCapacity must be a power of two of at least the vertex count`);
    }
    const {output} = props;
    validatePackedUint32View(output.segmentFlags, `${id} output.segmentFlags`);
    if (output.segmentFlags.length !== props.positions.length) {
      throw new Error(`${id} output.segmentFlags length must equal positions length`);
    }
    if (output.polygonInvalidCounts) {
      validatePackedUint32View(output.polygonInvalidCounts, `${id} output.polygonInvalidCounts`);
      if (output.polygonInvalidCounts.length !== props.polygonOffsets.length - 1) {
        throw new Error(`${id} output.polygonInvalidCounts length must equal the polygon count`);
      }
    }
    for (const [name, view] of [
      ['invalidSegmentCount', output.invalidSegmentCount],
      ['isValid', output.isValid]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} output.${name}`);
        if (view.length < 1) {
          throw new Error(`${id} output.${name} must hold one uint32`);
        }
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        output.segmentFlags,
        output.polygonInvalidCounts,
        output.invalidSegmentCount,
        output.isValid
      ],
      [props.positions, props.ringOffsets, props.polygonOffsets, props.parameters]
    );
  }

  /** Returns the coverage-validity nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {positions, ringOffsets, polygonOffsets, parameters, output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      positions,
      ringOffsets,
      polygonOffsets,
      parameters,
      output.segmentFlags,
      output.polygonInvalidCounts,
      output.invalidSegmentCount,
      output.isValid
    ]);
    const vertexCount = positions.length;
    const ringCount = ringOffsets.length - 1;
    const polygonCount = polygonOffsets.length - 1;
    const view = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', length);

    const ringFlip = view('ring-flip', ringCount);
    const featureOffsets = view('feature-offsets', polygonCount + 1);
    const table = view('segment-table', vertexCount * 8);
    const minima = createTransientView(graph, `${id}-bounds-minima`, 'float32x2', vertexCount);
    const maxima = createTransientView(graph, `${id}-bounds-maxima`, 'float32x2', vertexCount);
    const nodes: GPUCommandNode<Parameters>[] = [
      createRingOrientationNode<Parameters>(graph, {
        id: `${id}-ring-orientation`,
        operation: OPERATION,
        positions,
        ringOffsets,
        polygonOffsets,
        ringFlip
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-feature-offsets`,
        operation: OPERATION,
        variant: 'feature-offsets',
        bindings: [
          {name: 'featureOffsets', view: featureOffsets, type: 'u32', access: 'read_write'}
        ],
        invocationCount: polygonCount + 1,
        body: 'featureOffsets[featureOffsetsOffset + index] = index;'
      })
    ];
    nodes.push(
      createSegmentTableNode<Parameters>(graph, {
        id: `${id}-segment-table`,
        operation: OPERATION,
        geometry: {kind: 'polygons', positions, ringOffsets, polygonOffsets, featureOffsets},
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
      leafCapacity: this.leafCapacity,
      spatialSort: this.spatialSort
    });
    nodes.push(...bvhNodes);

    const traverse = (queryMinimum: string, queryMaximum: string, leafBody: string) => `{
    let queryMinimum = ${queryMinimum};
    let queryMaximum = ${queryMaximum};
    var node = 0u;
    loop {
      if (nodeOverlaps(node, queryMinimum, queryMaximum)) {
        if (node < INTERNAL_NODE_COUNT) {
          node = node * 2u + 1u;
          continue;
        }
        let other = leafIds[leafIdsOffset + node - INTERNAL_NODE_COUNT];
        if (other < SEGMENT_COUNT && segmentValid(other) && segmentFeature(other) != polygon) {
          ${leafBody}
        }
      }
      loop {
        if (node == 0u || (node & 1u) == 1u) { break; }
        node = (node - 1u) / 2u;
      }
      if (node == 0u) { break; }
      node = node + 1u;
    }
  }`;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-probe`,
        operation: OPERATION,
        variant: 'probe',
        bindings: [
          {name: 'segmentTable', view: table, type: 'u32', access: 'read'},
          {name: 'nodeMinima', view: bvh.nodeMinima, type: 'f32', access: 'read'},
          {name: 'nodeMaxima', view: bvh.nodeMaxima, type: 'f32', access: 'read'},
          {name: 'leafIds', view: bvh.leafIds, type: 'u32', access: 'read'},
          {name: 'ringFlip', view: ringFlip, type: 'u32', access: 'read'},
          {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
          {name: 'segmentFlags', view: output.segmentFlags, type: 'u32', access: 'read_write'}
        ],
        invocationCount: vertexCount,
        declarations: `${SEGMENT_PREDICATES_WGSL}
${getSegmentTableAccessorsWGSL('segment')}
const SEGMENT_COUNT: u32 = ${vertexCount}u;
const INTERNAL_NODE_COUNT: u32 = ${bvh.internalNodeCount}u;
const FLAG_UNMATCHED: u32 = ${GPU_COVERAGE_VALIDITY_FLAG.unmatched}u;
const FLAG_CROSSING: u32 = ${GPU_COVERAGE_VALIDITY_FLAG.crossing}u;
const FLAG_GAP: u32 = ${GPU_COVERAGE_VALIDITY_FLAG.gap}u;
const FLAG_OVERLAP: u32 = ${GPU_COVERAGE_VALIDITY_FLAG.overlap}u;
const FLAG_UNCERTAIN: u32 = ${GPU_COVERAGE_VALIDITY_FLAG.uncertain}u;
const MATCHED_BIT: u32 = 0x80000000u;
const FLOAT32_MAXIMUM: f32 = 3.402823466e+38;

fn nodeOverlaps(node: u32, queryMinimum: vec2f, queryMaximum: vec2f) -> bool {
  let component = node * 2u;
  let minimum = vec2f(nodeMinima[nodeMinimaOffset + component], nodeMinima[nodeMinimaOffset + component + 1u]);
  let maximum = vec2f(nodeMaxima[nodeMaximaOffset + component], nodeMaxima[nodeMaximaOffset + component + 1u]);
  return all(minimum <= queryMaximum) && all(queryMinimum <= maximum);
}

fn isFlipped(row: u32) -> bool {
  return ringFlip[ringFlipOffset + segmentRing(row)] != 0u;
}

fn samePoint(first: vec2f, second: vec2f) -> bool {
  return first.x == second.x && first.y == second.y;
}

fn distanceToSegment(point: vec2f, start: vec2f, end: vec2f) -> f32 {
  let direction = end - start;
  let lengthSquared = dot(direction, direction);
  let fraction = select(clamp(dot(point - start, direction) / lengthSquared, 0.0, 1.0), 0.0, lengthSquared == 0.0);
  return length(point - (start + direction * fraction));
}

// Length of the part of segment (start, end) that projects onto segment (otherStart, otherEnd), times
// the length of the latter, so that no division is needed and ties with gapWidth stay exact.
fn projectedLengthScaled(start: vec2f, end: vec2f, otherStart: vec2f, otherEnd: vec2f) -> f32 {
  let direction = otherEnd - otherStart;
  let lengthSquared = dot(direction, direction);
  let first = dot(start - otherStart, direction);
  let second = dot(end - otherStart, direction);
  return min(max(first, second), lengthSquared) - max(min(first, second), 0.0);
}

// The segments run side by side: both endpoints of the shorter lie within gapWidth of the longer,
// and each projects onto the other over more than gapWidth.
fn isNearlyParallel(a: vec2f, b: vec2f, c: vec2f, d: vec2f, gapWidth: f32) -> bool {
  let aLongest = distance(a, b) >= distance(c, d);
  let longStart = select(c, a, aLongest);
  let longEnd = select(d, b, aLongest);
  let shortStart = select(a, c, aLongest);
  let shortEnd = select(b, d, aLongest);
  if (distanceToSegment(shortStart, longStart, longEnd) > gapWidth ||
      distanceToSegment(shortEnd, longStart, longEnd) > gapWidth) {
    return false;
  }
  let shortOnLong = projectedLengthScaled(shortStart, shortEnd, longStart, longEnd);
  let longOnShort = projectedLengthScaled(longStart, longEnd, shortStart, shortEnd);
  return shortOnLong > gapWidth * distance(longStart, longEnd) &&
    longOnShort > gapWidth * distance(shortStart, shortEnd);
}

// Flags of segment row against segment other (another polygon). Bit 31 reports a matched edge.
fn comparePair(row: u32, other: u32, gapWidth: f32) -> u32 {
  let a = segmentStart(row);
  let b = segmentEnd(row);
  let c = segmentStart(other);
  let d = segmentEnd(other);
  let sameDirection = samePoint(a, c) && samePoint(b, d);
  if (sameDirection || (samePoint(a, d) && samePoint(b, c))) {
    let effectivelySame = sameDirection != (isFlipped(row) != isFlipped(other));
    return select(MATCHED_BIT, FLAG_OVERLAP, effectivelySame);
  }
  let hit = classifySegments(a, b, c, d, false);
  if (hit.kind == KIND_UNCERTAIN) { return FLAG_UNCERTAIN; }
  if (hit.kind == KIND_PROPER) { return FLAG_CROSSING; }
  if (hit.kind == KIND_OVERLAP) { return FLAG_UNMATCHED; }
  if (hit.kind == KIND_TOUCH) {
    // A shared vertex is fine; an endpoint of either segment inside the other is a T-junction.
    let onSegmentRow = samePoint(hit.point, a) || samePoint(hit.point, b);
    let onSegmentOther = samePoint(hit.point, c) || samePoint(hit.point, d);
    if (!(onSegmentRow && onSegmentOther)) { return FLAG_UNMATCHED; }
  }
  if (gapWidth > 0.0 && isNearlyParallel(a, b, c, d, gapWidth)) { return FLAG_GAP; }
  return 0u;
}

// Contribution of segment other to the winding number of point (effective interior on the left).
fn windingContribution(other: u32, point: vec2f) -> i32 {
  var start = segmentStart(other);
  var end = segmentEnd(other);
  if (isFlipped(other)) {
    let swap = start;
    start = end;
    end = swap;
  }
  let side = (end.x - start.x) * (point.y - start.y) - (point.x - start.x) * (end.y - start.y);
  if (start.y <= point.y) {
    if (end.y > point.y && side > 0.0) { return 1; }
  } else if (end.y <= point.y && side < 0.0) {
    return -1;
  }
  return 0;
}`,
        body: `var flags = 0u;
  if (segmentValid(index)) {
    let gapWidth = max(parameters[parametersOffset], 0.0);
    let polygon = segmentFeature(index);
    let a = segmentStart(index);
    let b = segmentEnd(index);
    var matched = false;
    ${traverse(
      'min(a, b) - vec2f(gapWidth)',
      'max(a, b) + vec2f(gapWidth)',
      `let result = comparePair(index, other, gapWidth);
          matched = matched || (result & MATCHED_BIT) != 0u;
          flags = flags | (result & ~MATCHED_BIT);`
    )}
    // A segment matched by an identical segment of another polygon is not tested against the rest,
    // as in GEOS; only a same-side duplicate (FLAG_OVERLAP) stays flagged.
    if (matched) {
      flags = flags & FLAG_OVERLAP;
    }
    if (flags == 0u && !matched) {
      let midpoint = a * 0.5 + b * 0.5;
      var winding = 0;
      ${traverse(
        'vec2f(midpoint.x, midpoint.y)',
        'vec2f(FLOAT32_MAXIMUM, midpoint.y)',
        'winding = winding + windingContribution(other, midpoint);'
      )}
      if (winding > 0) { flags = FLAG_OVERLAP; }
    }
  }
  segmentFlags[segmentFlagsOffset + index] = flags;`
      })
    );

    // Counts per polygon and in total, with integer atomics.
    const needsCounts = Boolean(
      output.polygonInvalidCounts || output.invalidSegmentCount || output.isValid
    );
    if (needsCounts) {
      const polygonCounts =
        output.polygonInvalidCounts ?? view('polygon-invalid-counts', polygonCount);
      const total = output.invalidSegmentCount ?? view('invalid-segment-count', 1);
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-clear-polygon-counts`,
          operation: OPERATION,
          view: polygonCounts,
          type: 'u32',
          value: '0u'
        }),
        createFillNode<Parameters>(graph, {
          id: `${id}-clear-total`,
          operation: OPERATION,
          view: total,
          type: 'u32',
          value: '0u'
        })
      );
      const countBindings: WGSLKernelBinding[] = [
        {name: 'segmentTable', view: table, type: 'u32', access: 'read'},
        {name: 'segmentFlags', view: output.segmentFlags, type: 'u32', access: 'read'},
        {name: 'polygonCounts', view: polygonCounts, type: 'atomic<u32>', access: 'read_write'},
        {name: 'total', view: total, type: 'atomic<u32>', access: 'read_write'}
      ];
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-count`,
          operation: OPERATION,
          variant: 'count',
          bindings: countBindings,
          invocationCount: vertexCount,
          declarations: `const SEGMENT_NONE: u32 = 0xffffffffu;
${getSegmentTableAccessorsWGSL('segment')}`,
          body: `if (segmentFlags[segmentFlagsOffset + index] != 0u && segmentValid(index)) {
    atomicAdd(&polygonCounts[polygonCountsOffset + segmentFeature(index)], 1u);
    atomicAdd(&total[totalOffset], 1u);
  }`
        })
      );
      if (output.isValid) {
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-finalize`,
            operation: OPERATION,
            variant: 'finalize',
            bindings: [
              {name: 'total', view: total, type: 'u32', access: 'read'},
              {name: 'isValid', view: output.isValid, type: 'u32', access: 'read_write'}
            ],
            invocationCount: 1,
            body: 'isValid[isValidOffset] = select(0u, 1u, total[totalOffset] == 0u);'
          })
        );
      }
    }
    return nodes;
  }
}
