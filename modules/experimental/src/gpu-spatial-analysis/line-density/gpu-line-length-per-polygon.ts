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
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {getSortedSegmentSumNodes} from '../../utils/sorted-segment-sums';
import {
  createFillNode,
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../utils/wgsl-kernel-nodes';
import {GEODESIC_WGSL, GPU_GEODESIC_MEAN_EARTH_RADIUS} from '../geometry-measures/geodesic-wgsl';
import {FIND_PATH_WGSL} from '../line-segmentize/line-segmentize-kernels';
import {GPUSpatialJoinCandidates} from '../spatial-join/spatial-join-candidates';
import {createFeatureRingsNode} from '../spatial-join/spatial-join-geometry';
import type {GPUSpatialJoinPolygons} from '../spatial-join/spatial-join-types';

const OPERATION = 'GPULineLengthPerPolygon';

/** Caller-owned outputs of {@link GPULineLengthPerPolygon}. */
export type GPULineLengthPerPolygonOutput = {
  /**
   * Total length of the part of every line that lies inside each polygon feature, one row per
   * polygon feature, in position units (`'planar'`) or sphere-radius units (`'spherical'`).
   */
  lengths: GraphDataView<'float32'>;
  /** Optional `sum(length * pathWeight)` per polygon feature. Requires `pathWeights`. */
  weightedLengths?: GraphDataView<'float32'>;
  /**
   * Optional number of segments with a positive length inside each polygon feature. A segment
   * that only touches the polygon boundary is not counted.
   */
  segmentCounts?: GraphDataView<'uint32'>;
  /**
   * One-row scalar receiving `1` when any capacity overflowed (candidate pairs, the BVH, or the
   * per-pair crossing bound), in which case lengths can be low, else `0`.
   */
  overflow: GraphDataView<'uint32'>;
};

/** Properties for {@link GPULineLengthPerPolygon}. */
export type GPULineLengthPerPolygonProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'line-length-per-polygon'`. */
  id?: string;
  /** Packed line vertices sorted by path: planar coordinates, or longitude/latitude degrees. */
  positions: GraphDataView<'float32x2'>;
  /** `pathCount + 1` monotonic row offsets; path `p` owns rows `[pathOffsets[p], pathOffsets[p + 1])`. */
  pathOffsets: GraphDataView<'uint32'>;
  /** Optional per-path value (`pathCount` rows) that multiplies the length for `weightedLengths`. */
  pathWeights?: GraphDataView<'float32'>;
  /** Polygon or multipolygon features, in the layout of `GPUSpatialPredicateJoin`. */
  polygons: GPUSpatialJoinPolygons;
  /** `'planar'` (default) or `'spherical'` (longitude/latitude degrees, great-circle piece lengths). */
  coordinateSystem?: 'planar' | 'spherical';
  /** Sphere radius of `'spherical'`. Defaults to {@link GPU_GEODESIC_MEAN_EARTH_RADIUS}. */
  radius?: number;
  /**
   * Compile-time capacity of (segment, polygon) bounding-box candidate pairs. Default
   * `max(1024, 8 * positions.length)`.
   */
  maximumCandidatePairs?: number;
  /**
   * Compile-time bound on edge crossings kept per (segment, polygon) pair. Default 64. Extra
   * crossings are ignored and set `overflow`.
   */
  maximumCrossings?: number;
  /** Output columns. */
  output: GPULineLengthPerPolygonOutput;
};

/**
 * Total line length inside each polygon (QGIS "Sum line lengths", PostGIS
 * `sum(ST_Length(ST_Intersection(line, polygon)))` per polygon), optionally weighted by a per-path
 * value, plus the number of segments that contribute.
 *
 * Every vertex row `i` that is not the last of its path defines one segment. The segments are
 * joined to the polygons with `GPUSpatialJoinCandidates` (bounding boxes, BVH probe). For each
 * candidate pair the segment is cut at every crossing with a polygon ring edge (at most
 * `maximumCrossings` kept, insertion-sorted by parameter `t`), and every piece between consecutive
 * cuts is kept when its midpoint is inside the feature (even/odd fill over all rings, so holes and
 * multipolygon parts work, as in the spatial-join contributors). Piece lengths are summed per pair,
 * then per polygon by a stable sort and fixed-order segmented sum, so results are bitwise
 * reproducible. No float atomics (segment counts use integer atomics).
 *
 * Pieces lying exactly along a ring edge are ambiguous (midpoint on the boundary) and may count or
 * not. Lines are not unioned: overlapping lines count once each, as in QGIS. `'spherical'` clips
 * straight in longitude/latitude (as {@link GPULineDensity} does) and measures each piece as the
 * great-circle distance between its endpoints.
 *
 * Precision: f32 inputs and f32 clip parameters (relative length error about `1e-6` of the
 * segment length; a sliver piece near a boundary can flip its midpoint test).
 */
export class GPULineLengthPerPolygon implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPULineLengthPerPolygonProps;
  /** Number of polygon features. */
  readonly polygonCount: number;
  /** Resolved coordinate system. */
  readonly coordinateSystem: 'planar' | 'spherical';
  /** Resolved sphere radius. */
  readonly radius: number;
  /** Resolved candidate pair capacity. */
  readonly maximumCandidatePairs: number;
  /** Resolved crossing bound. */
  readonly maximumCrossings: number;

  constructor(props: GPULineLengthPerPolygonProps) {
    this.id = props.id ?? 'line-length-per-polygon';
    this.props = props;
    const {id} = this;
    this.coordinateSystem = props.coordinateSystem ?? 'planar';
    this.radius = props.radius ?? GPU_GEODESIC_MEAN_EARTH_RADIUS;
    this.maximumCandidatePairs =
      props.maximumCandidatePairs ?? Math.max(1024, 8 * props.positions.length);
    this.maximumCrossings = props.maximumCrossings ?? 64;
    if (this.coordinateSystem !== 'planar' && this.coordinateSystem !== 'spherical') {
      throw new Error(`${id} coordinateSystem must be 'planar' or 'spherical'`);
    }
    if (!Number.isFinite(this.radius) || this.radius <= 0) {
      throw new Error(`${id} radius must be a positive finite number`);
    }
    for (const [name, value] of [
      ['maximumCandidatePairs', this.maximumCandidatePairs],
      ['maximumCrossings', this.maximumCrossings]
    ] as const) {
      if (!Number.isSafeInteger(value) || value < 1) {
        throw new Error(`${id} ${name} must be a positive integer`);
      }
    }
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    if (props.positions.length < 2) {
      throw new Error(`${id} needs at least two positions`);
    }
    validatePackedUint32View(props.pathOffsets, `${id} pathOffsets`);
    if (props.pathOffsets.length < 2) {
      throw new Error(`${id} pathOffsets must contain at least two rows`);
    }
    if (props.pathWeights) {
      validatePackedView(props.pathWeights, ['float32'], `${id} pathWeights`);
      if (props.pathWeights.length !== props.pathOffsets.length - 1) {
        throw new Error(`${id} pathWeights must hold one row per path`);
      }
    }
    if (props.polygons.kind !== 'polygons') {
      throw new Error(`${id} polygons must have kind 'polygons'`);
    }
    for (const [name, view] of [
      ['featureOffsets', props.polygons.featureOffsets],
      ['polygonOffsets', props.polygons.polygonOffsets],
      ['ringOffsets', props.polygons.ringOffsets]
    ] as const) {
      validatePackedUint32View(view, `${id} polygons.${name}`);
    }
    validatePackedView(props.polygons.positions, ['float32x2'], `${id} polygons.positions`);
    this.polygonCount = props.polygons.featureOffsets.length - 1;
    if (this.polygonCount < 1) {
      throw new Error(`${id} polygons must contain at least one feature`);
    }
    const {lengths, weightedLengths, segmentCounts, overflow} = props.output;
    validatePackedView(lengths, ['float32'], `${id} output.lengths`);
    if (lengths.length !== this.polygonCount) {
      throw new Error(`${id} output.lengths must hold ${this.polygonCount} rows`);
    }
    if (weightedLengths) {
      if (!props.pathWeights) {
        throw new Error(`${id} output.weightedLengths requires pathWeights`);
      }
      validatePackedView(weightedLengths, ['float32'], `${id} output.weightedLengths`);
      if (weightedLengths.length !== this.polygonCount) {
        throw new Error(`${id} output.weightedLengths must hold ${this.polygonCount} rows`);
      }
    }
    if (segmentCounts) {
      validatePackedUint32View(segmentCounts, `${id} output.segmentCounts`);
      if (segmentCounts.length !== this.polygonCount) {
        throw new Error(`${id} output.segmentCounts must hold ${this.polygonCount} rows`);
      }
    }
    validatePackedUint32View(overflow, `${id} output.overflow`);
    if (overflow.length < 1) {
      throw new Error(`${id} output.overflow must contain one uint32 row`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [lengths, weightedLengths, segmentCounts, overflow],
      [
        props.positions,
        props.pathOffsets,
        props.pathWeights,
        props.polygons.positions,
        props.polygons.featureOffsets,
        props.polygons.polygonOffsets,
        props.polygons.ringOffsets
      ]
    );
  }

  /** Returns the segment build, candidate join, ring clip, per-pair finish and sorted sum nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, polygonCount, maximumCandidatePairs, maximumCrossings} = this;
    const {polygons, output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.pathOffsets,
      props.pathWeights,
      polygons.positions,
      polygons.featureOffsets,
      polygons.polygonOffsets,
      polygons.ringOffsets,
      output.lengths,
      output.weightedLengths,
      output.segmentCounts,
      output.overflow
    ]);
    const spherical = this.coordinateSystem === 'spherical';
    const rowCount = props.positions.length;
    const pathCount = props.pathOffsets.length - 1;
    const nodes: GPUCommandNode<Parameters>[] = [];

    // One two-vertex segment feature per vertex row; non-segments are NaN so they never match.
    const segmentPositions = createTransientView(
      graph,
      `${id}-segment-positions`,
      'float32x2',
      2 * rowCount
    );
    const segmentOffsets = createTransientView(
      graph,
      `${id}-segment-offsets`,
      'uint32',
      rowCount + 1
    );
    const segmentWeights = createTransientView(graph, `${id}-segment-weights`, 'float32', rowCount);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-build-segments`,
        operation: OPERATION,
        variant: 'build-segments',
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          {name: 'pathOffsets', view: props.pathOffsets, type: 'u32', access: 'read'},
          {
            name: 'pathWeights',
            view: props.pathWeights ?? output.lengths,
            type: 'f32',
            access: 'read'
          },
          {name: 'segmentPositions', view: segmentPositions, type: 'f32', access: 'read_write'},
          {name: 'segmentOffsets', view: segmentOffsets, type: 'u32', access: 'read_write'},
          {name: 'segmentWeights', view: segmentWeights, type: 'f32', access: 'read_write'}
        ],
        invocationCount: rowCount + 1,
        declarations: `const PATH_COUNT: u32 = ${pathCount}u;
const ROW_COUNT: u32 = ${rowCount}u;
const HAS_WEIGHTS: bool = ${props.pathWeights ? 'true' : 'false'};
${FIND_PATH_WGSL}`,
        body: `segmentOffsets[segmentOffsetsOffset + index] = 2u * index;
  if (index < ROW_COUNT) {
    let path = findPath(index);
    var valid = path != NO_PATH;
    var weight = 1.0;
    if (valid) {
      valid = index + 1u < min(pathOffsets[pathOffsetsOffset + path + 1u], ROW_COUNT);
      if (HAS_WEIGHTS) { weight = pathWeights[pathWeightsOffset + path]; }
    }
    var nanBits = 0x7fc00000u;
    let nan = bitcast<f32>(nanBits);
    var a = vec2f(nan);
    var b = vec2f(nan);
    if (valid) {
      a = vec2f(positions[positionsOffset + 2u * index], positions[positionsOffset + 2u * index + 1u]);
      b = vec2f(positions[positionsOffset + 2u * index + 2u], positions[positionsOffset + 2u * index + 3u]);
    }
    segmentPositions[segmentPositionsOffset + 4u * index] = a.x;
    segmentPositions[segmentPositionsOffset + 4u * index + 1u] = a.y;
    segmentPositions[segmentPositionsOffset + 4u * index + 2u] = b.x;
    segmentPositions[segmentPositionsOffset + 4u * index + 3u] = b.y;
    segmentWeights[segmentWeightsOffset + index] = select(0.0, weight, valid);
  }`
      })
    );

    const pairs = {
      leftIds: createTransientView(graph, `${id}-pair-left`, 'uint32', maximumCandidatePairs),
      rightIds: createTransientView(graph, `${id}-pair-right`, 'uint32', maximumCandidatePairs),
      count: createTransientView(graph, `${id}-pair-count`, 'uint32', 1),
      overflow: createTransientView(graph, `${id}-pair-overflow`, 'uint32', 1)
    };
    nodes.push(
      ...new GPUSpatialJoinCandidates({
        id: `${id}-candidates`,
        left: {kind: 'lines', positions: segmentPositions, lineOffsets: segmentOffsets},
        right: polygons,
        pairs
      }).getCommandNodes(graph)
    );

    const featureRings = createTransientView(
      graph,
      `${id}-feature-rings`,
      'uint32x2',
      polygonCount
    );
    nodes.push(
      createFeatureRingsNode<Parameters>(graph, `${id}-polygon-rings`, {
        featureCount: polygonCount,
        geometry: polygons,
        featureRings
      })
    );

    const crossingOverflow = createTransientView(graph, `${id}-crossing-overflow`, 'uint32', 1);
    const insideLengths = createTransientView(
      graph,
      `${id}-inside-lengths`,
      'float32',
      maximumCandidatePairs
    );
    nodes.push(
      createFillNode<Parameters>(graph, {
        id: `${id}-fill-crossing-overflow`,
        operation: OPERATION,
        view: crossingOverflow,
        type: 'u32',
        value: '0u'
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-clip`,
        operation: OPERATION,
        variant: spherical ? 'clip-spherical' : 'clip-planar',
        bindings: [
          {name: 'leftIds', view: pairs.leftIds, type: 'u32', access: 'read'},
          {name: 'rightIds', view: pairs.rightIds, type: 'u32', access: 'read'},
          {name: 'segmentPositions', view: segmentPositions, type: 'f32', access: 'read'},
          {name: 'featureRings', view: featureRings, type: 'u32', access: 'read'},
          {name: 'ringOffsets', view: polygons.ringOffsets, type: 'u32', access: 'read'},
          {name: 'polygonPositions', view: polygons.positions, type: 'f32', access: 'read'},
          {name: 'insideLengths', view: insideLengths, type: 'f32', access: 'read_write'},
          {
            name: 'crossingOverflow',
            view: crossingOverflow,
            type: 'atomic<u32>',
            access: 'read_write'
          }
        ],
        invocationCount: maximumCandidatePairs,
        declarations: `const NO_FEATURE: u32 = 0xffffffffu;
const VERTEX_COUNT: u32 = ${polygons.positions.length}u;
const MAXIMUM_CROSSINGS: u32 = ${maximumCrossings}u;
const RADIUS: f32 = ${getWGSLFloatLiteral(this.radius)};
const EDGE_TOLERANCE: f32 = 1.0e-6;
${spherical ? GEODESIC_WGSL : ''}

fn getVertex(row: u32) -> vec2f {
  return vec2f(polygonPositions[polygonPositionsOffset + 2u * row], polygonPositions[polygonPositionsOffset + 2u * row + 1u]);
}

fn cross2(a: vec2f, b: vec2f) -> f32 { return a.x * b.y - a.y * b.x; }

fn getPieceLength(a: vec2f, b: vec2f, t0: f32, t1: f32) -> f32 {
  ${
    spherical
      ? 'return geodesicCentralAngle(mix(a, b, t0), mix(a, b, t1)) * RADIUS;'
      : 'return (t1 - t0) * length(b - a);'
  }
}

// Even/odd fill over every ring of the feature.
fn isInside(point: vec2f, ringStart: u32, ringEnd: u32) -> bool {
  var inside = false;
  for (var ring = ringStart; ring < ringEnd; ring++) {
    let first = ringOffsets[ringOffsetsOffset + ring];
    let last = min(ringOffsets[ringOffsetsOffset + ring + 1u], VERTEX_COUNT);
    if (last < first + 3u) { continue; }
    var previous = getVertex(last - 1u);
    for (var row = first; row < last; row++) {
      let current = getVertex(row);
      if ((current.y > point.y) != (previous.y > point.y)) {
        let crossingX = (previous.x - current.x) * (point.y - current.y) / (previous.y - current.y) + current.x;
        if (point.x < crossingX) { inside = !inside; }
      }
      previous = current;
    }
  }
  return inside;
}`,
        body: `var inside = 0.0;
  let segment = leftIds[leftIdsOffset + index];
  let feature = rightIds[rightIdsOffset + index];
  if (segment != NO_FEATURE && feature != NO_FEATURE) {
    let a = vec2f(segmentPositions[segmentPositionsOffset + 4u * segment], segmentPositions[segmentPositionsOffset + 4u * segment + 1u]);
    let b = vec2f(segmentPositions[segmentPositionsOffset + 4u * segment + 2u], segmentPositions[segmentPositionsOffset + 4u * segment + 3u]);
    let delta = b - a;
    let ringStart = featureRings[featureRingsOffset + 2u * feature];
    let ringEnd = featureRings[featureRingsOffset + 2u * feature + 1u];
    if (a.x == a.x && a.y == a.y && b.x == b.x && b.y == b.y && (delta.x != 0.0 || delta.y != 0.0)) {
      var crossings = array<f32, ${maximumCrossings}>();
      var crossingCount = 0u;
      var truncated = false;
      for (var ring = ringStart; ring < ringEnd; ring++) {
        let first = ringOffsets[ringOffsetsOffset + ring];
        let last = min(ringOffsets[ringOffsetsOffset + ring + 1u], VERTEX_COUNT);
        if (last < first + 3u) { continue; }
        var previous = getVertex(last - 1u);
        for (var row = first; row < last; row++) {
          let current = getVertex(row);
          let edge = current - previous;
          let denominator = cross2(delta, edge);
          if (denominator != 0.0) {
            let offset = previous - a;
            let t = cross2(offset, edge) / denominator;
            let u = cross2(offset, delta) / denominator;
            // Extra breakpoints are harmless (pieces are tested by midpoint), so edges are
            // inclusive with a small tolerance and a vertex crossing is never dropped.
            if (t > 0.0 && t < 1.0 && u >= -EDGE_TOLERANCE && u <= 1.0 + EDGE_TOLERANCE) {
              if (crossingCount < MAXIMUM_CROSSINGS) {
                var slot = crossingCount;
                crossingCount++;
                loop {
                  if (slot == 0u || crossings[slot - 1u] <= t) { break; }
                  crossings[slot] = crossings[slot - 1u];
                  slot--;
                }
                crossings[slot] = t;
              } else {
                truncated = true;
              }
            }
          }
          previous = current;
        }
      }
      if (truncated) { atomicStore(&crossingOverflow[crossingOverflowOffset], 1u); }
      var start = 0.0;
      for (var piece = 0u; piece <= crossingCount; piece++) {
        let end = select(1.0, crossings[min(piece, MAXIMUM_CROSSINGS - 1u)], piece < crossingCount);
        if (end > start) {
          if (isInside(a + delta * (0.5 * (start + end)), ringStart, ringEnd)) {
            inside += getPieceLength(a, b, start, end);
          }
          start = end;
        }
      }
    }
  }
  insideLengths[insideLengthsOffset + index] = inside;`
      })
    );

    // Per-pair keys and contributions; segment counts via integer atomics are order independent.
    const counts =
      output.segmentCounts ?? createTransientView(graph, `${id}-counts`, 'uint32', polygonCount);
    const keys = createTransientView(graph, `${id}-keys`, 'uint32', maximumCandidatePairs);
    const weighted = createTransientView(graph, `${id}-weighted`, 'float32', maximumCandidatePairs);
    nodes.push(
      createFillNode<Parameters>(graph, {
        id: `${id}-fill-counts`,
        operation: OPERATION,
        view: counts,
        type: 'u32',
        value: '0u'
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-finish-pairs`,
        operation: OPERATION,
        variant: 'finish-pairs',
        bindings: [
          {name: 'leftIds', view: pairs.leftIds, type: 'u32', access: 'read'},
          {name: 'rightIds', view: pairs.rightIds, type: 'u32', access: 'read'},
          {name: 'insideLengths', view: insideLengths, type: 'f32', access: 'read'},
          {name: 'segmentWeights', view: segmentWeights, type: 'f32', access: 'read'},
          {name: 'keys', view: keys, type: 'u32', access: 'read_write'},
          {name: 'weighted', view: weighted, type: 'f32', access: 'read_write'},
          {name: 'counts', view: counts, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: maximumCandidatePairs,
        declarations: `const NO_FEATURE: u32 = 0xffffffffu;
const POLYGON_COUNT: u32 = ${polygonCount}u;`,
        body: `let segment = leftIds[leftIdsOffset + index];
  let feature = rightIds[rightIdsOffset + index];
  let inside = insideLengths[insideLengthsOffset + index];
  var key = POLYGON_COUNT;
  var contribution = 0.0;
  if (segment != NO_FEATURE && feature < POLYGON_COUNT && inside > 0.0) {
    key = feature;
    contribution = inside * segmentWeights[segmentWeightsOffset + segment];
    atomicAdd(&counts[countsOffset + feature], 1u);
  }
  keys[keysOffset + index] = key;
  weighted[weightedOffset + index] = contribution;`
      }),
      ...getSortedSegmentSumNodes<Parameters>(graph, {
        id: `${id}-sum`,
        operation: OPERATION,
        segmentCount: polygonCount,
        segmentKeys: keys,
        segmentCounts: counts,
        sumContributions: insideLengths,
        sums: output.lengths,
        reductions: output.weightedLengths
          ? [{name: 'weighted', contributions: weighted, output: output.weightedLengths}]
          : []
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: OPERATION,
        variant: 'publish',
        bindings: [
          {name: 'pairOverflow', view: pairs.overflow, type: 'u32', access: 'read'},
          {name: 'crossingOverflow', view: crossingOverflow, type: 'u32', access: 'read'},
          {name: 'overflow', view: output.overflow, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 1,
        body: `overflow[overflowOffset] = select(0u, 1u, pairOverflow[pairOverflowOffset] != 0u || crossingOverflow[crossingOverflowOffset] != 0u);`
      })
    );
    return nodes;
  }
}
