// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUOverflowStatusPort} from '../../analysis-core/contracts/index';
import {
  getGPUGreatCircleRadius,
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT
} from '../../analysis-core/contracts/spatial-context';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  getStorageBindingElementLimit,
  validateMaximumCount,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {getSortedSegmentSumNodes} from '../../utils/sorted-segment-sums';
import {
  createFillNode,
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../utils/wgsl-kernel-nodes';
import {
  resolveGPUSpatialContext,
  resolveGPUSecondarySpatialContext,
  GPU_LINEAR_EDGE_MODELS,
  type GPULonLatPolygonStatusPort,
  type GPUSpatialContext,
  type GPUSpatialContextSupport
} from '../../analysis-core/contracts/index';
import {IS_NAN_BITS_WGSL} from '../../utils/wgsl-common';
import {EXACT_GEOMETRY_WGSL, getExactRingWindingWGSL} from '../../utils/exact-geometry-wgsl';
import {EXACT_ORIENTATION_WGSL} from '../../utils/exact-orientation-wgsl';
import {GEODESIC_WGSL} from '../geometry-measures/geodesic-wgsl';
import {FIND_LINE_WGSL} from '../line-segmentize/line-segmentize-kernels';
import {GPUSpatialJoinCandidates} from '../spatial-join/gpu-spatial-join-candidates';
import {createFeatureRingsNode} from '../spatial-join/spatial-join-geometry';
import type {
  GPULineGeometryFields,
  GPUPackedGeometryPort,
  GPUPolygonGeometryPort
} from '../../analysis-core/contracts/index';

import {
  createLonLatPolygonCheckNode,
  hasLonLatPolygonStatus
} from '../../analysis-core/lon-lat-ring-wgsl';

const OPERATION = 'GPULineLengthPerPolygon';

/** Planar/native, or longitude-latitude with the great-circle metric. */
const LINE_LENGTH_SPATIAL_CONTEXT_SUPPORT: GPUSpatialContextSupport = Object.freeze({
  ...GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  edgeModels: GPU_LINEAR_EDGE_MODELS,
  measure: 'distance'
});

/** Caller-owned outputs of {@link GPULineLengthPerPolygon}. */
export type GPULineLengthPerPolygonOutput = {
  /**
   * Total length of the part of every line that lies inside each polygon feature, one row per
   * polygon feature, in the context's length unit (position units for planar, meters by default
   * for longitude/latitude).
   */
  lengths: GraphDataView<'float32'>;
  /** Optional `sum(length * lineWeight)` per polygon feature. Requires `lineWeights`. */
  weightedLengths?: GraphDataView<'float32'>;
  /**
   * Optional number of segments with a positive length inside each polygon feature. A segment
   * that only touches the polygon boundary is not counted.
   */
  segmentCounts?: GraphDataView<'uint32'>;
  /**
   * One-row status scalar: `overflow` receives `1` when any capacity overflowed (candidate pairs,
   * the BVH, or the per-pair crossing bound), in which case lengths can be low, else `0`.
   * `antimeridianUncertainCount` counts unprepared crossing rings; `poleUncertainCount` counts
   * uncapped pole rings. Both optional counters are rewritten every lon/lat encoding and left
   * untouched in planar contexts.
   */
  status: GPUOverflowStatusPort & GPULonLatPolygonStatusPort;
};

/**
 * Largest `positions` row count of {@link GPULineLengthPerPolygon} on `device`: the segment table
 * holds two `float32x2` vertices (16 bytes) per row in one storage binding, so 8388608 on a default
 * device (128 MiB bindings).
 */
export function getGPULineLengthPerPolygonMaximumPositionCount(device: Device): number {
  return getStorageBindingElementLimit(device, 16);
}

/**
 * Largest `pairCapacity` of {@link GPULineLengthPerPolygon} on `device`: the candidate pair table of
 * `GPUSpatialJoinCandidates` holds 8 bytes per pair in one storage binding, so 16777216 on a
 * default device (128 MiB bindings).
 */
export function getGPULineLengthPerPolygonMaximumPairCapacity(device: Device): number {
  return getStorageBindingElementLimit(device, 8);
}

/** Properties for {@link GPULineLengthPerPolygon}. */
export type GPULineLengthPerPolygonProps = GPULineGeometryFields & {
  /** Prefix for generated node and transient IDs. Defaults to `'line-length-per-polygon'`. */
  id?: string;
  /** Optional per-line value (`lineCount` rows) that multiplies the length for `weightedLengths`. */
  lineWeights?: GraphDataView<'float32'>;
  /** Polygon or multipolygon features in the four-level layout; rows are features. */
  polygons: GPUPackedGeometryPort<GPUPolygonGeometryPort>;
  /**
   * Planar/native (default when omitted) or longitude-latitude with the `'great-circle'` metric
   * (positions in degrees, piece lengths in the context's length unit, meters by default). Replaces
   * the former `coordinateSystem` and `radius` props. Other contexts throw at construction.
   * Edges must be linear. Prepare polygons that cross ±180 or enclose a pole with
   * `GPULonLatPolygonPreparation`; optional status counters report unprepared rings.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * Context of `polygons` when declared separately from `spatialContext`. Must be compatible
   * with `spatialContext` (datum, coordinate space, metric, units, planar frame).
   * Longitude/latitude must use linear edges.
   * Omitted means `spatialContext` describes both inputs.
   */
  polygonsSpatialContext?: GPUSpatialContext;
  /**
   * Compile-time capacity of (segment, polygon) bounding-box candidate pairs, at most
   * {@link getGPULineLengthPerPolygonMaximumPairCapacity} (16777216 on a default device). Default
   * `min(max(1024, 8 * positions.length), getGPULineLengthPerPolygonMaximumPairCapacity(device))`.
   */
  pairCapacity?: number;
  /**
   * Compile-time bound on edge crossings kept per (segment, polygon) pair, 1 to 1024. Default 64.
   * Extra crossings are ignored and set `status.overflow`. The crossings live in a private array,
   * which must stay within the 8,192 B per-function limit of core WebGPU.
   */
  maximumCrossingCount?: number;
  /** Output columns. */
  output: GPULineLengthPerPolygonOutput;
};

/**
 * Total line length inside each polygon (QGIS "Sum line lengths", PostGIS
 * `sum(ST_Length(ST_Intersection(line, polygon)))` per polygon), optionally weighted by a per-line
 * value, plus the number of segments that contribute.
 *
 * Every vertex row `i` that is not the last of its line defines one segment. The segments are
 * joined to the polygons with `GPUSpatialJoinCandidates` (bounding boxes, BVH probe). For each
 * candidate pair the segment is cut at every crossing with a polygon ring edge (at most
 * `maximumCrossingCount` kept, insertion-sorted by parameter `t`), and every piece between consecutive
 * cuts is kept when its midpoint is inside the feature (even/odd fill over all rings, so holes and
 * multipolygon parts work, as in the spatial-join contributors). Piece lengths are summed per pair,
 * then per polygon by a stable sort and fixed-order segmented sum, so results are bitwise
 * reproducible. No float atomics (segment counts use integer atomics).
 *
 * Pieces lying exactly along a ring edge are ambiguous (midpoint on the boundary) and may count or
 * not. Lines are not unioned: overlapping lines count once each, as in QGIS. A longitude-latitude context clips
 * straight in longitude/latitude (as {@link GPULineDensity} does) and measures each piece as the
 * great-circle distance between its endpoints.
 *
 * Limits: `positions` has at most {@link getGPULineLengthPerPolygonMaximumPositionCount} rows (8388608 on a default device).
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
  /** Resolved coordinate and metric contract. */
  readonly spatialContext: GPUSpatialContext;
  /** Resolved coordinate system derived from `spatialContext`. */
  readonly coordinateSystem: 'planar' | 'spherical';
  /** Mean sphere radius in the context's length unit unless overridden; 1 for planar. */
  readonly radius: number;
  /** Resolved candidate pair capacity. */
  readonly pairCapacity: number;
  /** Resolved crossing bound. */
  readonly maximumCrossingCount: number;

  constructor(props: GPULineLengthPerPolygonProps) {
    this.id = props.id ?? 'line-length-per-polygon';
    this.props = props;
    const {id} = this;
    this.spatialContext = resolveGPUSpatialContext(
      id,
      props.spatialContext,
      LINE_LENGTH_SPATIAL_CONTEXT_SUPPORT
    );
    resolveGPUSecondarySpatialContext(
      id,
      'polygons',
      props.polygonsSpatialContext,
      this.spatialContext,
      LINE_LENGTH_SPATIAL_CONTEXT_SUPPORT.edgeModels
    );
    const spherical = this.spatialContext.coordinateSpace === 'longitude-latitude';
    this.coordinateSystem = spherical ? 'spherical' : 'planar';
    this.radius = getGPUGreatCircleRadius(this.spatialContext) ?? 1;
    // The default is clamped to the device limit at declaration, see getCommandNodes.
    this.pairCapacity = props.pairCapacity ?? Math.max(1024, 8 * props.positions.length);
    this.maximumCrossingCount = props.maximumCrossingCount ?? 64;
    validateCapacity(id, 'pairCapacity', this.pairCapacity);
    validateCapacity(id, 'maximumCrossingCount', this.maximumCrossingCount, 1, 1024);
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    if (props.positions.length < 2) {
      throw new Error(`${id} needs at least two positions`);
    }
    validatePackedView(props.lineOffsets, ['uint32'], `${id} lineOffsets`);
    if (props.lineOffsets.length < 2) {
      throw new Error(`${id} lineOffsets needs two rows`);
    }
    if (props.lineWeights) {
      validatePackedView(props.lineWeights, ['float32'], `${id} lineWeights`);
      if (props.lineWeights.length !== props.lineOffsets.length - 1) {
        throw new Error(`${id} lineWeights must hold one row per line`);
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
      validatePackedView(view, ['uint32'], `${id} polygons.${name}`);
    }
    validatePackedView(props.polygons.positions, ['float32x2'], `${id} polygons.positions`);
    this.polygonCount = props.polygons.featureOffsets.length - 1;
    if (this.polygonCount < 1) {
      throw new Error(`${id} polygons must contain at least one feature`);
    }
    const {lengths, weightedLengths, segmentCounts, status} = props.output;
    const {overflow} = status;
    validatePackedView(lengths, ['float32'], `${id} output.lengths`);
    if (lengths.length !== this.polygonCount) {
      throw new Error(`${id} output.lengths must hold ${this.polygonCount} rows`);
    }
    if (weightedLengths) {
      if (!props.lineWeights) {
        throw new Error(`${id} output.weightedLengths requires lineWeights`);
      }
      validatePackedView(weightedLengths, ['float32'], `${id} output.weightedLengths`);
      if (weightedLengths.length !== this.polygonCount) {
        throw new Error(`${id} output.weightedLengths must hold ${this.polygonCount} rows`);
      }
    }
    if (segmentCounts) {
      validatePackedView(segmentCounts, ['uint32'], `${id} output.segmentCounts`);
      if (segmentCounts.length !== this.polygonCount) {
        throw new Error(`${id} output.segmentCounts must hold ${this.polygonCount} rows`);
      }
    }
    validatePackedView(overflow, ['uint32'], `${id} output.status.overflow`);
    if (overflow.length < 1) {
      throw new Error(`${id} output.status.overflow must contain one uint32 row`);
    }
    for (const name of ['antimeridianUncertainCount', 'poleUncertainCount'] as const) {
      const view = status[name];
      if (view) {
        validatePackedView(view, ['uint32'], `${id} output.status.${name}`);
        if (view.length < 1) {
          throw new Error(`${id} output.status.${name} must contain one uint32 row`);
        }
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        lengths,
        weightedLengths,
        segmentCounts,
        overflow,
        status.antimeridianUncertainCount,
        status.poleUncertainCount
      ],
      [
        props.positions,
        props.lineOffsets,
        props.lineWeights,
        props.polygons.positions,
        props.polygons.featureOffsets,
        props.polygons.polygonOffsets,
        props.polygons.ringOffsets
      ]
    );
  }

  /** Returns optional polygon checks, segment build, candidate join, clip and sorted sum nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, polygonCount, maximumCrossingCount} = this;
    const {polygons, output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.lineOffsets,
      props.lineWeights,
      polygons.positions,
      polygons.featureOffsets,
      polygons.polygonOffsets,
      polygons.ringOffsets,
      output.lengths,
      output.weightedLengths,
      output.segmentCounts,
      output.status.overflow,
      output.status.antimeridianUncertainCount,
      output.status.poleUncertainCount
    ]);
    validateMaximumCount(
      id,
      'positions rows',
      props.positions.length,
      getGPULineLengthPerPolygonMaximumPositionCount(graph.device)
    );
    const maximumPairCapacity = getGPULineLengthPerPolygonMaximumPairCapacity(graph.device);
    // A default capacity clamps to the device limit; an explicit one above it is refused.
    const pairCapacity = props.pairCapacity ?? Math.min(this.pairCapacity, maximumPairCapacity);
    validateCapacity(id, 'pairCapacity', pairCapacity, 1, maximumPairCapacity);
    const spherical = this.coordinateSystem === 'spherical';
    const rowCount = props.positions.length;
    const lineCount = props.lineOffsets.length - 1;
    const nodes: GPUCommandNode<Parameters>[] = [];
    if (
      this.spatialContext.coordinateSpace === 'longitude-latitude' &&
      hasLonLatPolygonStatus(output.status)
    ) {
      const status = output.status;
      for (const name of ['antimeridianUncertainCount', 'poleUncertainCount'] as const) {
        const view = status[name];
        if (view) {
          nodes.push(
            createFillNode(graph, {
              id: `${id}-clear-${name}`,
              operation: OPERATION,
              view,
              type: 'u32',
              value: '0u',
              componentCount: 1
            })
          );
        }
      }
      nodes.push(
        createLonLatPolygonCheckNode(graph, {
          id: `${id}-lon-lat-polygon-check-0`,
          operation: OPERATION,
          positions: polygons.positions,
          ringOffsets: polygons.ringOffsets,
          status
        })
      );
    }

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
          {name: 'lineOffsets', view: props.lineOffsets, type: 'u32', access: 'read'},
          {
            name: 'lineWeights',
            view: props.lineWeights ?? output.lengths,
            type: 'f32',
            access: 'read'
          },
          {name: 'segmentPositions', view: segmentPositions, type: 'f32', access: 'read_write'},
          {name: 'segmentOffsets', view: segmentOffsets, type: 'u32', access: 'read_write'},
          {name: 'segmentWeights', view: segmentWeights, type: 'f32', access: 'read_write'}
        ],
        invocationCount: rowCount + 1,
        declarations: `const LINE_COUNT: u32 = ${lineCount}u;
const ROW_COUNT: u32 = ${rowCount}u;
const HAS_WEIGHTS: bool = ${props.lineWeights ? 'true' : 'false'};
${FIND_LINE_WGSL}`,
        body: `segmentOffsets[segmentOffsetsOffset + index] = 2u * index;
  if (index < ROW_COUNT) {
    let line = findLine(index);
    var valid = line != NO_LINE;
    var weight = 1.0;
    if (valid) {
      valid = index + 1u < min(lineOffsets[lineOffsetsOffset + line + 1u], ROW_COUNT);
      if (HAS_WEIGHTS) { weight = lineWeights[lineWeightsOffset + line]; }
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
      leftIds: createTransientView(graph, `${id}-pair-left`, 'uint32', pairCapacity),
      rightIds: createTransientView(graph, `${id}-pair-right`, 'uint32', pairCapacity),
      status: {
        count: createTransientView(graph, `${id}-pair-count`, 'uint32', 1),
        overflow: createTransientView(graph, `${id}-pair-overflow`, 'uint32', 1)
      }
    };
    nodes.push(
      ...new GPUSpatialJoinCandidates({
        id: `${id}-candidates`,
        left: {kind: 'lines', positions: segmentPositions, lineOffsets: segmentOffsets},
        right: polygons,
        output: pairs
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
      pairCapacity
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
        invocationCount: pairCapacity,
        declarations: `const NO_FEATURE: u32 = 0xffffffffu;
const VERTEX_COUNT: u32 = ${polygons.positions.length}u;
const MAXIMUM_CROSSINGS: u32 = ${maximumCrossingCount}u;
const RADIUS: f32 = ${getWGSLFloatLiteral(this.radius)};
const EDGE_TOLERANCE: f32 = 1.0e-6;
${spherical ? GEODESIC_WGSL : ''}
${IS_NAN_BITS_WGSL}
${EXACT_ORIENTATION_WGSL}
${EXACT_GEOMETRY_WGSL}

fn getVertex(row: u32) -> vec2f {
  return vec2f(polygonPositions[polygonPositionsOffset + 2u * row], polygonPositions[polygonPositionsOffset + 2u * row + 1u]);
}

var<private> segmentOrigin: vec2f;

fn getRelativeVertex(row: u32) -> vec2f {
  return getVertex(row) - segmentOrigin;
}

fn cross2(a: vec2f, b: vec2f) -> f32 { return a.x * b.y - a.y * b.x; }

fn getPieceLength(a: vec2f, b: vec2f, t0: f32, t1: f32) -> f32 {
  ${
    spherical
      ? 'return geodesicCentralAngle(mix(a, b, t0), mix(a, b, t1)) * RADIUS;'
      : 'return (t1 - t0) * length(b - a);'
  }
}

${getExactRingWindingWGSL('getRingWinding', 'getRelativeVertex')}
// Even/odd fill over every ring of the feature, with the shared exact point-in-ring rule
// (utils/exact-geometry-wgsl.ts). Called from one site.
fn isInside(point: vec2f, ringStart: u32, ringEnd: u32) -> bool {
  let probe = ExactProbe(point, point, point, EXACT_PROBE_POINT);
  var winding = 0;
  for (var ring = ringStart; ring < ringEnd; ring++) {
    let first = ringOffsets[ringOffsetsOffset + ring];
    let last = min(ringOffsets[ringOffsetsOffset + ring + 1u], VERTEX_COUNT);
    winding += getRingWinding(probe, first, last).x;
  }
  return (winding & 1) != 0;
}`,
        body: `var inside = 0.0;
  let segment = leftIds[leftIdsOffset + index];
  let feature = rightIds[rightIdsOffset + index];
  if (segment != NO_FEATURE && feature != NO_FEATURE) {
    let a = vec2f(segmentPositions[segmentPositionsOffset + 4u * segment], segmentPositions[segmentPositionsOffset + 4u * segment + 1u]);
    let b = vec2f(segmentPositions[segmentPositionsOffset + 4u * segment + 2u], segmentPositions[segmentPositionsOffset + 4u * segment + 3u]);
    let delta = b - a;
    segmentOrigin = a;
    let ringStart = featureRings[featureRingsOffset + 2u * feature];
    let ringEnd = featureRings[featureRingsOffset + 2u * feature + 1u];
    if (!isNanBits(a.x) && !isNanBits(a.y) && !isNanBits(b.x) && !isNanBits(b.y) && (delta.x != 0.0 || delta.y != 0.0)) {
      var crossings = array<f32, ${maximumCrossingCount}>();
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
          // Classify in the segment's local frame. Adding the world origin can round a
          // genuine interior midpoint onto a hole edge, losing a contributing segment.
          if (isInside(delta * (0.5 * (start + end)), ringStart, ringEnd)) {
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

    // Per-pair keys and contributions. Segment counts (integer atomics, order independent) are
    // only built for the public `segmentCounts` output; the sums find segment sizes by search.
    const counts = output.segmentCounts;
    const keys = createTransientView(graph, `${id}-keys`, 'uint32', pairCapacity);
    const weighted = createTransientView(graph, `${id}-weighted`, 'float32', pairCapacity);
    if (counts) {
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-fill-counts`,
          operation: OPERATION,
          view: counts,
          type: 'u32',
          value: '0u'
        })
      );
    }
    nodes.push(
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
          ...(counts
            ? [
                {
                  name: 'counts',
                  view: counts,
                  type: 'atomic<u32>' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        invocationCount: pairCapacity,
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
    ${counts ? 'atomicAdd(&counts[countsOffset + feature], 1u);' : ''}
  }
  keys[keysOffset + index] = key;
  weighted[weightedOffset + index] = contribution;`
      }),
      ...getSortedSegmentSumNodes<Parameters>(graph, {
        id: `${id}-sum`,
        operation: OPERATION,
        segmentCount: polygonCount,
        segmentKeys: keys,
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
          {name: 'pairOverflow', view: pairs.status.overflow, type: 'u32', access: 'read'},
          {name: 'crossingOverflow', view: crossingOverflow, type: 'u32', access: 'read'},
          {name: 'overflow', view: output.status.overflow, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 1,
        body: `overflow[overflowOffset] = select(0u, 1u, pairOverflow[pairOverflowOffset] != 0u || crossingOverflow[crossingOverflowOffset] != 0u);`
      })
    );
    return nodes;
  }
}
