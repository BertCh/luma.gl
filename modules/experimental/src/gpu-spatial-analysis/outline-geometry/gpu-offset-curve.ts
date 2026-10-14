// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import type {
  GPULineGeometryFields,
  GPUPolygonGeometryFields
} from '../../analysis-core/contracts/index';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {FIND_LINE_WGSL} from './gpu-outline-geometry';

/** Spatial contexts supported by this distance-measuring operation. */
const SPATIAL_CONTEXT_SUPPORT = Object.freeze({
  ...GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  measure: 'distance'
} as const);

const OPERATION = 'GPUOffsetCurve';

/** Number of float32 elements in a {@link GPUOffsetCurve} parameter buffer. */
export const GPU_OFFSET_CURVE_PARAMETER_LENGTH = 4;

/** Default mitre limit, matching `shapely.offset_curve`. */
export const GPU_OFFSET_CURVE_DEFAULT_MITRE_LIMIT = 5;

/** Default number of segments per quarter circle of round joins, matching `shapely`. */
export const GPU_OFFSET_CURVE_DEFAULT_QUAD_SEGMENTS = 8;

/**
 * Largest `quadSegments`. Each vertex keeps its `2 * quadSegments + 1` join points in a private
 * array, which must stay within the 8,192 B per-function limit of core WebGPU (513 x 8 B here).
 */
export const GPU_OFFSET_CURVE_MAXIMUM_QUAD_SEGMENTS = 256;

/** Corner treatment of {@link GPUOffsetCurve}, matching shapely `join_style`. */
export type GPUOffsetCurveJoinStyle = 'round' | 'mitre' | 'bevel';

/** Input geometry kind of {@link GPUOffsetCurve}. */
export type GPUOffsetCurveGeometryType =
  /** Open lines: the first and last vertex offset along their only segment. */
  | 'lines'
  /**
   * Polygon rings, each a closed ring with an implicit closing edge (do not repeat the first
   * vertex). Takes the four-level polygon layout; the ring count is preserved.
   */
  | 'polygons';

/** CPU description of the per-frame parameters of {@link GPUOffsetCurve}. */
export type GPUOffsetCurveSettings = {
  /** Signed offset in position units: positive is the left of the line direction, negative the right. */
  distance: number;
  /** Mitre length over `abs(distance)` beyond which a mitre corner is clipped. Default 5, at least 1. */
  mitreLimit?: number;
};

/**
 * Packs {@link GPUOffsetCurveSettings} into the 4-element float32 layout
 * `[distance, mitreLimit, 0, 0]`.
 *
 * @param settings Settings to encode.
 * @param target Optional destination of at least 4 elements.
 * @throws If a value is not finite, `mitreLimit` is below 1, or `target` is too short.
 */
export function getGPUOffsetCurveParameterValues(
  settings: GPUOffsetCurveSettings,
  target: Float32Array = new Float32Array(GPU_OFFSET_CURVE_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_OFFSET_CURVE_PARAMETER_LENGTH) {
    throw new Error(`Offset curve target must hold ${GPU_OFFSET_CURVE_PARAMETER_LENGTH} elements`);
  }
  const mitreLimit = settings.mitreLimit ?? GPU_OFFSET_CURVE_DEFAULT_MITRE_LIMIT;
  if (!Number.isFinite(settings.distance)) {
    throw new Error('Offset curve distance must be a finite number');
  }
  if (!Number.isFinite(mitreLimit) || mitreLimit < 1) {
    throw new Error('Offset curve mitreLimit must be a finite number of at least 1');
  }
  target.set([settings.distance, mitreLimit, 0, 0]);
  return target;
}

/**
 * Number of output rows each input vertex owns: `2 * quadSegments + 1` for round joins (a corner
 * turns by at most half a circle), 2 for mitre and bevel joins.
 */
export function getGPUOffsetCurveRowsPerVertex(
  joinStyle: GPUOffsetCurveJoinStyle,
  quadSegments: number,
  insideTurns: GPUOffsetCurveInsideTurns = 'intersection'
): number {
  // A buffer-style inside turn emits three points (offset end, vertex, offset start).
  const minimum = insideTurns === 'buffer' ? 3 : 2;
  return Math.max(joinStyle === 'round' ? 2 * quadSegments + 1 : 2, minimum);
}

/**
 * Inside-of-turn rule of {@link GPUOffsetCurve}.
 *
 * - `'intersection'` (default): the intersection of the two offset lines (`shapely.offset_curve`).
 * - `'buffer'`: the intersection only while it lies on both offset segments; otherwise the corner
 *   is routed through the input vertex (`a + dn1, a, a + dn2`), as the GEOS buffer builder does, so
 *   a raw buffer boundary keeps a winding of at least one everywhere inside the buffer instead of
 *   the inverted far-away corner a tight switchback produces.
 */
export type GPUOffsetCurveInsideTurns = 'intersection' | 'buffer';

/** Caller-owned outputs of {@link GPUOffsetCurve}. */
export type GPUOffsetCurveOutput = {
  /**
   * Offset points, exactly `positions.length * getGPUOffsetCurveRowsPerVertex(...)` rows. Input
   * vertex `i` owns rows `[i * k, (i + 1) * k)`; its first `counts[i]` rows are the offset points in
   * line order and any remaining rows repeat the last of them (zero-length, safe to draw).
   */
  positions: GraphDataView<'float32x2'>;
  /** Optional per-input-vertex number of distinct offset points, `positions.length` rows. */
  counts?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUOffsetCurve}.
 *
 * Per-frame (no recompile): the contents of `parameters` (signed distance, mitre limit) and every
 * input buffer. Compile-time: view lengths, `geometryType`, `joinStyle`, `quadSegments`.
 */
export type GPUOffsetCurveProps = (
  | ({
      /** Open lines (default). Takes `positions` and `lineOffsets`. */
      geometryType?: 'lines';
    } & GPULineGeometryFields)
  | ({
      /**
       * Closed rings with an implicit closing edge. Takes the four-level polygon layout
       * (`positions`, `featureOffsets`, `polygonOffsets`, `ringOffsets`); every ring is offset on
       * its own, so the ring count is preserved: output ring `r` owns the rows
       * `[ringOffsets[r] * k, ringOffsets[r + 1] * k)` and `featureOffsets` and `polygonOffsets`
       * apply to it unchanged.
       */
      geometryType: 'polygons';
    } & GPUPolygonGeometryFields)
) & {
  /** Prefix for generated node IDs. Defaults to `'offset-curve'`. */
  id?: string;
  /**
   * Planar only. Spatial context of `positions`; omitted means native planar coordinates.
   * Longitude/latitude (or any other non-planar) contexts throw at construction: project
   * positions to a planar frame first, for example with `createGPULocalFrame` and
   * `GPULocalFrameProjection`, and pass the frame's context so its `frame` identity is kept.
   */
  spatialContext?: GPUSpatialContext;
  /** Corner treatment on the outside of a turn. Default `'round'`. */
  joinStyle?: GPUOffsetCurveJoinStyle;
  /**
   * Segments per quarter circle of round joins, 1 to {@link GPU_OFFSET_CURVE_MAXIMUM_QUAD_SEGMENTS}.
   * Default 8, like shapely.
   */
  quadSegments?: number;
  /** Inside-of-turn rule, see {@link GPUOffsetCurveInsideTurns}. Default `'intersection'`. */
  insideTurns?: GPUOffsetCurveInsideTurns;
  /** Per-frame packed float32 view written with {@link getGPUOffsetCurveParameterValues}. */
  parameters: GraphDataView<'float32'>;
  /** Output points. */
  output: GPUOffsetCurveOutput;
};

/**
 * One-sided offset of lines and rings (`shapely.offset_curve`, GeoPandas `offset_curve`, geo
 * `OffsetCurve`-style raw offset).
 *
 * For every input vertex the kernel emits the offset corner: on the outside of a turn a round arc
 * (`quadSegments` per quarter circle, GEOS rounding rule), a mitre point (clipped like GEOS when
 * the mitre exceeds `mitreLimit`) or a bevel pair; on the inside of a turn the single intersection
 * of the two offset segments; at line ends the plain normal offset. Positive distance offsets to the
 * left of the line direction, negative to the right. Planar coordinates only.
 *
 * **Render-grade, not GEOS-clean.** GEOS removes the loops that form when an offset segment is
 * shorter than the inset at an inside corner, and drops pieces closer than `distance` to the input.
 * This contributor does not: tight inside corners, distances larger than the local segment length
 * and concave regions can leave self-crossing loops. Away from loops the points equal
 * `shapely.offset_curve` (checked in the spec against GEOS, including mitre clipping and ring
 * closure). GEOS clamps `quad_segs` to at least 8 for offset curves, so compare round joins at
 * `quadSegments >= 8`; lower values here give coarser, cheaper arcs. On the outside of a ring GEOS
 * leaves the corner at the first vertex open (a quirk); this contributor joins it. Repeated vertices are treated as line ends on that side, so remove duplicates first,
 * and rings must not repeat the first vertex.
 *
 * Output size is fixed up front ({@link getGPUOffsetCurveRowsPerVertex} rows per input vertex), so
 * there is no capacity, overflow flag or scan; one invocation per vertex, no atomics, deterministic.
 * The distance and mitre limit are read per frame. Precision: f32 from the f32 positions.
 *
 * Planar only: `spatialContext` defaults to native planar coordinates and a longitude/latitude
 * context throws at construction. Project first, for example with `createGPULocalFrame` and
 * `GPULocalFrameProjection`, and pass the frame's context so its `frame` identity is kept.
 */
export class GPUOffsetCurve implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUOffsetCurveProps;
  /** Resolved spatial context (planar). */
  readonly spatialContext: GPUSpatialContext;
  /** Resolved geometry kind. */
  readonly geometryType: GPUOffsetCurveGeometryType;
  /** Line offsets, or the polygon ring offsets, as read by the kernel. */
  readonly boundaryOffsets: GraphDataView<'uint32'>;
  /** Resolved join style. */
  readonly joinStyle: GPUOffsetCurveJoinStyle;
  /** Resolved quarter-circle segment count. */
  readonly quadSegments: number;
  /** Output rows per input vertex. */
  readonly rowsPerVertex: number;
  /** Resolved inside-of-turn rule. */
  readonly insideTurns: GPUOffsetCurveInsideTurns;

  constructor(props: GPUOffsetCurveProps) {
    this.id = props.id ?? 'offset-curve';
    this.props = props;
    this.spatialContext = resolveGPUSpatialContext(
      this.id,
      props.spatialContext,
      SPATIAL_CONTEXT_SUPPORT
    );
    this.geometryType = props.geometryType ?? 'lines';
    // Rings of a polygon are offset like closed lines, so the kernel reads them as line offsets.
    this.boundaryOffsets =
      props.geometryType === 'polygons' ? props.ringOffsets : props.lineOffsets;
    this.joinStyle = props.joinStyle ?? 'round';
    this.quadSegments = props.quadSegments ?? GPU_OFFSET_CURVE_DEFAULT_QUAD_SEGMENTS;
    this.insideTurns = props.insideTurns ?? 'intersection';
    this.rowsPerVertex = getGPUOffsetCurveRowsPerVertex(
      this.joinStyle,
      this.quadSegments,
      this.insideTurns
    );
    const {id} = this;
    if (this.insideTurns !== 'intersection' && this.insideTurns !== 'buffer') {
      throw new Error(`${id} insideTurns must be 'intersection' or 'buffer'`);
    }
    if (this.geometryType !== 'lines' && this.geometryType !== 'polygons') {
      throw new Error(`${id} geometryType must be 'lines' or 'polygons'`);
    }
    if (!['round', 'mitre', 'bevel'].includes(this.joinStyle)) {
      throw new Error(`${id} joinStyle must be 'round', 'mitre' or 'bevel'`);
    }
    validateCapacity(
      id,
      'quadSegments',
      this.quadSegments,
      1,
      GPU_OFFSET_CURVE_MAXIMUM_QUAD_SEGMENTS
    );
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    if (props.positions.length < 1) {
      throw new Error(`${id} needs at least one position`);
    }
    const offsetsName = props.geometryType === 'polygons' ? 'ringOffsets' : 'lineOffsets';
    validatePackedView(this.boundaryOffsets, ['uint32'], `${id} ${offsetsName}`);
    if (this.boundaryOffsets.length < 2) {
      throw new Error(`${id} ${offsetsName} needs two rows`);
    }
    if (props.geometryType === 'polygons') {
      validatePackedView(props.featureOffsets, ['uint32'], `${id} featureOffsets`);
      validatePackedView(props.polygonOffsets, ['uint32'], `${id} polygonOffsets`);
    }
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_OFFSET_CURVE_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_OFFSET_CURVE_PARAMETER_LENGTH} float32 values`
      );
    }
    validatePackedView(props.output.positions, ['float32x2'], `${id} output.positions`);
    const expectedLength = props.positions.length * this.rowsPerVertex;
    if (props.output.positions.length !== expectedLength) {
      throw new Error(`${id} output.positions must hold ${expectedLength} rows`);
    }
    if (expectedLength > 0x7fffffff) {
      throw new Error(`${id} output is too large`);
    }
    if (props.output.counts) {
      validatePackedView(props.output.counts, ['uint32'], `${id} output.counts`);
      if (props.output.counts.length !== props.positions.length) {
        throw new Error(`${id} output.counts must hold ${props.positions.length} rows`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.output.positions, props.output.counts],
      [props.positions, this.boundaryOffsets, props.parameters]
    );
  }

  /** Returns the single generation node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      this.boundaryOffsets,
      props.parameters,
      props.output.positions,
      props.output.counts
    ]);
    const bindings: WGSLKernelBinding[] = [
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      {name: 'lineOffsets', view: this.boundaryOffsets, type: 'u32', access: 'read'},
      {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
      {name: 'outputPositions', view: props.output.positions, type: 'f32', access: 'read_write'}
    ];
    if (props.output.counts) {
      bindings.push({
        name: 'outputCounts',
        view: props.output.counts,
        type: 'u32',
        access: 'read_write'
      });
    }
    const hasCounts = Boolean(props.output.counts);
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-generate`,
        operation: OPERATION,
        variant: `${this.geometryType}-${this.joinStyle}-${this.insideTurns}`,
        bindings,
        invocationCount: props.positions.length,
        declarations: /* wgsl */ `
const LINE_COUNT: u32 = ${this.boundaryOffsets.length - 1}u;
const ROW_COUNT: u32 = ${props.positions.length}u;
const ROWS_PER_VERTEX: u32 = ${this.rowsPerVertex}u;
const QUAD_SEGMENTS: f32 = ${this.quadSegments}.0;
const IS_RING: bool = ${this.geometryType === 'polygons'};
const HALF_PI: f32 = 1.5707963267948966;

fn getPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}

fn getLeftNormal(direction: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(-direction.y, direction.x);
}

fn rotateVector(v: vec2<f32>, angle: f32) -> vec2<f32> {
  let c = cos(angle);
  let s = sin(angle);
  return vec2<f32>(c * v.x - s * v.y, s * v.x + c * v.y);
}
${FIND_LINE_WGSL}`,
        body: /* wgsl */ `let offsetDistance = parameters[parametersOffset];
  let mitreLimit = max(parameters[parametersOffset + 1u], 1.0);
  let magnitude = abs(offsetDistance);
  let center = getPosition(index);
  var points = array<vec2<f32>, ${this.rowsPerVertex}>();
  var count = 1u;
  points[0] = center;

  var hasIncoming = false;
  var hasOutgoing = false;
  var incoming = vec2<f32>(0.0);
  var outgoing = vec2<f32>(0.0);
  var incomingLength = 0.0;
  var outgoingLength = 0.0;
  let line = findLine(index);
  if (line != NO_LINE) {
    let lineStart = lineOffsets[lineOffsetsOffset + line];
    let lineEnd = min(lineOffsets[lineOffsetsOffset + line + 1u], ROW_COUNT);
    let lineLength = lineEnd - lineStart;
    var previousRow = ROW_COUNT;
    var nextRow = ROW_COUNT;
    if (index > lineStart) {
      previousRow = index - 1u;
    } else if (IS_RING && lineLength >= 3u) {
      previousRow = lineEnd - 1u;
    }
    if (index + 1u < lineEnd) {
      nextRow = index + 1u;
    } else if (IS_RING && lineLength >= 3u) {
      nextRow = lineStart;
    }
    if (previousRow < ROW_COUNT) {
      let delta = center - getPosition(previousRow);
      let deltaLength = length(delta);
      if (deltaLength > 0.0) {
        incoming = delta / deltaLength;
        incomingLength = deltaLength;
        hasIncoming = true;
      }
    }
    if (nextRow < ROW_COUNT) {
      let delta = getPosition(nextRow) - center;
      let deltaLength = length(delta);
      if (deltaLength > 0.0) {
        outgoing = delta / deltaLength;
        outgoingLength = deltaLength;
        hasOutgoing = true;
      }
    }
  }

  if (hasIncoming && !hasOutgoing) {
    points[0] = center + getLeftNormal(incoming) * offsetDistance;
  } else if (!hasIncoming && hasOutgoing) {
    points[0] = center + getLeftNormal(outgoing) * offsetDistance;
  } else if (hasIncoming && hasOutgoing) {
    let normalIn = getLeftNormal(incoming);
    let normalOut = getLeftNormal(outgoing);
    let a = normalIn * offsetDistance;
    let b = normalOut * offsetDistance;
    let turn = incoming.x * outgoing.y - incoming.y * outgoing.x;
    let cosTurn = clamp(dot(incoming, outgoing), -1.0, 1.0);
    if (turn * offsetDistance > 0.0 && 1.0 + cosTurn > 1e-6) {
      // Inside of the turn: intersection of the two offset segments.
      ${getInsideTurnSource(this.insideTurns)}${
        this.insideTurns === 'buffer'
          ? `
    } else if (turn * offsetDistance > 0.0) {
      // A full reversal on the inside: route through the vertex.
      points[0] = center + a;
      points[1] = center;
      points[2] = center + b;
      count = 3u;`
          : ''
      }
    } else if (length(a - b) < magnitude * 1e-3) {
      points[0] = center + a;
    } else {
      ${getJoinSource(this.joinStyle)}
    }
  }

  let base = index * ROWS_PER_VERTEX;
  for (var slot = 0u; slot < ROWS_PER_VERTEX; slot++) {
    let point = points[min(slot, count - 1u)];
    let at = outputPositionsOffset + 2u * (base + slot);
    outputPositions[at] = point.x;
    outputPositions[at + 1u] = point.y;
  }
  ${hasCounts ? 'outputCounts[outputCountsOffset + index] = count;' : ''}`
      })
    ];
  }
}

/** WGSL statements for the inside-of-turn corner; set `points` and `count`. @internal */
function getInsideTurnSource(insideTurns: GPUOffsetCurveInsideTurns): string {
  if (insideTurns === 'intersection') {
    return 'points[0] = center + (a + b) / (1.0 + cosTurn);';
  }
  // The intersection lies |d| tan(theta / 2) back along each segment (theta the turn angle); where
  // that overshoots either segment the corner goes through the vertex (GEOS buffer rule).
  return `let inset = magnitude * sqrt(max(1.0 - cosTurn, 0.0) / (1.0 + cosTurn));
      if (inset > incomingLength || inset > outgoingLength) {
        points[0] = center + a;
        points[1] = center;
        points[2] = center + b;
        count = 3u;
      } else {
        points[0] = center + (a + b) / (1.0 + cosTurn);
      }`;
}

/** WGSL statements for the outside-of-turn corner; set `points` and `count`. @internal */
function getJoinSource(joinStyle: GPUOffsetCurveJoinStyle): string {
  if (joinStyle === 'bevel') {
    return `points[0] = center + a;
      points[1] = center + b;
      count = 2u;`;
  }
  if (joinStyle === 'round') {
    // GEOS: round(turnAngle / (pi / 2 / quadSegments)) uniform steps, both ends included.
    return `let turnAngle = acos(cosTurn);
      let steps = u32(floor(turnAngle * QUAD_SEGMENTS / HALF_PI + 0.5));
      let direction = select(1.0, -1.0, offsetDistance > 0.0);
      points[0] = center + a;
      if (steps >= 1u) {
        for (var arcStep = 1u; arcStep < steps; arcStep++) {
          points[arcStep] = center + rotateVector(a, direction * turnAngle * f32(arcStep) / f32(steps));
        }
        points[steps] = center + b;
        count = steps + 1u;
      } else {
        points[1] = center + b;
        count = 2u;
      }`;
  }
  return `let cosHalf = sqrt(max(0.5 * (1.0 + cosTurn), 0.0));
      if (cosHalf > 0.0 && 1.0 / cosHalf <= mitreLimit) {
        points[0] = center + (a + b) / (1.0 + cosTurn);
      } else {
        // Clipped mitre: the offset lines cut by the line perpendicular to the bisector at mitreLimit * distance.
        var bisector = a + b;
        if (length(bisector) < 1e-6 * magnitude) {
          bisector = incoming * magnitude;
        }
        let tipDirection = normalize(bisector);
        let clipMiddle = center + tipDirection * (mitreLimit * magnitude);
        let halfLength = magnitude * (1.0 - mitreLimit * cosHalf) / sqrt(max(1.0 - cosHalf * cosHalf, 1e-12));
        let perpendicular = getLeftNormal(tipDirection) * select(-1.0, 1.0, dot(a, getLeftNormal(tipDirection)) >= 0.0);
        points[0] = clipMiddle + perpendicular * halfLength;
        points[1] = clipMiddle - perpendicular * halfLength;
        count = 2u;
      }`;
}
