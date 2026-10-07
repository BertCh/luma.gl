// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GraphVectorView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {FIND_PATH_WGSL} from './gpu-outline-geometry';

const OPERATION = 'GPUOffsetCurve';

/** Number of float32 elements in a {@link GPUOffsetCurve} parameter buffer. */
export const GPU_OFFSET_CURVE_PARAMETER_LENGTH = 4;

/** Default mitre limit, matching `shapely.offset_curve`. */
export const GPU_OFFSET_CURVE_DEFAULT_MITRE_LIMIT = 5;

/** Default number of segments per quarter circle of round joins, matching `shapely`. */
export const GPU_OFFSET_CURVE_DEFAULT_QUAD_SEGMENTS = 8;

/** Corner treatment of {@link GPUOffsetCurve}, matching shapely `join_style`. */
export type GPUOffsetCurveJoinStyle = 'round' | 'mitre' | 'bevel';

/** Input geometry kind of {@link GPUOffsetCurve}. */
export type GPUOffsetCurveGeometryType =
  /** Open paths: the first and last vertex offset along their only segment. */
  | 'lines'
  /** Closed rings with an implicit closing edge (do not repeat the first vertex). */
  | 'rings';

/** CPU description of the per-frame parameters of {@link GPUOffsetCurve}. */
export type GPUOffsetCurveParameters = {
  /** Signed offset in position units: positive is the left of the path direction, negative the right. */
  distance: number;
  /** Mitre length over `abs(distance)` beyond which a mitre corner is clipped. Default 5, at least 1. */
  mitreLimit?: number;
};

/**
 * Packs {@link GPUOffsetCurveParameters} into the 4-element float32 layout
 * `[distance, mitreLimit, 0, 0]`.
 *
 * @param parameters Parameters to encode.
 * @param target Optional destination of at least 4 elements.
 * @throws If a value is not finite, `mitreLimit` is below 1, or `target` is too short.
 */
export function getGPUOffsetCurveParameterValues(
  parameters: GPUOffsetCurveParameters,
  target: Float32Array = new Float32Array(GPU_OFFSET_CURVE_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_OFFSET_CURVE_PARAMETER_LENGTH) {
    throw new Error(`Offset curve target must hold ${GPU_OFFSET_CURVE_PARAMETER_LENGTH} elements`);
  }
  const mitreLimit = parameters.mitreLimit ?? GPU_OFFSET_CURVE_DEFAULT_MITRE_LIMIT;
  if (!Number.isFinite(parameters.distance)) {
    throw new Error('Offset curve distance must be a finite number');
  }
  if (!Number.isFinite(mitreLimit) || mitreLimit < 1) {
    throw new Error('Offset curve mitreLimit must be a finite number of at least 1');
  }
  target.set([parameters.distance, mitreLimit, 0, 0]);
  return target;
}

/**
 * Number of output rows each input vertex owns: `2 * quadSegments + 1` for round joins (a corner
 * turns by at most half a circle), 2 for mitre and bevel joins.
 */
export function getGPUOffsetCurveRowsPerVertex(
  joinStyle: GPUOffsetCurveJoinStyle,
  quadSegments: number
): number {
  return joinStyle === 'round' ? 2 * quadSegments + 1 : 2;
}

/** Caller-owned outputs of {@link GPUOffsetCurve}. */
export type GPUOffsetCurveOutput = {
  /**
   * Offset points, exactly `positions.length * getGPUOffsetCurveRowsPerVertex(...)` rows. Input
   * vertex `i` owns rows `[i * k, (i + 1) * k)`; its first `counts[i]` rows are the offset points in
   * path order and any remaining rows repeat the last of them (zero-length, safe to draw).
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
export type GPUOffsetCurveProps = {
  /** Prefix for generated node IDs. Defaults to `'offset-curve'`. */
  id?: string;
  /** Packed planar positions. */
  positions: GraphDataView<'float32x2'>;
  /** `pathCount + 1` monotonic row offsets; path `p` owns rows `[pathOffsets[p], pathOffsets[p + 1])`. */
  pathOffsets: GraphDataView<'uint32'>;
  /** `'lines'` (default) or `'rings'`. */
  geometryType?: GPUOffsetCurveGeometryType;
  /** Corner treatment on the outside of a turn. Default `'round'`. */
  joinStyle?: GPUOffsetCurveJoinStyle;
  /** Segments per quarter circle of round joins, at least 1. Default 8, like shapely. */
  quadSegments?: number;
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
 * of the two offset segments; at path ends the plain normal offset. Positive distance offsets to the
 * left of the path direction, negative to the right. Planar coordinates only.
 *
 * **Render-grade, not GEOS-clean.** GEOS removes the loops that form when an offset segment is
 * shorter than the inset at an inside corner, and drops pieces closer than `distance` to the input.
 * This contributor does not: tight inside corners, distances larger than the local segment length
 * and concave regions can leave self-crossing loops. Away from loops the points equal
 * `shapely.offset_curve` (checked in the spec against GEOS, including mitre clipping and ring
 * closure). GEOS clamps `quad_segs` to at least 8 for offset curves, so compare round joins at
 * `quadSegments >= 8`; lower values here give coarser, cheaper arcs. On the outside of a ring GEOS
 * leaves the corner at the first vertex open (a quirk); this contributor joins it. Repeated vertices are treated as path ends on that side, so remove duplicates first,
 * and rings must not repeat the first vertex.
 *
 * Output size is fixed up front ({@link getGPUOffsetCurveRowsPerVertex} rows per input vertex), so
 * there is no capacity, overflow flag or scan; one invocation per vertex, no atomics, deterministic.
 * The distance and mitre limit are read per frame. Precision: f32 from the f32 positions.
 */
export class GPUOffsetCurve implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUOffsetCurveProps;
  /** Resolved geometry kind. */
  readonly geometryType: GPUOffsetCurveGeometryType;
  /** Resolved join style. */
  readonly joinStyle: GPUOffsetCurveJoinStyle;
  /** Resolved quarter-circle segment count. */
  readonly quadSegments: number;
  /** Output rows per input vertex. */
  readonly rowsPerVertex: number;

  constructor(props: GPUOffsetCurveProps) {
    this.id = props.id ?? 'offset-curve';
    this.props = props;
    this.geometryType = props.geometryType ?? 'lines';
    this.joinStyle = props.joinStyle ?? 'round';
    this.quadSegments = props.quadSegments ?? GPU_OFFSET_CURVE_DEFAULT_QUAD_SEGMENTS;
    this.rowsPerVertex = getGPUOffsetCurveRowsPerVertex(this.joinStyle, this.quadSegments);
    const {id} = this;
    for (const [name, view] of Object.entries({
      positions: props.positions,
      pathOffsets: props.pathOffsets,
      parameters: props.parameters,
      outputPositions: props.output.positions,
      outputCounts: props.output.counts
    })) {
      if ((view as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
      }
    }
    if (this.geometryType !== 'lines' && this.geometryType !== 'rings') {
      throw new Error(`${id} geometryType must be 'lines' or 'rings'`);
    }
    if (!['round', 'mitre', 'bevel'].includes(this.joinStyle)) {
      throw new Error(`${id} joinStyle must be 'round', 'mitre' or 'bevel'`);
    }
    if (!Number.isInteger(this.quadSegments) || this.quadSegments < 1) {
      throw new Error(`${id} quadSegments must be a positive integer`);
    }
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    if (props.positions.length < 1) {
      throw new Error(`${id} needs at least one position`);
    }
    validatePackedUint32View(props.pathOffsets, `${id} pathOffsets`);
    if (props.pathOffsets.length < 2) {
      throw new Error(`${id} pathOffsets must contain at least two rows`);
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
      validatePackedUint32View(props.output.counts, `${id} output.counts`);
      if (props.output.counts.length !== props.positions.length) {
        throw new Error(`${id} output.counts must hold ${props.positions.length} rows`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.output.positions, props.output.counts],
      [props.positions, props.pathOffsets, props.parameters]
    );
  }

  /** Returns the single generation node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.pathOffsets,
      props.parameters,
      props.output.positions,
      props.output.counts
    ]);
    const bindings: WGSLKernelBinding[] = [
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      {name: 'pathOffsets', view: props.pathOffsets, type: 'u32', access: 'read'},
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
        variant: `${this.geometryType}-${this.joinStyle}`,
        bindings,
        invocationCount: props.positions.length,
        declarations: /* wgsl */ `
const PATH_COUNT: u32 = ${props.pathOffsets.length - 1}u;
const ROW_COUNT: u32 = ${props.positions.length}u;
const ROWS_PER_VERTEX: u32 = ${this.rowsPerVertex}u;
const QUAD_SEGMENTS: f32 = ${this.quadSegments}.0;
const IS_RING: bool = ${this.geometryType === 'rings'};
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
${FIND_PATH_WGSL}`,
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
  let path = findPath(index);
  if (path != NO_PATH) {
    let pathStart = pathOffsets[pathOffsetsOffset + path];
    let pathEnd = min(pathOffsets[pathOffsetsOffset + path + 1u], ROW_COUNT);
    let pathLength = pathEnd - pathStart;
    var previousRow = ROW_COUNT;
    var nextRow = ROW_COUNT;
    if (index > pathStart) {
      previousRow = index - 1u;
    } else if (IS_RING && pathLength >= 3u) {
      previousRow = pathEnd - 1u;
    }
    if (index + 1u < pathEnd) {
      nextRow = index + 1u;
    } else if (IS_RING && pathLength >= 3u) {
      nextRow = pathStart;
    }
    if (previousRow < ROW_COUNT) {
      let delta = center - getPosition(previousRow);
      let deltaLength = length(delta);
      if (deltaLength > 0.0) {
        incoming = delta / deltaLength;
        hasIncoming = true;
      }
    }
    if (nextRow < ROW_COUNT) {
      let delta = getPosition(nextRow) - center;
      let deltaLength = length(delta);
      if (deltaLength > 0.0) {
        outgoing = delta / deltaLength;
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
      points[0] = center + (a + b) / (1.0 + cosTurn);
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
