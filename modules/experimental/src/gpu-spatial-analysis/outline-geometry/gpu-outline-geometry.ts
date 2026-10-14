// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPUSphereRadius,
  resolveGPUSpatialContext,
  type GPUSpatialContext,
  type GPUSpatialContextSupport
} from '../../analysis-core/contracts/spatial-context';
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
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';

const OPERATION = 'GPUOutlineGeometry';

/** Planar coordinates, or longitude/latitude with a sphere (great-circle) metric. */
const SPATIAL_CONTEXT_SUPPORT: GPUSpatialContextSupport = Object.freeze({
  coordinateSpaces: Object.freeze(['planar', 'longitude-latitude'] as const),
  metrics: Object.freeze(['native', 'great-circle'] as const)
});

/** Number of float32 elements in a {@link GPUOutlineGeometry} parameter buffer. */
export const GPU_OUTLINE_GEOMETRY_PARAMETER_LENGTH = 4;

/** Default number of triangles in the round join (and cap) disc of one vertex. */
export const GPU_OUTLINE_GEOMETRY_DEFAULT_JOIN_SEGMENTS = 16;

/** Input geometry kind of {@link GPUOutlineGeometry}. */
export type GPUOutlineGeometryType =
  /** Every position is a point; the outline is a disc. */
  | 'points'
  /** Open lines; the outline is a round-capped stroke. */
  | 'lines'
  /** Polygon rings (the closing edge is implicit); the outline is a round-joined stroke. */
  | 'polygons';

/** CPU description of the per-frame parameters of {@link GPUOutlineGeometry}. */
export type GPUOutlineGeometrySettings = {
  /**
   * Offset distance, in position units (planar context) or in the context's `units` on its sphere
   * (longitude/latitude context, meters by default). Must be non-negative; `0` draws nothing visible.
   */
  distance: number;
};

/**
 * Packs {@link GPUOutlineGeometrySettings} into the 4-element float32 layout
 * `[distance, 0, 0, 0]`. Write the result into a `GPUParameterBuffer` between encodings to change
 * the distance without recompiling.
 *
 * @param settings Settings to encode.
 * @param target Optional destination of at least 4 elements.
 * @throws If the distance is negative or not finite, or `target` is too short.
 */
export function getGPUOutlineGeometryParameterValues(
  settings: GPUOutlineGeometrySettings,
  target: Float32Array = new Float32Array(GPU_OUTLINE_GEOMETRY_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_OUTLINE_GEOMETRY_PARAMETER_LENGTH) {
    throw new Error(
      `Outline geometry target must hold ${GPU_OUTLINE_GEOMETRY_PARAMETER_LENGTH} elements`
    );
  }
  if (!Number.isFinite(settings.distance) || settings.distance < 0) {
    throw new Error('Outline geometry distance must be a non-negative finite number');
  }
  target.set([settings.distance, 0, 0, 0]);
  return target;
}

/**
 * Number of output vertices each input vertex owns: `joinSegments` disc triangles plus two
 * triangles of the quad toward the next vertex, three vertices each.
 */
export function getGPUOutlineGeometryVerticesPerInput(joinSegments: number): number {
  return 3 * (joinSegments + 2);
}

/** Caller-owned outputs of {@link GPUOutlineGeometry}. */
export type GPUOutlineGeometryOutput = {
  /**
   * Triangle-list vertices, exactly `positions.length * getGPUOutlineGeometryVerticesPerInput(joinSegments)`
   * rows. Input vertex `i` owns rows `[i * k, (i + 1) * k)` with `k` the per-input vertex count.
   * Unused triangles are degenerate (every vertex at the input vertex), so draw the whole buffer.
   */
  positions: GraphDataView<'float32x2'>;
};

/**
 * Properties for {@link GPUOutlineGeometry}.
 *
 * Per-frame (no recompile): the contents of `parameters` (distance) and every input buffer.
 * Compile-time: view lengths, `geometryType`, `spatialContext`, `joinSegments`.
 */
export type GPUOutlineGeometryProps = (
  | {
      /** Every position is a point. Takes only `positions`. */
      geometryType: 'points';
      /** Packed positions: planar coordinates, or longitude/latitude degrees for a sphere context. */
      positions: GraphDataView<'float32x2'>;
    }
  | ({
      /** Open lines. Takes `positions` and `lineOffsets`. */
      geometryType: 'lines';
    } & GPULineGeometryFields)
  | ({
      /**
       * Polygon rings, each stroked as a closed ring. Takes the four-level polygon layout
       * (`positions`, `featureOffsets`, `polygonOffsets`, `ringOffsets`). The output has one
       * block of vertices per input vertex, so the ring structure is that of the input.
       */
      geometryType: 'polygons';
    } & GPUPolygonGeometryFields)
) & {
  /** Prefix for generated node IDs. Defaults to `'outline-geometry'`. */
  id?: string;
  /**
   * Spatial context of `positions` and of the offset `distance`. Omitted means native planar
   * coordinates with distances in position units. `{coordinateSpace: 'longitude-latitude', metric:
   * 'great-circle'}` makes positions longitude/latitude degrees and distances meters (or the
   * context's `units`) on a sphere of `sphereRadius` (default the mean Earth radius), applied with
   * the local east/north scale of each vertex: an equirectangular approximation, accurate for
   * distances small against the radius and wrong near the poles and across the antimeridian.
   * This replaces the removed `coordinateSystem` and `radius` props. Other contexts throw.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * Triangles in each round join and cap disc, at least 3. Default
   * {@link GPU_OUTLINE_GEOMETRY_DEFAULT_JOIN_SEGMENTS}. The disc is inscribed, so the picture is
   * up to `distance * (1 - cos(pi / joinSegments))` narrower than the true offset (1.9% at 16).
   */
  joinSegments?: number;
  /** Per-frame packed float32 view written with {@link getGPUOutlineGeometryParameterValues}. */
  parameters: GraphDataView<'float32'>;
  /** Output triangles. */
  output: GPUOutlineGeometryOutput;
};

/**
 * Builds render-only buffer geometry for points, lines and rings: for each input vertex a round
 * join disc and a rectangle toward the next vertex, as non-indexed triangles ready to draw
 * (turf `buffer` and `lineOffset` for the picture only).
 *
 * **Picture only.** Triangles of neighboring vertices overlap, and nothing is unioned or
 * dissolved, so the geometry is correct for drawing opaque or stencil-combined fills but wrong for
 * area, overlay, selection or point-in-polygon queries on the buffered shape. Use
 * `GPUBufferSelection` and `dwithin` predicates for queries, and `GPUDistanceField` for a unioned
 * look. Translucent fills show overlaps as darker bands; draw into a stencil or a mask first.
 * Self-intersections and holes are not interpreted. No negative buffers and no one-sided
 * (`lineOffset`) strokes; both are open.
 *
 * Output size is fixed up front: every input vertex owns exactly
 * {@link getGPUOutlineGeometryVerticesPerInput} vertices, so there is no capacity, no overflow flag and no
 * scan. The kernel runs one invocation per input vertex with no atomics and is deterministic. The
 * distance is read per frame. Zero-length segments and missing next vertices (line ends, points)
 * emit a degenerate quad. Triangles are counter-clockwise for a positive distance.
 *
 * Precision: f32 from the f32 positions.
 */
export class GPUOutlineGeometry implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUOutlineGeometryProps;
  /** Resolved spatial context (planar, or longitude/latitude with a great-circle metric). */
  readonly spatialContext: GPUSpatialContext;
  /** Sphere radius in the context's length unit; only used by longitude/latitude contexts. */
  readonly radius: number;
  /** Resolved join disc triangle count. */
  readonly joinSegments: number;
  /** Output vertices per input vertex. */
  readonly verticesPerInput: number;
  /** Line offsets or polygon ring offsets as read by the kernel; absent for points. */
  readonly boundaryOffsets?: GraphDataView<'uint32'>;

  constructor(props: GPUOutlineGeometryProps) {
    this.id = props.id ?? 'outline-geometry';
    this.props = props;
    this.spatialContext = resolveGPUSpatialContext(
      this.id,
      props.spatialContext,
      SPATIAL_CONTEXT_SUPPORT
    );
    this.radius = getGPUSphereRadius(this.spatialContext);
    this.joinSegments = props.joinSegments ?? GPU_OUTLINE_GEOMETRY_DEFAULT_JOIN_SEGMENTS;
    this.verticesPerInput = getGPUOutlineGeometryVerticesPerInput(this.joinSegments);
    const {id} = this;
    this.boundaryOffsets =
      props.geometryType === 'lines'
        ? props.lineOffsets
        : props.geometryType === 'polygons'
          ? props.ringOffsets
          : undefined;
    if (!['points', 'lines', 'polygons'].includes(props.geometryType)) {
      throw new Error(`${id} geometryType must be 'points', 'lines' or 'polygons'`);
    }
    validateCapacity(id, 'joinSegments', this.joinSegments, 3);
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    if (props.positions.length < 1) {
      throw new Error(`${id} needs at least one position`);
    }
    if (props.geometryType !== 'points') {
      const offsetsName = props.geometryType === 'lines' ? 'lineOffsets' : 'ringOffsets';
      if (!this.boundaryOffsets) {
        throw new Error(`${id} ${offsetsName} is required`);
      }
      validatePackedView(this.boundaryOffsets, ['uint32'], `${id} ${offsetsName}`);
      if (this.boundaryOffsets.length < 2) {
        throw new Error(`${id} ${offsetsName} needs two rows`);
      }
      if (props.geometryType === 'polygons') {
        validatePackedView(props.featureOffsets, ['uint32'], `${id} featureOffsets`);
        validatePackedView(props.polygonOffsets, ['uint32'], `${id} polygonOffsets`);
      }
    }
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_OUTLINE_GEOMETRY_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_OUTLINE_GEOMETRY_PARAMETER_LENGTH} float32 values`
      );
    }
    validatePackedView(props.output.positions, ['float32x2'], `${id} output.positions`);
    const expectedLength = props.positions.length * this.verticesPerInput;
    if (props.output.positions.length !== expectedLength) {
      throw new Error(`${id} output.positions must hold ${expectedLength} rows`);
    }
    if (expectedLength > 0x7fffffff) {
      throw new Error(`${id} output is too large`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.output.positions],
      [props.positions, this.boundaryOffsets, props.parameters]
    );
  }

  /** Returns the single generation node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, joinSegments} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      this.boundaryOffsets,
      props.parameters,
      props.output.positions
    ]);
    const hasLines = props.geometryType !== 'points';
    const bindings: WGSLKernelBinding[] = [
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'}
    ];
    if (hasLines && this.boundaryOffsets) {
      bindings.push({name: 'lineOffsets', view: this.boundaryOffsets, type: 'u32', access: 'read'});
    }
    bindings.push(
      {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
      {name: 'outputPositions', view: props.output.positions, type: 'f32', access: 'read_write'}
    );
    const spherical = this.spatialContext.coordinateSpace === 'longitude-latitude';
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-generate`,
        operation: OPERATION,
        variant: props.geometryType,
        bindings,
        invocationCount: props.positions.length,
        declarations: /* wgsl */ `
const LINE_COUNT: u32 = ${this.boundaryOffsets ? this.boundaryOffsets.length - 1 : 0}u;
const ROW_COUNT: u32 = ${props.positions.length}u;
const JOIN_SEGMENTS: u32 = ${joinSegments}u;
const VERTICES_PER_INPUT: u32 = ${this.verticesPerInput}u;
const RADIUS: f32 = ${getWGSLFloatLiteral(this.radius)};
const DEGREES_PER_RADIAN: f32 = 57.29577951308232;
const TAU: f32 = 6.283185307179586;

fn getPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}

fn writeVertex(slot: u32, base: u32, p: vec2<f32>) {
  let at = outputPositionsOffset + 2u * (base + slot);
  outputPositions[at] = p.x;
  outputPositions[at + 1u] = p.y;
}

// Moves p by distance d along the metric direction (east, north).
fn offsetPoint(p: vec2<f32>, direction: vec2<f32>, d: f32) -> vec2<f32> {
  ${
    spherical
      ? `let latitudeScale = d / RADIUS * DEGREES_PER_RADIAN;
  let longitudeScale = latitudeScale / max(cos(radians(p.y)), 1e-6);
  return p + vec2<f32>(direction.x * longitudeScale, direction.y * latitudeScale);`
      : 'return p + direction * d;'
  }
}

// Unit metric direction of the segment a to b, or zero for a zero-length segment.
fn getDirection(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  ${
    spherical
      ? `let middleLatitude = radians(0.5 * (a.y + b.y));
  let metric = vec2<f32>((b.x - a.x) * cos(middleLatitude), b.y - a.y);`
      : 'let metric = b - a;'
  }
  let metricLength = length(metric);
  if (metricLength <= 0.0) {
    return vec2<f32>(0.0);
  }
  return metric / metricLength;
}
${hasLines ? FIND_LINE_WGSL : ''}`,
        body: /* wgsl */ `let distance = max(parameters[parametersOffset], 0.0);
  let base = index * VERTICES_PER_INPUT;
  let center = getPosition(index);
  // Round join and cap disc: JOIN_SEGMENTS counter-clockwise triangles around the vertex.
  // Each rim point is shared by two triangles, so it is computed once (one sin/cos per segment).
  var rimA = offsetPoint(center, vec2<f32>(1.0, 0.0), distance);
  for (var k = 0u; k < JOIN_SEGMENTS; k++) {
    let angleB = TAU * f32(k + 1u) / f32(JOIN_SEGMENTS);
    let rimB = offsetPoint(center, vec2<f32>(cos(angleB), sin(angleB)), distance);
    writeVertex(3u * k, base, center);
    writeVertex(3u * k + 1u, base, rimA);
    writeVertex(3u * k + 2u, base, rimB);
    rimA = rimB;
  }
  // Rectangle toward the next vertex, or a degenerate pair of triangles.
  var nextRow = ROW_COUNT;
  ${hasLines ? getNextRowSource(props.geometryType) : ''}
  var quad = array<vec2<f32>, 6>(center, center, center, center, center, center);
  if (nextRow < ROW_COUNT) {
    let next = getPosition(nextRow);
    let direction = getDirection(center, next);
    if (dot(direction, direction) > 0.0) {
      let normal = vec2<f32>(-direction.y, direction.x);
      let aLeft = offsetPoint(center, normal, distance);
      let aRight = offsetPoint(center, -normal, distance);
      let bRight = offsetPoint(next, -normal, distance);
      let bLeft = offsetPoint(next, normal, distance);
      quad = array<vec2<f32>, 6>(aLeft, aRight, bRight, aLeft, bRight, bLeft);
    }
  }
  for (var vertex = 0u; vertex < 6u; vertex++) {
    writeVertex(3u * JOIN_SEGMENTS + vertex, base, quad[vertex]);
  }`
      })
    ];
  }
}

/**
 * WGSL `findLine(row) -> u32`: upper-bound binary search over the bound `lineOffsets`.
 *
 * @internal
 */
export const FIND_LINE_WGSL = /* wgsl */ `
const NO_LINE: u32 = 0xffffffffu;

fn findLine(row: u32) -> u32 {
  var low = 0u;
  var high = LINE_COUNT + 1u;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (lineOffsets[lineOffsetsOffset + middle] <= row) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  if (low == 0u || low > LINE_COUNT) {
    return NO_LINE;
  }
  return low - 1u;
}
`;

/** WGSL statements that set `nextRow` for lines or rings. @internal */
function getNextRowSource(geometryType: GPUOutlineGeometryType): string {
  return `let line = findLine(index);
  if (line != NO_LINE) {
    let lineStart = lineOffsets[lineOffsetsOffset + line];
    let lineEnd = min(lineOffsets[lineOffsetsOffset + line + 1u], ROW_COUNT);
    if (index + 1u < lineEnd) {
      nextRow = index + 1u;
    }${
      geometryType === 'polygons'
        ? ` else if (lineEnd - lineStart >= 2u) {
      nextRow = lineStart;
    }`
        : ''
    }
  }`;
}
