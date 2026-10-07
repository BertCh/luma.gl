// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
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

const OPERATION = 'GPUShapeGenerator';

/** Number of float32 elements in a {@link GPUShapeGenerator} parameter buffer. */
export const GPU_SHAPE_GENERATOR_PARAMETER_LENGTH = 2;

/** Mean Earth radius in meters, the value turf uses for `units: 'meters'`. */
export const GPU_SHAPE_GENERATOR_EARTH_RADIUS = 6371008.8;

/**
 * Kind of ring produced by {@link GPUShapeGenerator}.
 *
 * - `'circle'`: closed ring of `segmentCount + 1` vertices, counter-clockwise (turf `circle`).
 * - `'sector'`: pie slice `[center, arc..., center]` of `segmentCount + 3` vertices, the arc
 *   running clockwise from the first to the second bearing (turf `sector`).
 * - `'ellipse'`: closed ring of `segmentCount + 1` vertices, counter-clockwise (turf `ellipse`).
 */
export type GPUShapeType = 'circle' | 'sector' | 'ellipse';

/**
 * How radii are measured.
 *
 * - `'planar'`: centers and radii share the coordinate units; bearings are degrees clockwise from
 *   +Y.
 * - `'geodesic'`: centers are `[longitude, latitude]` degrees, radii are meters on a sphere of
 *   {@link GPU_SHAPE_GENERATOR_EARTH_RADIUS}, bearings are degrees clockwise from north (turf
 *   `destination`).
 */
export type GPUShapeCoordinateSystem = 'planar' | 'geodesic';

/** CPU description of the per-frame parameters of {@link GPUShapeGenerator}. */
export type GPUShapeGeneratorParameters = {
  /**
   * Segments (arc intervals) per shape this frame, clamped on the GPU to
   * `[minimum, maximumSegments]` (minimum 3 for circles and ellipses, 1 for sectors).
   */
  segmentCount: number;
  /** Multiplier applied to every radius. Defaults to 1. */
  radiusScale?: number;
};

/**
 * Packs {@link GPUShapeGeneratorParameters} into the 2-element float32 layout
 * `[segmentCount, radiusScale]`. Write the result into a `GPUParameterBuffer` between encodings.
 *
 * @param parameters Parameters to encode.
 * @param target Optional destination of at least 2 elements.
 * @throws If a value is not finite, `segmentCount < 1`, `radiusScale < 0` or `target` is short.
 */
export function getGPUShapeGeneratorParameterValues(
  parameters: GPUShapeGeneratorParameters,
  target: Float32Array = new Float32Array(GPU_SHAPE_GENERATOR_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_SHAPE_GENERATOR_PARAMETER_LENGTH) {
    throw new Error(
      `Shape generator target must hold ${GPU_SHAPE_GENERATOR_PARAMETER_LENGTH} elements`
    );
  }
  const {segmentCount, radiusScale = 1} = parameters;
  if (!Number.isFinite(segmentCount) || !Number.isFinite(radiusScale)) {
    throw new Error('Shape generator parameters must be finite');
  }
  if (segmentCount < 1 || radiusScale < 0) {
    throw new Error('Shape generator segmentCount must be at least 1 and radiusScale >= 0');
  }
  target.set([Math.floor(segmentCount), radiusScale]);
  return target;
}

/** Smallest segment count of a shape type. */
export function getGPUShapeMinimumSegments(shape: GPUShapeType): number {
  return shape === 'sector' ? 1 : 3;
}

/** Vertices per shape for a segment count (already clamped to the shape's range). */
export function getGPUShapeVertexCount(shape: GPUShapeType, segmentCount: number): number {
  return shape === 'sector' ? segmentCount + 3 : segmentCount + 1;
}

/** Caller-owned outputs of {@link GPUShapeGenerator}. */
export type GPUShapeGeneratorOutput = {
  /**
   * Ring vertices. Feature `f` owns rows `[f * V, (f + 1) * V)` where `V` is
   * {@link getGPUShapeVertexCount} of the clamped per-frame segment count; rows from
   * `featureCount * V` on are not written. Capacity must hold
   * `featureCount * getGPUShapeVertexCount(shape, maximumSegments)` rows.
   */
  positions: GraphDataView<'float32x2'>;
  /**
   * GeoArrow-style ring offsets, `featureCount + 1` rows: `offsets[f] = f * V`. With one ring per
   * feature these are both the ring and the polygon offsets.
   */
  offsets: GraphDataView<'uint32'>;
  /** Optional one-row total vertex count `featureCount * V` of the current frame. */
  vertexCount?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUShapeGenerator}.
 *
 * Per-frame (no recompile): `parameters` (segment count, radius scale) and the contents of the
 * input columns. Compile-time: `shape`, `coordinateSystem`, `ellipseSpacing`, `maximumSegments`,
 * the feature count and which optional inputs and outputs are present.
 */
export type GPUShapeGeneratorProps = {
  /** Prefix for generated node IDs. Defaults to `'shape-generator'`. */
  id?: string;
  /** Kind of ring. */
  shape: GPUShapeType;
  /** How radii and bearings are interpreted. Defaults to `'planar'`. */
  coordinateSystem?: GPUShapeCoordinateSystem;
  /**
   * Ellipse vertex spacing. Defaults to `'parameter'` (equal angles). `'arc-length'` uses
   * approximately equal distances along the unprojected ellipse, before rotation and spherical
   * destination. Uses a 256-interval quarter-ellipse lookup table (1028 scratch bytes per feature).
   * Only valid for `'ellipse'`; this choice is compile time.
   */
  ellipseSpacing?: 'parameter' | 'arc-length';
  /** Largest segment count the parameter buffer may request (compile time). Defaults to 64. */
  maximumSegments?: number;
  /** One center per feature: planar `[x, y]` or geodesic `[longitude, latitude]` degrees. */
  centers: GraphDataView<'float32x2'>;
  /**
   * Radii: one `float32` per feature for `'circle'` and `'sector'`; `float32x2` semi-axes
   * `[x, y]` for `'ellipse'`. Non-finite or negative radii become zero. One zero ellipse axis
   * produces a line; two zero axes (or a zero circle/sector radius) collapse to the center.
   */
  radii: GraphDataView<'float32'> | GraphDataView<'float32x2'>;
  /** `'sector'` only: `[startBearing, endBearing]` degrees per feature. */
  bearings?: GraphDataView<'float32x2'>;
  /** `'ellipse'` only: rotation in degrees clockwise per feature. Defaults to 0. */
  rotations?: GraphDataView<'float32'>;
  /** Per-frame packed float32 view written with {@link getGPUShapeGeneratorParameterValues}. */
  parameters: GraphDataView<'float32'>;
  /** Output rings. */
  output: GPUShapeGeneratorOutput;
};

/**
 * Generates circle, sector and ellipse rings around per-feature centers (turf `circle`, `sector`,
 * `ellipse`; the "buffer a point" workload), one invocation per output vertex.
 *
 * The segment count is a per-frame parameter up to the compile-time `maximumSegments`, so a
 * slider changes ring smoothness without recompiling; every feature uses the same count, so ring
 * offsets are `f * V` and positions are packed with no gaps. Planar rings offset the center by
 * `radius * [sin bearing, cos bearing]`; geodesic rings apply the spherical `destination`
 * formula turf uses. No atomics; deterministic.
 *
 * Differences from turf: vertices are evenly spaced in bearing (circle, sector arc) or, by default,
 * in the ellipse parameter. Use `ellipseSpacing: 'arc-length'` for approximately equal ellipse arc
 * intervals (turf uses arc length); circles run counter-clockwise
 * from north like turf; a sector whose bearings coincide sweeps a full turn but keeps its center
 * spokes (turf returns a circle) so that every feature has the same vertex count; ellipse
 * `rotations` are applied about the center.
 *
 * Precision: f32. Geodesic output coordinates are f32 degrees, a rounding of about 0.5 m at
 * mid-latitudes regardless of radius; there is no antimeridian wrapping.
 */
export class GPUShapeGenerator implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUShapeGeneratorProps;
  /** Number of features. */
  readonly featureCount: number;
  /** Compile-time maximum segment count. */
  readonly maximumSegments: number;

  constructor(props: GPUShapeGeneratorProps) {
    const id = props.id ?? 'shape-generator';
    this.id = id;
    this.props = props;
    if (props.shape !== 'circle' && props.shape !== 'sector' && props.shape !== 'ellipse') {
      throw new Error(`${id} unknown shape ${String(props.shape)}`);
    }
    if (
      props.coordinateSystem !== undefined &&
      props.coordinateSystem !== 'planar' &&
      props.coordinateSystem !== 'geodesic'
    ) {
      throw new Error(`${id} coordinateSystem must be 'planar' or 'geodesic'`);
    }
    if (
      props.ellipseSpacing !== undefined &&
      (props.shape !== 'ellipse' ||
        (props.ellipseSpacing !== 'parameter' && props.ellipseSpacing !== 'arc-length'))
    ) {
      throw new Error(`${id} ellipseSpacing must be 'parameter' or 'arc-length' for ellipses`);
    }
    this.maximumSegments = props.maximumSegments ?? 64;
    const minimum = getGPUShapeMinimumSegments(props.shape);
    if (
      !Number.isSafeInteger(this.maximumSegments) ||
      this.maximumSegments < minimum ||
      this.maximumSegments > 65536
    ) {
      throw new Error(`${id} maximumSegments must be an integer in [${minimum}, 65536]`);
    }
    for (const [name, view] of Object.entries({
      centers: props.centers,
      radii: props.radii,
      bearings: props.bearings,
      rotations: props.rotations,
      parameters: props.parameters,
      ...props.output
    })) {
      if ((view as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
      }
    }
    validatePackedView(props.centers, ['float32x2'], `${id} centers`);
    this.featureCount = props.centers.length;
    if (this.featureCount < 1) {
      throw new Error(`${id} centers must hold at least one row`);
    }
    validatePackedView(
      props.radii,
      [props.shape === 'ellipse' ? 'float32x2' : 'float32'],
      `${id} radii`
    );
    if (props.radii.length !== this.featureCount) {
      throw new Error(`${id} radii must have one row per center`);
    }
    if (props.shape === 'sector') {
      if (!props.bearings) {
        throw new Error(`${id} sector shapes need bearings`);
      }
      validatePackedView(props.bearings, ['float32x2'], `${id} bearings`);
      if (props.bearings.length !== this.featureCount) {
        throw new Error(`${id} bearings must have one row per center`);
      }
    } else if (props.bearings) {
      throw new Error(`${id} ${props.shape} shapes have no bearings`);
    }
    if (props.rotations) {
      if (props.shape !== 'ellipse') {
        throw new Error(`${id} only ellipse shapes have rotations`);
      }
      validatePackedView(props.rotations, ['float32'], `${id} rotations`);
      if (props.rotations.length !== this.featureCount) {
        throw new Error(`${id} rotations must have one row per center`);
      }
    }
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_SHAPE_GENERATOR_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_SHAPE_GENERATOR_PARAMETER_LENGTH} float32 values`
      );
    }
    const {positions, offsets, vertexCount} = props.output;
    validatePackedView(positions, ['float32x2'], `${id} output.positions`);
    const capacity = this.featureCount * getGPUShapeVertexCount(props.shape, this.maximumSegments);
    if (positions.length < capacity) {
      throw new Error(`${id} output.positions must hold at least ${capacity} rows`);
    }
    validatePackedUint32View(offsets, `${id} output.offsets`);
    if (offsets.length !== this.featureCount + 1) {
      throw new Error(`${id} output.offsets must hold featureCount + 1 rows`);
    }
    if (vertexCount) {
      validatePackedUint32View(vertexCount, `${id} output.vertexCount`);
      if (vertexCount.length < 1) {
        throw new Error(`${id} output.vertexCount must hold one uint32`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [positions, offsets, vertexCount],
      [props.centers, props.radii, props.bearings, props.rotations, props.parameters]
    );
  }

  /** Returns vertex and offsets nodes, preceded by an arc-length table node when requested. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, featureCount, maximumSegments} = this;
    const {positions, offsets, vertexCount} = props.output;
    validateGraphViewsBelongToGraph(id, graph, [
      props.centers,
      props.radii,
      props.bearings,
      props.rotations,
      props.parameters,
      positions,
      offsets,
      vertexCount
    ]);
    const {shape} = props;
    const geodesic = props.coordinateSystem === 'geodesic';
    const arcLengths =
      props.ellipseSpacing === 'arc-length'
        ? createTransientView(graph, `${id}-arc-lengths`, 'float32', featureCount * 257)
        : undefined;
    const nodes: GPUCommandNode<Parameters>[] = [];
    const ellipseDeclarations = `
const QUARTER_TURN: f32 = 1.5707963267948966;
const ARC_INTERVALS: u32 = 256u;

fn getEllipseRadii(feature: u32) -> vec2<f32> {
  var axes = vec2<f32>(
    radii[radiiOffset + 2u * feature], radii[radiiOffset + 2u * feature + 1u]) *
    parameters[parametersOffset + 1u];
  if ((bitcast<u32>(axes.x) & 0x7f800000u) == 0x7f800000u || axes.x < 0.0) {
    axes.x = 0.0;
  }
  if ((bitcast<u32>(axes.y) & 0x7f800000u) == 0x7f800000u || axes.y < 0.0) {
    axes.y = 0.0;
  }
  return axes;
}`;
    if (arcLengths) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-arc-lengths`,
          operation: OPERATION,
          variant: 'ellipse-arc-lengths',
          bindings: [
            {name: 'radii', view: props.radii, type: 'f32', access: 'read'},
            {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
            {name: 'arcLengths', view: arcLengths, type: 'f32', access: 'read_write'}
          ],
          invocationCount: featureCount,
          declarations: ellipseDeclarations,
          body: `let axes = getEllipseRadii(index);
  // Normalize before squaring to keep the table independent of world-coordinate scale.
  let maximumRadius = max(axes.x, axes.y);
  var normalizedAxes = vec2<f32>(0.0);
  if (maximumRadius > 0.0) { normalizedAxes = axes / maximumRadius; }
  let base = arcLengthsOffset + index * (ARC_INTERVALS + 1u);
  arcLengths[base] = 0.0;
  var total = 0.0;
  for (var interval = 1u; interval <= ARC_INTERVALS; interval++) {
    // The exact chord uses the midpoint derivative times 2 sin(delta / 2), avoiding
    // cancellation from subtracting nearby ellipse coordinates. Quarter symmetry lets
    // every vertex reuse one table. The polygonal length converges quadratically.
    let halfStep = QUARTER_TURN / (2.0 * f32(ARC_INTERVALS));
    let midpoint = (2.0 * f32(interval) - 1.0) * halfStep;
    total += 2.0 * sin(halfStep) * length(normalizedAxes * vec2<f32>(sin(midpoint), cos(midpoint)));
    arcLengths[base + interval] = total;
  }`
        })
      );
    }
    const maximumVertices = getGPUShapeVertexCount(shape, maximumSegments);
    const constants = `const FEATURE_COUNT: u32 = ${featureCount}u;
const MAXIMUM_SEGMENTS: u32 = ${maximumSegments}u;
const MINIMUM_SEGMENTS: u32 = ${getGPUShapeMinimumSegments(shape)}u;
const MAXIMUM_VERTICES: u32 = ${maximumVertices}u;
const IS_SECTOR: bool = ${shape === 'sector'};

fn getSegmentCount(rawSegments: f32) -> u32 {
  return clamp(u32(max(rawSegments, 0.0)), MINIMUM_SEGMENTS, MAXIMUM_SEGMENTS);
}

fn getVertexCount(segments: u32) -> u32 {
  return select(segments + 1u, segments + 3u, IS_SECTOR);
}`;

    const vertexBindings: WGSLKernelBinding[] = [
      {name: 'centers', view: props.centers, type: 'f32', access: 'read'},
      {name: 'radii', view: props.radii, type: 'f32', access: 'read'},
      ...(props.bearings
        ? [{name: 'bearings', view: props.bearings, type: 'f32' as const, access: 'read' as const}]
        : []),
      ...(props.rotations
        ? [
            {
              name: 'rotations',
              view: props.rotations,
              type: 'f32' as const,
              access: 'read' as const
            }
          ]
        : []),
      {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
      ...(arcLengths
        ? [{name: 'arcLengths', view: arcLengths, type: 'f32' as const, access: 'read' as const}]
        : []),
      {name: 'positionsOut', view: positions, type: 'f32', access: 'read_write'}
    ];

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-vertices`,
        operation: OPERATION,
        variant: `${shape}-${geodesic ? 'geodesic' : 'planar'}${props.rotations ? '-rotated' : ''}${arcLengths ? '-arc-length' : ''}`,
        bindings: vertexBindings,
        invocationCount: featureCount * maximumVertices,
        declarations: `${constants}
const EARTH_RADIUS: f32 = ${GPU_SHAPE_GENERATOR_EARTH_RADIUS};
const DEGREES: f32 = 0.017453292519943295;
const TWO_PI: f32 = 6.283185307179586;
const NOT_A_NUMBER: u32 = 0x7fc00000u;
${shape === 'ellipse' ? ellipseDeclarations : ''}
${
  arcLengths
    ? `
fn getEllipsePhase(feature: u32, step: u32, segments: u32, axes: vec2<f32>) -> f32 {
  let fraction = f32(step) / f32(segments);
  // Equal axes already have equal arc intervals. This also handles collapsed points.
  if (axes.x == axes.y) { return TWO_PI * fraction; }
  // Integer quadrant arithmetic keeps cardinal vertices and the closure exact.
  let quadrant = (4u * step) / segments;
  let quarterFraction = f32((4u * step) % segments) / f32(segments);
  let reverse = (quadrant & 1u) != 0u;
  let targetFraction = select(quarterFraction, 1.0 - quarterFraction, reverse);
  let base = arcLengthsOffset + feature * (ARC_INTERVALS + 1u);
  let targetLength = targetFraction * arcLengths[base + ARC_INTERVALS];
  var lower = 0u;
  var upper = ARC_INTERVALS;
  for (var iteration = 0u; iteration < 8u; iteration++) {
    let middle = (lower + upper) / 2u;
    if (arcLengths[base + middle] < targetLength) { lower = middle; } else { upper = middle; }
  }
  let start = arcLengths[base + lower];
  let span = arcLengths[base + upper] - start;
  var weight = 0.0;
  if (span > 0.0) { weight = clamp((targetLength - start) / span, 0.0, 1.0); }
  let phase = QUARTER_TURN * (f32(lower) + weight) / f32(ARC_INTERVALS);
  return f32(quadrant) * QUARTER_TURN + select(phase, QUARTER_TURN - phase, reverse);
}`
    : ''
}

fn isFinite32(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}

// Point at the given distance (world units, or meters for geodesic) and bearing (radians
// clockwise from north) from the center.
fn getDestination(center: vec2<f32>, distance: f32, bearing: f32) -> vec2<f32> {
${
  geodesic
    ? `  let angularDistance = distance / EARTH_RADIUS;
  let latitude = center.y * DEGREES;
  let sinDistance = sin(angularDistance);
  let cosDistance = cos(angularDistance);
  let sinLatitude = sin(latitude);
  let cosLatitude = cos(latitude);
  let sinLatitude2 = clamp(
    sinLatitude * cosDistance + cosLatitude * sinDistance * cos(bearing), -1.0, 1.0);
  let latitude2 = asin(sinLatitude2);
  // Longitude difference directly, so the small offset is not lost to radians rounding.
  let longitudeDelta = atan2(
    sin(bearing) * sinDistance * cosLatitude, cosDistance - sinLatitude * sinLatitude2);
  return vec2<f32>(center.x + longitudeDelta / DEGREES, latitude2 / DEGREES);`
    : '  return center + distance * vec2<f32>(sin(bearing), cos(bearing));'
}
}`,
        body: `let feature = index / MAXIMUM_VERTICES;
  let local = index % MAXIMUM_VERTICES;
  let segments = getSegmentCount(parameters[parametersOffset]);
  let vertices = getVertexCount(segments);
  if (local >= vertices) {
    return;
  }
  let radiusScale = parameters[parametersOffset + 1u];
  let center = vec2<f32>(
    centers[centersOffset + 2u * feature], centers[centersOffset + 2u * feature + 1u]);
  var position = center;
${
  shape === 'ellipse'
    ? `  let radii2 = getEllipseRadii(feature);
  // Counter-clockwise from the x semi-axis, which points along bearing 90 degrees + rotation.
  let phase = ${arcLengths ? 'getEllipsePhase(feature, local % segments, segments, radii2)' : 'TWO_PI * f32(local % segments) / f32(segments)'};
  let east = radii2.x * cos(phase);
  let north = radii2.y * sin(phase);
  ${props.rotations ? 'let tilt = rotations[rotationsOffset + feature] * DEGREES;' : 'let tilt: f32 = 0.0;'}
  // Rotate the (east, north) offset clockwise by tilt.
  let rotatedEast = east * cos(tilt) + north * sin(tilt);
  let rotatedNorth = north * cos(tilt) - east * sin(tilt);
  ${
    arcLengths && !geodesic
      ? 'position = center + vec2<f32>(rotatedEast, rotatedNorth);'
      : `let distance = length(vec2<f32>(rotatedEast, rotatedNorth));
  var bearing = atan2(rotatedEast, rotatedNorth);
  ${
    arcLengths
      ? `// Avoid backend atan2 signed-zero ambiguity for an east/west degenerate axis.
  if (rotatedNorth == 0.0) {
    bearing = select(-QUARTER_TURN, QUARTER_TURN, rotatedEast >= 0.0);
  }`
      : ''
  }
  position = getDestination(center, distance, bearing);`
  }`
    : `  var radius = radii[radiiOffset + feature] * radiusScale;
  if (!isFinite32(radius) || radius < 0.0) { radius = 0.0; }
${
  shape === 'sector'
    ? `  var bearing1 = bearings[bearingsOffset + 2u * feature];
  var bearing2 = bearings[bearingsOffset + 2u * feature + 1u];
  bearing1 = bearing1 - 360.0 * floor(bearing1 / 360.0);
  bearing2 = bearing2 - 360.0 * floor(bearing2 / 360.0);
  let sweep = select(bearing2 - bearing1 + 360.0, bearing2 - bearing1, bearing1 < bearing2);
  // Slot 0 and the last slot are the center; slots 1..=segments+1 are the arc.
  if (local != 0u && local != vertices - 1u) {
    let step = f32(local - 1u) / f32(segments);
    position = getDestination(center, radius, (bearing1 + sweep * step) * DEGREES);
  }`
    : `  // Counter-clockwise ring, bearings 0, -360 / n, ...; the last vertex repeats the first.
  let ringStep = f32(local % segments) / f32(segments);
  position = getDestination(center, radius, -ringStep * TWO_PI);`
}`
}
  if (!isFinite32(center.x) || !isFinite32(center.y)) {
    position = vec2<f32>(bitcast<f32>(NOT_A_NUMBER | (feature & 0u)));
  }
  let base = positionsOutOffset + 2u * (feature * vertices + local);
  positionsOut[base] = position.x;
  positionsOut[base + 1u] = position.y;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-offsets`,
        operation: OPERATION,
        variant: 'offsets',
        bindings: [
          {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
          {name: 'offsetsOut', view: offsets, type: 'u32', access: 'read_write'},
          ...(vertexCount
            ? [
                {
                  name: 'vertexCountOut',
                  view: vertexCount,
                  type: 'u32' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        invocationCount: featureCount + 1,
        declarations: constants,
        body: `let vertices = getVertexCount(getSegmentCount(parameters[parametersOffset]));
  offsetsOut[offsetsOutOffset + index] = index * vertices;
  ${vertexCount ? 'if (index == FEATURE_COUNT) { vertexCountOut[vertexCountOutOffset] = index * vertices; }' : ''}`
      })
    );
    return nodes;
  }
}
