// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  COORDINATE_SYSTEM,
  Layer,
  project32,
  type LayerContext,
  type LayerProps,
  type UpdateParameters
} from '@deck.gl/core';
import {Buffer, type RenderPass} from '@luma.gl/core';
import {Model} from '@luma.gl/engine';

const NO_FEATURE = 0xffffffff;
const STYLE_BYTE_LENGTH = 32;
const BLEND_PARAMETERS = {
  depthWriteEnabled: false,
  depthCompare: 'always',
  blend: true,
  blendColorOperation: 'add',
  blendAlphaOperation: 'add',
  blendColorSrcFactor: 'src-alpha',
  blendColorDstFactor: 'one-minus-src-alpha',
  blendAlphaSrcFactor: 'one',
  blendAlphaDstFactor: 'one-minus-src-alpha'
} as const;

const SNAP_SHADER = /* wgsl */ `
struct SnapStyle {
  color: vec4<f32>,
  widthPixels: f32,
  _padding0: f32,
  _padding1: f32,
  _padding2: f32,
};

@group(0) @binding(auto) var<uniform> snapStyle: SnapStyle;
@group(0) @binding(auto) var<storage, read> snapPoints: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> snapFeatureIds: array<u32>;
@group(0) @binding(auto) var<storage, read> snapSegments: array<vec4<f32>>;

struct SnapVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) side: f32,
};

// Closest point to \`point\` on the segment from \`start\` to \`end\`.
fn getClosestSegmentPoint(point: vec2<f32>, start: vec2<f32>, end: vec2<f32>) -> vec2<f32> {
  let edge = end - start;
  let lengthSquared = dot(edge, edge);
  var t = 0.0;
  if (lengthSquared > 0.0) {
    t = clamp(dot(point - start, edge) / lengthSquared, 0.0, 1.0);
  }
  return start + edge * t;
}

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> SnapVertexOutput {
  // (t from the point to its snapped location, side across the line)
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
  );
  var output: SnapVertexOutput;
  let featureId = snapFeatureIds[instanceIndex];
  if (featureId == ${NO_FEATURE}u) {
    output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    output.side = 0.0;
    return output;
  }
  let point = snapPoints[instanceIndex];
  let segment = snapSegments[featureId];
  let snapped = getClosestSegmentPoint(point, segment.xy, segment.zw);
  var startClip = project_position_to_clipspace(vec3<f32>(point, 0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  var endClip = project_position_to_clipspace(vec3<f32>(snapped, 0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  // Deck's project32 matrices use OpenGL depth; WebGPU clip space requires [0, w].
  startClip.z = (startClip.z + startClip.w) * 0.5;
  endClip.z = (endClip.z + endClip.w) * 0.5;
  let corner = corners[vertexIndex];
  let screenDirection = (endClip.xy / endClip.w - startClip.xy / startClip.w) * project.viewportSize;
  let directionLength = length(screenDirection);
  var direction = vec2<f32>(1.0, 0.0);
  if (directionLength > 1e-6) {
    direction = screenDirection / directionLength;
  }
  let normal = vec2<f32>(-direction.y, direction.x);
  var clipPosition = mix(startClip, endClip, corner.x);
  clipPosition = vec4<f32>(
    clipPosition.xy + project_pixel_size_to_clipspace(normal * corner.y * snapStyle.widthPixels * 0.5),
    clipPosition.z,
    clipPosition.w
  );
  output.position = clipPosition;
  output.side = corner.y;
  return output;
}

@fragment fn fragmentMain(input: SnapVertexOutput) -> @location(0) vec4<f32> {
  let coverage = 1.0 - smoothstep(0.55, 1.0, abs(input.side));
  return vec4<f32>(snapStyle.color.rgb, snapStyle.color.a * coverage);
}
`;

/** Props for {@link NearestSnapLayer}. */
export type NearestSnapLayerProps = LayerProps & {
  /** `float32x2` point positions in planar meters. */
  positions: Buffer;
  /** uint32 nearest segment row per point, or `0xffffffff` for none. */
  nearestFeatureIds: Buffer;
  /** `float32x4` road segments `x0, y0, x1, y1` in planar meters. */
  segments: Buffer;
  /** Number of points. */
  instanceCount: number;
  /** RGBA, 0-255. */
  color?: readonly [number, number, number, number];
  /** Line width in CSS pixels. Defaults to 1.5. */
  widthPixels?: number;
};

/**
 * Draws a line from every point to the closest location on its nearest segment. The vertex shader
 * reads the join output and the segment geometry directly, so nothing is read back.
 */
export class NearestSnapLayer extends Layer<NearestSnapLayerProps> {
  static override layerName = 'NearestSnapLayer';
  static override defaultProps = {
    coordinateSystem: COORDINATE_SYSTEM.METER_OFFSETS,
    parameters: BLEND_PARAMETERS
  };

  override getAttributeManager() {
    return null;
  }

  override initializeState({device}: LayerContext): void {
    if (device.type !== 'webgpu') throw new Error(`${this.id} requires WebGPU`);
    const styleBuffer = device.createBuffer({
      id: `${this.id}-style`,
      byteLength: STYLE_BYTE_LENGTH,
      usage: Buffer.UNIFORM | Buffer.COPY_DST
    });
    const model = new Model(device, {
      ...this.getShaders({modules: [project32], source: SNAP_SHADER}),
      id: `${this.id}-model`,
      topology: 'triangle-list',
      isInstanced: true,
      vertexCount: 6,
      instanceCount: 0,
      bufferLayout: [],
      bindings: this.getBindings(styleBuffer),
      parameters: BLEND_PARAMETERS
    });
    this.setState({model, styleBuffer});
  }

  override updateState(parameters: UpdateParameters<this>): void {
    super.updateState(parameters);
    const {model, styleBuffer} = this.state as {model: Model; styleBuffer: Buffer};
    model.setBindings(this.getBindings(styleBuffer));
  }

  override getModels(): Model[] {
    return [(this.state as {model: Model}).model];
  }

  override draw({renderPass}: {renderPass: RenderPass}): void {
    const {model, styleBuffer} = this.state as {model: Model; styleBuffer: Buffer};
    const color = this.props.color ?? [255, 255, 255, 220];
    styleBuffer.write(
      Float32Array.of(
        color[0] / 255,
        color[1] / 255,
        color[2] / 255,
        color[3] / 255,
        this.props.widthPixels ?? 1.5,
        0,
        0,
        0
      )
    );
    model.setInstanceCount(this.props.instanceCount);
    model.draw(renderPass);
  }

  override finalizeState(context: LayerContext): void {
    const {model, styleBuffer} = this.state as {model: Model; styleBuffer: Buffer};
    model.destroy();
    styleBuffer.destroy();
    super.finalizeState(context);
  }

  private getBindings(styleBuffer: Buffer): Record<string, Buffer> {
    return {
      snapStyle: styleBuffer,
      snapPoints: this.props.positions,
      snapFeatureIds: this.props.nearestFeatureIds,
      snapSegments: this.props.segments
    };
  }
}

const NEIGHBOR_STYLE_BYTE_LENGTH = 32;

const NEIGHBOR_SHADER = /* wgsl */ `
struct NeighborStyle {
  alpha: f32,
  widthPixels: f32,
  stride: u32,
  shown: u32,
  colorByRank: u32,
  maximumDistance: f32,
  _padding0: f32,
  _padding1: f32,
};

@group(0) @binding(auto) var<uniform> neighborStyle: NeighborStyle;
@group(0) @binding(auto) var<storage, read> neighborQueries: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> neighborIds: array<u32>;
@group(0) @binding(auto) var<storage, read> neighborFeet: array<vec2<f32>>;

struct NeighborVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) side: f32,
  @location(1) color: vec4<f32>,
};

const RANK_COLORS = array<vec3<f32>, 8>(
  vec3<f32>(1.0, 1.0, 1.0),
  vec3<f32>(0.31, 0.79, 1.0),
  vec3<f32>(1.0, 0.58, 0.28),
  vec3<f32>(0.74, 0.48, 1.0),
  vec3<f32>(0.34, 0.92, 0.66),
  vec3<f32>(1.0, 0.41, 0.66),
  vec3<f32>(0.96, 0.86, 0.34),
  vec3<f32>(0.42, 0.62, 1.0)
);

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> NeighborVertexOutput {
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
  );
  var output: NeighborVertexOutput;
  let query = instanceIndex / neighborStyle.stride;
  let slot = instanceIndex % neighborStyle.stride;
  let featureId = neighborIds[instanceIndex];
  if (featureId == ${NO_FEATURE}u || slot >= neighborStyle.shown) {
    output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    output.side = 0.0;
    output.color = vec4<f32>(0.0);
    return output;
  }
  let point = neighborQueries[query];
  let foot = neighborFeet[instanceIndex];
  var startClip = project_position_to_clipspace(vec3<f32>(point, 0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  var endClip = project_position_to_clipspace(vec3<f32>(foot, 0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  // Deck's project32 matrices use OpenGL depth; WebGPU clip space requires [0, w].
  startClip.z = (startClip.z + startClip.w) * 0.5;
  endClip.z = (endClip.z + endClip.w) * 0.5;
  let corner = corners[vertexIndex];
  let screenDirection = (endClip.xy / endClip.w - startClip.xy / startClip.w) * project.viewportSize;
  let directionLength = length(screenDirection);
  var direction = vec2<f32>(1.0, 0.0);
  if (directionLength > 1e-6) {
    direction = screenDirection / directionLength;
  }
  let normal = vec2<f32>(-direction.y, direction.x);
  var clipPosition = mix(startClip, endClip, corner.x);
  clipPosition = vec4<f32>(
    clipPosition.xy + project_pixel_size_to_clipspace(normal * corner.y * neighborStyle.widthPixels * 0.5),
    clipPosition.z,
    clipPosition.w
  );
  output.position = clipPosition;
  output.side = corner.y;
  let fade = 1.0 - 0.45 * f32(slot) / max(f32(neighborStyle.shown), 1.0);
  var rgb = RANK_COLORS[slot % 8u];
  if (neighborStyle.colorByRank == 0u) {
    let t = clamp(length(foot - point) / max(neighborStyle.maximumDistance, 1.0), 0.0, 1.0);
    rgb = mix(vec3<f32>(0.27, 0.0, 0.33), vec3<f32>(0.99, 0.91, 0.14), t);
  }
  output.color = vec4<f32>(rgb, neighborStyle.alpha * fade);
  return output;
}

@fragment fn fragmentMain(input: NeighborVertexOutput) -> @location(0) vec4<f32> {
  let coverage = 1.0 - smoothstep(0.55, 1.0, abs(input.side));
  return vec4<f32>(input.color.rgb, input.color.a * coverage);
}
`;

/** Props for {@link NeighborLinkLayer}. */
export type NeighborLinkLayerProps = LayerProps & {
  /** `float32x2` query positions in planar meters. */
  queries: Buffer;
  /** uint32 dense neighbor ids, `query * stride + slot`, `0xffffffff` for empty slots. */
  neighborIds: Buffer;
  /** `float32x2` foot points aligned with `neighborIds`. */
  footPoints: Buffer;
  /** Number of queries. */
  queryCount: number;
  /** Neighbor slots per query (the join's `neighborCapacity`). */
  stride: number;
  /** Slots drawn per query: the first `shown` of the ordered neighbors. */
  shown: number;
  /** `true` colors by neighbor rank, `false` by distance up to `maximumDistance`. */
  colorByRank?: boolean;
  /** Distance mapped to the end of the ramp when not coloring by rank. */
  maximumDistance?: number;
  /** Line width in CSS pixels. Defaults to 1.5. */
  widthPixels?: number;
  /** Peak alpha, 0-1. Defaults to 0.9. */
  alpha?: number;
};

/**
 * Draws a line from every query to the foot point of each of its first `shown` neighbors, straight
 * from the `GPUNearestFeatureJoin` neighbor outputs. Nothing is read back.
 */
export class NeighborLinkLayer extends Layer<NeighborLinkLayerProps> {
  static override layerName = 'NeighborLinkLayer';
  static override defaultProps = {
    coordinateSystem: COORDINATE_SYSTEM.METER_OFFSETS,
    parameters: BLEND_PARAMETERS
  };

  override getAttributeManager() {
    return null;
  }

  override initializeState({device}: LayerContext): void {
    if (device.type !== 'webgpu') throw new Error(`${this.id} requires WebGPU`);
    const styleBuffer = device.createBuffer({
      id: `${this.id}-style`,
      byteLength: NEIGHBOR_STYLE_BYTE_LENGTH,
      usage: Buffer.UNIFORM | Buffer.COPY_DST
    });
    const model = new Model(device, {
      ...this.getShaders({modules: [project32], source: NEIGHBOR_SHADER}),
      id: `${this.id}-model`,
      topology: 'triangle-list',
      isInstanced: true,
      vertexCount: 6,
      instanceCount: 0,
      bufferLayout: [],
      bindings: this.getBindings(styleBuffer),
      parameters: BLEND_PARAMETERS
    });
    this.setState({model, styleBuffer});
  }

  override updateState(parameters: UpdateParameters<this>): void {
    super.updateState(parameters);
    const {model, styleBuffer} = this.state as {model: Model; styleBuffer: Buffer};
    model.setBindings(this.getBindings(styleBuffer));
  }

  override getModels(): Model[] {
    return [(this.state as {model: Model}).model];
  }

  override draw({renderPass}: {renderPass: RenderPass}): void {
    const {model, styleBuffer} = this.state as {model: Model; styleBuffer: Buffer};
    const data = new ArrayBuffer(NEIGHBOR_STYLE_BYTE_LENGTH);
    const floats = new Float32Array(data);
    const words = new Uint32Array(data);
    floats[0] = this.props.alpha ?? 0.9;
    floats[1] = this.props.widthPixels ?? 1.5;
    words[2] = this.props.stride;
    words[3] = this.props.shown;
    words[4] = this.props.colorByRank === false ? 0 : 1;
    floats[5] = this.props.maximumDistance ?? 100;
    styleBuffer.write(new Uint8Array(data));
    model.setInstanceCount(this.props.queryCount * this.props.stride);
    model.draw(renderPass);
  }

  override finalizeState(context: LayerContext): void {
    const {model, styleBuffer} = this.state as {model: Model; styleBuffer: Buffer};
    model.destroy();
    styleBuffer.destroy();
    super.finalizeState(context);
  }

  private getBindings(styleBuffer: Buffer): Record<string, Buffer> {
    return {
      neighborStyle: styleBuffer,
      neighborQueries: this.props.queries,
      neighborIds: this.props.neighborIds,
      neighborFeet: this.props.footPoints
    };
  }
}

const WEIGHT_STYLE_BYTE_LENGTH = 16;

const WEIGHT_SHADER = /* wgsl */ `
struct WeightStyle {
  alpha: f32,
  widthPixels: f32,
  stride: u32,
  weightK: u32,
};

@group(0) @binding(auto) var<uniform> weightStyle: WeightStyle;
@group(0) @binding(auto) var<storage, read> weightQueries: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> weightOffsets: array<u32>;
@group(0) @binding(auto) var<storage, read> weightNeighbors: array<u32>;
@group(0) @binding(auto) var<storage, read> weightValues: array<f32>;
@group(0) @binding(auto) var<storage, read> weightJoinIds: array<u32>;
@group(0) @binding(auto) var<storage, read> weightJoinFeet: array<vec2<f32>>;

struct WeightVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) side: f32,
  @location(1) color: vec4<f32>,
};

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> WeightVertexOutput {
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
  );
  var output: WeightVertexOutput;
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  output.side = 0.0;
  output.color = vec4<f32>(0.0);
  let query = instanceIndex / weightStyle.weightK;
  let entry = instanceIndex % weightStyle.weightK;
  let begin = weightOffsets[query];
  let end = weightOffsets[query + 1u];
  if (begin + entry >= end) { return output; }
  let featureId = weightNeighbors[begin + entry];
  let weight = weightValues[begin + entry];
  var rowSum = 0.0;
  for (var index = begin; index < end; index++) { rowSum += weightValues[index]; }
  let share = select(0.0, weight / rowSum, rowSum > 0.0);
  // The CSR row is sorted by feature id; the join slot with the same id carries the foot point.
  var found = false;
  var foot = vec2<f32>(0.0);
  for (var slot = 0u; slot < weightStyle.stride; slot++) {
    if (weightJoinIds[query * weightStyle.stride + slot] == featureId) {
      foot = weightJoinFeet[query * weightStyle.stride + slot];
      found = true;
      break;
    }
  }
  if (!found) { return output; }
  let point = weightQueries[query];
  var startClip = project_position_to_clipspace(vec3<f32>(point, 0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  var endClip = project_position_to_clipspace(vec3<f32>(foot, 0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  startClip.z = (startClip.z + startClip.w) * 0.5;
  endClip.z = (endClip.z + endClip.w) * 0.5;
  let corner = corners[vertexIndex];
  let screenDirection = (endClip.xy / endClip.w - startClip.xy / startClip.w) * project.viewportSize;
  let directionLength = length(screenDirection);
  var direction = vec2<f32>(1.0, 0.0);
  if (directionLength > 1e-6) { direction = screenDirection / directionLength; }
  let normal = vec2<f32>(-direction.y, direction.x);
  let width = weightStyle.widthPixels * (0.4 + 5.0 * share);
  var clipPosition = mix(startClip, endClip, corner.x);
  clipPosition = vec4<f32>(
    clipPosition.xy + project_pixel_size_to_clipspace(normal * corner.y * width * 0.5),
    clipPosition.z,
    clipPosition.w
  );
  output.position = clipPosition;
  output.side = corner.y;
  let rgb = mix(vec3<f32>(0.2, 0.55, 1.0), vec3<f32>(1.0, 0.25, 0.7), clamp(share * 1.6, 0.0, 1.0));
  output.color = vec4<f32>(rgb, weightStyle.alpha * (0.35 + 0.65 * clamp(share * 2.0, 0.0, 1.0)));
  return output;
}

@fragment fn fragmentMain(input: WeightVertexOutput) -> @location(0) vec4<f32> {
  let coverage = 1.0 - smoothstep(0.55, 1.0, abs(input.side));
  return vec4<f32>(input.color.rgb, input.color.a * coverage);
}
`;

/** Props for {@link WeightedLinkLayer}. */
export type WeightedLinkLayerProps = LayerProps & {
  /** `float32x2` query positions in planar meters. */
  queries: Buffer;
  /** CSR row offsets of the `GPUNearestFeatureWeights` output (`queryCount + 1`). */
  offsets: Buffer;
  /** CSR neighbor feature ids, ascending within each row. */
  neighbors: Buffer;
  /** CSR weights aligned with `neighbors`. */
  weights: Buffer;
  /** The join's `neighborIds`, used to look up the foot point of each CSR neighbor. */
  joinIds: Buffer;
  /** The join's `neighborFootPoints`. */
  joinFootPoints: Buffer;
  /** Number of queries. */
  queryCount: number;
  /** Join slots per query. */
  stride: number;
  /** Compiled `k` of the weights (maximum entries per row). */
  weightK: number;
  /** Base line width in CSS pixels. Defaults to 2. */
  widthPixels?: number;
  /** Peak alpha, 0-1. Defaults to 0.95. */
  alpha?: number;
};

/**
 * Draws one link per CSR entry of `GPUNearestFeatureWeights`, from the query to the foot point of
 * the weighted feature, with width and color proportional to the row-standardized weight.
 */
export class WeightedLinkLayer extends Layer<WeightedLinkLayerProps> {
  static override layerName = 'WeightedLinkLayer';
  static override defaultProps = {
    coordinateSystem: COORDINATE_SYSTEM.METER_OFFSETS,
    parameters: BLEND_PARAMETERS
  };

  override getAttributeManager() {
    return null;
  }

  override initializeState({device}: LayerContext): void {
    if (device.type !== 'webgpu') throw new Error(`${this.id} requires WebGPU`);
    const styleBuffer = device.createBuffer({
      id: `${this.id}-style`,
      byteLength: WEIGHT_STYLE_BYTE_LENGTH,
      usage: Buffer.UNIFORM | Buffer.COPY_DST
    });
    const model = new Model(device, {
      ...this.getShaders({modules: [project32], source: WEIGHT_SHADER}),
      id: `${this.id}-model`,
      topology: 'triangle-list',
      isInstanced: true,
      vertexCount: 6,
      instanceCount: 0,
      bufferLayout: [],
      bindings: this.getBindings(styleBuffer),
      parameters: BLEND_PARAMETERS
    });
    this.setState({model, styleBuffer});
  }

  override updateState(parameters: UpdateParameters<this>): void {
    super.updateState(parameters);
    const {model, styleBuffer} = this.state as {model: Model; styleBuffer: Buffer};
    model.setBindings(this.getBindings(styleBuffer));
  }

  override getModels(): Model[] {
    return [(this.state as {model: Model}).model];
  }

  override draw({renderPass}: {renderPass: RenderPass}): void {
    const {model, styleBuffer} = this.state as {model: Model; styleBuffer: Buffer};
    const data = new ArrayBuffer(WEIGHT_STYLE_BYTE_LENGTH);
    const floats = new Float32Array(data);
    const words = new Uint32Array(data);
    floats[0] = this.props.alpha ?? 0.95;
    floats[1] = this.props.widthPixels ?? 2;
    words[2] = this.props.stride;
    words[3] = this.props.weightK;
    styleBuffer.write(new Uint8Array(data));
    model.setInstanceCount(this.props.queryCount * this.props.weightK);
    model.draw(renderPass);
  }

  override finalizeState(context: LayerContext): void {
    const {model, styleBuffer} = this.state as {model: Model; styleBuffer: Buffer};
    model.destroy();
    styleBuffer.destroy();
    super.finalizeState(context);
  }

  private getBindings(styleBuffer: Buffer): Record<string, Buffer> {
    return {
      weightStyle: styleBuffer,
      weightQueries: this.props.queries,
      weightOffsets: this.props.offsets,
      weightNeighbors: this.props.neighbors,
      weightValues: this.props.weights,
      weightJoinIds: this.props.joinIds,
      weightJoinFeet: this.props.joinFootPoints
    };
  }
}
