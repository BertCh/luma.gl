// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Mode-local Deck layers of the relief and flow-field modes. Each layer binds contributor outputs as
 * read-only storage buffers and draws them without CPU readback:
 *
 * - {@link FlowTextureLayer}: a LIC raster tinted by the flow speed.
 * - {@link FlowTrailLayer}: particle trails straight from the advection ring buffer (segment `k`
 *   of particle `i` joins ring slots in age order, derived from the frame word in the shader).
 * - {@link FlowPathLayer}: CSR streamline polylines (`pathOffsets` + `points`).
 * - {@link StorageModelLayer}: the shared base, also used by `relief-layers.ts`.
 *
 * All positions are planar meters rendered with `COORDINATE_SYSTEM.METER_OFFSETS`.
 */

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
import type {DrawCommandBuffer} from '@luma.gl/gpgpu/gpu-core';

/** Standard source-over blending used by every layer here. */
export const STORAGE_LAYER_BLEND_PARAMETERS = {
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

/** WGSL shared by the layers: projection of planar meters to a WebGPU clip position. */
export const STORAGE_LAYER_PROJECTION_WGSL = /* wgsl */ `
fn projectStoragePosition(position: vec2<f32>) -> vec4<f32> {
  var clipPosition = project_position_to_clipspace(vec3<f32>(position, 0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  // Deck's project32 matrices use OpenGL depth; WebGPU clip space requires [0, w].
  clipPosition.z = (clipPosition.z + clipPosition.w) * 0.5;
  return clipPosition;
}

// Non-finite floats are tested through the exponent bits: compilers may fold x != x.
fn isNonFiniteStorageFloat(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7fffffffu) >= 0x7f800000u;
}
`;

/** Flow speed ramp stops, slow to fast, shared by the WGSL ramp and the panel legends. */
export const FLOW_SPEED_RAMP: readonly (readonly [number, number, number])[] = [
  [51, 89, 230],
  [51, 217, 242],
  [217, 242, 102],
  [255, 191, 89]
];

/** WGSL `getFlowSpeedColor(t)` built from {@link FLOW_SPEED_RAMP}. */
export const FLOW_SPEED_RAMP_WGSL = /* wgsl */ `
fn getFlowSpeedColor(t: f32) -> vec3<f32> {
  var stops = array<vec3<f32>, 4>(
    ${FLOW_SPEED_RAMP.map(
      ([red, green, blue]) =>
        `vec3<f32>(${(red / 255).toFixed(4)}, ${(green / 255).toFixed(4)}, ${(blue / 255).toFixed(4)})`
    ).join(',\n    ')}
  );
  let scaled = clamp(t, 0.0, 1.0) * 3.0;
  let index = min(u32(floor(scaled)), 2u);
  return mix(stops[index], stops[index + 1u], scaled - f32(index));
}
`;

type StorageModelLayerState = {
  model: Model | null;
  styleBuffer: Buffer | null;
  placeholderBuffer: Buffer | null;
};

/** Props every storage-model layer accepts. */
export type StorageModelLayerProps = LayerProps & {
  /** Optional GPU-written indirect record; its vertex count must match the layer. */
  drawCommands?: DrawCommandBuffer | null;
  /** Record index inside `drawCommands`. Defaults to 0. */
  drawCommandIndex?: number;
};

/**
 * Shared lifecycle: one instanced triangle-list model whose bindings are storage buffers plus one
 * owned style uniform buffer named `layerStyle` in WGSL.
 */
export abstract class StorageModelLayer<
  PropsT extends StorageModelLayerProps
> extends Layer<PropsT> {
  static override layerName = 'StorageModelLayer';
  static override defaultProps = {
    coordinateSystem: COORDINATE_SYSTEM.METER_OFFSETS,
    parameters: STORAGE_LAYER_BLEND_PARAMETERS
  };

  override getAttributeManager() {
    return null;
  }

  /** Full WGSL module source (declares the `layerStyle` uniform and both entry points). */
  protected abstract getShaderSource(): string;
  /** Byte length of the `layerStyle` uniform (a multiple of 16). */
  protected abstract getStyleByteLength(): number;
  /** Writes the uniform contents. Called every draw. */
  protected abstract writeStyle(styleBuffer: Buffer): void;
  /** Storage bindings by WGSL variable name; use `placeholder` for unused ones. */
  protected abstract getStorageBindings(placeholder: Buffer): Record<string, Buffer>;
  /** Instances to draw when no `drawCommands` is set. */
  protected abstract getInstanceCount(): number;

  override initializeState({device}: LayerContext): void {
    if (device.type !== 'webgpu') throw new Error(`${this.id} requires WebGPU`);
    const styleBuffer = device.createBuffer({
      id: `${this.id}-style`,
      byteLength: this.getStyleByteLength(),
      usage: Buffer.UNIFORM | Buffer.COPY_DST
    });
    const placeholderBuffer = device.createBuffer({
      id: `${this.id}-placeholder`,
      byteLength: 16,
      usage: Buffer.STORAGE | Buffer.COPY_DST
    });
    this.setState({model: null, styleBuffer, placeholderBuffer} satisfies StorageModelLayerState);
    const model = new Model(device, {
      ...this.getShaders({modules: [project32], source: this.getShaderSource()}),
      id: `${this.id}-model`,
      topology: 'triangle-list',
      isInstanced: true,
      vertexCount: 6,
      instanceCount: 0,
      bufferLayout: [],
      bindings: {...this.getStorageBindings(placeholderBuffer), layerStyle: styleBuffer},
      parameters: STORAGE_LAYER_BLEND_PARAMETERS
    });
    this.setState({model});
  }

  override updateState(parameters: UpdateParameters<this>): void {
    super.updateState(parameters);
    const {model, styleBuffer, placeholderBuffer} = this.state as StorageModelLayerState;
    if (model && styleBuffer && placeholderBuffer) {
      model.setBindings({...this.getStorageBindings(placeholderBuffer), layerStyle: styleBuffer});
    }
  }

  override getModels(): Model[] {
    const model = (this.state as StorageModelLayerState).model;
    return model ? [model] : [];
  }

  override draw({renderPass}: {renderPass: RenderPass}): void {
    const {model, styleBuffer} = this.state as StorageModelLayerState;
    if (!model || !styleBuffer) return;
    this.writeStyle(styleBuffer);
    const {drawCommands, drawCommandIndex = 0} = this.props;
    if (drawCommands) {
      // Bind pipeline and bindings with an empty draw, then replay the GPU-written record.
      model.setInstanceCount(0);
      model.draw(renderPass);
      drawCommands.draw(renderPass, drawCommandIndex);
    } else {
      model.setInstanceCount(this.getInstanceCount());
      model.draw(renderPass);
    }
  }

  override finalizeState(context: LayerContext): void {
    const state = this.state as StorageModelLayerState;
    state.model?.destroy();
    state.styleBuffer?.destroy();
    state.placeholderBuffer?.destroy();
    this.setState({model: null, styleBuffer: null, placeholderBuffer: null});
    super.finalizeState(context);
  }
}

const QUAD_CORNERS_WGSL = /* wgsl */ `
var<private> QUAD_CORNERS = array<vec2<f32>, 6>(
  vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
  vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0)
);
`;

// -------------------------------------------------------------------------------------------------
// LIC raster
// -------------------------------------------------------------------------------------------------

const TEXTURE_SHADER = /* wgsl */ `
struct LayerStyle {
  minimum: vec2<f32>,
  maximum: vec2<f32>,
  speedRange: vec2<f32>,
  opacity: f32,
  contrast: f32,
  gridSize: vec2<u32>,
  padding: vec2<u32>,
};
@group(0) @binding(auto) var<uniform> layerStyle: LayerStyle;
@group(0) @binding(auto) var<storage, read> licValues: array<f32>;
@group(0) @binding(auto) var<storage, read> licSpeeds: array<f32>;
${STORAGE_LAYER_PROJECTION_WGSL}
${FLOW_SPEED_RAMP_WGSL}
${QUAD_CORNERS_WGSL}

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) local: vec2<f32>,
};

@vertex fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  let corner = QUAD_CORNERS[vertexIndex];
  var output: VertexOutput;
  output.position = projectStoragePosition(mix(layerStyle.minimum, layerStyle.maximum, corner));
  output.local = corner;
  return output;
}

fn readCell(column: i32, row: i32) -> vec2<f32> {
  let columns = i32(layerStyle.gridSize.x);
  let rows = i32(layerStyle.gridSize.y);
  let index = u32(clamp(row, 0, rows - 1) * columns + clamp(column, 0, columns - 1));
  return vec2<f32>(licValues[index], licSpeeds[index]);
}

@fragment fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
  // Row 0 is the south edge, so the row index grows with y. Bilinear between cell centers.
  let grid = vec2<f32>(layerStyle.gridSize);
  let position = input.local * grid - vec2<f32>(0.5);
  let base = vec2<i32>(floor(position));
  let fraction = position - floor(position);
  let a = readCell(base.x, base.y);
  let b = readCell(base.x + 1, base.y);
  let c = readCell(base.x, base.y + 1);
  let d = readCell(base.x + 1, base.y + 1);
  if (isNonFiniteStorageFloat(a.x) || isNonFiniteStorageFloat(b.x) ||
      isNonFiniteStorageFloat(c.x) || isNonFiniteStorageFloat(d.x)) {
    discard;
  }
  let value = mix(mix(a, b, fraction.x), mix(c, d, fraction.x), fraction.y);
  let stretched = clamp((value.x - 0.5) * layerStyle.contrast + 0.5, 0.0, 1.0);
  let speedRatio = (value.y - layerStyle.speedRange.x) /
    max(layerStyle.speedRange.y - layerStyle.speedRange.x, 1e-20);
  let alpha = layerStyle.opacity * stretched * stretched;
  return vec4<f32>(getFlowSpeedColor(speedRatio), alpha);
}
`;

/** Props for {@link FlowTextureLayer}. */
export type FlowTextureLayerProps = StorageModelLayerProps & {
  /** LIC values in `[0, 1]`, `columns * rows` floats, row 0 at the smallest y. */
  values: Buffer;
  /** Flow speed at each pixel center, same layout as `values`. */
  speeds: Buffer;
  /** `[columns, rows]` of the LIC raster. */
  gridSize: readonly [number, number];
  /** `[minX, minY, maxX, maxY]` meters covered by the raster. */
  bounds: readonly [number, number, number, number];
  /** Speeds mapped to the ends of the color ramp. */
  speedRange: readonly [number, number];
  /** Maximum alpha. Defaults to 0.9. */
  opacity?: number;
  /** Contrast stretch around the mean LIC value 0.5. Defaults to 3. */
  contrast?: number;
};

/** One quad over the raster bounds; fragments sample LIC value and speed from storage. */
export class FlowTextureLayer extends StorageModelLayer<FlowTextureLayerProps> {
  static override layerName = 'FlowTextureLayer';

  protected getShaderSource(): string {
    return TEXTURE_SHADER;
  }
  protected getStyleByteLength(): number {
    return 48;
  }
  protected getInstanceCount(): number {
    return 1;
  }
  protected getStorageBindings(placeholder: Buffer): Record<string, Buffer> {
    return {licValues: this.props.values, licSpeeds: this.props.speeds ?? placeholder};
  }
  protected writeStyle(styleBuffer: Buffer): void {
    const {bounds, speedRange, gridSize, opacity = 0.9, contrast = 3} = this.props;
    const data = new ArrayBuffer(48);
    const floats = new Float32Array(data);
    floats.set([bounds[0], bounds[1], bounds[2], bounds[3], speedRange[0], speedRange[1]]);
    floats[6] = opacity;
    floats[7] = contrast;
    new Uint32Array(data).set(gridSize, 8);
    styleBuffer.write(new Uint8Array(data));
  }
}

// -------------------------------------------------------------------------------------------------
// Particle trails
// -------------------------------------------------------------------------------------------------

const SEGMENT_EXPANSION_WGSL = /* wgsl */ `
// Expands the segment (startClip, endClip) into a screen-space quad corner; x is t along the
// segment, y the side across it.
fn expandSegment(startClip: vec4<f32>, endClip: vec4<f32>, corner: vec2<f32>, widthPixels: f32) -> vec4<f32> {
  let screenDirection = (endClip.xy / endClip.w - startClip.xy / startClip.w) * project.viewportSize;
  let directionLength = length(screenDirection);
  var direction = vec2<f32>(1.0, 0.0);
  if (directionLength > 1e-6) {
    direction = screenDirection / directionLength;
  }
  let normal = vec2<f32>(-direction.y, direction.x);
  var clipPosition = mix(startClip, endClip, corner.x);
  let along = direction * (corner.x * 2.0 - 1.0) * 0.5;
  return vec4<f32>(
    clipPosition.xy + project_pixel_size_to_clipspace((normal * corner.y + along) * widthPixels * 0.5),
    clipPosition.z,
    clipPosition.w
  );
}
`;

const SEGMENT_CORNERS_WGSL = /* wgsl */ `
var<private> SEGMENT_CORNERS = array<vec2<f32>, 6>(
  vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(0.0, 1.0),
  vec2<f32>(0.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
);
`;

const TRAIL_SHADER = /* wgsl */ `
struct LayerStyle {
  speedRange: vec2<f32>,
  widthPixels: f32,
  opacity: f32,
  ringLength: u32,
  particleCount: u32,
  padding: vec2<u32>,
};
@group(0) @binding(auto) var<uniform> layerStyle: LayerStyle;
@group(0) @binding(auto) var<storage, read> trailPositions: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> trailSpeeds: array<f32>;
@group(0) @binding(auto) var<storage, read> trailWords: array<u32>;
${STORAGE_LAYER_PROJECTION_WGSL}
${FLOW_SPEED_RAMP_WGSL}
${SEGMENT_CORNERS_WGSL}
${SEGMENT_EXPANSION_WGSL}

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) side: f32,
  @location(1) color: vec4<f32>,
};

fn hiddenOutput() -> VertexOutput {
  var output: VertexOutput;
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  output.side = 0.0;
  output.color = vec4<f32>(0.0);
  return output;
}

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {
  let segmentsPerParticle = layerStyle.ringLength - 1u;
  let particle = instanceIndex / segmentsPerParticle;
  let segment = instanceIndex % segmentsPerParticle;
  if (particle >= layerStyle.particleCount) {
    return hiddenOutput();
  }
  // Frame f writes slot f % L, so the oldest slot is (f + 1) % L; segment k joins the k-th and
  // (k + 1)-th slots in age order.
  let frame = trailWords[1];
  let startSlot = (frame + 1u + segment) % layerStyle.ringLength;
  let endSlot = (frame + 2u + segment) % layerStyle.ringLength;
  let start = trailPositions[particle * layerStyle.ringLength + startSlot];
  let end = trailPositions[particle * layerStyle.ringLength + endSlot];
  let speed = trailSpeeds[particle];
  if (isNonFiniteStorageFloat(start.x) || isNonFiniteStorageFloat(end.x) ||
      isNonFiniteStorageFloat(start.y) || isNonFiniteStorageFloat(end.y)) {
    return hiddenOutput();
  }
  let startClip = projectStoragePosition(start);
  let endClip = projectStoragePosition(end);
  // A respawn fills the ring with one position: skip zero-length segments (also hides dead cells).
  let pixelLength = length((endClip.xy / endClip.w - startClip.xy / startClip.w) * project.viewportSize);
  if (pixelLength < 0.02) {
    return hiddenOutput();
  }
  let age = f32(segment + 1u) / f32(segmentsPerParticle);
  let speedRatio = (speed - layerStyle.speedRange.x) /
    max(layerStyle.speedRange.y - layerStyle.speedRange.x, 1e-20);
  var output: VertexOutput;
  output.position = expandSegment(startClip, endClip, SEGMENT_CORNERS[vertexIndex], layerStyle.widthPixels);
  output.side = SEGMENT_CORNERS[vertexIndex].y;
  output.color = vec4<f32>(getFlowSpeedColor(speedRatio), age * age * layerStyle.opacity);
  return output;
}

@fragment fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
  let coverage = 1.0 - smoothstep(0.55, 1.0, abs(input.side));
  return vec4<f32>(input.color.rgb, input.color.a * coverage);
}
`;

/** Props for {@link FlowTrailLayer}. */
export type FlowTrailLayerProps = StorageModelLayerProps & {
  /** Trail ring `float32x2`: particle `i` owns rows `i * ringLength` to `i * ringLength + L - 1`. */
  trailPositions: Buffer;
  /** Per-particle speed from the advection contributor, for the color ramp. */
  speeds: Buffer;
  /** The advection word-parameter buffer; word 1 is the frame that selects the newest slot. */
  wordParameters: Buffer;
  /** Ring length `L` the contributor was built with. */
  ringLength: number;
  /** Number of particles. */
  particleCount: number;
  /** Speeds mapped to the ends of the color ramp. */
  speedRange: readonly [number, number];
  /** Line width in CSS pixels. Defaults to 1.4. */
  widthPixels?: number;
  /** Maximum alpha at the head of a trail. Defaults to 0.9. */
  opacity?: number;
};

/** Particle trails drawn straight from the advection ring buffer, tinted and faded by age. */
export class FlowTrailLayer extends StorageModelLayer<FlowTrailLayerProps> {
  static override layerName = 'FlowTrailLayer';

  protected getShaderSource(): string {
    return TRAIL_SHADER;
  }
  protected getStyleByteLength(): number {
    return 32;
  }
  protected getInstanceCount(): number {
    return this.props.particleCount * (this.props.ringLength - 1);
  }
  protected getStorageBindings(): Record<string, Buffer> {
    return {
      trailPositions: this.props.trailPositions,
      trailSpeeds: this.props.speeds,
      trailWords: this.props.wordParameters
    };
  }
  protected writeStyle(styleBuffer: Buffer): void {
    const {speedRange, widthPixels = 1.4, opacity = 0.9, ringLength, particleCount} = this.props;
    const data = new ArrayBuffer(32);
    new Float32Array(data).set([speedRange[0], speedRange[1], widthPixels, opacity]);
    new Uint32Array(data).set([ringLength, particleCount], 4);
    styleBuffer.write(new Uint8Array(data));
  }
}

// -------------------------------------------------------------------------------------------------
// Streamline polylines
// -------------------------------------------------------------------------------------------------

const PATH_SHADER = /* wgsl */ `
struct LayerStyle {
  color: vec4<f32>,
  widthPixels: f32,
  opacity: f32,
  padding: vec2<u32>,
};
@group(0) @binding(auto) var<uniform> layerStyle: LayerStyle;
@group(0) @binding(auto) var<storage, read> pathPoints: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> pathOffsets: array<u32>;
@group(0) @binding(auto) var<storage, read> pathLineCount: array<u32>;
@group(0) @binding(auto) var<storage, read> pathPointCount: array<u32>;
${STORAGE_LAYER_PROJECTION_WGSL}
${SEGMENT_CORNERS_WGSL}
${SEGMENT_EXPANSION_WGSL}

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) side: f32,
};

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {
  var output: VertexOutput;
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  output.side = 0.0;
  let next = instanceIndex + 1u;
  if (next >= pathPointCount[0]) {
    return output;
  }
  // Instance i joins points i and i + 1 unless i + 1 starts a line: binary search the line of
  // point i + 1 in the CSR offsets.
  var low = 0u;
  var high = pathLineCount[0];
  while (high - low > 1u) {
    let middle = (low + high) / 2u;
    if (pathOffsets[middle] <= next) {
      low = middle;
    } else {
      high = middle;
    }
  }
  if (pathOffsets[low] == next) {
    return output;
  }
  let start = pathPoints[instanceIndex];
  let end = pathPoints[next];
  if (start.x != start.x || end.x != end.x) {
    return output;
  }
  output.position = expandSegment(
    projectStoragePosition(start), projectStoragePosition(end),
    SEGMENT_CORNERS[vertexIndex], layerStyle.widthPixels
  );
  output.side = SEGMENT_CORNERS[vertexIndex].y;
  return output;
}

@fragment fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
  let coverage = 1.0 - smoothstep(0.55, 1.0, abs(input.side));
  return vec4<f32>(layerStyle.color.rgb, layerStyle.color.a * layerStyle.opacity * coverage);
}
`;

/** Props for {@link FlowPathLayer}. */
export type FlowPathLayerProps = StorageModelLayerProps & {
  /** `float32x2` polyline points, `pathOffsets`-indexed. */
  points: Buffer;
  /** CSR offsets, `lineCount + 1` uint32 values. */
  pathOffsets: Buffer;
  /** One uint32: number of published lines. */
  lineCount: Buffer;
  /** One uint32: number of published points. */
  pointCount: Buffer;
  /** Upper bound of instances to draw when `drawCommands` is not set. */
  pointCapacity: number;
  /** Line color, 0-255 channels. */
  color?: readonly [number, number, number, number?];
  /** Line width in CSS pixels. Defaults to 1. */
  widthPixels?: number;
  /** Alpha multiplier. Defaults to 1. */
  opacity?: number;
};

/** Streamline polylines drawn from CSR storage; line breaks are found in the vertex shader. */
export class FlowPathLayer extends StorageModelLayer<FlowPathLayerProps> {
  static override layerName = 'FlowPathLayer';

  protected getShaderSource(): string {
    return PATH_SHADER;
  }
  protected getStyleByteLength(): number {
    return 32;
  }
  protected getInstanceCount(): number {
    return this.props.pointCapacity;
  }
  protected getStorageBindings(): Record<string, Buffer> {
    return {
      pathPoints: this.props.points,
      pathOffsets: this.props.pathOffsets,
      pathLineCount: this.props.lineCount,
      pathPointCount: this.props.pointCount
    };
  }
  protected writeStyle(styleBuffer: Buffer): void {
    const {color = [255, 255, 255, 255], widthPixels = 1, opacity = 1} = this.props;
    styleBuffer.write(
      Float32Array.of(
        color[0] / 255,
        color[1] / 255,
        color[2] / 255,
        (color[3] ?? 255) / 255,
        widthPixels,
        opacity,
        0,
        0
      )
    );
  }
}
