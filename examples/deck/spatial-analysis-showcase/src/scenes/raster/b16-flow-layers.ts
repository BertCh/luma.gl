// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Deck layers of the wind-flow scene. Each binds contributor outputs as read-only storage buffers
 * and draws them without CPU readback:
 *
 * - {@link WindTextureLayer}: a line-integral-convolution raster tinted by the wind speed.
 * - {@link WindTrailLayer}: particle trails straight from the advection ring buffer.
 * - {@link WindPathLayer}: CSR streamline polylines (`pathOffsets` + `points`).
 *
 * The wind field is a regular longitude and latitude grid, and the contributors integrate in
 * degrees, so positions are `[longitude, latitude]` and the layers use `COORDINATE_SYSTEM.LNGLAT`.
 * Speed colors are looked up in meters per second from a separate speed raster (the advection
 * speed output is in degrees per second, which is not a speed on the ground).
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
import type {WindRamp} from './b16-colors';

export {WIND_RAMP, type WindRamp} from './b16-colors';

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

/** Where the speed raster sits: `[originX, originY, cellWidth, cellHeight]` and its size in cells. */
export type WindFieldPlacement = {
  extent: readonly [number, number, number, number];
  size: readonly [number, number];
};

const FIELD_WGSL = /* wgsl */ `
struct FieldStyle {
  ramp: array<vec4<f32>, 4>,
  fieldOrigin: vec2<f32>,
  fieldCell: vec2<f32>,
  fieldSize: vec2<u32>,
  speedRange: vec2<f32>,
};

fn projectWindPosition(position: vec2<f32>) -> vec4<f32> {
  var clipPosition = project_position_to_clipspace(vec3<f32>(position, 0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  // Deck's project32 matrices use OpenGL depth; WebGPU clip space requires [0, w].
  clipPosition.z = (clipPosition.z + clipPosition.w) * 0.5;
  return clipPosition;
}

// Non-finite floats are tested through the exponent bits: compilers may fold x != x.
fn isNonFiniteWind(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7fffffffu) >= 0x7f800000u;
}

fn getRampColor(ramp: array<vec4<f32>, 4>, t: f32) -> vec3<f32> {
  var stops = ramp;
  let scaled = clamp(t, 0.0, 1.0) * 3.0;
  let index = min(u32(floor(scaled)), 2u);
  return mix(stops[index].rgb, stops[index + 1u].rgb, scaled - f32(index));
}
`;

const SPEED_SAMPLER_WGSL = /* wgsl */ `
@group(0) @binding(auto) var<storage, read> speedField: array<f32>;

fn readSpeedCell(style: FieldStyle, column: i32, row: i32) -> f32 {
  let columns = i32(style.fieldSize.x);
  let rows = i32(style.fieldSize.y);
  return speedField[u32(clamp(row, 0, rows - 1) * columns + clamp(column, 0, columns - 1))];
}

// Bilinear speed in meters per second at a longitude and latitude.
fn sampleWindSpeed(style: FieldStyle, lonLat: vec2<f32>) -> f32 {
  let grid = (lonLat - style.fieldOrigin) / style.fieldCell - vec2<f32>(0.5);
  let base = vec2<i32>(floor(grid));
  let fraction = grid - floor(grid);
  let a = readSpeedCell(style, base.x, base.y);
  let b = readSpeedCell(style, base.x + 1, base.y);
  let c = readSpeedCell(style, base.x, base.y + 1);
  let d = readSpeedCell(style, base.x + 1, base.y + 1);
  return mix(mix(a, b, fraction.x), mix(c, d, fraction.x), fraction.y);
}
`;

type StorageModelLayerState = {
  model: Model | null;
  styleBuffer: Buffer | null;
  placeholderBuffer: Buffer | null;
};

/** Props every wind layer accepts. */
export type WindLayerProps = LayerProps & {
  /** Speed in meters per second per field cell, row 0 at the south edge. */
  speedField: Buffer;
  /** Placement and size of the speed raster. */
  field: WindFieldPlacement;
  /** Ramp stops, slow to fast. */
  ramp: WindRamp;
  /** Speeds (m/s) mapped to the ends of the ramp. */
  speedRange: readonly [number, number];
  /** Optional GPU-written indirect record; its vertex count must match the layer. */
  drawCommands?: DrawCommandBuffer | null;
  /** Record index inside `drawCommands`. Defaults to 0. */
  drawCommandIndex?: number;
};

/** Bytes of the shared `FieldStyle` block at the start of every layer uniform. */
const FIELD_STYLE_BYTES = 64 + 16 + 16;

function writeFieldStyle(floats: Float32Array, words: Uint32Array, props: WindLayerProps): void {
  props.ramp.forEach(([red, green, blue], index) => {
    floats.set([red / 255, green / 255, blue / 255, 1], index * 4);
  });
  floats.set(props.field.extent.slice(0, 2), 16);
  floats.set(props.field.extent.slice(2, 4), 18);
  words.set(props.field.size, 20);
  floats.set(props.speedRange, 22);
}

/** Shared lifecycle: one instanced triangle-list model with storage bindings and a uniform. */
abstract class WindModelLayer<PropsT extends WindLayerProps> extends Layer<PropsT> {
  static override layerName = 'WindModelLayer';
  static override defaultProps = {
    coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
    parameters: BLEND_PARAMETERS
  };

  override getAttributeManager() {
    return null;
  }

  protected abstract getShaderSource(): string;
  protected abstract getStyleByteLength(): number;
  protected abstract writeStyle(styleBuffer: Buffer): void;
  protected abstract getStorageBindings(placeholder: Buffer): Record<string, Buffer>;
  protected abstract getInstanceCount(): number;
  protected getVertexCount(): number {
    return 6;
  }

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
      vertexCount: this.getVertexCount(),
      instanceCount: 0,
      bufferLayout: [],
      bindings: {...this.getStorageBindings(placeholderBuffer), layerStyle: styleBuffer},
      parameters: BLEND_PARAMETERS
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

const SEGMENT_WGSL = /* wgsl */ `
var<private> SEGMENT_CORNERS = array<vec2<f32>, 6>(
  vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(0.0, 1.0),
  vec2<f32>(0.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
);

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

// -------------------------------------------------------------------------------------------------
// LIC raster
// -------------------------------------------------------------------------------------------------

/** Latitude strips of the LIC quad: Mercator is not linear in latitude, strips keep it exact. */
const LIC_STRIPS = 96;

const TEXTURE_SHADER = /* wgsl */ `
${FIELD_WGSL}
struct LayerStyle {
  field: FieldStyle,
  minimum: vec2<f32>,
  maximum: vec2<f32>,
  gridSize: vec2<u32>,
  opacity: f32,
  contrast: f32,
};
@group(0) @binding(auto) var<uniform> layerStyle: LayerStyle;
@group(0) @binding(auto) var<storage, read> licValues: array<f32>;
${SPEED_SAMPLER_WGSL}

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) local: vec2<f32>,
};

var<private> QUAD_CORNERS = array<vec2<f32>, 6>(
  vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
  vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0)
);

@vertex fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  let strip = vertexIndex / 6u;
  let corner = QUAD_CORNERS[vertexIndex % 6u];
  let local = vec2<f32>(corner.x, (f32(strip) + corner.y) / ${LIC_STRIPS}.0);
  var output: VertexOutput;
  output.position = projectWindPosition(mix(layerStyle.minimum, layerStyle.maximum, local));
  output.local = local;
  return output;
}

fn readCell(column: i32, row: i32) -> f32 {
  let columns = i32(layerStyle.gridSize.x);
  let rows = i32(layerStyle.gridSize.y);
  return licValues[u32(clamp(row, 0, rows - 1) * columns + clamp(column, 0, columns - 1))];
}

@fragment fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
  // Row 0 is the south edge, so the row index grows with latitude. Bilinear between centers.
  let grid = vec2<f32>(layerStyle.gridSize);
  let position = input.local * grid - vec2<f32>(0.5);
  let base = vec2<i32>(floor(position));
  let fraction = position - floor(position);
  let a = readCell(base.x, base.y);
  let b = readCell(base.x + 1, base.y);
  let c = readCell(base.x, base.y + 1);
  let d = readCell(base.x + 1, base.y + 1);
  if (isNonFiniteWind(a) || isNonFiniteWind(b) || isNonFiniteWind(c) || isNonFiniteWind(d)) {
    discard;
  }
  let value = mix(mix(a, b, fraction.x), mix(c, d, fraction.x), fraction.y);
  let stretched = clamp((value - 0.5) * layerStyle.contrast + 0.5, 0.0, 1.0);
  let lonLat = mix(layerStyle.minimum, layerStyle.maximum, input.local);
  let speed = sampleWindSpeed(layerStyle.field, lonLat);
  let speedRatio = (speed - layerStyle.field.speedRange.x) /
    max(layerStyle.field.speedRange.y - layerStyle.field.speedRange.x, 1e-20);
  let alpha = layerStyle.opacity * stretched * stretched;
  return vec4<f32>(getRampColor(layerStyle.field.ramp, speedRatio), alpha);
}
`;

/** Props for {@link WindTextureLayer}. */
export type WindTextureLayerProps = WindLayerProps & {
  /** LIC values in `[0, 1]`, `columns * rows` floats, row 0 at the smallest latitude. */
  values: Buffer;
  /** `[columns, rows]` of the LIC raster. */
  gridSize: readonly [number, number];
  /** `[west, south, east, north]` degrees covered by the raster. */
  bounds: readonly [number, number, number, number];
  /** Maximum alpha. Defaults to 0.9. */
  opacity?: number;
  /** Contrast stretch around the mean LIC value 0.5. Defaults to 3. */
  contrast?: number;
};

/** The LIC raster as latitude strips over the bounds; fragments read the LIC value and speed. */
export class WindTextureLayer extends WindModelLayer<WindTextureLayerProps> {
  static override layerName = 'WindTextureLayer';

  protected getShaderSource(): string {
    return TEXTURE_SHADER;
  }
  protected getStyleByteLength(): number {
    return FIELD_STYLE_BYTES + 32;
  }
  protected getInstanceCount(): number {
    return 1;
  }
  protected override getVertexCount(): number {
    return 6 * LIC_STRIPS;
  }
  protected getStorageBindings(placeholder: Buffer): Record<string, Buffer> {
    return {licValues: this.props.values ?? placeholder, speedField: this.props.speedField};
  }
  protected writeStyle(styleBuffer: Buffer): void {
    const {bounds, gridSize, opacity = 0.9, contrast = 3} = this.props;
    const data = new ArrayBuffer(FIELD_STYLE_BYTES + 32);
    const floats = new Float32Array(data);
    const words = new Uint32Array(data);
    writeFieldStyle(floats, words, this.props);
    const base = FIELD_STYLE_BYTES / 4;
    floats.set([bounds[0], bounds[1], bounds[2], bounds[3]], base);
    words.set(gridSize, base + 4);
    floats[base + 6] = opacity;
    floats[base + 7] = contrast;
    styleBuffer.write(new Uint8Array(data));
  }
}

// -------------------------------------------------------------------------------------------------
// Particle trails
// -------------------------------------------------------------------------------------------------

const TRAIL_SHADER = /* wgsl */ `
${FIELD_WGSL}
struct LayerStyle {
  field: FieldStyle,
  widthPixels: f32,
  opacity: f32,
  ringLength: u32,
  particleCount: u32,
};
@group(0) @binding(auto) var<uniform> layerStyle: LayerStyle;
@group(0) @binding(auto) var<storage, read> trailPositions: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> trailWords: array<u32>;
${SPEED_SAMPLER_WGSL}
${SEGMENT_WGSL}

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
  if (isNonFiniteWind(start.x) || isNonFiniteWind(end.x) ||
      isNonFiniteWind(start.y) || isNonFiniteWind(end.y)) {
    return hiddenOutput();
  }
  let startClip = projectWindPosition(start);
  let endClip = projectWindPosition(end);
  // A respawn fills the ring with one position: skip zero-length segments.
  let pixelLength = length((endClip.xy / endClip.w - startClip.xy / startClip.w) * project.viewportSize);
  if (pixelLength < 0.02) {
    return hiddenOutput();
  }
  let age = f32(segment + 1u) / f32(segmentsPerParticle);
  let speed = sampleWindSpeed(layerStyle.field, end);
  let speedRatio = (speed - layerStyle.field.speedRange.x) /
    max(layerStyle.field.speedRange.y - layerStyle.field.speedRange.x, 1e-20);
  var output: VertexOutput;
  output.position = expandSegment(startClip, endClip, SEGMENT_CORNERS[vertexIndex], layerStyle.widthPixels);
  output.side = SEGMENT_CORNERS[vertexIndex].y;
  output.color = vec4<f32>(getRampColor(layerStyle.field.ramp, speedRatio), age * age * layerStyle.opacity);
  return output;
}

@fragment fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
  let coverage = 1.0 - smoothstep(0.55, 1.0, abs(input.side));
  return vec4<f32>(input.color.rgb, input.color.a * coverage);
}
`;

/** Props for {@link WindTrailLayer}. */
export type WindTrailLayerProps = WindLayerProps & {
  /** Trail ring `float32x2`: particle `i` owns rows `i * ringLength` to `i * ringLength + L - 1`. */
  trailPositions: Buffer;
  /** The advection word-parameter buffer; word 1 is the frame that selects the newest slot. */
  wordParameters: Buffer;
  /** Ring length `L` the contributor was built with. */
  ringLength: number;
  /** Number of particles. */
  particleCount: number;
  /** Line width in CSS pixels. Defaults to 1.4. */
  widthPixels?: number;
  /** Maximum alpha at the head of a trail. Defaults to 0.9. */
  opacity?: number;
};

/** Particle trails drawn straight from the advection ring buffer, tinted by wind speed. */
export class WindTrailLayer extends WindModelLayer<WindTrailLayerProps> {
  static override layerName = 'WindTrailLayer';

  protected getShaderSource(): string {
    return TRAIL_SHADER;
  }
  protected getStyleByteLength(): number {
    return FIELD_STYLE_BYTES + 16;
  }
  protected getInstanceCount(): number {
    return this.props.particleCount * (this.props.ringLength - 1);
  }
  protected getStorageBindings(): Record<string, Buffer> {
    return {
      trailPositions: this.props.trailPositions,
      trailWords: this.props.wordParameters,
      speedField: this.props.speedField
    };
  }
  protected writeStyle(styleBuffer: Buffer): void {
    const {widthPixels = 1.4, opacity = 0.9, ringLength, particleCount} = this.props;
    const data = new ArrayBuffer(FIELD_STYLE_BYTES + 16);
    const floats = new Float32Array(data);
    const words = new Uint32Array(data);
    writeFieldStyle(floats, words, this.props);
    const base = FIELD_STYLE_BYTES / 4;
    floats[base] = widthPixels;
    floats[base + 1] = opacity;
    words[base + 2] = ringLength;
    words[base + 3] = particleCount;
    styleBuffer.write(new Uint8Array(data));
  }
}

// -------------------------------------------------------------------------------------------------
// Streamline polylines
// -------------------------------------------------------------------------------------------------

const PATH_SHADER = /* wgsl */ `
${FIELD_WGSL}
struct LayerStyle {
  field: FieldStyle,
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
${SEGMENT_WGSL}

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
  if (isNonFiniteWind(start.x) || isNonFiniteWind(end.x)) {
    return output;
  }
  output.position = expandSegment(
    projectWindPosition(start), projectWindPosition(end),
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

/** Props for {@link WindPathLayer}. */
export type WindPathLayerProps = WindLayerProps & {
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
export class WindPathLayer extends WindModelLayer<WindPathLayerProps> {
  static override layerName = 'WindPathLayer';

  protected getShaderSource(): string {
    return PATH_SHADER;
  }
  protected getStyleByteLength(): number {
    return FIELD_STYLE_BYTES + 32;
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
    const data = new ArrayBuffer(FIELD_STYLE_BYTES + 32);
    const floats = new Float32Array(data);
    const words = new Uint32Array(data);
    writeFieldStyle(floats, words, this.props);
    const base = FIELD_STYLE_BYTES / 4;
    floats.set([color[0] / 255, color[1] / 255, color[2] / 255, (color[3] ?? 255) / 255], base);
    floats[base + 4] = widthPixels;
    floats[base + 5] = opacity;
    styleBuffer.write(new Uint8Array(data));
  }
}
