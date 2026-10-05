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
import type {DrawCommandBuffer} from '@luma.gl/gpgpu/gpu-core';

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

/** Vertices of one vehicle marker (an arrow head triangle). */
export const VEHICLE_MARKER_VERTEX_COUNT = 3;

const SHARED_WGSL = /* wgsl */ `
fn getViridis(t: f32) -> vec3<f32> {
  let c0 = vec3<f32>(0.2777273272234177, 0.005407344544966578, 0.3340998053353061);
  let c1 = vec3<f32>(0.1050930431085774, 1.404613529898575, 1.384590162594685);
  let c2 = vec3<f32>(-0.3308618287255563, 0.214847559468213, 0.09509516302823659);
  let c3 = vec3<f32>(-4.634230498983486, -5.799100973351585, -19.33244095627987);
  let c4 = vec3<f32>(6.228269936347081, 14.17993336680509, 56.69055260068105);
  let c5 = vec3<f32>(4.776384997670288, -13.74514537774601, -65.35303263337234);
  let c6 = vec3<f32>(-5.435455855934631, 4.645852612178535, 26.3124352495832);
  return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}

fn getPlayheadClipPosition(position: vec2<f32>) -> vec4<f32> {
  var clipPosition = project_position_to_clipspace(vec3<f32>(position, 0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  // Deck's project32 matrices use OpenGL depth; WebGPU clip space requires [0, w].
  clipPosition.z = (clipPosition.z + clipPosition.w) * 0.5;
  return clipPosition;
}
`;

const VEHICLE_SHADER = /* wgsl */ `
struct VehicleStyle {
  sizePixels: f32,
  speedForFullColor: f32,
  opacity: f32,
  _padding0: f32,
};

@group(0) @binding(auto) var<uniform> vehicleStyle: VehicleStyle;
@group(0) @binding(auto) var<storage, read> vehicleIds: array<u32>;
@group(0) @binding(auto) var<storage, read> vehiclePositions: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> vehicleHeadings: array<f32>;
@group(0) @binding(auto) var<storage, read> vehicleSpeeds: array<f32>;

struct VehicleVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec3<f32>,
};
${SHARED_WGSL}

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> VehicleVertexOutput {
  // Arrow head pointing along +x before rotation: tip, left wing, right wing.
  let corners = array<vec2<f32>, 3>(
    vec2<f32>(1.0, 0.0), vec2<f32>(-0.7, 0.62), vec2<f32>(-0.7, -0.62)
  );
  let track = vehicleIds[instanceIndex];
  let heading = vehicleHeadings[track];
  let speed = vehicleSpeeds[track];
  let rotation = vec2<f32>(cos(heading), sin(heading));
  let corner = corners[vertexIndex];
  let rotated = vec2<f32>(
    corner.x * rotation.x - corner.y * rotation.y,
    corner.x * rotation.y + corner.y * rotation.x
  );
  var clipPosition = getPlayheadClipPosition(vehiclePositions[track]);
  clipPosition = vec4<f32>(
    clipPosition.xy + project_pixel_size_to_clipspace(rotated * vehicleStyle.sizePixels),
    clipPosition.z,
    clipPosition.w
  );
  var output: VehicleVertexOutput;
  output.position = clipPosition;
  output.color = getViridis(clamp(speed / vehicleStyle.speedForFullColor, 0.0, 1.0));
  return output;
}

@fragment fn fragmentMain(input: VehicleVertexOutput) -> @location(0) vec4<f32> {
  return vec4<f32>(input.color, vehicleStyle.opacity);
}
`;

/** Props for {@link VehicleMarkerLayer}. */
export type VehicleMarkerLayerProps = LayerProps & {
  /** Compact uint32 list of active track indices (contributor `activeTracks.ids`). */
  ids: Buffer;
  /** `float32x2` interpolated position per track, in planar meters (contributor `currentPositions`). */
  positions: Buffer;
  /** `float32` heading per track in radians, 0 along +x, counterclockwise (contributor `headings`). */
  headings: Buffer;
  /** `float32` speed per track in meters per second (contributor `speeds`). */
  speeds: Buffer;
  /** GPU-written indirect record with `vertexCount` 3 whose instance count is the active count. */
  drawCommands: DrawCommandBuffer;
  /** Arrow half-length in CSS pixels. Defaults to 9. */
  sizePixels?: number;
  /** Speed in meters per second at which the colormap reaches its end. Defaults to 14. */
  speedForFullColor?: number;
};

type LayerState = {model: Model; styleBuffer: Buffer};

/**
 * Draws one heading-oriented arrow per active track straight from the playhead contributor's
 * `activeTracks.ids`, `currentPositions`, `headings` and `speeds`. Track data is gathered by track
 * ID in the vertex shader and the instance count is the GPU-written indirect record, so moving
 * the playhead never touches the CPU.
 */
export class VehicleMarkerLayer extends Layer<VehicleMarkerLayerProps> {
  static override layerName = 'VehicleMarkerLayer';
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
      ...this.getShaders({modules: [project32], source: VEHICLE_SHADER}),
      id: `${this.id}-model`,
      topology: 'triangle-list',
      isInstanced: true,
      vertexCount: VEHICLE_MARKER_VERTEX_COUNT,
      instanceCount: 0,
      bufferLayout: [],
      bindings: this.getBindings(styleBuffer),
      parameters: BLEND_PARAMETERS
    });
    this.setState({model, styleBuffer});
  }

  override updateState(parameters: UpdateParameters<this>): void {
    super.updateState(parameters);
    const {model, styleBuffer} = this.state as LayerState;
    model.setBindings(this.getBindings(styleBuffer));
  }

  override getModels(): Model[] {
    return [(this.state as LayerState).model];
  }

  override draw({renderPass}: {renderPass: RenderPass}): void {
    const {model, styleBuffer} = this.state as LayerState;
    styleBuffer.write(
      Float32Array.of(
        this.props.sizePixels ?? 9,
        this.props.speedForFullColor ?? 14,
        this.props.opacity ?? 1,
        0,
        0,
        0,
        0,
        0
      )
    );
    // Bind pipeline and bindings with an empty draw, then replay the GPU-written record.
    model.setInstanceCount(0);
    model.draw(renderPass);
    this.props.drawCommands.draw(renderPass, 0);
  }

  override finalizeState(context: LayerContext): void {
    const {model, styleBuffer} = this.state as LayerState;
    model.destroy();
    styleBuffer.destroy();
    super.finalizeState(context);
  }

  private getBindings(styleBuffer: Buffer): Record<string, Buffer> {
    return {
      vehicleStyle: styleBuffer,
      vehicleIds: this.props.ids,
      vehiclePositions: this.props.positions,
      vehicleHeadings: this.props.headings,
      vehicleSpeeds: this.props.speeds
    };
  }
}

const RESAMPLED_TRAIL_SHADER = /* wgsl */ `
struct TrailStyle {
  color: vec4<f32>,
  sampleCount: u32,
  widthPixels: f32,
  _padding0: f32,
  _padding1: f32,
};

@group(0) @binding(auto) var<uniform> trailStyle: TrailStyle;
@group(0) @binding(auto) var<storage, read> trailSamples: array<vec2<f32>>;

struct TrailVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) side: f32,
};
${SHARED_WGSL}

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> TrailVertexOutput {
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
  );
  // Instance = track * (sampleCount - 1) + k joins samples k and k + 1 of that track.
  let segmentsPerTrack = trailStyle.sampleCount - 1u;
  let track = instanceIndex / segmentsPerTrack;
  let sample = track * trailStyle.sampleCount + instanceIndex % segmentsPerTrack;
  let startClip = getPlayheadClipPosition(trailSamples[sample]);
  let endClip = getPlayheadClipPosition(trailSamples[sample + 1u]);
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
    clipPosition.xy + project_pixel_size_to_clipspace(normal * corner.y * trailStyle.widthPixels * 0.5),
    clipPosition.z,
    clipPosition.w
  );
  var output: TrailVertexOutput;
  output.position = clipPosition;
  output.side = corner.y;
  return output;
}

@fragment fn fragmentMain(input: TrailVertexOutput) -> @location(0) vec4<f32> {
  let coverage = 1.0 - smoothstep(0.55, 1.0, abs(input.side));
  return vec4<f32>(trailStyle.color.rgb, trailStyle.color.a * coverage);
}
`;

/** Props for {@link ResampledTrailLayer}. */
export type ResampledTrailLayerProps = LayerProps & {
  /** Dense `float32x2` samples, row `track * sampleCount + k` (contributor `samples`). */
  samples: Buffer;
  /** Number of tracks. */
  trackCount: number;
  /** Fixed samples per track (at least 2). */
  sampleCount: number;
  /** RGBA 0-255. Defaults to a faint white. */
  color?: readonly [number, number, number, number];
  /** Line width in CSS pixels. Defaults to 1. */
  widthPixels?: number;
};

/**
 * Draws every resampled track as a fixed-length polyline: `sampleCount - 1` quads per track read
 * directly from the dense `[trackCount x sampleCount]` sample buffer.
 */
export class ResampledTrailLayer extends Layer<ResampledTrailLayerProps> {
  static override layerName = 'ResampledTrailLayer';
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
      ...this.getShaders({modules: [project32], source: RESAMPLED_TRAIL_SHADER}),
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
    const {model, styleBuffer} = this.state as LayerState;
    model.setBindings(this.getBindings(styleBuffer));
  }

  override getModels(): Model[] {
    return [(this.state as LayerState).model];
  }

  override draw({renderPass}: {renderPass: RenderPass}): void {
    const {model, styleBuffer} = this.state as LayerState;
    const {color = [255, 255, 255, 40], widthPixels = 1, sampleCount, trackCount} = this.props;
    const data = new ArrayBuffer(STYLE_BYTE_LENGTH);
    const floats = new Float32Array(data);
    floats.set([
      color[0] / 255,
      color[1] / 255,
      color[2] / 255,
      (color[3] / 255) * (this.props.opacity ?? 1)
    ]);
    new Uint32Array(data)[4] = sampleCount;
    floats[5] = widthPixels;
    styleBuffer.write(new Uint8Array(data));
    model.setInstanceCount(trackCount * (sampleCount - 1));
    model.draw(renderPass);
  }

  override finalizeState(context: LayerContext): void {
    const {model, styleBuffer} = this.state as LayerState;
    model.destroy();
    styleBuffer.destroy();
    super.finalizeState(context);
  }

  private getBindings(styleBuffer: Buffer): Record<string, Buffer> {
    return {trailStyle: styleBuffer, trailSamples: this.props.samples};
  }
}
