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
import type {DrawCommandBuffer} from '@luma.gl/gpgpu/gpu-core';
import {Model} from '@luma.gl/engine';

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

const STOP_SHADER = /* wgsl */ `
struct StopStyle {
  baseRadiusPixels: f32,
  radiusPerSqrtSecond: f32,
  maximumRadiusPixels: f32,
  durationForFullColor: f32,
  opacity: f32,
  ringWidthPixels: f32,
  _padding0: f32,
  _padding1: f32,
};

@group(0) @binding(auto) var<uniform> stopStyle: StopStyle;
@group(0) @binding(auto) var<storage, read> stopCentroids: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> stopDurations: array<f32>;

struct StopVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) corner: vec2<f32>,
  @location(1) color: vec3<f32>,
  @location(2) ringStart: f32,
};

// Light pink for short dwells through hot pink to deep red for the longest.
fn getStopColor(t: f32) -> vec3<f32> {
  let low = vec3<f32>(1.0, 0.78, 0.88);
  let middle = vec3<f32>(1.0, 0.16, 0.5);
  let high = vec3<f32>(0.75, 0.0, 0.2);
  if (t < 0.5) {
    return mix(low, middle, t * 2.0);
  }
  return mix(middle, high, (t - 0.5) * 2.0);
}

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> StopVertexOutput {
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
  );
  var output: StopVertexOutput;
  let duration = stopDurations[instanceIndex];
  let centroid = stopCentroids[instanceIndex];
  let radiusPixels = min(
    stopStyle.baseRadiusPixels + stopStyle.radiusPerSqrtSecond * sqrt(max(duration, 0.0)),
    stopStyle.maximumRadiusPixels
  );
  let corner = corners[vertexIndex];
  var clipPosition = project_position_to_clipspace(vec3<f32>(centroid, 0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  // Deck's project32 matrices use OpenGL depth; WebGPU clip space requires [0, w].
  clipPosition.z = (clipPosition.z + clipPosition.w) * 0.5;
  clipPosition = vec4<f32>(
    clipPosition.xy + project_pixel_size_to_clipspace(corner * radiusPixels),
    clipPosition.z,
    clipPosition.w
  );
  output.position = clipPosition;
  output.corner = corner;
  output.color = getStopColor(clamp(sqrt(duration / stopStyle.durationForFullColor), 0.0, 1.0));
  output.ringStart = 1.0 - stopStyle.ringWidthPixels / radiusPixels;
  return output;
}

@fragment fn fragmentMain(input: StopVertexOutput) -> @location(0) vec4<f32> {
  let radius = length(input.corner);
  if (radius > 1.0) { discard; }
  let edge = 1.0 - smoothstep(0.92, 1.0, radius);
  let ring = smoothstep(input.ringStart - 0.04, input.ringStart, radius);
  let rgb = mix(input.color, vec3<f32>(1.0), ring);
  let alpha = mix(0.72, 1.0, ring) * stopStyle.opacity * edge;
  return vec4<f32>(rgb, alpha);
}
`;

/** Props for {@link TrajectoryStopLayer}. */
export type TrajectoryStopLayerProps = LayerProps & {
  /** `float32x2` stop centroids in planar meters (rows past the stop count are never drawn). */
  centroids: Buffer;
  /** `float32` stop durations in seconds, aligned with `centroids`. */
  durations: Buffer;
  /** GPU-written indirect record with `vertexCount` 6 whose instance count is the stop count. */
  drawCommands: DrawCommandBuffer;
  /** Radius in CSS pixels of a zero-duration stop. Defaults to 3. */
  baseRadiusPixels?: number;
  /** Extra radius in CSS pixels per square root of a second of dwell. Defaults to 0.45. */
  radiusPerSqrtSecond?: number;
  /** Upper bound of the radius in CSS pixels. Defaults to 14. */
  maximumRadiusPixels?: number;
  /** Duration in seconds at which the color reaches its darkest value. Defaults to 600. */
  durationForFullColor?: number;
  /** Ring outline width in CSS pixels. Defaults to 1.5. */
  ringWidthPixels?: number;
};

/**
 * Draws one outlined disc per stop straight from the contributor's centroid and duration buffers. The
 * radius grows with the square root of the dwell duration and the fill color follows it. The
 * instance count is a GPU-written indirect draw record, so nothing is read back to draw.
 */
export class TrajectoryStopLayer extends Layer<TrajectoryStopLayerProps> {
  static override layerName = 'TrajectoryStopLayer';
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
      ...this.getShaders({modules: [project32], source: STOP_SHADER}),
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
    styleBuffer.write(
      Float32Array.of(
        this.props.baseRadiusPixels ?? 3,
        this.props.radiusPerSqrtSecond ?? 0.45,
        this.props.maximumRadiusPixels ?? 14,
        this.props.durationForFullColor ?? 600,
        this.props.opacity ?? 1,
        this.props.ringWidthPixels ?? 1.5,
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
    const {model, styleBuffer} = this.state as {model: Model; styleBuffer: Buffer};
    model.destroy();
    styleBuffer.destroy();
    super.finalizeState(context);
  }

  private getBindings(styleBuffer: Buffer): Record<string, Buffer> {
    return {
      stopStyle: styleBuffer,
      stopCentroids: this.props.centroids,
      stopDurations: this.props.durations
    };
  }
}
