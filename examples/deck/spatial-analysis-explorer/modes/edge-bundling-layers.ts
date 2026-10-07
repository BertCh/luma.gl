// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  Layer,
  project32,
  type LayerContext,
  type LayerProps,
  type UpdateParameters
} from '@deck.gl/core';
import {Buffer, type RenderPass} from '@luma.gl/core';
import {Model} from '@luma.gl/engine';
import type {DrawCommandBuffer} from '@luma.gl/gpgpu/gpu-core';

const STYLE_BYTE_LENGTH = 48;
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

const BUNDLED_PATH_SHADER = /* wgsl */ `
struct BundledPathStyle {
  startColor: vec4<f32>,
  endColor: vec4<f32>,
  pointsPerPath: u32,
  opacity: f32,
  padding: vec2<f32>,
};

@group(0) @binding(auto) var<uniform> pathStyle: BundledPathStyle;
@group(0) @binding(auto) var<storage, read> paths: array<vec2<f32>>;

struct PathVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
};

@vertex fn vertexMain(
  @builtin(vertex_index) pointIndex: u32,
  @builtin(instance_index) pathIndex: u32
) -> PathVertexOutput {
  var output: PathVertexOutput;
  // Same row layout the contributor writes: edge e, point i at row e * pointsPerPath + i.
  let position = paths[pathIndex * pathStyle.pointsPerPath + pointIndex];
  var clipPosition = project_position_to_clipspace(vec3<f32>(position, 0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  // Deck's project32 matrices use OpenGL depth; WebGPU clip space requires [0, w].
  clipPosition.z = (clipPosition.z + clipPosition.w) * 0.5;
  output.position = clipPosition;
  let t = f32(pointIndex) / f32(max(pathStyle.pointsPerPath, 2u) - 1u);
  var color = mix(pathStyle.startColor, pathStyle.endColor, t);
  color.a = color.a * pathStyle.opacity;
  output.color = color;
  return output;
}

@fragment fn fragmentMain(input: PathVertexOutput) -> @location(0) vec4<f32> {
  return input.color;
}
`;

/** Props for {@link BundledPathLayer}. */
export type BundledPathLayerProps = LayerProps & {
  /** `float32x2` `[longitude, latitude]` polyline rows, edge-major (contributor `paths`). */
  paths: Buffer;
  /** Indirect record `[pointsPerPath, pathCount, 0, 0]` written by the contributor `drawRecord`. */
  drawCommands: DrawCommandBuffer;
  /** Control points per path (the record's vertex count). */
  pointsPerPath: number;
  /** Color at the first point of every path, RGBA 0-255. */
  startColor?: readonly [number, number, number, number];
  /** Color at the last point of every path, RGBA 0-255. */
  endColor?: readonly [number, number, number, number];
};

type BundledPathLayerState = {
  model: Model;
  styleBuffer: Buffer;
};

/**
 * Draws bundled polylines as instanced line strips. The vertex shader reads the contributor's
 * `paths` rows straight from a storage buffer (`paths[instance * pointsPerPath + vertex]`), and
 * the vertex and instance counts come from the GPU-written indirect draw record, so the CPU never
 * touches the geometry. Translucent colors blend over each other, so crowded bundles read as denser.
 */
export class BundledPathLayer extends Layer<BundledPathLayerProps> {
  static override layerName = 'BundledPathLayer';
  static override defaultProps = {parameters: BLEND_PARAMETERS};

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
      ...this.getShaders({modules: [project32], source: BUNDLED_PATH_SHADER}),
      id: `${this.id}-model`,
      topology: 'line-strip',
      isInstanced: true,
      vertexCount: this.props.pointsPerPath,
      instanceCount: 0,
      bufferLayout: [],
      bindings: this.getBindings(styleBuffer),
      parameters: BLEND_PARAMETERS
    });
    this.setState({model, styleBuffer});
  }

  override updateState(parameters: UpdateParameters<this>): void {
    super.updateState(parameters);
    const {model, styleBuffer} = this.state as BundledPathLayerState;
    model.setBindings(this.getBindings(styleBuffer));
  }

  override getModels(): Model[] {
    return [(this.state as BundledPathLayerState).model];
  }

  override draw({renderPass}: {renderPass: RenderPass}): void {
    const {model, styleBuffer} = this.state as BundledPathLayerState;
    this.writeStyle(styleBuffer);
    // Bind pipeline and bindings with an empty draw, then replay the GPU-written record.
    model.setInstanceCount(0);
    model.draw(renderPass);
    this.props.drawCommands.draw(renderPass, 0);
  }

  override finalizeState(context: LayerContext): void {
    const {model, styleBuffer} = this.state as BundledPathLayerState;
    model.destroy();
    styleBuffer.destroy();
    super.finalizeState(context);
  }

  private getBindings(styleBuffer: Buffer): Record<string, Buffer> {
    return {pathStyle: styleBuffer, paths: this.props.paths};
  }

  private writeStyle(styleBuffer: Buffer): void {
    const props = this.props;
    const data = new ArrayBuffer(STYLE_BYTE_LENGTH);
    const floats = new Float32Array(data);
    const words = new Uint32Array(data);
    const startColor = props.startColor ?? [255, 150, 60, 255];
    const endColor = props.endColor ?? [60, 220, 255, 255];
    for (let channel = 0; channel < 4; channel++) {
      floats[channel] = startColor[channel] / 255;
      floats[4 + channel] = endColor[channel] / 255;
    }
    words[8] = props.pointsPerPath;
    floats[9] = props.opacity ?? 1;
    styleBuffer.write(new Uint8Array(data));
  }
}
