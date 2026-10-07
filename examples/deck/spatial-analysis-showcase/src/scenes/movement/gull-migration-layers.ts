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

/** A 0-255 RGBA color. */
export type GullVertexColor = readonly [number, number, number, number?];

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

const GULL_KEPT_VERTEX_SHADER = /* wgsl */ `
struct GullKeptVertexStyle {
  color: vec4<f32>,
  radiusPixels: f32,
  opacity: f32,
  _padding0: f32,
  _padding1: f32,
};

@group(0) @binding(auto) var<uniform> gullKeptVertexStyle: GullKeptVertexStyle;
@group(0) @binding(auto) var<storage, read> gullPositions: array<vec2<f32>>;
// This is the compact output.ids list written by GPULineSimplification, not a CPU-expanded list.
@group(0) @binding(auto) var<storage, read> gullKeptIds: array<u32>;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) corner: vec2<f32>,
};

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
  );
  let corner = corners[vertexIndex];
  let position = gullPositions[gullKeptIds[instanceIndex]];
  var clipPosition = project_position_to_clipspace(
    vec3<f32>(position, 0.0), vec3<f32>(0.0), vec3<f32>(0.0)
  );
  // Deck's project32 matrices use OpenGL depth; WebGPU clip space requires [0, w].
  clipPosition.z = (clipPosition.z + clipPosition.w) * 0.5;
  var output: VertexOutput;
  output.position = vec4<f32>(
    clipPosition.xy + project_pixel_size_to_clipspace(corner * gullKeptVertexStyle.radiusPixels),
    clipPosition.z,
    clipPosition.w
  );
  output.corner = corner;
  return output;
}

@fragment fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
  let radiusSquared = dot(input.corner, input.corner);
  if (radiusSquared > 1.0) { discard; }
  let rim = smoothstep(0.5, 0.82, radiusSquared);
  let coverage = 1.0 - smoothstep(0.86, 1.0, radiusSquared);
  let rgb = mix(gullKeptVertexStyle.color.rgb, gullKeptVertexStyle.color.rgb * 0.35, rim);
  return vec4<f32>(rgb, gullKeptVertexStyle.color.a * gullKeptVertexStyle.opacity * coverage);
}
`;

type LayerState = {model: Model; styleBuffer: Buffer};

/** Props for {@link GullKeptVertexLayer}. */
export type GullKeptVertexLayerProps = LayerProps & {
  /** `float32x2` longitude/latitude for every original gull vertex. */
  positions: Buffer;
  /** Ascending, compact original-vertex ids written by GPULineSimplification. */
  keptIds: Buffer;
  /** GPU-written record whose instance count is the compact kept-id count. */
  drawCommands: DrawCommandBuffer;
  color?: GullVertexColor;
  radiusPixels?: number;
  opacity?: number;
};

/** Draws the vertices retained by simplification without reading compact ids back to the CPU. */
export class GullKeptVertexLayer extends Layer<GullKeptVertexLayerProps> {
  static override layerName = 'GullKeptVertexLayer';
  static override defaultProps = {
    coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
    parameters: BLEND_PARAMETERS
  };

  override getAttributeManager() {
    return null;
  }

  override initializeState({device}: LayerContext): void {
    if (device.type !== 'webgpu') throw new Error(`${this.id} requires WebGPU`);
    const styleBuffer = device.createBuffer({
      id: `${this.id}-style`,
      byteLength: 32,
      usage: Buffer.UNIFORM | Buffer.COPY_DST
    });
    const model = new Model(device, {
      ...this.getShaders({modules: [project32], source: GULL_KEPT_VERTEX_SHADER}),
      id: `${this.id}-model`,
      topology: 'triangle-list',
      isInstanced: true,
      vertexCount: 6,
      instanceCount: 0,
      bufferLayout: [],
      bindings: {
        gullKeptVertexStyle: styleBuffer,
        gullPositions: this.props.positions,
        gullKeptIds: this.props.keptIds
      },
      parameters: BLEND_PARAMETERS
    });
    this.setState({model, styleBuffer} satisfies LayerState);
  }

  override updateState(parameters: UpdateParameters<this>): void {
    super.updateState(parameters);
    const {model, styleBuffer} = this.state as LayerState;
    model.setBindings({
      gullKeptVertexStyle: styleBuffer,
      gullPositions: this.props.positions,
      gullKeptIds: this.props.keptIds
    });
  }

  override getModels(): Model[] {
    return [(this.state as LayerState).model];
  }

  override draw({renderPass}: {renderPass: RenderPass}): void {
    const {model, styleBuffer} = this.state as LayerState;
    const [red, green, blue, alpha = 255] = this.props.color ?? [255, 150, 40, 255];
    styleBuffer.write(
      Float32Array.of(
        red / 255,
        green / 255,
        blue / 255,
        alpha / 255,
        this.props.radiusPixels ?? 3.8,
        this.props.opacity ?? 1,
        0,
        0
      )
    );
    // Prime the pipeline bindings, then replay the contributor-owned indirect record.
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
}
