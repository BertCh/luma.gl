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

const KEPT_SEGMENT_SHADER = /* wgsl */ `
struct KeptSegmentStyle {
  color: vec4<f32>,
  widthPixels: f32,
  opacity: f32,
  _padding0: f32,
  _padding1: f32,
};

@group(0) @binding(auto) var<uniform> keptSegmentStyle: KeptSegmentStyle;
@group(0) @binding(auto) var<storage, read> vertexPositions: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> keptIds: array<u32>;
@group(0) @binding(auto) var<storage, read> vertexLines: array<u32>;
@group(0) @binding(auto) var<storage, read> keptCount: array<u32>;

struct KeptSegmentOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) side: f32,
};

fn projectVertex(position: vec2<f32>) -> vec4<f32> {
  var clipPosition = project_position_to_clipspace(vec3<f32>(position, 0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  // Deck's project32 matrices use OpenGL depth; WebGPU clip space requires [0, w].
  clipPosition.z = (clipPosition.z + clipPosition.w) * 0.5;
  return clipPosition;
}

// Instance i connects kept vertices i and i + 1 when both belong to the same line.
@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> KeptSegmentOutput {
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
  );
  var output: KeptSegmentOutput;
  output.side = 0.0;
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  if (instanceIndex + 1u >= keptCount[0]) {
    return output;
  }
  let startRow = keptIds[instanceIndex];
  let endRow = keptIds[instanceIndex + 1u];
  if (vertexLines[startRow] != vertexLines[endRow]) {
    return output;
  }
  let startClip = projectVertex(vertexPositions[startRow]);
  let endClip = projectVertex(vertexPositions[endRow]);
  let corner = corners[vertexIndex];
  let screenDirection = (endClip.xy / endClip.w - startClip.xy / startClip.w) * project.viewportSize;
  let directionLength = length(screenDirection);
  var direction = vec2<f32>(1.0, 0.0);
  if (directionLength > 1e-6) {
    direction = screenDirection / directionLength;
  }
  let normal = vec2<f32>(-direction.y, direction.x);
  var clipPosition = mix(startClip, endClip, corner.x);
  let along = direction * (corner.x * 2.0 - 1.0) * 0.5;
  clipPosition = vec4<f32>(
    clipPosition.xy + project_pixel_size_to_clipspace((normal * corner.y + along) * keptSegmentStyle.widthPixels * 0.5),
    clipPosition.z,
    clipPosition.w
  );
  output.position = clipPosition;
  output.side = corner.y;
  return output;
}

@fragment fn fragmentMain(input: KeptSegmentOutput) -> @location(0) vec4<f32> {
  let coverage = 1.0 - smoothstep(0.55, 1.0, abs(input.side));
  return vec4<f32>(keptSegmentStyle.color.rgb, keptSegmentStyle.color.a * keptSegmentStyle.opacity * coverage);
}
`;

/** Props for {@link KeptSegmentLayer}. */
export type KeptSegmentLayerProps = LayerProps & {
  /** `float32x2` vertex positions in planar meters, one per original vertex. */
  positions: Buffer;
  /** Ascending kept vertex rows (the contributor's compact `output.ids`). */
  keptIds: Buffer;
  /** `uint32` line index of every original vertex, used to avoid joining two different lines. */
  vertexLines: Buffer;
  /** One `uint32` word: the number of valid entries in `keptIds`. */
  keptCount: Buffer;
  /** GPU-written indirect record (`vertexCount` 6) whose instance count is the kept count. */
  drawCommands: DrawCommandBuffer;
  /** Line width in CSS pixels. Defaults to 2. */
  widthPixels?: number;
  /** RGBA color with 0-255 channels. Defaults to opaque orange. */
  color?: readonly [number, number, number, number?];
};

/**
 * Draws the simplified polylines straight from the contributor's compact kept IDs: instance `i`
 * connects consecutive kept vertices `ids[i]` and `ids[i + 1]` when they belong to the same line.
 * The indirect record's instance count is the kept count, so nothing is read back to draw.
 */
export class KeptSegmentLayer extends Layer<KeptSegmentLayerProps> {
  static override layerName = 'KeptSegmentLayer';
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
      ...this.getShaders({modules: [project32], source: KEPT_SEGMENT_SHADER}),
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
    const [red, green, blue, alpha = 255] = this.props.color ?? [255, 150, 40, 255];
    styleBuffer.write(
      Float32Array.of(
        red / 255,
        green / 255,
        blue / 255,
        alpha / 255,
        this.props.widthPixels ?? 2,
        this.props.opacity ?? 1,
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
      keptSegmentStyle: styleBuffer,
      vertexPositions: this.props.positions,
      keptIds: this.props.keptIds,
      vertexLines: this.props.vertexLines,
      keptCount: this.props.keptCount
    };
  }
}
