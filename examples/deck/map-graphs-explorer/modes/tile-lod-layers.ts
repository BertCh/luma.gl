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

/** Vertices per tile instance: 6 for the translucent fill, then 4 edges of 6 vertices each. */
export const TILE_OUTLINE_VERTEX_COUNT = 30;

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

const TILE_OUTLINE_SHADER = /* wgsl */ `
struct TileOutlineStyle {
  baseColor: vec4<f32>,
  widthPixels: f32,
  fillAlpha: f32,
  levelColors: u32,
  maximumLevel: f32,
  useCount: u32,
  _padding0: u32,
  _padding1: u32,
  _padding2: u32,
};

@group(0) @binding(auto) var<uniform> tileStyle: TileOutlineStyle;
@group(0) @binding(auto) var<storage, read> tileBounds: array<vec4<f32>>;
@group(0) @binding(auto) var<storage, read> tileLevels: array<u32>;
@group(0) @binding(auto) var<storage, read> tileIds: array<u32>;
@group(0) @binding(auto) var<storage, read> tileCount: array<u32>;

struct TileVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) side: f32,
  @location(1) color: vec4<f32>,
  @location(2) isFill: f32,
};

fn getTileLevelColor(level: u32) -> vec3<f32> {
  // Cool-to-warm ramp: coarse tiles blue, fine tiles yellow.
  let t = clamp(f32(level) / max(tileStyle.maximumLevel, 1.0), 0.0, 1.0);
  let low = vec3<f32>(0.25, 0.55, 1.0);
  let middle = vec3<f32>(0.35, 0.95, 0.65);
  let high = vec3<f32>(1.0, 0.9, 0.25);
  return select(mix(low, middle, t * 2.0), mix(middle, high, t * 2.0 - 1.0), t > 0.5);
}

fn projectTilePosition(position: vec2<f32>) -> vec4<f32> {
  var clipPosition = project_position_to_clipspace(vec3<f32>(position, 0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  // Deck's project32 matrices use OpenGL depth; WebGPU clip space requires [0, w].
  clipPosition.z = (clipPosition.z + clipPosition.w) * 0.5;
  return clipPosition;
}

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> TileVertexOutput {
  var output: TileVertexOutput;
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  output.side = 0.0;
  output.color = vec4<f32>(0.0);
  output.isFill = 0.0;
  if (tileStyle.useCount != 0u && instanceIndex >= tileCount[0]) {
    return output;
  }
  let node = tileIds[instanceIndex];
  let bounds = tileBounds[node];
  var color = tileStyle.baseColor;
  if (tileStyle.levelColors != 0u) {
    color = vec4<f32>(getTileLevelColor(tileLevels[node]), color.a);
  }
  let quadCorners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0)
  );
  if (vertexIndex < 6u) {
    let corner = quadCorners[vertexIndex];
    output.position = projectTilePosition(mix(bounds.xy, bounds.zw, corner));
    output.color = vec4<f32>(color.rgb, tileStyle.fillAlpha);
    output.isFill = 1.0;
    return output;
  }
  let edgeIndex = (vertexIndex - 6u) / 6u;
  let cornerIndex = (vertexIndex - 6u) % 6u;
  // Counter-clockwise edges: bottom, right, top, left.
  let corners = array<vec2<f32>, 4>(
    bounds.xy, vec2<f32>(bounds.z, bounds.y), bounds.zw, vec2<f32>(bounds.x, bounds.w)
  );
  let start = corners[edgeIndex];
  let end = corners[(edgeIndex + 1u) % 4u];
  // (t along the edge, side across it)
  let edgeCorners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
  );
  let corner = edgeCorners[cornerIndex];
  let startClip = projectTilePosition(start);
  let endClip = projectTilePosition(end);
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
    clipPosition.xy + project_pixel_size_to_clipspace((normal * corner.y + along) * tileStyle.widthPixels * 0.5),
    clipPosition.z,
    clipPosition.w
  );
  output.position = clipPosition;
  output.side = corner.y;
  output.color = color;
  return output;
}

@fragment fn fragmentMain(input: TileVertexOutput) -> @location(0) vec4<f32> {
  if (input.isFill > 0.5) {
    return input.color;
  }
  let coverage = 1.0 - smoothstep(0.55, 1.0, abs(input.side));
  return vec4<f32>(input.color.rgb, input.color.a * coverage);
}
`;

/** Props for {@link TileOutlineLayer}. */
export type TileOutlineLayerProps = LayerProps & {
  /** `float32x4` per quadtree node: `minX, minY, maxX, maxY` in meters around `coordinateOrigin`. */
  bounds: Buffer;
  /** `uint32` quadtree level per node, used by `levelColors`. */
  levels: Buffer;
  /** Compact tile IDs (quadtree node indices): instance `i` draws node `ids[i]`. */
  ids: Buffer;
  /** GPU-written indirect record (`vertexCount` = {@link TILE_OUTLINE_VERTEX_COUNT}), or `null`. */
  drawCommands?: DrawCommandBuffer | null;
  /** Record index in `drawCommands`. Defaults to 0. */
  drawCommandIndex?: number;
  /** Fixed instance count when `drawCommands` is not set. */
  instanceCount?: number;
  /** Optional GPU `uint32` count buffer: instances at or beyond `count[0]` are skipped. */
  countBuffer?: Buffer | null;
  /** Color the tile by level (blue coarse to yellow fine) instead of `color`. */
  levelColors?: boolean;
  /** Level count minus one, the top of the color ramp. */
  maximumLevel?: number;
  /** Outline color when `levelColors` is false, and the alpha of every outline. */
  color?: readonly [number, number, number, number];
  /** Outline width in pixels. Defaults to 1.5. */
  widthPixels?: number;
  /** Alpha of the translucent tile fill. Defaults to 0. */
  fillAlpha?: number;
};

type TileOutlineLayerState = {
  model: Model | null;
  styleBuffer: Buffer | null;
  placeholderBuffer: Buffer | null;
};

/**
 * Draws tile rectangles (translucent fill plus a four-edge outline per instance) from GPU buffers.
 * The instance IDs come from the recipe's compact output and the instance count from its indirect
 * draw record, so nothing about the selection is known to the CPU.
 */
export class TileOutlineLayer extends Layer<TileOutlineLayerProps> {
  static override layerName = 'TileOutlineLayer';
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
    const placeholderBuffer = device.createBuffer({
      id: `${this.id}-placeholder`,
      byteLength: 16,
      usage: Buffer.STORAGE | Buffer.COPY_DST
    });
    const model = new Model(device, {
      ...this.getShaders({modules: [project32], source: TILE_OUTLINE_SHADER}),
      id: `${this.id}-model`,
      topology: 'triangle-list',
      isInstanced: true,
      vertexCount: TILE_OUTLINE_VERTEX_COUNT,
      instanceCount: 0,
      bufferLayout: [],
      bindings: this.getTileBindings(placeholderBuffer, styleBuffer),
      parameters: BLEND_PARAMETERS
    });
    this.setState({model, styleBuffer, placeholderBuffer} satisfies TileOutlineLayerState);
  }

  override updateState(parameters: UpdateParameters<this>): void {
    super.updateState(parameters);
    const {model, styleBuffer, placeholderBuffer} = this.state as TileOutlineLayerState;
    if (model && styleBuffer && placeholderBuffer) {
      model.setBindings(this.getTileBindings(placeholderBuffer, styleBuffer));
    }
  }

  override getModels(): Model[] {
    const model = (this.state as TileOutlineLayerState).model;
    return model ? [model] : [];
  }

  override draw({renderPass}: {renderPass: RenderPass}): void {
    const {model, styleBuffer} = this.state as TileOutlineLayerState;
    if (!model || !styleBuffer) return;
    this.writeTileStyle(styleBuffer);
    const {drawCommands, drawCommandIndex = 0, instanceCount = 0} = this.props;
    if (drawCommands) {
      // Bind pipeline and bindings with an empty draw, then replay the GPU-written record.
      model.setInstanceCount(0);
      model.draw(renderPass);
      drawCommands.draw(renderPass, drawCommandIndex);
    } else {
      model.setInstanceCount(instanceCount);
      model.draw(renderPass);
    }
  }

  override finalizeState(context: LayerContext): void {
    const {model, styleBuffer, placeholderBuffer} = this.state as TileOutlineLayerState;
    model?.destroy();
    styleBuffer?.destroy();
    placeholderBuffer?.destroy();
    this.setState({model: null, styleBuffer: null, placeholderBuffer: null});
    super.finalizeState(context);
  }

  private getTileBindings(placeholder: Buffer, styleBuffer: Buffer): Record<string, Buffer> {
    return {
      tileStyle: styleBuffer,
      tileBounds: this.props.bounds,
      tileLevels: this.props.levels,
      tileIds: this.props.ids,
      tileCount: this.props.countBuffer ?? placeholder
    };
  }

  private writeTileStyle(styleBuffer: Buffer): void {
    const {
      color = [255, 255, 255, 255],
      widthPixels = 1.5,
      fillAlpha = 0,
      levelColors = false,
      maximumLevel = 1,
      countBuffer
    } = this.props;
    const data = new ArrayBuffer(STYLE_BYTE_LENGTH);
    const floats = new Float32Array(data);
    const words = new Uint32Array(data);
    floats.set([color[0] / 255, color[1] / 255, color[2] / 255, color[3] / 255], 0);
    floats[4] = widthPixels;
    floats[5] = fillAlpha;
    words[6] = levelColors ? 1 : 0;
    floats[7] = maximumLevel;
    words[8] = countBuffer ? 1 : 0;
    styleBuffer.write(new Uint8Array(data));
  }
}
