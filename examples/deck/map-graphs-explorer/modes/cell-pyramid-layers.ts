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

const QUADBIN_CELL_SHADER = /* wgsl */ `
struct QuadbinStyle {
  // log10 points per square kilometer at the low and high end of the colormap, opacity, inset
  range: vec4<f32>,
};

@group(0) @binding(auto) var<uniform> quadbinStyle: QuadbinStyle;
@group(0) @binding(auto) var<storage, read> quadbinKeysEven: array<vec2<u32>>;
@group(0) @binding(auto) var<storage, read> quadbinKeysOdd: array<vec2<u32>>;
@group(0) @binding(auto) var<storage, read> quadbinCountsEven: array<u32>;
@group(0) @binding(auto) var<storage, read> quadbinCountsOdd: array<u32>;
@group(0) @binding(auto) var<storage, read> quadbinActiveLevel: array<u32>;
@group(0) @binding(auto) var<storage, read> quadbinFirstRow: array<u32>;

struct QuadbinVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
};

// Gathers the even-numbered bits of a 32-bit word into its low 16 bits.
fn compactEvenBits(value: u32) -> u32 {
  var bits = value & 0x55555555u;
  bits = (bits | (bits >> 1u)) & 0x33333333u;
  bits = (bits | (bits >> 2u)) & 0x0f0f0f0fu;
  bits = (bits | (bits >> 4u)) & 0x00ff00ffu;
  bits = (bits | (bits >> 8u)) & 0x0000ffffu;
  return bits;
}

// Decodes a Quadbin key stored as (low, high) words into (tileX, tileY, zoom). The 52-bit path
// interleaves the tile bits as y1 x1 y0 x0 ..., most significant pair first, then pads with ones.
fn decodeQuadbin(key: vec2<u32>) -> vec3<u32> {
  let zoom = (key.y >> 20u) & 0x1fu;
  // Morton word = path << 12 as a 64-bit integer, split into (high, low).
  let mortonHigh = (key.y << 12u) | (key.x >> 20u);
  let mortonLow = key.x << 12u;
  let x32 = compactEvenBits(mortonLow) | (compactEvenBits(mortonHigh) << 16u);
  let y32 = compactEvenBits(mortonLow >> 1u) | (compactEvenBits(mortonHigh >> 1u) << 16u);
  let shift = 32u - zoom;
  return vec3<u32>(x32 >> shift, y32 >> shift, zoom);
}

fn getMercatorLatitude(tileY: f32, tileCount: f32) -> f32 {
  return degrees(atan(sinh(3.14159265358979 * (1.0 - 2.0 * tileY / tileCount))));
}

// Inferno-like ramp.
fn getCountColor(t: f32) -> vec3<f32> {
  let stops = array<vec3<f32>, 5>(
    vec3<f32>(0.001, 0.0, 0.016),
    vec3<f32>(0.259, 0.039, 0.406),
    vec3<f32>(0.578, 0.148, 0.404),
    vec3<f32>(0.929, 0.411, 0.145),
    vec3<f32>(0.988, 1.0, 0.644)
  );
  let scaled = clamp(t, 0.0, 1.0) * 4.0;
  let index = min(u32(floor(scaled)), 3u);
  return mix(stops[index], stops[index + 1u], scaled - f32(index));
}

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> QuadbinVertexOutput {
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0)
  );
  var output: QuadbinVertexOutput;
  // Even and odd pyramid levels live in alternating slabs; the level is the GPU-visible word that
  // GPUCellLevelSelection also reads.
  let isOdd = (quadbinActiveLevel[0] & 1u) == 1u;
  let row = quadbinFirstRow[0] + instanceIndex;
  let count = select(quadbinCountsEven[row], quadbinCountsOdd[row], isOdd);
  if (count == 0u) {
    output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    output.color = vec4<f32>(0.0);
    return output;
  }
  let tile = decodeQuadbin(select(quadbinKeysEven[row], quadbinKeysOdd[row], isOdd));
  let tileCount = f32(1u << tile.z);
  let inset = quadbinStyle.range.w;
  let corner = mix(vec2<f32>(inset), vec2<f32>(1.0 - inset), corners[vertexIndex]);
  let longitude = (f32(tile.x) + corner.x) / tileCount * 360.0 - 180.0;
  let latitude = getMercatorLatitude(f32(tile.y) + corner.y, tileCount);
  var clip = project_position_to_clipspace(
    vec3<f32>(longitude, latitude, 0.0), vec3<f32>(0.0), vec3<f32>(0.0)
  );
  // Deck's project32 matrices use OpenGL depth; WebGPU clip space requires [0, w].
  clip.z = (clip.z + clip.w) * 0.5;
  output.position = clip;

  // Points per square kilometer: a Web Mercator tile is stretched by 1 / cos(latitude).
  let centerLatitude = getMercatorLatitude(f32(tile.y) + 0.5, tileCount);
  let edgeKilometers = 40075.016686 / tileCount * cos(radians(centerLatitude));
  let density = f32(count) / (edgeKilometers * edgeKilometers);
  let t = (log2(max(density, 1.0e-3)) * 0.30103 - quadbinStyle.range.x) /
    max(quadbinStyle.range.y - quadbinStyle.range.x, 1.0e-3);
  output.color = vec4<f32>(getCountColor(t), quadbinStyle.range.z);
  return output;
}

@fragment fn fragmentMain(input: QuadbinVertexOutput) -> @location(0) vec4<f32> {
  return input.color;
}
`;

/** Props for {@link QuadbinCellLayer}. */
export type QuadbinCellLayerProps = LayerProps & {
  /** `uint32x2` Quadbin keys as (low, high) words of even pyramid levels. */
  cellsEven: Buffer;
  /** `uint32x2` Quadbin keys of odd pyramid levels. */
  cellsOdd: Buffer;
  /** `uint32` point count per cell row of even levels. */
  countsEven: Buffer;
  /** `uint32` point count per cell row of odd levels. */
  countsOdd: Buffer;
  /** One-word first row of the active level, written by `GPUCellLevelSelection`. */
  firstRow: Buffer;
  /** One-word active level index, already clamped to the pyramid, whose parity picks the slab. */
  activeLevel: Buffer;
  /**
   * GPU-written indirect record (six vertices). Its instance count is the number of occupied
   * rows of the active level. Its first instance stays 0 because the shell does not request the
   * `indirect-first-instance` device feature; the shader adds `firstRow` instead.
   */
  drawCommands: DrawCommandBuffer;
  /** `[minimum, maximum]` of log10 points per square kilometer mapped to the colormap. */
  densityLogRange: readonly [number, number];
  /** Fill opacity, 0-1. Defaults to 0.78. */
  opacity?: number;
  /** Fraction of the cell edge trimmed on every side so neighbors show a seam. Defaults to 0.02. */
  inset?: number;
};

/**
 * Draws Quadbin cells as Web Mercator squares. The vertex shader decodes the two-word key into
 * tile x, y and zoom and colors by points per square kilometer, so cell tables never leave the
 * GPU and a level switch only changes the indirect draw record.
 */
export class QuadbinCellLayer extends Layer<QuadbinCellLayerProps> {
  static override layerName = 'QuadbinCellLayer';
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
      byteLength: STYLE_BYTE_LENGTH,
      usage: Buffer.UNIFORM | Buffer.COPY_DST
    });
    const model = new Model(device, {
      ...this.getShaders({modules: [project32], source: QUADBIN_CELL_SHADER}),
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
    const {densityLogRange, opacity = 0.78, inset = 0.02} = this.props;
    styleBuffer.write(Float32Array.of(densityLogRange[0], densityLogRange[1], opacity, inset));
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
      quadbinStyle: styleBuffer,
      quadbinKeysEven: this.props.cellsEven,
      quadbinKeysOdd: this.props.cellsOdd,
      quadbinCountsEven: this.props.countsEven,
      quadbinCountsOdd: this.props.countsOdd,
      quadbinActiveLevel: this.props.activeLevel,
      quadbinFirstRow: this.props.firstRow
    };
  }
}
