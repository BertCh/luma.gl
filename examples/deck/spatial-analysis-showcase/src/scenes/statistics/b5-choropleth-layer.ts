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
import {COLORMAP_INDEXES, getRampWgsl, type RampName} from '../../engine/ramps';

/** RGBA 0-255. */
export type ChoroplethColor = readonly [number, number, number, number?];

const STYLE_BYTE_LENGTH = 320;
const MAXIMUM_PALETTE_SIZE = 16;
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

const SHADER = /* wgsl */ `
struct ChoroplethStyle {
  noDataColor: vec4<f32>,
  palette: array<vec4<f32>, ${MAXIMUM_PALETTE_SIZE}>,
  valueRange: vec2<f32>,
  opacity: f32,
  mode: u32,
  colormap: u32,
  selectedRow: i32,
  paletteSize: u32,
  sqrtScale: u32,
  selectedBoost: f32,
  _padding0: u32,
  _padding1: u32,
  _padding2: u32,
};

@group(0) @binding(auto) var<uniform> choroplethStyle: ChoroplethStyle;
@group(0) @binding(auto) var<storage, read> choroplethPositions: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> choroplethFeatures: array<u32>;
@group(0) @binding(auto) var<storage, read> choroplethValues: array<u32>;

${getRampWgsl()}

struct ChoroplethVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
};

fn getChoroplethColor(row: u32) -> vec4<f32> {
  let raw = choroplethValues[row];
  var color = choroplethStyle.noDataColor;
  if (choroplethStyle.mode == 0u) {
    // Packed rgba8, r in the low byte (the GPUColorScale layout).
    color = unpack4x8unorm(raw);
  } else if (choroplethStyle.mode == 1u) {
    let isNonFinite = (raw & 0x7fffffffu) >= 0x7f800000u;
    if (!isNonFinite) {
      let value = bitcast<f32>(raw);
      let range = choroplethStyle.valueRange;
      var t = clamp((value - range.x) / max(range.y - range.x, 1e-20), 0.0, 1.0);
      if (choroplethStyle.sqrtScale != 0u) {
        t = sqrt(t);
      }
      color = vec4<f32>(clamp(spatialAnalysisSampleRamp(choroplethStyle.colormap, t), vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
    }
  } else {
    // Category: palette index, the largest value is no data.
    if (raw < choroplethStyle.paletteSize) {
      color = choroplethStyle.palette[raw];
    }
  }
  if (choroplethStyle.selectedRow >= 0 && i32(row) == choroplethStyle.selectedRow) {
    color = vec4<f32>(mix(color.rgb, vec3<f32>(1.0), choroplethStyle.selectedBoost), max(color.a, 0.9));
  }
  return color;
}

@vertex fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> ChoroplethVertexOutput {
  var output: ChoroplethVertexOutput;
  let row = choroplethFeatures[vertexIndex];
  let color = getChoroplethColor(row);
  if (color.a <= 0.0) {
    output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    output.color = vec4<f32>(0.0);
    return output;
  }
  let position = choroplethPositions[vertexIndex];
  var clip = project_position_to_clipspace(vec3<f32>(position, 0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  // Deck's project32 matrices use OpenGL depth; WebGPU clip space requires [0, w].
  clip.z = (clip.z + clip.w) * 0.5;
  output.position = clip;
  output.color = vec4<f32>(color.rgb, color.a * choroplethStyle.opacity);
  return output;
}

@fragment fn fragmentMain(input: ChoroplethVertexOutput) -> @location(0) vec4<f32> {
  return input.color;
}
`;

/** How the fill layer turns the per-feature `values` buffer into a color. */
export type ChoroplethFillLayerProps = LayerProps & {
  /** Triangle vertices, `float32x2` longitude and latitude. */
  positions: Buffer;
  /** Feature row of every triangle vertex (`uint32`). */
  featureRows: Buffer;
  /** Number of vertices to draw (three per triangle). */
  vertexCount: number;
  /** Per-feature values: packed rgba8 (`'packed'`), `float32` (`'ramp'`) or a class index (`'category'`). */
  values: Buffer;
  /** `'packed'` reads rgba8 colors, `'ramp'` normalizes a float32 through `ramp`, `'category'` indexes `palette`. */
  mode?: 'packed' | 'ramp' | 'category';
  /** Ramp name for `'ramp'`. */
  ramp?: RampName;
  /** `[min, max]` mapped to the ramp ends for `'ramp'`. */
  valueRange?: readonly [number, number];
  /** Apply `sqrt` after normalisation. */
  sqrtScale?: boolean;
  /** Palette (up to 16) for `'category'`. */
  palette?: readonly ChoroplethColor[];
  /** Color of non-finite values in `'ramp'` mode. Alpha 0 hides them. */
  noDataColor?: ChoroplethColor;
  /** Feature drawn lighter, or -1. */
  selectedRow?: number;
  /** Fill opacity multiplier, 0-1. */
  fillOpacity?: number;
};

/**
 * Filled polygons drawn straight from GPU storage buffers: a static triangulation (positions and
 * the feature row of every vertex) plus a per-feature `values` buffer that analysis contributors
 * write. A recolor never leaves the GPU. Longitude and latitude are projected by deck.gl, so the
 * fill lines up with the basemap at any scale (a continental map included).
 */
export class ChoroplethFillLayer extends Layer<ChoroplethFillLayerProps> {
  static override layerName = 'ChoroplethFillLayer';
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
      ...this.getShaders({modules: [project32], source: SHADER}),
      id: `${this.id}-model`,
      topology: 'triangle-list',
      isInstanced: true,
      vertexCount: this.props.vertexCount,
      instanceCount: 1,
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
    model.setVertexCount(this.props.vertexCount);
  }

  override getModels(): Model[] {
    return [(this.state as {model: Model}).model];
  }

  override draw({renderPass}: {renderPass: RenderPass}): void {
    const {model, styleBuffer} = this.state as {model: Model; styleBuffer: Buffer};
    this.writeStyle(styleBuffer);
    model.setInstanceCount(1);
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
      choroplethStyle: styleBuffer,
      choroplethPositions: this.props.positions,
      choroplethFeatures: this.props.featureRows,
      choroplethValues: this.props.values
    };
  }

  private writeStyle(styleBuffer: Buffer): void {
    const {
      mode = 'packed',
      ramp = 'viridis',
      valueRange = [0, 1],
      sqrtScale = false,
      palette = [],
      noDataColor = [0, 0, 0, 0],
      selectedRow = -1,
      fillOpacity = 0.85
    } = this.props;
    const data = new ArrayBuffer(STYLE_BYTE_LENGTH);
    const floats = new Float32Array(data);
    const words = new Uint32Array(data);
    const signed = new Int32Array(data);
    const writeColor = (offset: number, color: ChoroplethColor) => {
      floats.set(
        [color[0] / 255, color[1] / 255, color[2] / 255, (color[3] ?? 255) / 255],
        offset / 4
      );
    };
    writeColor(0, noDataColor);
    palette.slice(0, MAXIMUM_PALETTE_SIZE).forEach((color, index) => {
      writeColor(16 + index * 16, color);
    });
    const base = 16 + MAXIMUM_PALETTE_SIZE * 16;
    floats.set(valueRange, base / 4);
    floats[(base + 8) / 4] = fillOpacity;
    words[(base + 12) / 4] = mode === 'packed' ? 0 : mode === 'ramp' ? 1 : 2;
    words[(base + 16) / 4] = COLORMAP_INDEXES[ramp];
    signed[(base + 20) / 4] = selectedRow;
    words[(base + 24) / 4] = Math.min(palette.length, MAXIMUM_PALETTE_SIZE);
    words[(base + 28) / 4] = sqrtScale ? 1 : 0;
    floats[(base + 32) / 4] = 0.45;
    styleBuffer.write(new Uint8Array(data));
  }
}
