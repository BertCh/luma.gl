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

/**
 * Polygon fill layer of the regression chapter: one instance per triangle of a one-time
 * triangulation, colored in the vertex shader from a GPU storage buffer of per-feature values.
 * The values are never read back to draw: a regression, a regionalization or an accessibility
 * result written by a contributor appears on the map as soon as the buffer changes.
 *
 * Triangle corners are `lng, lat` degrees (`COORDINATE_SYSTEM.LNGLAT`), so continental data such
 * as counties lines up with the basemap. Triangle `t` draws value row `owners[t]`; the value is
 * `values[row * valueStride + valueOffset]`, which lets one layer draw one column of an interleaved
 * coefficient table.
 */
const FILL_SHADER = /* wgsl */ `
struct FillStyle {
  baseColor: vec4<f32>,
  noDataColor: vec4<f32>,
  palette: array<vec4<f32>, 8>,
  valueRange: vec2<f32>,
  opacity: f32,
  colormap: u32,
  valueFormat: u32,
  valueStride: u32,
  valueOffset: u32,
  paletteSize: u32,
  noDataValue: u32,
  selectedRow: u32,
};

@group(0) @binding(auto) var<uniform> fillStyle: FillStyle;
@group(0) @binding(auto) var<storage, read> fillTriangles: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> fillOwners: array<u32>;
@group(0) @binding(auto) var<storage, read> fillValues: array<u32>;

${getRampWgsl()}

fn getFillColor(row: u32) -> vec4<f32> {
  let colormap = fillStyle.colormap;
  if (colormap == 0u || fillStyle.valueFormat == 0u) {
    return fillStyle.baseColor;
  }
  let raw = fillValues[row * fillStyle.valueStride + fillStyle.valueOffset];
  if (fillStyle.valueFormat == 1u && raw == fillStyle.noDataValue) {
    return fillStyle.noDataColor;
  }
  if (colormap == 4u) {
    var index = raw;
    if (fillStyle.valueFormat == 2u) {
      index = u32(max(bitcast<f32>(raw), 0.0));
    }
    return fillStyle.palette[index % max(fillStyle.paletteSize, 1u)];
  }
  if (colormap == 5u) {
    let on = select(raw != 0u, bitcast<f32>(raw) != 0.0, fillStyle.valueFormat == 2u);
    return select(fillStyle.noDataColor, fillStyle.baseColor, on);
  }
  let value = select(f32(raw), bitcast<f32>(raw), fillStyle.valueFormat == 2u);
  let isNonFinite = fillStyle.valueFormat == 2u && (raw & 0x7fffffffu) >= 0x7f800000u;
  if (isNonFinite) {
    return fillStyle.noDataColor;
  }
  let range = fillStyle.valueRange;
  let t = clamp((value - range.x) / max(range.y - range.x, 1e-20), 0.0, 1.0);
  let rgb = spatialAnalysisSampleRamp(colormap, t);
  return vec4<f32>(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)), fillStyle.baseColor.a);
}

struct FillOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
};

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) triangle: u32
) -> FillOutput {
  var output: FillOutput;
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  output.color = vec4<f32>(0.0);
  let row = fillOwners[triangle];
  if (row == 0xffffffffu) {
    return output;
  }
  var color = getFillColor(row);
  if (color.a <= 0.0) {
    return output;
  }
  if (row == fillStyle.selectedRow) {
    color = vec4<f32>(mix(color.rgb, vec3<f32>(1.0), 0.45), 1.0);
  }
  var clipPosition = project_position_to_clipspace(
    vec3<f32>(fillTriangles[triangle * 3u + vertexIndex], 0.0),
    vec3<f32>(0.0),
    vec3<f32>(0.0)
  );
  // Deck's project32 matrices use OpenGL depth; WebGPU clip space requires [0, w].
  clipPosition.z = (clipPosition.z + clipPosition.w) * 0.5;
  output.position = clipPosition;
  output.color = vec4<f32>(color.rgb, color.a * fillStyle.opacity);
  return output;
}

@fragment fn fragmentMain(input: FillOutput) -> @location(0) vec4<f32> {
  return input.color;
}
`;

const STYLE_BYTE_LENGTH = 208;
const BLEND = {
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

/** RGBA 0-255. */
export type FillColor = readonly [number, number, number, number?];

/** Props of {@link PolygonFillLayer}. */
export type PolygonFillLayerProps = LayerProps & {
  /** `float32x2` triangle corners (`lng, lat`), three per triangle. */
  triangles: Buffer;
  /** uint32 value row of every triangle; `0xffffffff` hides it. */
  owners: Buffer;
  triangleCount: number;
  /** Per-row values. Omit for a uniform color. */
  values?: Buffer | null;
  valueFormat?: 'uint32' | 'float32';
  /** Words per value row (an interleaved coefficient table has several). Defaults to 1. */
  valueStride?: number;
  /** Word of the row to color by. Defaults to 0. */
  valueOffset?: number;
  /** A ramp, `'category'` (palette by value), `'mask'` or `'uniform'`. */
  colormap?: 'uniform' | 'category' | 'mask' | RampName;
  valueRange?: readonly [number, number];
  /** Color of `uniform` and the alpha of ramps. */
  color?: FillColor;
  noDataColor?: FillColor;
  noDataValue?: number;
  /** Up to 8 colors for `category`. */
  palette?: readonly FillColor[];
  opacity?: number;
  /** Value row drawn brighter, or -1. */
  selectedRow?: number;
};

type FillState = {model: Model; styleBuffer: Buffer; placeholder: Buffer};

/** One instance per triangle, colored from a storage buffer. See the file comment. */
export class PolygonFillLayer extends Layer<PolygonFillLayerProps> {
  static override layerName = 'PolygonFillLayer';
  static override defaultProps = {
    coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
    parameters: BLEND
  };

  override getAttributeManager() {
    return null;
  }

  private getBindings(
    state: Pick<FillState, 'styleBuffer' | 'placeholder'>
  ): Record<string, Buffer> {
    return {
      fillStyle: state.styleBuffer,
      fillTriangles: this.props.triangles,
      fillOwners: this.props.owners,
      fillValues: this.props.values ?? state.placeholder
    };
  }

  override initializeState({device}: LayerContext): void {
    if (device.type !== 'webgpu') throw new Error(`${this.id} requires WebGPU`);
    const styleBuffer = device.createBuffer({
      id: `${this.id}-style`,
      byteLength: STYLE_BYTE_LENGTH,
      usage: Buffer.UNIFORM | Buffer.COPY_DST
    });
    const placeholder = device.createBuffer({
      id: `${this.id}-placeholder`,
      byteLength: 16,
      usage: Buffer.STORAGE | Buffer.COPY_DST
    });
    const model = new Model(device, {
      ...this.getShaders({modules: [project32], source: FILL_SHADER}),
      id: `${this.id}-model`,
      topology: 'triangle-list',
      isInstanced: true,
      vertexCount: 3,
      instanceCount: 0,
      bufferLayout: [],
      bindings: this.getBindings({styleBuffer, placeholder}),
      parameters: BLEND
    });
    this.setState({model, styleBuffer, placeholder} satisfies FillState);
  }

  override updateState(parameters: UpdateParameters<this>): void {
    super.updateState(parameters);
    const state = this.state as FillState;
    state.model.setBindings(this.getBindings(state));
  }

  override getModels(): Model[] {
    return [(this.state as FillState).model];
  }

  override draw({renderPass}: {renderPass: RenderPass}): void {
    const {model, styleBuffer} = this.state as FillState;
    const props = this.props;
    const data = new ArrayBuffer(STYLE_BYTE_LENGTH);
    const floats = new Float32Array(data);
    const words = new Uint32Array(data);
    const writeColor = (offset: number, color: FillColor) =>
      floats.set(
        [color[0] / 255, color[1] / 255, color[2] / 255, (color[3] ?? 255) / 255],
        offset / 4
      );
    writeColor(0, props.color ?? [255, 255, 255, 255]);
    writeColor(16, props.noDataColor ?? [0, 0, 0, 0]);
    const palette = props.palette ?? [[255, 255, 255, 255]];
    for (let index = 0; index < 8; index++)
      writeColor(32 + index * 16, palette[index % palette.length]);
    floats.set(props.valueRange ?? [0, 1], 160 / 4);
    floats[168 / 4] = props.opacity ?? 1;
    words[172 / 4] = COLORMAP_INDEXES[props.colormap ?? 'uniform'];
    words[176 / 4] = props.values ? (props.valueFormat === 'uint32' ? 1 : 2) : 0;
    words[180 / 4] = props.valueStride ?? 1;
    words[184 / 4] = props.valueOffset ?? 0;
    words[188 / 4] = Math.min(8, palette.length);
    words[192 / 4] = props.noDataValue ?? 0xffffffff;
    words[196 / 4] =
      props.selectedRow !== undefined && props.selectedRow >= 0 ? props.selectedRow : 0xffffffff;
    styleBuffer.write(new Uint8Array(data));
    model.setInstanceCount(props.triangleCount);
    model.draw(renderPass);
  }

  override finalizeState(context: LayerContext): void {
    const {model, styleBuffer, placeholder} = this.state as FillState;
    model.destroy();
    styleBuffer.destroy();
    placeholder.destroy();
    super.finalizeState(context);
  }
}
