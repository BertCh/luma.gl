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
import {COLORMAP_INDEXES, getRampWgsl, type RampName} from '../../engine/ramps';

const STYLE_BYTE_LENGTH = 64;
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

const MODES = {density: 0, meanValue: 1, packed: 2, signed: 3} as const;

/** How {@link QuadbinTableLayer} colors a cell. */
export type QuadbinColorMode = keyof typeof MODES;

const QUADBIN_CELL_SHADER = /* wgsl */ `
${getRampWgsl()}

struct QuadbinStyle {
  // log10 observations per square kilometer at the low and high end, opacity, inset
  range: vec4<f32>,
  // mean/signed value at the low end, at the high end, signed limit, unused
  value: vec4<f32>,
  // mode, colormap index, useRowCount, signed transform (0 linear, 1 log2 ratio)
  config: vec4<u32>,
  // color of cells without a baseline (NaN ratio)
  noBaseline: vec4<f32>,
};

@group(0) @binding(auto) var<uniform> quadbinStyle: QuadbinStyle;
@group(0) @binding(auto) var<storage, read> quadbinKeysEven: array<vec2<u32>>;
@group(0) @binding(auto) var<storage, read> quadbinKeysOdd: array<vec2<u32>>;
@group(0) @binding(auto) var<storage, read> quadbinCountsEven: array<u32>;
@group(0) @binding(auto) var<storage, read> quadbinCountsOdd: array<u32>;
// Either an f32 value column (mean, signed) or packed rgba8 colors: one binding keeps the vertex
// stage within the default limit of 8 storage buffers.
@group(0) @binding(auto) var<storage, read> quadbinData: array<u32>;
@group(0) @binding(auto) var<storage, read> quadbinActiveLevel: array<u32>;
@group(0) @binding(auto) var<storage, read> quadbinFirstRow: array<u32>;
@group(0) @binding(auto) var<storage, read> quadbinRowCount: array<u32>;

struct QuadbinVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
};

fn compactEvenBits(value: u32) -> u32 {
  var bits = value & 0x55555555u;
  bits = (bits | (bits >> 1u)) & 0x33333333u;
  bits = (bits | (bits >> 2u)) & 0x0f0f0f0fu;
  bits = (bits | (bits >> 4u)) & 0x00ff00ffu;
  bits = (bits | (bits >> 8u)) & 0x0000ffffu;
  return bits;
}

// Decodes a Quadbin key stored as (low, high) words into (tileX, tileY, zoom).
fn decodeQuadbin(key: vec2<u32>) -> vec3<u32> {
  let zoom = (key.y >> 20u) & 0x1fu;
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

fn isFiniteValue(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
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
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  output.color = vec4<f32>(0.0);
  let mode = quadbinStyle.config.x;
  if (quadbinStyle.config.z != 0u && instanceIndex >= quadbinRowCount[0]) { return output; }
  // Even and odd pyramid levels live in alternating slabs; the level is the GPU-visible word that
  // GPUCellLevelSelection also reads.
  let isOdd = (quadbinActiveLevel[0] & 1u) == 1u;
  let row = quadbinFirstRow[0] + instanceIndex;
  var count = 1u;
  if (mode <= 1u) {
    count = select(quadbinCountsEven[row], quadbinCountsOdd[row], isOdd);
    if (count == 0u) { return output; }
  }
  let tile = decodeQuadbin(select(quadbinKeysEven[row], quadbinKeysOdd[row], isOdd));
  let tileCount = f32(1u << tile.z);
  let inset = quadbinStyle.range.w;
  let corner = mix(vec2<f32>(inset), vec2<f32>(1.0 - inset), corners[vertexIndex]);
  let longitude = (f32(tile.x) + corner.x) / tileCount * 360.0 - 180.0;
  let latitude = getMercatorLatitude(f32(tile.y) + corner.y, tileCount);

  var color = vec4<f32>(0.0);
  if (mode == 0u) {
    // Observations per square kilometer: a Web Mercator tile is stretched by 1 / cos(latitude).
    let centerLatitude = getMercatorLatitude(f32(tile.y) + 0.5, tileCount);
    let edgeKilometers = 40075.016686 / tileCount * cos(radians(centerLatitude));
    let density = f32(count) / (edgeKilometers * edgeKilometers);
    let t = (log2(max(density, 1.0e-3)) * 0.30103 - quadbinStyle.range.x) /
      max(quadbinStyle.range.y - quadbinStyle.range.x, 1.0e-3);
    color = vec4<f32>(spatialAnalysisSampleRamp(quadbinStyle.config.y, clamp(t, 0.0, 1.0)), quadbinStyle.range.z);
  } else if (mode == 1u) {
    let mean = bitcast<f32>(quadbinData[row]) / f32(count);
    let t = (mean - quadbinStyle.value.x) / max(quadbinStyle.value.y - quadbinStyle.value.x, 1.0e-6);
    color = vec4<f32>(spatialAnalysisSampleRamp(quadbinStyle.config.y, clamp(t, 0.0, 1.0)), quadbinStyle.range.z);
  } else if (mode == 2u) {
    let packed = quadbinData[row];
    let rgba = vec4<f32>(
      f32(packed & 255u), f32((packed >> 8u) & 255u), f32((packed >> 16u) & 255u), f32(packed >> 24u)
    ) / 255.0;
    if (rgba.a <= 0.0) { return output; }
    color = vec4<f32>(rgba.rgb, rgba.a * quadbinStyle.range.z);
  } else {
    var value = bitcast<f32>(quadbinData[row]);
    if (!isFiniteValue(value) || (quadbinStyle.config.w == 1u && value <= 0.0)) {
      color = vec4<f32>(quadbinStyle.noBaseline.rgb, quadbinStyle.noBaseline.a * quadbinStyle.range.z);
    } else {
      if (quadbinStyle.config.w == 1u) { value = log2(value); }
      let t = 0.5 + 0.5 * clamp(value / max(quadbinStyle.value.z, 1.0e-6), -1.0, 1.0);
      color = vec4<f32>(spatialAnalysisSampleRamp(quadbinStyle.config.y, t), quadbinStyle.range.z);
    }
  }
  var clip = project_position_to_clipspace(
    vec3<f32>(longitude, latitude, 0.0), vec3<f32>(0.0), vec3<f32>(0.0)
  );
  // Deck's project32 matrices use OpenGL depth; WebGPU clip space requires [0, w].
  clip.z = (clip.z + clip.w) * 0.5;
  output.position = clip;
  output.color = color;
  return output;
}

@fragment fn fragmentMain(input: QuadbinVertexOutput) -> @location(0) vec4<f32> {
  return input.color;
}
`;

/** Props for {@link QuadbinTableLayer}. */
export type QuadbinTableLayerProps = LayerProps & {
  /** `uint32x2` Quadbin keys as (low, high) words (even pyramid levels, or the only table). */
  cellsEven: Buffer;
  /** `uint32x2` Quadbin keys of odd pyramid levels. Defaults to `cellsEven`. */
  cellsOdd?: Buffer;
  /** `uint32` observations per cell row (even levels). Needed by the `density` and `meanValue` modes. */
  countsEven?: Buffer | null;
  /** `uint32` observations per cell row (odd levels). Defaults to `countsEven`. */
  countsOdd?: Buffer | null;
  /** `float32` per cell row: summed value (`meanValue`) or the compared column (`signed`). */
  values?: Buffer | null;
  /** Packed `rgba8` per cell row for the `packed` mode. */
  colors?: Buffer | null;
  /** One-word active level index whose parity picks the slab. Omit for a single table. */
  activeLevel?: Buffer | null;
  /** One-word first row of the active level. Omit for a single table. */
  firstRow?: Buffer | null;
  /** One-word valid row count; rows past it are skipped. Use with `instanceCount`. */
  rowCount?: Buffer | null;
  /** Rows drawn when `drawCommands` is not set. */
  instanceCount?: number;
  /** GPU-written indirect record (six vertices); its instance count is the row count. */
  drawCommands?: DrawCommandBuffer | null;
  mode: QuadbinColorMode;
  ramp?: RampName;
  /** `[minimum, maximum]` of log10 observations per square kilometer (`density`). */
  densityLogRange?: readonly [number, number];
  /** `[minimum, maximum]` mean value (`meanValue`). */
  meanRange?: readonly [number, number];
  /** Symmetric limit of the `signed` mode: value or log2 ratio mapped to the ramp ends. */
  signedLimit?: number;
  /** `signed` transform: `'log2'` colors `log2(value)` and treats NaN or non-positive as no baseline. */
  signedTransform?: 'linear' | 'log2';
  noBaselineColor?: readonly [number, number, number, number];
  opacity?: number;
  /** Fraction of the cell edge trimmed on every side. Defaults to 0.03. */
  inset?: number;
};

type LayerState = {model: Model; styleBuffer: Buffer; placeholder: Buffer};

/**
 * Draws Quadbin cells as Web Mercator squares straight from a `GPUCellAggregation`,
 * `GPUCellRollup`, `GPUCellPyramid` or `GPUCellTableCompare` table. The vertex shader decodes the
 * two-word key into tile x, y and zoom and colors by density, a mean, packed class colors or a
 * signed comparison column, so no table ever leaves the GPU.
 */
export class QuadbinTableLayer extends Layer<QuadbinTableLayerProps> {
  static override layerName = 'QuadbinTableLayer';
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
    const placeholder = device.createBuffer({
      id: `${this.id}-placeholder`,
      byteLength: 16,
      usage: Buffer.STORAGE | Buffer.COPY_DST
    });
    const model = new Model(device, {
      ...this.getShaders({modules: [project32], source: QUADBIN_CELL_SHADER}),
      id: `${this.id}-model`,
      topology: 'triangle-list',
      isInstanced: true,
      vertexCount: 6,
      instanceCount: 0,
      bufferLayout: [],
      bindings: this.getBindings(styleBuffer, placeholder),
      parameters: BLEND_PARAMETERS
    });
    this.setState({model, styleBuffer, placeholder} satisfies LayerState);
  }

  override updateState(parameters: UpdateParameters<this>): void {
    super.updateState(parameters);
    const {model, styleBuffer, placeholder} = this.state as LayerState;
    model.setBindings(this.getBindings(styleBuffer, placeholder));
  }

  override getModels(): Model[] {
    return [(this.state as LayerState).model];
  }

  override draw({renderPass}: {renderPass: RenderPass}): void {
    const {model, styleBuffer} = this.state as LayerState;
    const props = this.props;
    const data = new ArrayBuffer(STYLE_BYTE_LENGTH);
    const floats = new Float32Array(data);
    const words = new Uint32Array(data);
    const [densityMinimum, densityMaximum] = props.densityLogRange ?? [0, 4];
    const [meanMinimum, meanMaximum] = props.meanRange ?? [0, 1];
    floats.set([densityMinimum, densityMaximum, props.opacity ?? 0.8, props.inset ?? 0.03], 0);
    floats.set([meanMinimum, meanMaximum, props.signedLimit ?? 1, 0], 4);
    words.set(
      [
        MODES[props.mode],
        COLORMAP_INDEXES[props.ramp ?? 'viridis'],
        props.rowCount ? 1 : 0,
        props.signedTransform === 'log2' ? 1 : 0
      ],
      8
    );
    const noBaseline = props.noBaselineColor ?? [150, 150, 150, 200];
    floats.set(
      [noBaseline[0] / 255, noBaseline[1] / 255, noBaseline[2] / 255, noBaseline[3] / 255],
      12
    );
    styleBuffer.write(new Uint8Array(data));
    if (props.drawCommands) {
      // Bind pipeline and bindings with an empty draw, then replay the GPU-written record.
      model.setInstanceCount(0);
      model.draw(renderPass);
      props.drawCommands.draw(renderPass, 0);
    } else {
      model.setInstanceCount(props.instanceCount ?? 0);
      model.draw(renderPass);
    }
  }

  override finalizeState(context: LayerContext): void {
    const {model, styleBuffer, placeholder} = this.state as LayerState;
    model.destroy();
    styleBuffer.destroy();
    placeholder.destroy();
    super.finalizeState(context);
  }

  private getBindings(styleBuffer: Buffer, placeholder: Buffer): Record<string, Buffer> {
    const {props} = this;
    return {
      quadbinStyle: styleBuffer,
      quadbinKeysEven: props.cellsEven,
      quadbinKeysOdd: props.cellsOdd ?? props.cellsEven,
      quadbinCountsEven: props.countsEven ?? placeholder,
      quadbinCountsOdd: props.countsOdd ?? props.countsEven ?? placeholder,
      quadbinData: props.values ?? props.colors ?? placeholder,
      quadbinActiveLevel: props.activeLevel ?? placeholder,
      quadbinFirstRow: props.firstRow ?? placeholder,
      quadbinRowCount: props.rowCount ?? placeholder
    };
  }
}
