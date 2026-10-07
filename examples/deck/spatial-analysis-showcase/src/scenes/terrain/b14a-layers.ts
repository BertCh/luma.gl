// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Terrain-chapter deck.gl layer: one quad over the raster bounds whose fragments unpack a packed
 * RGBA8 color per cell from a storage buffer (the `color` outputs of `GPUReliefShading` and
 * `GPUReliefBlend`, or the colorize pass of `b14a-colorize.ts`), optionally multiplied by a
 * per-cell illumination float (`GPUSolarShadowMask`). Nothing is read back or repacked.
 */

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
import {getCompareState} from '../../engine/layers';

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

/**
 * Multiply blending (the same state as the engine layers): the fragment, written as
 * `mix(vec3(1), rgb, a)`, multiplies the colour already in the target and the destination alpha
 * stays.
 */
const MULTIPLY_BLEND_PARAMETERS = {
  ...BLEND_PARAMETERS,
  blendColorSrcFactor: 'dst',
  blendColorDstFactor: 'zero',
  blendAlphaSrcFactor: 'zero',
  blendAlphaDstFactor: 'one'
} as const;

/** Hatch geometry of `noDataHatch`, in CSS pixels. */
const NO_DATA_HATCH_PITCH_PIXELS = 4;
const NO_DATA_HATCH_WIDTH_PIXELS = 1;

/** Byte length of the layer style uniform (see `LayerStyle` in the shader). */
const STYLE_BYTE_LENGTH = 80;

const SHADER = /* wgsl */ `
struct LayerStyle {
  minimum: vec2<f32>,
  maximum: vec2<f32>,
  gridSize: vec2<u32>,
  lightStrength: f32,
  opacity: f32,
  lightGain: f32,
  lightFloor: f32,
  // Swipe compare: 0 not compared, 1 side a, 2 side b, 3 hidden entirely.
  compareSide: u32,
  compareDivider: f32,
  hatchColor: vec4<f32>,
  hatchEnabled: u32,
  devicePixelRatio: f32,
  multiply: u32,
  hatchPitch: f32,
};
@group(0) @binding(auto) var<uniform> layerStyle: LayerStyle;
@group(0) @binding(auto) var<storage, read> cellColors: array<u32>;
@group(0) @binding(auto) var<storage, read> cellLight: array<f32>;

fn projectCell(position: vec2<f32>) -> vec4<f32> {
  var clipPosition = project_position_to_clipspace(vec3<f32>(position, 0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  // Deck's project32 matrices use OpenGL depth; WebGPU clip space requires [0, w].
  clipPosition.z = (clipPosition.z + clipPosition.w) * 0.5;
  return clipPosition;
}

// Non-finite floats are tested through the exponent bits: compilers may fold x != x.
fn isNonFiniteCell(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7fffffffu) >= 0x7f800000u;
}

// True when a fragment lies on the hidden side of the swipe divider (fragment x is in device pixels).
fn isCompareHidden(fragmentPosition: vec4<f32>) -> bool {
  let side = layerStyle.compareSide;
  if (side == 0u) {
    return false;
  }
  if (side == 3u) {
    return true;
  }
  let isLeft = fragmentPosition.x < layerStyle.compareDivider;
  return select(isLeft, !isLeft, side == 1u);
}

// Coverage of the 45 degree hatch stripes at a fragment: pitch and width are in CSS pixels.
fn getHatchCoverage(fragmentPosition: vec4<f32>) -> f32 {
  let ratio = max(layerStyle.devicePixelRatio, 1e-3);
  let spacing = max(layerStyle.hatchPitch, 1.0);
  let phase = (fragmentPosition.x + fragmentPosition.y) / ratio * 0.7071067811865476 / spacing;
  let fraction = fract(phase);
  let stripeDistance = min(fraction, 1.0 - fraction) * spacing;
  let halfWidth = ${NO_DATA_HATCH_WIDTH_PIXELS.toFixed(1)} * 0.5;
  let softness = 0.5 / ratio;
  return clamp((halfWidth + softness - stripeDistance) / (2.0 * softness), 0.0, 1.0);
}

// Multiply blending needs mix(white, rgb, a): the blend state multiplies it into the target.
fn finishColor(color: vec4<f32>) -> vec4<f32> {
  if (layerStyle.multiply != 0u) {
    return vec4<f32>(mix(vec3<f32>(1.0), color.rgb, color.a), color.a);
  }
  return color;
}

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) local: vec2<f32>,
};

@vertex fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0)
  );
  let corner = corners[vertexIndex];
  var output: VertexOutput;
  output.position = projectCell(mix(layerStyle.minimum, layerStyle.maximum, corner));
  output.local = corner;
  return output;
}

@fragment fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
  if (isCompareHidden(input.position)) {
    discard;
  }
  let columns = i32(layerStyle.gridSize.x);
  let rows = i32(layerStyle.gridSize.y);
  let column = clamp(i32(floor(input.local.x * f32(columns))), 0, columns - 1);
  // Row 0 is the north edge (the raster the contributors read), so flip the row.
  let row = clamp(rows - 1 - i32(floor(input.local.y * f32(rows))), 0, rows - 1);
  let index = u32(row * columns + column);
  var color = unpack4x8unorm(cellColors[index]);
  if (color.a <= 0.0) {
    if (layerStyle.hatchEnabled == 0u) {
      discard;
    }
    let coverage = getHatchCoverage(input.position) * layerStyle.hatchColor.a;
    if (coverage <= 0.0) {
      discard;
    }
    return finishColor(vec4<f32>(layerStyle.hatchColor.rgb, coverage * layerStyle.opacity));
  }
  if (layerStyle.lightStrength > 0.0) {
    var light = cellLight[index];
    if (isNonFiniteCell(light)) {
      light = 0.0;
    }
    let lit = clamp(light * layerStyle.lightGain, 0.0, 1.0);
    color = vec4<f32>(color.rgb * mix(1.0, mix(layerStyle.lightFloor, 1.0, lit), layerStyle.lightStrength), color.a);
  }
  return finishColor(vec4<f32>(color.rgb, color.a * layerStyle.opacity));
}
`;

/** Props for {@link ColorRasterLayer}. */
export type ColorRasterLayerProps = LayerProps & {
  /** Packed RGBA8 colors (`uint32`, red in the low byte), row 0 at the north edge. */
  colors: Buffer;
  /** Optional illumination floats, same layout; multiplied into the color by `lightStrength`. */
  light?: Buffer | null;
  /** Share of the illumination applied to the color, 0 (none) to 1 (full). Defaults to 0. */
  lightStrength?: number;
  /** Multiplier on the illumination before it is clamped to `[0, 1]` (exposure). Defaults to 1. */
  lightGain?: number;
  /** Brightness of fully shaded cells relative to lit ones, `[0, 1]`. Defaults to 0. */
  lightFloor?: number;
  /** `[columns, rows]` of the raster. */
  gridSize: readonly [number, number];
  /** `[minX, minY, maxX, maxY]` meters of the raster's outer cell edges. */
  bounds: readonly [number, number, number, number];
  /** Layer alpha. Defaults to 1. */
  alpha?: number;
  /**
   * Swipe compare: which side of the divider this layer belongs to, read from the same module
   * state as the engine layers (`setCompareState` in `engine/layers.ts`). Layers without it are
   * never clipped.
   */
  compareSide?: 'a' | 'b';
  /**
   * Draws cells whose packed colour has alpha 0 as 45 degree stripes of this colour (RGBA 0-255,
   * alpha defaults to 255; 4 px pitch, 1 px line, in CSS pixels) instead of discarding them. The
   * stripes stay inside the raster quad. Defaults to off: alpha-0 cells are discarded.
   */
  noDataHatch?: readonly [number, number, number, number?];
  /**
   * `'normal'` (default) is alpha blending. `'multiply'` multiplies the colour into what the deck
   * canvas already drew (the relief ground), like `blending: 'multiply'` of the engine layers.
   */
  blendMode?: 'normal' | 'multiply';
};

type LayerState = {
  model: Model | null;
  styleBuffer: Buffer | null;
  placeholderBuffer: Buffer | null;
};

/** One quad over the raster bounds whose fragments unpack a color from storage. */
export class ColorRasterLayer extends Layer<ColorRasterLayerProps> {
  static override layerName = 'ColorRasterLayer';
  static override defaultProps = {
    coordinateSystem: COORDINATE_SYSTEM.METER_OFFSETS,
    parameters: BLEND_PARAMETERS
  };

  override getAttributeManager() {
    return null;
  }

  private getBindings(placeholder: Buffer): Record<string, Buffer> {
    return {cellColors: this.props.colors, cellLight: this.props.light ?? placeholder};
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
    this.setState({model: null, styleBuffer, placeholderBuffer} satisfies LayerState);
    const model = new Model(device, {
      ...this.getShaders({modules: [project32], source: SHADER}),
      id: `${this.id}-model`,
      topology: 'triangle-list',
      isInstanced: true,
      vertexCount: 6,
      instanceCount: 0,
      bufferLayout: [],
      bindings: {...this.getBindings(placeholderBuffer), layerStyle: styleBuffer},
      parameters: this.props.blendMode === 'multiply' ? MULTIPLY_BLEND_PARAMETERS : BLEND_PARAMETERS
    });
    this.setState({model});
  }

  override updateState(parameters: UpdateParameters<this>): void {
    super.updateState(parameters);
    const {model, styleBuffer, placeholderBuffer} = this.state as LayerState;
    if (model && styleBuffer && placeholderBuffer) {
      model.setBindings({...this.getBindings(placeholderBuffer), layerStyle: styleBuffer});
    }
  }

  override getModels(): Model[] {
    const model = (this.state as LayerState).model;
    return model ? [model] : [];
  }

  /**
   * Deck merges the layer's `parameters` prop (alpha blending) over the model's on every draw,
   * which would undo `multiply`; the blend factors of `blendMode` are applied last so they win.
   */
  override _drawLayer(options: Parameters<Layer['_drawLayer']>[0]): void {
    super._drawLayer({
      ...options,
      parameters: {
        ...options.parameters,
        ...(this.props.blendMode === 'multiply' ? MULTIPLY_BLEND_PARAMETERS : BLEND_PARAMETERS)
      }
    });
  }

  /** The divider (device pixels) and side code of the swipe compare, as the engine layers do. */
  private getCompareUniforms(): {side: number; divider: number} {
    const {compareSide} = this.props;
    const state = getCompareState();
    if (!compareSide || !state) return {side: 0, divider: 0};
    if (state.showing !== 'both') {
      // Only one side shows; the other is hidden entirely.
      return {side: state.showing === compareSide ? 0 : 3, divider: 0};
    }
    const viewport = this.context?.viewport;
    const ratio = this.getDevicePixelRatio();
    const position = Math.min(Math.max(state.position, 0), 1);
    const divider = ((viewport?.x ?? 0) + position * (viewport?.width ?? 1)) * ratio;
    return {side: compareSide === 'a' ? 1 : 2, divider};
  }

  private getDevicePixelRatio(): number {
    try {
      return this.context.device.getDefaultCanvasContext().cssToDeviceRatio();
    } catch {
      // No canvas context (headless device): CSS pixels are device pixels.
      return 1;
    }
  }

  override draw({renderPass}: {renderPass: RenderPass}): void {
    const {model, styleBuffer} = this.state as LayerState;
    if (!model || !styleBuffer) return;
    const {
      bounds,
      gridSize,
      lightStrength = 0,
      alpha = 1,
      lightGain = 1,
      lightFloor = 0,
      noDataHatch
    } = this.props;
    const data = new ArrayBuffer(STYLE_BYTE_LENGTH);
    const floats = new Float32Array(data);
    const words = new Uint32Array(data);
    floats.set([bounds[0], bounds[1], bounds[2], bounds[3]]);
    words.set(gridSize, 4);
    floats[6] = this.props.light ? lightStrength : 0;
    floats[7] = alpha;
    floats[8] = lightGain;
    floats[9] = lightFloor;
    const compare = this.getCompareUniforms();
    words[10] = compare.side;
    floats[11] = compare.divider;
    if (noDataHatch) {
      floats.set(
        [
          noDataHatch[0] / 255,
          noDataHatch[1] / 255,
          noDataHatch[2] / 255,
          (noDataHatch[3] ?? 255) / 255
        ],
        12
      );
      words[16] = 1;
    }
    floats[17] = this.getDevicePixelRatio();
    words[18] = this.props.blendMode === 'multiply' ? 1 : 0;
    floats[19] = NO_DATA_HATCH_PITCH_PIXELS;
    styleBuffer.write(new Uint8Array(data));
    model.setInstanceCount(1);
    model.draw(renderPass);
  }

  override finalizeState(context: LayerContext): void {
    const state = this.state as LayerState;
    state.model?.destroy();
    state.styleBuffer?.destroy();
    state.placeholderBuffer?.destroy();
    this.setState({model: null, styleBuffer: null, placeholderBuffer: null});
    super.finalizeState(context);
  }
}
