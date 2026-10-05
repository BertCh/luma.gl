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

const STYLE_BYTE_LENGTH = 176;
const PALETTE_SIZE = 8;

const ALPHA_BLEND = {
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

/** Source-alpha additive blending: overlapping translucent cells accumulate into a density. */
const ADDITIVE_BLEND = {
  ...ALPHA_BLEND,
  blendColorDstFactor: 'one'
} as const;

const COLOR_MODES = {uniform: 0, count: 1, category: 2, distance: 3} as const;

/** How {@link CellBoundaryLayer} colors a cell. */
export type CellColorMode = keyof typeof COLOR_MODES;

const CELL_SHADER = /* wgsl */ `
struct CellStyle {
  palette: array<vec4<f32>, ${PALETTE_SIZE}>,
  baseColor: vec4<f32>,
  // valueMaximum, filterMaximum, widthPixels, opacity
  numbers: vec4<f32>,
  // stride, colorMode, mode (0 fill, 1 outline), useRowCount
  config: vec4<u32>,
};

@group(0) @binding(auto) var<uniform> cellStyle: CellStyle;
@group(0) @binding(auto) var<storage, read> cellBoundaries: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> cellVertexCounts: array<u32>;
@group(0) @binding(auto) var<storage, read> cellValues: array<u32>;
@group(0) @binding(auto) var<storage, read> cellRowCounts: array<u32>;

struct CellVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
  @location(1) side: f32,
};

fn getHiddenCell() -> CellVertexOutput {
  var output: CellVertexOutput;
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  output.color = vec4<f32>(0.0);
  output.side = 0.0;
  return output;
}

fn projectCellPosition(position: vec2<f32>) -> vec4<f32> {
  var clipPosition = project_position_to_clipspace(vec3<f32>(position, 0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  // Deck's project32 matrices use OpenGL depth; WebGPU clip space requires [0, w].
  clipPosition.z = (clipPosition.z + clipPosition.w) * 0.5;
  return clipPosition;
}

fn getCellInferno(t: f32) -> vec3<f32> {
  let c0 = vec3<f32>(0.0002189403691192265, 0.001651004631001012, -0.01948089843709184);
  let c1 = vec3<f32>(0.1065134194856116, 0.5639564367884091, 3.932712388889277);
  let c2 = vec3<f32>(11.60249308247187, -3.972853965665698, -15.9423941062914);
  let c3 = vec3<f32>(-41.70399613139459, 17.43639888205313, 44.35414519872813);
  let c4 = vec3<f32>(77.162935699427, -33.40235894210092, -81.80730925738993);
  let c5 = vec3<f32>(-71.31942824499214, 32.62606426397723, 73.20951985803202);
  let c6 = vec3<f32>(25.13112622477341, -12.24266895238567, -23.07032500287172);
  return clamp(c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6))))), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Alpha 0 means hidden.
fn getCellColor(row: u32) -> vec4<f32> {
  let colorMode = cellStyle.config.y;
  if (colorMode == 0u) {
    return cellStyle.baseColor;
  }
  let raw = cellValues[row];
  if (colorMode == 1u) {
    if (raw == 0u) { return vec4<f32>(0.0); }
    let t = clamp(log(1.0 + f32(raw)) / max(log(1.0 + cellStyle.numbers.x), 1e-6), 0.0, 1.0);
    return vec4<f32>(getCellInferno(0.12 + 0.88 * t), cellStyle.baseColor.a);
  }
  if (colorMode == 2u) {
    return cellStyle.palette[raw % ${PALETTE_SIZE}u];
  }
  // Distance: hide beyond the filter, fade with distance.
  if (f32(raw) > cellStyle.numbers.y) { return vec4<f32>(0.0); }
  let fade = 1.0 - 0.55 * f32(raw) / max(cellStyle.numbers.y, 1.0);
  return vec4<f32>(cellStyle.baseColor.rgb, cellStyle.baseColor.a * fade);
}

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) row: u32
) -> CellVertexOutput {
  let stride = cellStyle.config.x;
  let vertexTotal = cellVertexCounts[row];
  if (vertexTotal < 3u) { return getHiddenCell(); }
  if (cellStyle.config.w != 0u && row >= cellRowCounts[0]) { return getHiddenCell(); }
  let color = getCellColor(row);
  if (color.a <= 0.0) { return getHiddenCell(); }

  var output: CellVertexOutput;
  output.color = color;
  output.side = 0.0;
  if (cellStyle.config.z == 0u) {
    // Fill: triangle fan from vertex 0.
    let triangle = vertexIndex / 3u;
    let corner = vertexIndex % 3u;
    if (triangle + 2u >= vertexTotal) { return getHiddenCell(); }
    var index = 0u;
    if (corner > 0u) { index = triangle + corner; }
    output.position = projectCellPosition(cellBoundaries[row * stride + index]);
    return output;
  }
  // Outline: one screen-space quad per edge.
  let edge = vertexIndex / 6u;
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
  );
  if (edge >= vertexTotal) { return getHiddenCell(); }
  let corner = corners[vertexIndex % 6u];
  let startClip = projectCellPosition(cellBoundaries[row * stride + edge]);
  let endClip = projectCellPosition(cellBoundaries[row * stride + (edge + 1u) % vertexTotal]);
  let screenDirection = (endClip.xy / endClip.w - startClip.xy / startClip.w) * project.viewportSize;
  let directionLength = length(screenDirection);
  var direction = vec2<f32>(1.0, 0.0);
  if (directionLength > 1e-6) { direction = screenDirection / directionLength; }
  let normal = vec2<f32>(-direction.y, direction.x);
  var clipPosition = mix(startClip, endClip, corner.x);
  let along = direction * (corner.x * 2.0 - 1.0) * 0.5;
  clipPosition = vec4<f32>(
    clipPosition.xy + project_pixel_size_to_clipspace((normal * corner.y + along) * cellStyle.numbers.z * 0.5),
    clipPosition.z,
    clipPosition.w
  );
  output.position = clipPosition;
  output.side = corner.y;
  return output;
}

@fragment fn fragmentMain(input: CellVertexOutput) -> @location(0) vec4<f32> {
  var coverage = 1.0;
  if (cellStyle.config.z == 1u) {
    coverage = 1.0 - smoothstep(0.55, 1.0, abs(input.side));
  }
  return vec4<f32>(input.color.rgb, input.color.a * cellStyle.numbers.w * coverage);
}
`;

/** Props for {@link CellBoundaryLayer}. */
export type CellBoundaryLayerProps = LayerProps & {
  /** `float32x2` longitude/latitude boundary rows, `stride` vertices per cell. */
  boundaries: Buffer;
  /** uint32 vertex count per cell; cells with fewer than 3 vertices are not drawn. */
  vertexCounts: Buffer;
  /** Boundary vertices reserved per cell (the `maximumVertexCount` of `GPUCellGeometry`). */
  stride: number;
  /** Rows to draw. Rows past the GPU row count (when `rowCounts` is set) are skipped. */
  instanceCount: number;
  /** Optional one-word uint32 buffer holding the number of valid rows, written by the recipe. */
  rowCounts?: Buffer | null;
  /** uint32 value per cell for the `count`, `category` and `distance` color modes. */
  values?: Buffer | null;
  /** `'fill'` draws a triangle fan per cell; `'outline'` draws screen-space edge quads. */
  mode?: 'fill' | 'outline';
  /** Color source. Defaults to `'uniform'`. */
  colorMode?: CellColorMode;
  /** RGBA 0-255 for `uniform` and `distance` modes and the alpha of `count`. */
  color?: readonly [number, number, number, number];
  /** Category colors (`palette[value % 8]`), RGBA 0-255. */
  palette?: readonly (readonly number[])[];
  /** Value that maps to the top of the log color scale in `count` mode. */
  valueMaximum?: number;
  /** In `distance` mode, cells with a value above this are hidden. */
  filterMaximum?: number;
  /** Outline width in CSS pixels. Defaults to 1.5. */
  widthPixels?: number;
  /** Overall opacity multiplier. Defaults to 1. */
  opacity?: number;
  /** Accumulates overlapping cells instead of compositing them. Fixed at layer creation. */
  additive?: boolean;
};

type CellLayerState = {model: Model; styleBuffer: Buffer; placeholder: Buffer};

/**
 * Draws cell polygons straight from `GPUCellGeometry` outputs: the fixed-stride boundary rows and
 * vertex counts are bound as storage buffers, so no cell geometry ever visits the CPU.
 */
export class CellBoundaryLayer extends Layer<CellBoundaryLayerProps> {
  static override layerName = 'CellBoundaryLayer';
  static override defaultProps = {
    parameters: ALPHA_BLEND
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
    const blend = this.props.additive ? ADDITIVE_BLEND : ALPHA_BLEND;
    const model = new Model(device, {
      ...this.getShaders({modules: [project32], source: CELL_SHADER}),
      id: `${this.id}-model`,
      topology: 'triangle-list',
      isInstanced: true,
      vertexCount: this.getVertexCount(),
      instanceCount: 0,
      bufferLayout: [],
      bindings: this.getBindings(styleBuffer, placeholder),
      parameters: blend
    });
    this.setState({model, styleBuffer, placeholder} satisfies CellLayerState);
  }

  override updateState(parameters: UpdateParameters<this>): void {
    super.updateState(parameters);
    const {model, styleBuffer, placeholder} = this.state as CellLayerState;
    model.setBindings(this.getBindings(styleBuffer, placeholder));
    model.setVertexCount(this.getVertexCount());
  }

  override getModels(): Model[] {
    return [(this.state as CellLayerState).model];
  }

  override draw({renderPass}: {renderPass: RenderPass}): void {
    const {model, styleBuffer} = this.state as CellLayerState;
    const words = new ArrayBuffer(STYLE_BYTE_LENGTH);
    const floats = new Float32Array(words);
    const integers = new Uint32Array(words);
    const palette = this.props.palette ?? [];
    for (let index = 0; index < PALETTE_SIZE; index++) {
      const color = palette[index % Math.max(palette.length, 1)] ?? [255, 255, 255, 255];
      floats.set(
        [color[0] / 255, color[1] / 255, color[2] / 255, (color[3] ?? 255) / 255],
        index * 4
      );
    }
    const base = this.props.color ?? [255, 255, 255, 255];
    floats.set([base[0] / 255, base[1] / 255, base[2] / 255, base[3] / 255], 32);
    floats.set(
      [
        this.props.valueMaximum ?? 1,
        this.props.filterMaximum ?? 1e9,
        this.props.widthPixels ?? 1.5,
        this.props.opacity ?? 1
      ],
      36
    );
    integers.set(
      [
        this.props.stride,
        COLOR_MODES[this.props.colorMode ?? 'uniform'],
        this.props.mode === 'outline' ? 1 : 0,
        this.props.rowCounts ? 1 : 0
      ],
      40
    );
    styleBuffer.write(new Uint8Array(words));
    model.setInstanceCount(this.props.instanceCount);
    model.draw(renderPass);
  }

  override finalizeState(context: LayerContext): void {
    const {model, styleBuffer, placeholder} = this.state as CellLayerState;
    model.destroy();
    styleBuffer.destroy();
    placeholder.destroy();
    super.finalizeState(context);
  }

  private getVertexCount(): number {
    const {stride, mode} = this.props;
    return mode === 'outline' ? stride * 6 : Math.max(stride - 2, 1) * 3;
  }

  private getBindings(styleBuffer: Buffer, placeholder: Buffer): Record<string, Buffer> {
    return {
      cellStyle: styleBuffer,
      cellBoundaries: this.props.boundaries,
      cellVertexCounts: this.props.vertexCounts,
      cellValues: this.props.values ?? placeholder,
      cellRowCounts: this.props.rowCounts ?? placeholder
    };
  }
}
