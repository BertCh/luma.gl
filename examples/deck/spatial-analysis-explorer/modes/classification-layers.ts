// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Mode-local layers and helpers of the classification and group-statistics modes. Both draw
 * packed `rgba8` colors that `GPUColorScale` and `GPUBivariateClassification` write straight into
 * a storage buffer: the layers unpack them in the fragment or vertex shader, so a recolor never
 * leaves the GPU.
 */

import {
  COORDINATE_SYSTEM,
  Layer,
  project32,
  type LayerContext,
  type LayerProps,
  type UpdateParameters
} from '@deck.gl/core';
import {Buffer, type Binding, type RenderPass} from '@luma.gl/core';
import {Computation, Model} from '@luma.gl/engine';
import {
  getViewBinding,
  getViewElementOffset,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {SpatialAnalysisRasterLayer} from '../spatial-analysis-layers';

/** Marker of the shared style color function that {@link PackedColorRasterLayer} replaces. */
const STYLE_COLOR_FUNCTION = 'fn getSpatialAnalysisColor(valueRow: u32) -> vec4<f32> {';

/**
 * The shared raster layer with its color lookup replaced: `values` is a `uint32` buffer of packed
 * `rgba8` colors (`r | g << 8 | b << 16 | a << 24`), one per cell, and alpha 0 discards the cell.
 *
 * The shared layers have no packed-color colormap, and `spatial-analysis-layers.ts` is not owned by this
 * mode, so the layer renames the shared function in the WGSL source and defines its own.
 */
export class PackedColorRasterLayer extends SpatialAnalysisRasterLayer {
  static override layerName = 'PackedColorRasterLayer';

  protected override getShaderSource(): string {
    const source = super.getShaderSource();
    if (!source.includes(STYLE_COLOR_FUNCTION)) {
      throw new Error('PackedColorRasterLayer: the shared raster shader changed');
    }
    return `${source.replace(STYLE_COLOR_FUNCTION, 'fn getSpatialAnalysisStyleColor(valueRow: u32) -> vec4<f32> {')}
fn getSpatialAnalysisColor(valueRow: u32) -> vec4<f32> {
  return unpack4x8unorm(styleValues[valueRow]);
}
`;
  }
}

/** Returns CSS `rgba()` for a packed `rgba8` color. */
export function getPackedColorCss(packed: number): string {
  return `rgba(${packed & 255},${(packed >>> 8) & 255},${(packed >>> 16) & 255},${((packed >>> 24) & 255) / 255})`;
}

/** Packs 0-255 channels into the `rgba8` layout of the classification contributors. */
export function packColor(r: number, g: number, b: number, a = 255): number {
  return ((r & 255) | ((g & 255) << 8) | ((b & 255) << 16) | ((a & 255) << 24)) >>> 0;
}

/** Piecewise-linear interpolation through `[r, g, b]` stops. `t` is clamped to `[0, 1]`. */
export function sampleRamp(
  stops: readonly (readonly number[])[],
  t: number
): [number, number, number] {
  const scaled = Math.min(Math.max(t, 0), 1) * (stops.length - 1);
  const index = Math.min(Math.floor(scaled), stops.length - 2);
  const fraction = scaled - index;
  const from = stops[index];
  const to = stops[index + 1];
  return [
    from[0] + (to[0] - from[0]) * fraction,
    from[1] + (to[1] - from[1]) * fraction,
    from[2] + (to[2] - from[2]) * fraction
  ];
}

/** Returns `count` packed colors sampled evenly along a ramp (a single class uses the midpoint). */
export function getRampPalette(
  stops: readonly (readonly number[])[],
  count: number,
  alpha = 255
): Uint32Array {
  const palette = new Uint32Array(count);
  for (let index = 0; index < count; index++) {
    const [r, g, b] = sampleRamp(stops, count === 1 ? 0.5 : index / (count - 1));
    palette[index] = packColor(Math.round(r), Math.round(g), Math.round(b), alpha);
  }
  return palette;
}

/** Formats a number compactly for legends: integers plain, large and tiny values in short form. */
export function formatCompact(value: number): string {
  if (!Number.isFinite(value)) return value > 0 ? '+inf' : value < 0 ? '-inf' : 'n/a';
  const magnitude = Math.abs(value);
  if (magnitude >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (magnitude >= 1e4) return `${(value / 1e3).toFixed(1)}k`;
  if (magnitude >= 100 || Number.isInteger(value)) return value.toFixed(0);
  if (magnitude >= 1) return value.toFixed(1);
  if (magnitude === 0) return '0';
  return value.toPrecision(2);
}

/** Renders a histogram as a short line of unicode bars. */
export function formatSparkline(bins: ArrayLike<number>): string {
  const bars = '▁▂▃▄▅▆▇█';
  let maximum = 0;
  for (let index = 0; index < bins.length; index++) maximum = Math.max(maximum, bins[index]);
  let text = '';
  for (let index = 0; index < bins.length; index++) {
    text +=
      bins[index] === 0 || maximum === 0
        ? ' '
        : bars[
            Math.min(bars.length - 1, Math.floor((bins[index] / maximum) * (bars.length - 1) + 0.5))
          ];
  }
  return text;
}

/**
 * Adds a free-form block to the active mode's panel (between the controls and the readouts). The shared control
 * section only offers a static legend, but the class legends here change with every readback.
 * Returns `null` when the panel markup is not found, for example when embedded elsewhere.
 */
export function createPanelBlock(): HTMLDivElement | null {
  const modeArea = document.querySelector<HTMLElement>('[data-mode]');
  if (!modeArea) return null;
  const block = document.createElement('div');
  block.style.cssText = 'margin-top:10px;color:#a9b8d0;font-size:11px';
  // Directly under the controls, above the readout block.
  modeArea.insertBefore(block, modeArea.querySelector('[data-mode-readouts]'));
  return block;
}

/** Escapes text for use in `innerHTML`. */
export function escapeHtml(text: string): string {
  return text.replace(/[&<>"]/g, character => `&#${character.charCodeAt(0)};`);
}

const STYLE_BYTE_LENGTH = 16;
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

const PACKED_QUADBIN_CELL_SHADER = /* wgsl */ `
struct PackedQuadbinStyle {
  // inset (fraction of the cell edge), opacity
  values: vec4<f32>,
};

@group(0) @binding(auto) var<uniform> packedQuadbinStyle: PackedQuadbinStyle;
@group(0) @binding(auto) var<storage, read> packedQuadbinKeys: array<vec2<u32>>;
@group(0) @binding(auto) var<storage, read> packedQuadbinColors: array<u32>;
@group(0) @binding(auto) var<storage, read> packedQuadbinCount: array<u32>;

struct PackedQuadbinVertexOutput {
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

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> PackedQuadbinVertexOutput {
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0)
  );
  var output: PackedQuadbinVertexOutput;
  // Rows at or past the GPU-written table count (and rows without a color) collapse off screen.
  let packed = packedQuadbinColors[instanceIndex];
  let key = packedQuadbinKeys[instanceIndex];
  if (instanceIndex >= packedQuadbinCount[0] || (packed >> 24u) == 0u || key.y == 0xffffffffu) {
    output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    output.color = vec4<f32>(0.0);
    return output;
  }
  let tile = decodeQuadbin(key);
  let tileCount = f32(1u << tile.z);
  let inset = packedQuadbinStyle.values.x;
  let corner = mix(vec2<f32>(inset), vec2<f32>(1.0 - inset), corners[vertexIndex]);
  let longitude = (f32(tile.x) + corner.x) / tileCount * 360.0 - 180.0;
  let latitude = getMercatorLatitude(f32(tile.y) + corner.y, tileCount);
  var clip = project_position_to_clipspace(
    vec3<f32>(longitude, latitude, 0.0), vec3<f32>(0.0), vec3<f32>(0.0)
  );
  // Deck's project32 matrices use OpenGL depth; WebGPU clip space requires [0, w].
  clip.z = (clip.z + clip.w) * 0.5;
  output.position = clip;
  let color = unpack4x8unorm(packed);
  output.color = vec4<f32>(color.rgb, color.a * packedQuadbinStyle.values.y);
  return output;
}

@fragment fn fragmentMain(input: PackedQuadbinVertexOutput) -> @location(0) vec4<f32> {
  return input.color;
}
`;

/** Props for {@link PackedQuadbinCellLayer}. */
export type PackedQuadbinCellLayerProps = LayerProps & {
  /** `uint32x2` Quadbin keys as `(low, high)` words, one per table row. */
  cells: Buffer;
  /** Packed `rgba8` color per table row. Alpha 0 hides the row. */
  colors: Buffer;
  /** One-word buffer holding the number of occupied rows (`output.count` of a cell table). */
  count: Buffer;
  /** Number of table rows (the capacity) to issue instances for. */
  capacity: number;
  /** Fraction of the cell edge trimmed on every side so neighbors show a seam. Defaults to 0.03. */
  inset?: number;
  /** Fill opacity multiplier, 0-1. Defaults to 0.85. */
  opacity?: number;
};

/**
 * Draws a capacity-bounded Quadbin cell table as Web Mercator squares, one instance per table
 * row, colored by a packed `rgba8` column. The vertex shader decodes the two-word key and reads
 * the occupied-row count from a GPU buffer, so neither keys nor colors ever reach the CPU.
 */
export class PackedQuadbinCellLayer extends Layer<PackedQuadbinCellLayerProps> {
  static override layerName = 'PackedQuadbinCellLayer';
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
      ...this.getShaders({modules: [project32], source: PACKED_QUADBIN_CELL_SHADER}),
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
    const {inset = 0.03, opacity = 0.85, capacity} = this.props;
    styleBuffer.write(Float32Array.of(inset, opacity, 0, 0));
    model.setInstanceCount(capacity);
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
      packedQuadbinStyle: styleBuffer,
      packedQuadbinKeys: this.props.cells,
      packedQuadbinColors: this.props.colors,
      packedQuadbinCount: this.props.count
    };
  }
}

/**
 * Adds a one-invocation-per-edge pass that copies one of two `GPUClassBreaks` results into the
 * shared `breaks` and `classCount` views: the alternate pair when the per-frame method code in
 * `parameters[0]` equals `alternateMethodCode`, otherwise the primary pair.
 *
 * One `GPUClassBreaks` that compiles quantile, standard-deviation and head/tail together binds nine
 * storage buffers in its finish kernel, one over the default limit, so those methods split across
 * two instances and this pass joins them without a recompile.
 */
export function addBreaksSelectPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    maximumClassCount: number;
    parameters: GraphDataView<'float32'>;
    alternateMethodCode: number;
    primaryBreaks: GraphDataView<'float32'>;
    primaryClassCount: GraphDataView<'uint32'>;
    alternateBreaks: GraphDataView<'float32'>;
    alternateClassCount: GraphDataView<'uint32'>;
    breaks: GraphDataView<'float32'>;
    classCount: GraphDataView<'uint32'>;
  }
): void {
  const edgeCount = props.maximumClassCount + 1;
  const bindings = [
    {name: 'parameters', view: props.parameters, access: 'read' as const, type: 'f32'},
    {name: 'primaryBreaks', view: props.primaryBreaks, access: 'read' as const, type: 'f32'},
    {
      name: 'primaryClassCount',
      view: props.primaryClassCount,
      access: 'read' as const,
      type: 'u32'
    },
    {name: 'alternateBreaks', view: props.alternateBreaks, access: 'read' as const, type: 'f32'},
    {
      name: 'alternateClassCount',
      view: props.alternateClassCount,
      access: 'read' as const,
      type: 'u32'
    },
    {name: 'selectedBreaks', view: props.breaks, access: 'read_write' as const, type: 'f32'},
    {name: 'selectedClassCount', view: props.classCount, access: 'read_write' as const, type: 'u32'}
  ];
  const declarations = bindings
    .map(
      (
        binding,
        location
      ) => `const ${binding.name}Offset: u32 = ${getViewElementOffset(binding.view)}u;
@group(0) @binding(${location}) var<storage, ${binding.access}> ${binding.name}: array<${binding.type}>;`
    )
    .join('\n');
  const source = /* wgsl */ `
${declarations}
@compute @workgroup_size(${edgeCount})
fn main(@builtin(local_invocation_id) invocation: vec3<u32>) {
  let useAlternate = u32(parameters[parametersOffset]) == ${props.alternateMethodCode}u;
  let index = invocation.x;
  selectedBreaks[selectedBreaksOffset + index] = select(
    primaryBreaks[primaryBreaksOffset + index],
    alternateBreaks[alternateBreaksOffset + index],
    useAlternate
  );
  if (index == 0u) {
    selectedClassCount[selectedClassCountOffset] = select(
      primaryClassCount[primaryClassCountOffset],
      alternateClassCount[alternateClassCountOffset],
      useAlternate
    );
  }
}`;
  graph.addComputePass({
    id: props.id,
    workload: {
      operation: 'ClassificationBreaksSelect',
      variant: props.id,
      commandCount: 1,
      maximumWorkgroupCount: 1,
      maximumInvocationCount: edgeCount,
      readByteLength: edgeCount * 8 + 12,
      writeByteLength: edgeCount * 4 + 4
    },
    resources: bindings.map(binding => ({
      buffer: binding.view as GraphDataView<'uint32'>,
      usage: binding.access === 'read' ? ('storage-read' as const) : ('storage-read-write' as const)
    })),
    compile: ({device}) => {
      const computation = new Computation(device, {
        id: props.id,
        source,
        shaderLayout: {
          bindings: bindings.map((binding, location) => ({
            name: binding.name,
            type: binding.access === 'read' ? ('read-only-storage' as const) : ('storage' as const),
            group: 0,
            location
          }))
        }
      });
      return {
        encode: ({computePass, getBuffer}) => {
          const resolved: Record<string, Binding> = {};
          for (const binding of bindings) {
            resolved[binding.name] = getViewBinding(
              binding.view as GraphDataView<'uint32'>,
              getBuffer
            );
          }
          computation.setBindings(resolved);
          computation.dispatch(computePass, 1);
        },
        destroy: () => computation.destroy()
      };
    }
  });
}
