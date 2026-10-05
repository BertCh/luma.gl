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
import {Buffer, type Device, type RenderPass} from '@luma.gl/core';
import {Model} from '@luma.gl/engine';
import type {DrawCommandBuffer} from '@luma.gl/gpgpu/gpu-core';
import {GPU_POINT_DENSITY_HEXAGON_WGSL} from '@luma.gl/experimental/gpu-spatial-analysis';

/**
 * Generic deck.gl layers that draw analysis contributor outputs straight from GPU storage buffers.
 *
 * None of these layers reads a buffer back or repacks it: positions, values, compact IDs, extents,
 * fade weights, clip fractions, and GPU-resident bounds are bound as read-only storage, and counts
 * can come from a GPU-written indirect draw record. All positions are planar meters rendered with
 * `COORDINATE_SYSTEM.METER_OFFSETS` around `coordinateOrigin`, optionally through an affine
 * `positionScale`/`positionOffset` (for example raster cell coordinates to meters).
 */

/** Element format of a `values` buffer. */
export type SpatialAnalysisValueFormat = 'none' | 'uint32' | 'float32';

/**
 * How a value becomes a color.
 *
 * - `uniform`: `color` for every row.
 * - `viridis`, `inferno`, `grayscale`: normalized scalar `(value - min) / (max - min)`, using
 *   `valueRange` or a GPU `[min, max]` `extent` buffer.
 * - `category`: `palette[value % paletteSize]` (or a built-in categorical palette).
 * - `mask`: `color` when the value is nonzero, otherwise `noDataColor`.
 */
export type SpatialAnalysisColormap =
  | 'uniform'
  | 'viridis'
  | 'inferno'
  | 'grayscale'
  | 'category'
  | 'mask';

/** RGBA color with 0-255 channels. */
export type SpatialAnalysisColor = readonly [number, number, number, number?];

/** Styling and value-mapping props shared by every spatial-analysis layer. */
export type SpatialAnalysisStyleProps = {
  /** Per-row values (`values[row / valueDivisor]`). Omit for uniform color. */
  values?: Buffer | null;
  /** Element format of `values`. Defaults to `'float32'` when `values` is set. */
  valueFormat?: SpatialAnalysisValueFormat;
  /** Rows sharing one value, for example 4 outline segments per tile. Defaults to 1. */
  valueDivisor?: number;
  /**
   * Optional uint32 per drawn row giving the `values` row that colors it, for example the start
   * node of each road segment or the feature row of each outline segment. Overrides `valueDivisor`.
   */
  valueIndices?: Buffer | null;
  /** Value-to-color mapping. Defaults to `'uniform'`. */
  colormap?: SpatialAnalysisColormap;
  /** `[min, max]` for scalar colormaps when no `extent` buffer is given. */
  valueRange?: readonly [number, number];
  /** Optional GPU `[min, max]` float32 buffer (for example a contributor `extent` output). */
  extent?: Buffer | null;
  /** Multiplier applied to scalar values before normalization. Defaults to 1. */
  valueScale?: number;
  /** Apply `t = sqrt(t)` after normalization to lift low densities. */
  sqrtScale?: boolean;
  /** Base color for `uniform` and `mask`. */
  color?: SpatialAnalysisColor;
  /** Color for `noDataValue`, NaN or infinite floats, or a zero mask. Alpha 0 hides those rows. */
  noDataColor?: SpatialAnalysisColor;
  /** uint32 sentinel treated as no data. Defaults to `0xffffffff`. */
  noDataValue?: number;
  /** Discard rows whose scalar value is `<= discardAtOrBelow` (for example empty density cells). */
  discardAtOrBelow?: number;
  /** Up to 8 colors for `category`. Defaults to a built-in 8-color palette. */
  palette?: readonly SpatialAnalysisColor[];
  /** Affine transform applied to positions before projection. Defaults to identity. */
  positionScale?: readonly [number, number];
  /** Affine transform offset applied after `positionScale`. */
  positionOffset?: readonly [number, number];
};

/** Instance-count source: a fixed number or a GPU-written indirect draw record. */
export type SpatialAnalysisInstanceProps = {
  /** Number of instances to draw when `drawCommands` is not set. */
  instanceCount?: number;
  /** GPU-written indirect record whose `vertexCount` matches the layer (6 for quads). */
  drawCommands?: DrawCommandBuffer | null;
  /** Record index inside `drawCommands`. Defaults to 0. */
  drawCommandIndex?: number;
  /** Optional compact row IDs: instance `i` draws row `ids[i]`. */
  ids?: Buffer | null;
};

const DEFAULT_PALETTE: readonly SpatialAnalysisColor[] = [
  [78, 201, 255, 255],
  [255, 148, 72, 255],
  [189, 122, 255, 255],
  [87, 235, 168, 255],
  [255, 105, 168, 255],
  [245, 220, 87, 255],
  [107, 158, 255, 255],
  [255, 92, 92, 255]
];

const STYLE_BYTE_LENGTH = 288;
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

const COLORMAP_INDEXES: Record<SpatialAnalysisColormap, number> = {
  uniform: 0,
  viridis: 1,
  inferno: 2,
  grayscale: 3,
  category: 4,
  mask: 5
};

/** Uniform layout shared by every layer; must match `SPATIAL_ANALYSIS_STYLE_WGSL`. */
const SPATIAL_ANALYSIS_STYLE_WGSL = /* wgsl */ `
struct SpatialAnalysisStyle {
  baseColor: vec4<f32>,
  noDataColor: vec4<f32>,
  palette: array<vec4<f32>, 8>,
  positionScale: vec2<f32>,
  positionOffset: vec2<f32>,
  valueRange: vec2<f32>,
  sizePixels: f32,
  opacity: f32,
  gridSize: vec2<u32>,
  valueFormat: u32,
  colormap: u32,
  useIds: u32,
  useExtent: u32,
  noDataValue: u32,
  valueDivisor: u32,
  paletteSize: u32,
  useWeights: u32,
  useClip: u32,
  binning: u32,
  hexagonRadius: f32,
  rowOrder: u32,
  valueScale: f32,
  discardAtOrBelow: f32,
  useDiscard: u32,
  sqrtScale: u32,
  useValueIndices: u32,
  _padding1: u32,
};

@group(0) @binding(auto) var<uniform> spatialAnalysisStyle: SpatialAnalysisStyle;
@group(0) @binding(auto) var<storage, read> styleValues: array<u32>;
@group(0) @binding(auto) var<storage, read> styleExtent: array<f32>;
@group(0) @binding(auto) var<storage, read> styleValueIndices: array<u32>;

// Maps a drawn row (point, segment, or cell) to the row of styleValues that colors it.
fn getSpatialAnalysisValueRow(row: u32) -> u32 {
  if (spatialAnalysisStyle.useValueIndices != 0u) {
    return styleValueIndices[row];
  }
  return row / max(spatialAnalysisStyle.valueDivisor, 1u);
}

fn spatialAnalysisViridis(t: f32) -> vec3<f32> {
  let c0 = vec3<f32>(0.2777273272234177, 0.005407344544966578, 0.3340998053353061);
  let c1 = vec3<f32>(0.1050930431085774, 1.404613529898575, 1.384590162594685);
  let c2 = vec3<f32>(-0.3308618287255563, 0.214847559468213, 0.09509516302823659);
  let c3 = vec3<f32>(-4.634230498983486, -5.799100973351585, -19.33244095627987);
  let c4 = vec3<f32>(6.228269936347081, 14.17993336680509, 56.69055260068105);
  let c5 = vec3<f32>(4.776384997670288, -13.74514537774601, -65.35303263337234);
  let c6 = vec3<f32>(-5.435455855934631, 4.645852612178535, 26.3124352495832);
  return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}

fn spatialAnalysisInferno(t: f32) -> vec3<f32> {
  let c0 = vec3<f32>(0.0002189403691192265, 0.001651004631001012, -0.01948089843709184);
  let c1 = vec3<f32>(0.1065134194856116, 0.5639564367884091, 3.932712388889277);
  let c2 = vec3<f32>(11.60249308247187, -3.972853965665698, -15.9423941062914);
  let c3 = vec3<f32>(-41.70399613139459, 17.43639888205313, 44.35414519872813);
  let c4 = vec3<f32>(77.162935699427, -33.40235894210092, -81.80730925738993);
  let c5 = vec3<f32>(-71.31942824499214, 32.62606426397723, 73.20951985803202);
  let c6 = vec3<f32>(25.13112622477341, -12.24266895238567, -23.07032500287172);
  return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}

// Returns the style color for one value row. Alpha 0 means "discard".
fn getSpatialAnalysisColor(valueRow: u32) -> vec4<f32> {
  let colormap = spatialAnalysisStyle.colormap;
  if (colormap == 0u || spatialAnalysisStyle.valueFormat == 0u) {
    return spatialAnalysisStyle.baseColor;
  }
  let raw = styleValues[valueRow];
  if (spatialAnalysisStyle.valueFormat == 1u && raw == spatialAnalysisStyle.noDataValue) {
    return spatialAnalysisStyle.noDataColor;
  }
  if (colormap == 4u) {
    let size = max(spatialAnalysisStyle.paletteSize, 1u);
    return spatialAnalysisStyle.palette[raw % size];
  }
  if (colormap == 5u) {
    let on = select(raw != 0u, bitcast<f32>(raw) != 0.0, spatialAnalysisStyle.valueFormat == 2u);
    return select(spatialAnalysisStyle.noDataColor, spatialAnalysisStyle.baseColor, on);
  }
  var value = select(f32(raw), bitcast<f32>(raw), spatialAnalysisStyle.valueFormat == 2u);
  // NaN and +/-infinity (for example unreached network costs) are no data. Test the exponent bits:
  // compilers may fold the self-comparison NaN test to false.
  let isNonFinite = spatialAnalysisStyle.valueFormat == 2u && (raw & 0x7fffffffu) >= 0x7f800000u;
  if (isNonFinite || abs(value) > 3.0e38) {
    return spatialAnalysisStyle.noDataColor;
  }
  value = value * spatialAnalysisStyle.valueScale;
  if (spatialAnalysisStyle.useDiscard != 0u && value <= spatialAnalysisStyle.discardAtOrBelow) {
    return vec4<f32>(0.0);
  }
  var range = spatialAnalysisStyle.valueRange;
  if (spatialAnalysisStyle.useExtent != 0u) {
    range = vec2<f32>(styleExtent[0], styleExtent[1]) * spatialAnalysisStyle.valueScale;
  }
  var t = clamp((value - range.x) / max(range.y - range.x, 1e-20), 0.0, 1.0);
  if (spatialAnalysisStyle.sqrtScale != 0u) {
    t = sqrt(t);
  }
  var rgb = vec3<f32>(t);
  if (colormap == 1u) {
    rgb = spatialAnalysisViridis(t);
  } else if (colormap == 2u) {
    rgb = spatialAnalysisInferno(t);
  }
  return vec4<f32>(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)), spatialAnalysisStyle.baseColor.a);
}

fn getSpatialAnalysisPosition(position: vec2<f32>) -> vec2<f32> {
  return position * spatialAnalysisStyle.positionScale + spatialAnalysisStyle.positionOffset;
}

fn projectSpatialAnalysisPosition(position: vec2<f32>) -> vec4<f32> {
  var clipPosition = project_position_to_clipspace(vec3<f32>(position, 0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  // Deck's project32 matrices use OpenGL depth; WebGPU clip space requires [0, w].
  clipPosition.z = (clipPosition.z + clipPosition.w) * 0.5;
  return clipPosition;
}
`;

const POINT_SHADER = /* wgsl */ `
${SPATIAL_ANALYSIS_STYLE_WGSL}
@group(0) @binding(auto) var<storage, read> pointPositions: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> pointIds: array<u32>;

struct PointVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) corner: vec2<f32>,
  @location(1) color: vec4<f32>,
};

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> PointVertexOutput {
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
  );
  var output: PointVertexOutput;
  var row = instanceIndex;
  if (spatialAnalysisStyle.useIds != 0u) {
    row = pointIds[instanceIndex];
  }
  let source = pointPositions[row];
  let color = getSpatialAnalysisColor(getSpatialAnalysisValueRow(row));
  if (source.x != source.x || source.y != source.y || color.a <= 0.0) {
    output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    output.color = vec4<f32>(0.0);
    output.corner = vec2<f32>(0.0);
    return output;
  }
  let corner = corners[vertexIndex];
  var clipPosition = projectSpatialAnalysisPosition(getSpatialAnalysisPosition(source));
  clipPosition = vec4<f32>(
    clipPosition.xy + project_pixel_size_to_clipspace(corner * spatialAnalysisStyle.sizePixels),
    clipPosition.z,
    clipPosition.w
  );
  output.position = clipPosition;
  output.corner = corner;
  output.color = color;
  return output;
}

@fragment fn fragmentMain(input: PointVertexOutput) -> @location(0) vec4<f32> {
  let radiusSquared = dot(input.corner, input.corner);
  if (radiusSquared > 1.0) { discard; }
  let coverage = 1.0 - smoothstep(0.6, 1.0, radiusSquared);
  return vec4<f32>(input.color.rgb, input.color.a * spatialAnalysisStyle.opacity * coverage);
}
`;

const SEGMENT_SHADER = /* wgsl */ `
${SPATIAL_ANALYSIS_STYLE_WGSL}
@group(0) @binding(auto) var<storage, read> segmentPositions: array<vec4<f32>>;
@group(0) @binding(auto) var<storage, read> segmentIds: array<u32>;
@group(0) @binding(auto) var<storage, read> segmentWeights: array<f32>;
@group(0) @binding(auto) var<storage, read> segmentClip: array<vec2<f32>>;

struct SegmentVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) side: f32,
  @location(1) color: vec4<f32>,
};

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> SegmentVertexOutput {
  // (t along the segment, side across it)
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
  );
  var output: SegmentVertexOutput;
  var row = instanceIndex;
  if (spatialAnalysisStyle.useIds != 0u) {
    row = segmentIds[instanceIndex];
  }
  let segment = segmentPositions[row];
  var color = getSpatialAnalysisColor(getSpatialAnalysisValueRow(row));
  if (spatialAnalysisStyle.useWeights != 0u) {
    color.a = color.a * segmentWeights[row];
  }
  var clip = vec2<f32>(0.0, 1.0);
  if (spatialAnalysisStyle.useClip != 0u) {
    clip = segmentClip[row];
  }
  if (segment.x != segment.x || segment.z != segment.z || color.a <= 0.0 || clip.y <= clip.x) {
    output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    output.color = vec4<f32>(0.0);
    output.side = 0.0;
    return output;
  }
  let start = getSpatialAnalysisPosition(segment.xy);
  let end = getSpatialAnalysisPosition(segment.zw);
  let clippedStart = mix(start, end, clip.x);
  let clippedEnd = mix(start, end, clip.y);
  let startClip = projectSpatialAnalysisPosition(clippedStart);
  let endClip = projectSpatialAnalysisPosition(clippedEnd);
  let corner = corners[vertexIndex];
  let screenDirection = (endClip.xy / endClip.w - startClip.xy / startClip.w) * project.viewportSize;
  let directionLength = length(screenDirection);
  var direction = vec2<f32>(1.0, 0.0);
  if (directionLength > 1e-6) {
    direction = screenDirection / directionLength;
  }
  let normal = vec2<f32>(-direction.y, direction.x);
  var clipPosition = mix(startClip, endClip, corner.x);
  // Extend caps by half the width so polylines join without gaps.
  let along = direction * (corner.x * 2.0 - 1.0) * 0.5;
  clipPosition = vec4<f32>(
    clipPosition.xy + project_pixel_size_to_clipspace((normal * corner.y + along) * spatialAnalysisStyle.sizePixels * 0.5),
    clipPosition.z,
    clipPosition.w
  );
  output.position = clipPosition;
  output.side = corner.y;
  output.color = color;
  return output;
}

@fragment fn fragmentMain(input: SegmentVertexOutput) -> @location(0) vec4<f32> {
  let coverage = 1.0 - smoothstep(0.55, 1.0, abs(input.side));
  return vec4<f32>(input.color.rgb, input.color.a * spatialAnalysisStyle.opacity * coverage);
}
`;

const RASTER_SHADER = /* wgsl */ `
${SPATIAL_ANALYSIS_STYLE_WGSL}
${GPU_POINT_DENSITY_HEXAGON_WGSL}
@group(0) @binding(auto) var<storage, read> rasterBounds: array<f32>;
@group(0) @binding(auto) var<storage, read> rasterHexagonRadius: array<f32>;

struct RasterVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) worldPosition: vec2<f32>,
};

@vertex fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> RasterVertexOutput {
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0)
  );
  let minimum = vec2<f32>(rasterBounds[0], rasterBounds[1]);
  let maximum = vec2<f32>(rasterBounds[2], rasterBounds[3]);
  var extent = maximum - minimum;
  let radius = rasterHexagonRadius[0];
  if (spatialAnalysisStyle.binning == 1u) {
    // Hexagon lattices extend half a cell beyond the cell-center bounds.
    extent = vec2<f32>(
      f32(spatialAnalysisStyle.gridSize.x) * 1.7320508 * radius,
      f32(spatialAnalysisStyle.gridSize.y) * 1.5 * radius
    );
  }
  let margin = select(vec2<f32>(0.0), vec2<f32>(radius), spatialAnalysisStyle.binning == 1u);
  let worldPosition = minimum - margin + corners[vertexIndex] * (extent + margin * 2.0);
  var output: RasterVertexOutput;
  output.position = projectSpatialAnalysisPosition(worldPosition);
  output.worldPosition = worldPosition;
  return output;
}

@fragment fn fragmentMain(input: RasterVertexOutput) -> @location(0) vec4<f32> {
  let minimum = vec2<f32>(rasterBounds[0], rasterBounds[1]);
  let maximum = vec2<f32>(rasterBounds[2], rasterBounds[3]);
  let columns = i32(spatialAnalysisStyle.gridSize.x);
  let rows = i32(spatialAnalysisStyle.gridSize.y);
  var column: i32;
  var row: i32;
  if (spatialAnalysisStyle.binning == 1u) {
    let cell = getPointDensityHexagonCell(
      input.worldPosition.x, input.worldPosition.y, minimum.x, minimum.y, rasterHexagonRadius[0]
    );
    column = cell.x;
    row = cell.y;
  } else {
    let local = (input.worldPosition - minimum) / max(maximum - minimum, vec2<f32>(1e-20));
    column = i32(floor(local.x * f32(columns)));
    row = i32(floor(local.y * f32(rows)));
    if (spatialAnalysisStyle.rowOrder == 1u) {
      row = rows - 1 - row;
    }
  }
  if (column < 0 || row < 0 || column >= columns || row >= rows) {
    discard;
  }
  let color = getSpatialAnalysisColor(getSpatialAnalysisValueRow(u32(row * columns + column)));
  if (color.a <= 0.0) {
    discard;
  }
  return vec4<f32>(color.rgb, color.a * spatialAnalysisStyle.opacity);
}
`;

type SpatialAnalysisLayerState = {
  model: Model | null;
  styleBuffer: Buffer | null;
  placeholderBuffer: Buffer | null;
  boundsBuffer: Buffer | null;
  radiusBuffer: Buffer | null;
};

type CommonLayerProps = LayerProps & SpatialAnalysisStyleProps & SpatialAnalysisInstanceProps;

/** Packs {@link SpatialAnalysisStyleProps} into the shared uniform layout. */
function writeStyle(
  buffer: Buffer,
  props: CommonLayerProps,
  extra: {
    sizePixels: number;
    useIds: boolean;
    useWeights?: boolean;
    useClip?: boolean;
    gridSize?: readonly [number, number];
    binning?: number;
    hexagonRadius?: number;
    rowOrder?: number;
  }
): void {
  const data = new ArrayBuffer(STYLE_BYTE_LENGTH);
  const floats = new Float32Array(data);
  const words = new Uint32Array(data);
  const writeColor = (offset: number, color: SpatialAnalysisColor) => {
    floats.set(
      [color[0] / 255, color[1] / 255, color[2] / 255, (color[3] ?? 255) / 255],
      offset / 4
    );
  };
  writeColor(0, props.color ?? [255, 255, 255, 255]);
  writeColor(16, props.noDataColor ?? [0, 0, 0, 0]);
  const palette = props.palette ?? DEFAULT_PALETTE;
  for (let index = 0; index < 8; index++) {
    writeColor(32 + index * 16, palette[index % palette.length] ?? [255, 255, 255, 255]);
  }
  const valueFormat = props.values ? (props.valueFormat ?? 'float32') : 'none';
  floats.set(props.positionScale ?? [1, 1], 160 / 4);
  floats.set(props.positionOffset ?? [0, 0], 168 / 4);
  floats.set(props.valueRange ?? [0, 1], 176 / 4);
  floats[184 / 4] = extra.sizePixels;
  floats[188 / 4] = props.opacity ?? 1;
  words.set(extra.gridSize ?? [1, 1], 192 / 4);
  words[200 / 4] = valueFormat === 'none' ? 0 : valueFormat === 'uint32' ? 1 : 2;
  words[204 / 4] = COLORMAP_INDEXES[props.colormap ?? 'uniform'];
  words[208 / 4] = extra.useIds ? 1 : 0;
  words[212 / 4] = props.extent ? 1 : 0;
  words[216 / 4] = props.noDataValue ?? 0xffffffff;
  words[220 / 4] = Math.max(1, props.valueDivisor ?? 1);
  words[224 / 4] = Math.min(8, palette.length);
  words[228 / 4] = extra.useWeights ? 1 : 0;
  words[232 / 4] = extra.useClip ? 1 : 0;
  words[236 / 4] = extra.binning ?? 0;
  floats[240 / 4] = extra.hexagonRadius ?? 1;
  words[244 / 4] = extra.rowOrder ?? 0;
  floats[248 / 4] = props.valueScale ?? 1;
  floats[252 / 4] = props.discardAtOrBelow ?? 0;
  words[256 / 4] = props.discardAtOrBelow === undefined ? 0 : 1;
  words[260 / 4] = props.sqrtScale ? 1 : 0;
  words[264 / 4] = props.valueIndices ? 1 : 0;
  buffer.write(new Uint8Array(data));
}

/** Shared lifecycle: one model with storage bindings and an owned style uniform buffer. */
abstract class SpatialAnalysisBaseLayer<PropsT extends CommonLayerProps> extends Layer<PropsT> {
  static override layerName = 'SpatialAnalysisBaseLayer';
  static override defaultProps = {
    coordinateSystem: COORDINATE_SYSTEM.METER_OFFSETS,
    parameters: BLEND_PARAMETERS
  };

  override getAttributeManager() {
    return null;
  }

  protected abstract getShaderSource(): string;
  protected abstract getVertexCount(): number;
  protected abstract getBindings(placeholder: Buffer): Record<string, Buffer>;
  protected abstract writeLayerStyle(styleBuffer: Buffer): void;

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
    const state: SpatialAnalysisLayerState = {
      model: null,
      styleBuffer,
      placeholderBuffer,
      boundsBuffer: this.createBoundsBuffer(device),
      radiusBuffer: this.createBoundsBuffer(device)
    };
    this.setState(state);
    const model = new Model(device, {
      ...this.getShaders({modules: [project32], source: this.getShaderSource()}),
      id: `${this.id}-model`,
      topology: 'triangle-list',
      isInstanced: true,
      vertexCount: this.getVertexCount(),
      instanceCount: 0,
      bufferLayout: [],
      bindings: {...this.getBindings(placeholderBuffer), spatialAnalysisStyle: styleBuffer},
      parameters: BLEND_PARAMETERS
    });
    this.setState({model});
  }

  protected createBoundsBuffer(_device: Device): Buffer | null {
    return null;
  }

  override updateState(parameters: UpdateParameters<this>): void {
    super.updateState(parameters);
    const {model, styleBuffer, placeholderBuffer} = this.state as SpatialAnalysisLayerState;
    if (model && styleBuffer && placeholderBuffer) {
      model.setBindings({
        ...this.getBindings(placeholderBuffer),
        spatialAnalysisStyle: styleBuffer
      });
    }
  }

  override getModels(): Model[] {
    const model = (this.state as SpatialAnalysisLayerState).model;
    return model ? [model] : [];
  }

  override draw({renderPass}: {renderPass: RenderPass}): void {
    const {model, styleBuffer} = this.state as SpatialAnalysisLayerState;
    if (!model || !styleBuffer) return;
    this.writeLayerStyle(styleBuffer);
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
    const state = this.state as SpatialAnalysisLayerState;
    state.model?.destroy();
    state.styleBuffer?.destroy();
    state.placeholderBuffer?.destroy();
    state.boundsBuffer?.destroy();
    state.radiusBuffer?.destroy();
    this.setState({
      model: null,
      styleBuffer: null,
      placeholderBuffer: null,
      boundsBuffer: null,
      radiusBuffer: null
    });
    super.finalizeState(context);
  }

  protected getStyleBindings(placeholder: Buffer): Record<string, Buffer> {
    return {
      styleValues: this.props.values ?? placeholder,
      styleExtent: this.props.extent ?? placeholder,
      styleValueIndices: this.props.valueIndices ?? placeholder
    };
  }
}

/** Props for {@link SpatialAnalysisPointLayer}. */
export type SpatialAnalysisPointLayerProps = CommonLayerProps & {
  /** `float32x2` positions, one per row. */
  positions: Buffer;
  /** Disc radius in CSS pixels. Defaults to 3. */
  radiusPixels?: number;
};

/** Screen-space discs at GPU-resident positions, optionally gathered through compact IDs. */
export class SpatialAnalysisPointLayer extends SpatialAnalysisBaseLayer<SpatialAnalysisPointLayerProps> {
  static override layerName = 'SpatialAnalysisPointLayer';

  protected getShaderSource(): string {
    return POINT_SHADER;
  }
  protected getVertexCount(): number {
    return 6;
  }
  protected getBindings(placeholder: Buffer): Record<string, Buffer> {
    return {
      ...this.getStyleBindings(placeholder),
      pointPositions: this.props.positions,
      pointIds: this.props.ids ?? placeholder
    };
  }
  protected writeLayerStyle(styleBuffer: Buffer): void {
    writeStyle(styleBuffer, this.props, {
      sizePixels: this.props.radiusPixels ?? 3,
      useIds: Boolean(this.props.ids)
    });
  }
}

/** Props for {@link SpatialAnalysisSegmentLayer}. */
export type SpatialAnalysisSegmentLayerProps = CommonLayerProps & {
  /** Segments as `float32x4` rows `x0, y0, x1, y1` (equivalently two consecutive `float32x2`). */
  segments: Buffer;
  /** Line width in CSS pixels. Defaults to 2. */
  widthPixels?: number;
  /** Optional float32 per row multiplying alpha, for example time-window fade weights. */
  weights?: Buffer | null;
  /** Optional `float32x2` per row `[clipStart, clipEnd]` fractions drawn of each segment. */
  clipFractions?: Buffer | null;
};

/** Screen-space-width line segments from GPU-resident endpoint pairs. */
export class SpatialAnalysisSegmentLayer extends SpatialAnalysisBaseLayer<SpatialAnalysisSegmentLayerProps> {
  static override layerName = 'SpatialAnalysisSegmentLayer';

  protected getShaderSource(): string {
    return SEGMENT_SHADER;
  }
  protected getVertexCount(): number {
    return 6;
  }
  protected getBindings(placeholder: Buffer): Record<string, Buffer> {
    return {
      ...this.getStyleBindings(placeholder),
      segmentPositions: this.props.segments,
      segmentIds: this.props.ids ?? placeholder,
      segmentWeights: this.props.weights ?? placeholder,
      segmentClip: this.props.clipFractions ?? placeholder
    };
  }
  protected writeLayerStyle(styleBuffer: Buffer): void {
    writeStyle(styleBuffer, this.props, {
      sizePixels: this.props.widthPixels ?? 2,
      useIds: Boolean(this.props.ids),
      useWeights: Boolean(this.props.weights),
      useClip: Boolean(this.props.clipFractions)
    });
  }
}

/** Props for {@link SpatialAnalysisRasterLayer}. */
export type SpatialAnalysisRasterLayerProps = LayerProps &
  SpatialAnalysisStyleProps & {
    /** `[columns, rows]` of the cell grid. */
    gridSize: readonly [number, number];
    /**
     * `[minX, minY, maxX, maxY]` meters as a literal, or a GPU buffer holding four float32 values
     * (for example the same `GPUParameterBuffer` a density contributor reads).
     * For hexagons, `minX, minY` is the lattice origin (center of hexagon 0, 0).
     */
    bounds: readonly [number, number, number, number] | Buffer;
    /** Cell shape. Defaults to `'grid'`. */
    binning?: 'grid' | 'hexagon';
    /**
     * Hexagon center-to-vertex radius in meters, as a literal or a GPU buffer whose first float32
     * is the radius (for example the contributor's per-frame `hexagonRadius` parameter buffer).
     */
    hexagonRadius?: number | Buffer;
    /** Which edge row 0 lies on. Defaults to `'south'` (row index grows with y). */
    rowOrigin?: 'south' | 'north';
  };

/** One quad over the grid bounds whose fragments read the cell value from a storage buffer. */
export class SpatialAnalysisRasterLayer extends SpatialAnalysisBaseLayer<
  SpatialAnalysisRasterLayerProps & SpatialAnalysisInstanceProps
> {
  static override layerName = 'SpatialAnalysisRasterLayer';

  protected getShaderSource(): string {
    return RASTER_SHADER;
  }
  protected getVertexCount(): number {
    return 6;
  }
  protected override createBoundsBuffer(device: Device): Buffer | null {
    return device.createBuffer({
      id: `${this.id}-bounds`,
      byteLength: 16,
      usage: Buffer.STORAGE | Buffer.COPY_DST
    });
  }
  protected getBindings(placeholder: Buffer): Record<string, Buffer> {
    const {boundsBuffer, radiusBuffer} = this.state as SpatialAnalysisLayerState;
    const {bounds, hexagonRadius} = this.props;
    return {
      ...this.getStyleBindings(placeholder),
      rasterBounds: bounds instanceof Buffer ? bounds : (boundsBuffer ?? placeholder),
      rasterHexagonRadius:
        hexagonRadius instanceof Buffer ? hexagonRadius : (radiusBuffer ?? placeholder)
    };
  }
  protected writeLayerStyle(styleBuffer: Buffer): void {
    const {bounds, hexagonRadius = 1} = this.props;
    const {boundsBuffer, radiusBuffer} = this.state as SpatialAnalysisLayerState;
    if (!(bounds instanceof Buffer) && boundsBuffer) {
      boundsBuffer.write(Float32Array.from(bounds));
    }
    if (!(hexagonRadius instanceof Buffer) && radiusBuffer) {
      radiusBuffer.write(Float32Array.of(hexagonRadius, 0, 0, 0));
    }
    writeStyle(styleBuffer, this.props, {
      sizePixels: 1,
      useIds: false,
      gridSize: this.props.gridSize,
      binning: this.props.binning === 'hexagon' ? 1 : 0,
      rowOrder: this.props.rowOrigin === 'north' ? 1 : 0
    });
  }
  override draw({renderPass}: {renderPass: RenderPass}): void {
    const {model, styleBuffer} = this.state as SpatialAnalysisLayerState;
    if (!model || !styleBuffer) return;
    this.writeLayerStyle(styleBuffer);
    model.setInstanceCount(1);
    model.draw(renderPass);
  }
}
