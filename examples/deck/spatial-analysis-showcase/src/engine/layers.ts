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
import type {ClassTable} from '../cartography/types';
import {evaluateZoomStops, type ZoomStops} from '../cartography/zoom';
import {COLORMAP_INDEXES, type RampName} from './ramps';
import {
  MAXIMUM_PALETTE_SIZE,
  RGBA_COLORMAP_INDEX,
  SPATIAL_ANALYSIS_STYLE_WGSL,
  buildPointShader,
  buildPolygonShader,
  buildRasterShader,
  buildSegmentShader,
  getSpatialAnalysisStyleWgsl,
  type SpatialAnalysisShaderFeatures
} from './layer-wgsl';

export {SPATIAL_ANALYSIS_STYLE_WGSL, getSpatialAnalysisStyleWgsl};
export type {SpatialAnalysisShaderFeatures};

/**
 * Generic deck.gl layers that draw analysis contributor outputs straight from GPU storage buffers.
 *
 * None of these layers reads a buffer back or repacks it: positions, values, compact IDs, extents,
 * fade weights, clip fractions, and GPU-resident bounds are bound as read-only storage, and counts
 * can come from a GPU-written indirect draw record. All positions are planar meters rendered with
 * `COORDINATE_SYSTEM.METER_OFFSETS` around `coordinateOrigin`, optionally through an affine
 * `positionScale`/`positionOffset` (for example raster cell coordinates to meters).
 *
 * Style prop groups of {@link SpatialAnalysisStyleProps} (all optional; defaults draw as before):
 *
 * - Values and colour: `values`, `valueFormat`, `valueDivisor`, `valueIndices`, `colormap` (a
 *   ramp, `uniform`, `category`, `mask`, or packed `rgba`), `valueRange`, `extent`, `valueScale`,
 *   `sqrtScale`, `color`, `noDataColor`, `noDataValue`, `discardAtOrBelow`, `palette`,
 *   `reverseRamp`, `rampRange` (ramp trim).
 * - Classes: `classBreaks`, `classColors` (exact colours with alpha), `classTable` (one object
 *   that sets breaks, colours and hatch).
 * - Zoom: `opacityStops`, and ZoomStops on the size props of the point and segment layers.
 * - Highlight: `highlightClasses`, `dimOpacity`, `highlightActive` (and the `highlight` channel).
 * - Per-row channels (one packed buffer): `instanceChannels`, `channelStride`, `channels`
 *   (`alpha`, `highlight`, `heading`, `width`), `alphaDomain`, `alphaOutput`, `widthDomain`,
 *   `widthRange`, `widthScale`.
 * - Hatch: `hatchClasses`, `hatchNoData`, `hatchColor`, `hatchSpacingPixels`, `hatchWidthPixels`.
 * - Compare: `compareSide` (with {@link setCompareState}).
 * - Compositing: `blending` (`normal`, `additive`, `multiply`) and the layer `opacity`.
 * - Layer specific: point `shape`, `fillOpacity`, `angleDegrees`; segment `dashArray`, `cap`;
 *   raster `outlineClasses`.
 *
 * Anchors that chapter subclasses patch by exact string replacement. They must stay in the shader
 * text the base class returns for default props (greps over this file prove it):
 *
 * - point: `project_pixel_size_to_clipspace(corner * spatialAnalysisStyle.sizePixels)` (b6);
 *   `@group(0) @binding(auto) var<storage, read> pointPositions` and everything before it (b4).
 * - segment: `segment.x != segment.x || segment.z != segment.z`, the adjacent lines
 *   `let segment = segmentPositions[row];` / `var color = getSpatialAnalysisColor(getSpatialAnalysisValueRow(row));`,
 *   `var<storage, read> segmentPositions: array<vec4<f32>>` (b15, b9); the `segmentPositions`
 *   binding line (b4).
 * - style: `fn getSpatialAnalysisColor(valueRow: u32) -> vec4<f32> {` (b2) and
 *   `return spatialAnalysisStyle.palette[raw % size];` (b13).
 * - raster: `@group(0) @binding(auto) var<storage, read> rasterBounds`,
 *   `output.position = projectSpatialAnalysisPosition(worldPosition);`, `@vertex fn vertexMain`,
 *   `@fragment fn fragmentMain`, `RasterVertexOutput`, `rasterBounds` (b15, b9, b16, b7, wildfire).
 *
 * Storage bindings per stage with default props: point 6, segment 7, polygon 5, raster 5. The
 * per-row `instanceChannels` buffer adds one binding, only while it is set, and no other feature
 * adds one: the maximum is point 7, segment 8, polygon 6, raster 6 (the device ceiling is 8).
 */

/** Element format of a `values` buffer. */
export type SpatialAnalysisValueFormat = 'none' | 'uint32' | 'float32';

/**
 * How a value becomes a color.
 *
 * - `uniform`: `color` for every row (or scalar values read as classes when `classColors` is set).
 * - any {@link RampName} (`viridis`, `magma`, `inferno`, `cividis`, `diverging`, `grayscale`): normalized scalar `(value - min) / (max - min)`, using
 *   `valueRange` or a GPU `[min, max]` `extent` buffer.
 * - `category`: `palette[value % paletteSize]` (or a built-in categorical palette).
 * - `mask`: `color` when the value is nonzero, otherwise `noDataColor`.
 * - `rgba`: `values` is a `uint32` per row holding RGBA8 (red in the low byte), drawn as is (times
 *   the layer opacity). Use `valueFormat: 'uint32'`. Used by the relief helper.
 */
export type SpatialAnalysisColormap = 'uniform' | 'category' | 'mask' | 'rgba' | RampName;

/** RGBA color with 0-255 channels. */
export type SpatialAnalysisColor = readonly [number, number, number, number?];

/**
 * How a layer's fragments combine with what is already drawn.
 *
 * - `normal`: alpha blending.
 * - `additive`: colors add up where marks overlap.
 * - `multiply`: the layer colour multiplies what Deck already drew *in the deck canvas* (for
 *   example a relief underlay layer drawn first), keeps the destination alpha and draws nothing
 *   where the destination is empty. It cannot reach the basemap canvas, which is a different
 *   canvas under the deck canvas.
 */
export type SpatialAnalysisBlending = 'normal' | 'additive' | 'multiply';

/**
 * Shape of a point mark. `circle` is a disc, `square` an axis-aligned square and `hexagon` a
 * flat-topped hexagon (both tile without seams); `triangle`, `diamond`, `star` and `cross` are
 * drawn shapes; `ring` is stroke only (see `outlineWidthPixels`); `chevron` and `arrow` point north
 * and rotate by the `heading` channel or `angleDegrees`.
 */
export type SpatialAnalysisPointShape =
  | 'circle'
  | 'square'
  | 'hexagon'
  | 'triangle'
  | 'diamond'
  | 'star'
  | 'cross'
  | 'ring'
  | 'chevron'
  | 'arrow';

/** Float indexes (inside one value row of `instanceChannels`) of the per-row channels. */
export type SpatialAnalysisChannels = {
  /** Value-by-alpha: maps through `alphaDomain` and `alphaOutput` to an alpha multiplier. */
  alpha?: number;
  /** Non-zero highlights the row; the other rows dim to `dimOpacity` (linked brushing). */
  highlight?: number;
  /** Points: heading in degrees clockwise from north, used by `chevron` and `arrow` shapes. */
  heading?: number;
  /** Segments: width value mapped through `widthDomain`, `widthRange` and `widthScale`. */
  width?: number;
};

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
  /** Up to 16 colors for `category`. Defaults to a built-in 8-color palette. */
  palette?: readonly SpatialAnalysisColor[];
  /**
   * Classed (stepped) color for ramp colormaps: interior class breaks, ascending, at most 15
   * (16 classes), in value units after `valueScale`. A value takes class `k` = the number of
   * breaks `<= value`, colored with the ramp at `k / (classCount - 1)` (`getClassColors`), so a
   * `classes` legend with the same `breaks` and ramp matches exactly. `valueRange`, `extent` and
   * `sqrtScale` are ignored while classed. Omit or pass `[]` for a continuous ramp.
   */
  classBreaks?: readonly number[];
  /**
   * Exact class colours (at most 16), low class first, used with `classBreaks`: class `k` is drawn
   * with `classColors[k]` including its alpha, so alpha 0 makes that class transparent (a "nothing
   * to say" class). They take the palette slots, so classed colours and `category` are exclusive,
   * and `reverseRamp` and `rampRange` are ignored. Works with any colormap; with `'uniform'` (or
   * none) the values are still read as scalars. The layer `opacity` still multiplies.
   */
  classColors?: readonly SpatialAnalysisColor[];
  /**
   * One classification object (`makeClassTable`) that sets `classBreaks`, `classColors` and
   * `hatchClasses` (and `hatchNoData` from `noData.hatched`), so layer and legend cannot drift
   * apart. Explicit props win over the table.
   */
  classTable?: ClassTable;
  /** Flips the ramp (high values take the low end). Match it with the legend's `reverseRamp`. */
  reverseRamp?: boolean;
  /**
   * Ramp trim `[t0, t1]`: after normalising, `sqrtScale` and `reverseRamp`, the ramp is sampled at
   * `mix(t0, t1, t)`, so a ramp can skip its near-white or near-black end. Applies to classed
   * ramps too. A legend applies the same trim (reverse first, then trim). Defaults to `[0, 1]`.
   */
  rampRange?: readonly [number, number];
  /**
   * Layer opacity by zoom (`[[zoom, opacity], ...]`, or a number), evaluated at the current
   * viewport zoom every frame. Overrides `opacity` when set.
   */
  opacityStops?: ZoomStops;
  /**
   * Highlight set: when set, rows whose class index (classed colour) or category index
   * (`category`: `value % paletteSize`, at most 16 groups) is not in the set are drawn at alpha
   * `dimOpacity` times their own, as are rows without a group (no data, continuous ramps). `[]`
   * dims every row, `null` or `undefined` highlights nothing (no dimming). A 16-bit mask in the
   * uniform, no buffer: use it for legend isolate and class brushing.
   */
  highlightClasses?: readonly number[] | null;
  /** Alpha multiplier of dimmed rows (see `highlightClasses`, `channels.highlight`). Default 0.12. */
  dimOpacity?: number;
  /**
   * Turns the `highlight` channel on or off without rebuilding the buffer. When `false`, no row is
   * dimmed by the channel. Defaults to `true`.
   */
  highlightActive?: boolean;
  /**
   * One packed float32 buffer of per-row data, read at `valueRow * channelStride + index` where
   * `valueRow` is `getSpatialAnalysisValueRow(row)` (like `values`). Its WGSL is compiled in, with
   * one extra storage binding, only while this prop is set; changing it between null and a buffer
   * rebuilds the layer's model once.
   */
  instanceChannels?: Buffer | null;
  /** Floats per value row of `instanceChannels`. Defaults to 1. */
  channelStride?: number;
  /** Float index of each channel inside a row of `instanceChannels`; omit the unused ones. */
  channels?: SpatialAnalysisChannels;
  /**
   * Value-by-alpha: the `[low, high]` of the `alpha` channel mapped linearly to `alphaOutput`,
   * clamped. NaN takes `alphaOutput[0]`. Defaults to `[0, 1]`.
   */
  alphaDomain?: readonly [number, number];
  /** Alpha multipliers at the two ends of `alphaDomain`. Defaults to `[0.25, 1]`. */
  alphaOutput?: readonly [number, number];
  /** Segments: `[low, high]` of the `width` channel mapped to `widthRange`. Defaults to `[0, 1]`. */
  widthDomain?: readonly [number, number];
  /** Segments: line width in CSS pixels at the two ends of `widthDomain`. Defaults to `[1, 6]`. */
  widthRange?: readonly [number, number];
  /**
   * Segments: `'sqrt'` (default, area-true for flows) or `'linear'` mapping of the `width`
   * channel, applied before `widthMinPixels` and `widthMaxPixels`.
   */
  widthScale?: 'linear' | 'sqrt';
  /**
   * Class (or category) indices drawn with hatch stripes (suppressed, low n, not significant).
   * Screen-space 45 degree stripes: a hatched class whose colour has alpha 0 draws only the
   * stripes. Fully supported on raster and polygon layers; points and segments carry the class
   * index to the fragment stage as a flat varying and hatch too.
   */
  hatchClasses?: readonly number[];
  /** Hatch the rows that are no data (NaN, infinity or `noDataValue`). */
  hatchNoData?: boolean;
  /** Stripe colour. Defaults to ink at 0.35 alpha, `[31, 41, 51, 90]`; use a light ink on dark grounds. */
  hatchColor?: SpatialAnalysisColor;
  /** Distance between stripes in CSS pixels, across the stripes. Defaults to 4. */
  hatchSpacingPixels?: number;
  /** Stripe width in CSS pixels. Defaults to 1. */
  hatchWidthPixels?: number;
  /**
   * Swipe compare: which side of a divider this layer belongs to. Which side shows, and where the
   * divider sits, is the module state set with {@link setCompareState}. Layers without
   * `compareSide` are never affected.
   */
  compareSide?: 'a' | 'b';
  /**
   * `'normal'` (default) alpha blending, `'additive'`: colors add up where marks overlap, so
   * dense clusters glow (use it on a dark ground with low-alpha points), or `'multiply'`: the
   * colour multiplies what the deck canvas already shows (see {@link SpatialAnalysisBlending}).
   * Switching it changes the render pipeline (cached), not the shader.
   */
  blending?: SpatialAnalysisBlending;
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

/** Byte length of the style uniform; the offsets are in `STYLE_OFFSETS`. */
const STYLE_BYTE_LENGTH = 704;
const MAXIMUM_CLASS_BREAKS = 15;
/** `channel*` uniform value for "this channel is not bound". */
const NO_CHANNEL = 0xffffffff;
const DEFAULT_HATCH_COLOR: SpatialAnalysisColor = [31, 41, 51, 90];
const POINT_SHAPES: Record<SpatialAnalysisPointShape, number> = {
  circle: 0,
  square: 1,
  hexagon: 2,
  triangle: 3,
  diamond: 4,
  star: 5,
  cross: 6,
  ring: 7,
  chevron: 8,
  arrow: 9
};
const CAP_CODES = {butt: 1, square: 2, round: 3} as const;
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
 * Additive blending that stays valid premultiplied alpha for the transparent Deck canvas: color
 * accumulates `rgb * a`, alpha accumulates `a`, both saturate at 1, so color never exceeds alpha.
 */
const ADDITIVE_BLEND_PARAMETERS = {
  ...BLEND_PARAMETERS,
  blendColorDstFactor: 'one',
  blendAlphaDstFactor: 'one'
} as const;

/**
 * Multiply blending: the fragment (shader output `mix(vec3(1), rgb, a)`) multiplies the colour
 * already in the target (`src` factor `dst`, `dst` factor `zero`), and the destination alpha stays
 * (`src` factor `zero`, `dst` factor `one`).
 */
const MULTIPLY_BLEND_PARAMETERS = {
  ...BLEND_PARAMETERS,
  blendColorSrcFactor: 'dst',
  blendColorDstFactor: 'zero',
  blendAlphaSrcFactor: 'zero',
  blendAlphaDstFactor: 'one'
} as const;

function getBlendParameters(blending: SpatialAnalysisBlending | undefined) {
  if (blending === 'additive') return ADDITIVE_BLEND_PARAMETERS;
  if (blending === 'multiply') return MULTIPLY_BLEND_PARAMETERS;
  return BLEND_PARAMETERS;
}

/** Which side of the swipe divider shows (see {@link setCompareState}). */
export type SpatialAnalysisCompareState = {
  /** Divider position as a fraction (0-1) of the viewport width, measured from the left. */
  position: number;
  /** `'both'`: side `a` left of the divider, side `b` right of it; or only one side. */
  showing: 'a' | 'b' | 'both';
};

let compareState: SpatialAnalysisCompareState | null = null;

/**
 * Sets the swipe-compare state read by every layer with a `compareSide`, each time it draws. In
 * `'both'` mode a fragment of side `a` is discarded right of the divider and one of side `b` left of
 * it; `'a'` and `'b'` show only that side; `null` (or layers without `compareSide`) shows
 * everything. This is module state, not a layer prop: call it, then ask the deck to redraw (for
 * example while dragging the divider, `deck.redraw('compare')`).
 */
export function setCompareState(state: SpatialAnalysisCompareState | null): void {
  compareState = state ? {position: state.position, showing: state.showing} : null;
}

/** The compare state last set with {@link setCompareState}. */
export function getCompareState(): SpatialAnalysisCompareState | null {
  return compareState;
}

/** Per-frame facts the style needs from the viewport and canvas; see `getStyleFrame`. */
export type SpatialAnalysisStyleFrame = {
  /** Viewport zoom (0 for views without one), for `ZoomStops`. */
  zoom: number;
  /** Device pixels per CSS pixel. */
  devicePixelRatio: number;
  /** Viewport left edge in CSS pixels. */
  viewportX: number;
  /** Viewport width in CSS pixels. */
  viewportWidth: number;
};

const DEFAULT_STYLE_FRAME: SpatialAnalysisStyleFrame = {
  zoom: 0,
  devicePixelRatio: 1,
  viewportX: 0,
  viewportWidth: 1
};

type CommonLayerProps = LayerProps & SpatialAnalysisStyleProps & SpatialAnalysisInstanceProps;

/** Props every spatial-analysis layer accepts: the layer props, style and instance sources. */
export type SpatialAnalysisCommonLayerProps = CommonLayerProps;

/** Mark size and stroke values a layer writes into the style next to its colour props. */
export type SpatialAnalysisStyleSizeOptions = {
  /** Base size in CSS pixels (point radius, line width), already evaluated at the zoom. */
  sizePixels: number;
  /** Base size in metres; overrides `sizePixels` when set. */
  sizeMeters?: number;
  minPixels?: number;
  maxPixels?: number;
  outlineColor?: SpatialAnalysisColor;
  outlineWidthPixels?: number;
  shape?: SpatialAnalysisPointShape;
  useSizeValues?: boolean;
  sizeMaximumValue?: number;
  sizeScale?: 'sqrt' | 'linear';
  /** Points: fill alpha multiplier (the outline keeps its own). Defaults to 1. */
  fillOpacity?: number;
  /** Points: heading in degrees of `chevron` and `arrow` when there is no heading channel. */
  angleDegrees?: number;
  /** Segments: `[dash, gap]` in CSS pixels. */
  dashArray?: readonly [number, number];
  /** Segments: end cap shape. */
  cap?: 'butt' | 'square' | 'round';
  /** Rasters: class-boundary outline. */
  outlineClasses?: {color: SpatialAnalysisColor; widthPixels?: number};
};

/**
 * Byte offsets of the style uniform; must match `SpatialAnalysisStyle` in `layer-wgsl.ts`.
 *
 * | offset | members (type) |
 * | --- | --- |
 * | 0, 16 | baseColor, noDataColor (vec4) |
 * | 32 | palette (16 x vec4, 256 bytes; classColors live here too) |
 * | 288-316 | positionScale, positionOffset, valueRange (vec2), sizePixels, opacity |
 * | 320 | gridSize (vec2 u32) |
 * | 328-396 | valueFormat ... classCount (u32/f32, 4 bytes each) |
 * | 400 | classBreaks (4 x vec4) |
 * | 464 | outlineColor (vec4), then outlineWidthPixels ... sizeScale up to 516 |
 * | 520, 524 | rampLow, rampHigh |
 * | 528, 544 | outlineClassColor, hatchColor (vec4) |
 * | 560-590 | alphaDomain, alphaOutput, widthDomain, widthRange (vec2 each) |
 * | 592-636 | hatch, dpr, class outline width, highlight, fill opacity, angle, useClassColors |
 * | 640-664 | channelStride and channel indexes, widthScale, highlightActive |
 * | 668-676 | dashLength, dashGap, cap |
 * | 680-688 | compareSide, compareDivider, multiply |
 * | 692-700 | padding to a multiple of 16 (704 total) |
 */
const STYLE_OFFSETS = {
  baseColor: 0,
  noDataColor: 16,
  palette: 32,
  positionScale: 288,
  positionOffset: 296,
  valueRange: 304,
  sizePixels: 312,
  opacity: 316,
  gridSize: 320,
  valueFormat: 328,
  colormap: 332,
  useIds: 336,
  useExtent: 340,
  noDataValue: 344,
  valueDivisor: 348,
  paletteSize: 352,
  useWeights: 356,
  useClip: 360,
  binning: 364,
  hexagonRadius: 368,
  rowOrder: 372,
  valueScale: 376,
  discardAtOrBelow: 380,
  useDiscard: 384,
  sqrtScale: 388,
  useValueIndices: 392,
  classCount: 396,
  classBreaks: 400,
  outlineColor: 464,
  outlineWidthPixels: 480,
  reverseRamp: 484,
  sizeUnits: 488,
  sizeMeters: 492,
  minSizePixels: 496,
  maxSizePixels: 500,
  shape: 504,
  useSizeValues: 508,
  sizeMaximumValue: 512,
  sizeScale: 516,
  rampLow: 520,
  rampHigh: 524,
  outlineClassColor: 528,
  hatchColor: 544,
  alphaDomain: 560,
  alphaOutput: 568,
  widthDomain: 576,
  widthRange: 584,
  hatchMask: 592,
  hatchNoData: 596,
  hatchSpacing: 600,
  hatchWidth: 604,
  devicePixelRatio: 608,
  outlineClassWidth: 612,
  highlightMask: 616,
  highlightUse: 620,
  dimOpacity: 624,
  fillOpacity: 628,
  angleDegrees: 632,
  useClassColors: 636,
  channelStride: 640,
  channelAlpha: 644,
  channelHighlight: 648,
  channelHeading: 652,
  channelWidth: 656,
  widthScale: 660,
  highlightActive: 664,
  dashLength: 668,
  dashGap: 672,
  cap: 676,
  compareSide: 680,
  compareDivider: 684,
  multiply: 688
} as const;

/** The class breaks, colours and hatch of a style: explicit props first, then `classTable`. */
function resolveClassStyle(props: SpatialAnalysisStyleProps) {
  const table = props.classTable;
  return {
    breaks: (props.classBreaks ?? table?.breaks ?? []).slice(0, MAXIMUM_CLASS_BREAKS),
    colors: props.classColors ?? table?.colors,
    hatchClasses: props.hatchClasses ?? table?.hatched,
    hatchNoData: props.hatchNoData ?? table?.noData?.hatched ?? false
  };
}

/** True when a layer must carry the hatch flag to its fragment stage. */
function hasHatch(props: SpatialAnalysisStyleProps): boolean {
  const {hatchClasses, hatchNoData} = resolveClassStyle(props);
  return Boolean(hatchClasses?.length) || hatchNoData;
}

/** Style-uniform colormap index of a colormap name (`rgba` has its own constant). */
function getColormapIndex(colormap: SpatialAnalysisColormap | undefined): number {
  if (colormap === 'rgba') return RGBA_COLORMAP_INDEX;
  return COLORMAP_INDEXES[colormap ?? 'uniform'];
}

/** Bit mask of class or category indices below `limit`. */
function getIndexMask(indices: readonly number[] | null | undefined, limit: number): number {
  let mask = 0;
  for (const index of indices ?? []) {
    if (Number.isInteger(index) && index >= 0 && index < limit) mask |= 1 << index;
  }
  return mask >>> 0;
}

/** The side and divider the compare uniforms take for a layer (see {@link setCompareState}). */
function getCompareUniforms(
  side: 'a' | 'b' | undefined,
  frame: SpatialAnalysisStyleFrame
): {side: number; divider: number} {
  if (!side || !compareState) return {side: 0, divider: 0};
  const {showing, position} = compareState;
  if (showing === 'both') {
    const divider =
      (frame.viewportX + Math.min(Math.max(position, 0), 1) * frame.viewportWidth) *
      frame.devicePixelRatio;
    return {side: side === 'a' ? 1 : 2, divider};
  }
  // Only one side shows; the other is hidden entirely (code 3).
  return {side: showing === side ? 0 : 3, divider: 0};
}

/**
 * Packs {@link SpatialAnalysisStyleProps} into the shared uniform layout (`STYLE_OFFSETS`). Layer
 * subclasses in other files call it from `writeLayerStyle` with their size and shape options; pass
 * the `frame` of {@link SpatialAnalysisBaseLayer.getStyleFrame} so zoom stops, hatch, outlines and
 * compare see the current viewport.
 */
export function writeStyle(
  buffer: Buffer,
  props: CommonLayerProps,
  extra: SpatialAnalysisStyleSizeOptions & {
    useIds: boolean;
    useWeights?: boolean;
    useClip?: boolean;
    gridSize?: readonly [number, number];
    binning?: number;
    hexagonRadius?: number;
    rowOrder?: number;
    frame?: SpatialAnalysisStyleFrame;
  }
): void {
  const frame = extra.frame ?? DEFAULT_STYLE_FRAME;
  const data = new ArrayBuffer(STYLE_BYTE_LENGTH);
  const floats = new Float32Array(data);
  const words = new Uint32Array(data);
  const o = STYLE_OFFSETS;
  const writeColor = (offset: number, color: SpatialAnalysisColor) => {
    floats.set(
      [color[0] / 255, color[1] / 255, color[2] / 255, (color[3] ?? 255) / 255],
      offset / 4
    );
  };
  const colormap = props.colormap ?? 'uniform';
  const classStyle = resolveClassStyle(props);
  const breaks = classStyle.breaks;
  // Exact class colours take the palette slots; they are exclusive with the category palette.
  const useClassColors =
    breaks.length > 0 &&
    Boolean(classStyle.colors?.length) &&
    colormap !== 'category' &&
    colormap !== 'mask';
  writeColor(o.baseColor, props.color ?? [255, 255, 255, 255]);
  writeColor(o.noDataColor, props.noDataColor ?? [0, 0, 0, 0]);
  const palette = useClassColors
    ? classStyle.colors!
    : props.palette?.length
      ? props.palette
      : DEFAULT_PALETTE;
  const paletteSize = Math.min(MAXIMUM_PALETTE_SIZE, palette.length);
  for (let index = 0; index < MAXIMUM_PALETTE_SIZE; index++) {
    writeColor(o.palette + index * 16, palette[index % paletteSize] ?? [255, 255, 255, 255]);
  }
  const valueFormat = props.values ? (props.valueFormat ?? 'float32') : 'none';
  floats.set(props.positionScale ?? [1, 1], o.positionScale / 4);
  floats.set(props.positionOffset ?? [0, 0], o.positionOffset / 4);
  floats.set(props.valueRange ?? [0, 1], o.valueRange / 4);
  floats[o.sizePixels / 4] = extra.sizePixels;
  floats[o.opacity / 4] =
    props.opacityStops !== undefined
      ? evaluateZoomStops(props.opacityStops, frame.zoom)
      : (props.opacity ?? 1);
  words.set(extra.gridSize ?? [1, 1], o.gridSize / 4);
  words[o.valueFormat / 4] = valueFormat === 'none' ? 0 : valueFormat === 'uint32' ? 1 : 2;
  words[o.colormap / 4] = getColormapIndex(colormap);
  words[o.useIds / 4] = extra.useIds ? 1 : 0;
  words[o.useExtent / 4] = props.extent ? 1 : 0;
  words[o.noDataValue / 4] = props.noDataValue ?? 0xffffffff;
  words[o.valueDivisor / 4] = Math.max(1, props.valueDivisor ?? 1);
  words[o.paletteSize / 4] = paletteSize;
  words[o.useWeights / 4] = extra.useWeights ? 1 : 0;
  words[o.useClip / 4] = extra.useClip ? 1 : 0;
  words[o.binning / 4] = extra.binning ?? 0;
  floats[o.hexagonRadius / 4] = extra.hexagonRadius ?? 1;
  words[o.rowOrder / 4] = extra.rowOrder ?? 0;
  floats[o.valueScale / 4] = props.valueScale ?? 1;
  floats[o.discardAtOrBelow / 4] = props.discardAtOrBelow ?? 0;
  words[o.useDiscard / 4] = props.discardAtOrBelow === undefined ? 0 : 1;
  words[o.sqrtScale / 4] = props.sqrtScale ? 1 : 0;
  words[o.useValueIndices / 4] = props.valueIndices ? 1 : 0;
  words[o.classCount / 4] = breaks.length ? breaks.length + 1 : 0;
  floats.set(breaks, o.classBreaks / 4);
  writeColor(o.outlineColor, extra.outlineColor ?? [255, 255, 255, 255]);
  floats[o.outlineWidthPixels / 4] = Math.max(0, extra.outlineWidthPixels ?? 0);
  words[o.reverseRamp / 4] = props.reverseRamp ? 1 : 0;
  words[o.sizeUnits / 4] = extra.sizeMeters !== undefined ? 1 : 0;
  floats[o.sizeMeters / 4] = extra.sizeMeters ?? 0;
  floats[o.minSizePixels / 4] = extra.minPixels ?? 0;
  floats[o.maxSizePixels / 4] = extra.maxPixels ?? 1e6;
  words[o.shape / 4] = POINT_SHAPES[extra.shape ?? 'circle'];
  words[o.useSizeValues / 4] = extra.useSizeValues ? 1 : 0;
  floats[o.sizeMaximumValue / 4] = extra.sizeMaximumValue ?? 1;
  words[o.sizeScale / 4] = extra.sizeScale === 'linear' ? 1 : 0;
  const [rampLow, rampHigh] = props.rampRange ?? [0, 1];
  floats[o.rampLow / 4] = rampLow;
  floats[o.rampHigh / 4] = rampHigh;
  writeColor(o.outlineClassColor, extra.outlineClasses?.color ?? [0, 0, 0, 255]);
  floats[o.outlineClassWidth / 4] = extra.outlineClasses
    ? Math.max(0, extra.outlineClasses.widthPixels ?? 1)
    : 0;
  writeColor(o.hatchColor, props.hatchColor ?? DEFAULT_HATCH_COLOR);
  words[o.hatchMask / 4] = getIndexMask(classStyle.hatchClasses, 32);
  words[o.hatchNoData / 4] = classStyle.hatchNoData ? 1 : 0;
  floats[o.hatchSpacing / 4] = props.hatchSpacingPixels ?? 4;
  floats[o.hatchWidth / 4] = props.hatchWidthPixels ?? 1;
  floats[o.devicePixelRatio / 4] = frame.devicePixelRatio;
  words[o.highlightMask / 4] = getIndexMask(props.highlightClasses, MAXIMUM_PALETTE_SIZE);
  words[o.highlightUse / 4] = props.highlightClasses ? 1 : 0;
  floats[o.dimOpacity / 4] = props.dimOpacity ?? 0.12;
  floats[o.fillOpacity / 4] = extra.fillOpacity ?? 1;
  floats[o.angleDegrees / 4] = extra.angleDegrees ?? 0;
  words[o.useClassColors / 4] = useClassColors ? 1 : 0;
  const channels = props.instanceChannels ? (props.channels ?? {}) : {};
  words[o.channelStride / 4] = Math.max(1, Math.floor(props.channelStride ?? 1));
  words[o.channelAlpha / 4] = channels.alpha ?? NO_CHANNEL;
  words[o.channelHighlight / 4] = channels.highlight ?? NO_CHANNEL;
  words[o.channelHeading / 4] = channels.heading ?? NO_CHANNEL;
  words[o.channelWidth / 4] = channels.width ?? NO_CHANNEL;
  floats.set(props.alphaDomain ?? [0, 1], o.alphaDomain / 4);
  floats.set(props.alphaOutput ?? [0.25, 1], o.alphaOutput / 4);
  floats.set(props.widthDomain ?? [0, 1], o.widthDomain / 4);
  floats.set(props.widthRange ?? [1, 6], o.widthRange / 4);
  words[o.widthScale / 4] = props.widthScale === 'linear' ? 1 : 0;
  words[o.highlightActive / 4] = props.highlightActive === false ? 0 : 1;
  const dashArray = extra.dashArray;
  floats[o.dashLength / 4] = dashArray && dashArray[0] > 0 ? dashArray[0] : 0;
  floats[o.dashGap / 4] = dashArray ? Math.max(0, dashArray[1]) : 0;
  words[o.cap / 4] = extra.cap ? CAP_CODES[extra.cap] : 0;
  const compare = getCompareUniforms(props.compareSide, frame);
  words[o.compareSide / 4] = compare.side;
  floats[o.compareDivider / 4] = compare.divider;
  words[o.multiply / 4] = props.blending === 'multiply' ? 1 : 0;
  buffer.write(new Uint8Array(data));
}

/** Names of the `var<...> name:` declarations in a WGSL source. */
function getDeclaredBindings(source: string): Set<string> {
  return new Set(
    [...source.matchAll(/var<(?:uniform|storage)[^>]*>\s+(\w+)\s*:/g)].map(match => match[1])
  );
}

/** Per-layer GPU state of a spatial-analysis layer (`this.state`). */
export type SpatialAnalysisLayerState = {
  model: Model | null;
  styleBuffer: Buffer | null;
  placeholderBuffer: Buffer | null;
  boundsBuffer: Buffer | null;
  radiusBuffer: Buffer | null;
  /** Storage and uniform variable names the layer's shader declares. */
  declaredBindings: Set<string> | null;
  /** The shader source the current model was built from. */
  source: string;
  blending: SpatialAnalysisBlending;
};

/**
 * Shared lifecycle: one model with storage bindings and an owned style uniform buffer.
 *
 * Subclasses (in other files too) implement `getShaderSource`, `getVertexCount`, `getBindings` and
 * `writeLayerStyle` (call {@link writeStyle} with `frame: this.getStyleFrame()`). The model is
 * rebuilt whenever `getShaderSource()` returns a different string than the one it was built from,
 * so props that change the shader (hatch, channels, `tessellation`) take effect without a new id.
 * Bindings are filtered to the names the shader declares, so patched shaders never warn.
 */
export abstract class SpatialAnalysisBaseLayer<
  PropsT extends CommonLayerProps
> extends Layer<PropsT> {
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

  /**
   * The optional shader features the current props ask for: per-row channels and hatch. Layers
   * add their own (segment line style, raster class outline).
   */
  protected getShaderFeatures(): SpatialAnalysisShaderFeatures {
    return {
      channels: Boolean(this.props.instanceChannels),
      hatch: hasHatch(this.props)
    };
  }

  /**
   * The viewport and canvas facts the style needs this frame: zoom for zoom stops, the device
   * pixel ratio for hatch and outlines, and the viewport extent for swipe compare.
   */
  protected getStyleFrame(): SpatialAnalysisStyleFrame {
    const viewport = this.context?.viewport;
    if (!viewport) return DEFAULT_STYLE_FRAME;
    let devicePixelRatio = 1;
    try {
      devicePixelRatio = this.context.device.getDefaultCanvasContext().cssToDeviceRatio();
    } catch {
      // No canvas context (headless device): CSS pixels are device pixels.
    }
    return {
      zoom: Number.isFinite(viewport.zoom) ? viewport.zoom : 0,
      devicePixelRatio,
      viewportX: viewport.x ?? 0,
      viewportWidth: viewport.width
    };
  }

  /** Builds the model for `source` and records the bindings it declares. */
  private createModel(device: Device, source: string, blending: SpatialAnalysisBlending): Model {
    const state = this.state as SpatialAnalysisLayerState;
    this.setState({declaredBindings: getDeclaredBindings(source), source});
    return new Model(device, {
      ...this.getShaders({modules: [project32], source}),
      id: `${this.id}-model`,
      topology: 'triangle-list',
      isInstanced: true,
      vertexCount: this.getVertexCount(),
      instanceCount: 0,
      bufferLayout: [],
      bindings: this.getModelBindings(state.placeholderBuffer!, state.styleBuffer!),
      parameters: getBlendParameters(blending)
    });
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
    const blending = this.props.blending ?? 'normal';
    const state: SpatialAnalysisLayerState = {
      model: null,
      styleBuffer,
      placeholderBuffer,
      boundsBuffer: this.createBoundsBuffer(device),
      radiusBuffer: this.createBoundsBuffer(device),
      declaredBindings: null,
      source: '',
      blending
    };
    this.setState(state);
    this.setState({model: this.createModel(device, this.getShaderSource(), blending)});
  }

  protected createBoundsBuffer(_device: Device): Buffer | null {
    return null;
  }

  /**
   * The layer's bindings plus the style uniform, keeping only names the shader declares, so a
   * subclass with a patched shader never triggers "binding not found" warnings.
   */
  private getModelBindings(placeholder: Buffer, styleBuffer: Buffer): Record<string, Buffer> {
    const declared = (this.state as SpatialAnalysisLayerState).declaredBindings;
    const bindings: Record<string, Buffer> = {
      ...this.getBindings(placeholder),
      spatialAnalysisStyle: styleBuffer
    };
    if (!declared) return bindings;
    return Object.fromEntries(Object.entries(bindings).filter(([name]) => declared.has(name)));
  }

  override updateState(parameters: UpdateParameters<this>): void {
    super.updateState(parameters);
    const state = this.state as SpatialAnalysisLayerState;
    const {styleBuffer, placeholderBuffer} = state;
    if (!state.model || !styleBuffer || !placeholderBuffer) return;
    const blending = this.props.blending ?? 'normal';
    const source = this.getShaderSource();
    if (source !== state.source) {
      // A prop changed the shader text (a feature variant, tessellation): rebuild the model.
      state.model.destroy();
      this.setState({model: this.createModel(this.context.device, source, blending), blending});
      return;
    }
    const model = state.model;
    model.setBindings(this.getModelBindings(placeholderBuffer, styleBuffer));
    if (blending !== state.blending) {
      model.setParameters(getBlendParameters(blending));
      this.setState({blending});
    }
  }

  override getModels(): Model[] {
    const model = (this.state as SpatialAnalysisLayerState).model;
    return model ? [model] : [];
  }

  /**
   * Deck merges the layer's `parameters` prop (the default `BLEND_PARAMETERS`) over the model's
   * pipeline parameters on every draw, which would undo `additive` and `multiply`. The blend
   * factors of the `blending` prop are applied last, so they win.
   */
  override _drawLayer(options: Parameters<Layer['_drawLayer']>[0]): void {
    super._drawLayer({
      ...options,
      parameters: {...options.parameters, ...getBlendParameters(this.props.blending)}
    });
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

  /** The bindings of the shared style block: values, extent, value indices and channels. */
  protected getStyleBindings(placeholder: Buffer): Record<string, Buffer> {
    return {
      styleValues: this.props.values ?? placeholder,
      styleExtent: this.props.extent ?? placeholder,
      styleValueIndices: this.props.valueIndices ?? placeholder,
      styleChannels: this.props.instanceChannels ?? placeholder
    };
  }
}

/** Props for {@link SpatialAnalysisPointLayer}. */
export type SpatialAnalysisPointLayerProps = CommonLayerProps & {
  /** `float32x2` positions, one per row. */
  positions: Buffer;
  /** Mark radius in CSS pixels, or by zoom (`ZoomStops`). Defaults to 3. */
  radiusPixels?: ZoomStops;
  /**
   * Mark radius in ground metres (half the side for squares, circumradius for hexagons).
   * Overrides `radiusPixels`: marks grow with zoom, foreshorten under pitch, and metre-sized
   * squares or hexagons tile like cells.
   */
  radiusMeters?: number;
  /** Lower clamp of the drawn radius in CSS pixels (keeps metre-sized marks visible when zoomed out). */
  radiusMinPixels?: ZoomStops;
  /** Upper clamp of the drawn radius in CSS pixels. */
  radiusMaxPixels?: ZoomStops;
  /** Mark shape. Defaults to `'circle'` (soft-edged when there is no outline). */
  shape?: SpatialAnalysisPointShape;
  /** Outline (halo) colour around each mark. */
  outlineColor?: SpatialAnalysisColor;
  /**
   * Outline width in CSS pixels, drawn outside the fill radius, or by zoom. A 1-1.5 px outline in
   * the ground colour separates overlapping points; a dark outline frames proportional symbols.
   * For `shape: 'ring'` it is the stroke width (1.5 by default), drawn inside the radius in the
   * data colour.
   */
  outlineWidthPixels?: ZoomStops;
  /**
   * Fill alpha multiplier, independent of the outline: `fillOpacity: 0` with an outline draws
   * hollow dots. Defaults to 1.
   */
  fillOpacity?: number;
  /** Heading in degrees clockwise from north of `chevron` and `arrow` when there is no `heading` channel. */
  angleDegrees?: number;
  /**
   * Proportional symbols: a float32 per value row (same indexing as `values`) scaling the radius
   * as `radius * sqrt(value / sizeMaximumValue)` (`sizeScale: 'sqrt'`, area true, the default) or
   * linearly, before the pixel clamps. Use `getProportionalRadius` / `getSizeLegendEntries` from
   * `cartography/proportional` for the matching `size` legend.
   */
  sizeValues?: Buffer | null;
  /** Value drawn at the full radius. Defaults to 1. */
  sizeMaximumValue?: number;
  /** `'sqrt'` (default) or `'linear'`. */
  sizeScale?: 'sqrt' | 'linear';
};

/**
 * Marks at GPU-resident positions, optionally gathered through compact IDs: soft discs by
 * default, or crisp circles, squares, hexagons, triangles, diamonds, stars, crosses, rings and
 * oriented chevrons and arrows, with an optional outline, sized in pixels (also by zoom) or
 * metres, optionally proportional to a value.
 */
export class SpatialAnalysisPointLayer extends SpatialAnalysisBaseLayer<SpatialAnalysisPointLayerProps> {
  static override layerName = 'SpatialAnalysisPointLayer';

  protected getShaderSource(): string {
    return buildPointShader(this.getShaderFeatures());
  }
  protected getVertexCount(): number {
    return 6;
  }
  protected getBindings(placeholder: Buffer): Record<string, Buffer> {
    return {
      ...this.getStyleBindings(placeholder),
      pointPositions: this.props.positions,
      pointIds: this.props.ids ?? placeholder,
      pointSizeValues: this.props.sizeValues ?? placeholder
    };
  }
  protected writeLayerStyle(styleBuffer: Buffer): void {
    const {props} = this;
    const frame = this.getStyleFrame();
    const meters = props.radiusMeters;
    const radius = evaluateZoomStops(props.radiusPixels ?? 3, frame.zoom);
    writeStyle(styleBuffer, props, {
      sizePixels: meters !== undefined ? Math.max(radius, 1) : radius,
      ...(meters !== undefined ? {sizeMeters: meters} : {}),
      minPixels:
        props.radiusMinPixels === undefined
          ? undefined
          : evaluateZoomStops(props.radiusMinPixels, frame.zoom),
      maxPixels:
        props.radiusMaxPixels === undefined
          ? undefined
          : evaluateZoomStops(props.radiusMaxPixels, frame.zoom),
      shape: props.shape,
      outlineColor: props.outlineColor,
      outlineWidthPixels:
        props.outlineWidthPixels === undefined
          ? undefined
          : evaluateZoomStops(props.outlineWidthPixels, frame.zoom),
      fillOpacity: props.fillOpacity,
      angleDegrees: props.angleDegrees,
      useSizeValues: Boolean(props.sizeValues),
      sizeMaximumValue: props.sizeMaximumValue,
      sizeScale: props.sizeScale,
      useIds: Boolean(props.ids),
      frame
    });
  }
}

/** Props for {@link SpatialAnalysisSegmentLayer}. */
export type SpatialAnalysisSegmentLayerProps = CommonLayerProps & {
  /** Segments as `float32x4` rows `x0, y0, x1, y1` (equivalently two consecutive `float32x2`). */
  segments: Buffer;
  /** Line width in CSS pixels, or by zoom (`ZoomStops`). Defaults to 2. */
  widthPixels?: ZoomStops;
  /** Line width in ground metres; overrides `widthPixels` (a road drawn at its true width). */
  widthMeters?: number;
  /** Lower clamp of the drawn width in CSS pixels. */
  widthMinPixels?: ZoomStops;
  /** Upper clamp of the drawn width in CSS pixels. */
  widthMaxPixels?: ZoomStops;
  /** Casing colour drawn on both sides of the line (a halo that lifts it off the ground). */
  outlineColor?: SpatialAnalysisColor;
  /** Casing width in CSS pixels on each side, or by zoom. */
  outlineWidthPixels?: ZoomStops;
  /** Optional float32 per row multiplying alpha, for example time-window fade weights. */
  weights?: Buffer | null;
  /** Optional `float32x2` per row `[clipStart, clipEnd]` fractions drawn of each segment. */
  clipFractions?: Buffer | null;
  /**
   * Dash and gap lengths in CSS pixels, for example `[6, 4]`. The pattern restarts at the start of
   * every segment (it does not continue along a polyline) and begins with a dash, so a long
   * polyline made of short segments looks dashed only if its segments are longer than the dash.
   */
  dashArray?: readonly [number, number];
  /**
   * End caps: `'butt'` (flat, exactly at the endpoint), `'square'` (extends half the width) or
   * `'round'`. When omitted, the original behaviour: a short square cap that extends each end by a
   * quarter of the total line width, so polylines join without gaps.
   */
  cap?: 'butt' | 'square' | 'round';
};

/** Screen-space-width line segments from GPU-resident endpoint pairs. */
export class SpatialAnalysisSegmentLayer extends SpatialAnalysisBaseLayer<SpatialAnalysisSegmentLayerProps> {
  static override layerName = 'SpatialAnalysisSegmentLayer';

  protected override getShaderFeatures(): SpatialAnalysisShaderFeatures {
    return {
      ...super.getShaderFeatures(),
      lineStyle: Boolean(this.props.dashArray) || this.props.cap !== undefined
    };
  }
  protected getShaderSource(): string {
    return buildSegmentShader(this.getShaderFeatures());
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
    const {props} = this;
    const frame = this.getStyleFrame();
    writeStyle(styleBuffer, props, {
      sizePixels: evaluateZoomStops(props.widthPixels ?? 2, frame.zoom),
      ...(props.widthMeters !== undefined ? {sizeMeters: props.widthMeters} : {}),
      minPixels:
        props.widthMinPixels === undefined
          ? undefined
          : evaluateZoomStops(props.widthMinPixels, frame.zoom),
      maxPixels:
        props.widthMaxPixels === undefined
          ? undefined
          : evaluateZoomStops(props.widthMaxPixels, frame.zoom),
      outlineColor: props.outlineColor,
      outlineWidthPixels:
        props.outlineWidthPixels === undefined
          ? undefined
          : evaluateZoomStops(props.outlineWidthPixels, frame.zoom),
      dashArray: props.dashArray,
      cap: props.cap,
      useIds: Boolean(this.props.ids),
      useWeights: Boolean(this.props.weights),
      useClip: Boolean(this.props.clipFractions),
      frame
    });
  }
}

/** Props for {@link SpatialAnalysisPolygonLayer}. */
export type SpatialAnalysisPolygonLayerProps = LayerProps &
  SpatialAnalysisStyleProps & {
    /** `float32x2` triangle-list vertices in planar metres (three per triangle). */
    triangles: Buffer;
    /**
     * `uint32` feature row of every triangle vertex. The row indexes `values` (through
     * `valueIndices` when given), so one value per feature colours all its triangles.
     */
    features: Buffer;
    /** Number of triangle vertices to draw. */
    vertexCount: number;
  };

/**
 * Filled polygons (a choropleth) coloured per feature straight from a GPU value buffer: every
 * colormap of the shared style works (ramps, `classBreaks` and `classColors`, `category`
 * palettes, `mask`, packed `rgba`), alpha 0 hides a feature, and `hatchClasses` hatches classes.
 * Triangulate once on the CPU with `buildPolygonMesh` (`cartography/polygon-mesh`) and upload
 * with `createPolygonMeshBuffers`; draw the outlines with a {@link SpatialAnalysisSegmentLayer}
 * over `outlineSegments` (colour them per feature with `valueIndices: outlineFeatures`).
 */
export class SpatialAnalysisPolygonLayer extends SpatialAnalysisBaseLayer<
  SpatialAnalysisPolygonLayerProps & SpatialAnalysisInstanceProps
> {
  static override layerName = 'SpatialAnalysisPolygonLayer';

  protected getShaderSource(): string {
    return buildPolygonShader(this.getShaderFeatures());
  }
  protected getVertexCount(): number {
    return Math.max(0, this.props.vertexCount);
  }
  protected getBindings(placeholder: Buffer): Record<string, Buffer> {
    return {
      ...this.getStyleBindings(placeholder),
      polygonVertices: this.props.triangles,
      polygonFeatures: this.props.features
    };
  }
  protected writeLayerStyle(styleBuffer: Buffer): void {
    writeStyle(styleBuffer, this.props, {
      sizePixels: 1,
      useIds: false,
      frame: this.getStyleFrame()
    });
  }
  override draw({renderPass}: {renderPass: RenderPass}): void {
    const {model, styleBuffer} = this.state as SpatialAnalysisLayerState;
    if (!model || !styleBuffer || this.props.vertexCount <= 0) return;
    this.writeLayerStyle(styleBuffer);
    model.setVertexCount(this.props.vertexCount);
    model.setInstanceCount(1);
    model.draw(renderPass);
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
    /**
     * Opt-in: draws the raster as `tessellation x tessellation` sub-quads whose corners are each
     * projected exactly. Use it when the raster spans more than about 100 km (for example
     * 32-64 for a continental extent): with the default `1`, one quad is interpolated linearly
     * in screen space and, because Web Mercator's scale varies with latitude, cells land tens of
     * kilometers from their true position at 1,800 km. Changing it rebuilds the layer's model.
     */
    tessellation?: number;
    /**
     * Draws a line where the class (classed colour) or category of a cell differs from its east,
     * north, west or south neighbour (basin edges, HAND classes, travel-time bands): the line is
     * `widthPixels` (CSS pixels, default 1) wide, centred on the shared cell edge, in `color`
     * (its alpha counts), and drawn even where a class is transparent. Square grids only; rows
     * without a class or category (continuous ramps, no data) draw no edge.
     */
    outlineClasses?: {color: SpatialAnalysisColor; widthPixels?: number};
  };

/** One quad over the grid bounds whose fragments read the cell value from a storage buffer. */
export class SpatialAnalysisRasterLayer extends SpatialAnalysisBaseLayer<
  SpatialAnalysisRasterLayerProps & SpatialAnalysisInstanceProps
> {
  static override layerName = 'SpatialAnalysisRasterLayer';

  protected override getShaderFeatures(): SpatialAnalysisShaderFeatures {
    return {...super.getShaderFeatures(), classOutline: Boolean(this.props.outlineClasses)};
  }
  protected getShaderSource(): string {
    return buildRasterShader(this.getTessellation(), this.getShaderFeatures());
  }
  protected getVertexCount(): number {
    return 6 * this.getTessellation() ** 2;
  }
  /** Sub-quads per side, clamped to `1..256`. */
  protected getTessellation(): number {
    return Math.min(256, Math.max(1, Math.floor(this.props.tessellation ?? 1)));
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
      rowOrder: this.props.rowOrigin === 'north' ? 1 : 0,
      outlineClasses: this.props.outlineClasses,
      frame: this.getStyleFrame()
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
