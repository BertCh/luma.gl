// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPU_POINT_DENSITY_HEXAGON_WGSL} from '@luma.gl/experimental/gpu-spatial-analysis';
import {getRampWgsl} from './ramps';

/**
 * WGSL of the shared spatial-analysis layers: the style uniform and its colour functions, and the
 * point, segment, polygon and raster shaders built on them. `layers.ts` owns the TypeScript side
 * (props, uniform packing, lifecycle) and must stay in step with the struct below.
 *
 * Chapter subclasses patch these shaders by exact string replacement, so the shaders keep fixed
 * anchors (see the list at the top of `layers.ts`). Optional features are compiled in only when a
 * layer asks for them ({@link SpatialAnalysisShaderFeatures}); the default shader of every layer
 * binds no extra storage buffer.
 */

/** Largest number of palette colours (and therefore classes) a style holds. */
export const MAXIMUM_PALETTE_SIZE = 16;

/** Colormap index of `colormap: 'rgba'` (packed RGBA8 values), well above every ramp index. */
export const RGBA_COLORMAP_INDEX = 1000;

/**
 * Optional shader features. Each one that is set adds code, and `channels` adds one storage
 * binding, to the shader text; a layer whose features change rebuilds its model.
 */
export type SpatialAnalysisShaderFeatures = {
  /** Binds `styleChannels` and compiles the per-row alpha, highlight, heading and width reads. */
  channels?: boolean;
  /** Carries the hatch flag of each row to the fragment stage and draws the stripes. */
  hatch?: boolean;
  /** Segment layers: dash pattern and cap shapes. */
  lineStyle?: boolean;
  /** Raster layers: class-boundary outlines. */
  classOutline?: boolean;
};

/** The rendered-once ramp functions (the ramp source is large; build it a single time). */
const RAMP_WGSL = getRampWgsl();

/**
 * Uniform layout shared by every layer; must match `STYLE_OFFSETS` in `layers.ts`. Byte offsets
 * are noted per member. `array<vec4>` has stride 16, vec4 aligns to 16 and vec2 to 8.
 */
function getStyleStructWgsl(): string {
  return /* wgsl */ `
struct SpatialAnalysisStyle {
  baseColor: vec4<f32>,
  noDataColor: vec4<f32>,
  palette: array<vec4<f32>, ${MAXIMUM_PALETTE_SIZE}>,
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
  classCount: u32,
  classBreaks: array<vec4<f32>, 4>,
  outlineColor: vec4<f32>,
  outlineWidthPixels: f32,
  reverseRamp: u32,
  sizeUnits: u32,
  sizeMeters: f32,
  minSizePixels: f32,
  maxSizePixels: f32,
  shape: u32,
  useSizeValues: u32,
  sizeMaximumValue: f32,
  sizeScale: u32,
  rampLow: f32,
  rampHigh: f32,
  outlineClassColor: vec4<f32>,
  hatchColor: vec4<f32>,
  alphaDomain: vec2<f32>,
  alphaOutput: vec2<f32>,
  widthDomain: vec2<f32>,
  widthRange: vec2<f32>,
  hatchMask: u32,
  hatchNoData: u32,
  hatchSpacing: f32,
  hatchWidth: f32,
  devicePixelRatio: f32,
  outlineClassWidth: f32,
  highlightMask: u32,
  highlightUse: u32,
  dimOpacity: f32,
  fillOpacity: f32,
  angleDegrees: f32,
  useClassColors: u32,
  channelStride: u32,
  channelAlpha: u32,
  channelHighlight: u32,
  channelHeading: u32,
  channelWidth: u32,
  widthScale: u32,
  highlightActive: u32,
  dashLength: f32,
  dashGap: f32,
  cap: u32,
  compareSide: u32,
  compareDivider: f32,
  multiply: u32,
  _padding0: u32,
  _padding1: u32,
  _padding2: u32,
};
`;
}

/**
 * The shared style block: the `SpatialAnalysisStyle` uniform, the `styleValues`, `styleExtent` and
 * `styleValueIndices` storage bindings (plus `styleChannels` when `features.channels`), and the
 * colour functions every layer calls (`getSpatialAnalysisColor`, `getSpatialAnalysisGroup`,
 * `getSpatialAnalysisHatch`, `applySpatialAnalysisHatch`, the compare and multiply helpers).
 * Subclass shaders keep this block and replace what follows their layer's first binding.
 */
export function getSpatialAnalysisStyleWgsl(features: SpatialAnalysisShaderFeatures = {}): string {
  const channels = Boolean(features.channels);
  return /* wgsl */ `
${getStyleStructWgsl()}
@group(0) @binding(auto) var<uniform> spatialAnalysisStyle: SpatialAnalysisStyle;
@group(0) @binding(auto) var<storage, read> styleValues: array<u32>;
@group(0) @binding(auto) var<storage, read> styleExtent: array<f32>;
@group(0) @binding(auto) var<storage, read> styleValueIndices: array<u32>;
${channels ? '@group(0) @binding(auto) var<storage, read> styleChannels: array<f32>;' : ''}

const SPATIAL_ANALYSIS_NO_GROUP = 0xffffffffu;
const SPATIAL_ANALYSIS_RGBA_COLORMAP = ${RGBA_COLORMAP_INDEX}u;
const SPATIAL_ANALYSIS_DEGREES_TO_RADIANS = 0.017453292519943295;

// Maps a drawn row (point, segment, or cell) to the row of styleValues that colors it.
fn getSpatialAnalysisValueRow(row: u32) -> u32 {
  if (spatialAnalysisStyle.useValueIndices != 0u) {
    return styleValueIndices[row];
  }
  return row / max(spatialAnalysisStyle.valueDivisor, 1u);
}

${RAMP_WGSL}

// Class index of a value: the number of class breaks at or below it.
fn getSpatialAnalysisClassIndex(value: f32) -> u32 {
  var index = 0u;
  for (var i = 0u; i + 1u < spatialAnalysisStyle.classCount; i = i + 1u) {
    if (value >= spatialAnalysisStyle.classBreaks[i / 4u][i % 4u]) {
      index = i + 1u;
    }
  }
  return index;
}

// Ramp position of a value under classed (stepped) color: class k of n takes k / (n - 1).
fn getSpatialAnalysisClassPosition(value: f32) -> f32 {
  let count = spatialAnalysisStyle.classCount;
  if (count <= 1u) {
    return 0.5;
  }
  return f32(getSpatialAnalysisClassIndex(value)) / f32(count - 1u);
}

// True when the raw word of a float32 is a NaN or an infinity (test the exponent bits: compilers
// may fold the self-comparison NaN test to false).
fn isSpatialAnalysisNanWord(raw: u32) -> bool {
  return (raw & 0x7fffffffu) >= 0x7f800000u;
}

// True when the raw word of a value is no data: the uint32 sentinel, or a float32 NaN or infinity.
fn isSpatialAnalysisNoDataWord(raw: u32) -> bool {
  if (spatialAnalysisStyle.valueFormat == 1u) {
    return raw == spatialAnalysisStyle.noDataValue;
  }
  return isSpatialAnalysisNanWord(raw);
}

// Maps the value of one row to a color before highlighting. Alpha 0 means "discard".
fn getSpatialAnalysisMappedColor(valueRow: u32) -> vec4<f32> {
  let colormap = spatialAnalysisStyle.colormap;
  let classed = spatialAnalysisStyle.useClassColors != 0u && spatialAnalysisStyle.classCount > 0u;
  if ((colormap == 0u && !classed) || spatialAnalysisStyle.valueFormat == 0u) {
    return spatialAnalysisStyle.baseColor;
  }
  let raw = styleValues[valueRow];
  if (colormap == SPATIAL_ANALYSIS_RGBA_COLORMAP) {
    return unpack4x8unorm(raw);
  }
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
  // NaN and +/-infinity (for example unreached network costs) are no data.
  let isNonFinite = spatialAnalysisStyle.valueFormat == 2u && (raw & 0x7fffffffu) >= 0x7f800000u;
  if (isNonFinite || abs(value) > 3.0e38) {
    return spatialAnalysisStyle.noDataColor;
  }
  value = value * spatialAnalysisStyle.valueScale;
  if (spatialAnalysisStyle.useDiscard != 0u && value <= spatialAnalysisStyle.discardAtOrBelow) {
    return vec4<f32>(0.0);
  }
  if (classed) {
    // Exact class colours, including their alpha (alpha 0 draws nothing for that class).
    return spatialAnalysisStyle.palette[getSpatialAnalysisClassIndex(value)];
  }
  var t = 0.0;
  if (spatialAnalysisStyle.classCount > 0u) {
    t = getSpatialAnalysisClassPosition(value);
  } else {
    var range = spatialAnalysisStyle.valueRange;
    if (spatialAnalysisStyle.useExtent != 0u) {
      range = vec2<f32>(styleExtent[0], styleExtent[1]) * spatialAnalysisStyle.valueScale;
    }
    t = clamp((value - range.x) / max(range.y - range.x, 1e-20), 0.0, 1.0);
    if (spatialAnalysisStyle.sqrtScale != 0u) {
      t = sqrt(t);
    }
  }
  if (spatialAnalysisStyle.reverseRamp != 0u) {
    t = 1.0 - t;
  }
  // Ramp trim: sample the part of the ramp between rampLow and rampHigh (after reverse).
  t = spatialAnalysisStyle.rampLow + (spatialAnalysisStyle.rampHigh - spatialAnalysisStyle.rampLow) * t;
  let rgb = spatialAnalysisSampleRamp(colormap, t);
  return vec4<f32>(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)), spatialAnalysisStyle.baseColor.a);
}

// Class index (classed colour) or category index (raw % paletteSize) of a row, or NO_GROUP for rows
// that are not grouped (continuous ramps, masks, packed colours) or have no data.
fn getSpatialAnalysisGroup(valueRow: u32) -> u32 {
  if (spatialAnalysisStyle.valueFormat == 0u) {
    return SPATIAL_ANALYSIS_NO_GROUP;
  }
  let colormap = spatialAnalysisStyle.colormap;
  if (colormap == SPATIAL_ANALYSIS_RGBA_COLORMAP || colormap == 5u) {
    return SPATIAL_ANALYSIS_NO_GROUP;
  }
  let raw = styleValues[valueRow];
  if (isSpatialAnalysisNoDataWord(raw) && (spatialAnalysisStyle.valueFormat == 1u || colormap != 4u)) {
    return SPATIAL_ANALYSIS_NO_GROUP;
  }
  if (colormap == 4u) {
    return raw % max(spatialAnalysisStyle.paletteSize, 1u);
  }
  if (spatialAnalysisStyle.classCount == 0u || (colormap == 0u && spatialAnalysisStyle.useClassColors == 0u)) {
    return SPATIAL_ANALYSIS_NO_GROUP;
  }
  let value = select(f32(raw), bitcast<f32>(raw), spatialAnalysisStyle.valueFormat == 2u);
  return getSpatialAnalysisClassIndex(value * spatialAnalysisStyle.valueScale);
}

// Returns the style color for one value row: mapped, then dimmed (highlight) and faded (channels).
// Alpha 0 means "discard".
fn getSpatialAnalysisColor(valueRow: u32) -> vec4<f32> {
  var color = getSpatialAnalysisMappedColor(valueRow);
  var dimmed = false;
  if (spatialAnalysisStyle.highlightUse != 0u) {
    let group = getSpatialAnalysisGroup(valueRow);
    dimmed = group >= 16u || ((spatialAnalysisStyle.highlightMask >> group) & 1u) == 0u;
  }
${
  channels
    ? `  if (spatialAnalysisStyle.channelAlpha != SPATIAL_ANALYSIS_NO_GROUP) {
    let alphaValue = getSpatialAnalysisChannel(valueRow, spatialAnalysisStyle.channelAlpha);
    let domain = spatialAnalysisStyle.alphaDomain;
    var alphaT = clamp((alphaValue - domain.x) / max(domain.y - domain.x, 1e-20), 0.0, 1.0);
    if (isSpatialAnalysisNanWord(bitcast<u32>(alphaValue))) {
      alphaT = 0.0;
    }
    color.a = color.a * mix(spatialAnalysisStyle.alphaOutput.x, spatialAnalysisStyle.alphaOutput.y, alphaT);
  }
  if (spatialAnalysisStyle.channelHighlight != SPATIAL_ANALYSIS_NO_GROUP && spatialAnalysisStyle.highlightActive != 0u) {
    if (getSpatialAnalysisChannel(valueRow, spatialAnalysisStyle.channelHighlight) == 0.0) {
      dimmed = true;
    }
  }
`
    : ''
}  if (dimmed) {
    color.a = color.a * spatialAnalysisStyle.dimOpacity;
  }
  return color;
}

// True when the row is no data (NaN, infinity or the uint32 sentinel of a values buffer).
fn getSpatialAnalysisIsNoData(valueRow: u32) -> bool {
  if (spatialAnalysisStyle.valueFormat == 0u || spatialAnalysisStyle.colormap == SPATIAL_ANALYSIS_RGBA_COLORMAP) {
    return false;
  }
  return isSpatialAnalysisNoDataWord(styleValues[valueRow]);
}

// True when the row is drawn with hatch stripes (a hatched class or category, or no data).
fn getSpatialAnalysisHatch(valueRow: u32) -> bool {
  if (spatialAnalysisStyle.hatchMask == 0u && spatialAnalysisStyle.hatchNoData == 0u) {
    return false;
  }
  if (spatialAnalysisStyle.hatchNoData != 0u && getSpatialAnalysisIsNoData(valueRow)) {
    return true;
  }
  if (spatialAnalysisStyle.hatchMask != 0u) {
    let group = getSpatialAnalysisGroup(valueRow);
    return group < 32u && ((spatialAnalysisStyle.hatchMask >> group) & 1u) != 0u;
  }
  return false;
}

// Lays 45 degree stripes (CSS-pixel spacing and width) over a colour. The fragment position is in
// device pixels, so the stripes stay the same size on every display.
fn applySpatialAnalysisHatch(color: vec4<f32>, fragmentPosition: vec4<f32>) -> vec4<f32> {
  let ratio = max(spatialAnalysisStyle.devicePixelRatio, 1e-3);
  let spacing = max(spatialAnalysisStyle.hatchSpacing, 1.0);
  let phase = (fragmentPosition.x + fragmentPosition.y) / ratio * 0.7071067811865476 / spacing;
  let fraction = fract(phase);
  let stripeDistance = min(fraction, 1.0 - fraction) * spacing;
  let halfWidth = max(spatialAnalysisStyle.hatchWidth, 0.1) * 0.5;
  let softness = 0.5 / ratio;
  let stripe = clamp((halfWidth + softness - stripeDistance) / (2.0 * softness), 0.0, 1.0) * spatialAnalysisStyle.hatchColor.a;
  let alpha = stripe + color.a * (1.0 - stripe);
  if (alpha <= 0.0) {
    return vec4<f32>(0.0);
  }
  let rgb = (spatialAnalysisStyle.hatchColor.rgb * stripe + color.rgb * color.a * (1.0 - stripe)) / alpha;
  return vec4<f32>(rgb, alpha);
}

// Swipe compare: true when a fragment of this layer's side lies on the hidden side of the divider.
// compareSide 0 = not compared, 1 = side a, 2 = side b, 3 = hidden entirely.
fn isSpatialAnalysisCompareHidden(fragmentPosition: vec4<f32>) -> bool {
  let side = spatialAnalysisStyle.compareSide;
  if (side == 0u) {
    return false;
  }
  if (side == 3u) {
    return true;
  }
  let isLeft = fragmentPosition.x < spatialAnalysisStyle.compareDivider;
  return select(isLeft, !isLeft, side == 1u);
}

// Final color of a fragment. Multiply blending needs mix(white, rgb, a): the blend state then
// multiplies it into what is already drawn and keeps the destination alpha.
fn finishSpatialAnalysisColor(color: vec4<f32>) -> vec4<f32> {
  if (spatialAnalysisStyle.multiply != 0u) {
    return vec4<f32>(mix(vec3<f32>(1.0), color.rgb, color.a), color.a);
  }
  return color;
}
${
  channels
    ? `
// Reads one float of a value row from the packed per-row channel buffer.
fn getSpatialAnalysisChannel(valueRow: u32, index: u32) -> f32 {
  return styleChannels[valueRow * spatialAnalysisStyle.channelStride + index];
}
`
    : ''
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
}

/** The style block of the default feature set. */
export const SPATIAL_ANALYSIS_STYLE_WGSL = getSpatialAnalysisStyleWgsl();

/** `@location` declarations that carry the hatch flag from the vertex to the fragment stage. */
function getHatchVarying(features: SpatialAnalysisShaderFeatures, location: number): string {
  return features.hatch ? `  @location(${location}) @interpolate(flat) hatched: f32,\n` : '';
}

/** Shader source of {@link SpatialAnalysisPointLayer}: one instanced quad per mark. */
export function buildPointShader(features: SpatialAnalysisShaderFeatures = {}): string {
  const hatch = Boolean(features.hatch);
  const channels = Boolean(features.channels);
  return /* wgsl */ `
${getSpatialAnalysisStyleWgsl(features)}
@group(0) @binding(auto) var<storage, read> pointPositions: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> pointIds: array<u32>;
@group(0) @binding(auto) var<storage, read> pointSizeValues: array<f32>;

struct PointVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) corner: vec2<f32>,
  @location(1) color: vec4<f32>,
  @location(2) fillFraction: f32,
  @location(3) totalPixels: f32,
${getHatchVarying(features, 4)}};

// Unclamped fill radius in CSS pixels: radiusPixels, or radiusMeters at the current zoom.
fn getSpatialAnalysisPointBasePixels() -> f32 {
  if (spatialAnalysisStyle.sizeUnits == 1u) {
    return project_meter_size_to_pixel(spatialAnalysisStyle.sizeMeters);
  }
  return spatialAnalysisStyle.sizePixels;
}

// Fill radius in CSS pixels of one row: base size, scaled by its size value, then clamped.
fn getSpatialAnalysisPointRadius(row: u32) -> f32 {
  var radius = getSpatialAnalysisPointBasePixels();
  if (spatialAnalysisStyle.useSizeValues != 0u) {
    let sizeValue = pointSizeValues[getSpatialAnalysisValueRow(row)];
    var t = clamp(sizeValue / max(spatialAnalysisStyle.sizeMaximumValue, 1e-20), 0.0, 1.0);
    if (sizeValue != sizeValue) {
      t = 0.0;
    }
    if (spatialAnalysisStyle.sizeScale == 0u) {
      t = sqrt(t);
    }
    radius = radius * t;
  }
  return clamp(radius, spatialAnalysisStyle.minSizePixels, spatialAnalysisStyle.maxSizePixels);
}

// Width in CSS pixels of the outline drawn outside the fill. A ring is all stroke: its stroke
// (outlineWidthPixels, 1.5 by default) lies inside the radius, so it adds no outline.
fn getSpatialAnalysisPointOutlineWidth() -> f32 {
  if (spatialAnalysisStyle.shape == 7u) {
    return 0.0;
  }
  return max(spatialAnalysisStyle.outlineWidthPixels, 0.0);
}

// Heading in radians, clockwise from north: the heading channel, or angleDegrees.
fn getSpatialAnalysisPointHeading(row: u32) -> f32 {
${
  channels
    ? `  if (spatialAnalysisStyle.channelHeading != SPATIAL_ANALYSIS_NO_GROUP) {
    let heading = getSpatialAnalysisChannel(getSpatialAnalysisValueRow(row), spatialAnalysisStyle.channelHeading);
    if (isSpatialAnalysisNanWord(bitcast<u32>(heading))) {
      return 0.0;
    }
    return heading * SPATIAL_ANALYSIS_DEGREES_TO_RADIANS;
  }
`
    : ''
}  return spatialAnalysisStyle.angleDegrees * SPATIAL_ANALYSIS_DEGREES_TO_RADIANS;
}

// Normalized distance of a point in the mark's quad from its centre: 1 on the shape's edge (for
// the polygon shapes, 1 plus the signed distance to the boundary in quad units).
fn getSpatialAnalysisShapeDistance(corner: vec2<f32>) -> f32 {
  let p = abs(corner);
  // Mirrored about the vertical axis, for the north-pointing marks.
  let q = vec2<f32>(abs(corner.x), corner.y);
  let shape = spatialAnalysisStyle.shape;
  if (shape == 1u) {
    return max(p.x, p.y);
  }
  if (shape == 2u) {
    // Flat-topped hexagon of circumradius 1.
    return max(p.y * 1.1547005, p.x + p.y * 0.5773503);
  }
  if (shape == 3u) {
    // Triangle pointing north (circumradius 1).
    return 1.0 + max(-corner.y - 0.5, max(-0.8660254 * corner.x + 0.5 * corner.y, 0.8660254 * corner.x + 0.5 * corner.y) - 0.5);
  }
  if (shape == 4u) {
    return 1.0 + 0.7071068 * (p.x + p.y - 1.0);
  }
  if (shape == 5u) {
    // Five-point star: fold the plane onto one wedge between a tip and the next inner vertex.
    let sector = 1.2566371;
    var angle = atan2(corner.x, corner.y);
    angle = angle - sector * floor(angle / sector);
    let folded = min(angle, sector - angle);
    let wedge = length(corner) * vec2<f32>(sin(folded), cos(folded));
    return 1.0 + dot(wedge, vec2<f32>(0.9367, 0.3503)) - 0.3503;
  }
  if (shape == 6u) {
    // Plus sign with arms a third of the width.
    return 1.0 + min(max(p.x - 1.0, p.y - 0.33), max(p.x - 0.33, p.y - 1.0));
  }
  if (shape == 8u) {
    // Chevron pointing north: two mirrored bars meeting at the tip.
    return 1.0 + max(dot(q, vec2<f32>(0.9157, 0.402)) - 0.402, dot(q, vec2<f32>(-0.4856, -0.8741)) - 0.2098);
  }
  if (shape == 9u) {
    // Arrow pointing north: a triangular head over a shaft.
    let head = max(dot(q, vec2<f32>(0.8131, 0.582)) - 0.582, 0.05 - corner.y);
    let shaft = max(q.x - 0.2, abs(corner.y + 0.4) - 0.45);
    return 1.0 + min(head, shaft);
  }
  // Circles and rings.
  return length(corner);
}

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
${hatch ? '  let hatched = select(0.0, 1.0, getSpatialAnalysisHatch(getSpatialAnalysisValueRow(row)));\n' : ''}  if (source.x != source.x || source.y != source.y || ${hatch ? '(color.a <= 0.0 && hatched < 0.5)' : 'color.a <= 0.0'}) {
    output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    output.color = vec4<f32>(0.0);
    output.corner = vec2<f32>(0.0);
    output.fillFraction = 1.0;
    output.totalPixels = 1.0;
${hatch ? '    output.hatched = 0.0;\n' : ''}    return output;
  }
  let corner = corners[vertexIndex];
  let fillPixels = getSpatialAnalysisPointRadius(row);
  let totalPixels = fillPixels + getSpatialAnalysisPointOutlineWidth();
  let worldPosition = getSpatialAnalysisPosition(source);
  var clipPosition = projectSpatialAnalysisPosition(worldPosition);
  if (spatialAnalysisStyle.sizeUnits == 1u) {
    // Metre-sized marks are built in world space so they foreshorten under pitch and tile exactly.
    let metersPerPixel = spatialAnalysisStyle.sizeMeters / max(getSpatialAnalysisPointBasePixels(), 1e-6);
    clipPosition = projectSpatialAnalysisPosition(worldPosition + corner * totalPixels * metersPerPixel);
  } else {
    // Default sizes keep the factor at exactly 1.
    let sizeFactor = totalPixels / max(spatialAnalysisStyle.sizePixels, 1e-6);
    clipPosition = vec4<f32>(
      clipPosition.xy + project_pixel_size_to_clipspace(corner * spatialAnalysisStyle.sizePixels) * sizeFactor,
      clipPosition.z,
      clipPosition.w
    );
  }
  // Oriented marks (chevron, arrow) rotate their shape space, not the quad.
  var shapeCorner = corner;
  if (spatialAnalysisStyle.shape >= 8u) {
    let heading = getSpatialAnalysisPointHeading(row);
    let sine = sin(heading);
    let cosine = cos(heading);
    shapeCorner = vec2<f32>(
      corner.x * cosine - corner.y * sine,
      corner.x * sine + corner.y * cosine
    );
  }
  output.position = clipPosition;
  output.corner = shapeCorner;
  output.color = color;
  if (spatialAnalysisStyle.shape == 7u) {
    // Ring: the stroke is the outer band of the radius.
    let stroke = select(1.5, spatialAnalysisStyle.outlineWidthPixels, spatialAnalysisStyle.outlineWidthPixels > 0.0);
    output.fillFraction = clamp(1.0 - stroke / max(totalPixels, 1e-6), 0.0, 1.0);
  } else {
    output.fillFraction = fillPixels / max(totalPixels, 1e-6);
  }
  output.totalPixels = totalPixels;
${hatch ? '  output.hatched = hatched;\n' : ''}  return output;
}

@fragment fn fragmentMain(input: PointVertexOutput) -> @location(0) vec4<f32> {
  if (isSpatialAnalysisCompareHidden(input.position)) { discard; }
${hatch ? '  var fillColor = input.color;\n  if (input.hatched > 0.5) { fillColor = applySpatialAnalysisHatch(fillColor, input.position); }\n' : '  let fillColor = input.color;\n'}  if (spatialAnalysisStyle.shape == 0u && spatialAnalysisStyle.outlineWidthPixels <= 0.0) {
    // Soft-edged disc (the original look).
    let radiusSquared = dot(input.corner, input.corner);
    if (radiusSquared > 1.0) { discard; }
    let coverage = 1.0 - smoothstep(0.6, 1.0, radiusSquared);
    return finishSpatialAnalysisColor(vec4<f32>(fillColor.rgb, fillColor.a * spatialAnalysisStyle.opacity * coverage * spatialAnalysisStyle.fillOpacity));
  }
  let distance = getSpatialAnalysisShapeDistance(input.corner);
  if (distance > 1.0) { discard; }
  // One CSS pixel in corner units: circles and the drawn shapes get an anti-aliased rim; squares
  // and hexagons stay hard-edged so adjacent cells tile without seams.
  let pixel = 1.0 / max(input.totalPixels, 0.5);
  var coverage = 1.0;
  if (spatialAnalysisStyle.shape == 0u || spatialAnalysisStyle.shape >= 3u) {
    coverage = 1.0 - smoothstep(1.0 - pixel, 1.0, distance);
  }
  if (spatialAnalysisStyle.shape == 7u) {
    let stroke = smoothstep(input.fillFraction - pixel * 0.5, input.fillFraction + pixel * 0.5, distance);
    return finishSpatialAnalysisColor(vec4<f32>(fillColor.rgb, fillColor.a * spatialAnalysisStyle.opacity * coverage * stroke));
  }
  var color = fillColor;
  if (spatialAnalysisStyle.outlineWidthPixels > 0.0) {
    let outline = smoothstep(input.fillFraction - pixel * 0.5, input.fillFraction + pixel * 0.5, distance);
    if (spatialAnalysisStyle.fillOpacity < 1.0) {
      // Premultiplied mix, so a transparent fill leaves no halo of its own colour.
      let fillAlpha = fillColor.a * spatialAnalysisStyle.fillOpacity;
      let alpha = mix(fillAlpha, spatialAnalysisStyle.outlineColor.a, outline);
      let premultiplied = mix(fillColor.rgb * fillAlpha, spatialAnalysisStyle.outlineColor.rgb * spatialAnalysisStyle.outlineColor.a, outline);
      color = vec4<f32>(premultiplied / max(alpha, 1e-5), alpha);
    } else {
      color = mix(fillColor, spatialAnalysisStyle.outlineColor, outline);
    }
  } else {
    color.a = color.a * spatialAnalysisStyle.fillOpacity;
  }
  return finishSpatialAnalysisColor(vec4<f32>(color.rgb, color.a * spatialAnalysisStyle.opacity * coverage));
}
`;
}

/** Shader source of {@link SpatialAnalysisSegmentLayer}: one instanced quad per segment. */
export function buildSegmentShader(features: SpatialAnalysisShaderFeatures = {}): string {
  const hatch = Boolean(features.hatch);
  const channels = Boolean(features.channels);
  const lineStyle = Boolean(features.lineStyle);
  const hatchLocation = lineStyle ? 6 : 4;
  return /* wgsl */ `
${getSpatialAnalysisStyleWgsl(features)}
@group(0) @binding(auto) var<storage, read> segmentPositions: array<vec4<f32>>;
@group(0) @binding(auto) var<storage, read> segmentIds: array<u32>;
@group(0) @binding(auto) var<storage, read> segmentWeights: array<f32>;
@group(0) @binding(auto) var<storage, read> segmentClip: array<vec2<f32>>;

struct SegmentVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) side: f32,
  @location(1) color: vec4<f32>,
  @location(2) fillFraction: f32,
  @location(3) totalPixels: f32,
${lineStyle ? '  @location(4) along: f32,\n  @location(5) segmentLength: f32,\n' : ''}${getHatchVarying(features, hatchLocation)}};

// Line width in CSS pixels: widthPixels, or widthMeters at the current zoom, then clamped.
fn getSpatialAnalysisLineWidthPixels() -> f32 {
  var width = spatialAnalysisStyle.sizePixels;
  if (spatialAnalysisStyle.sizeUnits == 1u) {
    width = project_meter_size_to_pixel(spatialAnalysisStyle.sizeMeters);
  }
  return clamp(width, spatialAnalysisStyle.minSizePixels, spatialAnalysisStyle.maxSizePixels);
}
${
  channels
    ? `
// Line width of one value row: the width channel mapped from widthDomain to widthRange (sqrt or
// linear), or the base width, then clamped.
fn getSpatialAnalysisRowLineWidthPixels(valueRow: u32) -> f32 {
  if (spatialAnalysisStyle.channelWidth == SPATIAL_ANALYSIS_NO_GROUP) {
    return getSpatialAnalysisLineWidthPixels();
  }
  let widthValue = getSpatialAnalysisChannel(valueRow, spatialAnalysisStyle.channelWidth);
  let domain = spatialAnalysisStyle.widthDomain;
  var t = clamp((widthValue - domain.x) / max(domain.y - domain.x, 1e-20), 0.0, 1.0);
  if (isSpatialAnalysisNanWord(bitcast<u32>(widthValue))) {
    t = 0.0;
  }
  if (spatialAnalysisStyle.widthScale == 0u) {
    t = sqrt(t);
  }
  let width = mix(spatialAnalysisStyle.widthRange.x, spatialAnalysisStyle.widthRange.y, t);
  return clamp(width, spatialAnalysisStyle.minSizePixels, spatialAnalysisStyle.maxSizePixels);
}
`
    : ''
}${
  lineStyle
    ? `
// How far a segment's quad extends past its endpoints, in CSS pixels: a quarter of the width by
// default (a short square cap so polylines join), none for butt caps, half the width for square
// and round caps.
fn getSpatialAnalysisCapExtension(totalWidth: f32) -> f32 {
  let cap = spatialAnalysisStyle.cap;
  if (cap == 1u) {
    return 0.0;
  }
  if (cap >= 2u) {
    return totalWidth * 0.5;
  }
  return totalWidth * 0.25;
}
`
    : ''
}
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
${hatch ? '  let hatched = select(0.0, 1.0, getSpatialAnalysisHatch(getSpatialAnalysisValueRow(row)));\n' : ''}  var clip = vec2<f32>(0.0, 1.0);
  if (spatialAnalysisStyle.useClip != 0u) {
    clip = segmentClip[row];
  }
  if (segment.x != segment.x || segment.z != segment.z || ${hatch ? '(color.a <= 0.0 && hatched < 0.5)' : 'color.a <= 0.0'} || clip.y <= clip.x) {
    output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    output.color = vec4<f32>(0.0);
    output.side = 0.0;
    output.fillFraction = 1.0;
    output.totalPixels = 1.0;
${lineStyle ? '    output.along = 0.0;\n    output.segmentLength = 0.0;\n' : ''}${hatch ? '    output.hatched = 0.0;\n' : ''}    return output;
  }
  let lineWidth = ${channels ? 'getSpatialAnalysisRowLineWidthPixels(getSpatialAnalysisValueRow(row))' : 'getSpatialAnalysisLineWidthPixels()'};
  let totalWidth = lineWidth + 2.0 * max(spatialAnalysisStyle.outlineWidthPixels, 0.0);
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
${
  lineStyle
    ? `  // Extend the ends by the cap extension so polylines join without gaps.
  let extension = getSpatialAnalysisCapExtension(totalWidth);
  let along = direction * (corner.x * 2.0 - 1.0) * extension;
  clipPosition = vec4<f32>(
    clipPosition.xy + project_pixel_size_to_clipspace(normal * corner.y * totalWidth * 0.5 + along),
    clipPosition.z,
    clipPosition.w
  );
  // The projected length is twice the pixel length (clip space spans 2) in device pixels.
  let segmentLength = directionLength * 0.5 / max(project.devicePixelRatio, 1e-3);
`
    : `  // Extend caps by half the width so polylines join without gaps.
  let along = direction * (corner.x * 2.0 - 1.0) * 0.5;
  clipPosition = vec4<f32>(
    clipPosition.xy + project_pixel_size_to_clipspace((normal * corner.y + along) * totalWidth * 0.5),
    clipPosition.z,
    clipPosition.w
  );
`
}  output.position = clipPosition;
  output.side = corner.y;
  output.color = color;
  output.fillFraction = lineWidth / max(totalWidth, 1e-6);
  output.totalPixels = totalWidth;
${lineStyle ? '  output.along = corner.x * segmentLength + (corner.x * 2.0 - 1.0) * extension;\n  output.segmentLength = segmentLength;\n' : ''}${hatch ? '  output.hatched = hatched;\n' : ''}  return output;
}

@fragment fn fragmentMain(input: SegmentVertexOutput) -> @location(0) vec4<f32> {
  if (isSpatialAnalysisCompareHidden(input.position)) { discard; }
${hatch ? '  var fillColor = input.color;\n  if (input.hatched > 0.5) { fillColor = applySpatialAnalysisHatch(fillColor, input.position); }\n' : '  let fillColor = input.color;\n'}  var distance = abs(input.side);
${
  lineStyle
    ? `  var lineCoverage = 1.0;
  let halfWidth = max(input.totalPixels * 0.5, 1e-3);
  let overshoot = max(max(-input.along, input.along - input.segmentLength), 0.0);
  if (spatialAnalysisStyle.cap == 3u && overshoot > 0.0) {
    // Round cap: distance from the end point, relative to the half width.
    distance = max(distance, length(vec2<f32>(overshoot, input.side * halfWidth)) / halfWidth);
  }
  if (spatialAnalysisStyle.dashLength > 0.0) {
    let period = spatialAnalysisStyle.dashLength + spatialAnalysisStyle.dashGap;
    // The pattern restarts at every segment and begins with a dash.
    let position = select(input.along, input.segmentLength, input.along > input.segmentLength);
    let inPeriod = max(position, 0.0) - period * floor(max(position, 0.0) / period);
    let softness = 1.0 / max(spatialAnalysisStyle.devicePixelRatio, 1e-3);
    if (input.along <= 0.0) {
      lineCoverage = 1.0;
    } else if (input.along >= input.segmentLength) {
      lineCoverage = select(0.0, 1.0, inPeriod < spatialAnalysisStyle.dashLength);
    } else {
      lineCoverage = clamp(min(inPeriod, spatialAnalysisStyle.dashLength - inPeriod) / softness + 0.5, 0.0, 1.0);
    }
  }
`
    : ''
}  if (spatialAnalysisStyle.outlineWidthPixels <= 0.0) {
    let coverage = 1.0 - smoothstep(0.55, 1.0, distance);
    return finishSpatialAnalysisColor(vec4<f32>(fillColor.rgb, fillColor.a * spatialAnalysisStyle.opacity * coverage${lineStyle ? ' * lineCoverage' : ''}));
  }
  // Cased line: the fill core, then the outline color out to the full width.
  let pixel = 2.0 / max(input.totalPixels, 0.5);
  let coverage = 1.0 - smoothstep(1.0 - pixel, 1.0, distance);
  let outline = smoothstep(input.fillFraction - pixel * 0.5, input.fillFraction + pixel * 0.5, distance);
  let color = mix(fillColor, spatialAnalysisStyle.outlineColor, outline);
  return finishSpatialAnalysisColor(vec4<f32>(color.rgb, color.a * spatialAnalysisStyle.opacity * coverage${lineStyle ? ' * lineCoverage' : ''}));
}
`;
}

/** Shader source of {@link SpatialAnalysisPolygonLayer}: one triangle vertex per row. */
export function buildPolygonShader(features: SpatialAnalysisShaderFeatures = {}): string {
  const hatch = Boolean(features.hatch);
  return /* wgsl */ `
${getSpatialAnalysisStyleWgsl(features)}
@group(0) @binding(auto) var<storage, read> polygonVertices: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> polygonFeatures: array<u32>;

struct PolygonVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
${getHatchVarying(features, 1)}};

@vertex fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> PolygonVertexOutput {
  var output: PolygonVertexOutput;
  let source = polygonVertices[vertexIndex];
  let valueRow = getSpatialAnalysisValueRow(polygonFeatures[vertexIndex]);
  let color = getSpatialAnalysisColor(valueRow);
${hatch ? '  let hatched = select(0.0, 1.0, getSpatialAnalysisHatch(valueRow));\n' : ''}  if (source.x != source.x || source.y != source.y || ${hatch ? '(color.a <= 0.0 && hatched < 0.5)' : 'color.a <= 0.0'}) {
    output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    output.color = vec4<f32>(0.0);
${hatch ? '    output.hatched = 0.0;\n' : ''}    return output;
  }
  output.position = projectSpatialAnalysisPosition(getSpatialAnalysisPosition(source));
  output.color = color;
${hatch ? '  output.hatched = hatched;\n' : ''}  return output;
}

@fragment fn fragmentMain(input: PolygonVertexOutput) -> @location(0) vec4<f32> {
  if (isSpatialAnalysisCompareHidden(input.position)) { discard; }
${
  hatch
    ? `  var fillColor = input.color;
  if (input.hatched > 0.5) { fillColor = applySpatialAnalysisHatch(fillColor, input.position); }
  return finishSpatialAnalysisColor(vec4<f32>(fillColor.rgb, fillColor.a * spatialAnalysisStyle.opacity));
`
    : `  return finishSpatialAnalysisColor(vec4<f32>(input.color.rgb, input.color.a * spatialAnalysisStyle.opacity));
`
}}
`;
}

/**
 * Fraction of a cell edge pixel that lies on a class boundary, for `outlineClasses`: the fragment
 * is within half the outline width (device pixels) of an edge its cell shares with a neighbour of a
 * different class or category. `fraction` is the position inside the cell (x east, y north),
 * `unitsPerPixel` the cell units per device pixel.
 */
const CLASS_EDGE_WGSL = /* wgsl */ `
fn getSpatialAnalysisGridGroup(column: i32, row: i32) -> u32 {
  let columns = i32(spatialAnalysisStyle.gridSize.x);
  return getSpatialAnalysisGroup(getSpatialAnalysisValueRow(u32(row * columns + column)));
}

fn getSpatialAnalysisClassEdge(
  column: i32,
  row: i32,
  fraction: vec2<f32>,
  unitsPerPixel: vec2<f32>
) -> f32 {
  let columns = i32(spatialAnalysisStyle.gridSize.x);
  let rows = i32(spatialAnalysisStyle.gridSize.y);
  let northStep = select(1, -1, spatialAnalysisStyle.rowOrder == 1u);
  let reach = spatialAnalysisStyle.outlineClassWidth * 0.5 * spatialAnalysisStyle.devicePixelRatio;
  let here = getSpatialAnalysisGridGroup(column, row);
  var edge = 0.0;
  // West, east, south, north: distance in device pixels, then the neighbouring cell.
  let distances = vec4<f32>(
    fraction.x / max(unitsPerPixel.x, 1e-9),
    (1.0 - fraction.x) / max(unitsPerPixel.x, 1e-9),
    fraction.y / max(unitsPerPixel.y, 1e-9),
    (1.0 - fraction.y) / max(unitsPerPixel.y, 1e-9)
  );
  let neighbours = array<vec2<i32>, 4>(
    vec2<i32>(column - 1, row),
    vec2<i32>(column + 1, row),
    vec2<i32>(column, row - northStep),
    vec2<i32>(column, row + northStep)
  );
  for (var side = 0u; side < 4u; side = side + 1u) {
    let neighbour = neighbours[side];
    if (distances[side] < reach && neighbour.x >= 0 && neighbour.y >= 0 && neighbour.x < columns && neighbour.y < rows) {
      if (getSpatialAnalysisGridGroup(neighbour.x, neighbour.y) != here) {
        edge = max(edge, 1.0 - smoothstep(reach - 1.0, reach, distances[side]));
      }
    }
  }
  return edge;
}
`;

/**
 * Raster shader source for a quad drawn as `subdivisions x subdivisions` sub-quads.
 *
 * Deck projects each vertex exactly, but the varying `worldPosition` (local meters) is
 * interpolated linearly in screen space. Web Mercator's scale changes with latitude, so one quad
 * spanning hundreds of kilometers samples cells up to ~40 km away from where the cells really are
 * (1,800 km extent). Projecting every sub-quad corner exactly keeps the interpolation error small
 * (it falls with the square of the sub-quad size).
 */
export function buildRasterShader(
  subdivisions: number,
  features: SpatialAnalysisShaderFeatures = {}
): string {
  const count = Math.max(1, Math.floor(subdivisions));
  const hatch = Boolean(features.hatch);
  const classOutline = Boolean(features.classOutline);
  const source = /* wgsl */ `
${getSpatialAnalysisStyleWgsl(features)}
${GPU_POINT_DENSITY_HEXAGON_WGSL}
@group(0) @binding(auto) var<storage, read> rasterBounds: array<f32>;
@group(0) @binding(auto) var<storage, read> rasterHexagonRadius: array<f32>;
${classOutline ? CLASS_EDGE_WGSL : ''}
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
  let subdivisions = __SUBDIVISIONS__u;
  let quad = vertexIndex / 6u;
  let unitPosition = (corners[vertexIndex % 6u] + vec2<f32>(f32(quad % subdivisions), f32(quad / subdivisions))) / f32(subdivisions);
  let radius = rasterHexagonRadius[0];
  if (spatialAnalysisStyle.binning == 1u) {
    // Hexagon lattices extend half a cell beyond the cell-center bounds.
    extent = vec2<f32>(
      f32(spatialAnalysisStyle.gridSize.x) * 1.7320508 * radius,
      f32(spatialAnalysisStyle.gridSize.y) * 1.5 * radius
    );
  }
  let margin = select(vec2<f32>(0.0), vec2<f32>(radius), spatialAnalysisStyle.binning == 1u);
  let worldPosition = minimum - margin + unitPosition * (extent + margin * 2.0);
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
${
  classOutline
    ? `  // Derivatives first, in uniform control flow: cell units per device pixel.
  let gridCoordinate = (input.worldPosition - minimum) / max(maximum - minimum, vec2<f32>(1e-20)) * vec2<f32>(f32(columns), f32(rows));
  let gridWidth = fwidth(gridCoordinate);
`
    : ''
}  if (isSpatialAnalysisCompareHidden(input.position)) {
    discard;
  }
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
  let valueRow = getSpatialAnalysisValueRow(u32(row * columns + column));
  var color = getSpatialAnalysisColor(valueRow);
${
  classOutline
    ? `  if (spatialAnalysisStyle.binning == 0u && spatialAnalysisStyle.outlineClassWidth > 0.0) {
    let edge = getSpatialAnalysisClassEdge(column, row, fract(gridCoordinate), gridWidth) * spatialAnalysisStyle.outlineClassColor.a;
    let alpha = edge + color.a * (1.0 - edge);
    if (edge > 0.0 && alpha > 0.0) {
      color = vec4<f32>((spatialAnalysisStyle.outlineClassColor.rgb * edge + color.rgb * color.a * (1.0 - edge)) / alpha, alpha);
    }
  }
`
    : ''
}${
  hatch
    ? `  let hatched = getSpatialAnalysisHatch(valueRow);
  if (color.a <= 0.0 && !hatched) {
    discard;
  }
  if (hatched) {
    color = applySpatialAnalysisHatch(color, input.position);
  }
  if (color.a <= 0.0) {
    discard;
  }
`
    : `  if (color.a <= 0.0) {
    discard;
  }
`
}  return finishSpatialAnalysisColor(vec4<f32>(color.rgb, color.a * spatialAnalysisStyle.opacity));
}
`;
  return source.replaceAll('__SUBDIVISIONS__', String(count));
}
