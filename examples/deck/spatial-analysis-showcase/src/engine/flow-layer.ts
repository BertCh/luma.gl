// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LayerContext} from '@deck.gl/core';
import {Buffer} from '@luma.gl/core';
import {evaluateZoomStops, type ZoomStops} from '../cartography/zoom';
import {getSortedOrder} from './draw-order';
import {
  SpatialAnalysisBaseLayer,
  getSpatialAnalysisStyleWgsl,
  writeStyle,
  type SpatialAnalysisColor,
  type SpatialAnalysisCommonLayerProps,
  type SpatialAnalysisShaderFeatures
} from './layers';

/**
 * Origin-destination flows (SYNTHESIS G13, section 1.3 "Flows"): every row is one curved arrow
 * drawn from GPU buffers, expanded entirely in the vertex shader (no CPU tessellation).
 *
 * - **Shape.** A quadratic Bezier whose control point sits `curvature` (default 0.15) times the
 *   flow length to the left of the travel direction, so A to B and B to A bend to opposite sides
 *   and separate. The body is 24 segments; the end is trimmed by `endOffsetPixels` (the node
 *   radius) and finished with an arrowhead of length `max(2.2 w, 7 px)`. The parameter is spaced
 *   by chord length, so trims and arrowheads are within about 2 percent of their pixel size.
 * - **Width.** `0.5 + k * sqrt(w / maxValue)` pixels, clamped to 0.5-12 px (`k` from
 *   `maxWidthPixels`), where `w` is the row weight and `maxValue` is ONE number for the whole
 *   dataset, so toggles between flow sets stay comparable. The weight is the `width` channel of
 *   `instanceChannels` when `channels.width` is set, otherwise the numeric `values` (scaled by
 *   `valueScale`). Rows with neither draw at `widthPixels`.
 * - **Colour.** The shared style: ramps, classes, categories, per-row alpha. A 0.8 px halo in the
 *   ground colour (`outlineColor`) separates flows from each other and from the ground; it is off
 *   on additive blending unless asked for.
 * - **Order.** Rows draw in order (or through `ids`): draw the heaviest last, for example
 *   `ids: resources.createBuffer('flow-order', getFlowDrawOrder(weights))`.
 *
 * Storage bindings per stage: 5 with default props (3 style, endpoints, ids); `instanceChannels`
 * adds one and separate `origins` and `destinations` add one: at most 7 (the ceiling is 8). One
 * extra uniform buffer carries the flow parameters.
 *
 * Feeding from chapter data: build one `float32x4` buffer of `x0, y0, x1, y1` planar metres per
 * flow (the segment layout), or two `float32x2` buffers, and one `float32` weight per flow.
 */

/** Props of {@link SpatialAnalysisFlowLayer}. */
export type SpatialAnalysisFlowLayerProps = SpatialAnalysisCommonLayerProps & {
  /** `float32x4` rows `x0, y0, x1, y1` (origin then destination) in planar metres. */
  flows?: Buffer | null;
  /** `float32x2` origins, one per flow; use with `destinations` instead of `flows`. */
  origins?: Buffer | null;
  /** `float32x2` destinations, one per flow. */
  destinations?: Buffer | null;
  /** Number of flows to draw (the layer's `instanceCount`). */
  instanceCount?: number;
  /**
   * The weight drawn at the full `maxWidthPixels`: ONE value for the whole dataset (for example
   * the largest flow of any toggled state), so widths compare across toggles. Defaults to 1.
   */
  maxValue?: number;
  /** Width in CSS pixels of a flow with weight `maxValue`; 0.5-12. Defaults to 8. */
  maxWidthPixels?: number;
  /** Smallest drawn width in CSS pixels. Defaults to 0.5. */
  minWidthPixels?: number;
  /** Width in CSS pixels of rows that have no weight, or by zoom. Defaults to 2. */
  widthPixels?: ZoomStops;
  /**
   * Take the weight from the numeric `values`. Defaults to `true` unless `colormap` is
   * `category`, `mask` or `rgba` (those values are not weights; use `channels.width`).
   */
  widthFromValues?: boolean;
  /**
   * Bend as a fraction of the flow length; the sign picks the side (positive: left of travel).
   * Defaults to 0.15; 0 draws straight arrows.
   */
  curvature?: number;
  /** Trim at the destination, in CSS pixels (the destination node radius). Defaults to 0. */
  endOffsetPixels?: number;
  /** Trim at the origin, in CSS pixels (the origin node radius). Defaults to 0. */
  startOffsetPixels?: number;
  /** Draw arrowheads. Defaults to `true`. */
  arrowheads?: boolean;
  /** Halo colour: the ground colour. Defaults to the light paper at 0.92. */
  outlineColor?: SpatialAnalysisColor;
  /**
   * Halo width in CSS pixels on each side, or by zoom. Defaults to 0.8, and to 0 with
   * `blending: 'additive'` (a halo would erase the sum).
   */
  outlineWidthPixels?: ZoomStops;
};

/**
 * Draw order for flows: row indices ascending by weight, so the heaviest flow draws last (on top).
 * Upload the result as the layer's `ids`.
 */
export function getFlowDrawOrder(weights: ArrayLike<number>): Uint32Array {
  return getSortedOrder(weights, 'ascending');
}

const FLOW_SEGMENTS = 24;
const FLOW_BODY_VERTICES = FLOW_SEGMENTS * 6;
const FLOW_VERTEX_COUNT = FLOW_BODY_VERTICES + 3;
const DEFAULT_FLOW_HALO_COLOR: SpatialAnalysisColor = [243, 239, 230, 235];
const FLOW_ARROW_LENGTH_FACTOR = 2.2;
const FLOW_ARROW_MINIMUM_PIXELS = 7;
const FLOW_ARROW_WIDTH_RATIO = 0.45;
const FLOW_MAXIMUM_WIDTH_PIXELS = 12;

/** Shader source of {@link SpatialAnalysisFlowLayer}. */
export function buildFlowShader(features: SpatialAnalysisShaderFeatures, split: boolean): string {
  const channels = Boolean(features.channels);
  return /* wgsl */ `
${getSpatialAnalysisStyleWgsl(features)}
${
  split
    ? `@group(0) @binding(auto) var<storage, read> flowOrigins: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> flowDestinations: array<vec2<f32>>;`
    : '@group(0) @binding(auto) var<storage, read> flowEndpoints: array<vec4<f32>>;'
}
@group(0) @binding(auto) var<storage, read> flowIds: array<u32>;
// [0] maxValue, maxWidth, minWidth, curvature; [1] startOffset, endOffset, arrowMinLength,
// arrowLengthFactor; [2] arrowWidthRatio (0 = no arrowhead), baseWidth, halo, useWeight; [3] haloColor.
@group(0) @binding(auto) var<uniform> flowParams: array<vec4<f32>, 4>;

struct FlowVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
  @location(1) local: vec2<f32>,
  @location(2) @interpolate(flat) shape: vec4<f32>,
};

const FLOW_SEGMENTS = ${FLOW_SEGMENTS}u;
const FLOW_BODY_VERTICES = ${FLOW_BODY_VERTICES}u;

// Width in CSS pixels: 0.5 + k * sqrt(weight / maxValue) clamped, or the base width without weight.
fn getFlowWidthPixels(valueRow: u32) -> f32 {
  let limits = flowParams[0];
  let options = flowParams[2];
  var weight = 0.0;
  var hasWeight = false;
${
  channels
    ? `  if (spatialAnalysisStyle.channelWidth != SPATIAL_ANALYSIS_NO_GROUP) {
    weight = getSpatialAnalysisChannel(valueRow, spatialAnalysisStyle.channelWidth);
    hasWeight = true;
  }
`
    : ''
}  if (!hasWeight && options.w > 0.5 && spatialAnalysisStyle.valueFormat != 0u && spatialAnalysisStyle.colormap != SPATIAL_ANALYSIS_RGBA_COLORMAP) {
    let raw = styleValues[valueRow];
    weight = select(f32(raw), bitcast<f32>(raw), spatialAnalysisStyle.valueFormat == 2u) * spatialAnalysisStyle.valueScale;
    hasWeight = true;
  }
  var width = options.y;
  if (hasWeight) {
    var t = clamp(weight / max(limits.x, 1e-20), 0.0, 1.0);
    if (isSpatialAnalysisNanWord(bitcast<u32>(weight))) {
      t = 0.0;
    }
    if (spatialAnalysisStyle.widthScale == 0u) {
      t = sqrt(t);
    }
    width = 0.5 + (limits.y - 0.5) * t;
  }
  return clamp(width, limits.z, ${FLOW_MAXIMUM_WIDTH_PIXELS.toFixed(1)});
}

fn getFlowPoint(start: vec2<f32>, control: vec2<f32>, end: vec2<f32>, t: f32) -> vec2<f32> {
  let u = 1.0 - t;
  return u * u * start + 2.0 * u * t * control + t * t * end;
}

// Clip-space position of a point of the curve.
fn getFlowClip(start: vec2<f32>, control: vec2<f32>, end: vec2<f32>, t: f32) -> vec4<f32> {
  return projectSpatialAnalysisPosition(getFlowPoint(start, control, end, t));
}

// CSS-pixel screen position of a clip-space position (y up).
fn getFlowScreen(clip: vec4<f32>) -> vec2<f32> {
  return clip.xy / clip.w * project.viewportSize * 0.5 / max(project.devicePixelRatio, 1e-3);
}

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> FlowVertexOutput {
  var output: FlowVertexOutput;
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  output.color = vec4<f32>(0.0);
  output.local = vec2<f32>(0.0);
  output.shape = vec4<f32>(0.0);
  var row = instanceIndex;
  if (spatialAnalysisStyle.useIds != 0u) {
    row = flowIds[instanceIndex];
  }
${
  split
    ? '  let ends = vec4<f32>(flowOrigins[row], flowDestinations[row]);'
    : '  let ends = flowEndpoints[row];'
}
  let valueRow = getSpatialAnalysisValueRow(row);
  let color = getSpatialAnalysisColor(valueRow);
  if (ends.x != ends.x || ends.y != ends.y || ends.z != ends.z || ends.w != ends.w || color.a <= 0.0) {
    return output;
  }
  let limits = flowParams[0];
  let offsets = flowParams[1];
  let options = flowParams[2];
  let start = getSpatialAnalysisPosition(ends.xy);
  let end = getSpatialAnalysisPosition(ends.zw);
  let chord = end - start;
  let groundLength = length(chord);
  if (groundLength <= 0.0) {
    return output;
  }
  // Bend to the left of the travel direction (positive curvature), so A to B and B to A separate.
  let perpendicular = vec2<f32>(-chord.y, chord.x) / groundLength;
  let control = (start + end) * 0.5 + perpendicular * limits.w * groundLength;
  let startScreen = getFlowScreen(projectSpatialAnalysisPosition(start));
  let endScreen = getFlowScreen(projectSpatialAnalysisPosition(end));
  let lengthPixels = length(endScreen - startScreen);
  let width = getFlowWidthPixels(valueRow);
  let halo = options.z;
  let usable = lengthPixels - offsets.x - offsets.y;
  if (usable < 2.0) {
    return output;
  }
  var arrowLength = 0.0;
  if (options.x > 0.0) {
    arrowLength = min(max(offsets.w * width, offsets.z), usable * 0.7);
  }
  let tStart = offsets.x / lengthPixels;
  let tTip = 1.0 - offsets.y / lengthPixels;
  let tBase = tTip - arrowLength / lengthPixels;

  if (vertexIndex < FLOW_BODY_VERTICES) {
    let corners = array<vec2<f32>, 6>(
      vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(0.0, 1.0),
      vec2<f32>(0.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
    );
    let corner = corners[vertexIndex % 6u];
    let s = (f32(vertexIndex / 6u) + corner.x) / f32(FLOW_SEGMENTS);
    let t = mix(tStart, tBase, s);
    var clip = getFlowClip(start, control, end, t);
    let before = getFlowScreen(getFlowClip(start, control, end, max(t - 0.01, 0.0)));
    let after = getFlowScreen(getFlowClip(start, control, end, min(t + 0.01, 1.0)));
    var direction = vec2<f32>(1.0, 0.0);
    let directionLength = length(after - before);
    if (directionLength > 1e-6) {
      direction = (after - before) / directionLength;
    }
    let normal = vec2<f32>(-direction.y, direction.x);
    let across = corner.y * (width * 0.5 + halo);
    clip = vec4<f32>(clip.xy + project_pixel_size_to_clipspace(normal * across), clip.z, clip.w);
    output.position = clip;
    output.local = vec2<f32>(0.0, across);
    output.shape = vec4<f32>(0.0, width * 0.5, halo, 0.0);
  } else {
    if (arrowLength <= 0.0) {
      return output;
    }
    let baseClip = getFlowClip(start, control, end, tBase);
    let tipScreen = getFlowScreen(getFlowClip(start, control, end, tTip));
    let baseScreen = getFlowScreen(baseClip);
    var direction = vec2<f32>(1.0, 0.0);
    let directionLength = length(tipScreen - baseScreen);
    if (directionLength > 1e-6) {
      direction = (tipScreen - baseScreen) / directionLength;
    }
    let normal = vec2<f32>(-direction.y, direction.x);
    // Arrowhead frame: u along the direction from the base, v across. The drawn triangle is
    // grown by the halo along its two sides (not behind the base, which joins the body).
    let halfWidth = max(arrowLength * options.x, width * 0.5 + 1.0);
    let sideLength = length(vec2<f32>(halfWidth, arrowLength));
    let cornerV = (halfWidth * arrowLength + halo * sideLength) / arrowLength;
    let tipU = arrowLength + halo * sideLength / halfWidth;
    var local = vec2<f32>(tipU, 0.0);
    let index = vertexIndex - FLOW_BODY_VERTICES;
    if (index == 0u) {
      local = vec2<f32>(0.0, -cornerV);
    } else if (index == 1u) {
      local = vec2<f32>(0.0, cornerV);
    }
    let offset = direction * local.x + normal * local.y;
    output.position = vec4<f32>(baseClip.xy + project_pixel_size_to_clipspace(offset), baseClip.z, baseClip.w);
    output.local = local;
    output.shape = vec4<f32>(1.0, arrowLength, halfWidth, halo);
  }
  output.color = color;
  return output;
}

@fragment fn fragmentMain(input: FlowVertexOutput) -> @location(0) vec4<f32> {
  if (isSpatialAnalysisCompareHidden(input.position)) { discard; }
  // Signed distance in pixels inside the fill (positive) and the halo width around it.
  var distance = 0.0;
  var haloWidth = 0.0;
  if (input.shape.x < 0.5) {
    distance = input.shape.y - abs(input.local.y);
    haloWidth = input.shape.z;
  } else {
    let arrowLength = input.shape.y;
    let halfWidth = input.shape.z;
    let sideLength = length(vec2<f32>(halfWidth, arrowLength));
    distance = (halfWidth * arrowLength - halfWidth * input.local.x - arrowLength * abs(input.local.y)) / sideLength;
    haloWidth = input.shape.w;
  }
  let fillCoverage = clamp(distance + 0.5, 0.0, 1.0);
  let haloCoverage = clamp(distance + haloWidth + 0.5, 0.0, 1.0);
  let haloColor = flowParams[3];
  let alpha = mix(haloColor.a * haloCoverage, input.color.a, fillCoverage) * spatialAnalysisStyle.opacity;
  if (alpha <= 0.0) { discard; }
  return finishSpatialAnalysisColor(vec4<f32>(mix(haloColor.rgb, input.color.rgb, fillCoverage), alpha));
}
`;
}

/**
 * Curved origin-destination arrows with magnitude-true widths from GPU buffers. See the module
 * notes above and {@link SpatialAnalysisFlowLayerProps}.
 */
export class SpatialAnalysisFlowLayer extends SpatialAnalysisBaseLayer<SpatialAnalysisFlowLayerProps> {
  static override layerName = 'SpatialAnalysisFlowLayer';

  private paramsBuffer: Buffer | null = null;

  protected override getShaderFeatures(): SpatialAnalysisShaderFeatures {
    return {channels: Boolean(this.props.instanceChannels)};
  }

  /** True when the endpoints come as two `float32x2` buffers instead of one `float32x4`. */
  private hasSplitEndpoints(): boolean {
    return !this.props.flows && Boolean(this.props.origins && this.props.destinations);
  }

  protected getShaderSource(): string {
    return buildFlowShader(this.getShaderFeatures(), this.hasSplitEndpoints());
  }

  protected getVertexCount(): number {
    return FLOW_VERTEX_COUNT;
  }

  private getParamsBuffer(): Buffer {
    this.paramsBuffer ??= this.context.device.createBuffer({
      id: `${this.id}-flow-params`,
      byteLength: 64,
      usage: Buffer.UNIFORM | Buffer.COPY_DST
    });
    return this.paramsBuffer;
  }

  protected getBindings(placeholder: Buffer): Record<string, Buffer> {
    const {flows, origins, destinations, ids} = this.props;
    return {
      ...this.getStyleBindings(placeholder),
      flowEndpoints: flows ?? placeholder,
      flowOrigins: origins ?? placeholder,
      flowDestinations: destinations ?? placeholder,
      flowIds: ids ?? placeholder,
      flowParams: this.getParamsBuffer()
    };
  }

  protected writeLayerStyle(styleBuffer: Buffer): void {
    const {props} = this;
    const frame = this.getStyleFrame();
    const baseWidth = evaluateZoomStops(props.widthPixels ?? 2, frame.zoom);
    writeStyle(styleBuffer, props, {sizePixels: baseWidth, useIds: Boolean(props.ids), frame});

    const additive = props.blending === 'additive';
    const halo =
      props.outlineWidthPixels === undefined
        ? additive
          ? 0
          : 0.8
        : Math.max(0, evaluateZoomStops(props.outlineWidthPixels, frame.zoom));
    const haloColor = props.outlineColor ?? DEFAULT_FLOW_HALO_COLOR;
    const colormap = props.colormap ?? 'uniform';
    const fromValues =
      props.widthFromValues ??
      !(colormap === 'category' || colormap === 'mask' || colormap === 'rgba');
    const maxWidth = Math.min(FLOW_MAXIMUM_WIDTH_PIXELS, Math.max(0.5, props.maxWidthPixels ?? 8));
    const parameters = new Float32Array(16);
    parameters.set([
      Math.max(props.maxValue ?? 1, 1e-20),
      maxWidth,
      Math.max(0.5, props.minWidthPixels ?? 0.5),
      props.curvature ?? 0.15,
      Math.max(0, props.startOffsetPixels ?? 0),
      Math.max(0, props.endOffsetPixels ?? 0),
      FLOW_ARROW_MINIMUM_PIXELS,
      FLOW_ARROW_LENGTH_FACTOR,
      props.arrowheads === false ? 0 : FLOW_ARROW_WIDTH_RATIO,
      baseWidth,
      halo,
      fromValues ? 1 : 0,
      haloColor[0] / 255,
      haloColor[1] / 255,
      haloColor[2] / 255,
      halo > 0 ? (haloColor[3] ?? 255) / 255 : 0
    ]);
    this.getParamsBuffer().write(parameters);
  }

  override finalizeState(context: LayerContext): void {
    this.paramsBuffer?.destroy();
    this.paramsBuffer = null;
    super.finalizeState(context);
  }
}
