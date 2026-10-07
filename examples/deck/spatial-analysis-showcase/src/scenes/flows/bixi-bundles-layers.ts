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
import {
  getCompareState,
  SpatialAnalysisSegmentLayer,
  type SpatialAnalysisSegmentLayerProps
} from '../../engine/layers';
import type {PaletteColor} from '../../engine/ramps';

/**
 * Layers of the bixi-bundles scene.
 *
 * `BundleRibbonLayer` draws the polylines of `GPUEdgeBundling` (or, with `straight`, the straight
 * line between their pinned ends) as ribbons straight from the contributor's `paths` buffer. The
 * width follows the square root of one float per pair (rides), the colour is one palette slot per
 * pair, and the instances are drawn heaviest last so the trunks sit on top of the weak pairs.
 * `RideTrailLayer` is the shared segment layer with the age fade squared.
 */

const NORMAL_PARAMETERS = {
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

const PALETTE_SLOTS = 8;
const STYLE_BYTES = 208;
const HIGHLIGHT_DIM_ALPHA = 0.12;

const RIBBON_SHADER = /* wgsl */ `
struct BundleStyle {
  palette: array<vec4<f32>, ${PALETTE_SLOTS}>,
  flatColor: vec4<f32>,
  maximumValue: f32,
  widthMin: f32,
  widthMax: f32,
  opacity: f32,
  pointsPerPath: u32,
  pathCount: u32,
  useMask: u32,
  useFlatColor: u32,
  straight: u32,
  widthByValue: u32,
  compareSide: u32,
  heaviestLast: u32,
  compareDivider: f32,
  highlightMask: u32,
  dimAlpha: f32,
  padding: f32,
};

@group(0) @binding(auto) var<uniform> bundleStyle: BundleStyle;
@group(0) @binding(auto) var<storage, read> bundlePaths: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> bundleValues: array<f32>;
@group(0) @binding(auto) var<storage, read> bundleClasses: array<f32>;
@group(0) @binding(auto) var<storage, read> bundleMask: array<u32>;

struct BundleVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) side: f32,
  @location(1) color: vec4<f32>,
  @location(2) widths: vec2<f32>,
};

fn getHiddenBundleVertex() -> BundleVertexOutput {
  var output: BundleVertexOutput;
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  output.side = 0.0;
  output.color = vec4<f32>(0.0);
  output.widths = vec2<f32>(0.0);
  return output;
}

// Control point of a path; with straight on, the straight line between the pinned ends.
fn getBundlePoint(path: u32, point: u32) -> vec2<f32> {
  let pointsPerPath = bundleStyle.pointsPerPath;
  if (bundleStyle.straight != 0u) {
    let first = bundlePaths[path * pointsPerPath];
    let last = bundlePaths[path * pointsPerPath + pointsPerPath - 1u];
    return mix(first, last, f32(point) / f32(max(pointsPerPath, 2u) - 1u));
  }
  return bundlePaths[path * pointsPerPath + point];
}

fn projectBundlePoint(position: vec2<f32>) -> vec4<f32> {
  var clipPosition = project_position_to_clipspace(vec3<f32>(position, 0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  // Deck's project32 matrices use OpenGL depth; WebGPU clip space requires [0, w].
  clipPosition.z = (clipPosition.z + clipPosition.w) * 0.5;
  return clipPosition;
}

fn getBundleScreen(clipPosition: vec4<f32>) -> vec2<f32> {
  return clipPosition.xy / clipPosition.w * project.viewportSize;
}

fn getBundleDirection(fromPoint: vec2<f32>, toPoint: vec2<f32>, fallback: vec2<f32>) -> vec2<f32> {
  let delta = toPoint - fromPoint;
  let deltaLength = length(delta);
  if (deltaLength < 1e-4) {
    return fallback;
  }
  return delta / deltaLength;
}

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> BundleVertexOutput {
  let segmentsPerPath = max(bundleStyle.pointsPerPath, 2u) - 1u;
  let slot = instanceIndex / segmentsPerPath;
  let segment = instanceIndex % segmentsPerPath;
  // Paths are ranked heaviest first: drawing the last rank first puts the heaviest on top.
  var path = slot;
  if (bundleStyle.heaviestLast != 0u) {
    path = bundleStyle.pathCount - 1u - slot;
  }
  if (bundleStyle.useMask != 0u && bundleMask[path] == 0u) {
    return getHiddenBundleVertex();
  }
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
  );
  let corner = corners[vertexIndex];
  let lastPoint = max(bundleStyle.pointsPerPath, 2u) - 1u;
  let point = segment + u32(corner.x);
  let hasPrevious = point > 0u;
  let hasNext = point < lastPoint;

  let startPosition = getBundlePoint(path, segment);
  let endPosition = getBundlePoint(path, segment + 1u);
  if (startPosition.x != startPosition.x || endPosition.x != endPosition.x) {
    return getHiddenBundleVertex();
  }
  let startClip = projectBundlePoint(startPosition);
  let endClip = projectBundlePoint(endPosition);
  let segmentDirection = getBundleDirection(
    getBundleScreen(startClip), getBundleScreen(endClip), vec2<f32>(1.0, 0.0)
  );

  // Both quads that meet at a control point share its offset (the mean of the two directions), so
  // the ribbon has no gaps or overlaps at the joints.
  var centerClip = startClip;
  if (point != segment) {
    centerClip = endClip;
  }
  let center = getBundleScreen(centerClip);
  var incoming = segmentDirection;
  if (hasPrevious) {
    let previousClip = projectBundlePoint(getBundlePoint(path, select(point, point - 1u, hasPrevious)));
    incoming = getBundleDirection(getBundleScreen(previousClip), center, segmentDirection);
  }
  var outgoing = segmentDirection;
  if (hasNext) {
    let nextClip = projectBundlePoint(getBundlePoint(path, select(point, point + 1u, hasNext)));
    outgoing = getBundleDirection(center, getBundleScreen(nextClip), segmentDirection);
  }
  let tangent = getBundleDirection(vec2<f32>(0.0), incoming + outgoing, segmentDirection);
  let normal = vec2<f32>(-tangent.y, tangent.x);
  let miter = 1.0 / max(dot(tangent, segmentDirection), 0.5);

  let value = bundleValues[path];
  let strength = sqrt(clamp(value / max(bundleStyle.maximumValue, 1e-20), 0.0, 1.0));
  var widthPixels = bundleStyle.widthMin;
  if (bundleStyle.widthByValue != 0u) {
    widthPixels = mix(bundleStyle.widthMin, bundleStyle.widthMax, strength);
  }
  // The quad is a pixel wider than the line so the edge can fade; coverage is set in the fragment.
  let halfWidth = widthPixels * 0.5;
  let halfGeometry = halfWidth + 0.5;

  let classIndex = u32(max(bundleClasses[path], 0.0));
  var color = bundleStyle.flatColor;
  if (bundleStyle.useFlatColor == 0u) {
    color = bundleStyle.palette[min(classIndex, ${PALETTE_SLOTS - 1}u)];
  }
  if (bundleStyle.highlightMask != 0u && ((bundleStyle.highlightMask >> min(classIndex, 31u)) & 1u) == 0u) {
    color.a = color.a * bundleStyle.dimAlpha;
  }
  color.a = color.a * bundleStyle.opacity;

  var output: BundleVertexOutput;
  output.position = vec4<f32>(
    centerClip.xy + project_pixel_size_to_clipspace(normal * corner.y * halfGeometry * miter),
    centerClip.z,
    centerClip.w
  );
  output.side = corner.y;
  output.color = color;
  output.widths = vec2<f32>(halfGeometry, halfWidth);
  return output;
}

@fragment fn fragmentMain(input: BundleVertexOutput) -> @location(0) vec4<f32> {
  // Swipe compare: side a shows left of the divider, side b right of it (3 = hidden entirely).
  if (bundleStyle.compareSide == 3u) {
    discard;
  }
  if (bundleStyle.compareSide != 0u) {
    let isLeft = input.position.x < bundleStyle.compareDivider;
    if (select(isLeft, !isLeft, bundleStyle.compareSide == 1u)) {
      discard;
    }
  }
  let edgeDistance = abs(input.side) * input.widths.x;
  let coverage = clamp(input.widths.y + 0.5 - edgeDistance, 0.0, 1.0);
  return vec4<f32>(input.color.rgb, input.color.a * coverage);
}
`;

/** Props of {@link BundleRibbonLayer}. */
export type BundleRibbonLayerProps = LayerProps & {
  /** `float32x2` `[longitude, latitude]` polyline rows, edge-major (the `GPUEdgeBundling` `paths`). */
  paths: Buffer;
  /** Control points per path. */
  pointsPerPath: number;
  /** Number of paths (the contributor's edge count), ranked heaviest first. */
  pathCount: number;
  /** One float32 per path: the value the width follows (rides on the pair). */
  values: Buffer;
  /** One float32 per path: the palette slot of the path (a class index, or 0 and 1). */
  classes: Buffer;
  /** Optional uint32 liveness per path; zero hides the path (the contributor's `edgeMask`). */
  edgeMask?: Buffer | null;
  /** Palette slots, RGBA 0-255 with alpha (at most 8). */
  palette: readonly PaletteColor[];
  /** When set, every path is drawn in this one colour and the palette is ignored. */
  flatColor?: PaletteColor;
  /** The ONE value that maps to the widest ribbon. */
  maximumValue: number;
  /** Narrowest width in CSS pixels (and the width when `widthByValue` is off). */
  widthMinPixels?: number;
  /** Widest width in CSS pixels when `widthByValue` is on. */
  widthMaxPixels?: number;
  /** Width follows `0.5 + (max - min) * sqrt(value / maximumValue)`. */
  widthByValue?: boolean;
  /** Draw the straight line between the pinned ends of each path instead of the bundle. */
  straight?: boolean;
  /** Draw the lowest-ranked paths first so the heaviest are on top. Defaults to true. */
  heaviestLast?: boolean;
  /** Class indices drawn at full strength; the others are dimmed (an isolated legend class). */
  highlightClasses?: readonly number[] | null;
  /** Swipe compare: which side of the divider this layer belongs to. */
  compareSide?: 'a' | 'b';
};

type RibbonState = {model: Model; styleBuffer: Buffer};

/** Bundled polylines as width-by-value ribbons with one palette slot per path. */
export class BundleRibbonLayer extends Layer<BundleRibbonLayerProps> {
  static override layerName = 'BixiBundleRibbonLayer';
  static override defaultProps = {
    coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
    parameters: NORMAL_PARAMETERS
  };

  override getAttributeManager() {
    return null;
  }

  override initializeState({device}: LayerContext): void {
    if (device.type !== 'webgpu') throw new Error(`${this.id} requires WebGPU`);
    const styleBuffer = device.createBuffer({
      id: `${this.id}-style`,
      byteLength: STYLE_BYTES,
      usage: Buffer.UNIFORM | Buffer.COPY_DST
    });
    const model = new Model(device, {
      ...this.getShaders({modules: [project32], source: RIBBON_SHADER}),
      id: `${this.id}-model`,
      topology: 'triangle-list',
      isInstanced: true,
      vertexCount: 6,
      instanceCount: this.getInstanceCount(),
      bufferLayout: [],
      bindings: this.getBindings(styleBuffer),
      parameters: this.props.parameters ?? NORMAL_PARAMETERS
    });
    this.setState({model, styleBuffer});
  }

  override updateState(parameters: UpdateParameters<this>): void {
    super.updateState(parameters);
    const {model, styleBuffer} = this.state as RibbonState;
    model.setBindings(this.getBindings(styleBuffer));
    model.setInstanceCount(this.getInstanceCount());
  }

  override getModels(): Model[] {
    return [(this.state as RibbonState).model];
  }

  override draw({renderPass}: {renderPass: RenderPass}): void {
    const {model, styleBuffer} = this.state as RibbonState;
    this.writeStyle(styleBuffer);
    // Deck may reuse or reset model draw state between passes; make the instanced segment count
    // authoritative at the point of drawing, as the shared engine layers do.
    model.setInstanceCount(this.getInstanceCount());
    model.draw(renderPass);
  }

  override finalizeState(context: LayerContext): void {
    const {model, styleBuffer} = this.state as RibbonState;
    model.destroy();
    styleBuffer.destroy();
    super.finalizeState(context);
  }

  private getInstanceCount(): number {
    return this.props.pathCount * Math.max(this.props.pointsPerPath - 1, 1);
  }

  private getBindings(styleBuffer: Buffer): Record<string, Buffer> {
    // A path without a liveness buffer binds the values buffer, which is never read then.
    return {
      bundleStyle: styleBuffer,
      bundlePaths: this.props.paths,
      bundleValues: this.props.values,
      bundleClasses: this.props.classes,
      bundleMask: this.props.edgeMask ?? this.props.values
    };
  }

  private writeStyle(styleBuffer: Buffer): void {
    const props = this.props;
    const data = new ArrayBuffer(STYLE_BYTES);
    const floats = new Float32Array(data);
    const words = new Uint32Array(data);
    props.palette.slice(0, PALETTE_SLOTS).forEach((color, slot) => {
      floats.set([color[0] / 255, color[1] / 255, color[2] / 255, color[3] / 255], slot * 4);
    });
    const flat = props.flatColor ?? [255, 255, 255, 255];
    floats.set([flat[0] / 255, flat[1] / 255, flat[2] / 255, flat[3] / 255], 32);
    floats[36] = props.maximumValue;
    floats[37] = props.widthMinPixels ?? 1;
    floats[38] = props.widthMaxPixels ?? props.widthMinPixels ?? 1;
    floats[39] = props.opacity ?? 1;
    words[40] = props.pointsPerPath;
    words[41] = props.pathCount;
    words[42] = props.edgeMask ? 1 : 0;
    words[43] = props.flatColor ? 1 : 0;
    words[44] = props.straight ? 1 : 0;
    words[45] = props.widthByValue ? 1 : 0;
    const compare = this.getCompareUniforms();
    words[46] = compare.side;
    words[47] = props.heaviestLast === false ? 0 : 1;
    floats[48] = compare.divider;
    words[49] = (props.highlightClasses ?? []).reduce(
      (bits, index) => (index >= 0 && index < 32 ? bits | (1 << index) : bits),
      0
    );
    floats[50] = HIGHLIGHT_DIM_ALPHA;
    styleBuffer.write(new Uint8Array(data));
  }

  /** The swipe state as the engine layers read it: the divider is in device pixels. */
  private getCompareUniforms(): {side: number; divider: number} {
    const {compareSide} = this.props;
    const compare = getCompareState();
    if (!compareSide || !compare) return {side: 0, divider: 0};
    if (compare.showing !== 'both') {
      return {side: compare.showing === compareSide ? 0 : 3, divider: 0};
    }
    const viewport = this.context.viewport;
    let devicePixelRatio = 1;
    try {
      devicePixelRatio = this.context.device.getDefaultCanvasContext().cssToDeviceRatio();
    } catch {
      // No canvas context (headless device): CSS pixels are device pixels.
    }
    const divider =
      ((viewport?.x ?? 0) + Math.min(Math.max(compare.position, 0), 1) * (viewport?.width ?? 1)) *
      devicePixelRatio;
    return {side: compareSide === 'a' ? 1 : 2, divider};
  }
}

/** The weight line of the shared segment shader that {@link RideTrailLayer} patches. */
const WEIGHT_LINE = 'color.a = color.a * segmentWeights[row];';

/**
 * The trail segments of `GPUTimeWindowFilter`, drawn with age as alpha only: the contributor's
 * fade weight runs linearly from 1 at the bike to 0 at the tail of the window, and this layer
 * squares it, so a trail has alpha `(1 - age / tail)^2`. Everything else is the shared segment
 * layer: the compact ids, the clip fractions at the window edges and the indirect draw count.
 */
export class RideTrailLayer extends SpatialAnalysisSegmentLayer {
  static override layerName = 'BixiRideTrailLayer';

  constructor(props: SpatialAnalysisSegmentLayerProps) {
    super(props);
  }

  protected override getShaderSource(): string {
    const source = super.getShaderSource();
    if (!source.includes(WEIGHT_LINE)) {
      throw new Error('RideTrailLayer could not find the segment weight line');
    }
    return source.replace(
      WEIGHT_LINE,
      'color.a = color.a * segmentWeights[row] * segmentWeights[row];'
    );
  }
}
