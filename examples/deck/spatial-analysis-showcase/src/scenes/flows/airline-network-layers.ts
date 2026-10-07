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
import {getCompareState} from '../../engine/layers';
import type {PaletteColor} from '../../engine/ramps';
import {BETWEEN_GROUPS_INDEX, EGO_ROUTE_ALPHA} from './airline-network-palette';

/**
 * Route layers of the airline-network scene. Both draw straight from GPU buffers (the
 * `GPUGreatCircleArcs` path output, the `GPUEdgeBundling` paths) and colour every route from ONE
 * float per route (`categories`):
 *
 * - `NaN` hides the route (a display filter),
 * - `0` to `11` is a palette slot (the chapter's seven hues, grey other, the between-groups ink),
 * - `16 + slot` marks the route as part of a selection: with `egoActive` the marked routes are drawn
 *   at the selection alpha and every other route is multiplied by `dimAlpha`.
 *
 * Each layer draws one of two passes (`'within'` the groups, `'between'` them) so the between-group
 * routes can be drawn last, over the others. The palette carries the per-slot alpha.
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

/** Additive light: overlapping routes add their colour (a dark ground only). */
export const ADDITIVE_ROUTE_PARAMETERS = {
  ...NORMAL_PARAMETERS,
  blendColorDstFactor: 'one',
  blendAlphaDstFactor: 'one'
} as const;

/** Blend state of the route layers on a light ground. */
export const NORMAL_ROUTE_PARAMETERS = NORMAL_PARAMETERS;

const PALETTE_SLOTS = 12;
const STYLE_BYTES = 256;
const DRAW_PASS_CODES = {all: 0, within: 1, between: 2} as const;

/** Which routes a layer draws: every route, only those inside a group, or only those between groups. */
export type RouteDrawPass = keyof typeof DRAW_PASS_CODES;

const ROUTE_STYLE_WGSL = /* wgsl */ `
struct RouteStyle {
  palette: array<vec4<f32>, ${PALETTE_SLOTS}>,
  neutral: vec4<f32>,
  positionOffset: vec2<f32>,
  widthPixels: f32,
  opacity: f32,
  pathOffsetCount: u32,
  drawPass: u32,
  colorMode: u32,
  compareSide: u32,
  compareDivider: f32,
  dimAlpha: f32,
  egoActive: u32,
  betweenIndex: u32,
};

@group(0) @binding(auto) var<uniform> routeStyle: RouteStyle;

struct RouteVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) side: f32,
  @location(1) color: vec4<f32>,
};

// The colour of a route from its category float; alpha 0 means "not drawn in this layer".
fn getRouteColor(raw: f32) -> vec4<f32> {
  if (raw != raw) {
    return vec4<f32>(0.0);
  }
  let emphasised = raw >= 16.0;
  var slot = u32(max(raw, 0.0));
  if (emphasised) {
    slot = u32(max(raw - 16.0, 0.0));
  }
  let isBetween = slot == routeStyle.betweenIndex;
  if (routeStyle.drawPass == 1u && isBetween) {
    return vec4<f32>(0.0);
  }
  if (routeStyle.drawPass == 2u && !isBetween) {
    return vec4<f32>(0.0);
  }
  var color = routeStyle.palette[min(slot, ${PALETTE_SLOTS - 1}u)];
  if (routeStyle.colorMode == 1u) {
    color = routeStyle.neutral;
  }
  if (routeStyle.egoActive != 0u) {
    if (emphasised) {
      color.a = max(color.a, ${EGO_ROUTE_ALPHA.toFixed(2)});
    } else {
      color.a = color.a * routeStyle.dimAlpha;
    }
  }
  return color;
}

fn getHiddenRouteVertex() -> RouteVertexOutput {
  var output: RouteVertexOutput;
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  output.side = 0.0;
  output.color = vec4<f32>(0.0);
  return output;
}

fn projectRoutePosition(position: vec2<f32>) -> vec4<f32> {
  var clipPosition = project_position_to_clipspace(
    vec3<f32>(position + routeStyle.positionOffset, 0.0),
    vec3<f32>(0.0),
    vec3<f32>(0.0)
  );
  // Deck's project32 matrices use OpenGL depth; WebGPU clip space requires [0, w].
  clipPosition.z = (clipPosition.z + clipPosition.w) * 0.5;
  return clipPosition;
}

// Expands a segment between two projected points into a screen-space quad corner.
fn expandRouteSegment(
  startClip: vec4<f32>,
  endClip: vec4<f32>,
  vertexIndex: u32,
  color: vec4<f32>
) -> RouteVertexOutput {
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
  );
  let corner = corners[vertexIndex];
  let screenDirection = (endClip.xy / endClip.w - startClip.xy / startClip.w) * project.viewportSize;
  let directionLength = length(screenDirection);
  var direction = vec2<f32>(1.0, 0.0);
  if (directionLength > 1e-6) {
    direction = screenDirection / directionLength;
  }
  let normal = vec2<f32>(-direction.y, direction.x);
  var clipPosition = mix(startClip, endClip, corner.x);
  let along = direction * (corner.x * 2.0 - 1.0) * 0.5;
  clipPosition = vec4<f32>(
    clipPosition.xy + project_pixel_size_to_clipspace(
      (normal * corner.y + along) * routeStyle.widthPixels * 0.5
    ),
    clipPosition.z,
    clipPosition.w
  );
  var output: RouteVertexOutput;
  output.position = clipPosition;
  output.side = corner.y;
  output.color = color;
  return output;
}

@fragment fn fragmentMain(input: RouteVertexOutput) -> @location(0) vec4<f32> {
  // Swipe compare: side a shows left of the divider, side b right of it (3 = hidden entirely).
  if (routeStyle.compareSide == 3u) {
    discard;
  }
  if (routeStyle.compareSide != 0u) {
    let isLeft = input.position.x < routeStyle.compareDivider;
    if (select(isLeft, !isLeft, routeStyle.compareSide == 1u)) {
      discard;
    }
  }
  let coverage = 1.0 - smoothstep(0.55, 1.0, abs(input.side));
  return vec4<f32>(input.color.rgb, input.color.a * routeStyle.opacity * coverage);
}
`;

const ARC_SHADER = /* wgsl */ `
${ROUTE_STYLE_WGSL}
@group(0) @binding(auto) var<storage, read> pathPositions: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> pathOffsets: array<u32>;
@group(0) @binding(auto) var<storage, read> pathVertexCount: array<u32>;
@group(0) @binding(auto) var<storage, read> pathCategories: array<f32>;

// Instance i joins output vertices i and i + 1 of the same path.
@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> RouteVertexOutput {
  let vertex = instanceIndex;
  let count = pathVertexCount[0];
  if (vertex + 1u >= count) {
    return getHiddenRouteVertex();
  }
  // First offset row that is greater than the vertex: the next path start (or the end row).
  var low = 0u;
  var high = routeStyle.pathOffsetCount;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (pathOffsets[middle] <= vertex) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  let path = max(low, 1u) - 1u;
  if (low < routeStyle.pathOffsetCount && pathOffsets[low] == vertex + 1u) {
    return getHiddenRouteVertex();
  }
  let start = pathPositions[vertex];
  let end = pathPositions[vertex + 1u];
  if (start.x != start.x || end.x != end.x) {
    return getHiddenRouteVertex();
  }
  let color = getRouteColor(pathCategories[path]);
  if (color.a <= 0.0) {
    return getHiddenRouteVertex();
  }
  return expandRouteSegment(projectRoutePosition(start), projectRoutePosition(end), vertexIndex, color);
}
`;

const BUNDLE_SHADER = /* wgsl */ `
${ROUTE_STYLE_WGSL}
@group(0) @binding(auto) var<storage, read> bundlePaths: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> bundleCategories: array<f32>;

// One line strip per path; a hidden path collapses every vertex off screen.
@vertex fn vertexMain(
  @builtin(vertex_index) pointIndex: u32,
  @builtin(instance_index) pathIndex: u32
) -> RouteVertexOutput {
  let color = getRouteColor(bundleCategories[pathIndex]);
  if (color.a <= 0.0) {
    return getHiddenRouteVertex();
  }
  let position = bundlePaths[pathIndex * routeStyle.pathOffsetCount + pointIndex];
  var output: RouteVertexOutput;
  output.position = projectRoutePosition(position);
  output.side = 0.0;
  output.color = color;
  return output;
}
`;

/** Props shared by the route layers. */
export type RouteLayerStyleProps = {
  /** One float32 category per route (see the module comment); `NaN` hides the route. */
  categories: Buffer;
  /** Palette colours by slot, RGBA 0-255, alpha included (at most 12). */
  palette: readonly PaletteColor[];
  /** The single colour of `colorMode: 'neutral'`. */
  neutral: PaletteColor;
  /** `'category'` colours by slot; `'neutral'` draws every route in {@link RouteLayerStyleProps.neutral}. */
  colorMode?: 'category' | 'neutral';
  /** Which routes this layer draws. Defaults to every route. */
  drawPass?: RouteDrawPass;
  /** Line width in CSS pixels (arcs; bundles are 1 px line strips). */
  widthPixels?: number;
  /** Degrees added to every longitude and latitude (world copies: `[-360, 0]`, `[360, 0]`). */
  positionOffset?: readonly [number, number];
  /** Swipe compare: which side of the divider this layer belongs to. */
  compareSide?: 'a' | 'b';
  /** A selection is active: marked routes are bright, the rest multiplied by `dimAlpha`. */
  egoActive?: boolean;
  /** Alpha multiplier of routes outside the selection. */
  dimAlpha?: number;
};

type RouteLayerState = {model: Model; styleBuffer: Buffer};

/** Writes the shared style uniform of a route layer; `pathOffsetCount` is layer specific. */
function writeRouteStyle(
  buffer: Buffer,
  props: RouteLayerStyleProps & {opacity?: number},
  pathOffsetCount: number,
  context: LayerContext
): void {
  const data = new ArrayBuffer(STYLE_BYTES);
  const floats = new Float32Array(data);
  const words = new Uint32Array(data);
  props.palette.slice(0, PALETTE_SLOTS).forEach((color, slot) => {
    floats.set([color[0] / 255, color[1] / 255, color[2] / 255, (color[3] ?? 255) / 255], slot * 4);
  });
  const neutral = props.neutral;
  floats.set([neutral[0] / 255, neutral[1] / 255, neutral[2] / 255, neutral[3] / 255], 48);
  floats.set(props.positionOffset ?? [0, 0], 52);
  floats[54] = props.widthPixels ?? 0.75;
  floats[55] = props.opacity ?? 1;
  words[56] = pathOffsetCount;
  words[57] = DRAW_PASS_CODES[props.drawPass ?? 'all'];
  words[58] = props.colorMode === 'neutral' ? 1 : 0;
  // Swipe compare, as the engine layers do: the divider is in device pixels.
  const compare = getCompareState();
  let compareSide = 0;
  let compareDivider = 0;
  if (props.compareSide && compare) {
    if (compare.showing === 'both') {
      const viewport = context.viewport;
      let devicePixelRatio = 1;
      try {
        devicePixelRatio = context.device.getDefaultCanvasContext().cssToDeviceRatio();
      } catch {
        // No canvas context (headless device): CSS pixels are device pixels.
      }
      compareSide = props.compareSide === 'a' ? 1 : 2;
      compareDivider =
        ((viewport?.x ?? 0) + Math.min(Math.max(compare.position, 0), 1) * (viewport?.width ?? 1)) *
        devicePixelRatio;
    } else if (compare.showing !== props.compareSide) {
      compareSide = 3;
    }
  }
  words[59] = compareSide;
  floats[60] = compareDivider;
  floats[61] = props.dimAlpha ?? 1;
  words[62] = props.egoActive ? 1 : 0;
  words[63] = BETWEEN_GROUPS_INDEX;
  buffer.write(new Uint8Array(data));
}

/** Props of {@link CategoryArcLayer}. */
export type CategoryArcLayerProps = LayerProps &
  RouteLayerStyleProps & {
    /** `float32x2` `[longitude, latitude]` output vertices of `GPUGreatCircleArcs`. */
    positions: Buffer;
    /** Path start offsets (`pathOffsetCount` rows, uint32). */
    pathOffsets: Buffer;
    /** Number of rows in `pathOffsets` (path capacity + 1). */
    pathOffsetCount: number;
    /** One uint32: the number of valid vertices. */
    vertexCount: Buffer;
    /** Indirect record (`vertexCount` 6) whose instance count was copied from `vertexCount`. */
    drawCommands: DrawCommandBuffer;
  };

/**
 * Great-circle arcs drawn as screen-space quads straight from the contributor's path output, one
 * colour per route from a category column (see the module comment).
 */
export class CategoryArcLayer extends Layer<CategoryArcLayerProps> {
  static override layerName = 'AirlineNetworkCategoryArcLayer';
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
      ...this.getShaders({modules: [project32], source: ARC_SHADER}),
      id: `${this.id}-model`,
      topology: 'triangle-list',
      isInstanced: true,
      vertexCount: 6,
      instanceCount: 0,
      bufferLayout: [],
      bindings: this.getBindings(styleBuffer),
      parameters: this.props.parameters ?? NORMAL_PARAMETERS
    });
    this.setState({model, styleBuffer});
  }

  override updateState(parameters: UpdateParameters<this>): void {
    super.updateState(parameters);
    const {model, styleBuffer} = this.state as RouteLayerState;
    model.setBindings(this.getBindings(styleBuffer));
  }

  override getModels(): Model[] {
    return [(this.state as RouteLayerState).model];
  }

  override draw({renderPass}: {renderPass: RenderPass}): void {
    const {model, styleBuffer} = this.state as RouteLayerState;
    writeRouteStyle(styleBuffer, this.props, this.props.pathOffsetCount, this.context);
    // Bind the pipeline and bindings with an empty draw, then replay the GPU-written record.
    model.setInstanceCount(0);
    model.draw(renderPass);
    this.props.drawCommands.draw(renderPass, 0);
  }

  override finalizeState(context: LayerContext): void {
    const {model, styleBuffer} = this.state as RouteLayerState;
    model.destroy();
    styleBuffer.destroy();
    super.finalizeState(context);
  }

  private getBindings(styleBuffer: Buffer): Record<string, Buffer> {
    return {
      routeStyle: styleBuffer,
      pathPositions: this.props.positions,
      pathOffsets: this.props.pathOffsets,
      pathVertexCount: this.props.vertexCount,
      pathCategories: this.props.categories
    };
  }
}

/** Props of {@link CategoryPathLayer}. */
export type CategoryPathLayerProps = LayerProps &
  RouteLayerStyleProps & {
    /** `float32x2` `[longitude, latitude]` polyline rows, edge-major (the `GPUEdgeBundling` `paths`). */
    paths: Buffer;
    /** Control points per path. */
    pointsPerPath: number;
    /** Number of paths. */
    pathCount: number;
  };

/**
 * Bundled polylines drawn as 1 px line strips straight from the contributor's `paths` buffer,
 * coloured by a per-path category (for example the community of the route).
 */
export class CategoryPathLayer extends Layer<CategoryPathLayerProps> {
  static override layerName = 'AirlineNetworkCategoryPathLayer';
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
      ...this.getShaders({modules: [project32], source: BUNDLE_SHADER}),
      id: `${this.id}-model`,
      topology: 'line-strip',
      isInstanced: true,
      vertexCount: this.props.pointsPerPath,
      instanceCount: this.props.pathCount,
      bufferLayout: [],
      bindings: this.getBindings(styleBuffer),
      parameters: this.props.parameters ?? NORMAL_PARAMETERS
    });
    this.setState({model, styleBuffer});
  }

  override updateState(parameters: UpdateParameters<this>): void {
    super.updateState(parameters);
    const {model, styleBuffer} = this.state as RouteLayerState;
    model.setBindings(this.getBindings(styleBuffer));
    model.setInstanceCount(this.props.pathCount);
  }

  override getModels(): Model[] {
    return [(this.state as RouteLayerState).model];
  }

  override draw({renderPass}: {renderPass: RenderPass}): void {
    const {model, styleBuffer} = this.state as RouteLayerState;
    // The bundle shader reuses `pathOffsetCount` as the control points per path.
    writeRouteStyle(styleBuffer, this.props, this.props.pointsPerPath, this.context);
    model.draw(renderPass);
  }

  override finalizeState(context: LayerContext): void {
    const {model, styleBuffer} = this.state as RouteLayerState;
    model.destroy();
    styleBuffer.destroy();
    super.finalizeState(context);
  }

  private getBindings(styleBuffer: Buffer): Record<string, Buffer> {
    return {
      routeStyle: styleBuffer,
      bundlePaths: this.props.paths,
      bundleCategories: this.props.categories
    };
  }
}
