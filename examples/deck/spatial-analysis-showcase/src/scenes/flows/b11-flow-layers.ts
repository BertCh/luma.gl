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
import {COLORMAP_INDEXES, getRampWgsl, type RampName} from '../../engine/ramps';

/**
 * Layers of the flows chapter. They read contributor outputs straight from GPU storage buffers:
 * `FlowArcLayer` draws the top-K flows of `GPUFlowAggregation` (or the superedges of
 * `GPUNetworkCoarsening`) as curved arcs, `BundledPathLayer` draws the `GPUEdgeBundling` polylines
 * as ribbons, and `SizedDiscLayer` draws one disc per group with a per-row radius.
 */

/** Segments each arc is tessellated into. */
export const ARC_SEGMENTS = 24;

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

type Color = readonly [number, number, number, number];

function createPlaceholder(device: Device, id: string): Buffer {
  return device.createBuffer({
    id: `${id}-placeholder`,
    byteLength: 16,
    usage: Buffer.STORAGE | Buffer.COPY_DST
  });
}

function writeColor(floats: Float32Array, offset: number, color: Color): void {
  for (let channel = 0; channel < 4; channel++) floats[offset + channel] = color[channel] / 255;
}

// ---------------------------------------------------------------------------------------------
// Flow arcs
// ---------------------------------------------------------------------------------------------

const FLOW_ARC_STYLE_BYTES = 96;

const FLOW_ARC_SHADER = /* wgsl */ `
struct FlowArcStyle {
  originColor: vec4<f32>,
  destinationColor: vec4<f32>,
  bounds: vec4<f32>,
  gridSize: vec2<u32>,
  zoneKind: u32,
  hexagonRadius: f32,
  widthMinPixels: f32,
  widthMaxPixels: f32,
  bulge: f32,
  opacity: f32,
  limit: u32,
  weightFormat: u32,
  maximumWeight: f32,
  heaviestLast: u32,
};

const ARC_SEGMENTS: u32 = ${ARC_SEGMENTS}u;
const NO_ZONE: u32 = 0xffffffffu;
${GPU_POINT_DENSITY_HEXAGON_WGSL}

@group(0) @binding(auto) var<uniform> flowStyle: FlowArcStyle;
@group(0) @binding(auto) var<storage, read> flowOriginZoneIds: array<u32>;
@group(0) @binding(auto) var<storage, read> flowDestinationZoneIds: array<u32>;
@group(0) @binding(auto) var<storage, read> flowWeights: array<u32>;
@group(0) @binding(auto) var<storage, read> flowCount: array<u32>;
@group(0) @binding(auto) var<storage, read> zoneCenters: array<vec2<f32>>;

struct FlowVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) side: f32,
  @location(1) color: vec4<f32>,
};

fn getFlowWeight(row: u32) -> f32 {
  let raw = flowWeights[row];
  return select(bitcast<f32>(raw), f32(raw), flowStyle.weightFormat == 1u);
}

// Planar center of a zone: caller center, grid cell center or odd-r hexagon center.
fn getFlowZoneCenter(zone: u32) -> vec2<f32> {
  if (flowStyle.zoneKind == 0u) {
    return zoneCenters[zone];
  }
  let columns = flowStyle.gridSize.x;
  let column = i32(zone % columns);
  let row = i32(zone / columns);
  if (flowStyle.zoneKind == 1u) {
    return getPointDensityHexagonCenter(
      column, row, flowStyle.bounds.x, flowStyle.bounds.y, flowStyle.hexagonRadius
    );
  }
  let cellSize = (flowStyle.bounds.zw - flowStyle.bounds.xy) / vec2<f32>(flowStyle.gridSize);
  return flowStyle.bounds.xy + (vec2<f32>(f32(column), f32(row)) + 0.5) * cellSize;
}

// Quadratic Bezier whose control point sits to the right of the travel direction, so a flow and
// its reverse bulge to opposite sides and direction reads as a clockwise sweep.
fn getFlowArcPoint(a: vec2<f32>, b: vec2<f32>, t: f32) -> vec2<f32> {
  let delta = b - a;
  let control = (a + b) * 0.5 + vec2<f32>(delta.y, -delta.x) * flowStyle.bulge;
  let ab = mix(a, control, t);
  let cb = mix(control, b, t);
  return mix(ab, cb, t);
}

fn projectFlowPosition(position: vec2<f32>) -> vec4<f32> {
  var clipPosition = project_position_to_clipspace(vec3<f32>(position, 0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  // Deck's project32 matrices use OpenGL depth; WebGPU clip space requires [0, w].
  clipPosition.z = (clipPosition.z + clipPosition.w) * 0.5;
  return clipPosition;
}

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) drawIndex: u32
) -> FlowVertexOutput {
  var output: FlowVertexOutput;
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  output.side = 0.0;
  output.color = vec4<f32>(0.0);

  let shown = min(flowCount[0], flowStyle.limit);
  if (drawIndex >= shown) {
    return output;
  }
  // Heavy flows are the first rows of a sorted list: draw them last, on top.
  let row = select(drawIndex, shown - 1u - drawIndex, flowStyle.heaviestLast == 1u);
  let originZone = flowOriginZoneIds[row];
  let destinationZone = flowDestinationZoneIds[row];
  if (originZone == NO_ZONE || destinationZone == NO_ZONE || originZone == destinationZone) {
    return output;
  }
  let a = getFlowZoneCenter(originZone);
  let b = getFlowZoneCenter(destinationZone);

  let corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
  );
  let segment = vertexIndex / 6u;
  let corner = corners[vertexIndex % 6u];
  let t = (f32(segment) + corner.x) / f32(ARC_SEGMENTS);
  let halfStep = 0.5 / f32(ARC_SEGMENTS);
  let clipCenter = projectFlowPosition(getFlowArcPoint(a, b, t));
  let clipBefore = projectFlowPosition(getFlowArcPoint(a, b, max(t - halfStep, 0.0)));
  let clipAfter = projectFlowPosition(getFlowArcPoint(a, b, min(t + halfStep, 1.0)));
  let screenDirection =
    (clipAfter.xy / clipAfter.w - clipBefore.xy / clipBefore.w) * project.viewportSize;
  let directionLength = length(screenDirection);
  var direction = vec2<f32>(1.0, 0.0);
  if (directionLength > 1e-6) {
    direction = screenDirection / directionLength;
  }
  let normal = vec2<f32>(-direction.y, direction.x);

  var maximumWeight = flowStyle.maximumWeight;
  if (maximumWeight <= 0.0) {
    maximumWeight = getFlowWeight(0u);
  }
  let strength = sqrt(clamp(getFlowWeight(row) / max(maximumWeight, 1e-20), 0.0, 1.0));
  let widthPixels = mix(flowStyle.widthMinPixels, flowStyle.widthMaxPixels, strength);

  output.position = vec4<f32>(
    clipCenter.xy + project_pixel_size_to_clipspace(normal * corner.y * widthPixels * 0.5),
    clipCenter.z,
    clipCenter.w
  );
  output.side = corner.y;
  var color = mix(flowStyle.originColor, flowStyle.destinationColor, t);
  color.a = color.a * mix(0.3, 0.95, strength) * flowStyle.opacity;
  output.color = color;
  return output;
}

@fragment fn fragmentMain(input: FlowVertexOutput) -> @location(0) vec4<f32> {
  let coverage = 1.0 - smoothstep(0.6, 1.0, abs(input.side));
  return vec4<f32>(input.color.rgb, input.color.a * coverage);
}
`;

/** Props for {@link FlowArcLayer}. */
export type FlowArcLayerProps = LayerProps & {
  /** uint32 origin zone per flow row, `0xffffffff` for unused rows. */
  flowOriginZoneIds: Buffer;
  /** uint32 destination zone per flow row. */
  flowDestinationZoneIds: Buffer;
  /** Weight per flow row: float32 bits, or uint32 counts with `weightFormat: 'uint32'`. */
  flowWeights: Buffer;
  /** Element format of `flowWeights`. Defaults to `'float32'`. */
  weightFormat?: 'float32' | 'uint32';
  /** uint32 number of valid flow rows (contributor `output.count`). */
  flowCount: Buffer;
  /** GPU-written indirect record (`vertexCount` is `ARC_SEGMENTS * 6`); gives the instance count. */
  drawCommands?: DrawCommandBuffer | null;
  /** Instance count when there is no indirect record; rows beyond `flowCount` are still hidden. */
  instanceCount?: number;
  /**
   * Zone to position: `ids` reads `zoneCenters` (planar meters per zone), `hexagon` and `grid`
   * rebuild the lattice centers of the contributor from `bounds`, `gridSize`, `hexagonRadius`.
   */
  zoneKind: 'ids' | 'hexagon' | 'grid';
  /** Planar center per zone (`float32x2`) for `zoneKind: 'ids'`. */
  zoneCenters?: Buffer | null;
  /** `[columns, rows]` of the active lattice. */
  gridSize?: readonly [number, number];
  /** Lattice `[minX, minY, maxX, maxY]` in planar meters. */
  bounds?: readonly [number, number, number, number];
  /** Hexagon center-to-vertex radius in meters. */
  hexagonRadius?: number;
  /** Largest number of arcs drawn (the heaviest). Defaults to every row. */
  limit?: number;
  /** Weight that maps to `widthMaxPixels`; 0 or omitted uses row 0 (a weight-sorted list). */
  maximumWeight?: number;
  /** True when row 0 is the heaviest flow, so it is drawn last. Defaults to true. */
  heaviestLast?: boolean;
  /** Narrowest arc width in CSS pixels. Defaults to 1. */
  widthMinPixels?: number;
  /** Widest arc width in CSS pixels. Defaults to 8. */
  widthMaxPixels?: number;
  /** Origin end color. */
  originColor?: Color;
  /** Destination end color. */
  destinationColor?: Color;
  /** Control point offset as a fraction of the zone distance. Defaults to 0.13. */
  bulge?: number;
};

type ModelState = {model: Model; styleBuffer: Buffer; placeholder: Buffer};

/** One curved, direction-colored arc per aggregated flow, from GPU buffers and an indirect draw. */
export class FlowArcLayer extends Layer<FlowArcLayerProps> {
  static override layerName = 'FlowArcLayer';
  static override defaultProps = {
    coordinateSystem: COORDINATE_SYSTEM.METER_OFFSETS,
    parameters: BLEND_PARAMETERS
  };

  override getAttributeManager() {
    return null;
  }

  override initializeState({device}: LayerContext): void {
    if (device.type !== 'webgpu') throw new Error(`${this.id} requires WebGPU`);
    const styleBuffer = device.createBuffer({
      id: `${this.id}-style`,
      byteLength: FLOW_ARC_STYLE_BYTES,
      usage: Buffer.UNIFORM | Buffer.COPY_DST
    });
    const placeholder = createPlaceholder(device, this.id);
    const model = new Model(device, {
      ...this.getShaders({modules: [project32], source: FLOW_ARC_SHADER}),
      id: `${this.id}-model`,
      topology: 'triangle-list',
      isInstanced: true,
      vertexCount: ARC_SEGMENTS * 6,
      instanceCount: 0,
      bufferLayout: [],
      bindings: this.getBindings(styleBuffer, placeholder),
      parameters: BLEND_PARAMETERS
    });
    this.setState({model, styleBuffer, placeholder});
  }

  override updateState(parameters: UpdateParameters<this>): void {
    super.updateState(parameters);
    const {model, styleBuffer, placeholder} = this.state as ModelState;
    model.setBindings(this.getBindings(styleBuffer, placeholder));
  }

  override getModels(): Model[] {
    return [(this.state as ModelState).model];
  }

  override draw({renderPass}: {renderPass: RenderPass}): void {
    const {model, styleBuffer} = this.state as ModelState;
    this.writeStyle(styleBuffer);
    const {drawCommands, instanceCount = 0} = this.props;
    if (drawCommands) {
      // Bind pipeline and bindings with an empty draw, then replay the GPU-written record.
      model.setInstanceCount(0);
      model.draw(renderPass);
      drawCommands.draw(renderPass, 0);
    } else {
      model.setInstanceCount(instanceCount);
      model.draw(renderPass);
    }
  }

  override finalizeState(context: LayerContext): void {
    const {model, styleBuffer, placeholder} = this.state as ModelState;
    model.destroy();
    styleBuffer.destroy();
    placeholder.destroy();
    super.finalizeState(context);
  }

  private getBindings(styleBuffer: Buffer, placeholder: Buffer): Record<string, Buffer> {
    return {
      flowStyle: styleBuffer,
      flowOriginZoneIds: this.props.flowOriginZoneIds,
      flowDestinationZoneIds: this.props.flowDestinationZoneIds,
      flowWeights: this.props.flowWeights,
      flowCount: this.props.flowCount,
      zoneCenters: this.props.zoneCenters ?? placeholder
    };
  }

  private writeStyle(styleBuffer: Buffer): void {
    const props = this.props;
    const data = new ArrayBuffer(FLOW_ARC_STYLE_BYTES);
    const floats = new Float32Array(data);
    const words = new Uint32Array(data);
    writeColor(floats, 0, props.originColor ?? [255, 150, 60, 255]);
    writeColor(floats, 4, props.destinationColor ?? [60, 220, 255, 255]);
    floats.set(props.bounds ?? [0, 0, 1, 1], 8);
    words.set(props.gridSize ?? [1, 1], 12);
    words[14] = props.zoneKind === 'ids' ? 0 : props.zoneKind === 'hexagon' ? 1 : 2;
    floats[15] = props.hexagonRadius ?? 1;
    floats[16] = props.widthMinPixels ?? 1;
    floats[17] = props.widthMaxPixels ?? 8;
    floats[18] = props.bulge ?? 0.13;
    floats[19] = props.opacity ?? 1;
    words[20] = Math.max(0, Math.floor(props.limit ?? 0xffffff));
    words[21] = props.weightFormat === 'uint32' ? 1 : 0;
    floats[22] = props.maximumWeight ?? 0;
    words[23] = props.heaviestLast === false ? 0 : 1;
    styleBuffer.write(new Uint8Array(data));
  }
}

// ---------------------------------------------------------------------------------------------
// Bundled paths
// ---------------------------------------------------------------------------------------------

const PATH_STYLE_BYTES = 80;

const BUNDLED_PATH_SHADER = /* wgsl */ `
struct BundledPathStyle {
  startColor: vec4<f32>,
  endColor: vec4<f32>,
  valueRange: vec2<f32>,
  pointsPerPath: u32,
  colormap: u32,
  widthMin: f32,
  widthMax: f32,
  opacity: f32,
  useValues: u32,
  useMask: u32,
  widthByValue: u32,
  sqrtScale: u32,
  padding: u32,
};

@group(0) @binding(auto) var<uniform> pathStyle: BundledPathStyle;
@group(0) @binding(auto) var<storage, read> paths: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> edgeValues: array<f32>;
@group(0) @binding(auto) var<storage, read> edgeMask: array<u32>;

${getRampWgsl()}

struct PathVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) side: f32,
  @location(1) color: vec4<f32>,
};

fn projectPathPoint(position: vec2<f32>) -> vec4<f32> {
  var clipPosition = project_position_to_clipspace(vec3<f32>(position, 0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  // Deck's project32 matrices use OpenGL depth; WebGPU clip space requires [0, w].
  clipPosition.z = (clipPosition.z + clipPosition.w) * 0.5;
  return clipPosition;
}

@vertex fn vertexMain(
  @builtin(vertex_index) pointIndex: u32,
  @builtin(instance_index) pathIndex: u32
) -> PathVertexOutput {
  var output: PathVertexOutput;
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  output.side = 0.0;
  output.color = vec4<f32>(0.0);
  if (pathStyle.useMask != 0u && edgeMask[pathIndex] == 0u) {
    return output;
  }
  // Same row layout the contributor writes: edge e, point i at row e * pointsPerPath + i.
  let position = paths[pathIndex * pathStyle.pointsPerPath + pointIndex];
  output.position = projectPathPoint(position);
  let along = f32(pointIndex) / f32(max(pathStyle.pointsPerPath, 2u) - 1u);
  var rgba = mix(pathStyle.startColor, pathStyle.endColor, along);
  if (pathStyle.useValues != 0u) {
    let valueRange = pathStyle.valueRange;
    var tValue = clamp((edgeValues[pathIndex] - valueRange.x) / max(valueRange.y - valueRange.x, 1e-20), 0.0, 1.0);
    if (pathStyle.sqrtScale != 0u) {
      tValue = sqrt(tValue);
    }
    rgba = vec4<f32>(spatialAnalysisSampleRamp(pathStyle.colormap, tValue), pathStyle.startColor.a);
  }
  rgba.a = rgba.a * pathStyle.opacity;
  output.color = rgba;
  return output;
}

@fragment fn fragmentMain(input: PathVertexOutput) -> @location(0) vec4<f32> {
  return input.color;
}
`;

/** Props for {@link BundledPathLayer}. */
export type BundledPathLayerProps = LayerProps & {
  /** `float32x2` `[longitude, latitude]` polyline rows, edge-major (contributor `paths`). */
  paths: Buffer;
  /** Control points per path. */
  pointsPerPath: number;
  /** Number of paths (the contributor's edge count). */
  pathCount: number;
  /** Optional float32 value per path colored through `ramp`. */
  values?: Buffer | null;
  /** Value range mapped over the ramp. */
  valueRange?: readonly [number, number];
  /** Ramp of `values`. */
  ramp?: RampName;
  /** Apply a square root to the normalized value. */
  sqrtScale?: boolean;
  /** Optional uint32 per-path liveness; zero hides the path (the contributor's `edgeMask`). */
  edgeMask?: Buffer | null;
  /** Color at the first point (and everywhere when `values` is not set), RGBA 0-255. */
  startColor?: Color;
  /** Color at the last point when `values` is not set. */
  endColor?: Color;
  /** Ribbon width in CSS pixels, or the narrowest width when `widthByValue`. Defaults to 1. */
  widthPixels?: number;
  /** Widest ribbon when `widthByValue`. */
  widthMaxPixels?: number;
  /** Scale the width with the normalized value. */
  widthByValue?: boolean;
};

/** Bundled polylines drawn as 1 px line strips straight from the contributor's `paths` (width props are ignored). */
export class BundledPathLayer extends Layer<BundledPathLayerProps> {
  static override layerName = 'BundledPathLayer';
  static override defaultProps = {parameters: BLEND_PARAMETERS};

  override getAttributeManager() {
    return null;
  }

  override initializeState({device}: LayerContext): void {
    if (device.type !== 'webgpu') throw new Error(`${this.id} requires WebGPU`);
    const styleBuffer = device.createBuffer({
      id: `${this.id}-style`,
      byteLength: PATH_STYLE_BYTES,
      usage: Buffer.UNIFORM | Buffer.COPY_DST
    });
    const placeholder = createPlaceholder(device, this.id);
    const model = new Model(device, {
      ...this.getShaders({modules: [project32], source: BUNDLED_PATH_SHADER}),
      id: `${this.id}-model`,
      topology: 'line-strip',
      isInstanced: true,
      vertexCount: this.props.pointsPerPath,
      instanceCount: this.props.pathCount,
      bufferLayout: [],
      bindings: this.getBindings(styleBuffer, placeholder),
      parameters: BLEND_PARAMETERS
    });
    this.setState({model, styleBuffer, placeholder});
  }

  override updateState(parameters: UpdateParameters<this>): void {
    super.updateState(parameters);
    const {model, styleBuffer, placeholder} = this.state as ModelState;
    model.setBindings(this.getBindings(styleBuffer, placeholder));
    model.setInstanceCount(this.props.pathCount);
  }

  override getModels(): Model[] {
    return [(this.state as ModelState).model];
  }

  override draw({renderPass}: {renderPass: RenderPass}): void {
    const {model, styleBuffer} = this.state as ModelState;
    this.writeStyle(styleBuffer);
    model.draw(renderPass);
  }

  override finalizeState(context: LayerContext): void {
    const {model, styleBuffer, placeholder} = this.state as ModelState;
    model.destroy();
    styleBuffer.destroy();
    placeholder.destroy();
    super.finalizeState(context);
  }

  private getBindings(styleBuffer: Buffer, placeholder: Buffer): Record<string, Buffer> {
    return {
      pathStyle: styleBuffer,
      paths: this.props.paths,
      edgeValues: this.props.values ?? placeholder,
      edgeMask: this.props.edgeMask ?? placeholder
    };
  }

  private writeStyle(styleBuffer: Buffer): void {
    const props = this.props;
    const data = new ArrayBuffer(PATH_STYLE_BYTES);
    const floats = new Float32Array(data);
    const words = new Uint32Array(data);
    writeColor(floats, 0, props.startColor ?? [255, 150, 60, 255]);
    writeColor(floats, 4, props.endColor ?? [60, 220, 255, 255]);
    floats.set(props.valueRange ?? [0, 1], 8);
    words[10] = props.pointsPerPath;
    words[11] = COLORMAP_INDEXES[props.ramp ?? 'viridis'];
    floats[12] = props.widthPixels ?? 1;
    floats[13] = props.widthMaxPixels ?? props.widthPixels ?? 1;
    floats[14] = props.opacity ?? 1;
    words[15] = props.values ? 1 : 0;
    words[16] = props.edgeMask ? 1 : 0;
    words[17] = props.widthByValue ? 1 : 0;
    words[18] = props.sqrtScale ? 1 : 0;
    styleBuffer.write(new Uint8Array(data));
  }
}

// ---------------------------------------------------------------------------------------------
// Sized discs
// ---------------------------------------------------------------------------------------------

const DISC_STYLE_BYTES = 160;

const SIZED_DISC_SHADER = /* wgsl */ `
struct DiscStyle {
  palette: array<vec4<f32>, 8>,
  minRadius: f32,
  maxRadius: f32,
  maximumValue: f32,
  opacity: f32,
  valueFormat: u32,
  paletteSize: u32,
  useColors: u32,
  padding: u32,
};

@group(0) @binding(auto) var<uniform> discStyle: DiscStyle;
@group(0) @binding(auto) var<storage, read> discPositions: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> discValues: array<u32>;
@group(0) @binding(auto) var<storage, read> discColors: array<u32>;

struct DiscVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) corner: vec2<f32>,
  @location(1) color: vec4<f32>,
};

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) row: u32
) -> DiscVertexOutput {
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
  );
  var output: DiscVertexOutput;
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  output.corner = vec2<f32>(0.0);
  output.color = vec4<f32>(0.0);
  let raw = discValues[row];
  let value = select(bitcast<f32>(raw), f32(raw), discStyle.valueFormat == 1u);
  let source = discPositions[row];
  if (!(value > 0.0) || source.x != source.x || source.y != source.y) {
    return output;
  }
  let strength = sqrt(clamp(value / max(discStyle.maximumValue, 1e-20), 0.0, 1.0));
  let radius = mix(discStyle.minRadius, discStyle.maxRadius, strength);
  var rgba = vec4<f32>(0.9, 0.9, 0.9, 0.9);
  if (discStyle.useColors != 0u) {
    rgba = discStyle.palette[discColors[row] % max(discStyle.paletteSize, 1u)];
  }
  var clipPosition = project_position_to_clipspace(vec3<f32>(source, 0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  clipPosition.z = (clipPosition.z + clipPosition.w) * 0.5;
  let corner = corners[vertexIndex];
  output.position = vec4<f32>(
    clipPosition.xy + project_pixel_size_to_clipspace(corner * radius),
    clipPosition.z,
    clipPosition.w
  );
  output.corner = corner;
  output.color = rgba;
  return output;
}

@fragment fn fragmentMain(input: DiscVertexOutput) -> @location(0) vec4<f32> {
  let radiusSquared = dot(input.corner, input.corner);
  if (radiusSquared > 1.0) { discard; }
  let edge = smoothstep(0.72, 0.86, radiusSquared);
  let coverage = 1.0 - smoothstep(0.92, 1.0, radiusSquared);
  let rgb = mix(input.color.rgb, vec3<f32>(1.0), edge * 0.85);
  return vec4<f32>(rgb, input.color.a * discStyle.opacity * coverage);
}
`;

/** Props for {@link SizedDiscLayer}. */
export type SizedDiscLayerProps = LayerProps & {
  /** `float32x2` position per row (longitude and latitude). */
  positions: Buffer;
  /** Number of rows. */
  rowCount: number;
  /** Per-row size value; zero rows are hidden. */
  values: Buffer;
  /** Element format of `values`. Defaults to `'uint32'`. */
  valueFormat?: 'uint32' | 'float32';
  /** Value that maps to `maxRadiusPixels`; radius follows its square root. */
  maximumValue: number;
  /** Optional uint32 palette index per row. */
  colorIndices?: Buffer | null;
  /** Up to 8 palette colors. */
  palette?: readonly Color[];
  /** Smallest radius in CSS pixels. Defaults to 4. */
  minRadiusPixels?: number;
  /** Largest radius in CSS pixels. Defaults to 36. */
  maxRadiusPixels?: number;
};

/** One disc per row with a square-root radius, for supernodes of a coarsened network. */
export class SizedDiscLayer extends Layer<SizedDiscLayerProps> {
  static override layerName = 'SizedDiscLayer';
  static override defaultProps = {parameters: BLEND_PARAMETERS};

  override getAttributeManager() {
    return null;
  }

  override initializeState({device}: LayerContext): void {
    if (device.type !== 'webgpu') throw new Error(`${this.id} requires WebGPU`);
    const styleBuffer = device.createBuffer({
      id: `${this.id}-style`,
      byteLength: DISC_STYLE_BYTES,
      usage: Buffer.UNIFORM | Buffer.COPY_DST
    });
    const placeholder = createPlaceholder(device, this.id);
    const model = new Model(device, {
      ...this.getShaders({modules: [project32], source: SIZED_DISC_SHADER}),
      id: `${this.id}-model`,
      topology: 'triangle-list',
      isInstanced: true,
      vertexCount: 6,
      instanceCount: this.props.rowCount,
      bufferLayout: [],
      bindings: this.getBindings(styleBuffer, placeholder),
      parameters: BLEND_PARAMETERS
    });
    this.setState({model, styleBuffer, placeholder});
  }

  override updateState(parameters: UpdateParameters<this>): void {
    super.updateState(parameters);
    const {model, styleBuffer, placeholder} = this.state as ModelState;
    model.setBindings(this.getBindings(styleBuffer, placeholder));
    model.setInstanceCount(this.props.rowCount);
  }

  override getModels(): Model[] {
    return [(this.state as ModelState).model];
  }

  override draw({renderPass}: {renderPass: RenderPass}): void {
    const {model, styleBuffer} = this.state as ModelState;
    this.writeStyle(styleBuffer);
    model.draw(renderPass);
  }

  override finalizeState(context: LayerContext): void {
    const {model, styleBuffer, placeholder} = this.state as ModelState;
    model.destroy();
    styleBuffer.destroy();
    placeholder.destroy();
    super.finalizeState(context);
  }

  private getBindings(styleBuffer: Buffer, placeholder: Buffer): Record<string, Buffer> {
    return {
      discStyle: styleBuffer,
      discPositions: this.props.positions,
      discValues: this.props.values,
      discColors: this.props.colorIndices ?? placeholder
    };
  }

  private writeStyle(styleBuffer: Buffer): void {
    const props = this.props;
    const data = new ArrayBuffer(DISC_STYLE_BYTES);
    const floats = new Float32Array(data);
    const words = new Uint32Array(data);
    const palette = props.palette ?? [];
    for (let index = 0; index < 8; index++) {
      const color = palette[index % Math.max(palette.length, 1)];
      if (color) writeColor(floats, index * 4, color);
    }
    floats[32] = props.minRadiusPixels ?? 4;
    floats[33] = props.maxRadiusPixels ?? 36;
    floats[34] = props.maximumValue;
    floats[35] = props.opacity ?? 1;
    words[36] = props.valueFormat === 'float32' ? 0 : 1;
    words[37] = Math.min(8, palette.length);
    words[38] = props.colorIndices ? 1 : 0;
    styleBuffer.write(new Uint8Array(data));
  }
}
