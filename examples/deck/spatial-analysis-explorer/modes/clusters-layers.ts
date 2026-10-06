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
import {STORAGE_MODEL_STYLE_BYTE_LENGTH, StorageModelLayer} from './coverage-layers';

const STYLE_BYTE_LENGTH = 144;
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

const CENTROID_SHADER = /* wgsl */ `
struct CentroidStyle {
  palette: array<vec4<f32>, 8>,
  // basePixels, pixelsPerSqrtMember, maximumPixels, fillOpacity
  radius: vec4<f32>,
};

@group(0) @binding(auto) var<uniform> centroidStyle: CentroidStyle;
@group(0) @binding(auto) var<storage, read> clusterCentroids: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> clusterSizes: array<u32>;

struct CentroidVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) corner: vec2<f32>,
  @location(1) color: vec4<f32>,
  @location(2) ringWidth: f32,
};

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> CentroidVertexOutput {
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
  );
  var output: CentroidVertexOutput;
  let size = clusterSizes[instanceIndex];
  let centroid = clusterCentroids[instanceIndex];
  if (size == 0u || centroid.x != centroid.x || centroid.y != centroid.y) {
    output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    output.corner = vec2<f32>(0.0);
    output.color = vec4<f32>(0.0);
    output.ringWidth = 0.0;
    return output;
  }
  let radiusPixels = clamp(
    centroidStyle.radius.x + centroidStyle.radius.y * sqrt(f32(size)),
    centroidStyle.radius.x,
    centroidStyle.radius.z
  );
  let corner = corners[vertexIndex];
  var clipPosition = project_position_to_clipspace(vec3<f32>(centroid, 0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  // Deck's project32 matrices use OpenGL depth; WebGPU clip space requires [0, w].
  clipPosition.z = (clipPosition.z + clipPosition.w) * 0.5;
  clipPosition = vec4<f32>(
    clipPosition.xy + project_pixel_size_to_clipspace(corner * radiusPixels),
    clipPosition.z,
    clipPosition.w
  );
  output.position = clipPosition;
  output.corner = corner;
  output.color = centroidStyle.palette[instanceIndex % 8u];
  output.ringWidth = 2.0 / radiusPixels;
  return output;
}

@fragment fn fragmentMain(input: CentroidVertexOutput) -> @location(0) vec4<f32> {
  let distance = length(input.corner);
  if (distance > 1.0) { discard; }
  let edge = 1.0 - smoothstep(1.0 - input.ringWidth * 0.5, 1.0, distance);
  let ring = smoothstep(1.0 - input.ringWidth * 1.5, 1.0 - input.ringWidth, distance);
  let fill = vec4<f32>(input.color.rgb, centroidStyle.radius.w);
  let outline = vec4<f32>(mix(input.color.rgb, vec3<f32>(1.0), 0.75), 0.95);
  let color = mix(fill, outline, ring);
  return vec4<f32>(color.rgb, color.a * edge);
}
`;

/** Props for {@link ClusterCentroidLayer}. */
export type ClusterCentroidLayerProps = LayerProps & {
  /** `float32x2` cluster centroids in planar meters, indexed by compact cluster ID. */
  centroids: Buffer;
  /** uint32 member count per compact cluster ID. */
  sizes: Buffer;
  /** GPU-written indirect record (six vertices) whose instance count is the cluster count. */
  drawCommands: DrawCommandBuffer;
  /** Category colors, `palette[clusterId % 8]`, RGBA 0-255. */
  palette: readonly (readonly [number, number, number, number])[];
  /** Radius in CSS pixels of a one-point cluster. Defaults to 7. */
  basePixels?: number;
  /** Added radius in CSS pixels per square root of the member count. Defaults to 1.4. */
  pixelsPerSqrtMember?: number;
  /** Largest radius in CSS pixels. Defaults to 42. */
  maximumPixels?: number;
  /** Alpha of the disc fill, 0-1. Defaults to 0.3. */
  fillOpacity?: number;
};

/**
 * Draws one translucent disc with a light outline ring per cluster at its GPU-computed centroid.
 * The disc radius grows with the square root of the cluster size and the instance count comes from
 * an indirect draw record, so neither positions nor counts are read back.
 */
export class ClusterCentroidLayer extends Layer<ClusterCentroidLayerProps> {
  static override layerName = 'ClusterCentroidLayer';
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
      byteLength: STYLE_BYTE_LENGTH,
      usage: Buffer.UNIFORM | Buffer.COPY_DST
    });
    const model = new Model(device, {
      ...this.getShaders({modules: [project32], source: CENTROID_SHADER}),
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
    const style = new Float32Array(STYLE_BYTE_LENGTH / 4);
    for (let index = 0; index < 8; index++) {
      const color = this.props.palette[index % this.props.palette.length];
      style.set([color[0] / 255, color[1] / 255, color[2] / 255, color[3] / 255], index * 4);
    }
    style.set(
      [
        this.props.basePixels ?? 7,
        this.props.pixelsPerSqrtMember ?? 1.4,
        this.props.maximumPixels ?? 42,
        this.props.fillOpacity ?? 0.3
      ],
      32
    );
    styleBuffer.write(style);
    // Bind pipeline and bindings with an empty draw, then replay the GPU-written record.
    model.setInstanceCount(0);
    model.draw(renderPass);
    this.props.drawCommands.draw(renderPass, 0);
  }

  override finalizeState(context: LayerContext): void {
    const {model, styleBuffer} = this.state as {model: Model; styleBuffer: Buffer};
    model.destroy();
    styleBuffer.destroy();
    super.finalizeState(context);
  }

  private getBindings(styleBuffer: Buffer): Record<string, Buffer> {
    return {
      centroidStyle: styleBuffer,
      clusterCentroids: this.props.centroids,
      clusterSizes: this.props.sizes
    };
  }
}

// --- Group shape layers --------------------------------------------------------------------------
// Draw `GPUGroupGeometry` and `GPUGroupConvexHull` outputs straight from their storage buffers.

const SHAPE_PALETTE_SIZE = 8;

const SHAPE_PRELUDE_WGSL = /* wgsl */ `
struct ShapeStyle {
  palette: array<vec4<f32>, ${SHAPE_PALETTE_SIZE}>,
  // widthPixels, opacity, sizePixels, unused
  numbers: vec4<f32>,
  // groupCount, segments, shape (0 disc, 1 ring, 2 diamond), mode (0 centers, 1 medoid rows)
  config: vec4<u32>,
};

@group(0) @binding(auto) var<uniform> shapeStyle: ShapeStyle;

fn projectShapePosition(position: vec2<f32>) -> vec4<f32> {
  var clipPosition = project_position_to_clipspace(vec3<f32>(position, 0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  // Deck's project32 matrices use OpenGL depth; WebGPU clip space requires [0, w].
  clipPosition.z = (clipPosition.z + clipPosition.w) * 0.5;
  return clipPosition;
}

fn isFiniteVector(value: vec2<f32>) -> bool {
  return (bitcast<u32>(value.x) & 0x7f800000u) != 0x7f800000u &&
    (bitcast<u32>(value.y) & 0x7f800000u) != 0x7f800000u;
}

`;

const LINE_WGSL = /* wgsl */ `
struct LineOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
  @location(1) side: f32,
};

fn getHiddenLine() -> LineOutput {
  var output: LineOutput;
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  output.color = vec4<f32>(0.0);
  output.side = 0.0;
  return output;
}

// One screen-space quad (six vertices) along the segment from start to end.
fn getLineVertex(start: vec2<f32>, end: vec2<f32>, vertexIndex: u32, color: vec4<f32>) -> LineOutput {
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
  );
  if (!isFiniteVector(start) || !isFiniteVector(end)) { return getHiddenLine(); }
  let startClip = projectShapePosition(start);
  let endClip = projectShapePosition(end);
  let corner = corners[vertexIndex];
  let screenDirection = (endClip.xy / endClip.w - startClip.xy / startClip.w) * project.viewportSize;
  let directionLength = length(screenDirection);
  var direction = vec2<f32>(1.0, 0.0);
  if (directionLength > 1e-6) { direction = screenDirection / directionLength; }
  let normal = vec2<f32>(-direction.y, direction.x);
  var clipPosition = mix(startClip, endClip, corner.x);
  let along = direction * (corner.x * 2.0 - 1.0) * 0.5;
  clipPosition = vec4<f32>(
    clipPosition.xy + project_pixel_size_to_clipspace((normal * corner.y + along) * shapeStyle.numbers.x * 0.5),
    clipPosition.z,
    clipPosition.w
  );
  var output: LineOutput;
  output.position = clipPosition;
  output.color = vec4<f32>(color.rgb, color.a * shapeStyle.numbers.y);
  output.side = corner.y;
  return output;
}

@fragment fn fragmentMain(input: LineOutput) -> @location(0) vec4<f32> {
  let coverage = 1.0 - smoothstep(0.55, 1.0, abs(input.side));
  return vec4<f32>(input.color.rgb, input.color.a * coverage);
}
`;

const HULL_OUTLINE_SHADER = /* wgsl */ `
${SHAPE_PRELUDE_WGSL}
${LINE_WGSL}
@group(0) @binding(auto) var<storage, read> hullOffsets: array<u32>;
@group(0) @binding(auto) var<storage, read> hullCounts: array<u32>;
@group(0) @binding(auto) var<storage, read> hullPositions: array<vec2<f32>>;

// Instance = hull vertex slot. Its group is the last group whose offset is at or before the slot.
@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) slot: u32
) -> LineOutput {
  let groupCount = shapeStyle.config.x;
  if (slot >= hullOffsets[groupCount]) { return getHiddenLine(); }
  var low = 0u;
  var high = groupCount;
  while (high - low > 1u) {
    let middle = (low + high) / 2u;
    if (hullOffsets[middle] <= slot) { low = middle; } else { high = middle; }
  }
  let group = low;
  let count = hullCounts[group];
  let first = hullOffsets[group];
  if (count < 2u || slot >= first + count) { return getHiddenLine(); }
  var next = slot + 1u;
  if (next >= first + count) { next = first; }
  return getLineVertex(
    hullPositions[slot],
    hullPositions[next],
    vertexIndex,
    shapeStyle.palette[group % ${SHAPE_PALETTE_SIZE}u]
  );
}
`;

const ELLIPSE_OUTLINE_SHADER = /* wgsl */ `
${SHAPE_PRELUDE_WGSL}
${LINE_WGSL}
@group(0) @binding(auto) var<storage, read> groupCounts: array<u32>;
@group(0) @binding(auto) var<storage, read> groupCenters: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> groupEllipses: array<f32>;

fn getEllipsePoint(group: u32, step: u32) -> vec2<f32> {
  let segments = shapeStyle.config.y;
  let t = 6.283185307179586 * f32(step) / f32(segments);
  let angle = groupEllipses[group * 3u];
  let local = vec2<f32>(
    groupEllipses[group * 3u + 1u] * cos(t),
    groupEllipses[group * 3u + 2u] * sin(t)
  );
  return groupCenters[group] + vec2<f32>(
    local.x * cos(angle) - local.y * sin(angle),
    local.x * sin(angle) + local.y * cos(angle)
  );
}

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instance: u32
) -> LineOutput {
  let segments = shapeStyle.config.y;
  let group = instance / segments;
  let step = instance % segments;
  if (group >= shapeStyle.config.x || groupCounts[group] < 3u) { return getHiddenLine(); }
  return getLineVertex(
    getEllipsePoint(group, step),
    getEllipsePoint(group, step + 1u),
    vertexIndex,
    shapeStyle.palette[group % ${SHAPE_PALETTE_SIZE}u]
  );
}
`;

const BOUNDS_OUTLINE_SHADER = /* wgsl */ `
${SHAPE_PRELUDE_WGSL}
${LINE_WGSL}
@group(0) @binding(auto) var<storage, read> groupCounts: array<u32>;
@group(0) @binding(auto) var<storage, read> groupBounds: array<vec4<f32>>;

fn getBoundsCorner(bounds: vec4<f32>, corner: u32) -> vec2<f32> {
  let x = select(bounds.x, bounds.z, corner == 1u || corner == 2u);
  let y = select(bounds.y, bounds.w, corner == 2u || corner == 3u);
  return vec2<f32>(x, y);
}

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instance: u32
) -> LineOutput {
  let group = instance / 4u;
  let edge = instance % 4u;
  if (group >= shapeStyle.config.x || groupCounts[group] == 0u) { return getHiddenLine(); }
  let bounds = groupBounds[group];
  return getLineVertex(
    getBoundsCorner(bounds, edge),
    getBoundsCorner(bounds, (edge + 1u) % 4u),
    vertexIndex,
    shapeStyle.palette[group % ${SHAPE_PALETTE_SIZE}u]
  );
}
`;

const GROUP_MARKER_SHADER = /* wgsl */ `
${SHAPE_PRELUDE_WGSL}
@group(0) @binding(auto) var<storage, read> groupCounts: array<u32>;
@group(0) @binding(auto) var<storage, read> markerCenters: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> markerRows: array<u32>;
@group(0) @binding(auto) var<storage, read> markerPositions: array<vec2<f32>>;

struct MarkerOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) corner: vec2<f32>,
  @location(1) color: vec4<f32>,
};

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) group: u32
) -> MarkerOutput {
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
  );
  var output: MarkerOutput;
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  output.corner = vec2<f32>(0.0);
  output.color = vec4<f32>(0.0);
  if (group >= shapeStyle.config.x || groupCounts[group] == 0u) { return output; }
  var position = markerCenters[group];
  if (shapeStyle.config.w == 1u) {
    let row = markerRows[group];
    if (row == 0xffffffffu) { return output; }
    position = markerPositions[row];
  }
  if (!isFiniteVector(position)) { return output; }
  let corner = corners[vertexIndex];
  var clipPosition = projectShapePosition(position);
  clipPosition = vec4<f32>(
    clipPosition.xy + project_pixel_size_to_clipspace(corner * shapeStyle.numbers.z),
    clipPosition.z,
    clipPosition.w
  );
  output.position = clipPosition;
  output.corner = corner;
  output.color = vec4<f32>(shapeStyle.palette[0].rgb, shapeStyle.palette[0].a * shapeStyle.numbers.y);
  return output;
}

@fragment fn fragmentMain(input: MarkerOutput) -> @location(0) vec4<f32> {
  var inside = length(input.corner);
  if (shapeStyle.config.z == 2u) {
    inside = abs(input.corner.x) + abs(input.corner.y);
  }
  if (inside > 1.0) { discard; }
  var alpha = 1.0;
  if (shapeStyle.config.z == 1u) {
    alpha = smoothstep(0.45, 0.6, inside);
  }
  return vec4<f32>(input.color.rgb, input.color.a * alpha);
}
`;

type ShapeColor = readonly [number, number, number, number];

type ShapeStyleProps = {
  /** Number of groups the shape buffers hold. */
  groupCount: number;
  /** Colors indexed by `group % 8`, RGBA 0-255. Markers use the first entry. */
  palette: readonly ShapeColor[];
  /** Line width in CSS pixels. Defaults to 1.5. */
  widthPixels?: number;
  /** Overall opacity multiplier. Defaults to 1. */
  opacity?: number;
};

function writeShapeStyle(
  styleBuffer: Buffer,
  props: ShapeStyleProps,
  extra: {sizePixels?: number; segments?: number; shape?: number; mode?: number} = {}
): void {
  const data = new ArrayBuffer(STORAGE_MODEL_STYLE_BYTE_LENGTH);
  const floats = new Float32Array(data);
  const words = new Uint32Array(data);
  for (let index = 0; index < SHAPE_PALETTE_SIZE; index++) {
    const color = props.palette[index % props.palette.length];
    floats.set([color[0] / 255, color[1] / 255, color[2] / 255, color[3] / 255], index * 4);
  }
  floats.set([props.widthPixels ?? 1.5, props.opacity ?? 1, extra.sizePixels ?? 6, 0], 32);
  words.set([props.groupCount, extra.segments ?? 1, extra.shape ?? 0, extra.mode ?? 0], 36);
  styleBuffer.write(new Uint8Array(data));
}

/** Props for {@link HullOutlineLayer}. */
export type HullOutlineLayerProps = LayerProps &
  ShapeStyleProps & {
    /** uint32 `groupCount + 1` start of each hull in `hullPositions`. */
    hullOffsets: Buffer;
    /** uint32 emitted vertex count per group. */
    hullCounts: Buffer;
    /** `float32x2` hull vertices, counter-clockwise per group, ring not closed. */
    hullPositions: Buffer;
    /** Capacity of `hullPositions` (instances drawn; slots past the total are skipped). */
    slotCount: number;
  };

/** Draws every `GPUGroupConvexHull` ring as a closed outline, colored by group. */
export class HullOutlineLayer extends StorageModelLayer<HullOutlineLayerProps> {
  static override layerName = 'HullOutlineLayer';
  protected getShaderSource(): string {
    return HULL_OUTLINE_SHADER;
  }
  protected getVertexCount(): number {
    return 6;
  }
  protected getInstanceCount(): number {
    return this.props.slotCount;
  }
  protected getBindings(styleBuffer: Buffer): Record<string, Buffer> {
    return {
      shapeStyle: styleBuffer,
      hullOffsets: this.props.hullOffsets,
      hullCounts: this.props.hullCounts,
      hullPositions: this.props.hullPositions
    };
  }
  protected writeStyle(styleBuffer: Buffer): void {
    writeShapeStyle(styleBuffer, this.props);
  }
}

/** Props for {@link EllipseOutlineLayer}. */
export type EllipseOutlineLayerProps = LayerProps &
  ShapeStyleProps & {
    /** uint32 member count per group; groups with fewer than 3 are skipped. */
    counts: Buffer;
    /** `float32x2` ellipse centers per group, in the same plane as the ellipses. */
    centers: Buffer;
    /** float32 `[angle, sigmaX, sigmaY]` per group (`GPUGroupGeometry` `ellipses`). */
    ellipses: Buffer;
    /** Polyline segments per ellipse. Defaults to 48. */
    segments?: number;
  };

/** Draws `GPUGroupGeometry` standard deviational ellipses, colored by group. */
export class EllipseOutlineLayer extends StorageModelLayer<EllipseOutlineLayerProps> {
  static override layerName = 'EllipseOutlineLayer';
  protected getShaderSource(): string {
    return ELLIPSE_OUTLINE_SHADER;
  }
  protected getVertexCount(): number {
    return 6;
  }
  protected getInstanceCount(): number {
    return this.props.groupCount * (this.props.segments ?? 48);
  }
  protected getBindings(styleBuffer: Buffer): Record<string, Buffer> {
    return {
      shapeStyle: styleBuffer,
      groupCounts: this.props.counts,
      groupCenters: this.props.centers,
      groupEllipses: this.props.ellipses
    };
  }
  protected writeStyle(styleBuffer: Buffer): void {
    writeShapeStyle(styleBuffer, this.props, {segments: this.props.segments ?? 48});
  }
}

/** Props for {@link BoundsOutlineLayer}. */
export type BoundsOutlineLayerProps = LayerProps &
  ShapeStyleProps & {
    /** uint32 member count per group; empty groups are skipped. */
    counts: Buffer;
    /** `float32x4` `[minX, minY, maxX, maxY]` per group. */
    bounds: Buffer;
  };

/** Draws `GPUGroupGeometry` bounding boxes, colored by group. */
export class BoundsOutlineLayer extends StorageModelLayer<BoundsOutlineLayerProps> {
  static override layerName = 'BoundsOutlineLayer';
  protected getShaderSource(): string {
    return BOUNDS_OUTLINE_SHADER;
  }
  protected getVertexCount(): number {
    return 6;
  }
  protected getInstanceCount(): number {
    return this.props.groupCount * 4;
  }
  protected getBindings(styleBuffer: Buffer): Record<string, Buffer> {
    return {
      shapeStyle: styleBuffer,
      groupCounts: this.props.counts,
      groupBounds: this.props.bounds
    };
  }
  protected writeStyle(styleBuffer: Buffer): void {
    writeShapeStyle(styleBuffer, this.props);
  }
}

/** Props for {@link GroupMarkerLayer}. */
export type GroupMarkerLayerProps = LayerProps &
  ShapeStyleProps & {
    /** uint32 member count per group; empty groups are skipped. */
    counts: Buffer;
    /** `float32x2` per-group positions (used when `source` is `'centers'`). */
    centers: Buffer;
    /** uint32 point row per group, or `0xffffffff` (used when `source` is `'medoids'`). */
    medoidRows: Buffer;
    /** `float32x2` positions that `medoidRows` index (used when `source` is `'medoids'`). */
    positions: Buffer;
    /** Where each marker sits. */
    source: 'centers' | 'medoids';
    /** `'disc'`, `'ring'` or `'diamond'`. Defaults to `'disc'`. */
    shape?: 'disc' | 'ring' | 'diamond';
    /** Half-size in CSS pixels. Defaults to 6. */
    sizePixels?: number;
  };

const MARKER_SHAPES = {disc: 0, ring: 1, diamond: 2} as const;

/** One marker per group at its center or medoid, in the first palette color. */
export class GroupMarkerLayer extends StorageModelLayer<GroupMarkerLayerProps> {
  static override layerName = 'GroupMarkerLayer';
  protected getShaderSource(): string {
    return GROUP_MARKER_SHADER;
  }
  protected getVertexCount(): number {
    return 6;
  }
  protected getInstanceCount(): number {
    return this.props.groupCount;
  }
  protected getBindings(styleBuffer: Buffer): Record<string, Buffer> {
    return {
      shapeStyle: styleBuffer,
      groupCounts: this.props.counts,
      markerCenters: this.props.centers,
      markerRows: this.props.medoidRows,
      markerPositions: this.props.positions
    };
  }
  protected writeStyle(styleBuffer: Buffer): void {
    writeShapeStyle(styleBuffer, this.props, {
      sizePixels: this.props.sizePixels ?? 6,
      shape: MARKER_SHAPES[this.props.shape ?? 'disc'],
      mode: this.props.source === 'medoids' ? 1 : 0
    });
  }
}
