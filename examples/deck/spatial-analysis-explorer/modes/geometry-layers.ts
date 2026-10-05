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

/**
 * Bespoke layers of the Geometry mode.
 *
 * - {@link PathOutputLayer} draws a contributor's flat `GPULinePathOutput` (vertex rows plus path
 *   offsets) directly: instance `v` is the segment from vertex `v` to `v + 1`, hidden when `v + 1`
 *   starts a new path. The path of a vertex comes from a binary search over `pathOffsets` in the
 *   vertex shader, so nothing is read back or repacked, and the vertex count is the GPU-written
 *   indirect record.
 * - {@link PairSegmentLayer} draws one segment per row between two position columns (a point and
 *   its snapped foot point, or an event and its direction tick).
 *
 * Both work in `COORDINATE_SYSTEM.METER_OFFSETS` (planar meters) or `COORDINATE_SYSTEM.LNGLAT`
 * (degrees, including longitudes unwrapped past 180 by the great-circle contributors).
 */

export const GEOMETRY_BLEND_PARAMETERS = {
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

/** Color mapping of a geometry layer. */
export type GeometryColormap = 'viridis' | 'inferno' | 'diverging';

/** RGBA color, 0-255 channels. */
export type GeometryColor = readonly [number, number, number, number?];

const STYLE_BYTE_LENGTH = 96;

export const GEOMETRY_COMMON_WGSL = /* wgsl */ `
struct GeometryStyle {
  color: vec4<f32>,
  valueRange: vec2<f32>,
  positionOffset: vec2<f32>,
  widthPixels: f32,
  opacity: f32,
  colormap: u32,
  valueSource: u32,
  pathOffsetCount: u32,
  directionScale: f32,
  _padding0: u32,
  _padding1: u32,
  extraFloats: vec4<f32>,
  extraWords: vec4<u32>,
};

@group(0) @binding(auto) var<uniform> geometryStyle: GeometryStyle;
@group(0) @binding(auto) var<storage, read> geometryValues: array<f32>;

fn geometryViridis(t: f32) -> vec3<f32> {
  let c0 = vec3<f32>(0.2777273272234177, 0.005407344544966578, 0.3340998053353061);
  let c1 = vec3<f32>(0.1050930431085774, 1.404613529898575, 1.384590162594685);
  let c2 = vec3<f32>(-0.3308618287255563, 0.214847559468213, 0.09509516302823659);
  let c3 = vec3<f32>(-4.634230498983486, -5.799100973351585, -19.33244095627987);
  let c4 = vec3<f32>(6.228269936347081, 14.17993336680509, 56.69055260068105);
  let c5 = vec3<f32>(4.776384997670288, -13.74514537774601, -65.35303263337234);
  let c6 = vec3<f32>(-5.435455855934631, 4.645852612178535, 26.3124352495832);
  return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}

fn geometryInferno(t: f32) -> vec3<f32> {
  let c0 = vec3<f32>(0.0002189403691192265, 0.001651004631001012, -0.01948089843709184);
  let c1 = vec3<f32>(0.1065134194856116, 0.5639564367884091, 3.932712388889277);
  let c2 = vec3<f32>(11.60249308247187, -3.972853965665698, -15.9423941062914);
  let c3 = vec3<f32>(-41.70399613139459, 17.43639888205313, 44.35414519872813);
  let c4 = vec3<f32>(77.162935699427, -33.40235894210092, -81.80730925738993);
  let c5 = vec3<f32>(-71.31942824499214, 32.62606426397723, 73.20951985803202);
  let c6 = vec3<f32>(25.13112622477341, -12.24266895238567, -23.07032500287172);
  return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}

// Maps a scalar to a color. Alpha 0 means "no data".
fn getGeometryColor(value: f32) -> vec4<f32> {
  let bits = bitcast<u32>(value);
  if ((bits & 0x7fffffffu) >= 0x7f800000u) {
    return vec4<f32>(0.0);
  }
  let range = geometryStyle.valueRange;
  let t = clamp((value - range.x) / max(range.y - range.x, 1e-20), 0.0, 1.0);
  var rgb = geometryViridis(t);
  if (geometryStyle.colormap == 1u) {
    rgb = geometryInferno(t);
  } else if (geometryStyle.colormap == 2u) {
    // Blue (negative) through near-white to orange (positive).
    let low = vec3<f32>(0.23, 0.55, 1.0);
    let middle = vec3<f32>(0.93, 0.93, 0.93);
    let high = vec3<f32>(1.0, 0.45, 0.1);
    rgb = select(
      mix(low, middle, t * 2.0),
      mix(middle, high, t * 2.0 - 1.0),
      t >= 0.5
    );
  }
  return vec4<f32>(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)), geometryStyle.color.a);
}

fn projectGeometryPosition(position: vec2<f32>) -> vec4<f32> {
  var clipPosition = project_position_to_clipspace(
    vec3<f32>(position + geometryStyle.positionOffset, 0.0),
    vec3<f32>(0.0),
    vec3<f32>(0.0)
  );
  // Deck's project32 matrices use OpenGL depth; WebGPU clip space requires [0, w].
  clipPosition.z = (clipPosition.z + clipPosition.w) * 0.5;
  return clipPosition;
}

struct GeometryVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) side: f32,
  @location(1) color: vec4<f32>,
};

// Expands a segment between two projected points into a screen-space quad corner.
fn expandGeometrySegment(
  startClip: vec4<f32>,
  endClip: vec4<f32>,
  vertexIndex: u32,
  color: vec4<f32>
) -> GeometryVertexOutput {
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
      (normal * corner.y + along) * geometryStyle.widthPixels * 0.5
    ),
    clipPosition.z,
    clipPosition.w
  );
  var output: GeometryVertexOutput;
  output.position = clipPosition;
  output.side = corner.y;
  output.color = color;
  return output;
}

fn getHiddenGeometryVertex() -> GeometryVertexOutput {
  var output: GeometryVertexOutput;
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  output.side = 0.0;
  output.color = vec4<f32>(0.0);
  return output;
}

`;

const SEGMENT_FRAGMENT_WGSL = /* wgsl */ `
@fragment fn fragmentMain(input: GeometryVertexOutput) -> @location(0) vec4<f32> {
  let coverage = 1.0 - smoothstep(0.55, 1.0, abs(input.side));
  return vec4<f32>(input.color.rgb, input.color.a * geometryStyle.opacity * coverage);
}
`;

const PATH_SHADER = /* wgsl */ `
${GEOMETRY_COMMON_WGSL}
${SEGMENT_FRAGMENT_WGSL}
@group(0) @binding(auto) var<storage, read> pathPositions: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> pathOffsets: array<u32>;
@group(0) @binding(auto) var<storage, read> pathVertexCount: array<u32>;

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> GeometryVertexOutput {
  let vertex = instanceIndex;
  if (vertex + 1u >= pathVertexCount[0]) {
    return getHiddenGeometryVertex();
  }
  // First offset row that is greater than the vertex: the next path start (or the end row).
  var low = 0u;
  var high = geometryStyle.pathOffsetCount;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (pathOffsets[middle] <= vertex) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  if (low < geometryStyle.pathOffsetCount && pathOffsets[low] == vertex + 1u) {
    return getHiddenGeometryVertex();
  }
  let path = max(low, 1u) - 1u;
  let start = pathPositions[vertex];
  let end = pathPositions[vertex + 1u];
  if (start.x != start.x || end.x != end.x) {
    return getHiddenGeometryVertex();
  }
  var color = geometryStyle.color;
  if (geometryStyle.valueSource == 1u) {
    color = getGeometryColor(geometryValues[path]);
  } else if (geometryStyle.valueSource == 2u) {
    color = getGeometryColor(geometryValues[vertex]);
  } else if (geometryStyle.valueSource == 3u) {
    let palette = array<vec3<f32>, 6>(
      vec3<f32>(0.31, 0.79, 1.0), vec3<f32>(1.0, 0.58, 0.28), vec3<f32>(0.74, 0.48, 1.0),
      vec3<f32>(0.34, 0.92, 0.66), vec3<f32>(1.0, 0.41, 0.66), vec3<f32>(0.96, 0.86, 0.34)
    );
    color = vec4<f32>(palette[path % 6u], geometryStyle.color.a);
  }
  if (color.a <= 0.0) {
    return getHiddenGeometryVertex();
  }
  return expandGeometrySegment(
    projectGeometryPosition(start), projectGeometryPosition(end), vertexIndex, color
  );
}
`;

const PAIR_SHADER = /* wgsl */ `
${GEOMETRY_COMMON_WGSL}
${SEGMENT_FRAGMENT_WGSL}
@group(0) @binding(auto) var<storage, read> pairStarts: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> pairEnds: array<vec2<f32>>;

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> GeometryVertexOutput {
  let start = pairStarts[instanceIndex];
  var end = pairEnds[instanceIndex];
  if (geometryStyle.directionScale != 0.0) {
    // The second column is a unit direction: draw a tick of directionScale meters.
    end = start + end * geometryStyle.directionScale;
  }
  if (start.x != start.x || end.x != end.x || start.y != start.y || end.y != end.y) {
    return getHiddenGeometryVertex();
  }
  var color = geometryStyle.color;
  if (geometryStyle.valueSource == 2u) {
    color = getGeometryColor(geometryValues[instanceIndex]);
  }
  if (color.a <= 0.0) {
    return getHiddenGeometryVertex();
  }
  return expandGeometrySegment(
    projectGeometryPosition(start), projectGeometryPosition(end), vertexIndex, color
  );
}
`;

const FAN_SHADER = /* wgsl */ `
${GEOMETRY_COMMON_WGSL}
@group(0) @binding(auto) var<storage, read> fanSegments: array<vec4<f32>>;
@group(0) @binding(auto) var<storage, read> fanFeatureRows: array<u32>;
@group(0) @binding(auto) var<storage, read> fanCentroids: array<vec2<f32>>;

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> GeometryVertexOutput {
  let segment = fanSegments[instanceIndex];
  let feature = fanFeatureRows[instanceIndex];
  let centroid = fanCentroids[feature];
  let color = getGeometryColor(geometryValues[feature]);
  if (centroid.x != centroid.x || color.a <= 0.0) {
    return getHiddenGeometryVertex();
  }
  var position = centroid;
  if (vertexIndex == 1u) {
    position = segment.xy;
  } else if (vertexIndex == 2u) {
    position = segment.zw;
  }
  var output: GeometryVertexOutput;
  output.position = projectGeometryPosition(position);
  output.side = 0.0;
  output.color = color;
  return output;
}

@fragment fn fragmentMain(input: GeometryVertexOutput) -> @location(0) vec4<f32> {
  return vec4<f32>(input.color.rgb, input.color.a * geometryStyle.opacity);
}
`;

type GeometryLayerState = {
  model: Model | null;
  styleBuffer: Buffer | null;
  placeholderBuffer: Buffer | null;
};

/** Props shared by the geometry layers. */
export type GeometryStyleProps = {
  /** Fixed color, or the alpha of colormapped values. Defaults to white. */
  color?: GeometryColor;
  /** Line width in CSS pixels. Defaults to 2. */
  widthPixels?: number;
  /** Optional float32 values colored through `colormap`. Meaning depends on the layer. */
  values?: Buffer | null;
  /** Colormap for `values`. Defaults to viridis. */
  colormap?: GeometryColormap;
  /** `[min, max]` mapped to the colormap ends. */
  valueRange?: readonly [number, number];
  /** Offset added to every position before projection (for example `[-360, 0]` degrees). */
  positionOffset?: readonly [number, number];
};

/** Shared lifecycle of the geometry layers: one model, a style uniform and storage bindings. */
export abstract class GeometryBaseLayer<
  PropsT extends LayerProps & GeometryStyleProps
> extends Layer<PropsT> {
  static override layerName = 'GeometryBaseLayer';
  static override defaultProps = {
    coordinateSystem: COORDINATE_SYSTEM.METER_OFFSETS,
    parameters: GEOMETRY_BLEND_PARAMETERS
  };

  override getAttributeManager() {
    return null;
  }

  protected abstract getShaderSource(): string;
  protected abstract getBindings(placeholder: Buffer): Record<string, Buffer>;
  protected abstract getValueSource(): number;
  protected getVertexCount(): number {
    return 6;
  }
  protected getPathOffsetCount(): number {
    return 0;
  }
  protected getDirectionScale(): number {
    return 0;
  }
  /** Four layer-specific floats written to `extraFloats`. */
  protected getExtraFloats(): readonly number[] {
    return [0, 0, 0, 0];
  }
  /** Four layer-specific words written to `extraWords`. */
  protected getExtraWords(): readonly number[] {
    return [0, 0, 0, 0];
  }
  protected abstract drawInstances(model: Model, renderPass: RenderPass): void;

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
    const model = new Model(device, {
      ...this.getShaders({modules: [project32], source: this.getShaderSource()}),
      id: `${this.id}-model`,
      topology: 'triangle-list',
      isInstanced: true,
      vertexCount: this.getVertexCount(),
      instanceCount: 0,
      bufferLayout: [],
      bindings: {...this.getBindings(placeholderBuffer), geometryStyle: styleBuffer},
      parameters: GEOMETRY_BLEND_PARAMETERS
    });
    this.setState({model, styleBuffer, placeholderBuffer});
  }

  override updateState(parameters: UpdateParameters<this>): void {
    super.updateState(parameters);
    const {model, styleBuffer, placeholderBuffer} = this.state as GeometryLayerState;
    if (model && styleBuffer && placeholderBuffer) {
      model.setBindings({...this.getBindings(placeholderBuffer), geometryStyle: styleBuffer});
    }
  }

  override getModels(): Model[] {
    const model = (this.state as GeometryLayerState).model;
    return model ? [model] : [];
  }

  override draw({renderPass}: {renderPass: RenderPass}): void {
    const {model, styleBuffer} = this.state as GeometryLayerState;
    if (!model || !styleBuffer) return;
    const data = new ArrayBuffer(STYLE_BYTE_LENGTH);
    const floats = new Float32Array(data);
    const words = new Uint32Array(data);
    const [red, green, blue, alpha = 255] = this.props.color ?? [255, 255, 255, 255];
    floats.set([red / 255, green / 255, blue / 255, alpha / 255], 0);
    floats.set(this.props.valueRange ?? [0, 1], 4);
    floats.set(this.props.positionOffset ?? [0, 0], 6);
    floats[8] = this.props.widthPixels ?? 2;
    floats[9] = this.props.opacity ?? 1;
    words[10] = {viridis: 0, inferno: 1, diverging: 2}[this.props.colormap ?? 'viridis'];
    words[11] = this.getValueSource();
    words[12] = this.getPathOffsetCount();
    floats[13] = this.getDirectionScale();
    floats.set(this.getExtraFloats(), 16);
    words.set(this.getExtraWords(), 20);
    styleBuffer.write(new Uint8Array(data));
    this.drawInstances(model, renderPass);
  }

  override finalizeState(context: LayerContext): void {
    const state = this.state as GeometryLayerState;
    state.model?.destroy();
    state.styleBuffer?.destroy();
    state.placeholderBuffer?.destroy();
    this.setState({model: null, styleBuffer: null, placeholderBuffer: null});
    super.finalizeState(context);
  }
}

/** Which column colors a {@link PathOutputLayer}. */
export type PathColorSource = 'uniform' | 'path-value' | 'vertex-value' | 'path-index';

/** Props of {@link PathOutputLayer}. */
export type PathOutputLayerProps = LayerProps &
  GeometryStyleProps & {
    /** Output vertices, `float32x2`. */
    positions: Buffer;
    /** Path start offsets (`pathOffsetCount` rows, uint32). */
    pathOffsets: Buffer;
    /** Number of rows in `pathOffsets` (path capacity + 1). */
    pathOffsetCount: number;
    /** One uint32: number of valid vertices. */
    vertexCount: Buffer;
    /** Indirect record (`vertexCount` 6) whose instance count was copied from `vertexCount`. */
    drawCommands: DrawCommandBuffer;
    /** What colors a segment. `values` is indexed by path or by vertex. Defaults to uniform. */
    colorSource?: PathColorSource;
  };

/** Draws a flat path output straight from GPU buffers. See the file comment. */
export class PathOutputLayer extends GeometryBaseLayer<PathOutputLayerProps> {
  static override layerName = 'PathOutputLayer';

  protected getShaderSource(): string {
    return PATH_SHADER;
  }
  protected getBindings(placeholder: Buffer): Record<string, Buffer> {
    return {
      geometryValues: this.props.values ?? placeholder,
      pathPositions: this.props.positions,
      pathOffsets: this.props.pathOffsets,
      pathVertexCount: this.props.vertexCount
    };
  }
  protected getValueSource(): number {
    const source = this.props.colorSource ?? 'uniform';
    return {uniform: 0, 'path-value': 1, 'vertex-value': 2, 'path-index': 3}[source];
  }
  protected override getPathOffsetCount(): number {
    return this.props.pathOffsetCount;
  }
  protected drawInstances(model: Model, renderPass: RenderPass): void {
    // Bind pipeline and bindings with an empty draw, then replay the GPU-written record.
    model.setInstanceCount(0);
    model.draw(renderPass);
    this.props.drawCommands.draw(renderPass, 0);
  }
}

/** Props of {@link PairSegmentLayer}. */
export type PairSegmentLayerProps = LayerProps &
  GeometryStyleProps & {
    /** Segment starts, `float32x2`, one per instance. */
    starts: Buffer;
    /** Segment ends, `float32x2`, or unit directions when `directionScale` is set. */
    ends: Buffer;
    /** Number of segments. */
    instanceCount: number;
    /** When non-zero, `ends` holds unit directions and ticks are this many meters long. */
    directionScale?: number;
  };

/** One segment per row between two position columns, colored by an optional value column. */
export class PairSegmentLayer extends GeometryBaseLayer<PairSegmentLayerProps> {
  static override layerName = 'PairSegmentLayer';

  protected getShaderSource(): string {
    return PAIR_SHADER;
  }
  protected getBindings(placeholder: Buffer): Record<string, Buffer> {
    return {
      geometryValues: this.props.values ?? placeholder,
      pairStarts: this.props.starts,
      pairEnds: this.props.ends
    };
  }
  protected getValueSource(): number {
    return this.props.values ? 2 : 0;
  }
  protected override getDirectionScale(): number {
    return this.props.directionScale ?? 0;
  }
  protected drawInstances(model: Model, renderPass: RenderPass): void {
    model.setInstanceCount(this.props.instanceCount);
    model.draw(renderPass);
  }
}

/** Props of {@link PolygonFanLayer}. */
export type PolygonFanLayerProps = LayerProps &
  GeometryStyleProps & {
    /** Ring edges as `float32x4` rows `x0, y0, x1, y1`. */
    segments: Buffer;
    /** Feature row of every ring edge (uint32). */
    featureRows: Buffer;
    /** Per-feature fan centers, `float32x2` (for example the contributor's `centroids` output). */
    centroids: Buffer;
    /** Number of ring edges. */
    instanceCount: number;
  };

/**
 * Fills polygons as triangle fans from per-feature centers: one triangle per ring edge, colored by
 * the feature's value. Exact for polygons that are star-shaped around their center.
 */
export class PolygonFanLayer extends GeometryBaseLayer<PolygonFanLayerProps> {
  static override layerName = 'PolygonFanLayer';

  protected getShaderSource(): string {
    return FAN_SHADER;
  }
  protected override getVertexCount(): number {
    return 3;
  }
  protected getBindings(placeholder: Buffer): Record<string, Buffer> {
    return {
      geometryValues: this.props.values ?? placeholder,
      fanSegments: this.props.segments,
      fanFeatureRows: this.props.featureRows,
      fanCentroids: this.props.centroids
    };
  }
  protected getValueSource(): number {
    return 2;
  }
  protected drawInstances(model: Model, renderPass: RenderPass): void {
    model.setInstanceCount(this.props.instanceCount);
    model.draw(renderPass);
  }
}
