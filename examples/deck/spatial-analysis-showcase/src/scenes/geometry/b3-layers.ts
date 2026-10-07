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
import {COLORMAP_INDEXES, getRampWgsl, type RampName} from '../../engine/ramps';
import type {ClassTable} from '../../cartography/types';
import {B3_PALETTE} from './b3-palette';

/**
 * Bespoke layers of the Geometry chapter. They draw contributor outputs straight from GPU storage
 * buffers (no readback, no repacking) and color through the shared ramp table of `engine/ramps.ts`.
 *
 * - {@link PathOutputLayer}: a flat `GPULinePathOutput` (vertices + path offsets + GPU-written count).
 * - {@link PairSegmentLayer}: one segment per row between two position columns, or a tick along a
 *   unit direction column.
 * - {@link FeatureTriangleLayer}: a static triangle list (a triangulated polygon fill) colored by a
 *   per-feature GPU value column (a contributor output) through a ramp, a category palette or a flag.
 * - {@link TriangleListLayer}: a non-indexed triangle list of GPU positions in one color
 *   (for example the triangles `GPUOutlineGeometry` writes).
 * - {@link RingEdgeLayer}: closes rings of a GPU ring output into outlines (GPUShapeGenerator,
 *   GPURectangleClip polygon output).
 *
 * All work in `COORDINATE_SYSTEM.METER_OFFSETS` (planar meters) or `COORDINATE_SYSTEM.LNGLAT`.
 */

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

/** RGBA color, 0-255 channels. */
export type B3Color = readonly [number, number, number, number?];

export {B3_PALETTE};

const PALETTE_WGSL = B3_PALETTE.map(
  ([r, g, b]) =>
    `vec3<f32>(${(r / 255).toFixed(4)}, ${(g / 255).toFixed(4)}, ${(b / 255).toFixed(4)})`
).join(', ');

/** How a scalar value becomes a color. */
export type B3ValueMapping = 'ramp' | 'category' | 'flag' | 'class';

const STYLE_BYTE_LENGTH = 448;

export const B3_COMMON_WGSL = /* wgsl */ `
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
  secondaryColor: vec4<f32>,
  classPalette: array<vec4<f32>, 16>,
  classBreaks: array<vec4<f32>, 4>,
  classCount: u32,
  _classPadding0: u32,
  _classPadding1: u32,
  _classPadding2: u32,
};

@group(0) @binding(auto) var<uniform> geometryStyle: GeometryStyle;
@group(0) @binding(auto) var<storage, read> geometryValues: array<f32>;

${getRampWgsl()}

// Maps a scalar to a color. Alpha 0 means "no data". extraWords.x: 0 ramp, 1 category, 2 flag.
fn getGeometryColor(value: f32) -> vec4<f32> {
  let bits = bitcast<u32>(value);
  if ((bits & 0x7fffffffu) >= 0x7f800000u) {
    return vec4<f32>(0.0);
  }
  let mapping = geometryStyle.extraWords.x;
  if (mapping == 2u) {
    return select(geometryStyle.secondaryColor, geometryStyle.color, value != 0.0);
  }
  if (mapping == 1u) {
    let palette = array<vec3<f32>, 8>(${PALETTE_WGSL});
    return vec4<f32>(palette[u32(max(value, 0.0)) % 8u], geometryStyle.color.a);
  }
  if (mapping == 3u) {
    var classIndex = 0u;
    for (var index = 0u; index + 1u < geometryStyle.classCount; index = index + 1u) {
      if (value >= geometryStyle.classBreaks[index / 4u][index % 4u]) {
        classIndex = index + 1u;
      }
    }
    return geometryStyle.classPalette[classIndex];
  }
  let range = geometryStyle.valueRange;
  var t = clamp((value - range.x) / max(range.y - range.x, 1e-20), 0.0, 1.0);
  if (geometryStyle.extraWords.y != 0u) {
    t = sqrt(t);
  }
  let rgb = spatialAnalysisSampleRamp(geometryStyle.colormap, t);
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

const FLAT_FRAGMENT_WGSL = /* wgsl */ `
@fragment fn fragmentMain(input: GeometryVertexOutput) -> @location(0) vec4<f32> {
  return vec4<f32>(input.color.rgb, input.color.a * geometryStyle.opacity);
}
`;

const PATH_SHADER = /* wgsl */ `
${B3_COMMON_WGSL}
${SEGMENT_FRAGMENT_WGSL}
@group(0) @binding(auto) var<storage, read> pathPositions: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> pathOffsets: array<u32>;
@group(0) @binding(auto) var<storage, read> pathVertexCount: array<u32>;

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> GeometryVertexOutput {
  let vertex = instanceIndex;
  let count = pathVertexCount[0];
  // extraWords.w: closed rings. Their last vertex connects back to the path start.
  let closed = geometryStyle.extraWords.w != 0u;
  if (vertex >= count || (!closed && vertex + 1u >= count)) {
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
  let path = max(low, 1u) - 1u;
  var endIndex = vertex + 1u;
  if (vertex + 1u >= count || (low < geometryStyle.pathOffsetCount && pathOffsets[low] == vertex + 1u)) {
    if (!closed) {
      return getHiddenGeometryVertex();
    }
    endIndex = pathOffsets[path];
  }
  let start = pathPositions[vertex];
  let end = pathPositions[endIndex];
  if (start.x != start.x || end.x != end.x) {
    return getHiddenGeometryVertex();
  }
  var color = geometryStyle.color;
  if (geometryStyle.valueSource == 1u) {
    color = getGeometryColor(geometryValues[path]);
  } else if (geometryStyle.valueSource == 2u) {
    color = getGeometryColor(geometryValues[vertex]);
  } else if (geometryStyle.valueSource == 3u) {
    let palette = array<vec3<f32>, 8>(${PALETTE_WGSL});
    color = vec4<f32>(palette[path % 8u], geometryStyle.color.a);
  }
  if (color.a <= 0.0) {
    return getHiddenGeometryVertex();
  }
  return expandGeometrySegment(
    projectGeometryPosition(start), projectGeometryPosition(end), vertexIndex, color
  );
}
`;

const KEPT_SEGMENT_SHADER = /* wgsl */ `
${B3_COMMON_WGSL}
${SEGMENT_FRAGMENT_WGSL}
@group(0) @binding(auto) var<storage, read> keptVertexPositions: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> keptIds: array<u32>;
@group(0) @binding(auto) var<storage, read> keptVertexLines: array<u32>;
@group(0) @binding(auto) var<storage, read> keptCount: array<u32>;

// Instance i joins kept vertices i and i + 1 when they belong to the same line.
@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> GeometryVertexOutput {
  if (instanceIndex + 1u >= keptCount[0]) {
    return getHiddenGeometryVertex();
  }
  let startRow = keptIds[instanceIndex];
  let endRow = keptIds[instanceIndex + 1u];
  if (keptVertexLines[startRow] != keptVertexLines[endRow]) {
    return getHiddenGeometryVertex();
  }
  var color = geometryStyle.color;
  if (geometryStyle.valueSource == 3u) {
    let palette = array<vec3<f32>, 8>(${PALETTE_WGSL});
    color = vec4<f32>(palette[keptVertexLines[startRow] % 8u], geometryStyle.color.a);
  }
  return expandGeometrySegment(
    projectGeometryPosition(keptVertexPositions[startRow]),
    projectGeometryPosition(keptVertexPositions[endRow]),
    vertexIndex,
    color
  );
}
`;

const PAIR_SHADER = /* wgsl */ `
${B3_COMMON_WGSL}
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

const FEATURE_TRIANGLE_SHADER = /* wgsl */ `
${B3_COMMON_WGSL}
${FLAT_FRAGMENT_WGSL}
@group(0) @binding(auto) var<storage, read> triangleCorners: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> triangleFeatures: array<u32>;

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> GeometryVertexOutput {
  let feature = triangleFeatures[instanceIndex];
  var color = geometryStyle.color;
  if (geometryStyle.valueSource != 0u) {
    color = getGeometryColor(geometryValues[feature]);
  }
  let corner = triangleCorners[instanceIndex * 3u + vertexIndex];
  if (color.a <= 0.0 || corner.x != corner.x || corner.y != corner.y) {
    return getHiddenGeometryVertex();
  }
  var output: GeometryVertexOutput;
  output.position = projectGeometryPosition(corner);
  output.side = 0.0;
  output.color = color;
  return output;
}
`;

const TRIANGLE_LIST_SHADER = /* wgsl */ `
${B3_COMMON_WGSL}
${FLAT_FRAGMENT_WGSL}
@group(0) @binding(auto) var<storage, read> triangleVertices: array<vec2<f32>>;

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> GeometryVertexOutput {
  let position = triangleVertices[instanceIndex * 3u + vertexIndex];
  if (position.x != position.x || position.y != position.y) {
    return getHiddenGeometryVertex();
  }
  var output: GeometryVertexOutput;
  output.position = projectGeometryPosition(position);
  output.side = 0.0;
  output.color = geometryStyle.color;
  return output;
}
`;

const RING_EDGE_SHADER = /* wgsl */ `
${B3_COMMON_WGSL}
${SEGMENT_FRAGMENT_WGSL}
@group(0) @binding(auto) var<storage, read> ringPositions: array<vec2<f32>>;

// Rings are fixed-stride slots: slot s owns rows [s * stride, (s + 1) * stride); the ring closes.
@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> GeometryVertexOutput {
  let stride = geometryStyle.extraWords.z;
  let slot = instanceIndex / stride;
  let local = instanceIndex % stride;
  var next = local + 1u;
  if (next >= stride) {
    next = 0u;
  }
  let start = ringPositions[slot * stride + local];
  let end = ringPositions[slot * stride + next];
  if (start.x != start.x || end.x != end.x) {
    return getHiddenGeometryVertex();
  }
  var color = geometryStyle.color;
  if (geometryStyle.valueSource == 3u) {
    let palette = array<vec3<f32>, 8>(${PALETTE_WGSL});
    color = vec4<f32>(palette[slot % 8u], geometryStyle.color.a);
  } else if (geometryStyle.valueSource == 2u) {
    color = getGeometryColor(geometryValues[slot]);
  }
  if (color.a <= 0.0) {
    return getHiddenGeometryVertex();
  }
  return expandGeometrySegment(
    projectGeometryPosition(start), projectGeometryPosition(end), vertexIndex, color
  );
}
`;

type GeometryLayerState = {
  model: Model | null;
  styleBuffer: Buffer | null;
  placeholderBuffer: Buffer | null;
};

/** Props shared by the geometry layers. */
export type B3StyleProps = {
  /** Fixed color, or the alpha of colormapped values. Defaults to white. */
  color?: B3Color;
  /** Color of a zero value with `valueMapping: 'flag'`. Defaults to transparent. */
  secondaryColor?: B3Color;
  /** Line width in CSS pixels. Defaults to 2. */
  widthPixels?: number;
  /** Optional float32 values colored through `colormap` / `valueMapping`. Meaning depends on the layer. */
  values?: Buffer | null;
  /** How values become colors. Defaults to `'ramp'`. */
  valueMapping?: B3ValueMapping;
  /** Ramp name of `valueMapping: 'ramp'`. Defaults to viridis. Must match the legend. */
  colormap?: RampName;
  /** `[min, max]` mapped to the ramp ends. */
  valueRange?: readonly [number, number];
  /** Apply `sqrt` after normalising, matching a `sqrtScale` legend. */
  sqrtScale?: boolean;
  /** One fixed classification shared with a `classes` legend. */
  classTable?: ClassTable;
  /** Explicit fixed thresholds override `classTable.breaks`. */
  classBreaks?: readonly number[];
  /** Explicit fixed colours override `classTable.colors`. */
  classColors?: readonly B3Color[];
  /** Offset added to every position before projection (for example `[-360, 0]` degrees). */
  positionOffset?: readonly [number, number];
};

/** Shared lifecycle of the geometry layers: one model, a style uniform and storage bindings. */
export abstract class B3BaseLayer<PropsT extends LayerProps & B3StyleProps> extends Layer<PropsT> {
  static override layerName = 'B3BaseLayer';
  static override defaultProps = {
    coordinateSystem: COORDINATE_SYSTEM.METER_OFFSETS,
    parameters: BLEND_PARAMETERS
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
  /** Extra word z (ring stride of {@link RingEdgeLayer}). */
  protected getStride(): number {
    return 1;
  }
  /** Extra word w: 1 closes every path of a {@link PathOutputLayer}. */
  protected getClosed(): number {
    return 0;
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
      parameters: BLEND_PARAMETERS
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
    words[10] = COLORMAP_INDEXES[this.props.colormap ?? 'viridis'];
    words[11] = this.getValueSource();
    words[12] = this.getPathOffsetCount();
    floats[13] = this.getDirectionScale();
    words[20] = {ramp: 0, category: 1, flag: 2, class: 3}[this.props.valueMapping ?? 'ramp'];
    words[21] = this.props.sqrtScale ? 1 : 0;
    words[22] = this.getStride();
    words[23] = this.getClosed();
    const [sr, sg, sb, sa = 0] = this.props.secondaryColor ?? [0, 0, 0, 0];
    floats.set([sr / 255, sg / 255, sb / 255, sa / 255], 24);
    if ((this.props.valueMapping ?? 'ramp') === 'class') {
      const classBreaks = (this.props.classBreaks ?? this.props.classTable?.breaks ?? []).slice(
        0,
        15
      );
      const classColors = this.props.classColors ?? this.props.classTable?.colors ?? [];
      words[108] = classBreaks.length + 1;
      floats.set(classBreaks, 92);
      for (let index = 0; index < 16; index++) {
        const color = classColors[index] ??
          classColors[classColors.length - 1] ?? [255, 255, 255, 255];
        const [classRed, classGreen, classBlue, classAlpha = 255] = color;
        floats.set(
          [classRed / 255, classGreen / 255, classBlue / 255, classAlpha / 255],
          28 + index * 4
        );
      }
    }
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
  B3StyleProps & {
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
    /** Close every path back to its first vertex (polygon rings). Defaults to false. */
    closed?: boolean;
  };

/** Draws a flat path output straight from GPU buffers. */
export class PathOutputLayer extends B3BaseLayer<PathOutputLayerProps> {
  static override layerName = 'B3PathOutputLayer';

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
  protected override getClosed(): number {
    return this.props.closed ? 1 : 0;
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
  B3StyleProps & {
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
export class PairSegmentLayer extends B3BaseLayer<PairSegmentLayerProps> {
  static override layerName = 'B3PairSegmentLayer';

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

/** Props of {@link FeatureTriangleLayer}. */
export type FeatureTriangleLayerProps = LayerProps &
  B3StyleProps & {
    /** Triangle corners, `float32x2`, three rows per triangle (planar meters). */
    corners: Buffer;
    /** Feature row of every triangle (uint32), indexing `values`. */
    featureRows: Buffer;
    /** Number of triangles. */
    instanceCount: number;
  };

/** A triangulated polygon fill colored per feature from a GPU value column. */
export class FeatureTriangleLayer extends B3BaseLayer<FeatureTriangleLayerProps> {
  static override layerName = 'B3FeatureTriangleLayer';

  protected getShaderSource(): string {
    return FEATURE_TRIANGLE_SHADER;
  }
  protected override getVertexCount(): number {
    return 3;
  }
  protected getBindings(placeholder: Buffer): Record<string, Buffer> {
    return {
      geometryValues: this.props.values ?? placeholder,
      triangleCorners: this.props.corners,
      triangleFeatures: this.props.featureRows
    };
  }
  protected getValueSource(): number {
    return this.props.values ? 2 : 0;
  }
  protected drawInstances(model: Model, renderPass: RenderPass): void {
    model.setInstanceCount(this.props.instanceCount);
    model.draw(renderPass);
  }
}

/** Props of {@link TriangleListLayer}. */
export type TriangleListLayerProps = LayerProps &
  B3StyleProps & {
    /** `float32x2` triangle corners, three rows per triangle. */
    positions: Buffer;
    /** Number of triangles to draw. */
    triangleCount: number;
  };

/** Draws a triangle list of GPU-resident positions in one flat color. */
export class TriangleListLayer extends B3BaseLayer<TriangleListLayerProps> {
  static override layerName = 'B3TriangleListLayer';

  protected getShaderSource(): string {
    return TRIANGLE_LIST_SHADER;
  }
  protected override getVertexCount(): number {
    return 3;
  }
  protected getBindings(placeholder: Buffer): Record<string, Buffer> {
    return {geometryValues: placeholder, triangleVertices: this.props.positions};
  }
  protected getValueSource(): number {
    return 0;
  }
  protected drawInstances(model: Model, renderPass: RenderPass): void {
    model.setInstanceCount(this.props.triangleCount);
    model.draw(renderPass);
  }
}

/** Props of {@link RingEdgeLayer}. */
export type RingEdgeLayerProps = LayerProps &
  B3StyleProps & {
    /** Ring vertices, `float32x2`, `ringCount * stride` rows (fixed-stride slots). */
    positions: Buffer;
    /** Rows per ring slot; the ring closes from the last row to the first. */
    stride: number;
    /** Number of ring slots. */
    ringCount: number;
    /** Color each ring by its slot index (palette) or by `values[slot]`. Defaults to uniform. */
    colorSource?: 'uniform' | 'slot-value' | 'slot-index';
  };

/** Outlines fixed-stride closed rings such as the output of `GPUShapeGenerator`. */
export class RingEdgeLayer extends B3BaseLayer<RingEdgeLayerProps> {
  static override layerName = 'B3RingEdgeLayer';

  protected getShaderSource(): string {
    return RING_EDGE_SHADER;
  }
  protected getBindings(placeholder: Buffer): Record<string, Buffer> {
    return {
      geometryValues: this.props.values ?? placeholder,
      ringPositions: this.props.positions
    };
  }
  protected getValueSource(): number {
    const source = this.props.colorSource ?? 'uniform';
    return {uniform: 0, 'slot-value': 2, 'slot-index': 3}[source];
  }
  protected override getStride(): number {
    return this.props.stride;
  }
  protected drawInstances(model: Model, renderPass: RenderPass): void {
    model.setInstanceCount(this.props.ringCount * this.props.stride);
    model.draw(renderPass);
  }
}

/** Props of {@link KeptSegmentLayer}. */
export type KeptSegmentLayerProps = LayerProps &
  B3StyleProps & {
    /** `float32x2` positions of every original vertex. */
    positions: Buffer;
    /** Ascending kept vertex rows (the compact `ids` output of `GPULineSimplification`). */
    keptIds: Buffer;
    /** `uint32` line index of every original vertex, so two lines are never joined. */
    vertexLines: Buffer;
    /** One `uint32`: number of valid entries in `keptIds`. */
    keptCount: Buffer;
    /** Indirect record whose instance count was copied from `keptCount`. */
    drawCommands: DrawCommandBuffer;
    /** Color by the line index through the categorical palette. Defaults to uniform. */
    colorSource?: 'uniform' | 'line-index';
  };

/** Draws simplified polylines straight from the compact kept vertex IDs. */
export class KeptSegmentLayer extends B3BaseLayer<KeptSegmentLayerProps> {
  static override layerName = 'B3KeptSegmentLayer';

  protected getShaderSource(): string {
    return KEPT_SEGMENT_SHADER;
  }
  protected getBindings(placeholder: Buffer): Record<string, Buffer> {
    return {
      geometryValues: placeholder,
      keptVertexPositions: this.props.positions,
      keptIds: this.props.keptIds,
      keptVertexLines: this.props.vertexLines,
      keptCount: this.props.keptCount
    };
  }
  protected getValueSource(): number {
    return this.props.colorSource === 'line-index' ? 3 : 0;
  }
  protected drawInstances(model: Model, renderPass: RenderPass): void {
    model.setInstanceCount(0);
    model.draw(renderPass);
    this.props.drawCommands.draw(renderPass, 0);
  }
}
