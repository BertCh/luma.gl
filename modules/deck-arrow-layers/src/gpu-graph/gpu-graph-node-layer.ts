// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  Layer,
  picking,
  project32,
  type LayerContext,
  type LayerProps,
  type PickingInfo,
  type UpdateParameters
} from '@deck.gl/core';
import {Buffer, type RenderPass} from '@luma.gl/core';
import {Model} from '@luma.gl/engine';
import {
  GPU_GRAPH_MAXIMUM_PALETTE_LENGTH,
  type GPUGraphColor,
  type GPUGraphColorScale,
  type GPUGraphNodeColumn,
  type GPUGraphSizeScale
} from './gpu-graph-columns';

const GPU_GRAPH_DECK_POINT_VERTEX_COUNT = 65_536;

/** Node style uniform block size in bytes: 15 scalar words, 1 pad word, 3 colors, 8 palette stops. */
const NODE_STYLE_BYTE_LENGTH = 240;
const NODE_STYLE_WORD_COUNT = NODE_STYLE_BYTE_LENGTH / 4;

const FLAG_COLOR_COLUMN = 1;
const FLAG_SIZE_COLUMN = 2;
const FLAG_FILTER_MASK = 4;
const FLAG_HIGHLIGHT_MASK = 8;
const FLAG_PATH_RANKS = 16;
const FLAG_HIGHLIGHT_TINT = 32;
const FLAG_PATH_TINT = 64;

/** Default categorical palette (RGB 0-255), the eight hues of the original community colors. */
export const GPU_GRAPH_DEFAULT_COLOR_PALETTE: readonly GPUGraphColor[] = [
  [66, 201, 255],
  [255, 148, 71],
  [189, 122, 255],
  [87, 235, 168],
  [255, 105, 168],
  [245, 219, 87],
  [107, 158, 255],
  [158, 173, 199]
];

/** Default color for `null` column rows. */
const DEFAULT_NULL_COLOR: GPUGraphColor = [66, 79, 110];

/**
 * Counters exposed by {@link GPUGraphNodeLayer.getRenderStats} and
 * {@link GPUGraphEdgeLayer.getRenderStats}.
 */
export type GPUGraphLayerRenderStats = {
  /** Number of `draw()` calls, including picking passes and one per viewport. */
  drawCount: number;
  /** Number of times the layer uploaded its style uniform block (one on creation). */
  styleUniformWriteCount: number;
  /** Number of times the layer rebound storage buffers after creation. */
  bindingUpdateCount: number;
};

/**
 * GPU-resident, row-aligned graph columns consumed by a deck.gl node layer without staging or
 * copying. Every column is a caller-owned storage buffer with one 4-byte row per node.
 */
export type GPUGraphNodeLayerProps = LayerProps & {
  /** Interleaved-free `float32x2` node positions; the only vertex attribute. */
  positions: Buffer;
  /** Number of node rows and instances. */
  vertexCount: number;
  /** Forces true one-vertex point primitives without dropping or sampling resident instances. */
  pointMode?: boolean;
  /** Column mapped to color by {@link GPUGraphNodeLayerProps.colorScale}. Absent: `palette[0]`. */
  colorColumn?: GPUGraphNodeColumn;
  /** Color mapping. Defaults to {@link GPU_GRAPH_DEFAULT_COLOR_PALETTE} categorical. */
  colorScale?: GPUGraphColorScale;
  /** Column mapped to radius by {@link GPUGraphNodeLayerProps.sizeScale}. */
  sizeColumn?: GPUGraphNodeColumn;
  /** Radius mapping. Defaults to domain `[0, 1]` and range `[2, 8]` pixels. */
  sizeScale?: GPUGraphSizeScale;
  /** Radius when there is no `sizeColumn`. Default `max(1.4, min(6, 60 / sqrt(vertexCount)))`. */
  radiusPixels?: number;
  /** `u32` per node. Zero: the node is neither drawn nor pickable. */
  filterMask?: Buffer;
  /** `u32` per node. Nonzero: the node is highlighted (radius x1.45, full brightness). */
  highlightMask?: Buffer;
  /** Tint of highlighted nodes. Absent: highlighted nodes keep their scale color. */
  highlightColor?: GPUGraphColor;
  /** `u32` per node. Zero: off path. `k`: 1-based position along the extracted path. */
  pathRanks?: Buffer;
  /** Tint of on-path nodes. Absent: on-path nodes keep their scale color. */
  pathColor?: GPUGraphColor;
  /**
   * Dims nodes that are neither highlighted, on the path nor hovered.
   * Default: true when `highlightMask` or `pathRanks` is provided.
   */
  dimUnhighlighted?: boolean;
};

type NodeBindingKey = {
  colorColumn: Buffer;
  sizeColumn: Buffer;
  filterMask: Buffer;
  highlightMask: Buffer;
  pathRanks: Buffer;
};

type GPUGraphNodeLayerState = {
  model: Model | null;
  styleUniforms: Buffer | null;
  placeholder: Buffer | null;
  writtenStyle: Uint32Array | null;
  boundBuffers: NodeBindingKey | null;
  renderStats: GPUGraphLayerRenderStats;
};

const NODE_BLEND_PARAMETERS = {
  depthWriteEnabled: false,
  blend: true,
  blendColorOperation: 'add',
  blendAlphaOperation: 'add',
  blendColorSrcFactor: 'src-alpha',
  blendColorDstFactor: 'one-minus-src-alpha',
  blendAlphaSrcFactor: 'one',
  blendAlphaDstFactor: 'one-minus-src-alpha'
} as const;

/**
 * Direct instance vertex fetch plus up to five row-aligned storage columns.
 *
 * Storage buffers read in the vertex stage: colorColumn, sizeColumn, filterMask, highlightMask,
 * pathRanks (5 of the WebGPU default limit of 8). Absent inputs bind a 16-byte placeholder and
 * clear a flag in `nodeStyle`, so the bind group shape and the pipeline never change.
 */
export const GPU_GRAPH_DECK_NODE_SHADER = /* wgsl */ `
struct NodeStyle {
  radiusPixels: f32,
  opacity: f32,
  pointMode: u32,
  flags: u32,
  colorDomain: vec2<f32>,
  sizeDomain: vec2<f32>,
  sizeRange: vec2<f32>,
  colorType: u32,
  paletteLength: u32,
  colorFormat: u32,
  sizeFormat: u32,
  dimUnhighlighted: u32,
  _padding: u32,
  nullColor: vec4<f32>,
  highlightColor: vec4<f32>,
  pathColor: vec4<f32>,
  palette: array<vec4<f32>, 8>,
};

@group(0) @binding(auto) var<storage, read> colorColumn: array<u32>;
@group(0) @binding(auto) var<storage, read> sizeColumn: array<u32>;
@group(0) @binding(auto) var<storage, read> filterMask: array<u32>;
@group(0) @binding(auto) var<storage, read> highlightMask: array<u32>;
@group(0) @binding(auto) var<storage, read> pathRanks: array<u32>;
@group(0) @binding(auto) var<uniform> nodeStyle: NodeStyle;

struct NodeVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) corner: vec2<f32>,
  @location(1) color: vec4<f32>,
  @location(2) @interpolate(flat) pickingColor: vec3<f32>,
};

struct ColumnValue {
  value: f32,
  index: u32,
  isNull: bool,
};

fn hasFlag(flag: u32) -> bool {
  return (nodeStyle.flags & flag) != 0u;
}

fn getNodeCorner(vertexIndex: u32) -> vec2<f32> {
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
  );
  return corners[vertexIndex];
}

// format 0: uint32 (0xffffffff is null); format 1: float32 via bitcast (non-finite is null).
fn readColumnValue(raw: u32, format: u32) -> ColumnValue {
  var result: ColumnValue;
  if (format == 1u) {
    let value = bitcast<f32>(raw);
    result.isNull = (raw & 0x7f800000u) == 0x7f800000u;
    result.value = select(value, 0.0, result.isNull);
    result.index = u32(max(result.value, 0.0));
  } else {
    result.isNull = raw == 0xffffffffu;
    result.value = f32(raw);
    result.index = raw;
  }
  return result;
}

fn getLinearUnit(value: f32, domain: vec2<f32>) -> f32 {
  let span = domain.y - domain.x;
  return select(clamp((value - domain.x) / span, 0.0, 1.0), 0.0, abs(span) < 1e-20);
}

fn getNodeColor(index: u32) -> vec3<f32> {
  if (!hasFlag(${FLAG_COLOR_COLUMN}u)) {
    return nodeStyle.palette[0].rgb;
  }
  let column = readColumnValue(colorColumn[index], nodeStyle.colorFormat);
  if (column.isNull) {
    return nodeStyle.nullColor.rgb;
  }
  let stopCount = max(nodeStyle.paletteLength, 1u);
  if (nodeStyle.colorType == 1u) {
    return nodeStyle.palette[column.index % stopCount].rgb;
  }
  if (stopCount == 1u) {
    return nodeStyle.palette[0].rgb;
  }
  let position = getLinearUnit(column.value, nodeStyle.colorDomain) * f32(stopCount - 1u);
  let lower = min(u32(floor(position)), stopCount - 2u);
  return mix(nodeStyle.palette[lower].rgb, nodeStyle.palette[lower + 1u].rgb,
    position - f32(lower));
}

fn getNodeRadius(index: u32) -> f32 {
  if (!hasFlag(${FLAG_SIZE_COLUMN}u)) {
    return nodeStyle.radiusPixels;
  }
  let column = readColumnValue(sizeColumn[index], nodeStyle.sizeFormat);
  let unit = select(getLinearUnit(column.value, nodeStyle.sizeDomain), 0.0, column.isNull);
  return mix(nodeStyle.sizeRange.x, nodeStyle.sizeRange.y, unit);
}

fn encodeNodePickingColor(vertex: u32) -> vec3<f32> {
  let index = vertex + 1u;
  return vec3<f32>(
    f32(index % 256u),
    f32((index / 256u) % 256u),
    f32((index / 65536u) % 256u)
  ) / 255.0;
}

@vertex fn vertexMain(
  @location(0) nodePosition: vec2<f32>,
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> NodeVertexOutput {
  var output: NodeVertexOutput;
  // One gate for the draw and picking passes: filtered rows leave the clip volume entirely.
  if (hasFlag(${FLAG_FILTER_MASK}u) && filterMask[instanceIndex] == 0u) {
    output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    return output;
  }

  let corner = select(getNodeCorner(vertexIndex), vec2<f32>(0.0), nodeStyle.pointMode != 0u);
  let pickingColor = encodeNodePickingColor(instanceIndex);
  let isHighlighted = hasFlag(${FLAG_HIGHLIGHT_MASK}u) && highlightMask[instanceIndex] != 0u;
  let isOnPath = hasFlag(${FLAG_PATH_RANKS}u) && pathRanks[instanceIndex] != 0u;

  let highlightedObjectColor = picking_normalizeColor(picking.highlightedObjectColor);
  let isHovered = picking.isHighlightActive > 0.5 &&
    distance(pickingColor, highlightedObjectColor) < 0.00001;
  let isEmphasized = isHighlighted || isOnPath || isHovered;
  let radius = getNodeRadius(instanceIndex) * select(1.0, 1.45, isEmphasized);

  geometry.worldPosition = vec3<f32>(nodePosition, 0.0);
  geometry.pickingColor = pickingColor;
  var clipPosition = project_position_to_clipspace(
    vec3<f32>(nodePosition, 0.0),
    vec3<f32>(0.0),
    vec3<f32>(0.0)
  );
  // Deck's project32 matrices use OpenGL depth; WebGPU clip space requires [0, w].
  clipPosition.z = (clipPosition.z + clipPosition.w) * 0.5;
  clipPosition = vec4<f32>(
    clipPosition.xy + project_pixel_size_to_clipspace(corner * radius),
    clipPosition.z,
    clipPosition.w
  );

  var color = getNodeColor(instanceIndex);
  if (isHighlighted && hasFlag(${FLAG_HIGHLIGHT_TINT}u)) { color = nodeStyle.highlightColor.rgb; }
  if (isOnPath && hasFlag(${FLAG_PATH_TINT}u)) { color = nodeStyle.pathColor.rgb; }
  let inactiveBrightness = select(0.40, 0.78, nodeStyle.pointMode != 0u);
  let brightness = select(1.0, select(inactiveBrightness, 1.0, isEmphasized),
    nodeStyle.dimUnhighlighted != 0u);

  output.position = clipPosition;
  output.corner = corner;
  output.color = vec4<f32>(color * brightness, 0.95);
  output.pickingColor = pickingColor;
  return output;
}

@fragment fn fragmentMain(input: NodeVertexOutput) -> @location(0) vec4<f32> {
  let radiusSquared = dot(input.corner, input.corner);
  if (radiusSquared > 1.0) { discard; }
  if (picking.isActive > 0.5) {
    return vec4<f32>(input.pickingColor, 1.0);
  }
  let coverage = 1.0 - smoothstep(0.48, 1.0, radiusSquared);
  return vec4<f32>(input.color.rgb, input.color.a * nodeStyle.opacity * coverage);
}`;

/**
 * Deck layer whose actual instance vertex attribute is the progressive GPU Graph position buffer
 * and whose color, size, filter, highlight and path inputs are row-aligned GPU storage columns.
 *
 * @remarks
 * WebGPU only, by design. Graph columns are storage buffers read in the vertex stage, which
 * WebGL2 cannot bind, and the producers (GPU graph analytics) are WebGPU compute. Rather than
 * emulate columns with per-column vertex attributes, a WebGL2 device throws a clear error;
 * render CPU-side columns with a standard deck.gl layer instead.
 *
 * Style uniforms are packed in `initializeState` and `updateState` and uploaded only when the
 * packed bytes differ from the last upload, never in `draw()`. Buffers are rebound only when a
 * buffer identity or presence changes, which never rebuilds the pipeline. See
 * {@link GPUGraphNodeLayer.getRenderStats}.
 */
export class GPUGraphNodeLayer extends Layer<GPUGraphNodeLayerProps> {
  static override layerName = 'GPUGraphNodeLayer';
  // Deck's own default highlightColor ([0, 0, 128, 128], for autoHighlight) would otherwise always
  // look like a user tint. This layer applies hover emphasis in its vertex shader instead.
  static override defaultProps = {
    parameters: NODE_BLEND_PARAMETERS,
    highlightColor: null
  };

  override getAttributeManager() {
    return null;
  }

  /** Keeps Deck picking and lifecycle counts aligned with the resident vertex allocation. */
  override getNumInstances(): number {
    return this.props.vertexCount;
  }

  /**
   * Returns draw, style-upload and rebinding counters. They live in layer state, so they survive
   * Deck re-instantiating the layer with the same id. `drawCount` includes picking passes.
   */
  getRenderStats(): GPUGraphLayerRenderStats {
    const state = (this.state ?? {}) as Partial<GPUGraphNodeLayerState>;
    return {
      ...(state.renderStats ?? {
        drawCount: 0,
        styleUniformWriteCount: 0,
        bindingUpdateCount: 0
      })
    };
  }

  override initializeState({device}: LayerContext): void {
    if (device.type !== 'webgpu') {
      throw new Error(
        'GPUGraphNodeLayer requires WebGPU: graph columns are storage buffers read in the vertex stage, which WebGL2 cannot bind; render CPU-side columns with a standard deck.gl layer instead.'
      );
    }
    const styleUniforms = device.createBuffer({
      id: `${this.id}-style-uniforms`,
      byteLength: NODE_STYLE_BYTE_LENGTH,
      usage: Buffer.UNIFORM | Buffer.COPY_DST
    });
    const placeholder = device.createBuffer({
      id: `${this.id}-absent-column`,
      data: new Uint32Array(4),
      usage: Buffer.STORAGE | Buffer.COPY_DST
    });
    const boundBuffers = getBoundBuffers(this.props, placeholder);
    const pointMode = getPointMode(this.props);
    const model = new Model(device, {
      ...this.getShaders({
        modules: [project32, picking],
        source: GPU_GRAPH_DECK_NODE_SHADER
      }),
      id: `${this.id}-model`,
      topology: pointMode ? 'point-list' : 'triangle-list',
      isInstanced: true,
      vertexCount: pointMode ? 1 : 6,
      instanceCount: this.props.vertexCount,
      attributes: {nodePosition: this.props.positions},
      bufferLayout: [{name: 'nodePosition', format: 'float32x2', stepMode: 'instance'}],
      bindings: {...boundBuffers, nodeStyle: styleUniforms},
      parameters: NODE_BLEND_PARAMETERS
    });
    const state: GPUGraphNodeLayerState = {
      model,
      styleUniforms,
      placeholder,
      writtenStyle: null,
      boundBuffers,
      renderStats: {
        drawCount: 0,
        styleUniformWriteCount: 0,
        bindingUpdateCount: 0
      }
    };
    this.setState(state);
    // setState copies into the live state object, which is what later updates must mutate.
    writeStyleIfChanged(this.state as GPUGraphNodeLayerState, packNodeStyle(this.props));
  }

  /** Rebinds replaced columns and rewrites the style block only when its packed bytes change. */
  override updateState({props, oldProps}: UpdateParameters<this>): void {
    const state = this.state as GPUGraphNodeLayerState;
    const {model, styleUniforms, placeholder} = state;
    if (!model || !styleUniforms || !placeholder) return;
    if (props.positions !== oldProps.positions) {
      model.setAttributes({nodePosition: props.positions});
    }
    const boundBuffers = getBoundBuffers(props, placeholder);
    const previous = state.boundBuffers;
    if (
      !previous ||
      (Object.keys(boundBuffers) as (keyof NodeBindingKey)[]).some(
        key => boundBuffers[key] !== previous[key]
      )
    ) {
      model.setBindings({...boundBuffers, nodeStyle: styleUniforms});
      state.boundBuffers = boundBuffers;
      state.renderStats.bindingUpdateCount++;
    }
    writeStyleIfChanged(state, packNodeStyle(props));
    const pointMode = getPointMode(props);
    const topology = pointMode ? 'point-list' : 'triangle-list';
    if (model.topology !== topology) model.setTopology(topology);
    model.setVertexCount(pointMode ? 1 : 6);
    model.setInstanceCount(props.vertexCount);
  }

  override getModels(): Model[] {
    const model = (this.state as Partial<GPUGraphNodeLayerState> | undefined)?.model;
    return model ? [model] : [];
  }

  override draw({renderPass}: {renderPass: RenderPass}): void {
    const {model, renderStats} = this.state as GPUGraphNodeLayerState;
    if (!model) return;
    renderStats.drawCount++;
    model.draw(renderPass);
  }

  override getPickingInfo({info}: {info: PickingInfo}): PickingInfo {
    return info;
  }

  override finalizeState(context: LayerContext): void {
    const state = this.state as GPUGraphNodeLayerState;
    state.model?.destroy();
    state.styleUniforms?.destroy();
    state.placeholder?.destroy();
    this.setState({
      model: null,
      styleUniforms: null,
      placeholder: null
    } satisfies Partial<GPUGraphNodeLayerState>);
    super.finalizeState(context);
  }
}

function getPointMode(props: GPUGraphNodeLayerProps): boolean {
  return props.pointMode ?? props.vertexCount >= GPU_GRAPH_DECK_POINT_VERTEX_COUNT;
}

function getBoundBuffers(props: GPUGraphNodeLayerProps, placeholder: Buffer): NodeBindingKey {
  return {
    colorColumn: props.colorColumn?.buffer ?? placeholder,
    sizeColumn: props.sizeColumn?.buffer ?? placeholder,
    filterMask: props.filterMask ?? placeholder,
    highlightMask: props.highlightMask ?? placeholder,
    pathRanks: props.pathRanks ?? placeholder
  };
}

function writeStyleIfChanged(state: GPUGraphNodeLayerState, packed: Uint32Array): void {
  const written = state.writtenStyle;
  if (written && written.every((word, index) => word === packed[index])) return;
  state.styleUniforms?.write(packed);
  state.writtenStyle = packed;
  state.renderStats.styleUniformWriteCount++;
}

/** Packs every style-affecting prop; byte equality is the "inputs changed" test. */
function packNodeStyle(props: GPUGraphNodeLayerProps): Uint32Array {
  const words = new Uint32Array(NODE_STYLE_WORD_COUNT);
  const floats = new Float32Array(words.buffer);
  const colorScale = props.colorScale ?? {
    type: 'categorical',
    palette: GPU_GRAPH_DEFAULT_COLOR_PALETTE
  };
  const palette = colorScale.palette.slice(0, GPU_GRAPH_MAXIMUM_PALETTE_LENGTH);
  const dimUnhighlighted =
    props.dimUnhighlighted ?? Boolean(props.highlightMask || props.pathRanks);
  const highlightColor = Array.isArray(props.highlightColor)
    ? (props.highlightColor as GPUGraphColor)
    : null;
  const flags =
    (props.colorColumn ? FLAG_COLOR_COLUMN : 0) |
    (props.sizeColumn ? FLAG_SIZE_COLUMN : 0) |
    (props.filterMask ? FLAG_FILTER_MASK : 0) |
    (props.highlightMask ? FLAG_HIGHLIGHT_MASK : 0) |
    (props.pathRanks ? FLAG_PATH_RANKS : 0) |
    (highlightColor ? FLAG_HIGHLIGHT_TINT : 0) |
    (props.pathColor ? FLAG_PATH_TINT : 0);

  floats[0] =
    props.radiusPixels ??
    Math.max(1.4, Math.min(6, 60 / Math.sqrt(Math.max(props.vertexCount, 1))));
  floats[1] = props.opacity ?? 1;
  words[2] = Number(getPointMode(props));
  words[3] = flags;
  floats.set(colorScale.domain ?? [0, 1], 4);
  floats.set(props.sizeScale?.domain ?? [0, 1], 6);
  floats.set(props.sizeScale?.range ?? [2, 8], 8);
  words[10] = colorScale.type === 'categorical' ? 1 : 0;
  words[11] = Math.max(palette.length, 1);
  words[12] = props.colorColumn?.format === 'float32' ? 1 : 0;
  words[13] = props.sizeColumn?.format === 'float32' ? 1 : 0;
  words[14] = Number(dimUnhighlighted);
  setColor(floats, 16, colorScale.nullColor ?? DEFAULT_NULL_COLOR);
  setColor(floats, 20, highlightColor ?? [255, 214, 97]);
  setColor(floats, 24, props.pathColor ?? [255, 214, 97]);
  const stops = palette.length > 0 ? palette : GPU_GRAPH_DEFAULT_COLOR_PALETTE.slice(0, 1);
  stops.forEach((color, index) => setColor(floats, 28 + index * 4, color));
  return words;
}

function setColor(target: Float32Array, offset: number, color: GPUGraphColor): void {
  target[offset] = color[0] / 255;
  target[offset + 1] = color[1] / 255;
  target[offset + 2] = color[2] / 255;
  target[offset + 3] = 1;
}
