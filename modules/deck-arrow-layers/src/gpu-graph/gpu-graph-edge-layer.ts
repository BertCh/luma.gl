// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  Layer,
  project32,
  type LayerContext,
  type LayerProps,
  type UpdateParameters
} from '@deck.gl/core';
import {Buffer, type RenderPass} from '@luma.gl/core';
import {Model} from '@luma.gl/engine';
import type {GPUGraphColor} from './gpu-graph-columns';
import type {GPUGraphLayerRenderStats} from './gpu-graph-node-layer';

/** Edge style uniform block size in bytes: 4 scalar words and 3 colors. */
const EDGE_STYLE_BYTE_LENGTH = 64;
const EDGE_STYLE_WORD_COUNT = EDGE_STYLE_BYTE_LENGTH / 4;

const FLAG_FILTER_MASK = 1;
const FLAG_HIGHLIGHT_MASK = 2;
const FLAG_PATH_RANKS = 4;

const DEFAULT_EDGE_COLOR: GPUGraphColor = [82, 130, 179];
const DEFAULT_EDGE_HIGHLIGHT_COLOR: GPUGraphColor = [242, 184, 82];
const DEFAULT_EDGE_PATH_COLOR: GPUGraphColor = [255, 214, 97];

/**
 * One original aligned source/target GPU chunk and the shared resident graph state. Node-row
 * inputs (`filterMask`, `highlightMask`, `pathRanks`) are indexed by the edge endpoints.
 */
export type GPUGraphEdgeLayerProps = LayerProps & {
  /** `float32x2` node positions shared with the node layer, bound as storage. */
  positions: Buffer;
  /** `u32` source node row per edge. */
  sourceVertices: Buffer;
  /** `u32` target node row per edge. */
  targetVertices: Buffer;
  /** Number of edges in this chunk. */
  edgeCount: number;
  /** `u32` per node. An edge is drawn only if both endpoints are nonzero. */
  filterMask?: Buffer;
  /** `u32` per node. An edge is highlighted if both endpoints are nonzero. */
  highlightMask?: Buffer;
  /** `u32` per node (0: off path, k: 1-based rank). On path if both are nonzero and differ by 1. */
  pathRanks?: Buffer;
  /** Base edge color. Default `[82, 130, 179]`. */
  color?: GPUGraphColor;
  /** Color of highlighted edges. Default `[242, 184, 82]`. */
  highlightColor?: GPUGraphColor;
  /** Color of on-path edges, which win over highlighted ones. Default `[255, 214, 97]`. */
  pathColor?: GPUGraphColor;
};

type EdgeBindingKey = {
  positions: Buffer;
  sourceVertices: Buffer;
  targetVertices: Buffer;
  filterMask: Buffer;
  highlightMask: Buffer;
  pathRanks: Buffer;
};

type GPUGraphEdgeLayerState = {
  model: Model | null;
  styleUniforms: Buffer | null;
  placeholder: Buffer | null;
  writtenStyle: Uint32Array | null;
  boundBuffers: EdgeBindingKey | null;
  renderStats: GPUGraphLayerRenderStats;
};

const EDGE_BLEND_PARAMETERS = {
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
 * Chunk-local source/target storage directly addresses the live shared layout positions.
 *
 * Storage buffers read in the vertex stage: positions, sourceVertices, targetVertices,
 * filterMask, highlightMask, pathRanks (6 of the WebGPU default limit of 8). Absent masks bind a
 * 16-byte placeholder and clear a flag in `edgeStyle`.
 */
export const GPU_GRAPH_DECK_EDGE_SHADER = /* wgsl */ `
struct EdgeStyle {
  opacity: f32,
  flags: u32,
  _padding1: u32,
  _padding2: u32,
  color: vec4<f32>,
  highlightColor: vec4<f32>,
  pathColor: vec4<f32>,
};

@group(0) @binding(auto) var<storage, read> positions: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> sourceVertices: array<u32>;
@group(0) @binding(auto) var<storage, read> targetVertices: array<u32>;
@group(0) @binding(auto) var<storage, read> filterMask: array<u32>;
@group(0) @binding(auto) var<storage, read> highlightMask: array<u32>;
@group(0) @binding(auto) var<storage, read> pathRanks: array<u32>;
@group(0) @binding(auto) var<uniform> edgeStyle: EdgeStyle;

struct EdgeVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
};

fn hasFlag(flag: u32) -> bool {
  return (edgeStyle.flags & flag) != 0u;
}

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> EdgeVertexOutput {
  var output: EdgeVertexOutput;
  let sourceVertex = sourceVertices[instanceIndex];
  let targetVertex = targetVertices[instanceIndex];
  if (hasFlag(${FLAG_FILTER_MASK}u) &&
      (filterMask[sourceVertex] == 0u || filterMask[targetVertex] == 0u)) {
    output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    return output;
  }
  let vertex = select(sourceVertex, targetVertex, vertexIndex != 0u);
  let position = positions[vertex];
  let isHighlighted = hasFlag(${FLAG_HIGHLIGHT_MASK}u) &&
    highlightMask[sourceVertex] != 0u && highlightMask[targetVertex] != 0u;
  var isOnPath = false;
  if (hasFlag(${FLAG_PATH_RANKS}u)) {
    let sourceRank = pathRanks[sourceVertex];
    let targetRank = pathRanks[targetVertex];
    isOnPath = sourceRank != 0u && targetRank != 0u &&
      (max(sourceRank, targetRank) - min(sourceRank, targetRank)) == 1u;
  }

  geometry.worldPosition = vec3<f32>(position, 0.0);
  var clipPosition = project_position_to_clipspace(
    vec3<f32>(position, 0.0),
    vec3<f32>(0.0),
    vec3<f32>(0.0)
  );
  // Deck's OpenGL-style projection depth must be converted for WebGPU clipping.
  clipPosition.z = (clipPosition.z + clipPosition.w) * 0.5;
  output.position = clipPosition;
  output.color = select(select(edgeStyle.color, edgeStyle.highlightColor, isHighlighted),
    edgeStyle.pathColor, isOnPath);
  return output;
}

@fragment fn fragmentMain(input: EdgeVertexOutput) -> @location(0) vec4<f32> {
  return vec4<f32>(input.color.rgb, input.color.a * edgeStyle.opacity);
}`;

/**
 * Exactly one deck layer per nonempty original edge chunk; no implicit edge packing occurs.
 *
 * @remarks
 * WebGPU only, by design. Node masks and positions are storage buffers read in the vertex stage,
 * which WebGL2 cannot bind. A WebGL2 device throws; render CPU-side edges with a standard
 * deck.gl layer instead.
 *
 * Style uniforms are uploaded only when their packed bytes change and buffers are rebound only
 * when an identity or presence changes; neither happens in `draw()`.
 * See {@link GPUGraphEdgeLayer.getRenderStats}.
 */
export class GPUGraphEdgeLayer extends Layer<GPUGraphEdgeLayerProps> {
  static override layerName = 'GPUGraphEdgeLayer';
  // Deck's default highlightColor ([0, 0, 128, 128]) must not shadow DEFAULT_EDGE_HIGHLIGHT_COLOR.
  static override defaultProps = {
    parameters: EDGE_BLEND_PARAMETERS,
    highlightColor: null
  };

  override getAttributeManager() {
    return null;
  }

  /** Reports the original GPUData chunk population instead of the empty placeholder array. */
  override getNumInstances(): number {
    return this.props.edgeCount;
  }

  /**
   * Returns draw, style-upload and rebinding counters. They live in layer state, so they survive
   * Deck re-instantiating the layer with the same id.
   */
  getRenderStats(): GPUGraphLayerRenderStats {
    const state = (this.state ?? {}) as Partial<GPUGraphEdgeLayerState>;
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
        'GPUGraphEdgeLayer requires WebGPU: node masks and positions are storage buffers read in the vertex stage, which WebGL2 cannot bind; render CPU-side edges with a standard deck.gl layer instead.'
      );
    }
    const styleUniforms = device.createBuffer({
      id: `${this.id}-style-uniforms`,
      byteLength: EDGE_STYLE_BYTE_LENGTH,
      usage: Buffer.UNIFORM | Buffer.COPY_DST
    });
    const placeholder = device.createBuffer({
      id: `${this.id}-absent-mask`,
      data: new Uint32Array(4),
      usage: Buffer.STORAGE | Buffer.COPY_DST
    });
    const boundBuffers = getBoundBuffers(this.props, placeholder);
    const model = new Model(device, {
      ...this.getShaders({
        modules: [project32],
        source: GPU_GRAPH_DECK_EDGE_SHADER
      }),
      id: `${this.id}-model`,
      topology: 'line-list',
      isInstanced: true,
      vertexCount: 2,
      instanceCount: this.props.edgeCount,
      bufferLayout: [],
      bindings: {...boundBuffers, edgeStyle: styleUniforms},
      parameters: EDGE_BLEND_PARAMETERS
    });
    const state: GPUGraphEdgeLayerState = {
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
    writeStyleIfChanged(this.state as GPUGraphEdgeLayerState, packEdgeStyle(this.props));
  }

  /** Rebinds replaced buffers and rewrites the style block only when its packed bytes change. */
  override updateState({props}: UpdateParameters<this>): void {
    const state = this.state as GPUGraphEdgeLayerState;
    const {model, styleUniforms, placeholder} = state;
    if (!model || !styleUniforms || !placeholder) return;
    const boundBuffers = getBoundBuffers(props, placeholder);
    const previous = state.boundBuffers;
    if (
      !previous ||
      (Object.keys(boundBuffers) as (keyof EdgeBindingKey)[]).some(
        key => boundBuffers[key] !== previous[key]
      )
    ) {
      model.setBindings({...boundBuffers, edgeStyle: styleUniforms});
      state.boundBuffers = boundBuffers;
      state.renderStats.bindingUpdateCount++;
    }
    writeStyleIfChanged(state, packEdgeStyle(props));
    model.setInstanceCount(props.edgeCount);
  }

  override getModels(): Model[] {
    const model = (this.state as Partial<GPUGraphEdgeLayerState> | undefined)?.model;
    return model ? [model] : [];
  }

  override draw({renderPass}: {renderPass: RenderPass}): void {
    const {model, renderStats} = this.state as GPUGraphEdgeLayerState;
    if (!model) return;
    renderStats.drawCount++;
    model.draw(renderPass);
  }

  override finalizeState(context: LayerContext): void {
    const state = this.state as GPUGraphEdgeLayerState;
    state.model?.destroy();
    state.styleUniforms?.destroy();
    state.placeholder?.destroy();
    this.setState({
      model: null,
      styleUniforms: null,
      placeholder: null
    } satisfies Partial<GPUGraphEdgeLayerState>);
    super.finalizeState(context);
  }
}

function getBoundBuffers(props: GPUGraphEdgeLayerProps, placeholder: Buffer): EdgeBindingKey {
  return {
    positions: props.positions,
    sourceVertices: props.sourceVertices,
    targetVertices: props.targetVertices,
    filterMask: props.filterMask ?? placeholder,
    highlightMask: props.highlightMask ?? placeholder,
    pathRanks: props.pathRanks ?? placeholder
  };
}

function writeStyleIfChanged(state: GPUGraphEdgeLayerState, packed: Uint32Array): void {
  const written = state.writtenStyle;
  if (written && written.every((word, index) => word === packed[index])) return;
  state.styleUniforms?.write(packed);
  state.writtenStyle = packed;
  state.renderStats.styleUniformWriteCount++;
}

/** Packs every style-affecting prop; byte equality is the "inputs changed" test. */
function packEdgeStyle(props: GPUGraphEdgeLayerProps): Uint32Array {
  const words = new Uint32Array(EDGE_STYLE_WORD_COUNT);
  const floats = new Float32Array(words.buffer);
  const dimUnhighlighted = Boolean(props.highlightMask || props.pathRanks);
  floats[0] = props.opacity ?? 1;
  words[1] =
    (props.filterMask ? FLAG_FILTER_MASK : 0) |
    (props.highlightMask ? FLAG_HIGHLIGHT_MASK : 0) |
    (props.pathRanks ? FLAG_PATH_RANKS : 0);
  setColor(floats, 4, props.color ?? DEFAULT_EDGE_COLOR, dimUnhighlighted ? 0.07 : 0.2);
  const highlightColor = Array.isArray(props.highlightColor)
    ? (props.highlightColor as GPUGraphColor)
    : DEFAULT_EDGE_HIGHLIGHT_COLOR;
  setColor(floats, 8, highlightColor, 0.82);
  setColor(floats, 12, props.pathColor ?? DEFAULT_EDGE_PATH_COLOR, 0.95);
  return words;
}

function setColor(target: Float32Array, offset: number, color: GPUGraphColor, alpha: number): void {
  target[offset] = color[0] / 255;
  target[offset + 1] = color[1] / 255;
  target[offset + 2] = color[2] / 255;
  target[offset + 3] = alpha;
}
