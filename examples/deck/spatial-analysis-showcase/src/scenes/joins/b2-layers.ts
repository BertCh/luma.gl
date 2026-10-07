// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Layers local to the "joins" scenes.
 *
 * - {@link ZoneFillLayer}: fills a zone raster (cell to feature row, from `GPUPolygonRasterization`)
 *   with a per-feature value through the shared ramp table. Cells that hold no zone are hidden,
 *   which the shared raster layer cannot do itself (`0xffffffff` would index past the values).
 * - {@link PackedZoneFillLayer}: the same lookup for packed `rgba8` colors per feature, the output of
 *   `GPUColorScale` in the choropleth recipe.
 * - {@link LinkLayer}: a line from every query point to its matched feature, read straight from the
 *   join output (feature ids or foot points), so nothing is read back.
 */

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
import {SpatialAnalysisRasterLayer} from '../../engine/layers';

const STYLE_COLOR_FUNCTION = 'fn getSpatialAnalysisColor(valueRow: u32) -> vec4<f32> {';
const NO_ZONE = 0xffffffff;

/** Raster fill that looks a per-feature value up through a zone raster (`valueIndices`). */
export class ZoneFillLayer extends SpatialAnalysisRasterLayer {
  static override layerName = 'ZoneFillLayer';

  protected override getShaderSource(): string {
    const source = super.getShaderSource();
    if (!source.includes(STYLE_COLOR_FUNCTION)) {
      throw new Error('ZoneFillLayer: the shared raster shader changed');
    }
    return `${source.replace(STYLE_COLOR_FUNCTION, 'fn getSpatialAnalysisStyleColor(valueRow: u32) -> vec4<f32> {')}
fn getSpatialAnalysisColor(zone: u32) -> vec4<f32> {
  if (zone == ${NO_ZONE}u) { return vec4<f32>(0.0); }
  return getSpatialAnalysisStyleColor(zone);
}
`;
  }
}

/** Raster fill whose per-feature `values` are packed `rgba8` colors (alpha 0 hides the feature). */
export class PackedZoneFillLayer extends SpatialAnalysisRasterLayer {
  static override layerName = 'PackedZoneFillLayer';

  protected override getShaderSource(): string {
    const source = super.getShaderSource();
    if (!source.includes(STYLE_COLOR_FUNCTION)) {
      throw new Error('PackedZoneFillLayer: the shared raster shader changed');
    }
    return `${source.replace(STYLE_COLOR_FUNCTION, 'fn getSpatialAnalysisStyleColor(valueRow: u32) -> vec4<f32> {')}
fn getSpatialAnalysisColor(zone: u32) -> vec4<f32> {
  if (zone == ${NO_ZONE}u) { return vec4<f32>(0.0); }
  return unpack4x8unorm(styleValues[zone]);
}
`;
  }
}

const LINK_STYLE_BYTE_LENGTH = 48;
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

const LINK_SHADER = /* wgsl */ `
struct LinkStyle {
  color: vec4<f32>,
  widthPixels: f32,
  slotCapacity: u32,
  slotCount: u32,
  firstSlot: u32,
  queryStride: u32,
  useTargetIds: u32,
  fade: f32,
  _padding: u32,
};

@group(0) @binding(auto) var<uniform> linkStyle: LinkStyle;
@group(0) @binding(auto) var<storage, read> linkQueries: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> linkTargets: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> linkTargetIds: array<u32>;

struct LinkVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) side: f32,
  @location(1) alpha: f32,
};

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> LinkVertexOutput {
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
  );
  var output: LinkVertexOutput;
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  output.side = 0.0;
  output.alpha = 0.0;
  let slotCount = max(linkStyle.slotCount, 1u);
  let query = (instanceIndex / slotCount) * max(linkStyle.queryStride, 1u);
  let slotOffset = instanceIndex % slotCount;
  let slot = linkStyle.firstSlot + slotOffset;
  let index = query * linkStyle.slotCapacity + slot;
  var endPoint: vec2<f32>;
  if (linkStyle.useTargetIds != 0u) {
    let id = linkTargetIds[index];
    if (id == ${NO_ZONE}u) { return output; }
    endPoint = linkTargets[id];
  } else {
    endPoint = linkTargets[index];
  }
  let point = linkQueries[query];
  if (endPoint.x != endPoint.x || endPoint.y != endPoint.y || point.x != point.x) { return output; }
  var startClip = project_position_to_clipspace(vec3<f32>(point, 0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  var endClip = project_position_to_clipspace(vec3<f32>(endPoint, 0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  // Deck's project32 matrices use OpenGL depth; WebGPU clip space requires [0, w].
  startClip.z = (startClip.z + startClip.w) * 0.5;
  endClip.z = (endClip.z + endClip.w) * 0.5;
  let corner = corners[vertexIndex];
  let screenDirection = (endClip.xy / endClip.w - startClip.xy / startClip.w) * project.viewportSize;
  let directionLength = length(screenDirection);
  var direction = vec2<f32>(1.0, 0.0);
  if (directionLength > 1e-6) {
    direction = screenDirection / directionLength;
  }
  let normal = vec2<f32>(-direction.y, direction.x);
  var clipPosition = mix(startClip, endClip, corner.x);
  clipPosition = vec4<f32>(
    clipPosition.xy + project_pixel_size_to_clipspace(normal * corner.y * linkStyle.widthPixels * 0.5),
    clipPosition.z,
    clipPosition.w
  );
  output.position = clipPosition;
  output.side = corner.y;
  output.alpha = linkStyle.color.a * (1.0 - linkStyle.fade * f32(slotOffset) / f32(slotCount));
  return output;
}

@fragment fn fragmentMain(input: LinkVertexOutput) -> @location(0) vec4<f32> {
  let coverage = 1.0 - smoothstep(0.55, 1.0, abs(input.side));
  return vec4<f32>(linkStyle.color.rgb, input.alpha * coverage);
}
`;

/** Props for {@link LinkLayer}. */
export type LinkLayerProps = LayerProps & {
  /** `float32x2` query positions in planar meters. */
  queries: Buffer;
  /**
   * Link end points. Either `float32x2` foot points laid out `query * slotCapacity + slot`, or,
   * with `targetIds`, the positions of the features those ids index.
   */
  targets: Buffer;
  /** Optional `uint32` feature ids laid out `query * slotCapacity + slot` (`0xffffffff` = none). */
  targetIds?: Buffer | null;
  /** Number of query rows in the buffers. */
  queryCount: number;
  /** Slots stored per query. Defaults to 1. */
  slotCapacity?: number;
  /** Number of slots drawn per query, starting at `firstSlot`. Defaults to 1. */
  slotCount?: number;
  /** First slot drawn (1 skips the self match of a self join). Defaults to 0. */
  firstSlot?: number;
  /** Draw every n-th query only, so dense inputs stay readable. Defaults to 1. */
  queryStride?: number;
  /** RGBA, 0-255. */
  color?: readonly [number, number, number, number];
  /** Line width in CSS pixels. Defaults to 1.2. */
  widthPixels?: number;
  /** Alpha falloff across the slots, 0 (none) to 1 (last slot invisible). Defaults to 0.5. */
  fade?: number;
};

/** Draws a line from each query to its matched feature straight from the join output. */
export class LinkLayer extends Layer<LinkLayerProps> {
  static override layerName = 'LinkLayer';
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
      byteLength: LINK_STYLE_BYTE_LENGTH,
      usage: Buffer.UNIFORM | Buffer.COPY_DST
    });
    const placeholderBuffer = device.createBuffer({
      id: `${this.id}-placeholder`,
      byteLength: 16,
      usage: Buffer.STORAGE | Buffer.COPY_DST
    });
    const model = new Model(device, {
      ...this.getShaders({modules: [project32], source: LINK_SHADER}),
      id: `${this.id}-model`,
      topology: 'triangle-list',
      isInstanced: true,
      vertexCount: 6,
      instanceCount: 0,
      bufferLayout: [],
      bindings: this.getBindings(styleBuffer, placeholderBuffer),
      parameters: BLEND_PARAMETERS
    });
    this.setState({model, styleBuffer, placeholderBuffer});
  }

  override updateState(parameters: UpdateParameters<this>): void {
    super.updateState(parameters);
    const {model, styleBuffer, placeholderBuffer} = this.state as LinkState;
    model.setBindings(this.getBindings(styleBuffer, placeholderBuffer));
  }

  override getModels(): Model[] {
    return [(this.state as LinkState).model];
  }

  override draw({renderPass}: {renderPass: RenderPass}): void {
    const {model, styleBuffer} = this.state as LinkState;
    const {
      color = [255, 255, 255, 200],
      widthPixels = 1.2,
      slotCapacity = 1,
      slotCount = 1,
      firstSlot = 0,
      queryStride = 1,
      fade = 0.5,
      queryCount,
      targetIds
    } = this.props;
    const data = new ArrayBuffer(LINK_STYLE_BYTE_LENGTH);
    const floats = new Float32Array(data);
    const words = new Uint32Array(data);
    floats.set([color[0] / 255, color[1] / 255, color[2] / 255, color[3] / 255], 0);
    floats[4] = widthPixels;
    words[5] = slotCapacity;
    words[6] = slotCount;
    words[7] = firstSlot;
    words[8] = queryStride;
    words[9] = targetIds ? 1 : 0;
    floats[10] = fade;
    styleBuffer.write(new Uint8Array(data));
    model.setInstanceCount(Math.ceil(queryCount / Math.max(queryStride, 1)) * slotCount);
    model.draw(renderPass);
  }

  override finalizeState(context: LayerContext): void {
    const {model, styleBuffer, placeholderBuffer} = this.state as LinkState;
    model.destroy();
    styleBuffer.destroy();
    placeholderBuffer.destroy();
    super.finalizeState(context);
  }

  private getBindings(styleBuffer: Buffer, placeholder: Buffer): Record<string, Buffer> {
    return {
      linkStyle: styleBuffer,
      linkQueries: this.props.queries,
      linkTargets: this.props.targets,
      linkTargetIds: this.props.targetIds ?? placeholder
    };
  }
}

type LinkState = {model: Model; styleBuffer: Buffer; placeholderBuffer: Buffer};
