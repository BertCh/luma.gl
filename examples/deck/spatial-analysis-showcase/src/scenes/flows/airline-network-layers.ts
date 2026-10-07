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
import {NETWORK_PALETTE} from './airline-network-palette';

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

const STYLE_BYTES = 160;

const SHADER = /* wgsl */ `
struct CategoryPathStyle {
  palette: array<vec4<f32>, 8>,
  pointsPerPath: u32,
  opacity: f32,
  padding0: u32,
  padding1: u32,
};

@group(0) @binding(auto) var<uniform> pathStyle: CategoryPathStyle;
@group(0) @binding(auto) var<storage, read> paths: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> pathCategories: array<f32>;

struct PathVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
};

@vertex fn vertexMain(
  @builtin(vertex_index) pointIndex: u32,
  @builtin(instance_index) pathIndex: u32
) -> PathVertexOutput {
  var output: PathVertexOutput;
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  output.color = vec4<f32>(0.0);
  // NaN (a filtered-out route) hides the whole path.
  let category = pathCategories[pathIndex];
  if (category != category) {
    return output;
  }
  let position = paths[pathIndex * pathStyle.pointsPerPath + pointIndex];
  var clipPosition = project_position_to_clipspace(vec3<f32>(position, 0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  // Deck's project32 matrices use OpenGL depth; WebGPU clip space requires [0, w].
  clipPosition.z = (clipPosition.z + clipPosition.w) * 0.5;
  output.position = clipPosition;
  var rgba = pathStyle.palette[u32(max(category, 0.0)) % 8u];
  rgba.a = rgba.a * pathStyle.opacity;
  output.color = rgba;
  return output;
}

@fragment fn fragmentMain(input: PathVertexOutput) -> @location(0) vec4<f32> {
  return input.color;
}
`;

/** Props of {@link CategoryPathLayer}. */
export type CategoryPathLayerProps = LayerProps & {
  /** `float32x2` `[longitude, latitude]` polyline rows, edge-major (the `GPUEdgeBundling` `paths`). */
  paths: Buffer;
  /** Control points per path. */
  pointsPerPath: number;
  /** Number of paths. */
  pathCount: number;
  /** One float32 palette index per path; NaN hides the path. */
  categories: Buffer;
  /** Up to 8 palette colors, RGBA 0-255. Defaults to {@link NETWORK_PALETTE}. */
  palette?: readonly (readonly [number, number, number, number])[];
};

type ModelState = {model: Model; styleBuffer: Buffer};

/**
 * Bundled polylines drawn as 1 px line strips straight from the contributor's `paths` buffer,
 * colored by a per-path category (for example the community of the route).
 */
export class CategoryPathLayer extends Layer<CategoryPathLayerProps> {
  static override layerName = 'AirlineNetworkCategoryPathLayer';
  static override defaultProps = {parameters: BLEND_PARAMETERS};

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
      ...this.getShaders({modules: [project32], source: SHADER}),
      id: `${this.id}-model`,
      topology: 'line-strip',
      isInstanced: true,
      vertexCount: this.props.pointsPerPath,
      instanceCount: this.props.pathCount,
      bufferLayout: [],
      bindings: this.getBindings(styleBuffer),
      parameters: BLEND_PARAMETERS
    });
    this.setState({model, styleBuffer});
  }

  override updateState(parameters: UpdateParameters<this>): void {
    super.updateState(parameters);
    const {model, styleBuffer} = this.state as ModelState;
    model.setBindings(this.getBindings(styleBuffer));
    model.setInstanceCount(this.props.pathCount);
  }

  override getModels(): Model[] {
    return [(this.state as ModelState).model];
  }

  override draw({renderPass}: {renderPass: RenderPass}): void {
    const {model, styleBuffer} = this.state as ModelState;
    const data = new ArrayBuffer(STYLE_BYTES);
    const floats = new Float32Array(data);
    const words = new Uint32Array(data);
    const palette = this.props.palette ?? NETWORK_PALETTE;
    for (let index = 0; index < 8; index++) {
      const color = palette[index % palette.length];
      for (let channel = 0; channel < 4; channel++)
        floats[index * 4 + channel] = color[channel] / 255;
    }
    words[32] = this.props.pointsPerPath;
    floats[33] = this.props.opacity ?? 1;
    styleBuffer.write(new Uint8Array(data));
    model.draw(renderPass);
  }

  override finalizeState(context: LayerContext): void {
    const {model, styleBuffer} = this.state as ModelState;
    model.destroy();
    styleBuffer.destroy();
    super.finalizeState(context);
  }

  private getBindings(styleBuffer: Buffer): Record<string, Buffer> {
    return {
      pathStyle: styleBuffer,
      paths: this.props.paths,
      pathCategories: this.props.categories
    };
  }
}
