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
