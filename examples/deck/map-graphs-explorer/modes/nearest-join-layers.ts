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

const NO_FEATURE = 0xffffffff;
const STYLE_BYTE_LENGTH = 32;
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

const SNAP_SHADER = /* wgsl */ `
struct SnapStyle {
  color: vec4<f32>,
  widthPixels: f32,
  _padding0: f32,
  _padding1: f32,
  _padding2: f32,
};

@group(0) @binding(auto) var<uniform> snapStyle: SnapStyle;
@group(0) @binding(auto) var<storage, read> snapPoints: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> snapFeatureIds: array<u32>;
@group(0) @binding(auto) var<storage, read> snapSegments: array<vec4<f32>>;

struct SnapVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) side: f32,
};

// Closest point to \`point\` on the segment from \`start\` to \`end\`.
fn getClosestSegmentPoint(point: vec2<f32>, start: vec2<f32>, end: vec2<f32>) -> vec2<f32> {
  let edge = end - start;
  let lengthSquared = dot(edge, edge);
  var t = 0.0;
  if (lengthSquared > 0.0) {
    t = clamp(dot(point - start, edge) / lengthSquared, 0.0, 1.0);
  }
  return start + edge * t;
}

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> SnapVertexOutput {
  // (t from the point to its snapped location, side across the line)
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
  );
  var output: SnapVertexOutput;
  let featureId = snapFeatureIds[instanceIndex];
  if (featureId == ${NO_FEATURE}u) {
    output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    output.side = 0.0;
    return output;
  }
  let point = snapPoints[instanceIndex];
  let segment = snapSegments[featureId];
  let snapped = getClosestSegmentPoint(point, segment.xy, segment.zw);
  var startClip = project_position_to_clipspace(vec3<f32>(point, 0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  var endClip = project_position_to_clipspace(vec3<f32>(snapped, 0.0), vec3<f32>(0.0), vec3<f32>(0.0));
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
    clipPosition.xy + project_pixel_size_to_clipspace(normal * corner.y * snapStyle.widthPixels * 0.5),
    clipPosition.z,
    clipPosition.w
  );
  output.position = clipPosition;
  output.side = corner.y;
  return output;
}

@fragment fn fragmentMain(input: SnapVertexOutput) -> @location(0) vec4<f32> {
  let coverage = 1.0 - smoothstep(0.55, 1.0, abs(input.side));
  return vec4<f32>(snapStyle.color.rgb, snapStyle.color.a * coverage);
}
`;

/** Props for {@link NearestSnapLayer}. */
export type NearestSnapLayerProps = LayerProps & {
  /** `float32x2` point positions in planar meters. */
  positions: Buffer;
  /** uint32 nearest segment row per point, or `0xffffffff` for none. */
  nearestFeatureIds: Buffer;
  /** `float32x4` road segments `x0, y0, x1, y1` in planar meters. */
  segments: Buffer;
  /** Number of points. */
  instanceCount: number;
  /** RGBA, 0-255. */
  color?: readonly [number, number, number, number];
  /** Line width in CSS pixels. Defaults to 1.5. */
  widthPixels?: number;
};

/**
 * Draws a line from every point to the closest location on its nearest segment. The vertex shader
 * reads the join output and the segment geometry directly, so nothing is read back.
 */
export class NearestSnapLayer extends Layer<NearestSnapLayerProps> {
  static override layerName = 'NearestSnapLayer';
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
      ...this.getShaders({modules: [project32], source: SNAP_SHADER}),
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
    const color = this.props.color ?? [255, 255, 255, 220];
    styleBuffer.write(
      Float32Array.of(
        color[0] / 255,
        color[1] / 255,
        color[2] / 255,
        color[3] / 255,
        this.props.widthPixels ?? 1.5,
        0,
        0,
        0
      )
    );
    model.setInstanceCount(this.props.instanceCount);
    model.draw(renderPass);
  }

  override finalizeState(context: LayerContext): void {
    const {model, styleBuffer} = this.state as {model: Model; styleBuffer: Buffer};
    model.destroy();
    styleBuffer.destroy();
    super.finalizeState(context);
  }

  private getBindings(styleBuffer: Buffer): Record<string, Buffer> {
    return {
      snapStyle: styleBuffer,
      snapPoints: this.props.positions,
      snapFeatureIds: this.props.nearestFeatureIds,
      snapSegments: this.props.segments
    };
  }
}
