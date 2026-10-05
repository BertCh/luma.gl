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
import {GPU_POINT_DENSITY_HEXAGON_WGSL} from '@luma.gl/experimental/geospatial';

/** Segments each arc is tessellated into. The indirect draw record uses `ARC_SEGMENTS * 6` vertices. */
export const ARC_SEGMENTS = 24;

const STYLE_BYTE_LENGTH = 80;
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
};

const ARC_SEGMENTS: u32 = ${ARC_SEGMENTS}u;
const NO_ZONE: u32 = 0xffffffffu;
${GPU_POINT_DENSITY_HEXAGON_WGSL}

@group(0) @binding(auto) var<uniform> flowStyle: FlowArcStyle;
@group(0) @binding(auto) var<storage, read> flowOriginZoneIds: array<u32>;
@group(0) @binding(auto) var<storage, read> flowDestinationZoneIds: array<u32>;
@group(0) @binding(auto) var<storage, read> flowWeights: array<f32>;
@group(0) @binding(auto) var<storage, read> flowCount: array<u32>;

struct FlowVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) side: f32,
  @location(1) color: vec4<f32>,
};

// Planar center of a zone: grid cell center, or odd-r hexagon center (same lattice as the recipe).
fn getFlowZoneCenter(zone: u32) -> vec2<f32> {
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

// Quadratic Bezier from a to b whose control point sits to the right of the travel direction, so a
// flow and its reverse bulge to opposite sides and direction reads as a clockwise sweep.
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

  // Heaviest flows are rows 0.., so draw them last (on top): instance i is row count - 1 - i.
  let rowCount = flowCount[0];
  if (drawIndex >= rowCount) {
    return output;
  }
  let instanceIndex = rowCount - 1u - drawIndex;
  let originZone = flowOriginZoneIds[instanceIndex];
  let destinationZone = flowDestinationZoneIds[instanceIndex];
  if (originZone == NO_ZONE || destinationZone == NO_ZONE || originZone == destinationZone) {
    return output;
  }
  let a = getFlowZoneCenter(originZone);
  let b = getFlowZoneCenter(destinationZone);

  // Per quad: (position along the segment, side across the line).
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

  // flowWeights is sorted descending, so row 0 is the maximum and nothing is read back.
  let maximumWeight = max(flowWeights[0], 1e-20);
  let strength = sqrt(clamp(flowWeights[instanceIndex] / maximumWeight, 0.0, 1.0));
  let widthPixels = mix(flowStyle.widthMinPixels, flowStyle.widthMaxPixels, strength);

  output.position = vec4<f32>(
    clipCenter.xy + project_pixel_size_to_clipspace(normal * corner.y * widthPixels * 0.5),
    clipCenter.z,
    clipCenter.w
  );
  output.side = corner.y;
  var color = mix(flowStyle.originColor, flowStyle.destinationColor, t);
  color.a = color.a * mix(0.25, 0.95, strength) * flowStyle.opacity;
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
  /** uint32 origin zone per flow, `0xffffffff` for unused rows (recipe `flowOriginZoneIds`). */
  flowOriginZoneIds: Buffer;
  /** uint32 destination zone per flow (recipe `flowDestinationZoneIds`). */
  flowDestinationZoneIds: Buffer;
  /** float32 weight per flow, sorted descending (recipe `flowWeights`). Row 0 is the maximum. */
  flowWeights: Buffer;
  /** uint32 number of valid flow rows (recipe `output.count`); used to draw heavy flows last. */
  flowCount: Buffer;
  /** GPU-written indirect record whose `vertexCount` is `ARC_SEGMENTS * 6`. */
  drawCommands: DrawCommandBuffer;
  /** Zone lattice kind. */
  zoneKind: 'grid' | 'hexagon';
  /** `[columns, rows]` of the zone lattice. */
  gridSize: readonly [number, number];
  /** Zone lattice `[minX, minY, maxX, maxY]` in planar meters. */
  bounds: readonly [number, number, number, number];
  /** Hexagon center-to-vertex radius in meters. Ignored for grid zones. */
  hexagonRadius?: number;
  /** Narrowest arc width in CSS pixels. Defaults to 1. */
  widthMinPixels?: number;
  /** Widest arc width (the largest flow) in CSS pixels. Defaults to 8. */
  widthMaxPixels?: number;
  /** Origin end color, RGBA 0-255. */
  originColor?: readonly [number, number, number, number];
  /** Destination end color, RGBA 0-255. */
  destinationColor?: readonly [number, number, number, number];
  /** Control point offset as a fraction of the zone distance. Defaults to 0.13. */
  bulge?: number;
};

type FlowArcLayerState = {
  model: Model;
  styleBuffer: Buffer;
};

/**
 * Draws one curved, direction-colored arc per aggregated flow. The vertex shader reads the
 * recipe's top-K zone IDs and weights straight from storage buffers, converts zone IDs to lattice
 * centers, and tessellates a planar quadratic Bezier into screen-space-width quads. The instance
 * count comes from a GPU-written indirect draw record.
 */
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
      byteLength: STYLE_BYTE_LENGTH,
      usage: Buffer.UNIFORM | Buffer.COPY_DST
    });
    const model = new Model(device, {
      ...this.getShaders({modules: [project32], source: FLOW_ARC_SHADER}),
      id: `${this.id}-model`,
      topology: 'triangle-list',
      isInstanced: true,
      vertexCount: ARC_SEGMENTS * 6,
      instanceCount: 0,
      bufferLayout: [],
      bindings: this.getBindings(styleBuffer),
      parameters: BLEND_PARAMETERS
    });
    this.setState({model, styleBuffer});
  }

  override updateState(parameters: UpdateParameters<this>): void {
    super.updateState(parameters);
    const {model, styleBuffer} = this.state as FlowArcLayerState;
    model.setBindings(this.getBindings(styleBuffer));
  }

  override getModels(): Model[] {
    return [(this.state as FlowArcLayerState).model];
  }

  override draw({renderPass}: {renderPass: RenderPass}): void {
    const {model, styleBuffer} = this.state as FlowArcLayerState;
    this.writeStyle(styleBuffer);
    // Bind pipeline and bindings with an empty draw, then replay the GPU-written record.
    model.setInstanceCount(0);
    model.draw(renderPass);
    this.props.drawCommands.draw(renderPass, 0);
  }

  override finalizeState(context: LayerContext): void {
    const {model, styleBuffer} = this.state as FlowArcLayerState;
    model.destroy();
    styleBuffer.destroy();
    super.finalizeState(context);
  }

  private getBindings(styleBuffer: Buffer): Record<string, Buffer> {
    return {
      flowStyle: styleBuffer,
      flowOriginZoneIds: this.props.flowOriginZoneIds,
      flowDestinationZoneIds: this.props.flowDestinationZoneIds,
      flowWeights: this.props.flowWeights,
      flowCount: this.props.flowCount
    };
  }

  private writeStyle(styleBuffer: Buffer): void {
    const props = this.props;
    const data = new ArrayBuffer(STYLE_BYTE_LENGTH);
    const floats = new Float32Array(data);
    const words = new Uint32Array(data);
    const originColor = props.originColor ?? [255, 150, 60, 255];
    const destinationColor = props.destinationColor ?? [60, 220, 255, 255];
    for (let channel = 0; channel < 4; channel++) {
      floats[channel] = originColor[channel] / 255;
      floats[4 + channel] = destinationColor[channel] / 255;
    }
    floats.set(props.bounds, 8);
    words.set(props.gridSize, 12);
    words[14] = props.zoneKind === 'hexagon' ? 1 : 0;
    floats[15] = props.hexagonRadius ?? 1;
    floats[16] = props.widthMinPixels ?? 1;
    floats[17] = props.widthMaxPixels ?? 8;
    floats[18] = props.bulge ?? 0.13;
    floats[19] = props.opacity ?? 1;
    styleBuffer.write(new Uint8Array(data));
  }
}
