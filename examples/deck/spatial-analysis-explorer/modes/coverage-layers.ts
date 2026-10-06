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

/** Uniform bytes every layer built on {@link StorageModelLayer} owns (room for a palette and config). */
export const STORAGE_MODEL_STYLE_BYTE_LENGTH = 160;
const STYLE_BYTE_LENGTH = STORAGE_MODEL_STYLE_BYTE_LENGTH;
const PALETTE_SIZE = 8;
export const STORAGE_MODEL_BLEND = {
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

const ZONE_FILL_SHADER = /* wgsl */ `
struct ZoneFillStyle {
  palette: array<vec4<f32>, ${PALETTE_SIZE}>,
  // opacity, unused, unused, unused
  numbers: vec4<f32>,
};

@group(0) @binding(auto) var<uniform> zoneFillStyle: ZoneFillStyle;
@group(0) @binding(auto) var<storage, read> triangleVertices: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> triangleOwners: array<u32>;
@group(0) @binding(auto) var<storage, read> zoneColors: array<u32>;

struct ZoneFillOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
};

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) triangle: u32
) -> ZoneFillOutput {
  var output: ZoneFillOutput;
  let colorId = zoneColors[triangleOwners[triangle]];
  if (colorId == 0xffffffffu) {
    output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    output.color = vec4<f32>(0.0);
    return output;
  }
  var clipPosition = project_position_to_clipspace(
    vec3<f32>(triangleVertices[triangle * 3u + vertexIndex], 0.0),
    vec3<f32>(0.0),
    vec3<f32>(0.0)
  );
  // Deck's project32 matrices use OpenGL depth; WebGPU clip space requires [0, w].
  clipPosition.z = (clipPosition.z + clipPosition.w) * 0.5;
  output.position = clipPosition;
  let color = zoneFillStyle.palette[colorId % ${PALETTE_SIZE}u];
  output.color = vec4<f32>(color.rgb, color.a * zoneFillStyle.numbers.x);
  return output;
}

@fragment fn fragmentMain(input: ZoneFillOutput) -> @location(0) vec4<f32> {
  return input.color;
}
`;

const RING_OUTLINE_SHADER = /* wgsl */ `
struct RingOutlineStyle {
  color: vec4<f32>,
  widthPixels: f32,
  _padding0: f32,
  _padding1: f32,
  _padding2: f32,
};

@group(0) @binding(auto) var<uniform> ringOutlineStyle: RingOutlineStyle;
@group(0) @binding(auto) var<storage, read> ringVertices: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> keepMask: array<u32>;
@group(0) @binding(auto) var<storage, read> ringOffsets: array<u32>;
@group(0) @binding(auto) var<storage, read> vertexRings: array<u32>;

struct RingOutlineOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) side: f32,
};

fn projectRingVertex(position: vec2<f32>) -> vec4<f32> {
  var clipPosition = project_position_to_clipspace(vec3<f32>(position, 0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  // Deck's project32 matrices use OpenGL depth; WebGPU clip space requires [0, w].
  clipPosition.z = (clipPosition.z + clipPosition.w) * 0.5;
  return clipPosition;
}

// Instance v is a kept vertex; its edge runs to the next kept vertex of the same ring (cyclic).
@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) vertex: u32
) -> RingOutlineOutput {
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
  );
  var output: RingOutlineOutput;
  output.side = 0.0;
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  if (keepMask[vertex] == 0u) {
    return output;
  }
  let ring = vertexRings[vertex];
  let start = ringOffsets[ring];
  let end = ringOffsets[ring + 1u];
  var next = vertex + 1u;
  var found = false;
  for (var step = 0u; step < end - start; step++) {
    if (next >= end) {
      next = start;
    }
    if (next == vertex) {
      break;
    }
    if (keepMask[next] != 0u) {
      found = true;
      break;
    }
    next++;
  }
  if (!found) {
    return output;
  }
  let startClip = projectRingVertex(ringVertices[vertex]);
  let endClip = projectRingVertex(ringVertices[next]);
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
    clipPosition.xy + project_pixel_size_to_clipspace((normal * corner.y + along) * ringOutlineStyle.widthPixels * 0.5),
    clipPosition.z,
    clipPosition.w
  );
  output.position = clipPosition;
  output.side = corner.y;
  return output;
}

@fragment fn fragmentMain(input: RingOutlineOutput) -> @location(0) vec4<f32> {
  let coverage = 1.0 - smoothstep(0.55, 1.0, abs(input.side));
  return vec4<f32>(ringOutlineStyle.color.rgb, ringOutlineStyle.color.a * coverage);
}
`;

type StorageLayerState = {model: Model; styleBuffer: Buffer};

/** Shared lifecycle: one instanced model, an owned style uniform, storage bindings from props. */
export abstract class StorageModelLayer<PropsT extends LayerProps> extends Layer<PropsT> {
  static override layerName = 'StorageModelLayer';
  static override defaultProps = {
    coordinateSystem: COORDINATE_SYSTEM.METER_OFFSETS,
    parameters: STORAGE_MODEL_BLEND
  };

  protected abstract getShaderSource(): string;
  protected abstract getVertexCount(): number;
  protected abstract getInstanceCount(): number;
  protected abstract getBindings(styleBuffer: Buffer): Record<string, Buffer>;
  protected abstract writeStyle(styleBuffer: Buffer): void;

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
      ...this.getShaders({modules: [project32], source: this.getShaderSource()}),
      id: `${this.id}-model`,
      topology: 'triangle-list',
      isInstanced: true,
      vertexCount: this.getVertexCount(),
      instanceCount: 0,
      bufferLayout: [],
      bindings: this.getBindings(styleBuffer),
      parameters: STORAGE_MODEL_BLEND
    });
    this.setState({model, styleBuffer} satisfies StorageLayerState);
  }

  override updateState(parameters: UpdateParameters<this>): void {
    super.updateState(parameters);
    const {model, styleBuffer} = this.state as StorageLayerState;
    model.setBindings(this.getBindings(styleBuffer));
  }

  override getModels(): Model[] {
    return [(this.state as StorageLayerState).model];
  }

  override draw({renderPass}: {renderPass: RenderPass}): void {
    const {model, styleBuffer} = this.state as StorageLayerState;
    this.writeStyle(styleBuffer);
    model.setInstanceCount(this.getInstanceCount());
    model.draw(renderPass);
  }

  override finalizeState(context: LayerContext): void {
    const {model, styleBuffer} = this.state as StorageLayerState;
    model.destroy();
    styleBuffer.destroy();
    super.finalizeState(context);
  }
}

/** Props for {@link ZoneFillLayer}. */
export type ZoneFillLayerProps = LayerProps & {
  /** `float32x2` triangle corners in planar meters, three per triangle. */
  triangleVertices: Buffer;
  /** uint32 polygon row of each triangle. */
  triangleOwners: Buffer;
  /** uint32 color ID per polygon (a `GPUMapColoring` output); `0xffffffff` hides the polygon. */
  zoneColors: Buffer;
  /** Number of triangles. */
  triangleCount: number;
  /** Colors indexed by `colorId % 8`, RGBA 0-255. */
  palette: readonly (readonly [number, number, number, number])[];
  /** Overall opacity multiplier. Defaults to 1. */
  fillOpacity?: number;
};

/**
 * Fills polygon triangles with the palette color of their polygon's GPU color ID. The triangles
 * are a one-time CPU triangulation of the static input; the colors are read from the contributor's
 * output buffer every frame, so a new seed recolors the map without any CPU work.
 */
export class ZoneFillLayer extends StorageModelLayer<ZoneFillLayerProps> {
  static override layerName = 'ZoneFillLayer';

  protected getShaderSource(): string {
    return ZONE_FILL_SHADER;
  }
  protected getVertexCount(): number {
    return 3;
  }
  protected getInstanceCount(): number {
    return this.props.triangleCount;
  }
  protected getBindings(styleBuffer: Buffer): Record<string, Buffer> {
    return {
      zoneFillStyle: styleBuffer,
      triangleVertices: this.props.triangleVertices,
      triangleOwners: this.props.triangleOwners,
      zoneColors: this.props.zoneColors
    };
  }
  protected writeStyle(styleBuffer: Buffer): void {
    const style = new Float32Array(STYLE_BYTE_LENGTH / 4);
    for (let index = 0; index < PALETTE_SIZE; index++) {
      const color = this.props.palette[index % this.props.palette.length];
      style.set([color[0] / 255, color[1] / 255, color[2] / 255, color[3] / 255], index * 4);
    }
    style[PALETTE_SIZE * 4] = this.props.fillOpacity ?? 1;
    styleBuffer.write(style);
  }
}

/** Props for {@link RingOutlineLayer}. */
export type RingOutlineLayerProps = LayerProps & {
  /** `float32x2` ring vertices in planar meters (the contributor input), one row per vertex. */
  vertices: Buffer;
  /** uint32 keep flag per vertex (a `keepMask` output). */
  keepMask: Buffer;
  /** uint32 `ringCount + 1` ring-to-vertex offsets. */
  ringOffsets: Buffer;
  /** uint32 ring index of every vertex. */
  vertexRings: Buffer;
  /** Number of vertices (instances). */
  vertexCount: number;
  /** Line width in CSS pixels. Defaults to 2. */
  widthPixels?: number;
  /** RGBA 0-255. Defaults to white. */
  color?: readonly [number, number, number, number];
};

/**
 * Draws simplified closed rings straight from a GPU keep mask: every kept vertex draws the edge to
 * the next kept vertex of its ring, so nothing is compacted or read back to draw the outline.
 */
export class RingOutlineLayer extends StorageModelLayer<RingOutlineLayerProps> {
  static override layerName = 'RingOutlineLayer';

  protected getShaderSource(): string {
    return RING_OUTLINE_SHADER;
  }
  protected getVertexCount(): number {
    return 6;
  }
  protected getInstanceCount(): number {
    return this.props.vertexCount;
  }
  protected getBindings(styleBuffer: Buffer): Record<string, Buffer> {
    return {
      ringOutlineStyle: styleBuffer,
      ringVertices: this.props.vertices,
      keepMask: this.props.keepMask,
      ringOffsets: this.props.ringOffsets,
      vertexRings: this.props.vertexRings
    };
  }
  protected writeStyle(styleBuffer: Buffer): void {
    const color = this.props.color ?? [255, 255, 255, 255];
    styleBuffer.write(
      Float32Array.of(
        color[0] / 255,
        color[1] / 255,
        color[2] / 255,
        color[3] / 255,
        this.props.widthPixels ?? 2,
        0,
        0,
        0
      )
    );
  }
}

/** One triangulated polygon set: flat corner coordinates and the polygon row of each triangle. */
export type TriangulatedPolygons = {
  /** `x, y` per corner, three corners per triangle. */
  corners: Float32Array;
  /** Polygon row of each triangle. */
  owners: Uint32Array;
};

/**
 * Triangulates polygons given as GeoArrow-style offsets by ear clipping each ring. Holes are
 * ignored (every ring becomes its own fill), which is exact for the single-ring zone data of this
 * demo. It is a one-time CPU step on static input.
 */
export function triangulatePolygons(
  positions: Float32Array,
  ringOffsets: Uint32Array,
  polygonOffsets: Uint32Array
): TriangulatedPolygons {
  const corners: number[] = [];
  const owners: number[] = [];
  const polygonCount = polygonOffsets.length - 1;
  for (let polygon = 0; polygon < polygonCount; polygon++) {
    for (let ring = polygonOffsets[polygon]; ring < polygonOffsets[polygon + 1]; ring++) {
      const start = ringOffsets[ring];
      const end = ringOffsets[ring + 1];
      for (const triangle of earClip(positions, start, end)) {
        for (const vertex of triangle) {
          corners.push(positions[vertex * 2], positions[vertex * 2 + 1]);
        }
        owners.push(polygon);
      }
    }
  }
  return {corners: Float32Array.from(corners), owners: Uint32Array.from(owners)};
}

function earClip(positions: Float32Array, start: number, end: number): [number, number, number][] {
  const count = end - start;
  const triangles: [number, number, number][] = [];
  if (count < 3) return triangles;
  const x = (vertex: number) => positions[vertex * 2];
  const y = (vertex: number) => positions[vertex * 2 + 1];
  let area = 0;
  for (let vertex = start; vertex < end; vertex++) {
    const next = vertex + 1 < end ? vertex + 1 : start;
    area += x(vertex) * y(next) - x(next) * y(vertex);
  }
  // Work on a counter-clockwise copy of the vertex order.
  const order: number[] = [];
  for (let index = 0; index < count; index++) {
    order.push(area >= 0 ? start + index : end - 1 - index);
  }
  const cross = (a: number, b: number, c: number) =>
    (x(b) - x(a)) * (y(c) - y(a)) - (y(b) - y(a)) * (x(c) - x(a));
  const isInside = (a: number, b: number, c: number, point: number) => {
    const first = cross(a, b, point);
    const second = cross(b, c, point);
    const third = cross(c, a, point);
    return first >= 0 && second >= 0 && third >= 0;
  };
  let cursor = 0;
  let sinceClip = 0;
  while (order.length > 3) {
    const size = order.length;
    const previous = order[(cursor + size - 1) % size];
    const current = order[cursor % size];
    const next = order[(cursor + 1) % size];
    const orientation = cross(previous, current, next);
    let clip = false;
    if (orientation === 0) {
      // Degenerate corner: drop it without a triangle.
      order.splice(cursor % size, 1);
      sinceClip = 0;
      continue;
    }
    if (orientation > 0) {
      clip = true;
      for (let index = 0; index < size && clip; index++) {
        const other = order[index];
        if (other === previous || other === current || other === next) continue;
        const samePoint = (vertex: number) => x(vertex) === x(other) && y(vertex) === y(other);
        if (samePoint(previous) || samePoint(current) || samePoint(next)) continue;
        if (isInside(previous, current, next, other)) clip = false;
      }
    }
    if (clip || sinceClip > size) {
      // After a full unsuccessful pass the polygon is not simple: clip anyway to make progress.
      if (orientation > 0) triangles.push([previous, current, next]);
      order.splice(cursor % size, 1);
      sinceClip = 0;
    } else {
      cursor = (cursor + 1) % size;
      sinceClip++;
    }
  }
  if (order.length === 3 && cross(order[0], order[1], order[2]) > 0) {
    triangles.push([order[0], order[1], order[2]]);
  }
  return triangles;
}
