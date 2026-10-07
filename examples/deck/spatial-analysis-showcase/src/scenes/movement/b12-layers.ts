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

/**
 * Bespoke layers of the movement chapter. Each reads contributor output buffers directly (compact
 * ids, GPU-written draw records) so moving a playhead or editing a threshold never touches the CPU.
 *
 * Every layer takes `coordinateSystem`: the vessel scenes draw planar meters (`METER_OFFSETS`
 * around `coordinateOrigin`, the default), the gull scene draws longitude/latitude degrees
 * (`LNGLAT`) because its analysis meters live in an azimuthal projection.
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

/** Color with 0-255 channels. */
export type MovementColor = readonly [number, number, number, number?];

const SHARED_WGSL = /* wgsl */ `
fn getMovementClipPosition(position: vec2<f32>) -> vec4<f32> {
  var clipPosition = project_position_to_clipspace(vec3<f32>(position, 0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  // Deck's project32 matrices use OpenGL depth; WebGPU clip space requires [0, w].
  clipPosition.z = (clipPosition.z + clipPosition.w) * 0.5;
  return clipPosition;
}
`;

const CORNERS_WGSL = /* wgsl */ `
var<private> QUAD_CORNERS = array<vec2<f32>, 6>(
  vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
  vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
);
`;

type LayerState = {model: Model; styleBuffer: Buffer};

/** Shared lifecycle of the layers below: one model, one uniform buffer, GPU-written draw record. */
abstract class MovementLayer<PropsT extends LayerProps> extends Layer<PropsT> {
  static override layerName = 'MovementLayer';
  static override defaultProps = {
    coordinateSystem: COORDINATE_SYSTEM.METER_OFFSETS,
    parameters: BLEND_PARAMETERS
  };

  override getAttributeManager() {
    return null;
  }

  protected abstract getSource(): string;
  protected abstract getStyleByteLength(): number;
  protected abstract getBindings(styleBuffer: Buffer): Record<string, Buffer>;
  protected abstract writeStyle(styleBuffer: Buffer): void;
  protected abstract getDrawCommands(): DrawCommandBuffer | null;
  protected getDrawCommandIndex(): number {
    return 0;
  }
  protected getInstanceCount(): number {
    return 0;
  }

  override initializeState({device}: LayerContext): void {
    if (device.type !== 'webgpu') throw new Error(`${this.id} requires WebGPU`);
    const styleBuffer = device.createBuffer({
      id: `${this.id}-style`,
      byteLength: this.getStyleByteLength(),
      usage: Buffer.UNIFORM | Buffer.COPY_DST
    });
    const model = new Model(device, {
      ...this.getShaders({modules: [project32], source: this.getSource()}),
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
    const {model, styleBuffer} = this.state as LayerState;
    model.setBindings(this.getBindings(styleBuffer));
  }

  override getModels(): Model[] {
    return [(this.state as LayerState).model];
  }

  override draw({renderPass}: {renderPass: RenderPass}): void {
    const {model, styleBuffer} = this.state as LayerState;
    this.writeStyle(styleBuffer);
    const drawCommands = this.getDrawCommands();
    if (drawCommands) {
      // Bind pipeline and bindings with an empty draw, then replay the GPU-written record.
      model.setInstanceCount(0);
      model.draw(renderPass);
      drawCommands.draw(renderPass, this.getDrawCommandIndex());
    } else {
      model.setInstanceCount(this.getInstanceCount());
      model.draw(renderPass);
    }
  }

  override finalizeState(context: LayerContext): void {
    const {model, styleBuffer} = this.state as LayerState;
    model.destroy();
    styleBuffer.destroy();
    super.finalizeState(context);
  }
}

function writeColor(target: Float32Array, offset: number, color: MovementColor): void {
  target[offset] = color[0] / 255;
  target[offset + 1] = color[1] / 255;
  target[offset + 2] = color[2] / 255;
  target[offset + 3] = (color[3] ?? 255) / 255;
}

// ---------------------------------------------------------------------------------------------
// Vessel markers
// ---------------------------------------------------------------------------------------------

const VESSEL_STYLE_BYTE_LENGTH = 128 + 48 + 16 + 16;

const VESSEL_SHADER = /* wgsl */ `
struct VesselStyle {
  palette: array<vec4<f32>, 8>,
  outlineColor: vec4<f32>,
  sizePixels: f32,
  speedForFullColor: f32,
  opacity: f32,
  colorMode: u32,
  categoryFilter: u32,
  colormap: u32,
  selectedTrack: u32,
  paletteSize: u32,
  stoppedSpeed: f32,
  speedClassCount: u32,
  _padding2: u32,
  _padding3: u32,
  speedClassBreaks: vec4<f32>,
};

@group(0) @binding(auto) var<uniform> vesselStyle: VesselStyle;
@group(0) @binding(auto) var<storage, read> vesselIds: array<u32>;
@group(0) @binding(auto) var<storage, read> vesselPositions: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> vesselHeadings: array<f32>;
@group(0) @binding(auto) var<storage, read> vesselSpeeds: array<f32>;
@group(0) @binding(auto) var<storage, read> vesselCategories: array<u32>;

struct VesselVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) local: vec2<f32>,
  @location(1) color: vec4<f32>,
  @location(2) highlight: f32,
  @location(3) stopped: f32,
};
${SHARED_WGSL}
${CORNERS_WGSL}
${getRampWgsl()}

fn segmentDistance(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
  let ab = b - a;
  let t = clamp(dot(p - a, ab) / dot(ab, ab), 0.0, 1.0);
  return length(p - a - ab * t);
}

fn triangleSign(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
  return (p.x - b.x) * (a.y - b.y) - (a.x - b.x) * (p.y - b.y);
}

fn insideTriangle(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> bool {
  let d1 = triangleSign(p, a, b);
  let d2 = triangleSign(p, b, c);
  let d3 = triangleSign(p, c, a);
  let negative = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
  let positive = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
  return !(negative && positive);
}

// Signed distance to an arrowhead: tip a, wings b and c, notch n. Negative inside.
fn arrowDistance(p: vec2<f32>) -> f32 {
  let a = vec2<f32>(1.0, 0.0);
  let b = vec2<f32>(-0.8, 0.7);
  let n = vec2<f32>(-0.35, 0.0);
  let c = vec2<f32>(-0.8, -0.7);
  var d = segmentDistance(p, a, b);
  d = min(d, segmentDistance(p, b, n));
  d = min(d, segmentDistance(p, n, c));
  d = min(d, segmentDistance(p, c, a));
  let inside = insideTriangle(p, a, b, n) || insideTriangle(p, a, n, c);
  return select(d, -d, inside);
}

// Signed distance to the axis-aligned square drawn for a stopped vessel. Negative inside.
fn stoppedSquareDistance(p: vec2<f32>) -> f32 {
  let q = abs(p) - vec2<f32>(0.42);
  return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0);
}

// Class of a speed against up to four ascending breaks (m/s): 0 below the first break.
fn getSpeedClass(speed: f32) -> u32 {
  var speedClass = 0u;
  let breakCount = min(vesselStyle.speedClassCount, 5u) - 1u;
  for (var index = 0u; index < 4u; index = index + 1u) {
    if (index < breakCount && speed >= vesselStyle.speedClassBreaks[index]) {
      speedClass = index + 1u;
    }
  }
  return speedClass;
}

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> VesselVertexOutput {
  var output: VesselVertexOutput;
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  output.local = vec2<f32>(0.0);
  output.color = vec4<f32>(0.0);
  output.highlight = 0.0;
  output.stopped = 0.0;
  let track = vesselIds[instanceIndex];
  let category = vesselCategories[track];
  if (vesselStyle.categoryFilter != 0xffffffffu && vesselStyle.categoryFilter != category) {
    return output;
  }
  let heading = vesselHeadings[track];
  let speed = vesselSpeeds[track];
  let isStopped = vesselStyle.stoppedSpeed > 0.0 && speed < vesselStyle.stoppedSpeed;
  // A stopped vessel is an unrotated square: its heading carries no meaning.
  let rotation = select(vec2<f32>(cos(heading), sin(heading)), vec2<f32>(1.0, 0.0), isStopped);
  output.stopped = select(0.0, 1.0, isStopped);
  let corner = QUAD_CORNERS[vertexIndex] * 1.2;
  let rotated = vec2<f32>(
    corner.x * rotation.x - corner.y * rotation.y,
    corner.x * rotation.y + corner.y * rotation.x
  );
  var scale = vesselStyle.sizePixels;
  if (track == vesselStyle.selectedTrack) {
    scale = scale * 1.6;
    output.highlight = 1.0;
  }
  var clipPosition = getMovementClipPosition(vesselPositions[track]);
  clipPosition = vec4<f32>(
    clipPosition.xy + project_pixel_size_to_clipspace(rotated * scale),
    clipPosition.z,
    clipPosition.w
  );
  output.position = clipPosition;
  output.local = corner;
  if (vesselStyle.colorMode == 0u) {
    output.color = vesselStyle.palette[category % max(vesselStyle.paletteSize, 1u)];
  } else if (vesselStyle.colorMode == 2u) {
    output.color = vesselStyle.palette[min(getSpeedClass(speed), max(vesselStyle.paletteSize, 1u) - 1u)];
  } else {
    let t = clamp(speed / max(vesselStyle.speedForFullColor, 1e-6), 0.0, 1.0);
    output.color = vec4<f32>(spatialAnalysisSampleRamp(vesselStyle.colormap, t), 1.0);
  }
  return output;
}

@fragment fn fragmentMain(input: VesselVertexOutput) -> @location(0) vec4<f32> {
  let d = select(arrowDistance(input.local), stoppedSquareDistance(input.local), input.stopped > 0.5);
  let size = max(vesselStyle.sizePixels * select(1.0, 1.6, input.highlight > 0.5), 1.0);
  let aa = 1.0 / size;
  let cover = 1.0 - smoothstep(-aa * 0.5, aa * 0.5, d);
  if (cover <= 0.0) { discard; }
  let outlineWidth = 1.2 * aa;
  let edge = smoothstep(-outlineWidth - aa, -outlineWidth, d);
  var rgb = mix(input.color.rgb, vesselStyle.outlineColor.rgb, edge * vesselStyle.outlineColor.a);
  if (input.highlight > 0.5) {
    rgb = mix(rgb, vec3<f32>(1.0), edge);
  }
  return vec4<f32>(rgb, cover * vesselStyle.opacity * input.color.a);
}
`;

/** Props for {@link VesselMarkerLayer}. */
export type VesselMarkerLayerProps = LayerProps & {
  /** Compact `uint32` list of active track indices (contributor `activeTracks.ids`). */
  ids: Buffer;
  /** `float32x2` interpolated position per track. */
  positions: Buffer;
  /** `float32` heading per track in radians, counterclockwise from +x. */
  headings: Buffer;
  /** `float32` speed per track in meters per second. */
  speeds: Buffer;
  /** `uint32` category per track (palette row). */
  categories: Buffer;
  /** GPU-written indirect record with `vertexCount` 6 whose instance count is the active count. */
  drawCommands: DrawCommandBuffer;
  /** Arrow half-length in CSS pixels. Defaults to 8. */
  sizePixels?: number;
  /**
   * `'category'` colors by the palette, `'speed'` by a ramp over speed, `'speedClasses'` by the
   * palette row of the speed class (see `speedClassBreaks`). Defaults to `'category'`.
   */
  colorMode?: 'category' | 'speed' | 'speedClasses';
  /**
   * Up to four ascending speed breaks in m/s for `colorMode: 'speedClasses'`: class 0 is below
   * the first break and takes palette row 0, and so on (at most five classes).
   */
  speedClassBreaks?: readonly number[];
  /**
   * Speed in m/s below which a vessel is drawn as a small unrotated square instead of a
   * heading arrow (the chart-plotter convention: squares are stopped, arrows are moving).
   * 0 (the default) draws every vessel as an arrow.
   */
  stoppedSpeed?: number;
  /** Ramp used in speed mode. Defaults to `'viridis'`. */
  ramp?: RampName;
  /** Speed in m/s at which the ramp ends. Defaults to 12. */
  speedForFullColor?: number;
  /** Up to 8 category colors. */
  palette: readonly MovementColor[];
  /** Only this category is drawn (others are culled in the vertex shader). `null` draws all. */
  categoryFilter?: number | null;
  /** Track drawn larger with a white outline, or `null`. */
  selectedTrack?: number | null;
  /** Outline color. Defaults to a dark gray. */
  outlineColor?: MovementColor;
};

/**
 * One heading-oriented arrowhead per active track, gathered by track id from the playhead outputs.
 * Color is the vessel category, or the speed through a shared ramp.
 */
export class VesselMarkerLayer extends MovementLayer<VesselMarkerLayerProps> {
  static override layerName = 'VesselMarkerLayer';

  protected getSource(): string {
    return VESSEL_SHADER;
  }
  protected getStyleByteLength(): number {
    return VESSEL_STYLE_BYTE_LENGTH;
  }
  protected getDrawCommands(): DrawCommandBuffer {
    return this.props.drawCommands;
  }
  protected getBindings(styleBuffer: Buffer): Record<string, Buffer> {
    return {
      vesselStyle: styleBuffer,
      vesselIds: this.props.ids,
      vesselPositions: this.props.positions,
      vesselHeadings: this.props.headings,
      vesselSpeeds: this.props.speeds,
      vesselCategories: this.props.categories
    };
  }
  protected writeStyle(styleBuffer: Buffer): void {
    const props = this.props;
    const data = new ArrayBuffer(VESSEL_STYLE_BYTE_LENGTH);
    const floats = new Float32Array(data);
    const words = new Uint32Array(data);
    for (let index = 0; index < 8; index++) {
      writeColor(floats, index * 4, props.palette[index % props.palette.length]);
    }
    writeColor(floats, 32, props.outlineColor ?? [18, 22, 30, 230]);
    floats[36] = props.sizePixels ?? 8;
    floats[37] = props.speedForFullColor ?? 12;
    floats[38] = props.opacity ?? 1;
    words[39] = props.colorMode === 'speed' ? 1 : props.colorMode === 'speedClasses' ? 2 : 0;
    words[40] =
      props.categoryFilter === null || props.categoryFilter === undefined
        ? 0xffffffff
        : props.categoryFilter;
    words[41] = COLORMAP_INDEXES[props.ramp ?? 'viridis'];
    words[42] =
      props.selectedTrack === null || props.selectedTrack === undefined
        ? 0xffffffff
        : props.selectedTrack;
    words[43] = Math.min(8, props.palette.length);
    floats[44] = props.stoppedSpeed ?? 0;
    const speedClassBreaks = (props.speedClassBreaks ?? []).slice(0, 4);
    words[45] = speedClassBreaks.length + 1;
    for (let index = 0; index < 4; index++) {
      floats[48 + index] = speedClassBreaks[index] ?? 3e38;
    }
    styleBuffer.write(new Uint8Array(data));
  }
}

// ---------------------------------------------------------------------------------------------
// Stops
// ---------------------------------------------------------------------------------------------

const STOP_STYLE_BYTE_LENGTH = 32 + 16 + 80 + 16;

const STOP_SHADER = /* wgsl */ `
struct StopStyle {
  baseRadiusPixels: f32,
  radiusPerSqrtSecond: f32,
  maximumRadiusPixels: f32,
  durationForFullColor: f32,
  opacity: f32,
  ringWidthPixels: f32,
  classCount: f32,
  useRingColor: f32,
  classBreaks: vec4<f32>,
  classColors: array<vec4<f32>, 5>,
  ringColor: vec4<f32>,
};

@group(0) @binding(auto) var<uniform> stopStyle: StopStyle;
@group(0) @binding(auto) var<storage, read> stopCentroids: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> stopDurations: array<f32>;

struct StopVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) corner: vec2<f32>,
  @location(1) color: vec4<f32>,
  @location(2) ringStart: f32,
};
${SHARED_WGSL}
${CORNERS_WGSL}

// Light pink for short dwells through hot pink to deep red for the longest.
fn getStopColor(t: f32) -> vec3<f32> {
  let low = vec3<f32>(1.0, 0.78, 0.88);
  let middle = vec3<f32>(1.0, 0.16, 0.5);
  let high = vec3<f32>(0.75, 0.0, 0.2);
  if (t < 0.5) {
    return mix(low, middle, t * 2.0);
  }
  return mix(middle, high, (t - 0.5) * 2.0);
}

// Exact class colour of a duration (seconds) against up to four ascending breaks.
fn getStopClassColor(duration: f32) -> vec4<f32> {
  let classCount = u32(stopStyle.classCount);
  var stopClass = 0u;
  for (var index = 0u; index < 4u; index = index + 1u) {
    if (index + 1u < classCount && duration >= stopStyle.classBreaks[index]) {
      stopClass = index + 1u;
    }
  }
  return stopStyle.classColors[min(stopClass, 4u)];
}

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> StopVertexOutput {
  var output: StopVertexOutput;
  let duration = stopDurations[instanceIndex];
  let centroid = stopCentroids[instanceIndex];
  let radiusPixels = min(
    stopStyle.baseRadiusPixels + stopStyle.radiusPerSqrtSecond * sqrt(max(duration, 0.0)),
    stopStyle.maximumRadiusPixels
  );
  let corner = QUAD_CORNERS[vertexIndex];
  var clipPosition = getMovementClipPosition(centroid);
  clipPosition = vec4<f32>(
    clipPosition.xy + project_pixel_size_to_clipspace(corner * radiusPixels),
    clipPosition.z,
    clipPosition.w
  );
  output.position = clipPosition;
  output.corner = corner;
  if (stopStyle.classCount > 0.5) {
    output.color = getStopClassColor(duration);
  } else {
    output.color = vec4<f32>(
      getStopColor(clamp(sqrt(duration / stopStyle.durationForFullColor), 0.0, 1.0)),
      0.72
    );
  }
  output.ringStart = 1.0 - stopStyle.ringWidthPixels / radiusPixels;
  return output;
}

@fragment fn fragmentMain(input: StopVertexOutput) -> @location(0) vec4<f32> {
  let radius = length(input.corner);
  if (radius > 1.0) { discard; }
  let edge = 1.0 - smoothstep(0.92, 1.0, radius);
  let ring = smoothstep(input.ringStart - 0.04, input.ringStart, radius);
  let ringRgb = select(vec3<f32>(1.0), stopStyle.ringColor.rgb, stopStyle.useRingColor > 0.5);
  let ringAlpha = select(1.0, stopStyle.ringColor.a, stopStyle.useRingColor > 0.5);
  let rgb = mix(input.color.rgb, ringRgb, ring);
  let alpha = mix(input.color.a, ringAlpha, ring) * stopStyle.opacity * edge;
  return vec4<f32>(rgb, alpha);
}
`;

/** Props for {@link StopMarkerLayer}. */
export type StopMarkerLayerProps = LayerProps & {
  /** `float32x2` stop centroids. Rows past the stop count are never drawn. */
  centroids: Buffer;
  /** `float32` stop durations in seconds, aligned with `centroids`. */
  durations: Buffer;
  /** GPU-written indirect record with `vertexCount` 6 whose instance count is the stop count. */
  drawCommands: DrawCommandBuffer;
  /** Radius in CSS pixels of a zero-duration stop. Defaults to 3. */
  baseRadiusPixels?: number;
  /** Extra radius per square root of a second of dwell. Defaults to 0.45. */
  radiusPerSqrtSecond?: number;
  /** Upper bound of the radius in CSS pixels. Defaults to 14. */
  maximumRadiusPixels?: number;
  /** Duration in seconds at which the color reaches its darkest value. Defaults to 600. */
  durationForFullColor?: number;
  /**
   * Up to four ascending duration breaks in seconds. With `classColors` the disc takes the exact
   * colour of its duration class (class 0 below the first break) instead of the pink ramp.
   */
  classBreaks?: readonly number[];
  /** Exact class colours (alpha included), one more than `classBreaks`, at most five. */
  classColors?: readonly MovementColor[];
  /** Colour of the outer ring (a ground-coloured halo, for example). Defaults to white. */
  ringColor?: MovementColor;
  /** Ring width in CSS pixels. Defaults to 1.5. */
  ringWidthPixels?: number;
};

/** One outlined disc per stop; radius and color grow with the square root of the dwell. */
export class StopMarkerLayer extends MovementLayer<StopMarkerLayerProps> {
  static override layerName = 'StopMarkerLayer';

  protected getSource(): string {
    return STOP_SHADER;
  }
  protected getStyleByteLength(): number {
    return STOP_STYLE_BYTE_LENGTH;
  }
  protected getDrawCommands(): DrawCommandBuffer {
    return this.props.drawCommands;
  }
  protected getBindings(styleBuffer: Buffer): Record<string, Buffer> {
    return {
      stopStyle: styleBuffer,
      stopCentroids: this.props.centroids,
      stopDurations: this.props.durations
    };
  }
  protected writeStyle(styleBuffer: Buffer): void {
    const props = this.props;
    const floats = new Float32Array(STOP_STYLE_BYTE_LENGTH / 4);
    const classColors = (props.classColors ?? []).slice(0, 5);
    const classBreaks = (props.classBreaks ?? []).slice(0, 4);
    floats.set([
      props.baseRadiusPixels ?? 3,
      props.radiusPerSqrtSecond ?? 0.45,
      props.maximumRadiusPixels ?? 14,
      props.durationForFullColor ?? 600,
      props.opacity ?? 1,
      props.ringWidthPixels ?? 1.5,
      classColors.length > 0 ? Math.min(classColors.length, classBreaks.length + 1) : 0,
      props.ringColor ? 1 : 0
    ]);
    for (let index = 0; index < 4; index++) {
      floats[8 + index] = classBreaks[index] ?? 3e38;
    }
    for (let index = 0; index < 5; index++) {
      const color = classColors[Math.min(index, classColors.length - 1)];
      writeColor(floats, 12 + index * 4, color ?? [0, 0, 0, 0]);
    }
    writeColor(floats, 32, props.ringColor ?? [255, 255, 255, 255]);
    styleBuffer.write(floats);
  }
}

// ---------------------------------------------------------------------------------------------
// Kept segments (simplified lines)
// ---------------------------------------------------------------------------------------------

const KEPT_SEGMENT_SHADER = /* wgsl */ `
struct KeptSegmentStyle {
  color: vec4<f32>,
  widthPixels: f32,
  opacity: f32,
  _padding0: f32,
  _padding1: f32,
};

@group(0) @binding(auto) var<uniform> keptSegmentStyle: KeptSegmentStyle;
@group(0) @binding(auto) var<storage, read> vertexPositions: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> keptIds: array<u32>;
@group(0) @binding(auto) var<storage, read> vertexLines: array<u32>;
@group(0) @binding(auto) var<storage, read> keptCount: array<u32>;

struct KeptSegmentOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) side: f32,
};
${SHARED_WGSL}

// Instance i connects kept vertices i and i + 1 when both belong to the same line.
@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> KeptSegmentOutput {
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
  );
  var output: KeptSegmentOutput;
  output.side = 0.0;
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  if (instanceIndex + 1u >= keptCount[0]) {
    return output;
  }
  let startRow = keptIds[instanceIndex];
  let endRow = keptIds[instanceIndex + 1u];
  if (vertexLines[startRow] != vertexLines[endRow]) {
    return output;
  }
  let startClip = getMovementClipPosition(vertexPositions[startRow]);
  let endClip = getMovementClipPosition(vertexPositions[endRow]);
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
    clipPosition.xy + project_pixel_size_to_clipspace((normal * corner.y + along) * keptSegmentStyle.widthPixels * 0.5),
    clipPosition.z,
    clipPosition.w
  );
  output.position = clipPosition;
  output.side = corner.y;
  return output;
}

@fragment fn fragmentMain(input: KeptSegmentOutput) -> @location(0) vec4<f32> {
  let coverage = 1.0 - smoothstep(0.55, 1.0, abs(input.side));
  return vec4<f32>(keptSegmentStyle.color.rgb, keptSegmentStyle.color.a * keptSegmentStyle.opacity * coverage);
}
`;

/** Props for {@link KeptSegmentLayer}. */
export type KeptSegmentLayerProps = LayerProps & {
  /** `float32x2` vertex positions, one per original vertex. */
  positions: Buffer;
  /** Ascending kept vertex rows (the contributor's compact `output.ids`). */
  keptIds: Buffer;
  /** `uint32` line index of every original vertex, so two lines are never joined. */
  vertexLines: Buffer;
  /** One `uint32` word: the number of valid entries in `keptIds`. */
  keptCount: Buffer;
  /** GPU-written indirect record (`vertexCount` 6) whose instance count is the kept count. */
  drawCommands: DrawCommandBuffer;
  widthPixels?: number;
  color?: MovementColor;
};

/** Draws simplified polylines straight from compact kept ids. */
export class KeptSegmentLayer extends MovementLayer<KeptSegmentLayerProps> {
  static override layerName = 'KeptSegmentLayer';

  protected getSource(): string {
    return KEPT_SEGMENT_SHADER;
  }
  protected getStyleByteLength(): number {
    return 32;
  }
  protected getDrawCommands(): DrawCommandBuffer {
    return this.props.drawCommands;
  }
  protected getBindings(styleBuffer: Buffer): Record<string, Buffer> {
    return {
      keptSegmentStyle: styleBuffer,
      vertexPositions: this.props.positions,
      keptIds: this.props.keptIds,
      vertexLines: this.props.vertexLines,
      keptCount: this.props.keptCount
    };
  }
  protected writeStyle(styleBuffer: Buffer): void {
    const [red, green, blue, alpha = 255] = this.props.color ?? [255, 150, 40, 255];
    styleBuffer.write(
      Float32Array.of(
        red / 255,
        green / 255,
        blue / 255,
        alpha / 255,
        this.props.widthPixels ?? 2,
        this.props.opacity ?? 1,
        0,
        0
      )
    );
  }
}

// ---------------------------------------------------------------------------------------------
// Zone events
// ---------------------------------------------------------------------------------------------

const EVENT_SHADER = /* wgsl */ `
struct EventStyle {
  enterColor: vec4<f32>,
  exitColor: vec4<f32>,
  sizePixels: f32,
  opacity: f32,
  _padding0: f32,
  _padding1: f32,
};

@group(0) @binding(auto) var<uniform> eventStyle: EventStyle;
@group(0) @binding(auto) var<storage, read> eventPositions: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> eventTracks: array<u32>;
@group(0) @binding(auto) var<storage, read> eventTimes: array<f32>;
@group(0) @binding(auto) var<storage, read> eventTypes: array<u32>;
@group(0) @binding(auto) var<storage, read> trackStartTimes: array<f32>;
// [playhead seconds, pulse seconds, show-all flag, unused]
@group(0) @binding(auto) var<storage, read> eventClock: array<f32>;

struct EventVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) corner: vec2<f32>,
  @location(1) color: vec4<f32>,
  @location(2) pulse: f32,
};
${SHARED_WGSL}
${CORNERS_WGSL}

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> EventVertexOutput {
  var output: EventVertexOutput;
  let track = eventTracks[instanceIndex];
  let time = trackStartTimes[track] + eventTimes[instanceIndex];
  let age = eventClock[0] - time;
  let pulseSeconds = max(eventClock[1], 1.0);
  var color = select(eventStyle.enterColor, eventStyle.exitColor, eventTypes[instanceIndex] == 1u);
  var scale = 0.55;
  var ring = 0.0;
  if (age >= 0.0 && age <= pulseSeconds) {
    let phase = age / pulseSeconds;
    scale = 1.0 + 2.2 * phase;
    color.a = 1.0 - phase;
    ring = 1.0;
  } else if (eventClock[2] > 0.5) {
    color.a = 0.28;
  } else {
    color.a = 0.0;
  }
  let corner = QUAD_CORNERS[vertexIndex];
  var clipPosition = getMovementClipPosition(eventPositions[instanceIndex]);
  clipPosition = vec4<f32>(
    clipPosition.xy + project_pixel_size_to_clipspace(corner * eventStyle.sizePixels * scale),
    clipPosition.z,
    clipPosition.w
  );
  output.position = clipPosition;
  output.corner = corner;
  output.color = color;
  output.pulse = ring;
  return output;
}

@fragment fn fragmentMain(input: EventVertexOutput) -> @location(0) vec4<f32> {
  let radius = length(input.corner);
  if (radius > 1.0 || input.color.a <= 0.0) { discard; }
  let edge = 1.0 - smoothstep(0.85, 1.0, radius);
  // Pulsing events are rings; resting events are small solid dots.
  let ringMask = select(1.0, smoothstep(0.45, 0.7, radius), input.pulse > 0.5);
  return vec4<f32>(input.color.rgb, input.color.a * eventStyle.opacity * edge * ringMask);
}
`;

/** Props for {@link ZoneEventMarkerLayer}. */
export type ZoneEventMarkerLayerProps = LayerProps & {
  /** `float32x2` crossing position per event (`events.eventPositions`). */
  positions: Buffer;
  /** `uint32` track per event (`events.output.ids`). */
  eventTracks: Buffer;
  /** `float32` crossing time per event, relative to the track's first timestamp. */
  eventTimes: Buffer;
  /** `uint32` type per event: 0 enter, 1 exit. */
  eventTypes: Buffer;
  /** `float32` first timestamp of each track. */
  trackStartTimes: Buffer;
  /** `float32` `[playhead, pulseSeconds, showAll, 0]`, for example a `GPUParameterBuffer`. */
  clock: Buffer;
  /** GPU-written record with `vertexCount` 6 and the event count as instance count. */
  drawCommands: DrawCommandBuffer;
  drawCommandIndex?: number;
  enterColor?: MovementColor;
  exitColor?: MovementColor;
  sizePixels?: number;
};

/**
 * Enter and exit events at their interpolated crossing positions. An event pulses as a growing,
 * fading ring for the pulse duration after the playhead passes it; with the show-all flag the
 * other events rest as faint dots.
 */
export class ZoneEventMarkerLayer extends MovementLayer<ZoneEventMarkerLayerProps> {
  static override layerName = 'ZoneEventMarkerLayer';

  protected getSource(): string {
    return EVENT_SHADER;
  }
  protected getStyleByteLength(): number {
    return 48;
  }
  protected getDrawCommands(): DrawCommandBuffer {
    return this.props.drawCommands;
  }
  protected override getDrawCommandIndex(): number {
    return this.props.drawCommandIndex ?? 0;
  }
  protected getBindings(styleBuffer: Buffer): Record<string, Buffer> {
    return {
      eventStyle: styleBuffer,
      eventPositions: this.props.positions,
      eventTracks: this.props.eventTracks,
      eventTimes: this.props.eventTimes,
      eventTypes: this.props.eventTypes,
      trackStartTimes: this.props.trackStartTimes,
      eventClock: this.props.clock
    };
  }
  protected writeStyle(styleBuffer: Buffer): void {
    const data = new Float32Array(12);
    writeColor(data, 0, this.props.enterColor ?? [77, 230, 140, 255]);
    writeColor(data, 4, this.props.exitColor ?? [255, 115, 64, 255]);
    data[8] = this.props.sizePixels ?? 5;
    data[9] = this.props.opacity ?? 1;
    styleBuffer.write(data);
  }
}
