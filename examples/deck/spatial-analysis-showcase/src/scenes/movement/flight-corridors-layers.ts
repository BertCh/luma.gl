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
 * Bespoke 3-D layers of the flight scenes. They read longitude/latitude degrees and metres of
 * altitude straight from GPU storage buffers (the same buffers the analysis graphs read) and place
 * every vertex at `[longitude, latitude, altitude * elevationScale]`, so one code path draws the
 * flat map (`elevationScale: 0`) and the extruded view.
 */

export type FlightColor = readonly [number, number, number, number?];

/**
 * How a segment or marker becomes a color.
 *
 * - `uniform`: `color`.
 * - `altitude`: `ramp` over `valueRange` metres, from the elevation of the row.
 * - `value`: `ramp` over `valueRange` of `values[row] * valueScale` (float32).
 * - `category`: `palette[values[row] % 8]` (uint32).
 */
export type FlightColorMode = 'uniform' | 'altitude' | 'value' | 'category';

const COLOR_MODE_INDEXES: Record<FlightColorMode, number> = {
  uniform: 0,
  altitude: 1,
  value: 2,
  category: 3
};

/** Props shared by both flight layers. */
export type FlightLayerProps = LayerProps & {
  /** `float32x2` longitude, latitude per vertex. */
  lngLat: Buffer;
  /** `float32` altitude in metres per vertex. */
  elevations: Buffer;
  /** Optional per-row values (see {@link FlightColorMode}); float32 or uint32 words. */
  values?: Buffer | null;
  colorMode?: FlightColorMode;
  ramp?: RampName;
  /** `[min, max]` of the ramp: metres for `altitude`, `values * valueScale` for `value`. */
  valueRange?: readonly [number, number];
  valueScale?: number;
  palette?: readonly FlightColor[];
  color?: FlightColor;
  /** Metres of altitude drawn per metre. 0 flattens the map. Defaults to 0. */
  elevationScale?: number;
  /** Compact row ids written by a GPU graph. */
  ids?: Buffer | null;
  drawCommands?: DrawCommandBuffer | null;
  instanceCount?: number;
  opacity?: number;
};

const DEFAULT_PALETTE: readonly FlightColor[] = [
  [86, 180, 233, 255],
  [240, 150, 60, 255],
  [150, 150, 150, 255],
  [120, 200, 120, 255]
];

const STYLE_BYTE_LENGTH = 224;

const STYLE_WGSL = /* wgsl */ `
struct FlightStyle {
  baseColor: vec4<f32>,
  palette: array<vec4<f32>, 8>,
  sizePixels: f32,
  opacity: f32,
  elevationScale: f32,
  colormap: u32,
  valueRange: vec2<f32>,
  valueScale: f32,
  colorMode: u32,
  useIds: u32,
  useWeights: u32,
  useClip: u32,
  paletteSize: u32,
  minAltitude: f32,
  directionFilter: u32,
  coneCosine: f32,
  useHeadings: u32,
  valueIsFloat: u32,
  _padding0: u32,
  _padding1: u32,
  _padding2: u32,
};

@group(0) @binding(auto) var<uniform> flightStyle: FlightStyle;
@group(0) @binding(auto) var<storage, read> flightLngLat: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> flightElevations: array<f32>;
@group(0) @binding(auto) var<storage, read> flightValues: array<u32>;
@group(0) @binding(auto) var<storage, read> flightIds: array<u32>;

${getRampWgsl()}

// Color of one row from the elevation (metres) and its value word.
fn getFlightColor(row: u32, altitude: f32) -> vec4<f32> {
  let mode = flightStyle.colorMode;
  if (mode == 0u) { return flightStyle.baseColor; }
  var t = 0.0;
  if (mode == 3u) {
    return flightStyle.palette[flightValues[row] % max(flightStyle.paletteSize, 1u)];
  }
  var value = altitude;
  if (mode == 2u) {
    let raw = flightValues[row];
    value = select(f32(raw), bitcast<f32>(raw), flightStyle.valueIsFloat != 0u) * flightStyle.valueScale;
    if ((raw & 0x7fffffffu) >= 0x7f800000u) { return vec4<f32>(0.0); }
  }
  t = clamp((value - flightStyle.valueRange.x) / max(flightStyle.valueRange.y - flightStyle.valueRange.x, 1e-20), 0.0, 1.0);
  let rgb = spatialAnalysisSampleRamp(flightStyle.colormap, t);
  return vec4<f32>(rgb, flightStyle.baseColor.a);
}

fn projectFlight(lngLat: vec2<f32>, altitude: f32) -> vec4<f32> {
  var clipPosition = project_position_to_clipspace(
    vec3<f32>(lngLat, altitude * flightStyle.elevationScale), vec3<f32>(0.0), vec3<f32>(0.0)
  );
  // Deck's project32 matrices use OpenGL depth; WebGPU clip space requires [0, w].
  clipPosition.z = (clipPosition.z + clipPosition.w) * 0.5;
  return clipPosition;
}
`;

const SEGMENT_SHADER = /* wgsl */ `
${STYLE_WGSL}
@group(0) @binding(auto) var<storage, read> flightEndVertices: array<u32>;
@group(0) @binding(auto) var<storage, read> flightWeights: array<f32>;
@group(0) @binding(auto) var<storage, read> flightClip: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> flightHeadings: array<f32>;

struct SegmentOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) side: f32,
  @location(1) color: vec4<f32>,
};

fn hiddenSegment() -> SegmentOutput {
  var output: SegmentOutput;
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  output.color = vec4<f32>(0.0);
  output.side = 0.0;
  return output;
}

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> SegmentOutput {
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
  );
  var row = instanceIndex;
  if (flightStyle.useIds != 0u) { row = flightIds[instanceIndex]; }
  let endVertex = flightEndVertices[row];
  let startVertex = endVertex - 1u;
  let altitudeStart = flightElevations[startVertex];
  let altitudeEnd = flightElevations[endVertex];
  if (flightStyle.minAltitude > 0.0 && min(altitudeStart, altitudeEnd) < flightStyle.minAltitude) {
    return hiddenSegment();
  }
  if (flightStyle.directionFilter != 0u) {
    let heading = cos(flightHeadings[endVertex]);
    let east = heading > flightStyle.coneCosine;
    let west = heading < -flightStyle.coneCosine;
    if (!select(west, east, flightStyle.directionFilter == 1u)) { return hiddenSegment(); }
  }
  var color = getFlightColor(endVertex, 0.5 * (altitudeStart + altitudeEnd));
  if (flightStyle.useWeights != 0u) { color.a = color.a * flightWeights[row]; }
  var clip = vec2<f32>(0.0, 1.0);
  if (flightStyle.useClip != 0u) { clip = flightClip[row]; }
  if (color.a <= 0.0 || clip.y <= clip.x) { return hiddenSegment(); }
  let start = flightLngLat[startVertex];
  let end = flightLngLat[endVertex];
  let clippedStart = mix(start, end, clip.x);
  let clippedEnd = mix(start, end, clip.y);
  let startClip = projectFlight(clippedStart, mix(altitudeStart, altitudeEnd, clip.x));
  let endClip = projectFlight(clippedEnd, mix(altitudeStart, altitudeEnd, clip.y));
  let corner = corners[vertexIndex];
  let screenDirection = (endClip.xy / endClip.w - startClip.xy / startClip.w) * project.viewportSize;
  let directionLength = length(screenDirection);
  var direction = vec2<f32>(1.0, 0.0);
  if (directionLength > 1e-6) { direction = screenDirection / directionLength; }
  let normal = vec2<f32>(-direction.y, direction.x);
  var clipPosition = mix(startClip, endClip, corner.x);
  let along = direction * (corner.x * 2.0 - 1.0) * 0.5;
  clipPosition = vec4<f32>(
    clipPosition.xy + project_pixel_size_to_clipspace((normal * corner.y + along) * flightStyle.sizePixels * 0.5),
    clipPosition.z,
    clipPosition.w
  );
  var output: SegmentOutput;
  output.position = clipPosition;
  output.side = corner.y;
  output.color = color;
  return output;
}

@fragment fn fragmentMain(input: SegmentOutput) -> @location(0) vec4<f32> {
  let coverage = 1.0 - smoothstep(0.55, 1.0, abs(input.side));
  return vec4<f32>(input.color.rgb, input.color.a * flightStyle.opacity * coverage);
}
`;

const POINT_SHADER = /* wgsl */ `
${STYLE_WGSL}

struct PointOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) corner: vec2<f32>,
  @location(1) color: vec4<f32>,
};

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> PointOutput {
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
  );
  var output: PointOutput;
  var row = instanceIndex;
  if (flightStyle.useIds != 0u) { row = flightIds[instanceIndex]; }
  let source = flightLngLat[row];
  let altitude = flightElevations[row];
  let color = getFlightColor(row, altitude);
  if (source.x != source.x || source.y != source.y || color.a <= 0.0) {
    output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    output.color = vec4<f32>(0.0);
    output.corner = vec2<f32>(0.0);
    return output;
  }
  let corner = corners[vertexIndex];
  var clipPosition = projectFlight(source, altitude);
  clipPosition = vec4<f32>(
    clipPosition.xy + project_pixel_size_to_clipspace(corner * flightStyle.sizePixels),
    clipPosition.z,
    clipPosition.w
  );
  output.position = clipPosition;
  output.corner = corner;
  output.color = color;
  return output;
}

@fragment fn fragmentMain(input: PointOutput) -> @location(0) vec4<f32> {
  let radiusSquared = dot(input.corner, input.corner);
  if (radiusSquared > 1.0) { discard; }
  let rim = smoothstep(0.55, 0.8, radiusSquared);
  let rgb = mix(input.color.rgb, input.color.rgb * 0.35, rim);
  let coverage = 1.0 - smoothstep(0.85, 1.0, radiusSquared);
  return vec4<f32>(rgb, input.color.a * flightStyle.opacity * coverage);
}
`;

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

type FlightLayerState = {
  model: Model | null;
  styleBuffer: Buffer | null;
  placeholderBuffer: Buffer | null;
};

/** Shared lifecycle: one model with storage bindings and an owned style uniform buffer. */
abstract class FlightBaseLayer<PropsT extends FlightLayerProps> extends Layer<PropsT> {
  static override layerName = 'FlightBaseLayer';
  static override defaultProps = {
    coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
    parameters: BLEND_PARAMETERS
  };

  override getAttributeManager() {
    return null;
  }

  protected abstract getShaderSource(): string;
  protected abstract getBindings(placeholder: Buffer): Record<string, Buffer>;
  protected abstract writeStyle(styleBuffer: Buffer): void;

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
    this.setState({model: null, styleBuffer, placeholderBuffer} satisfies FlightLayerState);
    const model = new Model(device, {
      ...this.getShaders({modules: [project32], source: this.getShaderSource()}),
      id: `${this.id}-model`,
      topology: 'triangle-list',
      isInstanced: true,
      vertexCount: 6,
      instanceCount: 0,
      bufferLayout: [],
      bindings: {...this.getBindings(placeholderBuffer), flightStyle: styleBuffer},
      parameters: BLEND_PARAMETERS
    });
    this.setState({model});
  }

  override updateState(parameters: UpdateParameters<this>): void {
    super.updateState(parameters);
    const {model, styleBuffer, placeholderBuffer} = this.state as FlightLayerState;
    if (model && styleBuffer && placeholderBuffer) {
      model.setBindings({...this.getBindings(placeholderBuffer), flightStyle: styleBuffer});
    }
  }

  override getModels(): Model[] {
    const model = (this.state as FlightLayerState).model;
    return model ? [model] : [];
  }

  override draw({renderPass}: {renderPass: RenderPass}): void {
    const {model, styleBuffer} = this.state as FlightLayerState;
    if (!model || !styleBuffer) return;
    this.writeStyle(styleBuffer);
    const {drawCommands, instanceCount = 0} = this.props;
    if (drawCommands) {
      // Bind pipeline and bindings with an empty draw, then replay the GPU-written record.
      model.setInstanceCount(0);
      model.draw(renderPass);
      drawCommands.draw(renderPass, 0);
    } else {
      model.setInstanceCount(instanceCount);
      model.draw(renderPass);
    }
  }

  override finalizeState(context: LayerContext): void {
    const state = this.state as FlightLayerState;
    state.model?.destroy();
    state.styleBuffer?.destroy();
    state.placeholderBuffer?.destroy();
    this.setState({model: null, styleBuffer: null, placeholderBuffer: null});
    super.finalizeState(context);
  }

  protected getCommonBindings(placeholder: Buffer): Record<string, Buffer> {
    return {
      flightLngLat: this.props.lngLat,
      flightElevations: this.props.elevations,
      flightValues: this.props.values ?? placeholder,
      flightIds: this.props.ids ?? placeholder
    };
  }

  /** Packs the shared style uniform. */
  protected packStyle(
    styleBuffer: Buffer,
    extra: {
      sizePixels: number;
      useWeights?: boolean;
      useClip?: boolean;
      minAltitude?: number;
      directionFilter?: number;
      coneCosine?: number;
      valueIsFloat?: boolean;
    }
  ): void {
    const props = this.props;
    const data = new ArrayBuffer(STYLE_BYTE_LENGTH);
    const floats = new Float32Array(data);
    const words = new Uint32Array(data);
    const writeColor = (offset: number, color: FlightColor) => {
      floats.set(
        [color[0] / 255, color[1] / 255, color[2] / 255, (color[3] ?? 255) / 255],
        offset / 4
      );
    };
    writeColor(0, props.color ?? [255, 255, 255, 255]);
    const palette = props.palette ?? DEFAULT_PALETTE;
    for (let index = 0; index < 8; index++) {
      writeColor(16 + index * 16, palette[index % palette.length]);
    }
    const mode = props.colorMode ?? 'uniform';
    floats[36] = extra.sizePixels;
    floats[37] = props.opacity ?? 1;
    floats[38] = props.elevationScale ?? 0;
    words[39] = COLORMAP_INDEXES[props.ramp ?? 'viridis'];
    floats.set(props.valueRange ?? [0, 1], 40);
    floats[42] = props.valueScale ?? 1;
    words[43] = COLOR_MODE_INDEXES[mode];
    words[44] = props.ids ? 1 : 0;
    words[45] = extra.useWeights ? 1 : 0;
    words[46] = extra.useClip ? 1 : 0;
    words[47] = Math.min(8, palette.length);
    floats[48] = extra.minAltitude ?? 0;
    words[49] = extra.directionFilter ?? 0;
    floats[50] = extra.coneCosine ?? 0.7071;
    words[51] = 0;
    words[52] = extra.valueIsFloat ? 1 : 0;
    styleBuffer.write(new Uint8Array(data));
  }
}

/** Props for {@link FlightSegmentLayer}. */
export type FlightSegmentLayerProps = FlightLayerProps & {
  /** `uint32` end vertex per segment; the segment joins vertex `end - 1` to `end`. */
  endVertices: Buffer;
  /** Optional float32 per row multiplying alpha (time-window fade weights). */
  weights?: Buffer | null;
  /** Optional `float32x2` per row of drawn start and end fractions. */
  clipFractions?: Buffer | null;
  /** Segments whose lower end is below this altitude (metres) are hidden. 0 turns it off. */
  minAltitude?: number;
  /** `float32` heading in radians per vertex (of the step ending there), from `GPUTrajectoryMetrics`. */
  headings?: Buffer | null;
  /** `'east'` or `'west'` hides steps outside a cone of `directionHalfAngle` around that heading. */
  directionFilter?: 'all' | 'east' | 'west';
  /** Half angle of the direction cone in radians. Defaults to 45 degrees. */
  directionHalfAngle?: number;
  /** True when `values` holds float32 (default), false for uint32 categories. */
  valuesAreFloat?: boolean;
  /** Line width in CSS pixels. Defaults to 1.5. */
  widthPixels?: number;
};

/** Screen-width flight segments at longitude, latitude and altitude read from GPU buffers. */
export class FlightSegmentLayer extends FlightBaseLayer<FlightSegmentLayerProps> {
  static override layerName = 'FlightSegmentLayer';

  protected getShaderSource(): string {
    return SEGMENT_SHADER;
  }
  protected getBindings(placeholder: Buffer): Record<string, Buffer> {
    return {
      ...this.getCommonBindings(placeholder),
      flightEndVertices: this.props.endVertices,
      flightWeights: this.props.weights ?? placeholder,
      flightClip: this.props.clipFractions ?? placeholder,
      flightHeadings: this.props.headings ?? placeholder
    };
  }
  protected writeStyle(styleBuffer: Buffer): void {
    const {directionFilter = 'all', directionHalfAngle = Math.PI / 4} = this.props;
    this.packStyle(styleBuffer, {
      sizePixels: this.props.widthPixels ?? 1.5,
      useWeights: Boolean(this.props.weights),
      useClip: Boolean(this.props.clipFractions),
      minAltitude: this.props.minAltitude,
      directionFilter: this.props.headings
        ? directionFilter === 'east'
          ? 1
          : directionFilter === 'west'
            ? 2
            : 0
        : 0,
      coneCosine: Math.cos(directionHalfAngle),
      valueIsFloat: this.props.valuesAreFloat ?? true
    });
  }
}

/** Props for {@link FlightPointLayer}. */
export type FlightPointLayerProps = FlightLayerProps & {
  /** Disc radius in CSS pixels. Defaults to 3. */
  radiusPixels?: number;
  valuesAreFloat?: boolean;
};

/** Screen-space discs at GPU-resident longitude/latitude, lifted by their own altitude. */
export class FlightPointLayer extends FlightBaseLayer<FlightPointLayerProps> {
  static override layerName = 'FlightPointLayer';

  protected getShaderSource(): string {
    return POINT_SHADER;
  }
  protected getBindings(placeholder: Buffer): Record<string, Buffer> {
    return this.getCommonBindings(placeholder);
  }
  protected writeStyle(styleBuffer: Buffer): void {
    this.packStyle(styleBuffer, {
      sizePixels: this.props.radiusPixels ?? 3,
      valueIsFloat: this.props.valuesAreFloat ?? false
    });
  }
}
