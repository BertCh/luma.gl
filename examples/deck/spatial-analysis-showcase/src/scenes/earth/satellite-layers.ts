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
 * Bespoke layers of the satellite scenes. They read contributor output buffers directly and draw in
 * longitude/latitude degrees (`LNGLAT`) with an altitude in meters, which the layers can exaggerate
 * or compress (a log mapping keeps a 20,000 km GPS orbit on the same screen as a 550 km Starlink shell).
 *
 * Altitude display: `linear` is `scale * altitude`; `compressed` is `scale * 400 km * log2(1 + altitude / 400 km)`.
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
export type SatelliteColor = readonly [number, number, number, number?];

/** How altitude becomes screen height. */
export type AltitudeDisplay = 'linear' | 'compressed';

const SHARED_WGSL = /* wgsl */ `
fn getSatelliteDisplayAltitude(altitude: f32, scale: f32, mode: u32) -> f32 {
  if (mode == 1u) {
    return scale * 400000.0 * log2(1.0 + max(altitude, 0.0) / 400000.0);
  }
  return scale * altitude;
}

fn getSatelliteClipPosition(position: vec2<f32>, height: f32) -> vec4<f32> {
  var clipPosition = project_position_to_clipspace(vec3<f32>(position, height), vec3<f32>(0.0), vec3<f32>(0.0));
  // No depth test is used, so keep the point inside the depth range instead of clipping tall orbits.
  clipPosition.z = clipPosition.w * 0.5;
  return clipPosition;
}

// 0 at 200 km, 1 at 25,600 km (seven doublings), matching the legend extent.
fn getSatelliteAltitudeT(altitude: f32) -> f32 {
  return clamp(log2(max(altitude, 1.0) / 200000.0) / 7.0, 0.0, 1.0);
}

var<private> SEGMENT_CORNERS = array<vec2<f32>, 6>(
  vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(0.0, 1.0),
  vec2<f32>(0.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
);
var<private> QUAD_CORNERS = array<vec2<f32>, 6>(
  vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
  vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
);
`;

type LayerState = {model: Model; styleBuffer: Buffer};

/** Shared lifecycle: one model, one uniform buffer, optional GPU-written draw record. */
abstract class SatelliteBaseLayer<PropsT extends LayerProps> extends Layer<PropsT> {
  static override layerName = 'SatelliteBaseLayer';
  static override defaultProps = {
    coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
    parameters: BLEND_PARAMETERS
  };

  override getAttributeManager() {
    return null;
  }

  protected abstract getSource(): string;
  protected abstract getStyleByteLength(): number;
  protected abstract getBindings(styleBuffer: Buffer): Record<string, Buffer>;
  protected abstract writeStyle(styleBuffer: Buffer): void;
  protected getVertexCount(): number {
    return 6;
  }
  protected getDrawCommands(): DrawCommandBuffer | null {
    return null;
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
      vertexCount: this.getVertexCount(),
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
      drawCommands.draw(renderPass, 0);
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

function writeColor(target: Float32Array, offset: number, color: SatelliteColor): void {
  target[offset] = color[0] / 255;
  target[offset + 1] = color[1] / 255;
  target[offset + 2] = color[2] / 255;
  target[offset + 3] = (color[3] ?? 255) / 255;
}

function writePalette(target: Float32Array, palette: readonly SatelliteColor[]): void {
  for (let index = 0; index < 8; index++) {
    writeColor(target, index * 4, palette[index % palette.length]);
  }
}

// ---------------------------------------------------------------------------------------------
// Markers and stems
// ---------------------------------------------------------------------------------------------

const MARKER_STYLE_BYTE_LENGTH = 128 + 16 + 48;

const MARKER_SHADER = /* wgsl */ `
struct MarkerStyle {
  palette: array<vec4<f32>, 8>,
  outlineColor: vec4<f32>,
  sizePixels: f32,
  opacity: f32,
  altitudeScale: f32,
  altitudeMode: u32,
  colorMode: u32,
  groupFilter: u32,
  colormap: u32,
  paletteSize: u32,
  drawStems: u32,
  selectedTrack: u32,
  stemWidth: f32,
  _padding: u32,
};

@group(0) @binding(auto) var<uniform> markerStyle: MarkerStyle;
@group(0) @binding(auto) var<storage, read> markerIds: array<u32>;
@group(0) @binding(auto) var<storage, read> markerPositions: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> markerElevations: array<f32>;
@group(0) @binding(auto) var<storage, read> markerColors: array<u32>;
@group(0) @binding(auto) var<storage, read> markerGroups: array<u32>;
@group(0) @binding(auto) var<storage, read> markerSatellites: array<u32>;

struct MarkerOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) local: vec2<f32>,
  @location(1) color: vec4<f32>,
  @location(2) highlight: f32,
};
${SHARED_WGSL}
${getRampWgsl()}

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> MarkerOutput {
  var output: MarkerOutput;
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  output.local = vec2<f32>(0.0);
  output.color = vec4<f32>(0.0);
  output.highlight = 0.0;
  let track = markerIds[instanceIndex];
  if (markerColors[track] == 0xffffffffu) {
    return output;
  }
  if (markerStyle.groupFilter != 0xffffffffu && markerStyle.groupFilter != markerGroups[track]) {
    return output;
  }
  let altitude = markerElevations[track];
  let height = getSatelliteDisplayAltitude(altitude, markerStyle.altitudeScale, markerStyle.altitudeMode);
  var color: vec4<f32>;
  if (markerStyle.colorMode == 0u) {
    color = markerStyle.palette[markerColors[track] % max(markerStyle.paletteSize, 1u)];
  } else {
    color = vec4<f32>(spatialAnalysisSampleRamp(markerStyle.colormap, getSatelliteAltitudeT(altitude)), 1.0);
  }
  output.color = color;
  let groundPosition = markerPositions[track];
  let top = getSatelliteClipPosition(groundPosition, height);
  if (markerStyle.drawStems != 0u) {
    // A vertical line from the ground point to the satellite.
    let bottom = getSatelliteClipPosition(groundPosition, 0.0);
    let corner = SEGMENT_CORNERS[vertexIndex];
    let direction = (top.xy / top.w - bottom.xy / bottom.w) * project.viewportSize;
    let directionLength = length(direction);
    var unit = vec2<f32>(0.0, 1.0);
    if (directionLength > 1e-6) {
      unit = direction / directionLength;
    }
    let normal = vec2<f32>(-unit.y, unit.x);
    var clipPosition = mix(bottom, top, corner.x);
    clipPosition = vec4<f32>(
      clipPosition.xy + project_pixel_size_to_clipspace(normal * corner.y * markerStyle.stemWidth * 0.5),
      clipPosition.z,
      clipPosition.w
    );
    output.position = clipPosition;
    output.local = vec2<f32>(0.0, corner.y);
    output.color = vec4<f32>(color.rgb, color.a * 0.55);
    return output;
  }
  var scale = markerStyle.sizePixels;
  if (markerSatellites[track] == markerStyle.selectedTrack) {
    scale = scale * 1.8;
    output.highlight = 1.0;
  }
  let corner = QUAD_CORNERS[vertexIndex];
  output.position = vec4<f32>(
    top.xy + project_pixel_size_to_clipspace(corner * scale),
    top.z,
    top.w
  );
  output.local = corner;
  return output;
}

@fragment fn fragmentMain(input: MarkerOutput) -> @location(0) vec4<f32> {
  if (markerStyle.drawStems != 0u) {
    let coverage = 1.0 - smoothstep(0.55, 1.0, abs(input.local.y));
    return vec4<f32>(input.color.rgb, input.color.a * markerStyle.opacity * coverage);
  }
  let radiusSquared = dot(input.local, input.local);
  if (radiusSquared > 1.0) { discard; }
  let coverage = 1.0 - smoothstep(0.7, 1.0, radiusSquared);
  let ring = smoothstep(0.45, 0.62, radiusSquared);
  var rgb = mix(input.color.rgb, markerStyle.outlineColor.rgb, ring * markerStyle.outlineColor.a);
  if (input.highlight > 0.5) {
    rgb = mix(rgb, vec3<f32>(1.0), ring);
  }
  return vec4<f32>(rgb, coverage * markerStyle.opacity * input.color.a);
}
`;

/** Props for {@link SatelliteMarkerLayer}. */
export type SatelliteMarkerLayerProps = LayerProps & {
  /** Compact `uint32` list of active track indices (the playhead's `activeTracks.ids`). */
  ids: Buffer;
  /** `float32x2` interpolated longitude/latitude per track. */
  positions: Buffer;
  /** `float32` interpolated altitude in meters per track. */
  elevations: Buffer;
  /** `uint32` palette row per track (`colorMode: 'palette'`); `0xffffffff` hides the track in either mode. */
  colors: Buffer;
  /** `uint32` group per track, used by `groupFilter`. */
  groups: Buffer;
  /** `uint32` satellite index per track, used by `selectedSatellite`. */
  satellites: Buffer;
  drawCommands: DrawCommandBuffer;
  palette: readonly SatelliteColor[];
  /** `'palette'` colors by `colors`; `'altitude'` by a ramp over log altitude. */
  colorMode?: 'palette' | 'altitude';
  ramp?: RampName;
  sizePixels?: number;
  altitudeScale?: number;
  altitudeDisplay?: AltitudeDisplay;
  /** Draw vertical lines from the ground point to the satellite instead of markers. */
  stems?: boolean;
  stemWidthPixels?: number;
  /** Only this group is drawn; `null` draws all. */
  groupFilter?: number | null;
  selectedSatellite?: number | null;
  outlineColor?: SatelliteColor;
};

/** One disc (or one stem) per active satellite, gathered by track id from the playhead outputs. */
export class SatelliteMarkerLayer extends SatelliteBaseLayer<SatelliteMarkerLayerProps> {
  static override layerName = 'SatelliteMarkerLayer';
  protected getSource(): string {
    return MARKER_SHADER;
  }
  protected getStyleByteLength(): number {
    return MARKER_STYLE_BYTE_LENGTH;
  }
  protected getDrawCommands(): DrawCommandBuffer {
    return this.props.drawCommands;
  }
  protected getBindings(styleBuffer: Buffer): Record<string, Buffer> {
    return {
      markerStyle: styleBuffer,
      markerIds: this.props.ids,
      markerPositions: this.props.positions,
      markerElevations: this.props.elevations,
      markerColors: this.props.colors,
      markerGroups: this.props.groups,
      markerSatellites: this.props.satellites
    };
  }
  protected writeStyle(styleBuffer: Buffer): void {
    const props = this.props;
    const data = new ArrayBuffer(MARKER_STYLE_BYTE_LENGTH);
    const floats = new Float32Array(data);
    const words = new Uint32Array(data);
    writePalette(floats, props.palette);
    writeColor(floats, 32, props.outlineColor ?? [14, 18, 26, 230]);
    floats[36] = props.sizePixels ?? 6;
    floats[37] = props.opacity ?? 1;
    floats[38] = props.altitudeScale ?? 1;
    words[39] = props.altitudeDisplay === 'compressed' ? 1 : 0;
    words[40] = props.colorMode === 'altitude' ? 1 : 0;
    words[41] =
      props.groupFilter === null || props.groupFilter === undefined
        ? 0xffffffff
        : props.groupFilter;
    words[42] = COLORMAP_INDEXES[props.ramp ?? 'viridis'];
    words[43] = Math.min(8, props.palette.length);
    words[44] = props.stems ? 1 : 0;
    words[45] =
      props.selectedSatellite === null || props.selectedSatellite === undefined
        ? 0xffffffff
        : props.selectedSatellite;
    floats[46] = props.stemWidthPixels ?? 1;
    styleBuffer.write(new Uint8Array(data));
  }
}

// ---------------------------------------------------------------------------------------------
// Trails
// ---------------------------------------------------------------------------------------------

const TRAIL_STYLE_BYTE_LENGTH = 128 + 48;

const TRAIL_SHADER = /* wgsl */ `
struct TrailStyle {
  palette: array<vec4<f32>, 8>,
  widthPixels: f32,
  opacity: f32,
  altitudeScale: f32,
  altitudeMode: u32,
  colorMode: u32,
  groupFilter: u32,
  colormap: u32,
  paletteSize: u32,
  _padding0: u32,
  _padding1: u32,
  _padding2: u32,
  _padding3: u32,
};

@group(0) @binding(auto) var<uniform> trailStyle: TrailStyle;
@group(0) @binding(auto) var<storage, read> trailPositions: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> trailAltitudes: array<f32>;
@group(0) @binding(auto) var<storage, read> trailSegmentEnds: array<u32>;
@group(0) @binding(auto) var<storage, read> trailSegmentTracks: array<u32>;
@group(0) @binding(auto) var<storage, read> trailIds: array<u32>;
@group(0) @binding(auto) var<storage, read> trailWeights: array<f32>;
@group(0) @binding(auto) var<storage, read> trailClips: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> trailColors: array<u32>;
@group(0) @binding(auto) var<storage, read> trailGroups: array<u32>;

struct TrailOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) side: f32,
  @location(1) color: vec4<f32>,
};
${SHARED_WGSL}
${getRampWgsl()}

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> TrailOutput {
  var output: TrailOutput;
  output.side = 0.0;
  output.color = vec4<f32>(0.0);
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  let segment = trailIds[instanceIndex];
  let track = trailSegmentTracks[segment];
  if (trailColors[track] == 0xffffffffu) {
    return output;
  }
  if (trailStyle.groupFilter != 0xffffffffu && trailStyle.groupFilter != trailGroups[track]) {
    return output;
  }
  let clip = trailClips[segment];
  let weight = trailWeights[segment];
  if (clip.y <= clip.x || weight <= 0.0) {
    return output;
  }
  let endVertex = trailSegmentEnds[segment];
  let startVertex = endVertex - 1u;
  let a = trailPositions[startVertex];
  let b = trailPositions[endVertex];
  let altitudeA = trailAltitudes[startVertex];
  let altitudeB = trailAltitudes[endVertex];
  let start = mix(a, b, clip.x);
  let end = mix(a, b, clip.y);
  let startAltitude = mix(altitudeA, altitudeB, clip.x);
  let endAltitude = mix(altitudeA, altitudeB, clip.y);
  let startClip = getSatelliteClipPosition(start, getSatelliteDisplayAltitude(startAltitude, trailStyle.altitudeScale, trailStyle.altitudeMode));
  let endClip = getSatelliteClipPosition(end, getSatelliteDisplayAltitude(endAltitude, trailStyle.altitudeScale, trailStyle.altitudeMode));
  var color: vec4<f32>;
  if (trailStyle.colorMode == 0u) {
    color = trailStyle.palette[trailColors[track] % max(trailStyle.paletteSize, 1u)];
  } else {
    color = vec4<f32>(spatialAnalysisSampleRamp(trailStyle.colormap, getSatelliteAltitudeT(0.5 * (startAltitude + endAltitude))), 1.0);
  }
  let corner = SEGMENT_CORNERS[vertexIndex];
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
    clipPosition.xy + project_pixel_size_to_clipspace((normal * corner.y + along) * trailStyle.widthPixels * 0.5),
    clipPosition.z,
    clipPosition.w
  );
  output.position = clipPosition;
  output.side = corner.y;
  output.color = vec4<f32>(color.rgb, color.a * weight);
  return output;
}

@fragment fn fragmentMain(input: TrailOutput) -> @location(0) vec4<f32> {
  let coverage = 1.0 - smoothstep(0.55, 1.0, abs(input.side));
  return vec4<f32>(input.color.rgb, input.color.a * trailStyle.opacity * coverage);
}
`;

/** Props for {@link SatelliteTrailLayer}. */
export type SatelliteTrailLayerProps = LayerProps & {
  /** `float32x2` longitude/latitude per vertex. */
  positions: Buffer;
  /** `float32` altitude in meters per vertex. */
  altitudes: Buffer;
  /** `uint32` end vertex of every segment (the segment is `end - 1` to `end`). */
  segmentEnds: Buffer;
  /** `uint32` track of every segment. */
  segmentTracks: Buffer;
  /** Compact segment ids from `GPUTimeWindowFilter`. */
  ids: Buffer;
  /** Fade weights from `GPUTimeWindowFilter`. */
  weights: Buffer;
  /** Clip fractions from `GPUTimeWindowFilter`. */
  clipFractions: Buffer;
  drawCommands: DrawCommandBuffer;
  /** `uint32` palette row per track. */
  colors: Buffer;
  /** `uint32` group per track. */
  groups: Buffer;
  palette: readonly SatelliteColor[];
  colorMode?: 'palette' | 'altitude';
  ramp?: RampName;
  widthPixels?: number;
  altitudeScale?: number;
  altitudeDisplay?: AltitudeDisplay;
  groupFilter?: number | null;
};

/** Fading trails in 3D from the compact segment list of a time window. */
export class SatelliteTrailLayer extends SatelliteBaseLayer<SatelliteTrailLayerProps> {
  static override layerName = 'SatelliteTrailLayer';
  protected getSource(): string {
    return TRAIL_SHADER;
  }
  protected getStyleByteLength(): number {
    return TRAIL_STYLE_BYTE_LENGTH;
  }
  protected getDrawCommands(): DrawCommandBuffer {
    return this.props.drawCommands;
  }
  protected getBindings(styleBuffer: Buffer): Record<string, Buffer> {
    return {
      trailStyle: styleBuffer,
      trailPositions: this.props.positions,
      trailAltitudes: this.props.altitudes,
      trailSegmentEnds: this.props.segmentEnds,
      trailSegmentTracks: this.props.segmentTracks,
      trailIds: this.props.ids,
      trailWeights: this.props.weights,
      trailClips: this.props.clipFractions,
      trailColors: this.props.colors,
      trailGroups: this.props.groups
    };
  }
  protected writeStyle(styleBuffer: Buffer): void {
    const props = this.props;
    const data = new ArrayBuffer(TRAIL_STYLE_BYTE_LENGTH);
    const floats = new Float32Array(data);
    const words = new Uint32Array(data);
    writePalette(floats, props.palette);
    floats[32] = props.widthPixels ?? 1.5;
    floats[33] = props.opacity ?? 1;
    floats[34] = props.altitudeScale ?? 1;
    words[35] = props.altitudeDisplay === 'compressed' ? 1 : 0;
    words[36] = props.colorMode === 'altitude' ? 1 : 0;
    words[37] =
      props.groupFilter === null || props.groupFilter === undefined
        ? 0xffffffff
        : props.groupFilter;
    words[38] = COLORMAP_INDEXES[props.ramp ?? 'viridis'];
    words[39] = Math.min(8, props.palette.length);
    styleBuffer.write(new Uint8Array(data));
  }
}

// ---------------------------------------------------------------------------------------------
// Longitude/latitude cells
// ---------------------------------------------------------------------------------------------

const CELL_STYLE_BYTE_LENGTH = 16 + 16 + 16 + 16 + 16 + 16 + 16;

const CELL_SHADER = /* wgsl */ `
struct CellStyle {
  baseColor: vec4<f32>,
  noDataColor: vec4<f32>,
  grid: vec4<f32>,
  valueRange: vec2<f32>,
  discardAtOrBelow: f32,
  opacity: f32,
  size: vec2<u32>,
  colormap: u32,
  sqrtScale: u32,
  valueScale: f32,
  useDiscard: u32,
  uniformColor: u32,
  hatchNoData: u32,
  discardAbove: f32,
  useDiscardAbove: u32,
  elapsedClasses: u32,
  densityClasses: u32,
};

@group(0) @binding(auto) var<uniform> cellStyle: CellStyle;
@group(0) @binding(auto) var<storage, read> cellValues: array<f32>;

struct CellOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
  @location(1) local: vec2<f32>,
  @location(2) noData: f32,
};
${SHARED_WGSL}
${getRampWgsl()}

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> CellOutput {
  var output: CellOutput;
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  output.color = vec4<f32>(0.0);
  output.local = vec2<f32>(0.0);
  output.noData = 0.0;
  let column = instanceIndex % cellStyle.size.x;
  let row = instanceIndex / cellStyle.size.x;
  let raw = cellValues[instanceIndex];
  let bits = bitcast<u32>(raw);
  var value = raw * cellStyle.valueScale;
  var color: vec4<f32>;
  if ((bits & 0x7fffffffu) >= 0x7f800000u) {
    color = cellStyle.noDataColor;
  } else if (cellStyle.useDiscard != 0u && value <= cellStyle.discardAtOrBelow) {
    color = cellStyle.noDataColor;
    output.noData = 1.0;
  } else if (cellStyle.useDiscardAbove != 0u && value > cellStyle.discardAbove) {
    color = cellStyle.noDataColor;
    output.noData = 1.0;
  } else if (cellStyle.uniformColor != 0u) {
    color = cellStyle.baseColor;
  } else {
    var t = clamp((value - cellStyle.valueRange.x) / max(cellStyle.valueRange.y - cellStyle.valueRange.x, 1e-20), 0.0, 1.0);
    if (cellStyle.sqrtScale != 0u) {
      t = sqrt(t);
    }
    if (cellStyle.elapsedClasses != 0u) {
      // Ordered first-reach classes: 0-30, 30-60, 60-120 and 120-180 minutes.
      if (value <= 1800.0) {
        color = vec4<f32>(0.18, 0.66, 0.72, cellStyle.baseColor.a);
      } else if (value <= 3600.0) {
        color = vec4<f32>(0.34, 0.76, 0.56, cellStyle.baseColor.a);
      } else if (value <= 7200.0) {
        color = vec4<f32>(0.95, 0.70, 0.22, cellStyle.baseColor.a);
      } else {
        color = vec4<f32>(0.72, 0.34, 0.70, cellStyle.baseColor.a);
      }
    } else if (cellStyle.densityClasses != 0u) {
      // Six equal-interval, ordered ice-to-violet density classes; breaks are supplied by the
      // scene's observed maximum and shared with its class-table legend.
      let level = min(5u, u32(floor(t * 6.0)));
      let colors = array<vec3<f32>, 6>(
        vec3<f32>(0.80, 0.91, 0.96), vec3<f32>(0.61, 0.80, 0.91),
        vec3<f32>(0.45, 0.66, 0.84), vec3<f32>(0.43, 0.52, 0.76),
        vec3<f32>(0.48, 0.37, 0.68), vec3<f32>(0.34, 0.20, 0.52)
      );
      color = vec4<f32>(colors[level], cellStyle.baseColor.a);
    } else {
      color = vec4<f32>(spatialAnalysisSampleRamp(cellStyle.colormap, t), cellStyle.baseColor.a);
    }
  }
  if (color.a <= 0.0) {
    return output;
  }
  let corner = QUAD_CORNERS[vertexIndex] * 0.5 + vec2<f32>(0.5);
  let lngLat = vec2<f32>(
    cellStyle.grid.x + (f32(column) + corner.x) * cellStyle.grid.z,
    cellStyle.grid.y + (f32(row) + corner.y) * cellStyle.grid.w
  );
  output.position = getSatelliteClipPosition(lngLat, 0.0);
  output.color = color;
  output.local = corner;
  return output;
}

@fragment fn fragmentMain(input: CellOutput) -> @location(0) vec4<f32> {
  if (cellStyle.hatchNoData != 0u && input.noData > 0.5) {
    let stripe = fract((input.local.x + input.local.y) * 12.0);
    if (stripe > 0.22) { discard; }
  }
  return vec4<f32>(input.color.rgb, input.color.a * cellStyle.opacity);
}
`;

/** Props for {@link SatelliteCellLayer}. */
export type SatelliteCellLayerProps = LayerProps & {
  /** `float32` value per cell, row-major from the south-west corner (row 0 is the southernmost). */
  values: Buffer;
  /** `[columns, rows]`. */
  gridSize: readonly [number, number];
  /** `[minLongitude, minLatitude, cellWidth, cellHeight]` in degrees. */
  grid: readonly [number, number, number, number];
  colormap?: RampName;
  valueRange?: readonly [number, number];
  valueScale?: number;
  sqrtScale?: boolean;
  /** Values at or below this get `noDataColor` (alpha 0 hides the cell). */
  discardAtOrBelow?: number;
  /** Values above this get `noDataColor` too (for example cells first covered after the playhead). */
  discardAbove?: number;
  /** Color for NaN, infinite and discarded cells. Defaults to transparent. */
  noDataColor?: SatelliteColor;
  /** Draw every non-discarded cell in this color instead of the ramp. */
  color?: SatelliteColor;
  /** Render discarded cells as a sparse diagonal hatch rather than a filled value. */
  hatchNoData?: boolean;
  /** Use the satellite first-reach classes (0-30/30-60/60-120/120-180 minutes). */
  elapsedClasses?: boolean;
  /** Use six equal-interval ice-to-violet density classes instead of a continuous ramp. */
  densityClasses?: boolean;
};

/** Axis-aligned longitude/latitude cells (rectangles in Web Mercator) colored by a value buffer. */
export class SatelliteCellLayer extends SatelliteBaseLayer<SatelliteCellLayerProps> {
  static override layerName = 'SatelliteCellLayer';
  protected getSource(): string {
    return CELL_SHADER;
  }
  protected getStyleByteLength(): number {
    return CELL_STYLE_BYTE_LENGTH;
  }
  protected getInstanceCount(): number {
    return this.props.gridSize[0] * this.props.gridSize[1];
  }
  protected getBindings(styleBuffer: Buffer): Record<string, Buffer> {
    return {cellStyle: styleBuffer, cellValues: this.props.values};
  }
  protected writeStyle(styleBuffer: Buffer): void {
    const props = this.props;
    const data = new ArrayBuffer(CELL_STYLE_BYTE_LENGTH);
    const floats = new Float32Array(data);
    const words = new Uint32Array(data);
    writeColor(floats, 0, props.color ?? [255, 255, 255, 255]);
    writeColor(floats, 4, props.noDataColor ?? [0, 0, 0, 0]);
    floats.set(props.grid, 8);
    floats[12] = props.valueRange?.[0] ?? 0;
    floats[13] = props.valueRange?.[1] ?? 1;
    floats[14] = props.discardAtOrBelow ?? 0;
    floats[15] = props.opacity ?? 1;
    words[16] = props.gridSize[0];
    words[17] = props.gridSize[1];
    words[18] = COLORMAP_INDEXES[props.colormap ?? 'viridis'];
    words[19] = props.sqrtScale ? 1 : 0;
    floats[20] = props.valueScale ?? 1;
    words[21] = props.discardAtOrBelow === undefined ? 0 : 1;
    words[22] = props.color ? 1 : 0;
    words[23] = props.hatchNoData ? 1 : 0;
    floats[24] = props.discardAbove ?? 0;
    words[25] = props.discardAbove === undefined ? 0 : 1;
    words[26] = props.elapsedClasses ? 1 : 0;
    words[27] = props.densityClasses ? 1 : 0;
    styleBuffer.write(new Uint8Array(data));
  }
}

// ---------------------------------------------------------------------------------------------
// Swath outlines
// ---------------------------------------------------------------------------------------------

const SWATH_STYLE_BYTE_LENGTH = 48;

const SWATH_SHADER = /* wgsl */ `
struct SwathStyle {
  color: vec4<f32>,
  timeLimit: f32,
  opacity: f32,
  rowsPerInput: u32,
  _padding0: u32,
  _padding1: vec4<u32>,
};

@group(0) @binding(auto) var<uniform> swathStyle: SwathStyle;
@group(0) @binding(auto) var<storage, read> swathPositions: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> swathTimes: array<f32>;
@group(0) @binding(auto) var<storage, read> swathSatellites: array<u32>;
@group(0) @binding(auto) var<storage, read> swathVisible: array<u32>;

struct SwathOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
};
${SHARED_WGSL}

@vertex fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> SwathOutput {
  var output: SwathOutput;
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  output.color = vec4<f32>(0.0);
  let input = vertexIndex / swathStyle.rowsPerInput;
  if (swathTimes[input] > swathStyle.timeLimit || swathVisible[swathSatellites[input]] == 0u) {
    return output;
  }
  output.position = getSatelliteClipPosition(swathPositions[vertexIndex], 0.0);
  output.color = swathStyle.color;
  return output;
}

@fragment fn fragmentMain(input: SwathOutput) -> @location(0) vec4<f32> {
  return vec4<f32>(input.color.rgb, input.color.a * swathStyle.opacity);
}
`;

/** Props for {@link SatelliteSwathLayer}. */
export type SatelliteSwathLayerProps = LayerProps & {
  /** `float32x2` triangle-list vertices from `GPUOutlineGeometry`. */
  positions: Buffer;
  /** `float32` time of every input vertex (`positions.length / rowsPerInput` rows). */
  times: Buffer;
  /** `uint32` satellite index of every input vertex. */
  satellites: Buffer;
  /** `uint32` per satellite: 0 hides the satellite's geometry. */
  visibleSatellites: Buffer;
  /** Output rows per input vertex (`getGPUOutlineGeometryVerticesPerInput`). */
  rowsPerInput: number;
  /** Total triangle-list vertices. */
  vertexCount: number;
  /** Only geometry of input vertices at or before this time is drawn. */
  timeLimit: number;
  color: SatelliteColor;
};

/** Draws the triangles of `GPUOutlineGeometry` for the input vertices up to a time limit. */
export class SatelliteSwathLayer extends SatelliteBaseLayer<SatelliteSwathLayerProps> {
  static override layerName = 'SatelliteSwathLayer';
  protected getSource(): string {
    return SWATH_SHADER;
  }
  protected getStyleByteLength(): number {
    return SWATH_STYLE_BYTE_LENGTH;
  }
  protected getVertexCount(): number {
    return this.props.vertexCount;
  }
  protected getInstanceCount(): number {
    return 1;
  }
  protected getBindings(styleBuffer: Buffer): Record<string, Buffer> {
    return {
      swathStyle: styleBuffer,
      swathPositions: this.props.positions,
      swathTimes: this.props.times,
      swathSatellites: this.props.satellites,
      swathVisible: this.props.visibleSatellites
    };
  }
  protected writeStyle(styleBuffer: Buffer): void {
    const props = this.props;
    const data = new ArrayBuffer(SWATH_STYLE_BYTE_LENGTH);
    const floats = new Float32Array(data);
    const words = new Uint32Array(data);
    writeColor(floats, 0, props.color);
    floats[4] = props.timeLimit;
    floats[5] = props.opacity ?? 1;
    words[6] = props.rowsPerInput;
    styleBuffer.write(new Uint8Array(data));
  }
}
