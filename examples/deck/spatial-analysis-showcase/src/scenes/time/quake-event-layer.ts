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
import {Buffer, type Device, type RenderPass} from '@luma.gl/core';
import {Model} from '@luma.gl/engine';
import type {AnalysisFrame} from '../../engine/analysis-frame';
import {getAnalysisFrameWgsl} from '../../engine/layer-wgsl';
import {COLORMAP_INDEXES, getRampWgsl, type RampName} from '../../engine/ramps';

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

const STYLE_BYTE_LENGTH = 128;

/** The ramp functions, rendered once. */
const RAMP_WGSL = getRampWgsl();

/** Clip position of a `METER_OFFSETS` event position. */
const METER_OFFSETS_CLIP_WGSL = /* wgsl */ `fn getQuakeClipPosition(position: vec2<f32>) -> vec4<f32> {
  var clipPosition = project_position_to_clipspace(vec3<f32>(position, 0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  // Deck's project32 matrices use OpenGL depth; WebGPU clip space requires [0, w].
  clipPosition.z = (clipPosition.z + clipPosition.w) * 0.5;
  return clipPosition;
}`;

/** Clip position of an analysis-frame event position: the engine's frame inverse. */
function getFrameClipWgsl(frame: AnalysisFrame): string {
  return /* wgsl */ `${getAnalysisFrameWgsl(frame)}
fn getQuakeClipPosition(position: vec2<f32>) -> vec4<f32> {
  return projectSpatialAnalysisPosition(position);
}`;
}

/** Shader of {@link QuakeEventLayer}; positions are `METER_OFFSETS` or `frame` meters. */
function getQuakeShader(frame: AnalysisFrame | undefined): string {
  return /* wgsl */ `
struct QuakeStyle {
  palette: array<vec4<f32>, 3>,
  outlineColor: vec4<f32>,
  playhead: f32,
  fadeDays: f32,
  sizePixels: f32,
  sizeGrowth: f32,
  opacity: f32,
  ghostAlpha: f32,
  depthMax: f32,
  popStrength: f32,
  colorMode: u32,
  colormap: u32,
  staticMode: u32,
  shallowLimit: f32,
  deepLimit: f32,
  minimumMagnitude: f32,
  _padding0: f32,
  _padding1: f32,
};

@group(0) @binding(auto) var<uniform> quakeStyle: QuakeStyle;
@group(0) @binding(auto) var<storage, read> quakePositions: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> quakeTimes: array<f32>;
@group(0) @binding(auto) var<storage, read> quakeMagnitudes: array<f32>;
@group(0) @binding(auto) var<storage, read> quakeDepths: array<f32>;

struct QuakeVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) corner: vec2<f32>,
  @location(1) color: vec4<f32>,
};

${RAMP_WGSL}

${frame ? getFrameClipWgsl(frame) : METER_OFFSETS_CLIP_WGSL}

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> QuakeVertexOutput {
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
  );
  var output: QuakeVertexOutput;
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  output.corner = vec2<f32>(0.0);
  output.color = vec4<f32>(0.0);
  let magnitude = quakeMagnitudes[instanceIndex];
  if (magnitude < quakeStyle.minimumMagnitude) {
    return output;
  }
  let age = quakeStyle.playhead - quakeTimes[instanceIndex];
  var alpha = 1.0;
  var scale = 1.0;
  var ageFraction = 0.0;
  if (quakeStyle.staticMode == 0u) {
    if (age < 0.0) {
      return output;
    }
    if (age > quakeStyle.fadeDays) {
      if (quakeStyle.ghostAlpha <= 0.0) {
        return output;
      }
      alpha = quakeStyle.ghostAlpha;
      scale = 0.55;
      ageFraction = 1.0;
    } else {
      ageFraction = age / max(quakeStyle.fadeDays, 1e-3);
      alpha = max(pow(1.0 - ageFraction, 1.4), quakeStyle.ghostAlpha);
      // A new event swells and settles within the first few percent of its life.
      scale = 1.0 + quakeStyle.popStrength * exp(-age / max(quakeStyle.fadeDays * 0.06, 0.02));
    }
  }
  let depth = quakeDepths[instanceIndex];
  var rgb = vec3<f32>(1.0);
  if (quakeStyle.colorMode == 0u) {
    rgb = spatialAnalysisSampleRamp(quakeStyle.colormap, 1.0 - clamp(depth / max(quakeStyle.depthMax, 1.0), 0.0, 1.0));
  } else if (quakeStyle.colorMode == 1u) {
    rgb = spatialAnalysisSampleRamp(quakeStyle.colormap, 1.0 - ageFraction);
  } else {
    var slot = 0u;
    if (depth >= quakeStyle.deepLimit) {
      slot = 2u;
    } else if (depth >= quakeStyle.shallowLimit) {
      slot = 1u;
    }
    rgb = quakeStyle.palette[slot].rgb;
  }
  let radius = quakeStyle.sizePixels * pow(quakeStyle.sizeGrowth, magnitude - 4.0) * scale;
  let corner = corners[vertexIndex];
  var clipPosition = getQuakeClipPosition(quakePositions[instanceIndex]);
  clipPosition = vec4<f32>(
    clipPosition.xy + project_pixel_size_to_clipspace(corner * radius),
    clipPosition.z,
    clipPosition.w
  );
  output.position = clipPosition;
  output.corner = corner;
  output.color = vec4<f32>(rgb, alpha);
  return output;
}

@fragment fn fragmentMain(input: QuakeVertexOutput) -> @location(0) vec4<f32> {
  let radiusSquared = dot(input.corner, input.corner);
  if (radiusSquared > 1.0) {
    discard;
  }
  let ring = smoothstep(0.55, 0.85, radiusSquared);
  let rgb = mix(input.color.rgb, quakeStyle.outlineColor.rgb, ring * quakeStyle.outlineColor.a);
  let cover = 1.0 - smoothstep(0.88, 1.0, radiusSquared);
  return vec4<f32>(rgb, input.color.a * quakeStyle.opacity * cover * mix(0.62, 1.0, ring));
}
`;
}

/** Color with 0-255 channels. */
export type QuakeColor = readonly [number, number, number, number?];

/** Props for {@link QuakeEventLayer}. */
export type QuakeEventLayerProps = LayerProps & {
  /** `float32x2` planar meters per event (`METER_OFFSETS`, or `analysisFrame` meters). */
  positions: Buffer;
  /**
   * Positions are meters of this analysis frame; the shader inverts it on the GPU and the layer
   * draws as `COORDINATE_SYSTEM.LNGLAT`, as the engine's spatial-analysis layers do with the same
   * prop. `coordinateSystem` and `coordinateOrigin` are then ignored.
   */
  analysisFrame?: AnalysisFrame;
  /** `float32` event time in days. */
  times: Buffer;
  /** `float32` magnitude per event. */
  magnitudes: Buffer;
  /** `float32` depth in km per event. */
  depths: Buffer;
  /** Number of events. */
  instanceCount: number;
  /** Playhead in days. Ignored when `staticMode` is true. */
  playhead?: number;
  /** Length of an event's life in days: it fades from full to nothing over it. */
  fadeDays?: number;
  /** Radius in CSS pixels of a magnitude 4 event. Defaults to 2.5. */
  sizePixels?: number;
  /** Radius multiplier per magnitude unit. Defaults to 1.9. */
  sizeGrowth?: number;
  /** Alpha of events outside their life (past ones); 0 hides them. Defaults to 0. */
  ghostAlpha?: number;
  /** Extra swell of a new event as a fraction of its radius. Defaults to 1.2. */
  popStrength?: number;
  /** `'depth'` and `'age'` use `ramp`; `'class'` uses `palette` with the two depth limits. */
  colorMode?: 'depth' | 'age' | 'class';
  /** Ramp for the depth and age modes. Defaults to `'magma'`. */
  ramp?: RampName;
  /** Depth in km at which the depth ramp ends. Defaults to 300. */
  depthMax?: number;
  /** Shallow, intermediate and deep colors of the class mode. */
  palette?: readonly [QuakeColor, QuakeColor, QuakeColor];
  /** Depth limits of the class mode in km. Defaults to 70 and 300. */
  classLimits?: readonly [number, number];
  /** Ring color around each event. */
  outlineColor?: QuakeColor;
  /** Draw every event at full life, ignoring time. */
  staticMode?: boolean;
  /** Hide events below this magnitude. Defaults to 0. */
  minimumMagnitude?: number;
};

type QuakeLayerState = {model: Model; styleBuffer: Buffer; source: string};

/**
 * Earthquake discs: radius grows with magnitude, color follows depth, age or depth class, and in
 * playback mode each event appears at its time, swells, then fades over `fadeDays`. Everything is
 * decided in the vertex shader from the catalog buffers and one style uniform, so moving the
 * playhead writes 128 bytes.
 */
export class QuakeEventLayer extends Layer<QuakeEventLayerProps> {
  static override layerName = 'QuakeEventLayer';
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
    const source = getQuakeShader(this.props.analysisFrame);
    this.setState({model: this.createModel(device, source, styleBuffer), styleBuffer, source});
  }

  override updateState(parameters: UpdateParameters<this>): void {
    super.updateState(parameters);
    const {model, styleBuffer, source} = this.state as QuakeLayerState;
    const nextSource = getQuakeShader(this.props.analysisFrame);
    if (nextSource !== source) {
      // A different analysis frame is a different shader.
      model.destroy();
      this.setState({
        model: this.createModel(this.context.device, nextSource, styleBuffer),
        source: nextSource
      });
      return;
    }
    model.setBindings(this.getBindings(styleBuffer));
  }

  /** The analysis-frame hook emits longitude/latitude, whatever `coordinateSystem` says. */
  override _drawLayer(options: Parameters<Layer['_drawLayer']>[0]): void {
    const {shaderModuleProps} = options;
    if (!this.props.analysisFrame || !shaderModuleProps?.project) {
      super._drawLayer(options);
      return;
    }
    super._drawLayer({
      ...options,
      shaderModuleProps: {
        ...shaderModuleProps,
        project: {
          ...shaderModuleProps.project,
          coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
          coordinateOrigin: [0, 0, 0]
        }
      }
    });
  }

  override getModels(): Model[] {
    return [(this.state as QuakeLayerState).model];
  }

  override draw({renderPass}: {renderPass: RenderPass}): void {
    const {model, styleBuffer} = this.state as QuakeLayerState;
    this.writeStyle(styleBuffer);
    model.setInstanceCount(this.props.instanceCount);
    model.draw(renderPass);
  }

  override finalizeState(context: LayerContext): void {
    const {model, styleBuffer} = this.state as QuakeLayerState;
    model.destroy();
    styleBuffer.destroy();
    super.finalizeState(context);
  }

  private createModel(device: Device, source: string, styleBuffer: Buffer): Model {
    return new Model(device, {
      ...this.getShaders({modules: [project32], source}),
      id: `${this.id}-model`,
      topology: 'triangle-list',
      isInstanced: true,
      vertexCount: 6,
      instanceCount: 0,
      bufferLayout: [],
      bindings: this.getBindings(styleBuffer),
      parameters: BLEND_PARAMETERS
    });
  }

  private getBindings(styleBuffer: Buffer): Record<string, Buffer> {
    return {
      quakeStyle: styleBuffer,
      quakePositions: this.props.positions,
      quakeTimes: this.props.times,
      quakeMagnitudes: this.props.magnitudes,
      quakeDepths: this.props.depths
    };
  }

  private writeStyle(styleBuffer: Buffer): void {
    const props = this.props;
    const data = new ArrayBuffer(STYLE_BYTE_LENGTH);
    const floats = new Float32Array(data);
    const words = new Uint32Array(data);
    const palette = props.palette ?? [
      [255, 154, 98],
      [78, 209, 181],
      [185, 149, 255]
    ];
    palette.forEach((color, slot) => {
      floats.set(
        [color[0] / 255, color[1] / 255, color[2] / 255, (color[3] ?? 255) / 255],
        slot * 4
      );
    });
    const outline = props.outlineColor ?? [10, 14, 22, 200];
    floats.set(
      [outline[0] / 255, outline[1] / 255, outline[2] / 255, (outline[3] ?? 255) / 255],
      12
    );
    const limits = props.classLimits ?? [70, 300];
    floats[16] = props.playhead ?? 0;
    floats[17] = props.fadeDays ?? 30;
    floats[18] = props.sizePixels ?? 2.5;
    floats[19] = props.sizeGrowth ?? 1.9;
    floats[20] = props.opacity ?? 1;
    floats[21] = props.ghostAlpha ?? 0;
    floats[22] = props.depthMax ?? 300;
    floats[23] = props.popStrength ?? 1.2;
    words[24] = props.colorMode === 'age' ? 1 : props.colorMode === 'class' ? 2 : 0;
    words[25] = COLORMAP_INDEXES[props.ramp ?? 'magma'];
    words[26] = props.staticMode ? 1 : 0;
    floats[27] = limits[0];
    floats[28] = limits[1];
    floats[29] = props.minimumMagnitude ?? 0;
    styleBuffer.write(new Uint8Array(data));
  }
}
