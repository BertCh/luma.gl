// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Mode-local Deck layer of the relief mode: a raster of packed RGBA8 relief colors (the
 * `GPUReliefShading.color` output, one `pack4x8unorm` word per cell) optionally multiplied by the
 * per-cell sun illumination of `GPUSolarShadowMask`, so the "Light and shadow" composite never
 * leaves the GPU. The stock raster layer maps scalars through colormaps and cannot unpack colors.
 */

import type {Buffer} from '@luma.gl/core';
import {
  STORAGE_LAYER_PROJECTION_WGSL,
  StorageModelLayer,
  type StorageModelLayerProps
} from './flow-field-layers';

const SHADER = /* wgsl */ `
struct LayerStyle {
  minimum: vec2<f32>,
  maximum: vec2<f32>,
  gridSize: vec2<u32>,
  lightStrength: f32,
  opacity: f32,
  lightGain: f32,
  padding0: f32,
  padding1: f32,
  padding2: f32,
};
@group(0) @binding(auto) var<uniform> layerStyle: LayerStyle;
@group(0) @binding(auto) var<storage, read> reliefColors: array<u32>;
@group(0) @binding(auto) var<storage, read> reliefLight: array<f32>;
${STORAGE_LAYER_PROJECTION_WGSL}

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) local: vec2<f32>,
};

@vertex fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0)
  );
  let corner = corners[vertexIndex];
  var output: VertexOutput;
  output.position = projectStoragePosition(mix(layerStyle.minimum, layerStyle.maximum, corner));
  output.local = corner;
  return output;
}

@fragment fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
  let columns = i32(layerStyle.gridSize.x);
  let rows = i32(layerStyle.gridSize.y);
  let column = clamp(i32(floor(input.local.x * f32(columns))), 0, columns - 1);
  // Row 0 is the north edge (the raster the contributors read), so flip the row.
  let row = clamp(rows - 1 - i32(floor(input.local.y * f32(rows))), 0, rows - 1);
  let index = u32(row * columns + column);
  var color = unpack4x8unorm(reliefColors[index]);
  if (color.a <= 0.0) {
    discard;
  }
  if (layerStyle.lightStrength > 0.0) {
    var light = reliefLight[index];
    if (isNonFiniteStorageFloat(light)) {
      light = 0.0;
    }
    color = vec4<f32>(color.rgb * mix(1.0, clamp(light * layerStyle.lightGain, 0.0, 1.0), layerStyle.lightStrength), color.a);
  }
  return vec4<f32>(color.rgb, color.a * layerStyle.opacity);
}
`;

/** Props for {@link ReliefRasterLayer}. */
export type ReliefRasterLayerProps = StorageModelLayerProps & {
  /** Packed RGBA8 colors (`uint32`, red in the low byte), row 0 at the north edge. */
  colors: Buffer;
  /** Optional illumination floats, same layout; multiplied into the color by `lightStrength`. */
  light?: Buffer | null;
  /** Share of the illumination applied to the color, 0 (none) to 1 (full). Defaults to 0. */
  lightStrength?: number;
  /** `[columns, rows]` of the raster. */
  gridSize: readonly [number, number];
  /** `[minX, minY, maxX, maxY]` meters of the raster's outer cell edges. */
  bounds: readonly [number, number, number, number];
  /** Layer alpha. Defaults to 1. */
  opacity?: number;
  /** Multiplier on the illumination before it is clamped to `[0, 1]` (exposure). Defaults to 1. */
  lightGain?: number;
};

/** One quad over the raster bounds whose fragments unpack a relief color from storage. */
export class ReliefRasterLayer extends StorageModelLayer<ReliefRasterLayerProps> {
  static override layerName = 'ReliefRasterLayer';

  protected getShaderSource(): string {
    return SHADER;
  }
  protected getStyleByteLength(): number {
    return 48;
  }
  protected getInstanceCount(): number {
    return 1;
  }
  protected getStorageBindings(placeholder: Buffer): Record<string, Buffer> {
    return {reliefColors: this.props.colors, reliefLight: this.props.light ?? placeholder};
  }
  protected writeStyle(styleBuffer: Buffer): void {
    const {bounds, gridSize, lightStrength = 0, opacity = 1, lightGain = 1} = this.props;
    const data = new ArrayBuffer(48);
    new Float32Array(data).set([bounds[0], bounds[1], bounds[2], bounds[3]]);
    new Uint32Array(data).set(gridSize, 4);
    const floats = new Float32Array(data);
    floats[6] = this.props.light ? lightStrength : 0;
    floats[7] = opacity;
    floats[8] = lightGain;
    styleBuffer.write(new Uint8Array(data));
  }
}
