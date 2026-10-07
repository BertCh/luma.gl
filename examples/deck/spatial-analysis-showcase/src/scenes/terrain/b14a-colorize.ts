// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * The colorize pass of the terrain chapter: one compute node that turns a product raster (float32
 * values or uint32 class codes) into packed RGBA8 colors. Ramp colors come from the showcase's one
 * ramp table (`engine/ramps.ts`), so a legend and the map cannot drift apart. The pass is its own
 * tiny graph so a ramp, range or opacity change re-encodes only this pass, never the analysis.
 */

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import type {GPUParameterBuffer} from '@luma.gl/experimental/gpu-spatial-analysis';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {addKernelPass} from '../../engine/mode-kernels';
import {COLORMAP_INDEXES, getRampWgsl, type RampName} from '../../engine/ramps';
import type {SpatialAnalysisResources} from '../../engine/resources';

/** RGB color, 0-255. */
export type Rgb = readonly [number, number, number];

/** How a value becomes a color. */
export type PaintMode = 'ramp' | 'aspect' | 'classes';

/** Everything the colorize pass reads per frame. */
export type PaintSpec = {
  mode: PaintMode;
  ramp: RampName;
  /** Value mapped to the first ramp stop. */
  low: number;
  /** Value mapped to the last ramp stop. */
  high: number;
  /** Alpha of every colored cell, 0 to 1. */
  alpha: number;
  /** Ramp mode: fade the middle of a diverging ramp so the relief underneath shows through. */
  fadeMiddle?: boolean;
  /** Ramp mode: gamma applied to the normalized value before the ramp (1 = linear). */
  gamma?: number;
};

/** Floats of the paint parameter buffer. */
export const PAINT_PARAMETER_LENGTH = 8;

/** Writes a paint spec into its parameter buffer. */
export function writePaint(buffer: GPUParameterBuffer<'float32'>, paint: PaintSpec): void {
  buffer.write(
    Float32Array.of(
      paint.mode === 'ramp' ? 0 : paint.mode === 'aspect' ? 1 : 2,
      COLORMAP_INDEXES[paint.ramp],
      paint.low,
      paint.high,
      paint.alpha,
      paint.fadeMiddle ? 1 : 0,
      paint.gamma ?? 1,
      0
    )
  );
}

/** Saturation and value of the aspect wheel. */
const ASPECT_SATURATION = 0.62;
const ASPECT_VALUE = 0.95;

/** Color of an aspect (degrees clockwise from north) on the hue wheel; TypeScript twin of the WGSL. */
export function getAspectColor(aspectDegrees: number): Rgb {
  const hue = (((aspectDegrees % 360) + 360) % 360) / 360;
  const channel = (n: number) => {
    const k = (n + hue * 6) % 6;
    return (
      (ASPECT_VALUE - ASPECT_VALUE * ASPECT_SATURATION * Math.max(Math.min(k, 4 - k, 1), 0)) * 255
    );
  };
  return [Math.round(channel(5)), Math.round(channel(3)), Math.round(channel(1))];
}

/** Input of a colorize pass. */
export type ColorizeInput = {
  buffer: Buffer;
  format: 'float32' | 'uint32';
  length: number;
};

/**
 * Builds the colorize graph for one product raster.
 *
 * @param palette Class colors, index 0 is class 1. Used when `mode` is `'classes'`; class 0 (and
 *   non-finite floats) are transparent.
 */
export function createColorizeGraph(
  resources: SpatialAnalysisResources,
  device: Device,
  id: string,
  input: ColorizeInput,
  parameters: GPUParameterBuffer<'float32'>,
  colors: Buffer,
  palette: readonly Rgb[] = []
): CompiledGPUCommandGraph<void> {
  const graph = new GPUCommandGraph<void>(device, {id: `${id}-paint`});
  const isFloat = input.format === 'float32';
  const paletteSize = Math.max(1, palette.length);
  const paletteWgsl = (palette.length > 0 ? palette : [[0, 0, 0] as Rgb])
    .map(
      ([r, g, b]) =>
        `vec3<f32>(${(r / 255).toFixed(4)}, ${(g / 255).toFixed(4)}, ${(b / 255).toFixed(4)})`
    )
    .join(', ');
  addKernelPass(graph, {
    id: `${id}-colorize`,
    invocationCount: input.length,
    bindings: [
      {
        name: 'source',
        view: importGraphBuffer(graph, 'source', input.buffer, input.format, input.length),
        type: isFloat ? 'f32' : 'u32',
        access: 'read'
      },
      {
        name: 'paint',
        view: parameters.importToGraph(graph),
        type: 'f32',
        access: 'read'
      },
      {
        name: 'colors',
        view: importGraphBuffer(graph, 'colors', colors, 'uint32', input.length),
        type: 'u32',
        access: 'read_write'
      }
    ],
    declarations: /* wgsl */ `
${getRampWgsl()}
var<private> CLASS_COLORS: array<vec3<f32>, ${paletteSize}> = array<vec3<f32>, ${paletteSize}>(${paletteWgsl});
const CLASS_COUNT: u32 = ${palette.length}u;
const ASPECT_SATURATION: f32 = ${ASPECT_SATURATION};
const ASPECT_VALUE: f32 = ${ASPECT_VALUE};
fn isBad(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u;
}
fn aspectChannel(n: f32, hue: f32) -> f32 {
  let shifted = n + hue * 6.0;
  let k = shifted - 6.0 * floor(shifted / 6.0);
  return ASPECT_VALUE - ASPECT_VALUE * ASPECT_SATURATION * clamp(min(k, 4.0 - k), 0.0, 1.0);
}`,
    body: /* wgsl */ `
  let mode = u32(paint[paintOffset]);
  let rampIndex = u32(paint[paintOffset + 1u]);
  let low = paint[paintOffset + 2u];
  let high = paint[paintOffset + 3u];
  let alpha = paint[paintOffset + 4u];
  var color = vec4<f32>(0.0);
  ${
    isFloat
      ? `
  let value = source[sourceOffset + index];
  if (!isBad(value)) {
    if (mode == 1u) {
      if (value >= 0.0) {
        let hue = fract(value / 360.0);
        color = vec4<f32>(aspectChannel(5.0, hue), aspectChannel(3.0, hue), aspectChannel(1.0, hue), alpha);
      } else {
        color = vec4<f32>(0.78, 0.78, 0.78, alpha);
      }
    } else {
      let t = clamp((value - low) / max(high - low, 1e-30), 0.0, 1.0);
      let shaped = pow(t, max(paint[paintOffset + 6u], 0.05));
      var cellAlpha = alpha;
      if (paint[paintOffset + 5u] > 0.5) {
        cellAlpha = alpha * mix(0.18, 1.0, abs(2.0 * t - 1.0));
      }
      color = vec4<f32>(spatialAnalysisSampleRamp(rampIndex, shaped), cellAlpha);
    }
  }`
      : `
  let code = source[sourceOffset + index];
  if (mode == 2u) {
    if (code >= 1u && code <= CLASS_COUNT) {
      color = vec4<f32>(CLASS_COLORS[code - 1u], alpha);
    }
  } else if (code > 0u) {
    // Code rasters shown on a ramp (for example geomorphon ternary patterns); 0 is nodata.
    let t = clamp((f32(code) - low) / max(high - low, 1e-30), 0.0, 1.0);
    color = vec4<f32>(spatialAnalysisSampleRamp(rampIndex, t), alpha);
  }`
  }
  colors[colorsOffset + index] = pack4x8unorm(color);`
  });
  return resources.track(graph.compile());
}
