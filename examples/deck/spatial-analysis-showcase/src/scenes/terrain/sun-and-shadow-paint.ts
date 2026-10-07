// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * The paint pass of sun-and-shadow: one small compute kernel that turns a float raster of the
 * contributors into packed RGBA8 cells, with the veil colour, the light colours, the class table
 * and the anchors of a continuous gradient read from a parameter buffer. A ground change, a new
 * scale or a new exposure is therefore a buffer write and never a recompile.
 *
 * Four modes share the kernel:
 * - `veil`: sun visibility (0 = shadow, 1 = sun) becomes the indigo veil whose opacity is the
 *   hidden share of the sun disc, so sunlit cells are transparent and the penumbra blends;
 * - `light`: illumination becomes a multiply colour, cool in the shade and warm in the sun, that
 *   `ColorRasterLayer` multiplies into the relief ground;
 * - `classes`: the value (times `scale`) is classed at the table breaks;
 * - `gradient`: the table colours interpolated between the class anchors (continuous colour).
 */

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {addKernelPass} from '../../engine/mode-kernels';

/** Most classes one paint pass holds (the June sun-hours scale has ten). */
export const MAXIMUM_PAINT_CLASSES = 12;

/** Floats of the paint parameter buffer. */
export const SUN_PAINT_PARAMETER_LENGTH = 96;

const COLOR_A_OFFSET = 8;
const COLOR_B_OFFSET = 12;
const BREAKS_OFFSET = 16;
const COLORS_OFFSET = 32;
const ANCHORS_OFFSET = 80;

/** An RGBA colour with 0-255 channels. */
type Color255 = readonly [number, number, number, number?];

/** Everything the paint kernel reads per run. */
export type SunPaintSettings =
  | {
      mode: 'veil';
      /** The shadow colour; its alpha is the opacity of full shadow. */
      color: Color255;
    }
  | {
      mode: 'light';
      /** Illumination multiplier before it is clamped to 0-1. */
      exposure: number;
      /** Multiply colour in full sun, channels 0-1. */
      lit: readonly [number, number, number];
      /** Multiply colour in full shade, channels 0-1. */
      shade: readonly [number, number, number];
    }
  | {
      mode: 'classes';
      /** Interior class breaks in the unit of `value * scale`, ascending. */
      breaks: readonly number[];
      colors: readonly Color255[];
      /** Multiplier from the raster value to the table unit (Wh to kWh is 0.001). Defaults to 1. */
      scale?: number;
    }
  | {
      mode: 'gradient';
      /** The value of each class colour, ascending, one per colour. */
      anchors: readonly number[];
      colors: readonly Color255[];
      scale?: number;
    };

/** Packs {@link SunPaintSettings} into the parameter layout the kernel reads. */
export function getSunPaintParameters(
  settings: SunPaintSettings,
  target: Float32Array = new Float32Array(SUN_PAINT_PARAMETER_LENGTH)
): Float32Array {
  target.fill(0);
  switch (settings.mode) {
    case 'veil': {
      target[0] = 2;
      const [r, g, b, a = 255] = settings.color;
      target.set([r / 255, g / 255, b / 255, a / 255], COLOR_A_OFFSET);
      break;
    }
    case 'light':
      target[0] = 3;
      target[3] = settings.exposure;
      target.set(settings.lit, COLOR_A_OFFSET);
      target.set(settings.shade, COLOR_B_OFFSET);
      break;
    case 'classes':
    case 'gradient': {
      const classCount = Math.min(MAXIMUM_PAINT_CLASSES, settings.colors.length);
      target[0] = settings.mode === 'classes' ? 0 : 1;
      target[1] = classCount;
      target[2] = settings.scale ?? 1;
      for (let index = 0; index < classCount; index++) {
        const [r, g, b, a = 255] = settings.colors[index];
        target.set([r / 255, g / 255, b / 255, a / 255], COLORS_OFFSET + index * 4);
      }
      const values = settings.mode === 'classes' ? settings.breaks : settings.anchors;
      for (let index = 0; index < Math.min(values.length, MAXIMUM_PAINT_CLASSES - 1); index++) {
        target[(settings.mode === 'classes' ? BREAKS_OFFSET : ANCHORS_OFFSET) + index] =
          values[index];
      }
      if (settings.mode === 'gradient') {
        // The gradient reads one anchor per colour, including the last.
        target[ANCHORS_OFFSET + classCount - 1] = settings.anchors[classCount - 1];
      }
      break;
    }
  }
  return target;
}

/** Inputs of {@link addSunPaintPass}. */
export type SunPaintPassProps = {
  id: string;
  /** Number of cells. */
  count: number;
  /** The raster painted. */
  source: GraphDataView<'float32'>;
  /** Paint parameters, `SUN_PAINT_PARAMETER_LENGTH` floats. */
  settings: GraphDataView<'float32'>;
  /** Packed RGBA8 output (red in the low byte); alpha 0 cells are not drawn. */
  output: GraphDataView<'uint32'>;
};

/** Adds the paint kernel to a graph, after the nodes that write its input. */
export function addSunPaintPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: SunPaintPassProps
): void {
  addKernelPass(graph, {
    id: props.id,
    invocationCount: props.count,
    bindings: [
      {name: 'source', view: props.source, type: 'f32', access: 'read'},
      {name: 'paint', view: props.settings, type: 'f32', access: 'read'},
      {name: 'cells', view: props.output, type: 'u32', access: 'read_write'}
    ],
    declarations: /* wgsl */ `
const COLOR_A_OFFSET: u32 = ${COLOR_A_OFFSET}u;
const COLOR_B_OFFSET: u32 = ${COLOR_B_OFFSET}u;
const BREAKS_OFFSET: u32 = ${BREAKS_OFFSET}u;
const COLORS_OFFSET: u32 = ${COLORS_OFFSET}u;
const ANCHORS_OFFSET: u32 = ${ANCHORS_OFFSET}u;
fn isNotFinite(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u;
}
fn getTableColor(slot: u32) -> vec4<f32> {
  let base = paintOffset + COLORS_OFFSET + slot * 4u;
  return vec4<f32>(paint[base], paint[base + 1u], paint[base + 2u], paint[base + 3u]);
}`,
    body: /* wgsl */ `
  let mode = u32(paint[paintOffset]);
  let classCount = u32(paint[paintOffset + 1u]);
  let scale = paint[paintOffset + 2u];
  let exposure = paint[paintOffset + 3u];
  let raw = source[sourceOffset + index];
  var color = vec4<f32>(0.0);
  if (!isNotFinite(raw)) {
    if (mode == 2u) {
      // Veil: the hidden share of the sun disc is the opacity of the shadow colour.
      let hiddenShare = 1.0 - clamp(raw, 0.0, 1.0);
      let veilBase = paintOffset + COLOR_A_OFFSET;
      color = vec4<f32>(paint[veilBase], paint[veilBase + 1u], paint[veilBase + 2u], paint[veilBase + 3u] * hiddenShare);
    } else if (mode == 3u) {
      // Multiply colour: cool shade to warm sun by the clamped illumination.
      let lit = clamp(raw * exposure, 0.0, 1.0);
      let litBase = paintOffset + COLOR_A_OFFSET;
      let shadeBase = paintOffset + COLOR_B_OFFSET;
      let litColor = vec3<f32>(paint[litBase], paint[litBase + 1u], paint[litBase + 2u]);
      let shadeColor = vec3<f32>(paint[shadeBase], paint[shadeBase + 1u], paint[shadeBase + 2u]);
      color = vec4<f32>(mix(shadeColor, litColor, lit), 1.0);
    } else {
      let value = raw * scale;
      if (mode == 0u) {
        var found = 0u;
        for (var slot = 0u; slot + 1u < classCount; slot = slot + 1u) {
          if (value >= paint[paintOffset + BREAKS_OFFSET + slot]) {
            found = slot + 1u;
          }
        }
        color = getTableColor(found);
      } else {
        color = getTableColor(0u);
        for (var slot = 0u; slot + 1u < classCount; slot = slot + 1u) {
          let low = paint[paintOffset + ANCHORS_OFFSET + slot];
          let high = paint[paintOffset + ANCHORS_OFFSET + slot + 1u];
          if (value >= low) {
            let position = clamp((value - low) / max(high - low, 1e-6), 0.0, 1.0);
            color = mix(getTableColor(slot), getTableColor(slot + 1u), position);
          }
        }
      }
    }
  }
  cells[cellsOffset + index] = pack4x8unorm(color);`
  });
}
