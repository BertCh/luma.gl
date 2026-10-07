// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * The paint pass of terrain-basics: one small compute kernel that turns the float outputs of the
 * contributors into packed RGBA8 cells, with the class table, the breaks and the ramp read from a
 * parameter buffer, so a ground change, a new class table or frozen quantile breaks are buffer
 * writes and never a recompile.
 *
 * Three modes share one kernel (`classes`, `ramp`, `aspect`); the chapter colorize pass
 * (`b14a-colorize.ts`) cannot class a float raster or weight aspect by slope, which the story
 * needs. The kernel reads one or two float rasters: `classes` and `ramp` read the first, or the
 * second when `useSecond` is set (the Mercator slope next to the ground slope); `aspect` reads the
 * aspect from the first and the slope from the second.
 */

import type {GraphDataView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {addKernelPass} from '../../engine/mode-kernels';
import {COLORMAP_INDEXES, getRampWgsl, type RampName} from '../../engine/ramps';

/** Most classes one paint pass holds (seven slope classes fit). */
export const MAXIMUM_PAINT_CLASSES = 8;

/** Floats of the paint parameter buffer. */
export const PAINT_PARAMETER_LENGTH = 52;

const BREAKS_OFFSET = 12;
const COLORS_OFFSET = 20;
/** A break no value reaches: classes above the table's last real break stay empty. */
const UNREACHABLE_BREAK = 3.0e38;

/** How the paint pass colours a cell. */
export type PaintMode = 'classes' | 'ramp' | 'aspect';

/** Everything the paint kernel reads per run. */
export type PaintSettings = {
  mode: PaintMode;
  /** Interior class breaks, ascending, at most seven (classes mode). */
  breaks?: readonly number[];
  /** One RGBA colour per class, 0-255 channels, low class first (classes mode). */
  colors?: readonly (readonly [number, number, number, number?])[];
  /** Read the second raster instead of the first (classes and ramp modes). */
  useSecond?: boolean;
  ramp?: RampName;
  /** Ramp mode: value at the ramp start and end, and the value below which cells are not drawn. */
  low?: number;
  high?: number;
  floor?: number;
  /** Ramp and aspect modes: opacity of drawn cells, 0-1. */
  alpha?: number;
  /** Aspect mode: slope (degrees) where the colour starts to appear and where it is full. */
  flatDegrees?: number;
  fullDegrees?: number;
};

/** Packs {@link PaintSettings} into the parameter layout the kernel reads. */
export function getPaintParameterValues(
  settings: PaintSettings,
  target: Float32Array = new Float32Array(PAINT_PARAMETER_LENGTH)
): Float32Array {
  const colors = settings.colors ?? [];
  const classCount = Math.min(MAXIMUM_PAINT_CLASSES, Math.max(1, colors.length));
  target.fill(0);
  target[0] = settings.mode === 'classes' ? 0 : settings.mode === 'ramp' ? 1 : 2;
  target[1] = classCount;
  target[2] = settings.useSecond ? 1 : 0;
  target[3] = settings.ramp ? COLORMAP_INDEXES[settings.ramp] : 0;
  target[4] = settings.low ?? 0;
  target[5] = settings.high ?? 1;
  target[6] = settings.floor ?? -UNREACHABLE_BREAK;
  target[7] = settings.alpha ?? 1;
  target[8] = settings.flatDegrees ?? 5;
  target[9] = settings.fullDegrees ?? 25;
  for (let index = 0; index < MAXIMUM_PAINT_CLASSES - 1; index++) {
    const value = settings.breaks?.[index];
    target[BREAKS_OFFSET + index] =
      value !== undefined && Number.isFinite(value) ? value : UNREACHABLE_BREAK;
  }
  for (let index = 0; index < classCount; index++) {
    const color = colors[index] ?? [0, 0, 0, 0];
    target[COLORS_OFFSET + index * 4] = color[0] / 255;
    target[COLORS_OFFSET + index * 4 + 1] = color[1] / 255;
    target[COLORS_OFFSET + index * 4 + 2] = color[2] / 255;
    target[COLORS_OFFSET + index * 4 + 3] = (color[3] ?? 255) / 255;
  }
  return target;
}

/** Inputs of {@link addPaintPass}. */
export type PaintPassProps = {
  id: string;
  /** Number of cells. */
  count: number;
  /** The raster painted (aspect in aspect mode). */
  first: GraphDataView<'float32'>;
  /** The second raster (slope in aspect mode, the Mercator slope next to the ground one). */
  second?: GraphDataView<'float32'>;
  /** Paint parameters, `PAINT_PARAMETER_LENGTH` floats. */
  settings: GraphDataView<'float32'>;
  /** Packed RGBA8 output (red in the low byte); alpha 0 cells are not drawn. */
  output: GraphDataView<'uint32'>;
};

/** Adds the paint kernel to a graph, after the nodes that write its inputs. */
export function addPaintPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: PaintPassProps
): void {
  const hasSecond = props.second !== undefined;
  addKernelPass(graph, {
    id: props.id,
    invocationCount: props.count,
    bindings: [
      {name: 'firstRaster', view: props.first, type: 'f32', access: 'read'},
      ...(props.second
        ? [
            {
              name: 'secondRaster',
              view: props.second,
              type: 'f32' as const,
              access: 'read' as const
            }
          ]
        : []),
      {name: 'paint', view: props.settings, type: 'f32', access: 'read'},
      {name: 'cells', view: props.output, type: 'u32', access: 'read_write'}
    ],
    declarations: /* wgsl */ `
${getRampWgsl()}
const BREAKS_OFFSET: u32 = ${BREAKS_OFFSET}u;
const COLORS_OFFSET: u32 = ${COLORS_OFFSET}u;
const BREAK_COUNT: u32 = ${MAXIMUM_PAINT_CLASSES - 1}u;
fn isNotFinite(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u;
}
fn getClassOf(value: f32, classCount: u32) -> u32 {
  var found = 0u;
  for (var breakIndex = 0u; breakIndex < BREAK_COUNT; breakIndex = breakIndex + 1u) {
    if (breakIndex + 1u < classCount && value >= paint[paintOffset + BREAKS_OFFSET + breakIndex]) {
      found = breakIndex + 1u;
    }
  }
  return found;
}`,
    body: /* wgsl */ `
  let mode = u32(paint[paintOffset]);
  let classCount = u32(paint[paintOffset + 1u]);
  let useSecond = paint[paintOffset + 2u] > 0.5;
  let rampIndex = u32(paint[paintOffset + 3u]);
  let rampLow = paint[paintOffset + 4u];
  let rampHigh = paint[paintOffset + 5u];
  let rampFloor = paint[paintOffset + 6u];
  let alpha = paint[paintOffset + 7u];
  let firstValue = firstRaster[firstRasterOffset + index];
  ${hasSecond ? 'let secondValue = secondRaster[secondRasterOffset + index];' : 'let secondValue = firstValue;'}
  var color = vec4<f32>(0.0);
  if (mode == 2u) {
    // Aspect hue, fading in with the slope; flat ground (aspect -1) and no-data stay empty.
    if (!isNotFinite(firstValue) && !isNotFinite(secondValue) && firstValue >= 0.0) {
      let flatDegrees = paint[paintOffset + 8u];
      let fullDegrees = paint[paintOffset + 9u];
      let weight = clamp((secondValue - flatDegrees) / max(fullDegrees - flatDegrees, 1e-6), 0.0, 1.0);
      let hue = fract(firstValue / 360.0);
      color = vec4<f32>(spatialAnalysisSampleRamp(rampIndex, hue), alpha * weight);
    }
  } else {
    let value = select(firstValue, secondValue, useSecond);
    if (!isNotFinite(value)) {
      if (mode == 1u) {
        if (value >= rampFloor) {
          let fraction = clamp((value - rampLow) / max(rampHigh - rampLow, 1e-6), 0.0, 1.0);
          color = vec4<f32>(spatialAnalysisSampleRamp(rampIndex, fraction), alpha);
        }
      } else {
        let found = getClassOf(value, classCount);
        let base = paintOffset + COLORS_OFFSET + found * 4u;
        color = vec4<f32>(paint[base], paint[base + 1u], paint[base + 2u], paint[base + 3u]);
      }
    }
  }
  cells[cellsOffset + index] = pack4x8unorm(color);`
  });
}
