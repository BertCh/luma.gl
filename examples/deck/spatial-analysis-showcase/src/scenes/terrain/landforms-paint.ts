// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * The table-driven colorize pass of the landforms story. One compute node turns a product raster
 * (uint32 class codes, or float32 values classed at the breaks of a table) into packed RGBA8 colours
 * read straight from a class table, alpha included, so the layer and the legend cannot drift apart
 * and a change of table, ground or isolated class is a parameter write, never a recompile. Peak and
 * pit codes can be enlarged by a few cells so single 6.6 m cells stay visible at low zoom.
 */

import type {Buffer, Device} from '@luma.gl/core';
import type {GPUParameterBuffer} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import type {ClassTable} from '../../cartography/types';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {addKernelPass} from '../../engine/mode-kernels';
import type {SpatialAnalysisResources} from '../../engine/resources';

/** Floats of the paint parameter buffer. */
export const TABLE_PAINT_PARAMETER_LENGTH = 96;

const BREAK_OFFSET = 8;
const COLOR_OFFSET = 24;
const MAXIMUM_CLASSES = 16;

/** Alpha multiplier of the classes a legend isolation dims. */
const DIM_ALPHA = 0.12;

/** What the paint pass reads besides the table. */
export type TablePaintOptions = {
  /** `'codes'`: the uint32 code is the class index. `'breaks'`: float values classed at the breaks. */
  mode: 'codes' | 'breaks';
  /** Cells around a peak or pit that take its colour (0 = off). Codes mode only. */
  dilateRadius?: number;
  peakCode?: number;
  pitCode?: number;
  /** Class indices kept at full alpha; the others are dimmed. `null` shows everything. */
  isolate?: readonly number[] | null;
};

/** Writes a class table and its options into the paint parameter buffer. */
export function writeTablePaint(
  buffer: GPUParameterBuffer<'float32'>,
  table: ClassTable,
  options: TablePaintOptions
): void {
  const values = new Float32Array(TABLE_PAINT_PARAMETER_LENGTH);
  const classCount = Math.min(MAXIMUM_CLASSES, table.colors.length);
  let mask = 0;
  for (const index of options.isolate ?? []) {
    if (index >= 0 && index < MAXIMUM_CLASSES) mask |= 1 << index;
  }
  values[0] = options.mode === 'codes' ? 0 : 1;
  values[1] = classCount;
  values[2] = options.dilateRadius ?? 0;
  values[3] = options.peakCode ?? 0;
  values[4] = options.pitCode ?? 0;
  values[5] = mask;
  values[6] = DIM_ALPHA;
  table.breaks.slice(0, MAXIMUM_CLASSES - 1).forEach((value, index) => {
    values[BREAK_OFFSET + index] = value;
  });
  for (let index = 0; index < classCount; index++) {
    const color = table.colors[index];
    values[COLOR_OFFSET + index * 4] = color[0] / 255;
    values[COLOR_OFFSET + index * 4 + 1] = color[1] / 255;
    values[COLOR_OFFSET + index * 4 + 2] = color[2] / 255;
    values[COLOR_OFFSET + index * 4 + 3] = (color[3] ?? 255) / 255;
  }
  buffer.write(values);
}

/** Input of a table paint pass. */
export type TablePaintInput = {
  buffer: Buffer;
  format: 'float32' | 'uint32';
  width: number;
  height: number;
};

/**
 * Builds the paint graph of one product raster: `colors[index]` = packed RGBA of the cell's class
 * in the table. Non-finite floats, codes beyond the table and class alpha 0 are transparent.
 *
 * @param id Debug id, unique per graph.
 * @param parameters The buffer {@link writeTablePaint} fills.
 * @param colors The packed output, one uint32 per cell, which the layer draws.
 */
export function createTablePaintGraph(
  resources: SpatialAnalysisResources,
  device: Device,
  id: string,
  input: TablePaintInput,
  parameters: GPUParameterBuffer<'float32'>,
  colors: Buffer
): CompiledGPUCommandGraph<void> {
  const length = input.width * input.height;
  const graph = new GPUCommandGraph<void>(device, {id: `${id}-paint`});
  const isFloat = input.format === 'float32';
  addKernelPass(graph, {
    id: `${id}-table-paint`,
    invocationCount: length,
    bindings: [
      {
        name: 'source',
        view: importGraphBuffer(graph, 'source', input.buffer, input.format, length),
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
        view: importGraphBuffer(graph, 'colors', colors, 'uint32', length),
        type: 'u32',
        access: 'read_write'
      }
    ],
    declarations: /* wgsl */ `
const WIDTH: i32 = ${input.width};
const HEIGHT: i32 = ${input.height};
fn isBad(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u;
}
fn tableColor(classIndex: u32) -> vec4<f32> {
  let base = paintOffset + ${COLOR_OFFSET}u + classIndex * 4u;
  return vec4<f32>(paint[base], paint[base + 1u], paint[base + 2u], paint[base + 3u]);
}`,
    body: /* wgsl */ `
  let classCount = u32(paint[paintOffset + 1u]);
  var classIndex = 0xffffffffu;
  ${
    isFloat
      ? /* wgsl */ `
  let value = source[sourceOffset + index];
  if (!isBad(value)) {
    classIndex = 0u;
    for (var breakIndex = 0u; breakIndex + 1u < classCount; breakIndex = breakIndex + 1u) {
      if (value >= paint[paintOffset + ${BREAK_OFFSET}u + breakIndex]) {
        classIndex = breakIndex + 1u;
      }
    }
  }`
      : /* wgsl */ `
  let own = source[sourceOffset + index];
  classIndex = own;
  let reach = i32(paint[paintOffset + 2u]);
  let peakCode = u32(paint[paintOffset + 3u]);
  let pitCode = u32(paint[paintOffset + 4u]);
  if (reach > 0 && own != peakCode && own != pitCode) {
    let column = i32(index) % WIDTH;
    let row = i32(index) / WIDTH;
    var foundPeak = false;
    var foundPit = false;
    for (var rowOffset = -reach; rowOffset <= reach; rowOffset = rowOffset + 1) {
      for (var columnOffset = -reach; columnOffset <= reach; columnOffset = columnOffset + 1) {
        let neighbourColumn = column + columnOffset;
        let neighbourRow = row + rowOffset;
        if (neighbourColumn < 0 || neighbourRow < 0 || neighbourColumn >= WIDTH || neighbourRow >= HEIGHT) {
          continue;
        }
        let neighbour = source[sourceOffset + u32(neighbourRow * WIDTH + neighbourColumn)];
        foundPeak = foundPeak || neighbour == peakCode;
        foundPit = foundPit || neighbour == pitCode;
      }
    }
    if (foundPeak) {
      classIndex = peakCode;
    } else if (foundPit) {
      classIndex = pitCode;
    }
  }`
  }
  var color = vec4<f32>(0.0);
  if (classIndex < classCount) {
    color = tableColor(classIndex);
    let isolated = u32(paint[paintOffset + 5u]);
    if (isolated != 0u && ((isolated >> classIndex) & 1u) == 0u) {
      color.a = color.a * paint[paintOffset + 6u];
    }
  }
  colors[colorsOffset + index] = pack4x8unorm(color);`
  });
  return resources.track(graph.compile());
}
