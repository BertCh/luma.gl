// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Scene-local colorize pass of the relief-visualization scene: a float raster (the signed local
 * relief in metres) is classed against a table of breaks and exact RGBA colours that lives in a
 * parameter buffer, so the legend, the tooltip and the map read one `ClassTable` and a change of
 * breaks is a buffer write. The chapter colorize pass (`b14a-colorize.ts`) classes code rasters
 * only; this pass reads values.
 *
 * Generalisation for the common build: `createColorizeGraph` could take `mode: 'value-classes'`
 * with this table layout.
 */

import type {Buffer, Device} from '@luma.gl/core';
import type {GPUParameterBuffer} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import type {ClassTable} from '../../cartography/types';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {addKernelPass} from '../../engine/mode-kernels';
import type {SpatialAnalysisResources} from '../../engine/resources';

/** Most classes the table buffer holds. */
export const MAXIMUM_VALUE_CLASSES = 16;

const BREAKS_OFFSET = 4;
const COLORS_OFFSET = BREAKS_OFFSET + MAXIMUM_VALUE_CLASSES;

/** Floats of the class table parameter buffer: count, 3 spare, 16 breaks, 16 RGBA colours. */
export const VALUE_CLASS_TABLE_LENGTH = COLORS_OFFSET + 4 * MAXIMUM_VALUE_CLASSES;

/**
 * Packs a class table into the buffer layout of {@link createValueClassPaintGraph}: class count,
 * the interior breaks, then one RGBA colour (0-1) per class, alpha 0 for a class that is not drawn.
 *
 * @throws If the table has more than {@link MAXIMUM_VALUE_CLASSES} classes.
 */
export function packValueClassTable(
  table: ClassTable,
  target: Float32Array = new Float32Array(VALUE_CLASS_TABLE_LENGTH)
): Float32Array {
  const classCount = table.breaks.length + 1;
  if (classCount > MAXIMUM_VALUE_CLASSES) throw new Error('Too many classes for the paint pass');
  target.fill(0);
  target[0] = classCount;
  table.breaks.forEach((value, index) => {
    target[BREAKS_OFFSET + index] = value;
  });
  table.colors.forEach((color, index) => {
    const base = COLORS_OFFSET + index * 4;
    target[base] = color[0] / 255;
    target[base + 1] = color[1] / 255;
    target[base + 2] = color[2] / 255;
    target[base + 3] = (color[3] ?? 255) / 255;
  });
  return target;
}

/**
 * Compiles the pass that turns `source` (float32 values, one per cell) into packed RGBA8 `colors`:
 * the class of a value is the number of breaks at or below it; NaN and infinities are transparent.
 *
 * @param table A parameter buffer written with {@link packValueClassTable}.
 */
export function createValueClassPaintGraph(
  resources: SpatialAnalysisResources,
  device: Device,
  id: string,
  source: {buffer: Buffer; length: number},
  table: GPUParameterBuffer<'float32'>,
  colors: Buffer
): CompiledGPUCommandGraph<void> {
  const graph = new GPUCommandGraph<void>(device, {id: `${id}-class-paint`});
  addKernelPass(graph, {
    id: `${id}-class-colorize`,
    invocationCount: source.length,
    bindings: [
      {
        name: 'values',
        view: importGraphBuffer(graph, 'values', source.buffer, 'float32', source.length),
        type: 'f32',
        access: 'read'
      },
      {name: 'classTable', view: table.importToGraph(graph), type: 'f32', access: 'read'},
      {
        name: 'colors',
        view: importGraphBuffer(graph, 'colors', colors, 'uint32', source.length),
        type: 'u32',
        access: 'read_write'
      }
    ],
    declarations: /* wgsl */ `
const BREAKS_OFFSET: u32 = ${BREAKS_OFFSET}u;
const COLORS_OFFSET: u32 = ${COLORS_OFFSET}u;
fn isNotFinite(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u;
}`,
    body: /* wgsl */ `
  let value = values[valuesOffset + index];
  var color = vec4<f32>(0.0);
  if (!isNotFinite(value)) {
    let classCount = u32(classTable[classTableOffset]);
    var classIndex = 0u;
    for (var limit = 0u; limit + 1u < classCount; limit++) {
      if (value >= classTable[classTableOffset + BREAKS_OFFSET + limit]) {
        classIndex = limit + 1u;
      }
    }
    let base = classTableOffset + COLORS_OFFSET + 4u * classIndex;
    color = vec4<f32>(classTable[base], classTable[base + 1u], classTable[base + 2u], classTable[base + 3u]);
  }
  colors[colorsOffset + index] = pack4x8unorm(color);`
  });
  return resources.track(graph.compile());
}
