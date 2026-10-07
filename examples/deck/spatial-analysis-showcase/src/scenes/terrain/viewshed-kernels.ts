// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * The small scene kernels of the viewshed story, each one compute pass added to a contributor
 * graph with `addKernelPass`. The contributors write visibility codes (`GPU_TERRAIN_VISIBILITY`:
 * hidden 0, visible 1, out of range 2, no data 3, marginal 4) and counts; these passes turn them
 * into what the map draws and reads, so nothing is read back per cell:
 *
 * - {@link addClassifyPass}: code to class value of the chapter visibility table;
 * - {@link addFlipPass} and {@link addDilatePass}: the cells whose code differs between two earth
 *   models, grown by one cell so a few hundred of them are still visible at a wide zoom;
 * - {@link addCumulativeClassPass}: lookout counts, with cells beyond every lookout's reach set
 *   apart from the cells that none of them sees;
 * - {@link addGatherPass}: the code at a few named cells (summits), a few words to read back;
 * - {@link addCoarseUnseenPass}: a coarse mask of the unseen cells for the largest-gap search.
 */

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {addKernelPass} from '../../engine/mode-kernels';
import {VISIBILITY_CLASS} from './terrain-palettes';
import {
  BEYOND_REACH_CLASS,
  COARSE_STRIDE,
  getCoarseSize,
  LOOKOUT_SLOTS,
  NO_DATA_CLASS
} from './viewshed-style';

/**
 * Visibility codes to class values: visible 0, marginal 1, hidden 2, out of range 3 (the order of
 * the chapter visibility table, so layer and legend read one table) and no data as the uint32
 * sentinel, which the layer hatches.
 */
export function addClassifyPass(
  graph: GPUCommandGraph<void>,
  props: {id: string; codes: GraphDataView; classes: GraphDataView; count: number}
): void {
  addKernelPass(graph, {
    id: props.id,
    invocationCount: props.count,
    bindings: [
      {name: 'codes', view: props.codes, type: 'u32', access: 'read'},
      {name: 'classes', view: props.classes, type: 'u32', access: 'read_write'}
    ],
    body: /* wgsl */ `
  let code = codes[codesOffset + index];
  var value = ${VISIBILITY_CLASS.outOfRange}u;
  if (code == 1u) {
    value = ${VISIBILITY_CLASS.visible}u;
  } else if (code == 4u) {
    value = ${VISIBILITY_CLASS.marginal}u;
  } else if (code == 0u) {
    value = ${VISIBILITY_CLASS.hidden}u;
  } else if (code == 3u) {
    value = ${NO_DATA_CLASS}u;
  }
  classes[classesOffset + index] = value;`
  });
}

/** 1 where two visibility rasters differ (the cell flips between two models), else 0. */
export function addFlipPass(
  graph: GPUCommandGraph<void>,
  props: {
    id: string;
    first: GraphDataView;
    second: GraphDataView;
    flips: GraphDataView;
    count: number;
  }
): void {
  addKernelPass(graph, {
    id: props.id,
    invocationCount: props.count,
    bindings: [
      {name: 'first', view: props.first, type: 'u32', access: 'read'},
      {name: 'second', view: props.second, type: 'u32', access: 'read'},
      {name: 'flips', view: props.flips, type: 'u32', access: 'read_write'}
    ],
    body: /* wgsl */ `
  flips[flipsOffset + index] = select(0u, 1u, first[firstOffset + index] != second[secondOffset + index]);`
  });
}

/** Grows a 0/1 raster by one cell in every direction (3 x 3 maximum), for drawing. */
export function addDilatePass(
  graph: GPUCommandGraph<void>,
  props: {
    id: string;
    source: GraphDataView;
    grown: GraphDataView;
    width: number;
    height: number;
  }
): void {
  addKernelPass(graph, {
    id: props.id,
    invocationCount: props.width * props.height,
    bindings: [
      {name: 'source', view: props.source, type: 'u32', access: 'read'},
      {name: 'grown', view: props.grown, type: 'u32', access: 'read_write'}
    ],
    declarations: `const GRID_WIDTH: i32 = ${props.width};\nconst GRID_HEIGHT: i32 = ${props.height};`,
    body: /* wgsl */ `
  let column = i32(index) % GRID_WIDTH;
  let row = i32(index) / GRID_WIDTH;
  var hit = 0u;
  for (var rowStep = -1; rowStep <= 1; rowStep = rowStep + 1) {
    for (var columnStep = -1; columnStep <= 1; columnStep = columnStep + 1) {
      let neighbourColumn = column + columnStep;
      let neighbourRow = row + rowStep;
      if (neighbourColumn >= 0 && neighbourRow >= 0 && neighbourColumn < GRID_WIDTH && neighbourRow < GRID_HEIGHT) {
        hit = max(hit, source[sourceOffset + u32(neighbourRow * GRID_WIDTH + neighbourColumn)]);
      }
    }
  }
  grown[grownOffset + index] = hit;`
  });
}

/**
 * Lookout counts to class values: the count (0 to 6) where the cell is within `reach` metres of at
 * least one active lookout, else {@link BEYOND_REACH_CLASS}. `lookouts` holds `[column, row]` pairs
 * (parked lookouts have a negative column) and `reach` holds `[metres, ground metres per cell]`.
 */
export function addCumulativeClassPass(
  graph: GPUCommandGraph<void>,
  props: {
    id: string;
    counts: GraphDataView;
    lookouts: GraphDataView;
    reach: GraphDataView;
    classes: GraphDataView;
    width: number;
    height: number;
  }
): void {
  addKernelPass(graph, {
    id: props.id,
    invocationCount: props.width * props.height,
    bindings: [
      {name: 'counts', view: props.counts, type: 'u32', access: 'read'},
      {name: 'lookouts', view: props.lookouts, type: 'f32', access: 'read'},
      {name: 'reach', view: props.reach, type: 'f32', access: 'read'},
      {name: 'classes', view: props.classes, type: 'u32', access: 'read_write'}
    ],
    declarations: `const GRID_WIDTH: u32 = ${props.width}u;\nconst SLOT_COUNT: u32 = ${LOOKOUT_SLOTS}u;`,
    body: /* wgsl */ `
  let column = f32(index % GRID_WIDTH);
  let row = f32(index / GRID_WIDTH);
  var within = false;
  for (var slot = 0u; slot < SLOT_COUNT; slot = slot + 1u) {
    let lookoutColumn = lookouts[lookoutsOffset + slot * 2u];
    let lookoutRow = lookouts[lookoutsOffset + slot * 2u + 1u];
    if (lookoutColumn >= 0.0) {
      let metres = length(vec2<f32>(column - lookoutColumn, row - lookoutRow)) * reach[reachOffset + 1u];
      if (metres <= reach[reachOffset]) {
        within = true;
      }
    }
  }
  classes[classesOffset + index] = select(${BEYOND_REACH_CLASS}u, counts[countsOffset + index], within);`
  });
}

/** Copies the value at `cells[i]` (a raster index, or the no-data sentinel) into `values[i]`. */
export function addGatherPass(
  graph: GPUCommandGraph<void>,
  props: {
    id: string;
    source: GraphDataView;
    cells: GraphDataView;
    values: GraphDataView;
    count: number;
  }
): void {
  addKernelPass(graph, {
    id: props.id,
    invocationCount: props.count,
    bindings: [
      {name: 'source', view: props.source, type: 'u32', access: 'read'},
      {name: 'cells', view: props.cells, type: 'u32', access: 'read'},
      {name: 'values', view: props.values, type: 'u32', access: 'read_write'}
    ],
    body: /* wgsl */ `
  let cell = cells[cellsOffset + index];
  values[valuesOffset + index] = select(${NO_DATA_CLASS}u, source[sourceOffset + cell], cell != ${NO_DATA_CLASS}u);`
  });
}

/**
 * A coarse mask of the cells no lookout sees: one invocation per coarse cell (`COARSE_STRIDE` fine
 * cells wide) writes 1 where the fine cell at its centre has count 0 and 0 elsewhere (seen, or
 * beyond every lookout's reach).
 */
export function addCoarseUnseenPass(
  graph: GPUCommandGraph<void>,
  props: {
    id: string;
    classes: GraphDataView;
    mask: GraphDataView;
    width: number;
    height: number;
  }
): void {
  const coarse = getCoarseSize(props.width, props.height);
  addKernelPass(graph, {
    id: props.id,
    invocationCount: coarse.width * coarse.height,
    bindings: [
      {name: 'classes', view: props.classes, type: 'u32', access: 'read'},
      {name: 'mask', view: props.mask, type: 'u32', access: 'read_write'}
    ],
    declarations: `const GRID_WIDTH: u32 = ${props.width}u;
const GRID_HEIGHT: u32 = ${props.height}u;
const COARSE_WIDTH: u32 = ${coarse.width}u;
const STRIDE: u32 = ${COARSE_STRIDE}u;`,
    body: /* wgsl */ `
  let column = min((index % COARSE_WIDTH) * STRIDE + STRIDE / 2u, GRID_WIDTH - 1u);
  let row = min((index / COARSE_WIDTH) * STRIDE + STRIDE / 2u, GRID_HEIGHT - 1u);
  mask[maskOffset + index] = select(0u, 1u, classes[classesOffset + row * GRID_WIDTH + column] == 0u);`
  });
}
