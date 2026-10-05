// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createMapGraphKernelNode, type MapGraphKernelBinding} from '../map-graph-kernels';
import {TERRAIN_ILLUMINATION_WGSL_CONSTANTS} from '../terrain-illumination/terrain-illumination-utils';

/**
 * Rounds to the nearest integer, ties to the nearest even integer, like Python's `round` and
 * `numpy.round` (`Math.round` rounds ties toward positive infinity instead).
 *
 * @internal
 */
export function roundHalfEven(value: number): number {
  const floor = Math.floor(value);
  const difference = value - floor;
  if (difference < 0.5) {
    return floor;
  }
  if (difference > 0.5) {
    return floor + 1;
  }
  return floor % 2 === 0 ? floor : floor + 1;
}

/** Graph-owned scratch shared by every mean filter of one recipe. @internal */
export type ReliefMeanFilterScratch = {
  /** Per-pixel row anchor, see {@link getReliefMeanFilterNodes}. */
  anchors: GraphDataView<'float32'>;
  /** Per-pixel sum of `z - anchor` over the clamped row window. */
  rowSums: GraphDataView<'float32'>;
  /** Per-pixel count of valid samples in the clamped row window. */
  rowCounts: GraphDataView<'uint32'>;
};

/** Creates the transient scratch consumed by {@link getReliefMeanFilterNodes}. @internal */
export function createReliefMeanFilterScratch<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  pixelCount: number
): ReliefMeanFilterScratch {
  return {
    anchors: createTransientView(graph, `${id}-mean-anchors`, 'float32', pixelCount),
    rowSums: createTransientView(graph, `${id}-mean-row-sums`, 'float32', pixelCount),
    rowCounts: createTransientView(graph, `${id}-mean-row-counts`, 'uint32', pixelCount)
  };
}

/** Properties for {@link getReliefMeanFilterNodes}. @internal */
export type ReliefMeanFilterProps = {
  /** Prefix for the two node IDs. */
  id: string;
  /** Operation name reported in the workload estimate. */
  operation: string;
  /** Grid width in pixels. */
  width: number;
  /** Grid height in pixels. */
  height: number;
  /** Window half-width in pixels; the window is `(2 * radius + 1)^2`. At least 1. */
  radius: number;
  /** Canonical float32 elevation. */
  values: GraphDataView<'float32'>;
  /** Canonical per-pixel validity, 1 where `values` is usable. */
  validity: GraphDataView<'uint32'>;
  /** Scratch shared by consecutive filters; filters must run in order. */
  scratch: ReliefMeanFilterScratch;
  /** Receives `scale * (z - mean(z))`, NaN where the centre is invalid. */
  deviation: GraphDataView<'float32'>;
  /** Optional per-frame multiplier read from a float32 settings view. */
  scale?: {settings: GraphDataView<'float32'>; index: number};
  /** Optional output receiving 1 where the centre is valid. */
  validityOutput?: GraphDataView<'uint32'>;
};

/**
 * Builds the two kernels of a separable, edge-clamped, nodata-aware mean filter that returns the
 * deviation `z - mean(z)` of every valid pixel from its `(2r + 1)^2` window mean.
 *
 * Window semantics follow RVT's `mean_filter` (Kokalj and Somrak 2019): coordinates outside the
 * grid replicate the edge pixel (`numpy.pad(mode='edge')`, so an edge sample may be counted
 * several times), and invalid samples are excluded from both the sum and the count.
 *
 * RVT evaluates the window with a float64 integral image. A float32 integral image or a plain
 * float32 running sum would lose the answer to cancellation (a 4000 m surface has a float32 step
 * of 0.5 mm, while relief is centimetres), and a prefix sum cannot be made precise on the GPU.
 * Instead the filter is computed in two exact-scale passes that never form a large absolute sum:
 *
 * 1. Rows. Each pixel picks an anchor `a`, its own value when valid, otherwise the first valid
 *    value in its clamped row window, otherwise 0. It stores `rowSum = sum_valid (z - a)`,
 *    `rowCount`, and `a`. Terms are small differences of nearby values, each exact or rounded at
 *    the scale of the difference rather than the scale of the elevation.
 * 2. Columns. For a valid centre `c`, `total = sum_rows (rowSum + rowCount * (a - c))` and
 *    `count = sum rowCount`. Since `rowSum + rowCount * a` is the true sum of the row window,
 *    `total` equals `sum(window) - count * c`, so `z - mean = -total / count`. The large values
 *    cancel inside the exact differences `a - c`, never inside a long accumulation.
 *
 * Cost is `O(radius)` per pixel per pass, independent of the grid size. The window always contains
 * the valid centre, so `count >= 1`.
 *
 * @internal
 */
export function getReliefMeanFilterNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: ReliefMeanFilterProps
): GPUCommandNode<Parameters>[] {
  const {width, height, radius, scratch} = props;
  const pixelCount = width * height;
  const declarations = `const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;
const RADIUS: i32 = ${radius};`;
  const rowPass = createMapGraphKernelNode<Parameters>(graph, {
    id: `${props.id}-rows`,
    operation: props.operation,
    variant: 'mean-rows',
    bindings: [
      {name: 'values', view: props.values, type: 'f32', access: 'read'},
      {name: 'validity', view: props.validity, type: 'u32', access: 'read'},
      {name: 'anchors', view: scratch.anchors, type: 'f32', access: 'read_write'},
      {name: 'rowSums', view: scratch.rowSums, type: 'f32', access: 'read_write'},
      {name: 'rowCounts', view: scratch.rowCounts, type: 'u32', access: 'read_write'}
    ],
    invocationCount: pixelCount,
    declarations,
    body: `let column = i32(index % WIDTH);
  let rowStart = index - u32(column);
  var anchor = 0.0;
  if (validity[validityOffset + index] != 0u) {
    anchor = values[valuesOffset + index];
  } else {
    for (var offset = -RADIUS; offset <= RADIUS; offset++) {
      let sampleIndex = rowStart + u32(clamp(column + offset, 0, i32(WIDTH) - 1));
      if (validity[validityOffset + sampleIndex] != 0u) {
        anchor = values[valuesOffset + sampleIndex];
        break;
      }
    }
  }
  var rowSum = 0.0;
  var rowCount = 0u;
  for (var offset = -RADIUS; offset <= RADIUS; offset++) {
    let sampleIndex = rowStart + u32(clamp(column + offset, 0, i32(WIDTH) - 1));
    if (validity[validityOffset + sampleIndex] != 0u) {
      rowSum += values[valuesOffset + sampleIndex] - anchor;
      rowCount += 1u;
    }
  }
  anchors[anchorsOffset + index] = anchor;
  rowSums[rowSumsOffset + index] = rowSum;
  rowCounts[rowCountsOffset + index] = rowCount;`
  });
  const columnBindings: MapGraphKernelBinding[] = [
    {name: 'values', view: props.values, type: 'f32', access: 'read'},
    {name: 'validity', view: props.validity, type: 'u32', access: 'read'},
    {name: 'anchors', view: scratch.anchors, type: 'f32', access: 'read'},
    {name: 'rowSums', view: scratch.rowSums, type: 'f32', access: 'read'},
    {name: 'rowCounts', view: scratch.rowCounts, type: 'u32', access: 'read'},
    {name: 'deviation', view: props.deviation, type: 'f32', access: 'read_write'}
  ];
  if (props.scale) {
    columnBindings.push({
      name: 'scaleSettings',
      view: props.scale.settings,
      type: 'f32',
      access: 'read'
    });
  }
  if (props.validityOutput) {
    columnBindings.push({
      name: 'validityOutput',
      view: props.validityOutput,
      type: 'u32',
      access: 'read_write'
    });
  }
  const columnPass = createMapGraphKernelNode<Parameters>(graph, {
    id: `${props.id}-columns`,
    operation: props.operation,
    variant: 'mean-columns',
    bindings: columnBindings,
    invocationCount: pixelCount,
    declarations: `${declarations}\n${TERRAIN_ILLUMINATION_WGSL_CONSTANTS}`,
    body: `let column = index % WIDTH;
  let row = i32(index / WIDTH);
  let isValid = validity[validityOffset + index] != 0u;
  var deviationValue = getNaN(index);
  if (isValid) {
    let center = values[valuesOffset + index];
    var total = 0.0;
    var count = 0u;
    for (var offset = -RADIUS; offset <= RADIUS; offset++) {
      let sampleIndex = u32(clamp(row + offset, 0, i32(HEIGHT) - 1)) * WIDTH + column;
      let rowCount = rowCounts[rowCountsOffset + sampleIndex];
      total += rowSums[rowSumsOffset + sampleIndex] +
        f32(rowCount) * (anchors[anchorsOffset + sampleIndex] - center);
      count += rowCount;
    }
    deviationValue = -total / f32(count) ${props.scale ? `* scaleSettings[scaleSettingsOffset + ${props.scale.index}u]` : ''};
  }
  deviation[deviationOffset + index] = deviationValue;
  ${props.validityOutput ? 'validityOutput[validityOutputOffset + index] = select(0u, 1u, isValid);' : ''}`
  });
  return [rowPass, columnPass];
}
