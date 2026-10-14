// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPURasterBufferBand} from '../../gpu-raster/index';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {TERRAIN_ILLUMINATION_WGSL_CONSTANTS} from '../terrain-illumination/terrain-illumination-utils';
import {
  addTerrainSummedAreaTableNodes,
  getGPUTerrainSummedAreaMaximumCellCount,
  TERRAIN_SUMMED_AREA_SUM_COUNT_BOX_WGSL,
  TERRAIN_SUMMED_AREA_WGSL_HELPERS
} from '../topographic-position/terrain-summed-area-table';

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

/** Graph-owned scratch shared by every mean filter of one contributor. @internal */
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
  /**
   * Window half-width in pixels; the window is `(2 * radius + 1)^2`. At least 1. With
   * `radiusSlot`, the largest radius the filter serves (the compile-time maximum).
   */
  radius: number;
  /**
   * Optional per-frame radius read from a float32 settings view. The word is an integer in
   * `[1, radius]`: larger values clamp to `radius`, and a value below 1 (or NaN) selects `radius`.
   * The arithmetic and order are those of the fixed-radius filter, so a per-frame radius equal to
   * `radius` is bit-identical. One pipeline pair serves every per-frame radius.
   */
  radiusSlot?: {settings: GraphDataView<'float32'>; index: number};
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

// Not GPURasterNeighborhood/GPURasterBoxBlur: their radius is capped at 8 and they average in absolute float32, losing centimetre relief.
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
  // With a per-frame radius the loops run to `radius` read from the settings, else to RADIUS.
  const {radiusSlot} = props;
  const radiusSource = radiusSlot
    ? `let radiusWord = radiusSettings[radiusSettingsOffset + ${radiusSlot.index}u];
  let radius = select(RADIUS, clamp(i32(radiusWord), 1, RADIUS), radiusWord >= 1.0);
  `
    : '';
  const loopRadius = radiusSlot ? 'radius' : 'RADIUS';
  const variantSuffix = radiusSlot ? '-radius' : '';
  const radiusBinding: WGSLKernelBinding[] = radiusSlot
    ? [{name: 'radiusSettings', view: radiusSlot.settings, type: 'f32', access: 'read'}]
    : [];
  const rowPass = createWGSLKernelNode<Parameters>(graph, {
    id: `${props.id}-rows`,
    operation: props.operation,
    variant: `mean-rows${variantSuffix}`,
    bindings: [
      {name: 'values', view: props.values, type: 'f32', access: 'read'},
      {name: 'validity', view: props.validity, type: 'u32', access: 'read'},
      {name: 'anchors', view: scratch.anchors, type: 'f32', access: 'read_write'},
      {name: 'rowSums', view: scratch.rowSums, type: 'f32', access: 'read_write'},
      {name: 'rowCounts', view: scratch.rowCounts, type: 'u32', access: 'read_write'},
      ...radiusBinding
    ],
    invocationCount: pixelCount,
    declarations,
    body: `let column = i32(index % WIDTH);
  let rowStart = index - u32(column);
  ${radiusSource}
  var anchor = 0.0;
  if (validity[validityOffset + index] != 0u) {
    anchor = values[valuesOffset + index];
  } else {
    for (var offset = -${loopRadius}; offset <= ${loopRadius}; offset++) {
      let sampleIndex = rowStart + u32(clamp(column + offset, 0, i32(WIDTH) - 1));
      if (validity[validityOffset + sampleIndex] != 0u) {
        anchor = values[valuesOffset + sampleIndex];
        break;
      }
    }
  }
  var rowSum = 0.0;
  var rowCount = 0u;
  for (var offset = -${loopRadius}; offset <= ${loopRadius}; offset++) {
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
  const columnBindings: WGSLKernelBinding[] = [
    ...radiusBinding,
    {name: 'values', view: props.values, type: 'f32', access: 'read'},
    {name: 'validity', view: props.validity, type: 'u32', access: 'read'},
    {name: 'anchors', view: scratch.anchors, type: 'f32', access: 'read'},
    {name: 'rowSums', view: scratch.rowSums, type: 'f32', access: 'read'},
    {name: 'rowCounts', view: scratch.rowCounts, type: 'u32', access: 'read'},
    {name: 'deviation', view: props.deviation, type: 'f32', access: 'read_write'}
  ];
  // The scale usually lives in the same settings view as the radius: bind it once, so the pass
  // stays within 8 storage buffers with every output (core limit).
  const scaleSettingsName =
    props.scale && radiusSlot && props.scale.settings === radiusSlot.settings
      ? 'radiusSettings'
      : 'scaleSettings';
  if (props.scale && scaleSettingsName === 'scaleSettings') {
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
  const columnPass = createWGSLKernelNode<Parameters>(graph, {
    id: `${props.id}-columns`,
    operation: props.operation,
    variant: `mean-columns${variantSuffix}`,
    bindings: columnBindings,
    invocationCount: pixelCount,
    declarations: `${declarations}\n${TERRAIN_ILLUMINATION_WGSL_CONSTANTS}`,
    body: `let column = index % WIDTH;
  let row = i32(index / WIDTH);
  ${radiusSource}
  let isValid = validity[validityOffset + index] != 0u;
  var deviationValue = getNaN(index);
  if (isValid) {
    let center = values[valuesOffset + index];
    var total = 0.0;
    var count = 0u;
    for (var offset = -${loopRadius}; offset <= ${loopRadius}; offset++) {
      let sampleIndex = u32(clamp(row + offset, 0, i32(HEIGHT) - 1)) * WIDTH + column;
      let rowCount = rowCounts[rowCountsOffset + sampleIndex];
      total += rowSums[rowSumsOffset + sampleIndex] +
        f32(rowCount) * (anchors[anchorsOffset + sampleIndex] - center);
      count += rowCount;
    }
    deviationValue = -total / f32(count) ${props.scale ? `* ${scaleSettingsName}[${scaleSettingsName}Offset + ${props.scale.index}u]` : ''};
  }
  deviation[deviationOffset + index] = deviationValue;
  ${props.validityOutput ? 'validityOutput[validityOutputOffset + index] = select(0u, 1u, isValid);' : ''}`
  });
  return [rowPass, columnPass];
}

/**
 * Default elevation quantum of the exact summed-area table behind the mean filters of
 * {@link GPUSimpleLocalRelief} and {@link GPUMultiScaleRelief}: `2^-16` elevation units (15 um
 * for metres). With it a quantized elevation fits a signed 32-bit integer for `|z - anchor| < 32768`.
 * Float32 elevations of magnitude 128 or more are multiples of `2^-16` already, so for them the
 * quantization is exact.
 */
export const GPU_RELIEF_DEFAULT_QUANTUM = 2 ** -16;

/**
 * Largest mean-filter radius the summed-area path accepts. The exact deviation sum of a window
 * differs from the centre by up to `2^32` quantized units per sample (two signed 32-bit values),
 * so `(2R + 1)^2 * 2^32 < 2^63` bounds `R` by 23170.
 */
export const GPU_RELIEF_SUMMED_AREA_MAXIMUM_RADIUS = 23170;

/**
 * Returns whether a grid is served by the O(1) summed-area mean filter: grids up to
 * {@link getGPUTerrainSummedAreaMaximumCellCount} cells on `device`. Larger grids use the separable
 * O(radius) filter of {@link getReliefMeanFilterNodes}. The choice depends on the grid size and the
 * device binding limit only.
 *
 * @internal
 */
export function usesReliefSummedArea(device: Device, pixelCount: number): boolean {
  return pixelCount <= getGPUTerrainSummedAreaMaximumCellCount(device);
}

/**
 * Throws unless `quantum` is a positive power of two.
 *
 * @internal
 */
export function validateReliefMeanFilter(id: string, quantum: number): void {
  if (
    !(quantum > 0) ||
    !Number.isFinite(quantum) ||
    2 ** Math.round(Math.log2(quantum)) !== quantum
  ) {
    throw new Error(`${id} quantum must be a positive power of two`);
  }
}

/**
 * Throws unless, on the summed-area path (see {@link usesReliefSummedArea}), `radius` is at most
 * {@link GPU_RELIEF_SUMMED_AREA_MAXIMUM_RADIUS}. Call at declaration, where the device is known.
 *
 * @internal
 */
export function validateReliefMeanFilterRadius(
  id: string,
  device: Device,
  pixelCount: number,
  radius: number
): void {
  if (usesReliefSummedArea(device, pixelCount) && radius > GPU_RELIEF_SUMMED_AREA_MAXIMUM_RADIUS) {
    throw new Error(`${id} radius must be at most ${GPU_RELIEF_SUMMED_AREA_MAXIMUM_RADIUS}`);
  }
}

/** Per-filter properties of {@link ReliefMeanFilterSetup.getFilterNodes}. @internal */
export type ReliefMeanFilterNodeProps = Omit<
  ReliefMeanFilterProps,
  'width' | 'height' | 'values' | 'validity' | 'scratch'
>;

/** Result of {@link createReliefMeanFilterSetup}. @internal */
export type ReliefMeanFilterSetup<Parameters> = {
  /** Which algorithm serves the grid. */
  method: 'summed-area' | 'separable';
  /** Nodes to run once before any filter (the table build; empty for the separable method). */
  nodes: GPUCommandNode<Parameters>[];
  /** Returns the nodes of one filter; call order is run order. */
  getFilterNodes: (props: ReliefMeanFilterNodeProps) => GPUCommandNode<Parameters>[];
};

/**
 * Chooses the mean filter algorithm for a grid and builds what all filters of one contributor
 * share. Grids up to {@link getGPUTerrainSummedAreaMaximumCellCount} cells use
 * {@link getReliefSummedAreaFilterNodes} over one exact summed-area table built here (the table
 * is shared by every filter, so MSRM builds it once); larger grids use the separable
 * {@link getReliefMeanFilterNodes} with shared scratch.
 *
 * @internal
 */
export function createReliefMeanFilterSetup<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    width: number;
    height: number;
    /** Canonical elevation band. */
    band: GPURasterBufferBand;
    /** Power-of-two elevation quantum of the summed-area path. */
    quantum: number;
  }
): ReliefMeanFilterSetup<Parameters> {
  const {id, width, height, band} = props;
  const pixelCount = width * height;
  const values = band.storage.values as GraphDataView<'float32'>;
  if (!usesReliefSummedArea(graph.device, pixelCount)) {
    const scratch = createReliefMeanFilterScratch(graph, id, pixelCount);
    const validity = band.validity as GraphDataView<'uint32'>;
    return {
      method: 'separable',
      nodes: [],
      getFilterNodes: filterProps =>
        getReliefMeanFilterNodes(graph, {...filterProps, width, height, values, validity, scratch})
    };
  }
  const table = addTerrainSummedAreaTableNodes(graph, {
    id,
    width,
    height,
    elevation: band,
    quantum: props.quantum,
    includeSquares: false,
    originRelative: true
  });
  return {
    method: 'summed-area',
    nodes: table.nodes,
    getFilterNodes: filterProps =>
      getReliefSummedAreaFilterNodes(graph, {
        ...filterProps,
        width,
        height,
        values,
        table,
        quantum: props.quantum
      })
  };
}

/**
 * Builds the single kernel of the O(1)-per-pixel mean filter over an exact summed-area table
 * (without squares) of elevations quantized to integers `q = round((z - anchor) / quantum)`. Output and
 * window semantics are those of {@link getReliefMeanFilterNodes}.
 *
 * Edge replication. RVT pads the grid with `numpy.pad(mode='edge')`, so the clamped window gives
 * every grid column and row an integer weight: 1 inside, and `1 + overhang` on the edge column
 * or row, where `overhang` counts the window columns (rows) beyond that edge; with a width of 1
 * both overhangs land on the same column. The weight is separable and the window sum is
 *
 *     box(clipped) + overhangLeft * strip(column 0) + overhangRight * strip(last column)
 *       + overhangTop * strip(row 0) + overhangBottom * strip(last row)
 *       + the four overhangX * overhangY * corner cells,
 *
 * every term a summed-area box query, read from the 64-bit modular table, so the weighted sum `S`
 * and the weighted count `N` of valid samples (invalid samples are zero in both planes) are exact
 * integers. The deviation needs no large absolute value: with `D = N * q_c - S` formed exactly in
 * 64 bits (`q_c` is the centre's quantized value, `|D| < 2^63` by the radius bound),
 *
 *     z - mean = ((z_c - anchor) - q_c * quantum) + quantum * D / N.
 *
 * The first term is the centre's own quantization residual: `z_c * 2^-k` scaling is exact in
 * float32, so it is formed without rounding. `D` converts to float32 once (relative error
 * `2^-24`), then one division and one multiplication. The result differs from the float64 RVT
 * mean filter by at most `quantum / 2` (quantization of the window samples; zero when every
 * `|z| >= 128` at the default quantum) plus about three float32 ULP of the deviation itself, with
 * no dependence on the absolute elevation magnitude (the subtraction additionally rounds at the
 * relief-span scale). Elevations with `|z - anchor| >= 2^31 * quantum` count as
 * invalid, like the topographic-position table.
 *
 * @internal
 */
export function getReliefSummedAreaFilterNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: ReliefMeanFilterNodeProps & {
    width: number;
    height: number;
    values: GraphDataView<'float32'>;
    table: {
      lowTable: GraphDataView<'uint32'>;
      highTable: GraphDataView<'uint32'>;
      anchor?: GraphDataView<'float32'>;
    };
    quantum: number;
  }
): GPUCommandNode<Parameters>[] {
  const {width, height, radius, radiusSlot} = props;
  const bindings: WGSLKernelBinding[] = [
    ...(props.table.anchor
      ? [{name: 'anchor', view: props.table.anchor, type: 'f32' as const, access: 'read' as const}]
      : []),
    {name: 'lowTable', view: props.table.lowTable, type: 'u32', access: 'read'},
    {name: 'highTable', view: props.table.highTable, type: 'u32', access: 'read'},
    {name: 'values', view: props.values, type: 'f32', access: 'read'},
    {name: 'deviation', view: props.deviation, type: 'f32', access: 'read_write'}
  ];
  if (radiusSlot) {
    bindings.push({name: 'radiusSettings', view: radiusSlot.settings, type: 'f32', access: 'read'});
  }
  const scaleSettingsName =
    props.scale && radiusSlot && props.scale.settings === radiusSlot.settings
      ? 'radiusSettings'
      : 'scaleSettings';
  if (props.scale && scaleSettingsName === 'scaleSettings') {
    bindings.push({
      name: 'scaleSettings',
      view: props.scale.settings,
      type: 'f32',
      access: 'read'
    });
  }
  if (props.validityOutput) {
    bindings.push({
      name: 'validityOutput',
      view: props.validityOutput,
      type: 'u32',
      access: 'read_write'
    });
  }
  const radiusSource = radiusSlot
    ? `let radiusWord = radiusSettings[radiusSettingsOffset + ${radiusSlot.index}u];
  let radius = select(RADIUS, clamp(i32(radiusWord), 1, RADIUS), radiusWord >= 1.0);`
    : 'let radius = RADIUS;';
  return [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${props.id}-deviation`,
      operation: props.operation,
      variant: `mean-summed-area${radiusSlot ? '-radius' : ''}`,
      bindings,
      invocationCount: width * height,
      declarations: `const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;
const PIXEL_COUNT: u32 = ${width * height}u;
const RADIUS: i32 = ${radius};
const QUANTUM: f32 = ${getWGSLFloatLiteral(props.quantum)};
const INVERSE_QUANTUM: f32 = ${getWGSLFloatLiteral(1 / props.quantum)};
${TERRAIN_SUMMED_AREA_WGSL_HELPERS}
${TERRAIN_SUMMED_AREA_SUM_COUNT_BOX_WGSL}
// Runtime operand so the NaN is not constant-folded away.
fn getNaN(index: u32) -> f32 { return bitcast<f32>(0x7fc00000u | (index & 0u)); }
// total + weight * strip, modulo 2^64 for the sum.
fn addWeighted(total: TerrainBoxSums, strip: TerrainBoxSums, weight: u32) -> TerrainBoxSums {
  return TerrainBoxSums(
    add64(total.sum, multiply64By32(strip.sum, weight)),
    total.count + strip.count * weight
  );
}`,
      body: `let column = i32(index % WIDTH);
  let row = i32(index / WIDTH);
  ${radiusSource}
  // A single-cell box returns the centre's quantized elevation and validity exactly.
  let centre = readBoxSums(column, column, row, row);
  let isValid = centre.count == 1u;
  var deviationValue = getNaN(index);
  if (isValid) {
    let firstColumn = max(column - radius, 0i);
    let lastColumn = min(column + radius, i32(WIDTH) - 1i);
    let firstRow = max(row - radius, 0i);
    let lastRow = min(row + radius, i32(HEIGHT) - 1i);
    let overhangLeft = u32(max(radius - column, 0i));
    let overhangRight = u32(max(column + radius - (i32(WIDTH) - 1i), 0i));
    let overhangTop = u32(max(radius - row, 0i));
    let overhangBottom = u32(max(row + radius - (i32(HEIGHT) - 1i), 0i));
    var total = readBoxSums(firstColumn, lastColumn, firstRow, lastRow);
    if (overhangLeft > 0u) {
      total = addWeighted(total, readBoxSums(0i, 0i, firstRow, lastRow), overhangLeft);
    }
    if (overhangRight > 0u) {
      total = addWeighted(total, readBoxSums(i32(WIDTH) - 1i, i32(WIDTH) - 1i, firstRow, lastRow), overhangRight);
    }
    if (overhangTop > 0u) {
      total = addWeighted(total, readBoxSums(firstColumn, lastColumn, 0i, 0i), overhangTop);
      if (overhangLeft > 0u) {
        total = addWeighted(total, readBoxSums(0i, 0i, 0i, 0i), overhangLeft * overhangTop);
      }
      if (overhangRight > 0u) {
        total = addWeighted(total, readBoxSums(i32(WIDTH) - 1i, i32(WIDTH) - 1i, 0i, 0i), overhangRight * overhangTop);
      }
    }
    if (overhangBottom > 0u) {
      total = addWeighted(total, readBoxSums(firstColumn, lastColumn, i32(HEIGHT) - 1i, i32(HEIGHT) - 1i), overhangBottom);
      if (overhangLeft > 0u) {
        total = addWeighted(total, readBoxSums(0i, 0i, i32(HEIGHT) - 1i, i32(HEIGHT) - 1i), overhangLeft * overhangBottom);
      }
      if (overhangRight > 0u) {
        total = addWeighted(total, readBoxSums(i32(WIDTH) - 1i, i32(WIDTH) - 1i, i32(HEIGHT) - 1i, i32(HEIGHT) - 1i), overhangRight * overhangBottom);
      }
    }
    // D = N * q_c - S, exact modulo 2^64 and |D| < 2^63.
    let quantized = bitcast<i32>(centre.sum.x);
    var centreTotal = multiply64By32(vec2<u32>(total.count, 0u), u32(abs(quantized)));
    if (quantized < 0i) {
      centreTotal = negate64(centreTotal);
    }
    let difference = subtract64(centreTotal, total.sum);
    // The centre's own quantization residual is exact: scaling by a power of two does not round.
    let residual = ((values[valuesOffset + index]${props.table.anchor ? ' - anchor[anchorOffset]' : ''}) * INVERSE_QUANTUM - f32(quantized)) * QUANTUM;
    deviationValue = residual + signed64ToFloat(difference) / f32(total.count) * QUANTUM;
    ${props.scale ? `deviationValue = deviationValue * ${scaleSettingsName}[${scaleSettingsName}Offset + ${props.scale.index}u];` : ''}
  }
  deviation[deviationOffset + index] = deviationValue;
  ${props.validityOutput ? 'validityOutput[validityOutputOffset + index] = select(0u, 1u, isValid);' : ''}`
    })
  ];
}
