// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPURasterBand, GPURasterMetadata} from '../../gpu-raster/index';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  getStorageBindingElementLimit,
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {
  getRasterBandViews,
  getRasterBufferBandNodes,
  validateRasterBandBelongsToGraph,
  validateRasterBuffersDistinct,
  validateRasterGrid
} from '../../gpu-raster/raster-band-utils';
import {
  addTerrainSummedAreaTableNodes,
  getGPUTerrainSummedAreaMaximumCellCount,
  TERRAIN_SUMMED_AREA_BOX_WGSL,
  TERRAIN_SUMMED_AREA_WGSL_HELPERS
} from './terrain-summed-area-table';

/** Default elevation quantum, 1/256 elevation unit (the Terrarium encoding step in metres). */
export const GPU_TERRAIN_TOPOGRAPHIC_POSITION_DEFAULT_QUANTUM = 1 / 256;

/**
 * Largest `grid` cells x `scales` product of {@link GPUTerrainTopographicPosition} on `device` when
 * `output.topographicPositionIndex` or `output.deviationFromMean` is requested: each holds one
 * float32 per cell and scale in one storage binding, so 33554432 on a default device (128 MiB
 * bindings).
 */
export function getGPUTerrainTopographicPositionMaximumPlaneCount(device: Device): number {
  return getStorageBindingElementLimit(device, 4);
}

/** Largest number of scales one {@link GPUTerrainTopographicPosition} evaluates. */
export const GPU_TERRAIN_TOPOGRAPHIC_POSITION_MAXIMUM_SCALES = 64;

/**
 * One square neighbourhood scale, in cells.
 *
 * The topographic position index uses the square annulus of cells whose Chebyshev distance from the
 * centre is in `(innerRadius, radius]`. The deviation from mean elevation uses the full
 * `(2 * radius + 1)²` window including the centre, as Lindsay et al. (2015) define it.
 */
export type GPUTerrainTopographicPositionScale = {
  /** Outer radius in cells, a positive integer. */
  radius: number;
  /** Inner radius in cells excluded from the TPI neighbourhood. Defaults to 0 (exclude the centre). */
  innerRadius?: number;
};

/** Caller-owned outputs of {@link GPUTerrainTopographicPosition}; at least one is required. */
export type GPUTerrainTopographicPositionOutput = {
  /**
   * Optional TPI planes: `scales.length * width * height` values, plane `k` holding
   * `z - mean(annulus k)` in elevation units. NaN where the centre or the whole annulus is invalid.
   */
  topographicPositionIndex?: GraphDataView<'float32'>;
  /**
   * Optional DEV planes: `scales.length * width * height` values, plane `k` holding
   * `(z - mean) / standardDeviation` over window `k` (population statistics; 0 where the window
   * has zero variance). NaN where the centre is invalid.
   */
  deviationFromMean?: GraphDataView<'float32'>;
  /** Optional per-pixel signed DEV of the scale with the largest |DEV| (DEVmax magnitude). */
  maximumDeviation?: GraphDataView<'float32'>;
  /** Optional per-pixel radius in cells of the scale with the largest |DEV|; first scale on ties. */
  maximumDeviationRadius?: GraphDataView<'uint32'>;
  /** Optional per-pixel 1 where the centre elevation is valid, else 0. */
  outputValidity?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUTerrainTopographicPosition}.
 *
 * Every scale is in cells, so the contributor is independent of the cell-size model: with projected
 * (`'uniform'`) metres, Web Mercator, or geographic degrees alike, a radius of `r` cells covers
 * `r` rows and `r` columns. On geographic grids the window is therefore narrower in ground metres
 * east-west than north-south by cos(latitude). Outputs are in elevation units (TPI) or
 * dimensionless (DEV).
 *
 * Topology: grid size, scales, quantum, and which outputs exist. There are no per-frame settings.
 */
export type GPUTerrainTopographicPositionProps = {
  /** Prefix for node and transient IDs. Defaults to `'terrain-topographic-position'`. */
  id?: string;
  /**
   * Raster grid: its width and height size the kernels. At most
   * {@link getGPUTerrainSummedAreaMaximumCellCount} cells (11184810 on a default device).
   */
  grid: GPURasterMetadata;
  /** Elevation band, buffer or texture; scale, offset, nodata, and validity are honored. */
  elevation: GPURasterBand;
  /** One to {@link GPU_TERRAIN_TOPOGRAPHIC_POSITION_MAXIMUM_SCALES} neighbourhood scales. */
  scales: readonly GPUTerrainTopographicPositionScale[];
  /**
   * Elevation quantum of the exact integer summed-area table. Must be a power of two. Defaults to
   * 1/256. Box sums are exact for quantized elevations while
   * `(2R + 1)² · (relief / quantum)² < 2^64`, where `relief` is the elevation range inside the
   * largest window of radius `R`; with the default that is `relief < 2^32 / (256 · (2R + 1))`,
   * e.g. 16.7 km at `R = 500`.
   */
  quantum?: number;
  /** Caller-owned outputs. */
  output: GPUTerrainTopographicPositionOutput;
};

/**
 * Multiscale topographic position: TPI over square annuli, deviation from mean elevation (DEV),
 * and the maximum-deviation scale (DEVmax, Lindsay, Cockburn & Russell 2015), all from one exact
 * summed-area table.
 *
 * A float32 summed-area table loses the box sums it exists to provide: on a 1024² tile at 1000 m
 * the table entries reach 10^9, whose float32 spacing is 64 elevation units. This contributor instead
 * quantizes elevations to integers (`quantum`, default 1/256), builds a 64-bit modular table with
 * the GPU Core `GPUScanUint64`, `GPUScan`, and `GPUTranspose` primitives, and recovers every box
 * sum exactly. Variance is then formed relative to the centre cell in exact integer arithmetic,
 * `Σ(q - q_c)² = Σq² - 2 q_c Σq + N q_c²` (mod 2^64), and then re-centred, again exactly, on the
 * integer `s` nearest the window mean: `Σ(d - s)² = Σd² - 2 s Σd + N s²` with `d = q - q_c`. The
 * float32 variance is `Σ(d - s)² / N - r²` with a residual mean `|r|` of about half a quantum, so
 * it does not cancel when the centre is far from the window mean (large radii on slopes). The
 * float32 rounding left is that of the conversions, divisions and square root. The quantization
 * changes each elevation by at most half a quantum.
 *
 * Every scale costs O(1) per pixel. Windows are clipped to the raster and to valid cells (the
 * count of valid cells is a third table plane), which differs from WhiteboxTools only along the
 * border band, where Whitebox's clamped corner indices drop the first row and column.
 */
export class GPUTerrainTopographicPosition implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUTerrainTopographicPositionProps;
  /** Receptive field in pixels (`GPURasterHaloStage` contract): the largest radius. */
  readonly requiredHalo: number;

  constructor(props: GPUTerrainTopographicPositionProps) {
    this.id = props.id ?? 'terrain-topographic-position';
    this.props = props;
    const {id} = this;
    const {output} = props;
    const pixelCount = validateRasterGrid(id, props.grid);
    const {scales} = props;
    if (
      !Array.isArray(scales) ||
      scales.length < 1 ||
      scales.length > GPU_TERRAIN_TOPOGRAPHIC_POSITION_MAXIMUM_SCALES
    ) {
      throw new Error(
        `${id} scales must contain 1 to ${GPU_TERRAIN_TOPOGRAPHIC_POSITION_MAXIMUM_SCALES} entries`
      );
    }
    for (const scale of scales) {
      const innerRadius = scale.innerRadius ?? 0;
      if (
        !Number.isSafeInteger(scale.radius) ||
        scale.radius < 1 ||
        !Number.isSafeInteger(innerRadius) ||
        innerRadius < 0 ||
        innerRadius >= scale.radius
      ) {
        throw new Error(`${id} scales need integer radius >= 1 and 0 <= innerRadius < radius`);
      }
    }
    const quantum = props.quantum ?? GPU_TERRAIN_TOPOGRAPHIC_POSITION_DEFAULT_QUANTUM;
    if (
      !(quantum > 0) ||
      !Number.isFinite(quantum) ||
      2 ** Math.round(Math.log2(quantum)) !== quantum
    ) {
      throw new Error(`${id} quantum must be a positive power of two`);
    }
    this.requiredHalo = Math.max(...scales.map(scale => scale.radius));
    if (
      !output.topographicPositionIndex &&
      !output.deviationFromMean &&
      !output.maximumDeviation &&
      !output.maximumDeviationRadius &&
      !output.outputValidity
    ) {
      throw new Error(`${id} requires at least one output`);
    }
    for (const [name, view, length] of [
      ['topographicPositionIndex', output.topographicPositionIndex, pixelCount * scales.length],
      ['deviationFromMean', output.deviationFromMean, pixelCount * scales.length],
      ['maximumDeviation', output.maximumDeviation, pixelCount]
    ] as const) {
      if (view) {
        validatePackedView(view, ['float32'], `${id} output.${name}`);
        if (view.length !== length) {
          throw new Error(`${id} output.${name} must contain ${length} values`);
        }
      }
    }
    for (const [name, view] of [
      ['maximumDeviationRadius', output.maximumDeviationRadius],
      ['outputValidity', output.outputValidity]
    ] as const) {
      if (view) {
        validatePackedView(view, ['uint32'], `${id} output.${name}`);
        if (view.length !== pixelCount) {
          throw new Error(`${id} output.${name} must contain one value per pixel`);
        }
      }
    }
    validateRasterBuffersDistinct(
      id,
      [
        output.topographicPositionIndex,
        output.deviationFromMean,
        output.maximumDeviation,
        output.maximumDeviationRadius,
        output.outputValidity
      ],
      getRasterBandViews(props.elevation)
    );
  }

  /** Returns elevation canonicalization, summed-area table, and evaluation nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {output, scales} = props;
    const {width, height} = props.grid;
    validateRasterBandBelongsToGraph(id, graph, props.elevation, []);
    validateGraphViewsBelongToGraph(id, graph, [
      output.topographicPositionIndex,
      output.deviationFromMean,
      output.maximumDeviation,
      output.maximumDeviationRadius,
      output.outputValidity
    ]);
    const pixelCount = width * height;
    validateMaximumCount(
      id,
      'grid cells',
      pixelCount,
      getGPUTerrainSummedAreaMaximumCellCount(graph.device)
    );
    if (output.topographicPositionIndex || output.deviationFromMean) {
      validateMaximumCount(
        id,
        'grid cells x scales',
        pixelCount * props.scales.length,
        getGPUTerrainTopographicPositionMaximumPlaneCount(graph.device)
      );
    }
    const quantum = props.quantum ?? GPU_TERRAIN_TOPOGRAPHIC_POSITION_DEFAULT_QUANTUM;
    const source = getRasterBufferBandNodes(graph, id, props.elevation, props.grid, true);
    const summedArea = addTerrainSummedAreaTableNodes(graph, {
      id,
      width,
      height,
      elevation: source.band,
      quantum
    });
    const bindings: WGSLKernelBinding[] = [
      {name: 'lowTable', view: summedArea.lowTable, type: 'u32', access: 'read'},
      {name: 'highTable', view: summedArea.highTable, type: 'u32', access: 'read'}
    ];
    const outputs: [string, GraphDataView | undefined, 'f32' | 'u32'][] = [
      ['tpiValues', output.topographicPositionIndex, 'f32'],
      ['deviationValues', output.deviationFromMean, 'f32'],
      ['maximumDeviationValues', output.maximumDeviation, 'f32'],
      ['maximumRadiusValues', output.maximumDeviationRadius, 'u32'],
      ['validityValues', output.outputValidity, 'u32']
    ];
    for (const [name, view, type] of outputs) {
      if (view) {
        bindings.push({name, view, type, access: 'read_write'});
      }
    }
    const scaleList = scales
      .map(scale => `vec2<i32>(${scale.radius}i, ${scale.innerRadius ?? 0}i)`)
      .join(', ');
    const evaluate = createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-evaluate`,
      operation: 'GPUTerrainTopographicPosition',
      variant: 'evaluate',
      bindings,
      invocationCount: pixelCount,
      declarations: `const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;
const PIXEL_COUNT: u32 = ${pixelCount}u;
const SCALE_COUNT: u32 = ${scales.length}u;
const QUANTUM: f32 = ${getWGSLFloatLiteral(quantum)};
const INVERSE_QUANTUM: f32 = ${getWGSLFloatLiteral(1 / quantum)};
const SCALES = array<vec2<i32>, ${scales.length}>(${scaleList});
${TERRAIN_SUMMED_AREA_WGSL_HELPERS}
${TERRAIN_SUMMED_AREA_BOX_WGSL}
fn readClippedBox(column: i32, row: i32, radius: i32) -> TerrainBoxSums {
  return readBoxSums(
    max(column - radius, 0i),
    min(column + radius, i32(WIDTH) - 1i),
    max(row - radius, 0i),
    min(row + radius, i32(HEIGHT) - 1i)
  );
}
// Σ(q - q_c) over a box, exact modulo 2^64.
fn getCentredSum(box: TerrainBoxSums, magnitude: u32, isNegative: bool) -> vec2<u32> {
  var centreTotal = multiply64By32(vec2<u32>(box.count, 0u), magnitude);
  if (isNegative) {
    centreTotal = negate64(centreTotal);
  }
  return subtract64(box.sum, centreTotal);
}`,
      body: `let column = i32(index % WIDTH);
  let row = i32(index / WIDTH);
  // A single-cell box returns the centre's quantized elevation and validity exactly.
  let centre = readBoxSums(column, column, row, row);
  let isValid = centre.count == 1u;
  let invalidValue = bitcast<f32>(0x7fc00000u | (index & 0u));
  let quantized = bitcast<i32>(centre.sum.x);
  let magnitude = u32(abs(quantized));
  let isNegative = quantized < 0i;
  let centreSquare = multiplyWide32(magnitude, magnitude);
  var maximumDeviation = invalidValue;
  var maximumAbsolute = -1.0;
  var maximumRadius = 0u;
  for (var scaleIndex = 0u; scaleIndex < SCALE_COUNT; scaleIndex++) {
    let scale = SCALES[scaleIndex];
    let outer = readClippedBox(column, row, scale.x);
    // Population statistics relative to the centre: d1 = Σ(q - q_c), d2 = Σ(q - q_c)².
    let centredSum = getCentredSum(outer, magnitude, isNegative);
    var twiceCentreSum = multiply64By32(outer.sum, magnitude);
    twiceCentreSum = add64(twiceCentreSum, twiceCentreSum);
    if (isNegative) {
      twiceCentreSum = negate64(twiceCentreSum);
    }
    let centredSquare = add64(
      subtract64(outer.square, twiceCentreSum),
      multiply64By32(centreSquare, outer.count)
    );
    let count = f32(outer.count);
    let meanOffset = signed64ToFloat(centredSum) / count;
    // Review C-16: d2 / n - (d1 / n)^2 in f32 cancels when the centre is far from the window mean.
    // Re-centre exactly on the integer s nearest that mean: sum (d - s)^2 = d2 - 2 s d1 + n s^2 in
    // modular 64-bit integers, so only the residual mean (|r| about 0.5 quanta) is squared in f32.
    let shiftValue = round(meanOffset);
    let shiftMagnitude = u32(abs(shiftValue));
    let shiftNegative = shiftValue < 0.0;
    var shiftedCross = multiply64By32(centredSum, shiftMagnitude);
    shiftedCross = add64(shiftedCross, shiftedCross);
    var shiftTotal = multiply64By32(vec2<u32>(outer.count, 0u), shiftMagnitude);
    if (shiftNegative) {
      shiftedCross = negate64(shiftedCross);
      shiftTotal = negate64(shiftTotal);
    }
    let shiftedSquare = add64(
      subtract64(centredSquare, shiftedCross),
      multiply64By32(multiplyWide32(shiftMagnitude, shiftMagnitude), outer.count)
    );
    let residualMean = signed64ToFloat(subtract64(centredSum, shiftTotal)) / count;
    let variance = unsigned64ToFloat(shiftedSquare) / count - residualMean * residualMean;
    var deviation = 0.0;
    if (variance > 0.0) {
      deviation = -meanOffset / sqrt(variance);
    }
    if (!isValid) {
      deviation = invalidValue;
    }
    ${
      output.deviationFromMean
        ? 'deviationValues[deviationValuesOffset + scaleIndex * PIXEL_COUNT + index] = deviation;'
        : ''
    }
    if (isValid && abs(deviation) > maximumAbsolute) {
      maximumAbsolute = abs(deviation);
      maximumDeviation = deviation;
      maximumRadius = u32(scale.x);
    }
    ${
      output.topographicPositionIndex
        ? `let inner = readClippedBox(column, row, scale.y);
    let annulus = TerrainBoxSums(
      subtract64(outer.sum, inner.sum),
      subtract64(outer.square, inner.square),
      outer.count - inner.count
    );
    var tpi = invalidValue;
    if (isValid && annulus.count > 0u) {
      tpi = -signed64ToFloat(getCentredSum(annulus, magnitude, isNegative)) / f32(annulus.count) * QUANTUM;
    }
    tpiValues[tpiValuesOffset + scaleIndex * PIXEL_COUNT + index] = tpi;`
        : ''
    }
  }
  ${output.maximumDeviation ? 'maximumDeviationValues[maximumDeviationValuesOffset + index] = maximumDeviation;' : ''}
  ${output.maximumDeviationRadius ? 'maximumRadiusValues[maximumRadiusValuesOffset + index] = maximumRadius;' : ''}
  ${output.outputValidity ? 'validityValues[validityValuesOffset + index] = select(0u, 1u, isValid);' : ''}`
    });
    return [...source.nodes, ...summedArea.nodes, evaluate];
  }
}
