// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPURasterBand} from '../../gpu-raster/index';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {validateGraphViewsBelongToGraph} from '../../utils/gpu-contributor-utils';
import {
  getTerrainBandViews,
  getTerrainElevationNodes,
  validateTerrainBandBelongsToGraph,
  validateTerrainBuffersDistinct,
  validateTerrainGrid
} from '../terrain-analysis/terrain-analysis-utils';
import {
  addTerrainSummedAreaTableNodes,
  TERRAIN_SUMMED_AREA_BOX_WGSL,
  TERRAIN_SUMMED_AREA_WGSL_HELPERS
} from './terrain-summed-area-table';

/** Default elevation quantum, 1/256 elevation unit (the Terrarium encoding step in metres). */
export const GPU_TERRAIN_TOPOGRAPHIC_POSITION_DEFAULT_QUANTUM = 1 / 256;

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
  /** Grid width in pixels. */
  width: number;
  /** Grid height in pixels. */
  height: number;
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
  validity?: GraphDataView<'uint32'>;
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
 * `Σ(q - q_c)² = Σq² - 2 q_c Σq + N q_c²` (mod 2^64), so the only float32 rounding left is in the
 * final division and square root. The quantization changes each elevation by at most half a
 * quantum.
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
    const pixelCount = validateTerrainGrid(id, props.width, props.height);
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
      !props.topographicPositionIndex &&
      !props.deviationFromMean &&
      !props.maximumDeviation &&
      !props.maximumDeviationRadius &&
      !props.validity
    ) {
      throw new Error(`${id} requires at least one output`);
    }
    for (const [name, view, length] of [
      ['topographicPositionIndex', props.topographicPositionIndex, pixelCount * scales.length],
      ['deviationFromMean', props.deviationFromMean, pixelCount * scales.length],
      ['maximumDeviation', props.maximumDeviation, pixelCount]
    ] as const) {
      if (view) {
        validatePackedView(view, ['float32'], `${id} ${name}`);
        if (view.length !== length) {
          throw new Error(`${id} ${name} must contain ${length} values`);
        }
      }
    }
    for (const [name, view] of [
      ['maximumDeviationRadius', props.maximumDeviationRadius],
      ['validity', props.validity]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} ${name}`);
        if (view.length !== pixelCount) {
          throw new Error(`${id} ${name} must contain one value per pixel`);
        }
      }
    }
    validateTerrainBuffersDistinct(
      id,
      [
        props.topographicPositionIndex,
        props.deviationFromMean,
        props.maximumDeviation,
        props.maximumDeviationRadius,
        props.validity
      ],
      getTerrainBandViews(props.elevation)
    );
  }

  /** Returns elevation canonicalization, summed-area table, and evaluation nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {width, height, scales} = props;
    validateTerrainBandBelongsToGraph(id, graph, props.elevation, []);
    validateGraphViewsBelongToGraph(id, graph, [
      props.topographicPositionIndex,
      props.deviationFromMean,
      props.maximumDeviation,
      props.maximumDeviationRadius,
      props.validity
    ]);
    const pixelCount = width * height;
    const quantum = props.quantum ?? GPU_TERRAIN_TOPOGRAPHIC_POSITION_DEFAULT_QUANTUM;
    const source = getTerrainElevationNodes(graph, id, props.elevation, width, height, true);
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
      ['tpiValues', props.topographicPositionIndex, 'f32'],
      ['deviationValues', props.deviationFromMean, 'f32'],
      ['maximumDeviationValues', props.maximumDeviation, 'f32'],
      ['maximumRadiusValues', props.maximumDeviationRadius, 'u32'],
      ['validityValues', props.validity, 'u32']
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
    let variance = unsigned64ToFloat(centredSquare) / count - meanOffset * meanOffset;
    var deviation = 0.0;
    if (variance > 0.0) {
      deviation = -meanOffset / sqrt(variance);
    }
    if (!isValid) {
      deviation = invalidValue;
    }
    ${
      props.deviationFromMean
        ? 'deviationValues[deviationValuesOffset + scaleIndex * PIXEL_COUNT + index] = deviation;'
        : ''
    }
    if (isValid && abs(deviation) > maximumAbsolute) {
      maximumAbsolute = abs(deviation);
      maximumDeviation = deviation;
      maximumRadius = u32(scale.x);
    }
    ${
      props.topographicPositionIndex
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
  ${props.maximumDeviation ? 'maximumDeviationValues[maximumDeviationValuesOffset + index] = maximumDeviation;' : ''}
  ${props.maximumDeviationRadius ? 'maximumRadiusValues[maximumRadiusValuesOffset + index] = maximumRadius;' : ''}
  ${props.validity ? 'validityValues[validityValuesOffset + index] = select(0u, 1u, isValid);' : ''}`
    });
    return [...source.nodes, ...summedArea.nodes, evaluate];
  }
}
