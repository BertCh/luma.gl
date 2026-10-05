// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUReduction,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPURasterBand} from '../../gpu-raster/index';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {validateGraphViewsBelongToGraph} from '../../utils/gpu-contributor-utils';
import {
  getTerrainBandViews,
  getTerrainElevationNodes,
  TERRAIN_WGSL_HELPERS,
  validateTerrainBandBelongsToGraph,
  validateTerrainBuffersDistinct,
  validateTerrainGrid,
  validateTerrainSettings
} from '../terrain-analysis/terrain-analysis-utils';
import {
  TERRAIN_GEOMORPHOMETRY_CELL_SLOTS,
  type TerrainGeomorphometryCellSettings,
  writeTerrainGeomorphometryCellSettings
} from '../terrain-curvature/terrain-geomorphometry-utils';
import {
  GPUTerrainTopographicPosition,
  type GPUTerrainTopographicPositionScale
} from './gpu-terrain-topographic-position';
import {
  getTerrainGroundCellSizeWGSL,
  type GPUTerrainCellSizeMode,
  validateTerrainCellSizeMode
} from '../terrain-grid-utils';

/** Number of float32 values read from `GPUTerrainWeissLandformsProps.settings`. */
export const GPU_TERRAIN_WEISS_LANDFORMS_PARAMETER_LENGTH = 8;

/**
 * Weiss (2001) landform classes written by {@link GPUTerrainWeissLandforms}; 0 marks invalid cells.
 */
export const GPU_TERRAIN_WEISS_LANDFORMS = {
  /** Small and large TPI both low. */
  canyon: 1,
  /** Small TPI low, large TPI mid. */
  midslopeDrainage: 2,
  /** Small TPI low, large TPI high. */
  uplandDrainage: 3,
  /** Small TPI mid, large TPI low. */
  uShapedValley: 4,
  /** Both mid and slope at or below the slope threshold. */
  plain: 5,
  /** Both mid and slope above the slope threshold. */
  openSlope: 6,
  /** Small TPI mid, large TPI high. */
  upperSlope: 7,
  /** Small TPI high, large TPI low. */
  localRidge: 8,
  /** Small TPI high, large TPI mid. */
  midslopeRidge: 9,
  /** Small and large TPI both high. */
  mountainTop: 10
} as const;

/** One Weiss landform class name. */
export type GPUTerrainWeissLandform = keyof typeof GPU_TERRAIN_WEISS_LANDFORMS;

/**
 * How {@link GPUTerrainWeissLandforms} standardizes topographic position before thresholding.
 *
 * - `'global'` (Weiss 2001, Jenness Land Facets): z-score of each TPI scale against the mean and
 *   population standard deviation of that scale over every valid cell of the grid.
 * - `'local'`: the deviation from mean elevation (DEV) of each scale's full window, which is the
 *   TPI standardized by its own neighbourhood. Cheaper (no reductions) and tile-local.
 */
export type GPUTerrainWeissStandardization = 'global' | 'local';

/**
 * CPU-side description packed by {@link getGPUTerrainWeissLandformsParameterValues}.
 *
 * Cell size model: `cellSize` is in projected metres (`'uniform'`), equatorial Web Mercator metres
 * (`'web-mercator'`, scaled by cos(latitude) per row), or degrees (`'geographic'`, 111319.49 m per
 * degree with the x size scaled by cos(latitude) per row). It only affects the slope used to split
 * plains from open slopes; TPI scales are in cells.
 */
export type GPUTerrainWeissLandformsSettings = TerrainGeomorphometryCellSettings & {
  /** Standardized position threshold separating low, mid, and high. Defaults to 1. */
  standardThreshold?: number;
  /** Slope in degrees at or below which a mid/mid cell is a plain. Defaults to 5. */
  slopeThresholdDegrees?: number;
};

/**
 * Packs settings into the 8-float layout read by {@link GPUTerrainWeissLandforms}:
 * `[cellSizeX, cellSizeY, zFactor, northEdge, southEdge, standardThreshold,
 * slopeThresholdDegrees, 0]`.
 */
export function getGPUTerrainWeissLandformsParameterValues(
  settings: GPUTerrainWeissLandformsSettings,
  target: Float32Array = new Float32Array(GPU_TERRAIN_WEISS_LANDFORMS_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_TERRAIN_WEISS_LANDFORMS_PARAMETER_LENGTH) {
    throw new Error('Weiss landform settings target must hold 8 values');
  }
  writeTerrainGeomorphometryCellSettings(settings, target);
  target[5] = settings.standardThreshold ?? 1;
  target[6] = settings.slopeThresholdDegrees ?? 5;
  target[7] = 0;
  return target;
}

/**
 * Properties for {@link GPUTerrainWeissLandforms}.
 *
 * Topology: grid size, scales, standardization, quantum, which outputs exist, and `cellSizeMode`. Per-frame: `settings` and elevation contents. See
 * {@link GPUTerrainWeissLandformsSettings} for the cell-size model.
 */
export type GPUTerrainWeissLandformsProps = {
  /** Prefix for node and transient IDs. Defaults to `'terrain-weiss-landforms'`. */
  id?: string;
  /** Grid width in pixels. */
  width: number;
  /** Grid height in pixels. */
  height: number;
  /** Elevation band, buffer or texture; scale, offset, nodata, and validity are honored. */
  elevation: GPURasterBand;
  /** Per-frame settings with at least 8 float32 values. */
  settings: GraphDataView<'float32'>;
  /** Small neighbourhood in cells. Defaults to `{radius: 3}`. */
  smallScale?: GPUTerrainTopographicPositionScale;
  /** Large neighbourhood in cells. Defaults to `{radius: 15}`. */
  largeScale?: GPUTerrainTopographicPositionScale;
  /** Standardization. Defaults to `'global'`. */
  standardization?: GPUTerrainWeissStandardization;
  /** Elevation quantum of the summed-area table, see `GPUTerrainTopographicPositionProps`. */
  quantum?: number;
  /** Optional class per pixel, 1 to 10 as in {@link GPU_TERRAIN_WEISS_LANDFORMS}; 0 when invalid. */
  landforms?: GraphDataView<'uint32'>;
  /** Optional standardized positions: two planes of `width * height` values, small then large. */
  standardizedPosition?: GraphDataView<'float32'>;
  /** Optional per-pixel 1 where the class is valid, else 0. */
  validity?: GraphDataView<'uint32'>;
  /** Cell size interpretation for the slope. Defaults to `'uniform'`. */
  cellSizeMode?: GPUTerrainCellSizeMode;
};

/**
 * Weiss (2001) topographic-position landform classification into ten classes from TPI at a small
 * and a large scale plus slope.
 *
 * TPI comes from {@link GPUTerrainTopographicPosition} (square annuli, exact summed-area table).
 * With `'global'` standardization, `GPUReduction` sums form each scale's grid mean and, in a second
 * pass over centred squares, its population standard deviation, so no float32 `E[x²] - E[x]²`
 * cancellation occurs. Slope is Horn's 3×3 slope with clamped borders. A cell is invalid when its
 * elevation, either position value, or its slope is invalid. Global statistics make the classes
 * depend on the whole tile, as in the original ArcView method; use `'local'` for tile-independent
 * classes.
 *
 * Classes (`s`, `l` = small and large standardized position, `t` = `standardThreshold`):
 * `s ≤ -t`: canyon (`l ≤ -t`), midslope drainage, upland drainage (`l ≥ t`);
 * `-t < s < t`: U-shaped valley (`l ≤ -t`), plain or open slope (split by slope), upper slope
 * (`l ≥ t`); `s ≥ t`: local ridge (`l ≤ -t`), midslope ridge, mountain top (`l ≥ t`).
 */
export class GPUTerrainWeissLandforms implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUTerrainWeissLandformsProps;
  /** Receptive field in pixels for the local terms; global standardization reads the whole grid. */
  readonly requiredHalo: number;

  constructor(props: GPUTerrainWeissLandformsProps) {
    this.id = props.id ?? 'terrain-weiss-landforms';
    this.props = props;
    const {id} = this;
    const pixelCount = validateTerrainGrid(id, props.width, props.height);
    const scales = this.getScales();
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
    this.requiredHalo = Math.max(1, ...scales.map(scale => scale.radius));
    if (!['global', 'local'].includes(props.standardization ?? 'global')) {
      throw new Error(`${id} standardization must be global or local`);
    }
    validateTerrainCellSizeMode(id, props.cellSizeMode);
    validateTerrainSettings(id, props.settings, GPU_TERRAIN_WEISS_LANDFORMS_PARAMETER_LENGTH);
    if (!props.landforms && !props.standardizedPosition && !props.validity) {
      throw new Error(`${id} requires at least one output`);
    }
    if (props.standardizedPosition) {
      validatePackedView(props.standardizedPosition, ['float32'], `${id} standardizedPosition`);
      if (props.standardizedPosition.length !== 2 * pixelCount) {
        throw new Error(`${id} standardizedPosition must contain two values per pixel`);
      }
    }
    for (const [name, view] of [
      ['landforms', props.landforms],
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
      [props.landforms, props.standardizedPosition, props.validity],
      [...getTerrainBandViews(props.elevation), props.settings]
    );
  }

  /** Returns position, slope, optional global statistics, and classification nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {width, height} = props;
    validateTerrainBandBelongsToGraph(id, graph, props.elevation, []);
    validateGraphViewsBelongToGraph(id, graph, [
      props.settings,
      props.landforms,
      props.standardizedPosition,
      props.validity
    ]);
    const pixelCount = width * height;
    const isGlobal = (props.standardization ?? 'global') === 'global';
    const source = getTerrainElevationNodes(graph, id, props.elevation, width, height, true);
    const elevationValidity = source.band.validity;
    if (!elevationValidity) {
      throw new Error(`${id} canonical elevation is missing validity`);
    }
    const nodes: GPUCommandNode<Parameters>[] = [...source.nodes];
    const position = createTransientView(graph, `${id}-position`, 'float32', 2 * pixelCount);
    const positionValidity = createTransientView(
      graph,
      `${id}-position-validity`,
      'uint32',
      pixelCount
    );
    nodes.push(
      ...new GPUTerrainTopographicPosition({
        id: `${id}-position`,
        width,
        height,
        elevation: source.band,
        scales: this.getScales(),
        quantum: props.quantum,
        ...(isGlobal ? {topographicPositionIndex: position} : {deviationFromMean: position}),
        validity: positionValidity
      }).getCommandNodes(graph)
    );
    const slope = createTransientView(graph, `${id}-slope`, 'float32', pixelCount);
    const mask = createTransientView(graph, `${id}-mask`, 'uint32', pixelCount);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-slope`,
        operation: 'GPUTerrainWeissLandforms',
        variant: `slope-${props.cellSizeMode ?? 'uniform'}`,
        bindings: [
          {
            name: 'elevationValues',
            view: source.band.storage.values,
            type: 'f32',
            access: 'read'
          },
          {name: 'elevationValidity', view: elevationValidity, type: 'u32', access: 'read'},
          {name: 'settings', view: props.settings, type: 'f32', access: 'read'},
          {name: 'positionValues', view: position, type: 'f32', access: 'read'},
          {name: 'positionValidity', view: positionValidity, type: 'u32', access: 'read'},
          {name: 'slopeValues', view: slope, type: 'f32', access: 'read_write'},
          {name: 'maskValues', view: mask, type: 'u32', access: 'read_write'}
        ],
        invocationCount: pixelCount,
        declarations: `const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;
const PIXEL_COUNT: u32 = ${pixelCount}u;
${TERRAIN_WGSL_HELPERS}
${getTerrainGroundCellSizeWGSL(props.cellSizeMode ?? 'uniform', TERRAIN_GEOMORPHOMETRY_CELL_SLOTS)}
fn readElevation(column: i32, row: i32) -> vec2<f32> {
  let sampleColumn = u32(clamp(column, 0i, i32(WIDTH) - 1i));
  let sampleRow = u32(clamp(row, 0i, i32(HEIGHT) - 1i));
  let sampleIndex = sampleRow * WIDTH + sampleColumn;
  let value = elevationValues[elevationValuesOffset + sampleIndex];
  let isValid = elevationValidity[elevationValidityOffset + sampleIndex] != 0u && isFiniteValue(value);
  return vec2<f32>(value, select(0.0, 1.0, isValid));
}`,
        body: `let column = i32(index % WIDTH);
  let row = i32(index / WIDTH);
  let centre = readElevation(column, row);
  var isValid = centre.y != 0.0;
  // Horn weights over centre-relative differences, which are exact in float32.
  var eastSum = 0.0;
  var southSum = 0.0;
  for (var dy = -1i; dy <= 1i; dy++) {
    for (var dx = -1i; dx <= 1i; dx++) {
      if (dx == 0i && dy == 0i) {
        continue;
      }
      let sample = readElevation(column + dx, row + dy);
      isValid = isValid && sample.y != 0.0;
      let weight = select(1.0, 2.0, dx == 0i || dy == 0i);
      let difference = sample.x - centre.x;
      eastSum += f32(dx) * weight * difference;
      southSum += f32(dy) * weight * difference;
    }
  }
  let groundCell = getGroundCellSize(u32(row));
  let zFactor = settings[settingsOffset + 2u];
  // Only the gradient magnitude matters, so the row direction does not.
  let eastGradient = zFactor * eastSum / (8.0 * groundCell.x);
  let rowGradient = zFactor * southSum / (8.0 * groundCell.y);
  let slope = atan(length(vec2<f32>(eastGradient, rowGradient))) * 57.29577951308232;
  let small = positionValues[positionValuesOffset + index];
  let large = positionValues[positionValuesOffset + PIXEL_COUNT + index];
  isValid = isValid && isFiniteValue(slope) && isFiniteValue(small) && isFiniteValue(large) &&
    positionValidity[positionValidityOffset + index] != 0u;
  slopeValues[slopeValuesOffset + index] = select(bitcast<f32>(0x7fc00000u | (index & 0u)), slope, isValid);
  maskValues[maskValuesOffset + index] = select(0u, 1u, isValid);`
      })
    );
    const classifyBindings: WGSLKernelBinding[] = [
      {name: 'positionValues', view: position, type: 'f32', access: 'read'},
      {name: 'slopeValues', view: slope, type: 'f32', access: 'read'},
      {name: 'settings', view: props.settings, type: 'f32', access: 'read'}
    ];
    if (isGlobal) {
      const statistics = this.addGlobalStatisticsNodes(graph, nodes, position, mask, pixelCount);
      classifyBindings.push(
        {name: 'statisticsValues', view: statistics.values, type: 'f32', access: 'read'},
        {name: 'statisticsCount', view: statistics.count, type: 'u32', access: 'read'}
      );
    }
    for (const [name, view, type] of [
      ['landformValues', props.landforms, 'u32'],
      ['standardizedValues', props.standardizedPosition, 'f32'],
      ['validityValues', props.validity, 'u32']
    ] as const) {
      if (view) {
        classifyBindings.push({name, view, type, access: 'read_write'});
      }
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-classify`,
        operation: 'GPUTerrainWeissLandforms',
        variant: isGlobal ? 'classify-global' : 'classify-local',
        bindings: classifyBindings,
        invocationCount: pixelCount,
        declarations: `const PIXEL_COUNT: u32 = ${pixelCount}u;
${TERRAIN_WGSL_HELPERS}
fn standardize(value: f32, scale: u32) -> f32 {
  ${
    isGlobal
      ? `let count = f32(statisticsCount[statisticsCountOffset]);
  let mean = statisticsValues[statisticsValuesOffset + scale] / count;
  let deviation = sqrt(statisticsValues[statisticsValuesOffset + 2u + scale] / count);
  return select(0.0, (value - mean) / deviation, deviation > 0.0);`
      : 'return value;'
  }
}
// 0 = low (at or below -threshold), 1 = mid, 2 = high (at or above threshold).
fn getPositionClass(value: f32, threshold: f32) -> u32 {
  if (value <= -threshold) {
    return 0u;
  }
  return select(1u, 2u, value >= threshold);
}`,
        body: `let slope = slopeValues[slopeValuesOffset + index];
  let threshold = settings[settingsOffset + 5u];
  let slopeThreshold = settings[settingsOffset + 6u];
  let small = standardize(positionValues[positionValuesOffset + index], 0u);
  let large = standardize(positionValues[positionValuesOffset + PIXEL_COUNT + index], 1u);
  let isValid = isFiniteValue(slope) && isFiniteValue(small) && isFiniteValue(large) &&
    isFiniteValue(threshold) && threshold > 0.0 && isFiniteValue(slopeThreshold);
  let smallClass = getPositionClass(small, threshold);
  let largeClass = getPositionClass(large, threshold);
  // Low small position: canyon 1, midslope drainage 2, upland drainage 3.
  var landform = 1u + largeClass;
  if (smallClass == 1u) {
    // Mid small position: valley 4, plain 5, open slope 6, upper slope 7.
    landform = select(select(4u, 7u, largeClass == 2u), select(6u, 5u, slope <= slopeThreshold), largeClass == 1u);
  } else if (smallClass == 2u) {
    // High small position: local ridge 8, midslope ridge 9, mountain top 10.
    landform = 8u + largeClass;
  }
  let invalidValue = bitcast<f32>(0x7fc00000u | (index & 0u));
  ${props.landforms ? 'landformValues[landformValuesOffset + index] = select(0u, landform, isValid);' : ''}
  ${
    props.standardizedPosition
      ? `standardizedValues[standardizedValuesOffset + index] = select(invalidValue, small, isValid);
  standardizedValues[standardizedValuesOffset + PIXEL_COUNT + index] = select(invalidValue, large, isValid);`
      : ''
  }
  ${props.validity ? 'validityValues[validityValuesOffset + index] = select(0u, 1u, isValid);' : ''}`
      })
    );
    return nodes;
  }

  /** Returns the small and large scales with defaults applied. */
  private getScales(): GPUTerrainTopographicPositionScale[] {
    return [this.props.smallScale ?? {radius: 3}, this.props.largeScale ?? {radius: 15}];
  }

  /**
   * Adds the count, per-scale sum, centred-square, and per-scale square-sum reductions.
   *
   * `values` holds `[sumSmall, sumLarge, squareSmall, squareLarge]`.
   */
  private addGlobalStatisticsNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    nodes: GPUCommandNode<Parameters>[],
    position: GraphDataView<'float32'>,
    mask: GraphDataView<'uint32'>,
    pixelCount: number
  ): {values: GraphDataView<'float32'>; count: GraphDataView<'uint32'>} {
    const {id} = this;
    const values = createTransientView(graph, `${id}-statistics`, 'float32', 4);
    const count = createTransientView(graph, `${id}-statistics-count`, 'uint32', 1);
    const getPlane = (plane: number) =>
      graph.createDataView(position.buffer, {
        format: 'float32',
        length: pixelCount,
        byteOffset: position.byteOffset + plane * pixelCount * 4
      });
    const getStatistic = (element: number) =>
      graph.createDataView(values.buffer, {
        format: 'float32',
        length: 1,
        byteOffset: values.byteOffset + element * 4
      });
    nodes.push(
      ...new GPUReduction({
        id: `${id}-count`,
        input: mask,
        output: count,
        operation: 'sum'
      }).getCommandNodes(graph)
    );
    for (const plane of [0, 1]) {
      nodes.push(
        ...new GPUReduction({
          id: `${id}-sum-${plane}`,
          input: getPlane(plane),
          mask,
          output: getStatistic(plane),
          operation: 'sum'
        }).getCommandNodes(graph)
      );
    }
    const squares = createTransientView(graph, `${id}-squares`, 'float32', 2 * pixelCount);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-squares`,
        operation: 'GPUTerrainWeissLandforms',
        variant: 'centred-squares',
        bindings: [
          {name: 'positionValues', view: position, type: 'f32', access: 'read'},
          {name: 'maskValues', view: mask, type: 'u32', access: 'read'},
          {name: 'statisticsValues', view: values, type: 'f32', access: 'read'},
          {name: 'statisticsCount', view: count, type: 'u32', access: 'read'},
          {name: 'squareValues', view: squares, type: 'f32', access: 'read_write'}
        ],
        invocationCount: 2 * pixelCount,
        declarations: `const PIXEL_COUNT: u32 = ${pixelCount}u;`,
        body: `let scale = index / PIXEL_COUNT;
  let pixel = index - scale * PIXEL_COUNT;
  let mean = statisticsValues[statisticsValuesOffset + scale] / f32(statisticsCount[statisticsCountOffset]);
  let difference = positionValues[positionValuesOffset + index] - mean;
  squareValues[squareValuesOffset + index] = select(0.0, difference * difference, maskValues[maskValuesOffset + pixel] != 0u);`
      })
    );
    for (const plane of [0, 1]) {
      nodes.push(
        ...new GPUReduction({
          id: `${id}-square-sum-${plane}`,
          input: graph.createDataView(squares.buffer, {
            format: 'float32',
            length: pixelCount,
            byteOffset: squares.byteOffset + plane * pixelCount * 4
          }),
          mask,
          output: getStatistic(2 + plane),
          operation: 'sum'
        }).getCommandNodes(graph)
      );
    }
    return {values, count};
  }
}
