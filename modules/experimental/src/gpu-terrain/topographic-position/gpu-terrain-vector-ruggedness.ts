// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPURasterBand, GPURasterBorderMode, GPURasterMetadata} from '../../gpu-raster/index';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {
  getRasterBandViews,
  getRasterBufferBandNodes,
  RASTER_WGSL_HELPERS,
  validateRasterBandBelongsToGraph,
  validateRasterBuffersDistinct,
  validateParameterTarget,
  validateRasterGrid,
  validateRasterGridAxisAligned,
  validateRasterParameters
} from '../../gpu-raster/raster-band-utils';
import {
  TERRAIN_GEOMORPHOMETRY_CELL_SLOTS,
  writeTerrainGeomorphometryCellSettings,
  type TerrainGeomorphometryCellSettings
} from '../terrain-curvature/terrain-geomorphometry-utils';
import {
  getRasterGroundCellSizeWGSL,
  getRasterRowNorthSignWGSL,
  type GPURasterCellSizeMode,
  type GPURasterRowDirection,
  validateRasterCellSizeMode,
  validateRasterRowDirection
} from '../../analysis-core/raster-grid';

/** Window radius from which the window sum runs as two separable passes instead of one square loop. */
const SEPARABLE_RADIUS = 3;

/** Number of float32 values read from `GPUTerrainVectorRuggednessProps.parameters`. */
export const GPU_TERRAIN_VECTOR_RUGGEDNESS_PARAMETER_LENGTH = 8;

/** Index of the per-frame window radius in the packed parameters. */
const RADIUS_PARAMETER_INDEX = 5;

/**
 * CPU-side description packed by {@link getGPUTerrainVectorRuggednessParameterValues}.
 *
 * The cell-size model depends on `GPUTerrainVectorRuggednessProps.cellSizeMode`: projected metres
 * for `'uniform'`, EPSG:3857 metres for `'web-mercator'`, or degrees for `'geographic'` (x spacing
 * scales with cos(latitude) of each row). Cell size and latitude band come from the grid.
 */
export type GPUTerrainVectorRuggednessSettings = TerrainGeomorphometryCellSettings & {
  /**
   * Window half-width in pixels: the window is `(2 * radius + 1)` square. Integer in
   * `[1, maximumRadius]`. Defaults to `maximumRadius`. The packer throws for a value that is not
   * an integer of at least 1; a value above `maximumRadius` written to the buffer by other means
   * is clamped to `maximumRadius` by the kernel.
   */
  radius?: number;
};

/**
 * Packs settings into the 8-float layout read by {@link GPUTerrainVectorRuggedness}:
 * `[cellSizeX, cellSizeY, zFactor, northEdge, southEdge, radius, 0, 0]`, with the cell size and
 * the row-edge world y taken from `grid`. `radius` is 0 when `settings.radius` is omitted, which
 * the kernel reads as `maximumRadius`.
 */
export function getGPUTerrainVectorRuggednessParameterValues(
  settings: GPUTerrainVectorRuggednessSettings,
  grid: GPURasterMetadata,
  target: Float32Array = new Float32Array(GPU_TERRAIN_VECTOR_RUGGEDNESS_PARAMETER_LENGTH)
): Float32Array {
  validateParameterTarget(
    'getGPUTerrainVectorRuggednessParameterValues',
    target,
    GPU_TERRAIN_VECTOR_RUGGEDNESS_PARAMETER_LENGTH
  );
  if (
    settings.radius !== undefined &&
    (!Number.isSafeInteger(settings.radius) || settings.radius < 1)
  ) {
    throw new Error(
      'getGPUTerrainVectorRuggednessParameterValues radius must be an integer of at least 1'
    );
  }
  target.fill(0, 0, GPU_TERRAIN_VECTOR_RUGGEDNESS_PARAMETER_LENGTH);
  writeTerrainGeomorphometryCellSettings(settings, grid, target);
  target[RADIUS_PARAMETER_INDEX] = settings.radius ?? 0;
  return target;
}

/** Caller-owned outputs of {@link GPUTerrainVectorRuggedness}; at least one is required. */
export type GPUTerrainVectorRuggednessOutput = {
  /** Optional vector ruggedness measure per pixel in `[0, 1]`; NaN where invalid. */
  vectorRuggedness?: GraphDataView<'float32'>;
  /** Optional per-pixel 1 where the output is valid, else 0. */
  outputValidity?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUTerrainVectorRuggedness}.
 *
 * Cell-size model: `cellSizeMode` selects projected metres (`'uniform'`), `'web-mercator'`, or
 * `'geographic'` degrees with latitude-dependent row spacing; see
 * {@link GPUTerrainVectorRuggednessSettings}.
 *
 * Topology: grid size, elevation format, `maximumRadius`, `cellSizeMode`, `rowDirection`,
 * `borderMode`. Per-frame: `parameters` (cell size, latitude band, `radius`) and elevation.
 */
export type GPUTerrainVectorRuggednessProps = {
  /** Prefix for node and transient IDs. Defaults to `'terrain-vector-ruggedness'`. */
  id?: string;
  /** Raster grid: its width and height size the kernels. */
  grid: GPURasterMetadata;
  /** Elevation band, buffer or texture; scale, offset, nodata, and validity are honored. */
  elevation: GPURasterBand;
  /** Per-frame parameters with at least 8 float32 values, see {@link getGPUTerrainVectorRuggednessParameterValues}. */
  parameters: GraphDataView<'float32'>;
  /**
   * Largest window half-width in pixels the contributor serves. Integer >= 1, default 1. It sizes
   * `requiredHalo`, selects the direct or separable window sum, and bounds the loops; the
   * per-frame `radius` of the settings picks any value in `[1, maximumRadius]` with the same
   * pipelines.
   */
  maximumRadius?: number;
  /** Caller-owned outputs. */
  output: GPUTerrainVectorRuggednessOutput;
  /** Cell size interpretation. Defaults to `'uniform'`. */
  cellSizeMode?: GPURasterCellSizeMode;
  /** Direction in which the row index increases. Defaults to `'south'` (north-up rasters). */
  rowDirection?: GPURasterRowDirection;
  /**
   * Treatment of the raster border when computing Horn normals: `'clamp'` repeats edge samples
   * (default), `'nodata'` leaves border normals invalid.
   */
  borderMode?: GPURasterBorderMode;
};

/**
 * Computes the vector ruggedness measure (VRM; Sappington, Longshore and Thompson 2007).
 *
 * A first kernel derives a unit surface normal per cell with Horn's 3x3 method; a second sums the
 * normals of the `(2 * radius + 1)` square window (clipped to the raster and to cells with a valid
 * normal) and returns `1 - |sum| / count`, clamped to `[0, 1]`. `radius` is a per-frame setting in
 * `[1, maximumRadius]`, so one pipeline serves every radius. With `maximumRadius` below 3 the
 * kernel sums the square window directly; from 3 the clipped box sum runs as a row pass and a
 * column pass (two kernels instead of one), so cost is O(radius) per pixel instead of
 * O(radius^2). An invalid centre normal gives NaN and validity 0. Satisfies the
 * `GPURasterHaloStage` contract with `requiredHalo = maximumRadius + 1`.
 */
export class GPUTerrainVectorRuggedness implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUTerrainVectorRuggednessProps;
  /** Receptive field in pixels (`GPURasterHaloStage` contract): `maximumRadius + 1`. */
  readonly requiredHalo: number;

  constructor(props: GPUTerrainVectorRuggednessProps) {
    this.id = props.id ?? 'terrain-vector-ruggedness';
    this.props = props;
    const {id} = this;
    const {output} = props;
    const maximumRadius = props.maximumRadius ?? 1;
    if (!Number.isSafeInteger(maximumRadius) || maximumRadius < 1) {
      throw new Error(`${id} maximumRadius must be an integer of at least 1`);
    }
    this.requiredHalo = maximumRadius + 1;
    const pixelCount = validateRasterGrid(id, props.grid);
    if (!output.vectorRuggedness && !output.outputValidity) {
      throw new Error(`${id} requires at least one output`);
    }
    if (output.vectorRuggedness) {
      validatePackedView(output.vectorRuggedness, ['float32'], `${id} output.vectorRuggedness`);
      if (output.vectorRuggedness.length !== pixelCount) {
        throw new Error(`${id} output.vectorRuggedness must contain one value per pixel`);
      }
    }
    if (output.outputValidity) {
      validatePackedView(output.outputValidity, ['uint32'], `${id} output.outputValidity`);
      if (output.outputValidity.length !== pixelCount) {
        throw new Error(`${id} output.outputValidity must contain one value per pixel`);
      }
    }
    validateRasterParameters(id, props.parameters, GPU_TERRAIN_VECTOR_RUGGEDNESS_PARAMETER_LENGTH);
    validateRasterCellSizeMode(id, props.cellSizeMode);
    validateRasterGridAxisAligned(id, props.grid, props.cellSizeMode);
    validateRasterRowDirection(id, props.rowDirection);
    if (!['clamp', 'nodata'].includes(props.borderMode ?? 'clamp')) {
      throw new Error(`${id} borderMode must be clamp or nodata`);
    }
    validateRasterBuffersDistinct(
      id,
      [output.vectorRuggedness, output.outputValidity],
      [...getRasterBandViews(props.elevation), props.parameters]
    );
  }

  /** Returns the optional elevation canonicalization node, the normal kernel and the window kernel. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {output} = props;
    const {width, height} = props.grid;
    validateRasterBandBelongsToGraph(id, graph, props.elevation, []);
    validateGraphViewsBelongToGraph(id, graph, [
      props.parameters,
      output.vectorRuggedness,
      output.outputValidity
    ]);
    const pixelCount = width * height;
    const source = getRasterBufferBandNodes(graph, id, props.elevation, props.grid, true);
    const normals = (['x', 'y', 'z'] as const).map(component =>
      createTransientView(graph, `${id}-normal-${component}`, 'float32', pixelCount)
    );
    const normalNode = createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-normals`,
      operation: 'GPUTerrainVectorRuggedness',
      variant: `normals-${props.cellSizeMode ?? 'uniform'}-${props.borderMode ?? 'clamp'}`,
      bindings: [
        {name: 'elevation', view: source.band.storage.values, type: 'f32', access: 'read'},
        {
          name: 'elevationValidity',
          view: source.band.validity as GraphDataView<'uint32'>,
          type: 'u32',
          access: 'read'
        },
        {name: 'settings', view: props.parameters, type: 'f32', access: 'read'},
        {name: 'normalX', view: normals[0], type: 'f32', access: 'read_write'},
        {name: 'normalY', view: normals[1], type: 'f32', access: 'read_write'},
        {name: 'normalZ', view: normals[2], type: 'f32', access: 'read_write'}
      ],
      invocationCount: pixelCount,
      declarations: `const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;
${getRasterRowNorthSignWGSL(props.rowDirection ?? 'south')}
${RASTER_WGSL_HELPERS}
${getRasterGroundCellSizeWGSL(props.cellSizeMode ?? 'uniform', TERRAIN_GEOMORPHOMETRY_CELL_SLOTS)}`,
      body: `let row = i32(index / WIDTH);
  let column = i32(index % WIDTH);
  var isValid = true;
  var windowValues: array<f32, 9>;
  for (var k = 0; k < 9; k++) {
    var sampleRow = row + k / 3 - 1;
    var sampleColumn = column + k % 3 - 1;
    ${
      (props.borderMode ?? 'clamp') === 'nodata'
        ? `if (sampleRow < 0 || sampleRow >= i32(HEIGHT) || sampleColumn < 0 || sampleColumn >= i32(WIDTH)) {
      isValid = false;
      continue;
    }`
        : ''
    }
    sampleRow = clamp(sampleRow, 0, i32(HEIGHT) - 1);
    sampleColumn = clamp(sampleColumn, 0, i32(WIDTH) - 1);
    let sampleIndex = u32(sampleRow) * WIDTH + u32(sampleColumn);
    let sampleValue = elevation[elevationOffset + sampleIndex];
    isValid = isValid && elevationValidity[elevationValidityOffset + sampleIndex] != 0u &&
      isFiniteValue(sampleValue);
    windowValues[k] = sampleValue;
  }
  let groundCell = getGroundCellSize(u32(row));
  let zFactor = settings[settingsOffset + 2u];
  // Horn: weights 1-2-1 across three rows or columns, divided by 8 cell sizes.
  let eastGradient = zFactor *
    ((windowValues[2] + 2.0 * windowValues[5] + windowValues[8]) -
     (windowValues[0] + 2.0 * windowValues[3] + windowValues[6])) / (8.0 * groundCell.x);
  let northGradient = ROW_NORTH_SIGN * zFactor *
    ((windowValues[6] + 2.0 * windowValues[7] + windowValues[8]) -
     (windowValues[0] + 2.0 * windowValues[1] + windowValues[2])) / (8.0 * groundCell.y);
  isValid = isValid && groundCell.x > 0.0 && groundCell.y > 0.0 &&
    isFiniteValue(eastGradient) && isFiniteValue(northGradient);
  let normal = normalize(vec3<f32>(-eastGradient, -northGradient, 1.0));
  let invalidValue = bitcast<f32>(0x7fc00000u | (index & 0u));
  normalX[normalXOffset + index] = select(invalidValue, normal.x, isValid);
  normalY[normalYOffset + index] = select(invalidValue, normal.y, isValid);
  normalZ[normalZOffset + index] = select(invalidValue, normal.z, isValid);`
    });

    const maximumRadius = props.maximumRadius ?? 1;
    const settingsBinding: WGSLKernelBinding = {
      name: 'settings',
      view: props.parameters,
      type: 'f32',
      access: 'read'
    };
    const bindings: WGSLKernelBinding[] = [
      {name: 'normalX', view: normals[0], type: 'f32', access: 'read'},
      {name: 'normalY', view: normals[1], type: 'f32', access: 'read'},
      {name: 'normalZ', view: normals[2], type: 'f32', access: 'read'},
      settingsBinding
    ];
    if (output.vectorRuggedness) {
      bindings.push({
        name: 'vrmValues',
        view: output.vectorRuggedness,
        type: 'f32',
        access: 'read_write'
      });
    }
    if (output.outputValidity) {
      bindings.push({
        name: 'validityValues',
        view: output.outputValidity,
        type: 'u32',
        access: 'read_write'
      });
    }
    // Per-frame radius word: 0 (omitted) or invalid selects maximumRadius; larger values clamp.
    const radiusSource = `let radiusWord = settings[settingsOffset + ${RADIUS_PARAMETER_INDEX}u];
  let radius = select(RADIUS, clamp(i32(radiusWord), 1, RADIUS), radiusWord >= 1.0);`;
    const finishBody = (
      sumExpression: string
    ) => `let invalidValue = bitcast<f32>(0x7fc00000u | (index & 0u));
  let isValid = isFiniteValue(normalX[normalXOffset + index]);
  var vectorSum = vec3<f32>(0.0);
  var count = 0.0;
  if (isValid) {
    ${sumExpression}
  }
  // count >= 1 whenever the centre normal is valid.
  let ruggedness = clamp(1.0 - length(vectorSum) / max(count, 1.0), 0.0, 1.0);
  ${output.vectorRuggedness ? 'vrmValues[vrmValuesOffset + index] = select(invalidValue, ruggedness, isValid);' : ''}
  ${output.outputValidity ? 'validityValues[validityValuesOffset + index] = select(0u, 1u, isValid);' : ''}`;
    if (maximumRadius < SEPARABLE_RADIUS) {
      // Small windows: summing the (2r + 1)^2 normals directly is cheaper than two extra passes.
      const windowNode = createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-window`,
        operation: 'GPUTerrainVectorRuggedness',
        variant: 'window',
        bindings,
        invocationCount: pixelCount,
        declarations: `const WIDTH: i32 = ${width};
const HEIGHT: i32 = ${height};
const RADIUS: i32 = ${maximumRadius};
${RASTER_WGSL_HELPERS}`,
        body: `let row = i32(index) / WIDTH;
  let column = i32(index) % WIDTH;
  ${radiusSource}
  ${finishBody(`for (var sampleRow = max(row - radius, 0); sampleRow <= min(row + radius, HEIGHT - 1); sampleRow++) {
      for (var sampleColumn = max(column - radius, 0); sampleColumn <= min(column + radius, WIDTH - 1); sampleColumn++) {
        let sampleIndex = u32(sampleRow * WIDTH + sampleColumn);
        let normalXValue = normalX[normalXOffset + sampleIndex];
        if (isFiniteValue(normalXValue)) {
          vectorSum += vec3<f32>(normalXValue, normalY[normalYOffset + sampleIndex], normalZ[normalZOffset + sampleIndex]);
          count += 1.0;
        }
      }
    }`)}`
      });
      return [...source.nodes, normalNode, windowNode];
    }
    // Large windows: the clipped box sum of valid unit normals is separable, so a row pass followed
    // by a column pass costs O(r) per pixel instead of O(r^2). Normals have magnitude 1, so plain
    // float32 sums stay well conditioned (unlike elevation, no large offset cancels).
    const rowSums = (['x', 'y', 'z'] as const).map(component =>
      createTransientView(graph, `${id}-row-sum-${component}`, 'float32', pixelCount)
    );
    const rowCounts = createTransientView(graph, `${id}-row-count`, 'uint32', pixelCount);
    const rowNode = createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-window-rows`,
      operation: 'GPUTerrainVectorRuggedness',
      variant: 'window-rows',
      bindings: [
        {name: 'normalX', view: normals[0], type: 'f32', access: 'read'},
        {name: 'normalY', view: normals[1], type: 'f32', access: 'read'},
        {name: 'normalZ', view: normals[2], type: 'f32', access: 'read'},
        {name: 'rowSumX', view: rowSums[0], type: 'f32', access: 'read_write'},
        {name: 'rowSumY', view: rowSums[1], type: 'f32', access: 'read_write'},
        {name: 'rowSumZ', view: rowSums[2], type: 'f32', access: 'read_write'},
        {name: 'rowCounts', view: rowCounts, type: 'u32', access: 'read_write'},
        settingsBinding
      ],
      invocationCount: pixelCount,
      declarations: `const WIDTH: i32 = ${width};
const RADIUS: i32 = ${maximumRadius};
${RASTER_WGSL_HELPERS}`,
      body: `let column = i32(index) % WIDTH;
  let rowStart = i32(index) - column;
  ${radiusSource}
  var sum = vec3<f32>(0.0);
  var count = 0u;
  for (var sampleColumn = max(column - radius, 0); sampleColumn <= min(column + radius, WIDTH - 1); sampleColumn++) {
    let sampleIndex = u32(rowStart + sampleColumn);
    let normalXValue = normalX[normalXOffset + sampleIndex];
    if (isFiniteValue(normalXValue)) {
      sum += vec3<f32>(normalXValue, normalY[normalYOffset + sampleIndex], normalZ[normalZOffset + sampleIndex]);
      count += 1u;
    }
  }
  rowSumX[rowSumXOffset + index] = sum.x;
  rowSumY[rowSumYOffset + index] = sum.y;
  rowSumZ[rowSumZOffset + index] = sum.z;
  rowCounts[rowCountsOffset + index] = count;`
    });
    const columnNode = createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-window`,
      operation: 'GPUTerrainVectorRuggedness',
      variant: 'window-columns',
      bindings: [
        {name: 'normalX', view: normals[0], type: 'f32', access: 'read'},
        {name: 'rowSumX', view: rowSums[0], type: 'f32', access: 'read'},
        {name: 'rowSumY', view: rowSums[1], type: 'f32', access: 'read'},
        {name: 'rowSumZ', view: rowSums[2], type: 'f32', access: 'read'},
        {name: 'rowCounts', view: rowCounts, type: 'u32', access: 'read'},
        ...bindings.slice(3)
      ],
      invocationCount: pixelCount,
      declarations: `const WIDTH: i32 = ${width};
const HEIGHT: i32 = ${height};
const RADIUS: i32 = ${maximumRadius};
${RASTER_WGSL_HELPERS}`,
      body: `let row = i32(index) / WIDTH;
  let column = i32(index) % WIDTH;
  ${radiusSource}
  ${finishBody(`for (var sampleRow = max(row - radius, 0); sampleRow <= min(row + radius, HEIGHT - 1); sampleRow++) {
      let sampleIndex = u32(sampleRow * WIDTH + column);
      vectorSum += vec3<f32>(rowSumX[rowSumXOffset + sampleIndex], rowSumY[rowSumYOffset + sampleIndex], rowSumZ[rowSumZOffset + sampleIndex]);
      count += f32(rowCounts[rowCountsOffset + sampleIndex]);
    }`)}`
    });
    return [...source.nodes, normalNode, rowNode, columnNode];
  }
}
