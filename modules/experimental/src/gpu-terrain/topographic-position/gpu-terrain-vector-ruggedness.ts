// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPURasterBand, GPURasterBorderMode} from '../../gpu-raster/index';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {validateGraphViewsBelongToGraph} from '../../utils/gpu-contributor-utils';
import type {GPUTerrainCellSizeMode} from '../terrain-analysis/gpu-terrain-derivatives';
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
  getTerrainGroundCellSizeWGSL,
  validateTerrainCellSizeMode,
  validateTerrainRowDirection,
  writeTerrainGeomorphometryCellSettings
} from '../terrain-curvature/terrain-geomorphometry-utils';

/** Number of float32 values read from `GPUTerrainVectorRuggednessProps.settings`. */
export const GPU_TERRAIN_VECTOR_RUGGEDNESS_PARAMETER_LENGTH = 8;

/**
 * CPU-side description packed by {@link getGPUTerrainVectorRuggednessParameterValues}.
 *
 * The cell-size model depends on `GPUTerrainVectorRuggednessProps.cellSizeMode`: projected metres
 * for `'uniform'`, equatorial Web Mercator metres for `'web-mercator'`, or degrees for
 * `'geographic'` (x spacing scales with cos(latitude) of each row).
 */
export type GPUTerrainVectorRuggednessSettings = {
  /** `[x, y]` cell size: metres (uniform), equatorial Web Mercator metres, or degrees (geographic). */
  cellSize: readonly [number, number];
  /** Elevation multiplier converting elevation units into ground units. Defaults to 1. */
  zFactor?: number;
  /** Top edge of row 0: normalized Web Mercator y in `[0, 1]` or latitude degrees. Defaults to 0. */
  northEdge?: number;
  /** Bottom edge of the last row, same units as `northEdge`. Defaults to 0. */
  southEdge?: number;
};

/**
 * Packs settings into the 8-float layout read by {@link GPUTerrainVectorRuggedness}:
 * `[cellSizeX, cellSizeY, zFactor, northEdge, southEdge, 0, 0, 0]`.
 */
export function getGPUTerrainVectorRuggednessParameterValues(
  settings: GPUTerrainVectorRuggednessSettings,
  target: Float32Array = new Float32Array(GPU_TERRAIN_VECTOR_RUGGEDNESS_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_TERRAIN_VECTOR_RUGGEDNESS_PARAMETER_LENGTH) {
    throw new Error('Terrain vector ruggedness settings target must hold 8 values');
  }
  target.fill(0, 0, GPU_TERRAIN_VECTOR_RUGGEDNESS_PARAMETER_LENGTH);
  writeTerrainGeomorphometryCellSettings(settings, target);
  return target;
}

/**
 * Properties for {@link GPUTerrainVectorRuggedness}.
 *
 * Cell-size model: `cellSizeMode` selects projected metres (`'uniform'`), `'web-mercator'`, or
 * `'geographic'` degrees with latitude-dependent row spacing; see
 * {@link GPUTerrainVectorRuggednessSettings}.
 */
export type GPUTerrainVectorRuggednessProps = {
  /** Prefix for node and transient IDs. Defaults to `'terrain-vector-ruggedness'`. */
  id?: string;
  /** Grid width in pixels. */
  width: number;
  /** Grid height in pixels. */
  height: number;
  /** Elevation band, buffer or texture; scale, offset, nodata, and validity are honored. */
  elevation: GPURasterBand;
  /** Per-frame settings with at least 8 float32 values, see {@link getGPUTerrainVectorRuggednessParameterValues}. */
  settings: GraphDataView<'float32'>;
  /** Window half-width in pixels: the window is `(2 * radius + 1)` square. Integer >= 1, default 1. */
  radius?: number;
  /** Optional vector ruggedness measure per pixel in `[0, 1]`; NaN where invalid. */
  vectorRuggedness?: GraphDataView<'float32'>;
  /** Optional per-pixel 1 where the output is valid, else 0. */
  validity?: GraphDataView<'uint32'>;
  /** Cell size interpretation. Defaults to `'uniform'`. */
  cellSizeMode?: GPUTerrainCellSizeMode;
  /** Direction in which the row index increases. Defaults to `'south'` (north-up rasters). */
  rowDirection?: 'south' | 'north';
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
 * normal) and returns `1 - |sum| / count`, clamped to `[0, 1]`. The window sum is a direct loop,
 * so cost grows as O(radius^2) per pixel; it is intended for radii up to about 8. An invalid
 * centre normal gives NaN and validity 0. Satisfies the `GPURasterHaloStage` contract with
 * `requiredHalo = radius + 1`.
 */
export class GPUTerrainVectorRuggedness implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUTerrainVectorRuggednessProps;
  /** Receptive field in pixels (`GPURasterHaloStage` contract): `radius + 1`. */
  readonly requiredHalo: number;

  constructor(props: GPUTerrainVectorRuggednessProps) {
    this.id = props.id ?? 'terrain-vector-ruggedness';
    this.props = props;
    const {id} = this;
    const radius = props.radius ?? 1;
    if (!Number.isSafeInteger(radius) || radius < 1) {
      throw new Error(`${id} radius must be an integer of at least 1`);
    }
    this.requiredHalo = radius + 1;
    const pixelCount = validateTerrainGrid(id, props.width, props.height);
    if (!props.vectorRuggedness && !props.validity) {
      throw new Error(`${id} requires at least one output`);
    }
    if (props.vectorRuggedness) {
      validatePackedView(props.vectorRuggedness, ['float32'], `${id} vectorRuggedness`);
      if (props.vectorRuggedness.length !== pixelCount) {
        throw new Error(`${id} vectorRuggedness must contain one value per pixel`);
      }
    }
    if (props.validity) {
      validatePackedUint32View(props.validity, `${id} validity`);
      if (props.validity.length !== pixelCount) {
        throw new Error(`${id} validity must contain one value per pixel`);
      }
    }
    validateTerrainSettings(id, props.settings, GPU_TERRAIN_VECTOR_RUGGEDNESS_PARAMETER_LENGTH);
    validateTerrainCellSizeMode(id, props.cellSizeMode);
    validateTerrainRowDirection(id, props.rowDirection);
    if (!['clamp', 'nodata'].includes(props.borderMode ?? 'clamp')) {
      throw new Error(`${id} borderMode must be clamp or nodata`);
    }
    validateTerrainBuffersDistinct(
      id,
      [props.vectorRuggedness, props.validity],
      [...getTerrainBandViews(props.elevation), props.settings]
    );
  }

  /** Returns the optional elevation canonicalization node, the normal kernel and the window kernel. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {width, height} = props;
    validateTerrainBandBelongsToGraph(id, graph, props.elevation, []);
    validateGraphViewsBelongToGraph(id, graph, [
      props.settings,
      props.vectorRuggedness,
      props.validity
    ]);
    const pixelCount = width * height;
    const source = getTerrainElevationNodes(graph, id, props.elevation, width, height, true);
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
        {name: 'settings', view: props.settings, type: 'f32', access: 'read'},
        {name: 'normalX', view: normals[0], type: 'f32', access: 'read_write'},
        {name: 'normalY', view: normals[1], type: 'f32', access: 'read_write'},
        {name: 'normalZ', view: normals[2], type: 'f32', access: 'read_write'}
      ],
      invocationCount: pixelCount,
      declarations: `const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;
const ROW_NORTH_SIGN: f32 = ${(props.rowDirection ?? 'south') === 'south' ? '-1.0' : '1.0'};
${TERRAIN_WGSL_HELPERS}
${getTerrainGroundCellSizeWGSL(props.cellSizeMode ?? 'uniform')}`,
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

    const radius = props.radius ?? 1;
    const bindings: WGSLKernelBinding[] = [
      {name: 'normalX', view: normals[0], type: 'f32', access: 'read'},
      {name: 'normalY', view: normals[1], type: 'f32', access: 'read'},
      {name: 'normalZ', view: normals[2], type: 'f32', access: 'read'}
    ];
    if (props.vectorRuggedness) {
      bindings.push({
        name: 'vrmValues',
        view: props.vectorRuggedness,
        type: 'f32',
        access: 'read_write'
      });
    }
    if (props.validity) {
      bindings.push({
        name: 'validityValues',
        view: props.validity,
        type: 'u32',
        access: 'read_write'
      });
    }
    const windowNode = createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-window`,
      operation: 'GPUTerrainVectorRuggedness',
      variant: `window-radius-${radius}`,
      bindings,
      invocationCount: pixelCount,
      declarations: `const WIDTH: i32 = ${width};
const HEIGHT: i32 = ${height};
const RADIUS: i32 = ${radius};
${TERRAIN_WGSL_HELPERS}`,
      body: `let row = i32(index) / WIDTH;
  let column = i32(index) % WIDTH;
  let invalidValue = bitcast<f32>(0x7fc00000u | (index & 0u));
  let isValid = isFiniteValue(normalX[normalXOffset + index]);
  var vectorSum = vec3<f32>(0.0);
  var count = 0.0;
  if (isValid) {
    for (var sampleRow = max(row - RADIUS, 0); sampleRow <= min(row + RADIUS, HEIGHT - 1); sampleRow++) {
      for (var sampleColumn = max(column - RADIUS, 0); sampleColumn <= min(column + RADIUS, WIDTH - 1); sampleColumn++) {
        let sampleIndex = u32(sampleRow * WIDTH + sampleColumn);
        let normalXValue = normalX[normalXOffset + sampleIndex];
        if (isFiniteValue(normalXValue)) {
          vectorSum += vec3<f32>(normalXValue, normalY[normalYOffset + sampleIndex], normalZ[normalZOffset + sampleIndex]);
          count += 1.0;
        }
      }
    }
  }
  // count >= 1 whenever the centre normal is valid.
  let ruggedness = clamp(1.0 - length(vectorSum) / max(count, 1.0), 0.0, 1.0);
  ${props.vectorRuggedness ? 'vrmValues[vrmValuesOffset + index] = select(invalidValue, ruggedness, isValid);' : ''}
  ${props.validity ? 'validityValues[validityValuesOffset + index] = select(0u, 1u, isValid);' : ''}`
    });
    return [...source.nodes, normalNode, windowNode];
  }
}
