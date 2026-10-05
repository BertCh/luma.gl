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
import {createMapGraphKernelNode, type MapGraphKernelBinding} from '../map-graph-kernels';
import type {GPUMapGraphRecipe} from '../map-graph-types';
import {validateGraphViewsBelongToGraph} from '../map-graph-utils';
import {
  getTerrainBandViews,
  getTerrainElevationNodes,
  TERRAIN_WGSL_HELPERS,
  validateTerrainBandBelongsToGraph,
  validateTerrainBuffersDistinct,
  validateTerrainGrid
} from '../terrain-analysis/terrain-analysis-utils';

/** Terrain ruggedness index definition. */
export type GPUTerrainRuggednessAlgorithm = 'riley' | 'wilson';

/** Treatment of the outermost one-cell ring of the raster. */
export type GPUTerrainRuggednessEdgeMode = 'nodata' | 'extrapolate';

/**
 * Properties for {@link GPUTerrainRuggedness}.
 *
 * Every output is expressed in elevation units. The operators compare elevations only, so there
 * is no cell-size model: the result is independent of projected metres (`'uniform'`),
 * `'web-mercator'`, or `'geographic'` degrees and needs no settings buffer or z factor.
 */
export type GPUTerrainRuggednessProps = {
  /** Prefix for node and transient IDs. Defaults to `'terrain-ruggedness'`. */
  id?: string;
  /** Grid width in pixels. */
  width: number;
  /** Grid height in pixels. */
  height: number;
  /** Elevation band, buffer or texture; scale, offset, nodata, and validity are honored. */
  elevation: GPURasterBand;
  /** Optional topographic position index: centre minus the mean of its 8 neighbours. */
  topographicPositionIndex?: GraphDataView<'float32'>;
  /** Optional terrain ruggedness index, see `terrainRuggednessAlgorithm`. */
  terrainRuggednessIndex?: GraphDataView<'float32'>;
  /** Optional roughness: maximum minus minimum elevation of the 3x3 window. */
  roughness?: GraphDataView<'float32'>;
  /** Optional per-pixel 1 where every output is valid, else 0. */
  validity?: GraphDataView<'uint32'>;
  /**
   * `'riley'` (default, gdaldem 3.3+): square root of the summed squared neighbour differences.
   * `'wilson'`: mean absolute neighbour difference.
   */
  terrainRuggednessAlgorithm?: GPUTerrainRuggednessAlgorithm;
  /**
   * `'nodata'` (default, gdaldem default): the outer ring and any window with an invalid sample are
   * invalid. `'extrapolate'` (gdaldem `-compute_edges`): outside samples are linearly extrapolated
   * and invalid neighbours are replaced by the centre; needs width and height of at least 2.
   */
  edgeMode?: GPUTerrainRuggednessEdgeMode;
};

/**
 * Computes gdaldem-compatible topographic position index (TPI), terrain ruggedness index (TRI),
 * and roughness over a 3x3 window in one fused kernel.
 *
 * Outputs use elevation units (no cell size, no z factor). Invalid pixels receive NaN and
 * validity 0. The recipe satisfies the `GPURasterHaloStage` contract with a one-pixel halo.
 */
export class GPUTerrainRuggedness implements GPUMapGraphRecipe {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Stable recipe name. */
  readonly recipe = 'terrain-ruggedness';
  /** Validated properties. */
  readonly props: GPUTerrainRuggednessProps;
  /** Receptive field in pixels (`GPURasterHaloStage` contract). */
  readonly requiredHalo = 1;

  constructor(props: GPUTerrainRuggednessProps) {
    this.id = props.id ?? this.recipe;
    this.props = props;
    const {id} = this;
    const pixelCount = validateTerrainGrid(id, props.width, props.height);
    if (
      !props.topographicPositionIndex &&
      !props.terrainRuggednessIndex &&
      !props.roughness &&
      !props.validity
    ) {
      throw new Error(`${id} requires at least one output`);
    }
    for (const [name, view] of [
      ['topographicPositionIndex', props.topographicPositionIndex],
      ['terrainRuggednessIndex', props.terrainRuggednessIndex],
      ['roughness', props.roughness]
    ] as const) {
      if (view) {
        validatePackedView(view, ['float32'], `${id} ${name}`);
        if (view.length !== pixelCount) {
          throw new Error(`${id} ${name} must contain one value per pixel`);
        }
      }
    }
    if (props.validity) {
      validatePackedUint32View(props.validity, `${id} validity`);
      if (props.validity.length !== pixelCount) {
        throw new Error(`${id} validity must contain one value per pixel`);
      }
    }
    if (!['riley', 'wilson'].includes(props.terrainRuggednessAlgorithm ?? 'riley')) {
      throw new Error(`${id} terrainRuggednessAlgorithm must be riley or wilson`);
    }
    const edgeMode = props.edgeMode ?? 'nodata';
    if (!['nodata', 'extrapolate'].includes(edgeMode)) {
      throw new Error(`${id} edgeMode must be nodata or extrapolate`);
    }
    if (edgeMode === 'extrapolate' && (props.width < 2 || props.height < 2)) {
      throw new Error(`${id} edgeMode extrapolate requires width and height of at least 2`);
    }
    validateTerrainBuffersDistinct(
      id,
      [
        props.topographicPositionIndex,
        props.terrainRuggednessIndex,
        props.roughness,
        props.validity
      ],
      getTerrainBandViews(props.elevation)
    );
  }

  /** Returns the optional elevation canonicalization node and the window kernel. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    validateTerrainBandBelongsToGraph(id, graph, props.elevation, []);
    validateGraphViewsBelongToGraph(id, graph, [
      props.topographicPositionIndex,
      props.terrainRuggednessIndex,
      props.roughness,
      props.validity
    ]);
    const source = getTerrainElevationNodes(
      graph,
      id,
      props.elevation,
      props.width,
      props.height,
      true
    );
    const bindings: MapGraphKernelBinding[] = [
      {name: 'elevation', view: source.band.storage.values, type: 'f32', access: 'read'},
      {
        name: 'elevationValidity',
        view: source.band.validity as GraphDataView<'uint32'>,
        type: 'u32',
        access: 'read'
      }
    ];
    const outputs: [string, GraphDataView | undefined, 'f32' | 'u32'][] = [
      ['tpiValues', props.topographicPositionIndex, 'f32'],
      ['rugValues', props.terrainRuggednessIndex, 'f32'],
      ['roughnessValues', props.roughness, 'f32'],
      ['validityValues', props.validity, 'u32']
    ];
    for (const [name, view, type] of outputs) {
      if (view) {
        bindings.push({name, view, type, access: 'read_write'});
      }
    }
    const edgeMode = props.edgeMode ?? 'nodata';
    const algorithm = props.terrainRuggednessAlgorithm ?? 'riley';
    const node = createMapGraphKernelNode<Parameters>(graph, {
      id: `${id}-window`,
      operation: 'GPUTerrainRuggedness',
      variant: `${edgeMode}-${algorithm}`,
      bindings,
      invocationCount: props.width * props.height,
      declarations: getDeclarations(props.width, props.height, edgeMode),
      body: getBody(props, algorithm)
    });
    return [...source.nodes, node];
  }
}

function getDeclarations(
  width: number,
  height: number,
  edgeMode: GPUTerrainRuggednessEdgeMode
): string {
  const header = `const WIDTH: i32 = ${width};
const HEIGHT: i32 = ${height};
${TERRAIN_WGSL_HELPERS}
fn isSampleValid(row: i32, column: i32) -> bool {
  let sampleIndex = u32(row * WIDTH + column);
  return elevationValidity[elevationValidityOffset + sampleIndex] != 0u &&
    isFiniteValue(elevation[elevationOffset + sampleIndex]);
}
fn getSample(row: i32, column: i32) -> f32 {
  return elevation[elevationOffset + u32(row * WIDTH + column)];
}`;
  if (edgeMode === 'nodata') {
    return header;
  }
  // gdaldem -compute_edges: INTERPOL(a, b) = 2a - b is invalid when either operand is invalid.
  // Top and bottom rows clamp horizontal neighbours; other rows extrapolate left and right.
  return `${header}
struct WindowSample { value: f32, valid: bool };
fn getExtrapolated(
  rowA: i32, columnA: i32, rowB: i32, columnB: i32
) -> WindowSample {
  let valid = isSampleValid(rowA, columnA) && isSampleValid(rowB, columnB);
  return WindowSample(2.0 * getSample(rowA, columnA) - getSample(rowB, columnB), valid);
}
fn getWindowSample(row: i32, column: i32, deltaRow: i32, deltaColumn: i32) -> WindowSample {
  if (row == 0 || row == HEIGHT - 1) {
    let sampleColumn = clamp(column + deltaColumn, 0, WIDTH - 1);
    if (row == 0 && deltaRow < 0) {
      return getExtrapolated(0, sampleColumn, 1, sampleColumn);
    }
    if (row == HEIGHT - 1 && deltaRow > 0) {
      return getExtrapolated(HEIGHT - 1, sampleColumn, HEIGHT - 2, sampleColumn);
    }
    let sampleRow = row + deltaRow;
    return WindowSample(getSample(sampleRow, sampleColumn), isSampleValid(sampleRow, sampleColumn));
  }
  let sampleRow = row + deltaRow;
  if (column == 0 && deltaColumn < 0) {
    return getExtrapolated(sampleRow, 0, sampleRow, 1);
  }
  if (column == WIDTH - 1 && deltaColumn > 0) {
    return getExtrapolated(sampleRow, WIDTH - 1, sampleRow, WIDTH - 2);
  }
  let sampleColumn = column + deltaColumn;
  return WindowSample(getSample(sampleRow, sampleColumn), isSampleValid(sampleRow, sampleColumn));
}`;
}

function getBody(
  props: GPUTerrainRuggednessProps,
  algorithm: GPUTerrainRuggednessAlgorithm
): string {
  const edgeMode = props.edgeMode ?? 'nodata';
  const windowSource =
    edgeMode === 'nodata'
      ? `var isValid = row > 0 && row < HEIGHT - 1 && column > 0 && column < WIDTH - 1;
  if (isValid) {
    for (var k = 0; k < 9; k++) {
      let sampleRow = row + k / 3 - 1;
      let sampleColumn = column + k % 3 - 1;
      isValid = isValid && isSampleValid(sampleRow, sampleColumn);
      if (isValid) {
        windowValues[k] = getSample(sampleRow, sampleColumn);
      }
    }
  }`
      : `var isValid = isSampleValid(row, column);
  if (isValid) {
    for (var k = 0; k < 9; k++) {
      let windowSample = getWindowSample(row, column, k / 3 - 1, k % 3 - 1);
      // gdaldem replaces invalid window samples by the centre when computing edges.
      windowValues[k] = select(centerValue, windowSample.value, windowSample.valid);
    }
  }`;
  return `let row = i32(index) / WIDTH;
  let column = i32(index) % WIDTH;
  var windowValues: array<f32, 9>;
  let centerValue = getSample(row, column);
  ${windowSource}
  var differenceSum = 0.0;
  var absoluteSum = 0.0;
  var squareSum = 0.0;
  var minimum = windowValues[0];
  var maximum = windowValues[0];
  for (var k = 0; k < 9; k++) {
    // Differences against the centre are formed first so they stay exact near large offsets.
    let difference = windowValues[k] - centerValue;
    differenceSum += difference;
    absoluteSum += abs(difference);
    squareSum += difference * difference;
    minimum = min(minimum, windowValues[k]);
    maximum = max(maximum, windowValues[k]);
  }
  let topographicPosition = -differenceSum / 8.0;
  ${
    algorithm === 'riley'
      ? 'let ruggedness = sqrt(squareSum);'
      : 'let ruggedness = absoluteSum / 8.0;'
  }
  let roughness = maximum - minimum;
  isValid = isValid && isFiniteValue(topographicPosition) && isFiniteValue(ruggedness) &&
    isFiniteValue(roughness);
  let invalidValue = bitcast<f32>(0x7fc00000u | (index & 0u));
  ${props.topographicPositionIndex ? 'tpiValues[tpiValuesOffset + index] = select(invalidValue, topographicPosition, isValid);' : ''}
  ${props.terrainRuggednessIndex ? 'rugValues[rugValuesOffset + index] = select(invalidValue, ruggedness, isValid);' : ''}
  ${props.roughness ? 'roughnessValues[roughnessValuesOffset + index] = select(invalidValue, roughness, isValid);' : ''}
  ${props.validity ? 'validityValues[validityValuesOffset + index] = select(0u, 1u, isValid);' : ''}`;
}
