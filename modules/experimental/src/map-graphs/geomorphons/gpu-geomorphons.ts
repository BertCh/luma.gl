// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validatePackedUint32View,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPURasterBand} from '../../gpu-raster/index';
import {createMapGraphKernelNode, type MapGraphKernelBinding} from '../map-graph-kernels';
import type {GPUMapGraphRecipe} from '../map-graph-types';
import {validateGraphViewsBelongToGraph} from '../map-graph-utils';
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
  writeTerrainGeomorphometryCellSettings,
  type TerrainGeomorphometryCellSettings
} from '../terrain-curvature/terrain-geomorphometry-utils';

/**
 * How the zenith and nadir lines of sight of one direction are reduced to a ternary pattern digit.
 *
 * - `'anglev1'`: GRASS default; when either absolute angle exceeds its threshold the larger
 *   absolute angle wins and an exact tie is flat.
 * - `'anglev2'`: each threshold applies to its own angle; an exact tie resolves to +1.
 * - `'anglev2-distance'`: like `'anglev2'`, but an exact tie goes to the farther sample.
 */
export type GPUGeomorphonComparison = 'anglev1' | 'anglev2' | 'anglev2-distance';

/** Geomorphon landform codes written to the `forms` output; 0 marks an invalid cell. */
export const GPU_GEOMORPHON_FORMS = {
  flat: 1,
  peak: 2,
  ridge: 3,
  shoulder: 4,
  spur: 5,
  slope: 6,
  hollow: 7,
  footslope: 8,
  valley: 9,
  pit: 10
} as const;

/** One geomorphon landform code, see {@link GPU_GEOMORPHON_FORMS}. */
export type GPUGeomorphonForm = (typeof GPU_GEOMORPHON_FORMS)[keyof typeof GPU_GEOMORPHON_FORMS];

/** Number of float32 values read from `GPUGeomorphonsProps.settings`. */
export const GPU_GEOMORPHONS_PARAMETER_LENGTH = 8;

/**
 * CPU-side description packed by {@link getGPUGeomorphonsParameterValues}.
 *
 * The cell-size model follows `cellSizeMode` of the recipe: projected metres (`'uniform'`),
 * equatorial Web Mercator metres (`'web-mercator'`), or geographic degrees (`'geographic'`) with
 * latitude-dependent row spacing derived from `northEdge` and `southEdge`.
 */
export type GPUGeomorphonsSettings = TerrainGeomorphometryCellSettings & {
  /** Flatness threshold in degrees, greater than 0 and below 90. Defaults to 1. */
  flatThresholdDegrees?: number;
  /**
   * Distance in ground metres beyond which the flatness threshold is lowered to the height
   * `tan(threshold) * flatDistance`. 0 disables the rule. Defaults to 0.
   */
  flatDistance?: number;
};

/**
 * Packs settings into the 8-float layout read by {@link GPUGeomorphons}:
 * `[cellSizeX, cellSizeY, zFactor, northEdge, southEdge, flatThresholdDegrees, flatDistance, 0]`.
 */
export function getGPUGeomorphonsParameterValues(
  settings: GPUGeomorphonsSettings,
  target: Float32Array = new Float32Array(GPU_GEOMORPHONS_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_GEOMORPHONS_PARAMETER_LENGTH) {
    throw new Error('Geomorphons settings target must hold 8 values');
  }
  writeTerrainGeomorphometryCellSettings(settings, target);
  target[5] = settings.flatThresholdDegrees ?? 1;
  target[6] = settings.flatDistance ?? 0;
  target[7] = 0;
  return target;
}

/**
 * Properties for {@link GPUGeomorphons}.
 *
 * Topology: grid size, elevation band, search and skip radius, comparison mode, which outputs
 * exist, `cellSizeMode`, and `rowDirection`. Per-frame: `settings` and elevation contents.
 * Cell sizes are interpreted by `cellSizeMode`: projected metres (`'uniform'`), equatorial Web
 * Mercator metres (`'web-mercator'`), or geographic degrees (`'geographic'`) with
 * latitude-dependent row spacing.
 */
export type GPUGeomorphonsProps = {
  /** Prefix for node and transient IDs. Defaults to `'geomorphons'`. */
  id?: string;
  /** Grid width in pixels. */
  width: number;
  /** Grid height in pixels. */
  height: number;
  /** Elevation band, buffer or texture; scale, offset, nodata, and validity are honored. */
  elevation: GPURasterBand;
  /** Per-frame settings with at least 8 float32 values, see {@link getGPUGeomorphonsParameterValues}. */
  settings: GraphDataView<'float32'>;
  /** Search radius in cells along the row axis; at least `skipRadius + 2`. */
  searchRadius: number;
  /** Cells next to the centre that are ignored along every line of sight. Defaults to 0. */
  skipRadius?: number;
  /** Zenith/nadir reduction. Defaults to `'anglev1'`. */
  comparison?: GPUGeomorphonComparison;
  /** Optional landform per pixel, see {@link GPU_GEOMORPHON_FORMS}; 0 where invalid. */
  forms?: GraphDataView<'uint32'>;
  /** Optional rotation- and mirror-invariant ternary class (498 values); 0 where invalid. */
  ternary?: GraphDataView<'uint32'>;
  /** Optional 6561-valued ternary pattern code, sum of `(digit + 1) * 3^direction`; 0 where invalid. */
  pattern?: GraphDataView<'uint32'>;
  /** Optional per-pixel 1 where the geomorphon is valid, else 0. */
  validity?: GraphDataView<'uint32'>;
  /** Cell size interpretation. Defaults to `'uniform'`. */
  cellSizeMode?: GPUTerrainCellSizeMode;
  /** Direction in which the row index increases. Defaults to `'south'` (north-up rasters). */
  rowDirection?: 'south' | 'north';
};

/**
 * Classifies each cell into one of ten landforms with the geomorphon method of Jasiewicz and
 * Stepinski (2013), following the behaviour of GRASS `r.geomorphon`.
 *
 * Eight lines of sight (NE, N, NW, W, SW, S, SE, E) are searched out to `searchRadius` rows of
 * ground distance. Zenith and nadir samples are compared by exact integer cross-multiplication
 * instead of arctangents, so ties on integer terrain are exact. Cells that are invalid, or within
 * `skipRadius + 1` cells of the raster border, receive form 0, ternary 0, pattern 0, and
 * validity 0. Invalid samples along a line of sight are ignored; a direction whose first
 * neighbour is invalid, or that has no valid sample, contributes pattern digit 0.
 *
 * `requiredHalo` equals `searchRadius` cells along the row metric; for non-square geographic
 * cells the east-west reach in columns can exceed it. The cell size is the ground size of the
 * centre row, so the search distance is `searchRadius * cellSizeY` ground metres.
 */
export class GPUGeomorphons implements GPUMapGraphRecipe {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Stable recipe name. */
  readonly recipe = 'geomorphons';
  /** Validated properties. */
  readonly props: GPUGeomorphonsProps;
  /** Receptive field in pixels (`GPURasterHaloStage` contract), equal to `searchRadius`. */
  readonly requiredHalo: number;

  constructor(props: GPUGeomorphonsProps) {
    this.id = props.id ?? this.recipe;
    this.props = props;
    const {id} = this;
    const pixelCount = validateTerrainGrid(id, props.width, props.height);
    const skipRadius = props.skipRadius ?? 0;
    if (!Number.isSafeInteger(props.searchRadius) || props.searchRadius < 1) {
      throw new Error(`${id} searchRadius must be a positive integer`);
    }
    if (!Number.isSafeInteger(skipRadius) || skipRadius < 0) {
      throw new Error(`${id} skipRadius must be a non-negative integer`);
    }
    if (props.searchRadius < skipRadius + 2) {
      throw new Error(`${id} searchRadius must be at least skipRadius + 2`);
    }
    if (!['anglev1', 'anglev2', 'anglev2-distance'].includes(props.comparison ?? 'anglev1')) {
      throw new Error(`${id} comparison must be anglev1, anglev2, or anglev2-distance`);
    }
    if (!props.forms && !props.ternary && !props.pattern && !props.validity) {
      throw new Error(`${id} requires at least one output`);
    }
    for (const [name, view] of [
      ['forms', props.forms],
      ['ternary', props.ternary],
      ['pattern', props.pattern],
      ['validity', props.validity]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} ${name}`);
        if (view.length !== pixelCount) {
          throw new Error(`${id} ${name} must contain one value per pixel`);
        }
      }
    }
    validateTerrainSettings(id, props.settings, GPU_GEOMORPHONS_PARAMETER_LENGTH);
    validateTerrainCellSizeMode(id, props.cellSizeMode);
    validateTerrainRowDirection(id, props.rowDirection);
    validateTerrainBuffersDistinct(
      id,
      [props.forms, props.ternary, props.pattern, props.validity],
      [...getTerrainBandViews(props.elevation), props.settings]
    );
    this.requiredHalo = props.searchRadius;
  }

  /** Returns the optional elevation canonicalization nodes and the fused geomorphon kernel. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    validateTerrainBandBelongsToGraph(id, graph, props.elevation, []);
    validateGraphViewsBelongToGraph(id, graph, [
      props.settings,
      props.forms,
      props.ternary,
      props.pattern,
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
    const validity = source.band.validity;
    if (!validity || source.band.storage.kind !== 'buffer') {
      throw new Error(`${id} could not canonicalize elevation`);
    }
    const bindings: MapGraphKernelBinding[] = [
      {name: 'elevation', view: source.band.storage.values, type: 'f32', access: 'read'},
      {name: 'elevationValidity', view: validity, type: 'u32', access: 'read'},
      {name: 'settings', view: props.settings, type: 'f32', access: 'read'}
    ];
    for (const [name, view] of [
      ['formValues', props.forms],
      ['ternaryValues', props.ternary],
      ['patternValues', props.pattern],
      ['validityValues', props.validity]
    ] as const) {
      if (view) {
        bindings.push({name, view, type: 'u32', access: 'read_write'});
      }
    }
    const comparison = props.comparison ?? 'anglev1';
    const skipRadius = props.skipRadius ?? 0;
    const cellSizeMode = props.cellSizeMode ?? 'uniform';
    const rowDirection = props.rowDirection ?? 'south';
    return [
      ...source.nodes,
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-classify`,
        operation: 'GPUGeomorphons',
        variant: `${comparison}-${cellSizeMode}-skip${skipRadius}-radius${props.searchRadius}`,
        bindings,
        invocationCount: props.width * props.height,
        declarations: getDeclarations(props, comparison, skipRadius, cellSizeMode, rowDirection),
        body: getBody(props)
      })
    ];
  }
}

function getDeclarations(
  props: GPUGeomorphonsProps,
  comparison: GPUGeomorphonComparison,
  skipRadius: number,
  cellSizeMode: GPUTerrainCellSizeMode,
  rowDirection: 'south' | 'north'
): string {
  return /* wgsl */ `
const WIDTH: u32 = ${props.width}u;
const HEIGHT: u32 = ${props.height}u;
const SEARCH_RADIUS: i32 = ${props.searchRadius};
const SKIP_RADIUS: i32 = ${skipRadius};
const COMPARISON: u32 = ${comparison === 'anglev1' ? 0 : comparison === 'anglev2' ? 1 : 2}u;
const ROW_NORTH_SIGN: i32 = ${rowDirection === 'south' ? '-1' : '1'};
const DEGREES_TO_RADIANS: f32 = 0.017453292519943295;
${TERRAIN_WGSL_HELPERS}
${getTerrainGroundCellSizeWGSL(cellSizeMode)}

// Directions in GRASS order: NE, N, NW, W, SW, S, SE, E.
var<private> DIRECTION_COLUMN = array<i32, 8>(1, 0, -1, -1, -1, 0, 1, 1);
var<private> DIRECTION_NORTH = array<i32, 8>(1, 1, 1, 0, -1, -1, -1, 0);
// Landform codes by minus count (row) and plus count (column); 0 marks impossible pairs.
var<private> FORM_TABLE = array<u32, 81>(
  1u, 1u, 1u, 8u, 8u, 9u, 9u, 9u, 10u,
  1u, 1u, 8u, 8u, 8u, 9u, 9u, 9u, 0u,
  1u, 4u, 6u, 6u, 7u, 7u, 9u, 0u, 0u,
  4u, 4u, 6u, 6u, 6u, 7u, 0u, 0u, 0u,
  4u, 4u, 5u, 6u, 6u, 0u, 0u, 0u, 0u,
  3u, 3u, 5u, 5u, 0u, 0u, 0u, 0u, 0u,
  3u, 3u, 3u, 0u, 0u, 0u, 0u, 0u, 0u,
  3u, 3u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
  2u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u
);

fn isElevationValid(column: i32, row: i32) -> bool {
  let linear = u32(row) * WIDTH + u32(column);
  return elevationValidity[elevationValidityOffset + linear] != 0u &&
    isFiniteValue(elevation[elevationOffset + linear]);
}

fn getElevation(column: i32, row: i32) -> f32 {
  return elevation[elevationOffset + u32(row) * WIDTH + u32(column)];
}

// Re-encodes the digits cyclically shifted by shift, optionally mirrored, in base 3.
fn encodeRotation(digits: array<u32, 8>, shift: u32, mirrored: bool) -> u32 {
  var code = 0u;
  var power = 1u;
  for (var position = 0u; position < 8u; position++) {
    let source = (position + 8u - shift) % 8u;
    let digit = select(digits[source], digits[7u - source], mirrored);
    code += digit * power;
    power *= 3u;
  }
  return code;
}

// Reduces one line of sight, given zenith/nadir heights and step counts, to a digit in {-1, 0, 1}.
fn getPatternDigit(
  zenithHeight: f32, zenithCount: i32, nadirHeight: f32, nadirCount: i32,
  tangent: f32, flatDistance: f32, step: f32
) -> i32 {
  let zenithDistance = f32(zenithCount) * step;
  let nadirDistance = f32(nadirCount) * step;
  let zenithReference = select(zenithDistance, flatDistance, flatDistance > 0.0 && flatDistance < zenithDistance);
  let nadirReference = select(nadirDistance, flatDistance, flatDistance > 0.0 && flatDistance < nadirDistance);
  let zenithAbsolute = abs(zenithHeight);
  let nadirAbsolute = abs(nadirHeight);
  let zenithOver = zenithAbsolute > tangent * zenithReference;
  let nadirOver = nadirAbsolute > tangent * nadirReference;
  // Absolute angles compare as |height| / count, cross-multiplied to stay division free.
  let zenithScaled = zenithAbsolute * f32(nadirCount);
  let nadirScaled = nadirAbsolute * f32(zenithCount);
  if (COMPARISON == 0u) {
    if (!(zenithOver || nadirOver)) { return 0; }
    if (nadirScaled < zenithScaled) { return 1; }
    if (nadirScaled > zenithScaled) { return -1; }
    return 0;
  }
  if (!zenithOver && !nadirOver) { return 0; }
  if (zenithOver && !nadirOver) { return 1; }
  if (nadirOver && !zenithOver) { return -1; }
  if (nadirScaled < zenithScaled) { return 1; }
  if (nadirScaled > zenithScaled) { return -1; }
  if (COMPARISON == 2u) {
    if (nadirCount < zenithCount) { return 1; }
    if (nadirCount > zenithCount) { return -1; }
  }
  return 1;
}`;
}

function getBody(props: GPUGeomorphonsProps): string {
  const writes = [
    props.forms ? 'formValues[formValuesOffset + index] = outputForm;' : '',
    props.ternary ? 'ternaryValues[ternaryValuesOffset + index] = outputTernary;' : '',
    props.pattern ? 'patternValues[patternValuesOffset + index] = outputPattern;' : '',
    props.validity ? 'validityValues[validityValuesOffset + index] = select(0u, 1u, isValid);' : ''
  ].join('\n  ');
  return /* wgsl */ `
  let row = i32(index / WIDTH);
  let column = i32(index % WIDTH);
  let groundCell = getGroundCellSize(u32(row));
  let zFactor = settings[settingsOffset + 2u];
  let flatThreshold = settings[settingsOffset + 5u];
  let flatDistance = settings[settingsOffset + 6u];
  let settingsValid = isFiniteValue(groundCell.x) && isFiniteValue(groundCell.y) &&
    groundCell.x > 0.0 && groundCell.y > 0.0 && isFiniteValue(zFactor) &&
    isFiniteValue(flatThreshold) && flatThreshold > 0.0 && flatThreshold < 90.0 &&
    isFiniteValue(flatDistance) && flatDistance >= 0.0;
  let margin = SKIP_RADIUS + 1;
  let insideBand = row >= margin && row < i32(HEIGHT) - margin &&
    column >= margin && column < i32(WIDTH) - margin;
  let isValid = settingsValid && insideBand && isElevationValid(column, row);
  var outputForm = 0u;
  var outputTernary = 0u;
  var outputPattern = 0u;
  if (isValid) {
    let centerElevation = getElevation(column, row);
    let tangent = tan(flatThreshold * DEGREES_TO_RADIANS);
    // The strict search limit is shrunk by 1e-5 so a sample exactly at the limit is excluded even
    // when f32 cell sizes round (for example cos(0) or equal east-west and north-south sizes).
    let searchDistance = f32(SEARCH_RADIUS) * groundCell.y * 0.99999;
    var digits: array<u32, 8>;
    var minusCount = 0u;
    var plusCount = 0u;
    for (var direction = 0u; direction < 8u; direction++) {
      digits[direction] = 1u;
      let columnStep = DIRECTION_COLUMN[direction];
      let northStep = DIRECTION_NORTH[direction];
      let rowStep = northStep * ROW_NORTH_SIGN;
      if (!isElevationValid(column + columnStep, row + rowStep)) { continue; }
      let eastMetres = f32(columnStep) * groundCell.x;
      let northMetres = f32(northStep) * groundCell.y;
      // Axis directions use the exact cell size so the search limit has no square-root rounding.
      var step = sqrt(eastMetres * eastMetres + northMetres * northMetres);
      if (columnStep == 0) { step = groundCell.y; }
      if (northStep == 0) { step = groundCell.x; }
      var zenithHeight = 0.0;
      var zenithCount = 0;
      var nadirHeight = 0.0;
      var nadirCount = 0;
      for (var count = SKIP_RADIUS + 1; count <= i32(max(WIDTH, HEIGHT)); count++) {
        if (!(f32(count) * step < searchDistance)) { break; }
        let sampleColumn = column + count * columnStep;
        let sampleRow = row + count * rowStep;
        if (sampleColumn < 0 || sampleColumn >= i32(WIDTH) || sampleRow < 0 || sampleRow >= i32(HEIGHT)) { break; }
        if (!isElevationValid(sampleColumn, sampleRow)) { continue; }
        let height = zFactor * (getElevation(sampleColumn, sampleRow) - centerElevation);
        if (zenithCount == 0 || height * f32(zenithCount) > zenithHeight * f32(count)) {
          zenithHeight = height;
          zenithCount = count;
        }
        if (nadirCount == 0 || height * f32(nadirCount) < nadirHeight * f32(count)) {
          nadirHeight = height;
          nadirCount = count;
        }
      }
      if (zenithCount == 0) { continue; }
      let digit = getPatternDigit(
        zenithHeight, zenithCount, nadirHeight, nadirCount, tangent, flatDistance, step
      );
      digits[direction] = u32(digit + 1);
      if (digit > 0) { plusCount++; }
      if (digit < 0) { minusCount++; }
    }
    var code = 0u;
    var power = 1u;
    var rotated = 0xffffffffu;
    for (var position = 0u; position < 8u; position++) {
      code += digits[position] * power;
      power *= 3u;
      rotated = min(rotated, encodeRotation(digits, position, false));
      rotated = min(rotated, encodeRotation(digits, position, true));
    }
    outputPattern = code;
    outputTernary = rotated;
    outputForm = FORM_TABLE[minusCount * 9u + plusCount];
  }
  ${writes}`;
}
