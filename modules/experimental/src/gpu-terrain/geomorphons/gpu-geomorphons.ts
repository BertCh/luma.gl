// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPURasterBand, GPURasterMetadata} from '../../gpu-raster/index';
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
  RASTER_DEGREES_TO_RADIANS_WGSL,
  validateRasterCellSizeMode,
  validateRasterRowDirection
} from '../../analysis-core/raster-grid';

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

/** Number of float32 values read from `GPUGeomorphonsProps.parameters`. */
export const GPU_GEOMORPHONS_PARAMETER_LENGTH = 8;

/**
 * CPU-side description packed by {@link getGPUGeomorphonsParameterValues}.
 *
 * The cell-size model follows `cellSizeMode` of the contributor: projected metres (`'uniform'`),
 * equatorial Web Mercator metres (`'web-mercator'`), or geographic degrees (`'geographic'`) with
 * latitude-dependent row spacing derived from the row edges of the grid.
 */
export type GPUGeomorphonsSettings = TerrainGeomorphometryCellSettings & {
  /** Flatness threshold in degrees, greater than 0 and below 90. Defaults to 1. */
  flatThresholdDegrees?: number;
  /**
   * Distance in ground metres beyond which the flatness threshold is lowered to the height
   * `tan(threshold) * flatDistance`. 0 disables the rule. Defaults to 0. As in GRASS
   * `r.geomorphon` (`main.c`: `flat_distance <= skip_distance || flat_distance >=
   * search_distance` sets it to 0), a distance at or inside the skip distance
   * (`skipRadius` cells) or at or beyond the search distance (`searchRadius` cells) disables the
   * rule; both are measured with the row's north-south ground cell size.
   */
  flatDistance?: number;
};

/**
 * Packs settings into the 8-float layout read by {@link GPUGeomorphons}:
 * `[cellSizeX, cellSizeY, zFactor, northEdge, southEdge, flatThresholdDegrees, flatDistance, 0]`,
 * with the cell size and the row-edge world y taken from `grid`.
 */
export function getGPUGeomorphonsParameterValues(
  settings: GPUGeomorphonsSettings,
  grid: GPURasterMetadata,
  target: Float32Array = new Float32Array(GPU_GEOMORPHONS_PARAMETER_LENGTH)
): Float32Array {
  validateParameterTarget(
    'getGPUGeomorphonsParameterValues',
    target,
    GPU_GEOMORPHONS_PARAMETER_LENGTH
  );
  writeTerrainGeomorphometryCellSettings(settings, grid, target);
  target[5] = settings.flatThresholdDegrees ?? 1;
  target[6] = settings.flatDistance ?? 0;
  target[7] = 0;
  return target;
}

/** Caller-owned outputs of {@link GPUGeomorphons}; at least one is required. */
export type GPUGeomorphonsOutput = {
  /** Optional landform per pixel, see {@link GPU_GEOMORPHON_FORMS}; 0 where invalid. */
  forms?: GraphDataView<'uint32'>;
  /** Optional rotation- and mirror-invariant ternary class (498 values); 0 where invalid. */
  ternary?: GraphDataView<'uint32'>;
  /** Optional 6561-valued ternary pattern code, sum of `(digit + 1) * 3^direction`; 0 where invalid. */
  pattern?: GraphDataView<'uint32'>;
  /** Optional per-pixel 1 where the geomorphon is valid, else 0. */
  outputValidity?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUGeomorphons}.
 *
 * Topology: grid size, elevation band, search and skip radius, comparison mode, which outputs
 * exist, `cellSizeMode`, and `rowDirection`. Per-frame: `parameters` (including the grid cell
 * size and latitude band) and elevation contents.
 * Cell sizes are interpreted by `cellSizeMode`: projected metres (`'uniform'`), equatorial Web
 * Mercator metres (`'web-mercator'`), or geographic degrees (`'geographic'`) with
 * latitude-dependent row spacing.
 */
export type GPUGeomorphonsProps = {
  /** Prefix for node and transient IDs. Defaults to `'geomorphons'`. */
  id?: string;
  /** Raster grid: its width and height size the kernels. */
  grid: GPURasterMetadata;
  /** Elevation band, buffer or texture; scale, offset, nodata, and validity are honored. */
  elevation: GPURasterBand;
  /** Per-frame parameters with at least 8 float32 values, see {@link getGPUGeomorphonsParameterValues}. */
  parameters: GraphDataView<'float32'>;
  /** Search radius in cells along the row axis; at least `skipRadius + 2`. */
  searchRadius: number;
  /** Cells next to the centre that are ignored along every line of sight. Defaults to 0. */
  skipRadius?: number;
  /** Zenith/nadir reduction. Defaults to `'anglev1'`. */
  comparison?: GPUGeomorphonComparison;
  /** Caller-owned outputs. */
  output: GPUGeomorphonsOutput;
  /** Cell size interpretation. Defaults to `'uniform'`. */
  cellSizeMode?: GPURasterCellSizeMode;
  /** Direction in which the row index increases. Defaults to `'south'` (north-up rasters). */
  rowDirection?: GPURasterRowDirection;
};

/**
 * Classifies each cell into one of ten landforms with the geomorphon method of Jasiewicz and
 * Stepinski (2013), using the GRASS `r.geomorphon` angle-comparison variants.
 * These variants compare absolute extrema; they are not the original paper's signed
 * zenith/nadir angle-difference test. In particular, `anglev1` calls a constant-slope ray flat
 * when its absolute extrema tie, even when that slope exceeds the flatness threshold.
 *
 * Eight lines of sight (NE, N, NW, W, SW, S, SE, E) are searched out to `searchRadius` rows of
 * ground distance. Zenith and nadir samples are compared by f32 cross-multiplication
 * instead of arctangents. A tied extremum keeps the nearest sample. Ties are exact when the
 * height differences and their products with integer step counts are exactly representable
 * in f32; arbitrary floating terrain has ordinary f32 rounding. GRASS instead computes
 * distances from map coordinates and compares double-precision arctangents. Its coordinate
 * serialization and libm rounding can break mathematical ties, so bitwise GRASS classification
 * parity requires matching the tie convention as well as the search and skip radii.
 * Cells that are invalid, or within
 * `skipRadius + 1` cells of the raster border, receive form 0, ternary 0, pattern 0, and
 * validity 0. Invalid samples along a line of sight are ignored; a direction whose first
 * neighbour is invalid, or that has no valid sample, contributes pattern digit 0. GRASS keeps its
 * initial -90/+90 degree zenith/nadir angles for a line of sight with no sample after the skip,
 * which `anglev2` reads as a tie and turns into +1; here no sample means no relief evidence, so the
 * digit stays 0 in every comparison mode.
 *
 * `requiredHalo` equals `searchRadius` cells along the row metric; for non-square geographic
 * cells the east-west reach in columns can exceed it. The cell size is the ground size of the
 * centre row, so the search distance is `searchRadius * cellSizeY` ground metres.
 */
export class GPUGeomorphons implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUGeomorphonsProps;
  /** Receptive field in pixels (`GPURasterHaloStage` contract), equal to `searchRadius`. */
  readonly requiredHalo: number;

  constructor(props: GPUGeomorphonsProps) {
    this.id = props.id ?? 'geomorphons';
    this.props = props;
    const {id} = this;
    const {output} = props;
    const pixelCount = validateRasterGrid(id, props.grid);
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
    if (!output.forms && !output.ternary && !output.pattern && !output.outputValidity) {
      throw new Error(`${id} requires at least one output`);
    }
    for (const [name, view] of [
      ['forms', output.forms],
      ['ternary', output.ternary],
      ['pattern', output.pattern],
      ['outputValidity', output.outputValidity]
    ] as const) {
      if (view) {
        validatePackedView(view, ['uint32'], `${id} output.${name}`);
        if (view.length !== pixelCount) {
          throw new Error(`${id} output.${name} must contain one value per pixel`);
        }
      }
    }
    validateRasterParameters(id, props.parameters, GPU_GEOMORPHONS_PARAMETER_LENGTH);
    validateRasterCellSizeMode(id, props.cellSizeMode);
    validateRasterGridAxisAligned(id, props.grid, props.cellSizeMode);
    validateRasterRowDirection(id, props.rowDirection);
    validateRasterBuffersDistinct(
      id,
      [output.forms, output.ternary, output.pattern, output.outputValidity],
      [...getRasterBandViews(props.elevation), props.parameters]
    );
    this.requiredHalo = props.searchRadius;
  }

  /** Returns the optional elevation canonicalization nodes and the fused geomorphon kernel. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {output} = props;
    const {width, height} = props.grid;
    validateRasterBandBelongsToGraph(id, graph, props.elevation, []);
    validateGraphViewsBelongToGraph(id, graph, [
      props.parameters,
      output.forms,
      output.ternary,
      output.pattern,
      output.outputValidity
    ]);
    const source = getRasterBufferBandNodes(graph, id, props.elevation, props.grid, true);
    const validity = source.band.validity;
    if (!validity || source.band.storage.kind !== 'buffer') {
      throw new Error(`${id} could not canonicalize elevation`);
    }
    const bindings: WGSLKernelBinding[] = [
      {name: 'elevation', view: source.band.storage.values, type: 'f32', access: 'read'},
      {name: 'elevationValidity', view: validity, type: 'u32', access: 'read'},
      {name: 'settings', view: props.parameters, type: 'f32', access: 'read'}
    ];
    for (const [name, view] of [
      ['formValues', output.forms],
      ['ternaryValues', output.ternary],
      ['patternValues', output.pattern],
      ['validityValues', output.outputValidity]
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
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-classify`,
        operation: 'GPUGeomorphons',
        variant: `${comparison}-${cellSizeMode}-skip${skipRadius}-radius${props.searchRadius}`,
        bindings,
        invocationCount: width * height,
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
  cellSizeMode: GPURasterCellSizeMode,
  rowDirection: GPURasterRowDirection
): string {
  return /* wgsl */ `
const WIDTH: u32 = ${props.grid.width}u;
const HEIGHT: u32 = ${props.grid.height}u;
const SEARCH_RADIUS: i32 = ${props.searchRadius};
const SKIP_RADIUS: i32 = ${skipRadius};
const COMPARISON: u32 = ${comparison === 'anglev1' ? 0 : comparison === 'anglev2' ? 1 : 2}u;
${getRasterRowNorthSignWGSL(rowDirection, 'i32')}
${RASTER_DEGREES_TO_RADIANS_WGSL}
${RASTER_WGSL_HELPERS}
${getRasterGroundCellSizeWGSL(cellSizeMode, TERRAIN_GEOMORPHOMETRY_CELL_SLOTS)}

// Directions in GRASS order: NE, N, NW, W, SW, S, SE, E.
const DIRECTION_COLUMN = array<i32, 8>(1, 0, -1, -1, -1, 0, 1, 1);
const DIRECTION_NORTH = array<i32, 8>(1, 1, 1, 0, -1, -1, -1, 0);
// Landform codes by minus count (row) and plus count (column); 0 marks impossible pairs.
const FORM_TABLE = array<u32, 81>(
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
  // Canonical elevation is NaN exactly where a pixel is invalid, so no validity load is needed.
  return isFiniteValue(elevation[elevationOffset + u32(row) * WIDTH + u32(column)]);
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
    props.output.forms ? 'formValues[formValuesOffset + index] = outputForm;' : '',
    props.output.ternary ? 'ternaryValues[ternaryValuesOffset + index] = outputTernary;' : '',
    props.output.pattern ? 'patternValues[patternValuesOffset + index] = outputPattern;' : '',
    props.output.outputValidity
      ? 'validityValues[validityValuesOffset + index] = select(0u, 1u, isValid);'
      : ''
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
    // GRASS r.geomorphon ignores flat_distance at or inside the skip distance and at or beyond
    // the search distance (review C-13).
    let skipDistance = f32(SKIP_RADIUS) * groundCell.y;
    let flatReference = select(
      0.0,
      flatDistance,
      flatDistance > skipDistance && flatDistance < f32(SEARCH_RADIUS) * groundCell.y
    );
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
        let sampleElevation = getElevation(sampleColumn, sampleRow);
        if (!isFiniteValue(sampleElevation)) { continue; }
        let height = zFactor * (sampleElevation - centerElevation);
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
        zenithHeight, zenithCount, nadirHeight, nadirCount, tangent, flatReference, step
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
