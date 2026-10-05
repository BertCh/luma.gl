// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPURasterBand} from '../../gpu-raster/index';
import {
  getRasterGridWGSL,
  validateRasterCellSizeMode,
  writeRasterGridSettings,
  type RasterGridSettings
} from '../cost-distance/raster-grid-utils';
import {createMapGraphKernelNode} from '../map-graph-kernels';
import type {GPUMapGraphRecipe} from '../map-graph-types';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../map-graph-utils';
import type {GPUTerrainCellSizeMode} from '../terrain-analysis/index';
import {
  getTerrainBandViews,
  getTerrainElevationNodes,
  validateTerrainBandBelongsToGraph,
  validateTerrainGrid,
  validateTerrainSettings
} from '../terrain-analysis/terrain-analysis-utils';

const OPERATION = 'GPUTerrainFlowField';

/** Number of float32 values read from `GPUTerrainFlowFieldProps.settings`. */
export const GPU_TERRAIN_FLOW_FIELD_PARAMETER_LENGTH = 8;

/**
 * CPU-side description packed by {@link getGPUTerrainFlowFieldParameterValues}.
 *
 * Cell sizes follow the shared raster grid prefix: with `cellSizeMode` `'uniform'` they are
 * projected meters; with `'web-mercator'` (equatorial meters) or `'geographic'` (degrees) the
 * ground spacing is evaluated per row from `northEdge`/`southEdge`, so the gradient honors the
 * latitude-dependent cell width. Elevations are meters after the band's scale and offset.
 */
export type GPUTerrainFlowFieldSettings = RasterGridSettings & {
  /**
   * Horizontal wind `[x, y]` in the raster frame: x along increasing columns, y along increasing
   * rows (south for a north-up raster whose row 0 is the northern edge). Any speed unit; the
   * output velocities share it.
   */
  wind: readonly [number, number];
  /** Factor applied to the elevation gradient before deflection. Defaults to 1. */
  verticalExaggeration?: number;
};

/**
 * Packs `[cellSizeX, cellSizeY, northEdge, southEdge, windX, windY, verticalExaggeration, 0]`.
 *
 * @throws If `target` holds fewer than 8 values.
 */
export function getGPUTerrainFlowFieldParameterValues(
  settings: GPUTerrainFlowFieldSettings,
  target: Float32Array = new Float32Array(GPU_TERRAIN_FLOW_FIELD_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_TERRAIN_FLOW_FIELD_PARAMETER_LENGTH) {
    throw new Error('Terrain flow field settings target must hold 8 values');
  }
  writeRasterGridSettings(target, settings);
  target[4] = settings.wind[0];
  target[5] = settings.wind[1];
  target[6] = settings.verticalExaggeration ?? 1;
  target[7] = 0;
  return target;
}

/**
 * Properties for {@link GPUTerrainFlowField}.
 *
 * Compile-time: `width`, `height`, `cellSizeMode`, and the elevation format and calibration.
 * Per-frame (no recompile): the contents of `elevation` and `settings` (wind, exaggeration).
 */
export type GPUTerrainFlowFieldProps = {
  /** Prefix for node and transient IDs. Defaults to `'terrain-flow-field'`. Compile-time. */
  id?: string;
  /** Grid width in cells. Compile-time. */
  width: number;
  /** Grid height in cells. Compile-time. */
  height: number;
  /**
   * Elevation band, buffer or texture. Validity, nodata, scale, and offset are honored; cells
   * without a finite calibrated elevation are invalid.
   */
  elevation: GPURasterBand;
  /** Settings with at least 8 float32 values, see {@link getGPUTerrainFlowFieldParameterValues}. Per-frame. */
  settings: GraphDataView<'float32'>;
  /** Cell size interpretation. Defaults to `'uniform'`. Compile-time. */
  cellSizeMode?: GPUTerrainCellSizeMode;
  /**
   * Output field, `width * height` rows of `(u, v)` in the raster frame (u along increasing
   * columns, v along increasing rows), in the wind's unit. NaN where the cell or a neighbor used by
   * its gradient is invalid. This is the `velocities` format of `GPUParticleAdvection`,
   * `GPUStreamlines` and `GPULineIntegralConvolution` (row 0 is their smallest y).
   */
  velocities: GraphDataView<'float32x2'>;
};

/**
 * Terrain-following wind: a uniform horizontal wind deflected by the DEM so it flows around and
 * over relief, ready for the particle-advection and flow-texture recipes.
 *
 * Air cannot enter the ground, so the 3D wind is projected onto the terrain tangent plane. With
 * the surface gradient `g = (dh/dx, dh/dy)` the horizontal part of that projection is
 * `v = w - (w . g) g / (1 + |g|^2)`: wind across a slope is unchanged, wind straight up or down a
 * slope loses the component the ground takes (a vertical wall stops it). The gradient uses
 * central differences in ground meters, one-sided at the grid border; a cell whose own elevation
 * or any gradient neighbor is invalid gets NaN, which the field samplers treat as no data.
 * Deterministic, stateless, one kernel.
 *
 * Coordinate frame: wind and output are in the raster frame (x = columns, y = rows). Running the
 * particles in raster-local coordinates (`extent = [0, 0, 1, 1]` in cells, or ground cell sizes)
 * keeps the frames identical; for a north-up raster a wind toward the north is `[0, -speed]`.
 *
 * The field model comes from the Rigi terrain viewer's flow layer (MIT, same author), which pairs
 * it with a port of luma.gl's `FlowParticleSimulation`.
 */
export class GPUTerrainFlowField implements GPUMapGraphRecipe {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Stable recipe name. */
  readonly recipe = 'terrain-flow-field';
  /** Validated properties. */
  readonly props: GPUTerrainFlowFieldProps;

  constructor(props: GPUTerrainFlowFieldProps) {
    this.id = props.id ?? this.recipe;
    this.props = props;
    const {id} = this;
    const cellCount = validateTerrainGrid(id, props.width, props.height);
    validateTerrainSettings(id, props.settings, GPU_TERRAIN_FLOW_FIELD_PARAMETER_LENGTH);
    validateRasterCellSizeMode(id, props.cellSizeMode ?? 'uniform');
    validatePackedView(props.velocities, ['float32x2'], `${id} velocities`);
    if (props.velocities.length !== cellCount) {
      throw new Error(`${id} velocities must contain one row per cell`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.velocities],
      [...getTerrainBandViews(props.elevation), props.settings]
    );
  }

  /** Returns the elevation canonicalization node and the deflection kernel. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {width, height} = props;
    validateTerrainBandBelongsToGraph(id, graph, props.elevation, []);
    validateGraphViewsBelongToGraph(id, graph, [props.settings, props.velocities]);
    const source = getTerrainElevationNodes(graph, id, props.elevation, width, height, true);
    const elevation = source.band.storage.values as GraphDataView<'float32'>;
    const grid = {width, height, cellSizeMode: props.cellSizeMode ?? 'uniform'};
    const deflect = createMapGraphKernelNode<Parameters>(graph, {
      id: `${id}-deflect`,
      operation: OPERATION,
      variant: 'deflect',
      bindings: [
        {name: 'elevation', view: elevation, type: 'f32', access: 'read'},
        {name: 'settings', view: props.settings, type: 'f32', access: 'read'},
        {name: 'velocities', view: props.velocities, type: 'f32', access: 'read_write'}
      ],
      invocationCount: width * height,
      declarations: `${getRasterGridWGSL(grid)}
// Direction indices of the grid helpers: 2 = next row, 6 = previous row.
const DIRECTION_NEXT_ROW: u32 = 2u;
const DIRECTION_PREVIOUS_ROW: u32 = 6u;`,
      body: `let column = index % GRID_WIDTH;
  let row = index / GRID_WIDTH;
  let column0 = select(column - 1u, 0u, column == 0u);
  let column1 = min(column + 1u, GRID_WIDTH - 1u);
  let row0 = select(row - 1u, 0u, row == 0u);
  let row1 = min(row + 1u, GRID_HEIGHT - 1u);
  let center = elevation[elevationOffset + index];
  let previousColumn = elevation[elevationOffset + row * GRID_WIDTH + column0];
  let nextColumn = elevation[elevationOffset + row * GRID_WIDTH + column1];
  let previousRow = elevation[elevationOffset + row0 * GRID_WIDTH + column];
  let nextRow = elevation[elevationOffset + row1 * GRID_WIDTH + column];
  let output = 2u * index + velocitiesOffset;
  if (!isFiniteValue(center) || !isFiniteValue(previousColumn) || !isFiniteValue(nextColumn) ||
      !isFiniteValue(previousRow) || !isFiniteValue(nextRow)) {
    velocities[output] = getQuietNaN();
    velocities[output + 1u] = getQuietNaN();
    return;
  }
  let exaggeration = settings[settingsOffset + 6u];
  var gradientX = 0.0;
  if (column1 > column0) {
    let spanX = f32(column1 - column0) * getGroundCellSize(f32(row) + 0.5).x;
    gradientX = exaggeration * ((nextColumn - previousColumn) / spanX);
  }
  var gradientY = 0.0;
  if (row1 > row0) {
    var spanY = 0.0;
    if (row1 > row) { spanY = spanY + getD8Distance(DIRECTION_NEXT_ROW, row); }
    if (row0 < row) { spanY = spanY + getD8Distance(DIRECTION_PREVIOUS_ROW, row); }
    gradientY = exaggeration * ((nextRow - previousRow) / spanY);
  }
  let wind = vec2<f32>(settings[settingsOffset + 4u], settings[settingsOffset + 5u]);
  let gradient = vec2<f32>(gradientX, gradientY);
  let deflection = dot(wind, gradient) / (1.0 + dot(gradient, gradient));
  let velocity = wind - deflection * gradient;
  velocities[output] = velocity.x;
  velocities[output + 1u] = velocity.y;`
    });
    return [...source.nodes, deflect];
  }
}
