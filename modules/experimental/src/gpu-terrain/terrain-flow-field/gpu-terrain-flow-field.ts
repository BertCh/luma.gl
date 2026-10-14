// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPURasterBand, GPURasterMetadata} from '../../gpu-raster/index';
import {
  getRasterGridWGSL,
  writeRasterGridPrefix
} from '../../gpu-raster/cost-distance/raster-grid-utils';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {
  validateRasterCellSizeMode,
  type GPURasterCellSizeMode
} from '../../analysis-core/raster-grid';
import {
  getRasterBandViews,
  getRasterBufferBandNodes,
  validateParameterTarget,
  validateRasterBandBelongsToGraph,
  validateRasterGrid,
  validateRasterGridAxisAligned,
  validateRasterParameters
} from '../../gpu-raster/raster-band-utils';

const OPERATION = 'GPUTerrainFlowField';

/** Number of float32 values read from `GPUTerrainFlowFieldProps.parameters`. */
export const GPU_TERRAIN_FLOW_FIELD_PARAMETER_LENGTH = 8;

/**
 * CPU-side description packed by {@link getGPUTerrainFlowFieldParameterValues}.
 *
 * Cell sizes and the latitude band come from the grid passed next to it: with `cellSizeMode`
 * `'uniform'` the cell size is in projected meters; with `'web-mercator'` (equatorial meters) or
 * `'geographic'` (degrees) the ground spacing is evaluated per row from the grid's row edges, so
 * the gradient honors the latitude-dependent cell width. Elevations are meters after the band's
 * scale and offset.
 */
export type GPUTerrainFlowFieldSettings = {
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
 * Packs `[cellSizeX, cellSizeY, northEdge, southEdge, windX, windY, verticalExaggeration, 0]`,
 * with the cell size and the row-edge world y taken from `grid`.
 *
 * @throws If `target` holds fewer than 8 values.
 */
export function getGPUTerrainFlowFieldParameterValues(
  settings: GPUTerrainFlowFieldSettings,
  grid: GPURasterMetadata,
  target: Float32Array = new Float32Array(GPU_TERRAIN_FLOW_FIELD_PARAMETER_LENGTH)
): Float32Array {
  validateParameterTarget(
    'getGPUTerrainFlowFieldParameterValues',
    target,
    GPU_TERRAIN_FLOW_FIELD_PARAMETER_LENGTH
  );
  writeRasterGridPrefix(target, grid);
  target[4] = settings.wind[0];
  target[5] = settings.wind[1];
  target[6] = settings.verticalExaggeration ?? 1;
  target[7] = 0;
  return target;
}

/** Caller-owned outputs of {@link GPUTerrainFlowField}. */
export type GPUTerrainFlowFieldOutput = {
  /**
   * Output field, `width * height` rows of `(u, v)` in the raster frame (u along increasing
   * columns, v along increasing rows), in the wind's unit. NaN where the cell or a neighbor used by
   * its gradient is invalid. This is the `velocities` format of `GPUParticleAdvection`,
   * `GPUStreamlines` and `GPULineIntegralConvolution` (row 0 is their smallest y).
   */
  velocities: GraphDataView<'float32x2'>;
};

/**
 * Properties for {@link GPUTerrainFlowField}.
 *
 * Compile-time: `grid` size, `cellSizeMode`, and the elevation format and calibration.
 * Per-frame (no recompile): the contents of `elevation` and `parameters` (grid cell size and
 * latitude band, wind, exaggeration).
 */
export type GPUTerrainFlowFieldProps = {
  /** Prefix for node and transient IDs. Defaults to `'terrain-flow-field'`. Compile-time. */
  id?: string;
  /** Raster grid: its width and height size the kernels. */
  grid: GPURasterMetadata;
  /**
   * Elevation band, buffer or texture. Validity, nodata, scale, and offset are honored; cells
   * without a finite calibrated elevation are invalid.
   */
  elevation: GPURasterBand;
  /** Parameters with at least 8 float32 values, see {@link getGPUTerrainFlowFieldParameterValues}. Per-frame. */
  parameters: GraphDataView<'float32'>;
  /** Cell size interpretation. Defaults to `'uniform'`. Compile-time. */
  cellSizeMode?: GPURasterCellSizeMode;
  /** Caller-owned outputs. */
  output: GPUTerrainFlowFieldOutput;
};

/**
 * Terrain-following wind: a uniform horizontal wind deflected by the DEM so it flows around and
 * over relief, ready for the particle-advection and flow-texture contributors.
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
export class GPUTerrainFlowField implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUTerrainFlowFieldProps;

  constructor(props: GPUTerrainFlowFieldProps) {
    this.id = props.id ?? 'terrain-flow-field';
    this.props = props;
    const {id} = this;
    const {output} = props;
    const cellCount = validateRasterGrid(id, props.grid);
    validateRasterParameters(id, props.parameters, GPU_TERRAIN_FLOW_FIELD_PARAMETER_LENGTH);
    validateRasterCellSizeMode(id, props.cellSizeMode);
    validateRasterGridAxisAligned(id, props.grid, props.cellSizeMode);
    validatePackedView(output.velocities, ['float32x2'], `${id} output.velocities`);
    if (output.velocities.length !== cellCount) {
      throw new Error(`${id} output.velocities must contain one row per cell`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [output.velocities],
      [...getRasterBandViews(props.elevation), props.parameters]
    );
  }

  /** Returns the elevation canonicalization node and the deflection kernel. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {output} = props;
    const {width, height} = props.grid;
    validateRasterBandBelongsToGraph(id, graph, props.elevation, []);
    validateGraphViewsBelongToGraph(id, graph, [props.parameters, output.velocities]);
    const source = getRasterBufferBandNodes(graph, id, props.elevation, props.grid, true);
    const elevation = source.band.storage.values as GraphDataView<'float32'>;
    const grid = {width, height, cellSizeMode: props.cellSizeMode ?? 'uniform'};
    const deflect = createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-deflect`,
      operation: OPERATION,
      variant: 'deflect',
      bindings: [
        {name: 'elevation', view: elevation, type: 'f32', access: 'read'},
        {name: 'settings', view: props.parameters, type: 'f32', access: 'read'},
        {name: 'velocities', view: output.velocities, type: 'f32', access: 'read_write'}
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
    let spanX = f32(column1 - column0) * getGroundCellSizeAtRowCoordinate(f32(row) + 0.5).x;
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
