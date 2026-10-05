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
} from '../../gpu-raster/cost-distance/raster-grid-utils';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import type {GPUTerrainCellSizeMode} from '../terrain-analysis/index';
import {
  getTerrainBandViews,
  getTerrainElevationNodes,
  validateTerrainBandBelongsToGraph,
  validateTerrainGrid,
  validateTerrainSettings
} from '../terrain-analysis/terrain-analysis-utils';

const OPERATION = 'GPUTerrainHydrologicIndices';

/** Number of float32 values read from `GPUTerrainHydrologicIndicesProps.settings`. */
export const GPU_TERRAIN_HYDROLOGIC_INDICES_PARAMETER_LENGTH = 8;

/** Default lower bound of `tan(beta)` used by the indices. */
export const GPU_TERRAIN_HYDROLOGIC_INDICES_DEFAULT_MINIMUM_SLOPE = 0.001;

/**
 * CPU-side description packed by {@link getGPUTerrainHydrologicIndicesParameterValues}.
 *
 * Cell sizes follow the shared raster grid prefix: with `cellSizeMode` `'uniform'` they are
 * projected meters; with `'web-mercator'` (equatorial meters) or `'geographic'` (degrees) the
 * ground spacing is evaluated per row from `northEdge`/`southEdge` (latitude-dependent cell
 * width), for both the slope and the contour width.
 */
export type GPUTerrainHydrologicIndicesSettings = RasterGridSettings & {
  /**
   * Lower bound applied to `tan(beta)` before the indices are formed, so filled flats do not give
   * an infinite wetness index. Negative or non-finite values act as 0 (no clamp: zero slopes give
   * `+Infinity` wetness and zero stream power). Defaults to
   * {@link GPU_TERRAIN_HYDROLOGIC_INDICES_DEFAULT_MINIMUM_SLOPE} (0.001, about 0.06 degrees).
   */
  minimumSlope?: number;
};

/**
 * Packs `[cellSizeX, cellSizeY, northEdge, southEdge, minimumSlope, 0, 0, 0]`.
 *
 * @throws If `target` holds fewer than 8 values.
 */
export function getGPUTerrainHydrologicIndicesParameterValues(
  settings: GPUTerrainHydrologicIndicesSettings,
  target: Float32Array = new Float32Array(GPU_TERRAIN_HYDROLOGIC_INDICES_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_TERRAIN_HYDROLOGIC_INDICES_PARAMETER_LENGTH) {
    throw new Error('Terrain hydrologic indices settings target must hold 8 values');
  }
  writeRasterGridSettings(target, settings);
  target[4] = settings.minimumSlope ?? GPU_TERRAIN_HYDROLOGIC_INDICES_DEFAULT_MINIMUM_SLOPE;
  target[5] = 0;
  target[6] = 0;
  target[7] = 0;
  return target;
}

/**
 * Properties for {@link GPUTerrainHydrologicIndices}.
 *
 * Compile-time: `width`, `height`, `cellSizeMode`, the elevation format and calibration, and which
 * outputs are provided. Per-frame: the contents of `elevation`, `accumulation`, and `settings`.
 * Outputs never share a buffer with an input or with each other.
 */
export type GPUTerrainHydrologicIndicesProps = {
  /** Prefix for node and transient IDs. Defaults to `'terrain-hydrologic-indices'`. Compile-time. */
  id?: string;
  /** Grid width in cells. Compile-time. */
  width: number;
  /** Grid height in cells. Compile-time. */
  height: number;
  /**
   * Routing surface the slope is measured on, normally the `filledElevation` of
   * `GPUTerrainFlow` (or the raw DEM when it was not filled). Validity, nodata, scale and offset
   * are honored.
   */
  elevation: GPURasterBand;
  /**
   * Upslope contributing area in m^2 per cell, including the cell itself: `GPUTerrainFlow`
   * `accumulation` with `accumulationUnits: 'area'` (any routing). Non-finite values give NaN;
   * zero gives a wetness index of `-Infinity`.
   */
  accumulation: GraphDataView<'float32'>;
  /** Settings with at least 8 float32 values, see {@link getGPUTerrainHydrologicIndicesParameterValues}. Per-frame. */
  settings: GraphDataView<'float32'>;
  /** Cell size interpretation. Defaults to `'uniform'`. Compile-time. */
  cellSizeMode?: GPUTerrainCellSizeMode;
  /** `tan(beta)` after the `minimumSlope` clamp. Presence is compile-time. */
  slope?: GraphDataView<'float32'>;
  /** Specific catchment area `a = A / b` in meters. Presence is compile-time. */
  specificCatchmentArea?: GraphDataView<'float32'>;
  /** Topographic wetness index `ln(a / tan(beta))`. Presence is compile-time. */
  wetnessIndex?: GraphDataView<'float32'>;
  /** Stream power index `a * tan(beta)` in meters. Presence is compile-time. */
  streamPowerIndex?: GraphDataView<'float32'>;
};

/**
 * Flow-accumulation-based terrain indices on one raster tile: specific catchment area, the
 * topographic wetness index (Beven and Kirkby 1979) and the stream power index (Moore, Grayson and
 * Ladson 1991).
 *
 * Definitions, per valid cell with a finite contributing area `A`:
 * - `tan(beta)` is the steepest D8 descent on the routing surface, `max((z - z_n) / d_n, 0)` over
 *   valid neighbors with ground distances `d_n` (the slope `GPUTerrainFlow` routes D8 flow on),
 *   then raised to `minimumSlope`.
 * - Contour width `b = sqrt(groundX * groundY)` at the cell's row, so `b` is the cell size on
 *   square projected grids (the TauDEM and Whitebox convention) and the equal-area side otherwise.
 * - `a = A / b`, wetness `ln(a / tan(beta))`, stream power `a * tan(beta)`.
 *
 * Invalid cells and cells with a non-finite `A` get NaN. Implementations differ mainly in the
 * slope: Whitebox and SAGA usually take a 3x3 (Horn or Zevenbergen-Thorne) slope raster and
 * TauDEM the D-infinity slope; this recipe uses the D8 descent slope so `tan(beta)` and the flow
 * routing agree on flats and filled depressions. One kernel, no iteration.
 */
export class GPUTerrainHydrologicIndices implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUTerrainHydrologicIndicesProps;

  constructor(props: GPUTerrainHydrologicIndicesProps) {
    this.id = props.id ?? 'terrain-hydrologic-indices';
    this.props = props;
    const {id} = this;
    const cellCount = validateTerrainGrid(id, props.width, props.height);
    validateTerrainSettings(id, props.settings, GPU_TERRAIN_HYDROLOGIC_INDICES_PARAMETER_LENGTH);
    validateRasterCellSizeMode(id, props.cellSizeMode ?? 'uniform');
    const outputs = this.getOutputs();
    if (outputs.length === 0) {
      throw new Error(`${id} requires at least one output`);
    }
    for (const [name, view] of [['accumulation', props.accumulation], ...outputs] as const) {
      validatePackedView(view, ['float32'], `${id} ${name}`);
      if (view.length !== cellCount) {
        throw new Error(`${id} ${name} must contain one value per cell`);
      }
    }
    const outputBuffers = outputs.map(([, view]) => view.buffer);
    if (new Set(outputBuffers).size !== outputBuffers.length) {
      throw new Error(`${id} outputs must not share buffers with each other`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      outputs.map(([, view]) => view),
      [...getTerrainBandViews(props.elevation), props.accumulation, props.settings]
    );
  }

  /** Returns the elevation canonicalization node and the index kernel. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {width, height} = props;
    const outputs = this.getOutputs();
    validateTerrainBandBelongsToGraph(id, graph, props.elevation, []);
    validateGraphViewsBelongToGraph(id, graph, [
      props.accumulation,
      props.settings,
      ...outputs.map(([, view]) => view)
    ]);
    const source = getTerrainElevationNodes(graph, id, props.elevation, width, height, true);
    const elevation = source.band.storage.values as GraphDataView<'float32'>;
    const bindings: WGSLKernelBinding[] = [
      {name: 'elevation', view: elevation, type: 'f32', access: 'read'},
      {name: 'accumulation', view: props.accumulation, type: 'f32', access: 'read'},
      {name: 'settings', view: props.settings, type: 'f32', access: 'read'},
      ...outputs.map(([name, view]) => ({
        name,
        view,
        type: 'f32' as const,
        access: 'read_write' as const
      }))
    ];
    const write = (value: (name: string) => string) =>
      outputs.map(([name]) => `${name}[${name}Offset + index] = ${value(name)};`).join('\n  ');
    const values: Record<string, string> = {
      slope: 'tangent',
      specificCatchmentArea: 'catchment',
      wetnessIndex: 'wetness',
      streamPowerIndex: 'catchment * tangent'
    };
    const indices = createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-indices`,
      operation: OPERATION,
      variant: 'indices',
      bindings,
      invocationCount: width * height,
      declarations: getRasterGridWGSL({
        width,
        height,
        cellSizeMode: props.cellSizeMode ?? 'uniform'
      }),
      body: `let row = index / GRID_WIDTH;
  let center = elevation[elevationOffset + index];
  let area = accumulation[accumulationOffset + index];
  if (!isFiniteValue(center) || !isFiniteValue(area)) {
    ${write(() => 'getQuietNaN()')}
    return;
  }
  var steepest = 0.0;
  for (var direction = 0u; direction < 8u; direction++) {
    let neighbor = getD8Neighbor(index, direction);
    if (neighbor == GRID_NONE) { continue; }
    let neighborValue = elevation[elevationOffset + neighbor];
    if (!isFiniteValue(neighborValue)) { continue; }
    steepest = max(steepest, (center - neighborValue) / getD8Distance(direction, row));
  }
  var minimumSlope = settings[settingsOffset + 4u];
  if (!(minimumSlope >= 0.0) || !isFiniteValue(minimumSlope)) { minimumSlope = 0.0; }
  let tangent = max(steepest, minimumSlope);
  let ground = getGroundCellSize(f32(row) + 0.5);
  let catchment = area / sqrt(ground.x * ground.y);
  // WGSL leaves division by zero and log(0) unspecified, so the limits are written explicitly.
  var wetness = getInfinity();
  if (catchment <= 0.0) {
    wetness = -getInfinity();
  } else if (tangent > 0.0) {
    wetness = log(catchment / tangent);
  }
  ${write(name => values[name])}`
    });
    return [...source.nodes, indices];
  }

  private getOutputs(): [string, GraphDataView<'float32'>][] {
    const {props} = this;
    const outputs: [string, GraphDataView<'float32'> | undefined][] = [
      ['slope', props.slope],
      ['specificCatchmentArea', props.specificCatchmentArea],
      ['wetnessIndex', props.wetnessIndex],
      ['streamPowerIndex', props.streamPowerIndex]
    ];
    return outputs.filter((entry): entry is [string, GraphDataView<'float32'>] =>
      Boolean(entry[1])
    );
  }
}
