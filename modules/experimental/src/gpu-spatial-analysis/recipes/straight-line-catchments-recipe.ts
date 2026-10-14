// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {GPUDistanceField} from '../../gpu-raster/distance-field/index';
import type {GPUDistanceFieldMode} from '../../gpu-raster/distance-field/index';
import {GPURasterZonalStatistics} from '../../gpu-raster/raster-zonal-statistics/index';
import type {
  GPURasterZonalStatisticsOutput,
  GPURasterZonalStatisticsSumOrder
} from '../../gpu-raster/raster-zonal-statistics/index';
import type {GPURasterBand, GPURasterMetadata} from '../../gpu-raster/index';
import {makeGPURasterBand} from '../../gpu-raster/raster-grid/index';
import type {
  GPUPackedGeometryPort,
  GPUPointGeometryPort,
  GPURecipeStatusPort,
  GPUStatusScalar
} from '../../analysis-core/contracts/index';
import {
  assertRecipe,
  getOrCreateView,
  RecipeBuilder,
  type GPURecipeOverrides,
  type GPURecipeResult
} from './recipe-utils';

const ID = 'GPUStraightLineCatchmentsRecipe';

/** Native planar distances for the recipe's radius-based stages. */
const SPATIAL_CONTEXT_SUPPORT = Object.freeze({
  ...GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  measure: 'distance'
} as const);

/** Properties for {@link addStraightLineCatchmentsRecipe}. */
export type GPUStraightLineCatchmentsRecipeProps = GPURecipeOverrides<
  Record<never, never>,
  {
    /** Caller-owned catchment id (nearest facility) per cell. */
    allocation?: GraphDataView<'uint32'>;
    /** Caller-owned distance to the nearest facility per cell. */
    distances?: GraphDataView<'float32'>;
    /** Caller-owned per-catchment statistics. */
    statistics?: GPURasterZonalStatisticsOutput;
    /** Mirrors `result.status`. */
    status?: {
      /** Caller-owned one-row flag of the `zonal-statistics` stage, see `result.status.overflow`. */
      overflow?: GPUStatusScalar;
    };
  }
> & {
  /** Prefix for every node and transient ID. Defaults to `'straight-line-catchments'`. */
  id?: string;
  /**
   * Coordinate space of the grid and facility positions. Planar only: the distance field computes
   * Euclidean distances in ground units on a regular grid, which are not distances on the sphere
   * for longitude/latitude degrees. A longitude-latitude context throws; project the facilities
   * and choose the grid in a frame from `createGPULocalFrame` and pass its `spatialContext`, then
   * `parameters` cell sizes, `maximumDistance` and the `distances` output are in meters. Omitted
   * means native planar. The resolved context is returned as `result.spatialContext`.
   */
  spatialContext?: GPUSpatialContext;
  /** Raster grid of the catchments (width and height size the kernels). */
  grid: GPURasterMetadata;
  /** `getGPUDistanceFieldParameterValues` view (origin, cell size, optional `maximumDistance`). */
  parameters: GraphDataView<'float32'>;
  /**
   * Facility points in ground coordinates; the point count is the zone capacity. `sourceIds`
   * (optional) is the zone ID of each facility, defaulting to the row; it must be below the zone
   * capacity.
   */
  facilities: GPUPackedGeometryPort<GPUPointGeometryPort>;
  /** Optional one-row active facility count. */
  facilityCount?: GraphDataView<'uint32'>;
  /** Value raster summarised per catchment: a packed float32 view or a full raster band. */
  values: GraphDataView<'float32'> | GPURasterBand;
  /** Distance-field algorithm. Defaults to `'exact'`. */
  mode?: GPUDistanceFieldMode;
  /** Zonal sum accumulation. Defaults to the contributor default. */
  sumOrder?: GPURasterZonalStatisticsSumOrder;
};

/** Named outputs of {@link addStraightLineCatchmentsRecipe}. */
export type GPUStraightLineCatchmentsRecipeResult = GPURecipeResult & {
  /** Resolved coordinate context of the grid and facilities; always planar. */
  spatialContext: GPUSpatialContext;
  /** Nearest facility id per cell (the Voronoi zone). */
  allocation: GraphDataView<'uint32'>;
  /** Euclidean distance from each cell to its facility. */
  distances: GraphDataView<'float32'>;
  /** Per-catchment cell counts, value counts, sums, means, minimums and maximums. */
  statistics: Required<GPURasterZonalStatisticsOutput> & {
    status: {overflow: GraphDataView<'uint32'>};
  };
  /** Status words next to the per-stage `stages`. */
  status: {
    /**
     * One-row flag: 1 when a cell was allocated a zone id at or above the zone capacity (a
     * `seedIds` entry out of range), so that cell is missing from `statistics`. Stage
     * `zonal-statistics`.
     */
    overflow: GPUStatusScalar;
  } & GPURecipeStatusPort;
};

/**
 * Straight-line catchments recipe: nearest-facility zones on a raster and what falls in each.
 *
 * Chain: `GPUDistanceField` (Euclidean allocation, i.e. raster Voronoi) ->
 * `GPURasterZonalStatistics` over a value raster, with the allocation as the zone raster. Cells
 * with no facility within `maximumDistance` carry `0xffffffff` and are ignored by the statistics.
 *
 * Spatial context: planar only (see `GPUStraightLineCatchmentsRecipeProps.spatialContext`).
 * Distances are in the context's length unit, meters for a `createGPULocalFrame` context.
 *
 * Status: stage `zonal-statistics` reports `overflow` (a zone id past the zone capacity).
 */
export function addStraightLineCatchmentsRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUStraightLineCatchmentsRecipeProps
): GPUStraightLineCatchmentsRecipeResult {
  const id = props.id ?? 'straight-line-catchments';
  const spatialContext = resolveGPUSpatialContext(
    id,
    props.spatialContext,
    SPATIAL_CONTEXT_SUPPORT
  );
  const builder = new RecipeBuilder(graph);
  const {grid} = props;
  const cellCount = grid.width * grid.height;
  const outputs = props.outputs ?? {};
  const zoneCapacity = props.facilities.positions.length;
  assertRecipe(ID, cellCount >= 1, 'needs a non-empty grid');
  assertRecipe(ID, zoneCapacity >= 1, 'needs at least one facility row');

  const allocation = getOrCreateView(
    graph,
    `${id}-allocation`,
    'uint32',
    cellCount,
    outputs.allocation
  );
  const distances = getOrCreateView(
    graph,
    `${id}-distances`,
    'float32',
    cellCount,
    outputs.distances
  );
  builder.add(
    new GPUDistanceField({
      id: `${id}-distance-field`,
      grid,
      parameters: props.parameters,
      seedPositions: props.facilities.positions,
      seedIds: props.facilities.sourceIds,
      seedCount: props.facilityCount,
      mode: props.mode,
      output: {distances, allocation}
    })
  );

  const provided = outputs.statistics ?? {};
  const overflow = getOrCreateView(
    graph,
    `${id}-overflow`,
    'uint32',
    1,
    outputs.status?.overflow ?? provided.status?.overflow
  );
  const statistics = {
    cellCounts: getOrCreateView(
      graph,
      `${id}-cell-counts`,
      'uint32',
      zoneCapacity,
      provided.cellCounts
    ),
    valueCounts: getOrCreateView(
      graph,
      `${id}-value-counts`,
      'uint32',
      zoneCapacity,
      provided.valueCounts
    ),
    sums: getOrCreateView(graph, `${id}-sums`, 'float32', zoneCapacity, provided.sums),
    means: getOrCreateView(graph, `${id}-means`, 'float32', zoneCapacity, provided.means),
    minimums: getOrCreateView(graph, `${id}-minimums`, 'float32', zoneCapacity, provided.minimums),
    maximums: getOrCreateView(graph, `${id}-maximums`, 'float32', zoneCapacity, provided.maximums),
    status: {overflow}
  };
  const band: GPURasterBand =
    'storage' in props.values
      ? (props.values as GPURasterBand)
      : makeGPURasterBand(props.values as GraphDataView<'float32'>, grid, {id: `${id}-values`});
  builder.add(
    new GPURasterZonalStatistics({
      id: `${id}-zonal-statistics`,
      grid,
      zones: allocation,
      values: band,
      zoneCapacity,
      sumOrder: props.sumOrder,
      output: statistics
    })
  );
  return {
    spatialContext,
    contributors: builder.contributors,
    allocation,
    distances,
    statistics,
    outputs: {allocation, distances, statistics},
    intermediates: {},
    status: {overflow, stages: [{stage: 'zonal-statistics', status: {overflow}}]}
  };
}
