// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {resolveGPUSpatialContext, type GPUSpatialContext} from '../contracts/spatial-context';
import {GPUDistanceField} from '../../gpu-raster/distance-field/index';
import type {GPUDistanceFieldMode} from '../../gpu-raster/distance-field/index';
import {GPURasterZonalStatistics} from '../../gpu-raster/raster-zonal-statistics/index';
import type {
  GPURasterZonalStatisticsOutput,
  GPURasterZonalStatisticsSumOrder
} from '../../gpu-raster/raster-zonal-statistics/index';
import type {GPURasterBand} from '../../gpu-raster/index';
import {
  assertRecipe,
  getOrCreateView,
  RecipeBuilder,
  type GPURecipeOverrides,
  type GPURecipeResult
} from './recipe-utils';

const ID = 'GPUStraightLineCatchmentsRecipe';

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
  }
> & {
  /** Prefix for every node and transient ID. Defaults to `'straight-line-catchments'`. */
  id?: string;
  /**
   * Coordinate space of the grid and facility positions. Planar only: the distance field computes
   * Euclidean distances in ground units on a regular grid, which are not distances on the sphere
   * for longitude/latitude degrees. A longitude-latitude context throws; project the facilities
   * and choose the grid in a frame from `createGPULocalFrame` and pass its `spatialContext`, then
   * `settings` cell sizes, `maxDistance` and the `distances` output are in meters. Omitted means
   * native planar. The resolved context is returned as `result.spatialContext`.
   */
  spatialContext?: GPUSpatialContext;
  /** Grid width in cells. */
  width: number;
  /** Grid height in cells (row 0 is the minimum-y edge). */
  height: number;
  /** `getGPUDistanceFieldParameterValues` view (origin, cell size, optional `maxDistance`). */
  settings: GraphDataView<'float32'>;
  /** Facility positions in ground coordinates; its length is the zone capacity. */
  seedPositions: GraphDataView<'float32x2'>;
  /** Optional seed ID per facility; defaults to the row. Must be below the zone capacity. */
  seedIds?: GraphDataView<'uint32'>;
  /** Optional one-row active facility count. */
  seedCount?: GraphDataView<'uint32'>;
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
  statistics: Required<GPURasterZonalStatisticsOutput>;
};

/**
 * Straight-line catchments recipe: nearest-facility zones on a raster and what falls in each.
 *
 * Chain: `GPUDistanceField` (Euclidean allocation, i.e. raster Voronoi) ->
 * `GPURasterZonalStatistics` over a value raster, with the allocation as the zone raster. Cells
 * with no facility within `maxDistance` carry `0xffffffff` and are ignored by the statistics.
 *
 * Spatial context: planar only (see `GPUStraightLineCatchmentsRecipeProps.spatialContext`).
 * Distances are in the context's length unit, meters for a `createGPULocalFrame` context.
 */
export function addStraightLineCatchmentsRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUStraightLineCatchmentsRecipeProps
): GPUStraightLineCatchmentsRecipeResult {
  const id = props.id ?? 'straight-line-catchments';
  const spatialContext = resolveGPUSpatialContext(id, props.spatialContext);
  const builder = new RecipeBuilder(graph);
  const {width, height} = props;
  const cellCount = width * height;
  const outputs = props.outputs ?? {};
  const zoneCapacity = props.seedPositions.length;
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
      width,
      height,
      settings: props.settings,
      seedPositions: props.seedPositions,
      seedIds: props.seedIds,
      seedCount: props.seedCount,
      mode: props.mode,
      output: {distances, allocation}
    })
  );

  const provided = outputs.statistics ?? {};
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
    maximums: getOrCreateView(graph, `${id}-maximums`, 'float32', zoneCapacity, provided.maximums)
  };
  const band: GPURasterBand =
    'storage' in props.values
      ? (props.values as GPURasterBand)
      : ({
          id: `${id}-values`,
          format: 'float32',
          storage: {kind: 'buffer', values: props.values as GraphDataView<'float32'>}
        } as GPURasterBand);
  builder.add(
    new GPURasterZonalStatistics({
      id: `${id}-zonal-statistics`,
      width,
      height,
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
    status: {stages: []}
  };
}
