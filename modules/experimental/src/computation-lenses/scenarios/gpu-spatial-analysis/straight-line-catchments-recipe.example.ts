// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPUDistanceFieldParameterValues,
  GPU_DISTANCE_FIELD_PARAMETER_LENGTH,
  makeGPURasterMetadata
} from '@luma.gl/experimental/gpu-raster';
import {
  addStraightLineCatchmentsRecipe,
  GPUParameterBuffer,
  type GPUSpatialContext
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Inputs the lens scenario uploads from the Chicago North Side facilities and observations. */
export type StraightLineCatchmentsRecipeExampleInput = {
  /** Facility positions, projected to local-frame metres. */
  facilities: GraphDataView<'float32x2'>;
  /** Observations per grid cell, row-major from the top row of the extent. */
  observationCounts: GraphDataView<'float32'>;
  /** Planar context of the local frame. */
  spatialContext: GPUSpatialContext;
  /** Grid extent `[minX, minY, maxX, maxY]` in frame metres, a whole number of cells. */
  extent: readonly [number, number, number, number];
  /** Grid cell size in metres. */
  cellSize: number;
};

/** Nearest-facility catchments on a raster, with the observations summed per catchment. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {
    facilities,
    observationCounts,
    spatialContext,
    extent,
    cellSize
  }: StraightLineCatchmentsRecipeExampleInput
) {
  const [minX, minY, maxX, maxY] = extent;
  const width = Math.round((maxX - minX) / cellSize);
  const height = Math.round((maxY - minY) / cellSize);
  const grid = makeGPURasterMetadata({width, height, cellSize, origin: [minX, maxY]});
  const parameters = new GPUParameterBuffer(graph.device, {
    id: 'catchment-parameters',
    format: 'float32',
    length: GPU_DISTANCE_FIELD_PARAMETER_LENGTH,
    values: getGPUDistanceFieldParameterValues({}, grid)
  });
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const output = <Format extends 'float32' | 'uint32'>(
    id: string,
    format: Format,
    length: number
  ) => createTransientView(graph, id, format, length, usage);
  const zoneCount = facilities.length;
  const allocation = output('allocation', 'uint32', width * height);
  const distances = output('distances', 'float32', width * height);
  const statistics = {
    cellCounts: output('cellCounts', 'uint32', zoneCount),
    valueCounts: output('valueCounts', 'uint32', zoneCount),
    sums: output('sums', 'float32', zoneCount),
    means: output('means', 'float32', zoneCount),
    minimums: output('minimums', 'float32', zoneCount),
    maximums: output('maximums', 'float32', zoneCount)
  };
  // #region example
  const catchments = addStraightLineCatchmentsRecipe(graph, {
    spatialContext,
    grid,
    parameters: parameters.importToGraph(graph),
    facilities: {kind: 'points', positions: facilities},
    values: observationCounts,
    outputs: {allocation, distances, statistics}
  });
  // allocation colours each cell by its nearest facility; sums totals observations per catchment
  // #endregion example
  return {catchments, allocation, distances, statistics, grid, parameters};
}
