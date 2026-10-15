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
  addSpaceTimeHotSpotsRecipe,
  GPUParameterBuffer,
  GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH,
  type GPUSpatialContext
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Slices of the cube: one per month of 2023. */
export const SPACE_TIME_SLICE_COUNT = 12;

/** Inputs the lens scenario uploads from `s2-chicago-north-nature.json`. */
export type SpaceTimeHotSpotsRecipeExampleInput = {
  /** iNaturalist observations, projected to local-frame metres. */
  observations: GraphDataView<'float32x2'>;
  /** Observation times as little-endian Int64 epoch milliseconds. */
  timestamps: GraphDataView<'uint32x2'>;
  /** `getGPUCalendarBucketsParameterValues(0)`: UTC. */
  calendarParameters: GraphDataView<'sint32'>;
  /** Planar context of the local frame. */
  spatialContext: GPUSpatialContext;
  /** Lattice columns. */
  width: number;
  /** Lattice rows. */
  height: number;
  /** Covered extent `[minX, minY, maxX, maxY]`, a whole number of cells. */
  bounds: readonly [number, number, number, number];
};

/** Timestamped observations to emerging hot spots on an 800 m lattice and 12 monthly slices. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {
    observations,
    timestamps,
    calendarParameters,
    spatialContext,
    width,
    height,
    bounds
  }: SpaceTimeHotSpotsRecipeExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const cellCount = width * height;
  const binCount = cellCount * SPACE_TIME_SLICE_COUNT;
  const create = <Format extends 'float32' | 'uint32' | 'sint32'>(
    id: string,
    format: Format,
    length: number
  ) => createTransientView(graph, id, format, length, usage);
  const outputs = {
    cube: create('cube', 'uint32', binCount),
    giZScores: create('giZScores', 'float32', binCount),
    trendZ: create('trendZ', 'float32', cellCount),
    category: create('category', 'uint32', cellCount),
    hotSliceCount: create('hotSliceCount', 'uint32', cellCount),
    coldSliceCount: create('coldSliceCount', 'uint32', cellCount)
  };
  const parameters = new GPUParameterBuffer(graph.device, {
    id: 'emerging-hot-spot-parameters',
    format: 'float32',
    length: GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH
  });
  // #region example
  addSpaceTimeHotSpotsRecipe(graph, {
    timestamps,
    calendarParameters,
    slices: {step: 'month', origin: {year: 2023, month: 1}, count: 12},
    cells: {kind: 'lattice', positions: observations, spatialContext, width, height, bounds},
    settings: {radius: 1.5, temporalWindow: 1},
    parameters,
    outputs
  });
  // category per cell: the 17 ArcGIS emerging hot spot classes from the month-by-cell cube
  // #endregion example
  return {...outputs, parameters};
}
