// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  addDriveTimeCatchmentRecipe,
  type GPUNetworkTopology
} from '@luma.gl/experimental/gpu-network';

/** Uploaded inputs: roads, facility and demand longitude/latitude, and ascending drive-time breaks. */
export type DriveTimeCatchmentRecipeExampleInput = {
  graph: GPUCommandGraph;
  roads: GPUNetworkTopology & {nodePositions: GraphDataView<'float32x2'>};
  facilityPositions: GraphDataView<'float32x2'>;
  demandPositions: GraphDataView<'float32x2'>;
  costBreaks: GraphDataView<'float32'>;
};

/** Adds the recipe shown on the card; the lens compiles, executes and reads it back. */
export async function runExample(_device: Device, input: DriveTimeCatchmentRecipeExampleInput) {
  const {graph, roads, facilityPositions, demandPositions, costBreaks} = input;
  // #region example
  const catchment = addDriveTimeCatchmentRecipe(graph, {
    spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'},
    topology: roads,
    facilities: facilityPositions,
    demand: demandPositions,
    costBreaks,
    maximumIterations: 24
  });
  const {demandTimes, demandBands, bands} = catchment;
  // #endregion example
  return {catchment, demandTimes, demandBands, bands};
}
