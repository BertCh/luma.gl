// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {makeGPURasterBand, type GPURasterMetadata} from '@luma.gl/experimental/gpu-raster';
import {GPUTerrainSpikeRepair} from '@luma.gl/experimental/gpu-terrain';

/** Elevation with injected spikes and the repair contributor's published outputs. */
export type TerrainSpikeRepairExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  dem: GraphDataView<'float32'>;
  repairedElevation: GraphDataView<'float32'>;
  repairValidity: GraphDataView<'uint32'>;
  repairLabels: GraphDataView<'uint32'>;
  repairStatistics: GraphDataView<'uint32'>;
};

/** Adds component-based repair and returns the contributor whose buffers need disposal. */
export async function runExample(device: Device, input: TerrainSpikeRepairExampleInput) {
  const {graph, grid, dem, repairedElevation, repairValidity, repairLabels, repairStatistics} =
    input;
  // #region example
  const repair = new GPUTerrainSpikeRepair({
    grid,
    elevation: makeGPURasterBand(dem, grid),
    step: 900,
    jump: 200,
    tolerance: 80, // metres; match the injected encoding step
    agreement: 0.8,
    maximumComponentFraction: 0.25,
    maximumComponentIterations: 32,
    output: {
      values: repairedElevation,
      outputValidity: repairValidity,
      labels: repairLabels,
      statistics: repairStatistics
    }
  });
  graph.add(repair);
  // #endregion example
  return {repair};
}
