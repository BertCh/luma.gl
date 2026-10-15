// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {makeGPURasterBand, type GPURasterMetadata} from '@luma.gl/experimental/gpu-raster';
import {
  getGPUSolarIrradianceParameterValues,
  getGPUSolarIrradianceSunTable,
  getGPUTerrainDerivativesParameterValues,
  getGPUTerrainHorizonParameterValues,
  GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH,
  GPU_TERRAIN_HORIZON_PARAMETER_LENGTH,
  GPUParameterBuffer,
  GPUSolarIrradiance,
  GPUTerrainDerivatives,
  GPUTerrainHorizon
} from '@luma.gl/experimental/gpu-terrain';

/** Graph, grid and caller-owned views the example wires together. */
export type SolarIrradianceExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  dem: GraphDataView<'float32'>;
  /** Pixel-major horizon angles, `width * height * 16` values. */
  horizon: GraphDataView<'float32'>;
  /** Horn slope and aspect in degrees, written by the prerequisite `GPUTerrainDerivatives`. */
  slope: GraphDataView<'float32'>;
  aspect: GraphDataView<'float32'>;
  hillshade: GraphDataView<'float32'>;
  sunHours: GraphDataView<'float32'>;
  insolation: GraphDataView<'float32'>;
};

/**
 * Adds the horizon map, slope and aspect, and day-long irradiance to `graph`; returns the buffers
 * to destroy. The `GPUTerrainHorizon` (16 sectors, 32 cells) and `GPUTerrainDerivatives` steps
 * (see their cards) precede the shown snippet.
 */
export async function runExample(device: Device, input: SolarIrradianceExampleInput) {
  const {graph, grid, dem, horizon, slope, aspect, sunHours, insolation} = input;
  const horizonSettings = new GPUParameterBuffer(device, {
    id: 'horizon-settings',
    format: 'float32',
    length: GPU_TERRAIN_HORIZON_PARAMETER_LENGTH,
    values: getGPUTerrainHorizonParameterValues({}, grid)
  });
  graph.add(
    new GPUTerrainHorizon({
      grid,
      elevation: makeGPURasterBand(dem, grid),
      parameters: horizonSettings.importToGraph(graph),
      maximumRadius: 32,
      output: {horizon}
    })
  );
  // Prerequisite: Horn slope and aspect tilt each cell's surface toward or away from the sun.
  const derivativeSettings = new GPUParameterBuffer(device, {
    id: 'derivative-settings',
    format: 'float32',
    length: GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH,
    values: getGPUTerrainDerivativesParameterValues({}, grid)
  });
  graph.add(
    new GPUTerrainDerivatives({
      grid,
      elevation: makeGPURasterBand(dem, grid),
      parameters: derivativeSettings.importToGraph(graph),
      output: {slope, aspect, hillshade: input.hillshade}
    })
  );
  // #region example
  const buffer = (id: string, values: Float32Array) =>
    new GPUParameterBuffer(device, {id, format: 'float32', length: values.length, values});
  const {values, sampleCount} = getGPUSolarIrradianceSunTable({
    longitude: -120.95, // Greenville CA, 5 August 2021 in PDT
    latitude: 40.15,
    start: new Date('2021-08-05T07:00:00Z'),
    end: new Date('2021-08-06T07:00:00Z'),
    stepMinutes: 15
  });
  const sunTable = buffer('sun-table', values);
  const settings = buffer('settings', getGPUSolarIrradianceParameterValues({sampleCount}));
  graph.add(
    new GPUSolarIrradiance({
      grid, // 64 × 64 cells of 120 m, UTM 10N
      directionCount: 16,
      horizon, // from GPUTerrainHorizon, pixel-major
      slope, // degrees, with aspect from GPUTerrainDerivatives
      aspect,
      sunTable: sunTable.importToGraph(graph),
      sampleCapacity: sampleCount,
      parameters: settings.importToGraph(graph),
      output: {sunHours, insolation} // hours, Wh/m² on the sloped surface
    })
  );
  // #endregion example
  return {horizonSettings, derivativeSettings, sunTable, settings};
}
