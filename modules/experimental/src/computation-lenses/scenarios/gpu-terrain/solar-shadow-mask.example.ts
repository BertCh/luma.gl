// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {makeGPURasterBand, type GPURasterMetadata} from '@luma.gl/experimental/gpu-raster';
import {
  getGPUSolarShadowMaskParameterValues,
  getGPUTerrainHorizonParameterValues,
  getSolarPosition,
  GPUParameterBuffer,
  GPUSolarShadowMask,
  GPUTerrainHorizon
} from '@luma.gl/experimental/gpu-terrain';

/** Graph, grid and caller-owned views the example wires together. */
export type SolarShadowMaskExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  dem: GraphDataView<'float32'>;
  /** Pixel-major horizon angles, `width * height * 16` values. */
  horizon: GraphDataView<'float32'>;
  sunVisibility: GraphDataView<'float32'>;
};

/** Adds horizon map and shadow mask for one sun to `graph`; returns the buffers to destroy. */
export async function runExample(device: Device, input: SolarShadowMaskExampleInput) {
  const {graph, grid, dem, horizon, sunVisibility} = input;
  // #region example
  const buffer = (id: string, values: Float32Array) =>
    new GPUParameterBuffer(device, {id, format: 'float32', length: values.length, values});
  const sun = getSolarPosition(new Date('2021-08-05T13:45:00Z'), -120.95, 40.15); // 06:45 PDT
  const horizonSettings = buffer('horizon-settings', getGPUTerrainHorizonParameterValues({}, grid));
  const sunSettings = buffer('sun-settings', getGPUSolarShadowMaskParameterValues(sun));
  graph.add(
    new GPUTerrainHorizon({
      grid, // 64 × 64 cells of 120 m, UTM 10N
      elevation: makeGPURasterBand(dem, grid),
      parameters: horizonSettings.importToGraph(graph),
      maximumRadius: 32, // cells; 16 sectors by default
      output: {horizon}
    })
  );
  graph.add(
    new GPUSolarShadowMask({
      grid,
      directionCount: 16,
      horizon,
      parameters: sunSettings.importToGraph(graph),
      output: {sunVisibility} // fraction of the solar disk, 0–1
    })
  );
  // #endregion example
  return {horizonSettings, sunSettings};
}
