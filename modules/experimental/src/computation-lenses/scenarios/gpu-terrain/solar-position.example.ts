// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPUSolarPositionParameterValues,
  GPUParameterBuffer,
  GPUSolarPosition,
  GPU_SOLAR_POSITION_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-terrain';

/** Graph and caller-owned views the example wires together. */
export type SolarPositionExampleInput = {
  graph: GPUCommandGraph;
  /** One `[longitude, latitude]` row per site, degrees. */
  positions: GraphDataView<'float32x2'>;
  azimuth: GraphDataView<'float32'>;
  altitude: GraphDataView<'float32'>;
  daylight: GraphDataView<'uint32'>;
};

/** Adds the sun position of every site at one instant to `graph`; returns the buffer to destroy. */
export async function runExample(device: Device, input: SolarPositionExampleInput) {
  const {graph, positions, azimuth, altitude, daylight} = input;
  // #region example
  const parameters = new GPUParameterBuffer(device, {
    id: 'sun-parameters',
    format: 'float32',
    length: GPU_SOLAR_POSITION_PARAMETER_LENGTH,
    values: getGPUSolarPositionParameterValues({timestamp: new Date('2021-08-05T20:00:00Z')})
  });
  graph.add(
    new GPUSolarPosition({
      positions, // longitude, latitude per site
      parameters: parameters.importToGraph(graph),
      output: {azimuth, altitude, daylight} // degrees from north, degrees, 0 or 1
    })
  );
  // #endregion example
  return {parameters};
}
