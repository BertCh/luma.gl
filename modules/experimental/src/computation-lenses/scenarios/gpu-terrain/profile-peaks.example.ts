// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUParameterBuffer,
  GPUProfilePeaks,
  getGPUProfilePeaksParameterValues,
  GPU_PROFILE_PEAKS_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-terrain';

/** Graph and caller-owned views the example wires together. */
export type ProfilePeaksExampleInput = {
  graph: GPUCommandGraph;
  /** Elevation samples along one profile line, metres. */
  profile: GraphDataView<'float32'>;
  /** CSR span offsets: `[0, sampleCount]` for one profile. */
  offsets: GraphDataView<'uint32'>;
  prominence: GraphDataView<'float32'>;
  refinedIndex: GraphDataView<'float32'>;
  refinedValue: GraphDataView<'float32'>;
  peakMask: GraphDataView<'uint32'>;
  converged: GraphDataView<'uint32'>;
};

/** Adds profile peak detection on `profile` to `graph`; returns the parameter buffer to destroy. */
export async function runExample(device: Device, input: ProfilePeaksExampleInput) {
  const {graph, profile, offsets, prominence, refinedIndex, refinedValue, peakMask} = input;
  const {converged} = input;
  // #region example
  const parameters = new GPUParameterBuffer(device, {
    id: 'profile-peaks-parameters',
    format: 'float32',
    length: GPU_PROFILE_PEAKS_PARAMETER_LENGTH,
    values: getGPUProfilePeaksParameterValues({minimumProminence: 20}) // m
  });
  graph.add(
    new GPUProfilePeaks({
      values: profile, // 64 elevation samples along row 40
      offsets,
      parameters: parameters.importToGraph(graph),
      window: 8, // samples
      minimumSide: 4,
      output: {prominence, refinedIndex, refinedValue, peakMask, status: {converged}}
    })
  );
  // #endregion example
  return {parameters};
}
