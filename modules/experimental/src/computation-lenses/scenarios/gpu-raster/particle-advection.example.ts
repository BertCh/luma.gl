// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {type GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  type GPURasterMetadata,
  GPUParameterBuffer,
  GPUParticleAdvection,
  GPU_PARTICLE_ADVECTION_PARAMETER_LENGTH,
  GPU_PARTICLE_ADVECTION_UINT32_PARAMETER_LENGTH,
  getGPUParticleAdvectionParameterValues,
  getGPUParticleAdvectionUint32ParameterValues
} from '@luma.gl/experimental/gpu-raster';

/** Graph, wind grid and caller-owned views for particle-advection. */
export type ParticleAdvectionExampleInput = {
  graph: GPUCommandGraph;
  /** Wind grid: 0.25 degree cells, rows north to south. */
  grid: GPURasterMetadata;
  /** Wind in degrees per hour, rows from the south edge as the contributor expects. */
  velocities: GraphDataView<'float32x2'>;
  positions: GraphDataView<'float32x2'>;
  ages: GraphDataView<'uint32'>;
  generations: GraphDataView<'uint32'>;
  speeds: GraphDataView<'float32'>;
  /** Trail ring of 200 particles by 12 slots. */
  trails: GraphDataView<'float32x2'>;
};

/** Adds the recorded particle-advection computation; returns its owned parameter buffers. */
export async function runExample(device: Device, input: ParticleAdvectionExampleInput) {
  const {graph, grid, velocities, positions, ages, generations, speeds, trails} = input;
  // Prerequisite: one uint32 word buffer per step, which differs only in the frame index (the
  // trail slot). Seed 1, no ageing, so a particle only respawns by leaving the field.
  const frameWords = Array.from(
    {length: 12},
    (_, frame) =>
      new GPUParameterBuffer(device, {
        id: `advection-frame-${frame}`,
        format: 'uint32',
        length: GPU_PARTICLE_ADVECTION_UINT32_PARAMETER_LENGTH,
        values: getGPUParticleAdvectionUint32ParameterValues({seed: 1, frame, maximumAge: 0})
      })
  );
  const frameViews = frameWords.map(words => words.importToGraph(graph));
  // The grid is north-up; the contributor counts field rows from the southern edge.
  const [west, cell] = [grid.affine[2], grid.affine[0]];
  const south = grid.affine[5] - grid.height * cell;
  // #region example
  const extent = [west, south, cell, cell] as const; // degrees; the field's first row is the south
  const parameters = new GPUParameterBuffer(device, {
    id: 'advection-parameters',
    format: 'float32',
    length: GPU_PARTICLE_ADVECTION_PARAMETER_LENGTH,
    values: getGPUParticleAdvectionParameterValues(
      {fieldExtent: extent, timeStep: 2}, // 2 hours per step
      [grid.width, grid.height]
    )
  });
  const field = parameters.importToGraph(graph);
  for (let frame = 0; frame < 12; frame++) {
    graph.add(
      new GPUParticleAdvection({
        id: `advection-${frame}`,
        velocities, // degrees per hour
        fieldWidth: grid.width,
        fieldHeight: grid.height,
        parameters: field,
        uint32Parameters: frameViews[frame], // seed 1, this step's frame index
        state: {positions, ages, generations}, // 200 particles, updated in place
        output: {speeds, trails: {positions: trails, length: 12}} // one trail slot per step
      })
    );
  }
  // #endregion example
  return {parameters: [parameters, ...frameWords]};
}
