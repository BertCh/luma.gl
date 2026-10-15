// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createGPUConjugateGradientProgram,
  GPUProgramCompiler,
  type GPUCommandGraphRecordingOptions
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUData} from '@luma.gl/gpgpu/gpu-data';

/** Uploaded CSR system `(I + 4 L) x = elevation` and the device that lowers the solver. */
export type ConjugateGradientExampleInput = {
  device: Device;
  recording: GPUCommandGraphRecordingOptions;
  rowOffsets: GPUData;
  columnIndices: GPUData;
  values: GPUData;
  elevation: GPUData;
  smoothed: GPUData;
};

/**
 * Builds the CSR matrix `I + lambda * L` of the 5-point graph Laplacian `L` with zero-flux
 * (Neumann) boundaries: a cell's missing neighbours are dropped, so each row of `L` sums to zero.
 */
export function createScreenedPoissonSystem(side: number, lambda: number) {
  const rowOffsets = [0];
  const columnIndices: number[] = [];
  const values: number[] = [];
  for (let row = 0; row < side; row++) {
    for (let column = 0; column < side; column++) {
      const index = row * side + column;
      const neighbours = [
        ...(row > 0 ? [index - side] : []),
        ...(column > 0 ? [index - 1] : []),
        ...(column < side - 1 ? [index + 1] : []),
        ...(row < side - 1 ? [index + side] : [])
      ];
      // Columns stay ascending: up, left, the diagonal itself, right, down.
      for (const neighbour of [
        ...neighbours.filter(n => n < index),
        index,
        ...neighbours.filter(n => n > index)
      ]) {
        columnIndices.push(neighbour);
        values.push(neighbour === index ? 1 + lambda * neighbours.length : -lambda);
      }
      rowOffsets.push(columnIndices.length);
    }
  }
  return {
    rowOffsets: Uint32Array.from(rowOffsets),
    columnIndices: Uint32Array.from(columnIndices),
    values: Float32Array.from(values)
  };
}

/** Lowers the solver on the given device; the shared harness owns execution and readback. */
export async function runExample(
  device: Device,
  {recording, rowOffsets, columnIndices, values, elevation, smoothed}: ConjugateGradientExampleInput
) {
  // #region example
  const solver = createGPUConjugateGradientProgram({
    id: 'smooth',
    size: 256,
    nonZeros: columnIndices.length,
    maxIterations: 12,
    toleranceSquared: 35000
  });
  const compilation = new GPUProgramCompiler<undefined>(device).compile(
    solver.program,
    {
      vectors: {
        [solver.rowOffsets.id]: rowOffsets,
        [solver.columnIndices.id]: columnIndices,
        [solver.values.id]: values,
        [solver.rhs.id]: elevation,
        [solver.solution.id]: smoothed
      }
    },
    {recording}
  );
  // smoothed solves (I + 4 L) smoothed = elevation: a diffusion-smoothed 16 × 16 elevation block, in m.
  return {compilation, solver};
  // #endregion example
}
