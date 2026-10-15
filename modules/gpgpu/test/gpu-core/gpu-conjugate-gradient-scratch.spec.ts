// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUData} from '@luma.gl/gpgpu/gpu-data';
import {GPUProgramCompiler, createGPUConjugateGradientProgram} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, test} from 'vitest';

test.each([
  1, 3
])('257-row CG executes %s iterations with independent reduction scratch', async maximumIterations => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const size = 257;
  const solver = createGPUConjugateGradientProgram({
    id: 'scratch-regression',
    size,
    nonZeros: size,
    maxIterations: maximumIterations,
    toleranceSquared: 1e-10
  });
  // Two distinct positive eigenvalues require two CG iterations from a zero initial solution.
  const diagonal = Float32Array.from({length: size}, (_, index) => 1 + (index % 2));
  const arrays = [
    Uint32Array.from({length: size + 1}, (_, index) => index),
    Uint32Array.from({length: size}, (_, index) => index),
    diagonal,
    new Float32Array(size).fill(1),
    new Float32Array(size)
  ];
  const inputs = [
    solver.rowOffsets,
    solver.columnIndices,
    solver.values,
    solver.rhs,
    solver.solution
  ];
  const buffers = arrays.map(data =>
    device.createBuffer({
      data,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    })
  );
  const data = inputs.map(
    (input, index) =>
      new GPUData({
        buffer: buffers[index],
        ownsBuffer: true,
        format: input.format,
        length: input.length
      })
  );
  try {
    const compilation = new GPUProgramCompiler(device).compile(solver.program, {
      vectors: Object.fromEntries(inputs.map((input, index) => [input.id, data[index]]))
    });
    const compiled = compilation.graph.compile();
    try {
      // Re-encoding the same graph must not reuse stale partial sums from the previous solve.
      for (let execution = 0; execution < 2; execution++) {
        buffers[4].write(new Float32Array(size));
        const encoder = device.createCommandEncoder();
        compiled.encode(encoder, {parameters: undefined});
        device.submit(encoder.finish());
        const bytes = await buffers[4].readAsync();
        const solution = new Float32Array(bytes.buffer, bytes.byteOffset, size);
        const firstStep = size / diagonal.reduce((total, value) => total + value, 0);
        for (let index = 0; index < size; index++) {
          // CPU arithmetic is only the oracle for the GPU solve.
          const expected = maximumIterations === 1 ? firstStep : 1 / diagonal[index];
          expect(solution[index]).toBeCloseTo(expected, 5);
        }
      }
    } finally {
      compiled.destroy();
    }
  } finally {
    for (const input of data) input.destroy();
  }
});
