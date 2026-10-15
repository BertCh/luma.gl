// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUData} from '@luma.gl/gpgpu/gpu-data';
import {createGPUConjugateGradientProgram, GPUProgramCompiler} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, test} from 'vitest';

test('recording preserves convergent conjugate-gradient output, node order and untimed passes', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const buffers: Buffer[] = [];
  const makeData = (values: number[], format: 'uint32' | 'float32') => {
    const buffer = device.createBuffer({
      data: format === 'uint32' ? new Uint32Array(values) : new Float32Array(values),
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    buffers.push(buffer);
    return new GPUData({buffer, format, length: values.length});
  };
  // SPD system [4 1; 1 3] x = [1; 2], which converges in two iterations.
  const solver = createGPUConjugateGradientProgram({
    id: 'parity',
    size: 2,
    nonZeros: 4,
    maxIterations: 6,
    toleranceSquared: 1e-10
  });
  const solution = makeData([0, 0], 'float32');
  const bindings = {
    vectors: {
      [solver.rowOffsets.id]: makeData([0, 2, 4], 'uint32'),
      [solver.columnIndices.id]: makeData([0, 1, 0, 1], 'uint32'),
      [solver.values.id]: makeData([4, 1, 1, 3], 'float32'),
      [solver.rhs.id]: makeData([1, 2], 'float32'),
      [solver.solution.id]: solution
    }
  };
  const compiler = new GPUProgramCompiler(device);
  const production = compiler.compile(solver.program, bindings).graph.compile();
  const recorded = compiler.compile(solver.program, bindings, {recording: {}}).graph.compile();
  try {
    expect(recorded.stats.nodeOrder).toEqual(production.stats.nodeOrder);
    expect(recorded.stats.physicalTransientBufferCount).toBe(
      production.stats.physicalTransientBufferCount
    );
    expect(recorded.stats.physicalTransientBytes).toBe(production.stats.physicalTransientBytes);
    const run = async (compiled: typeof production, coalesceComputePasses: boolean) => {
      solution.buffer.write(new Float32Array([0, 0]));
      const commandEncoder = device.createCommandEncoder({
        measureExecutionTime: false,
        timeProfilingQuerySet: null
      });
      const encoding = compiled.encode(commandEncoder, {
        parameters: undefined,
        coalesceComputePasses
      });
      device.submit(commandEncoder.finish());
      const bytes = await solution.buffer.readAsync();
      expect(encoding.stats.timestampedNodeCount).toBe(0);
      const values = new Float32Array(bytes.buffer, bytes.byteOffset, 2);
      expect(values[0]).toBeCloseTo(1 / 11, 5);
      expect(values[1]).toBeCloseTo(7 / 11, 5);
      return {bytes: Array.from(bytes), passes: encoding.stats.computePassCount};
    };
    for (const coalesceComputePasses of [true, false]) {
      const baseline = await run(production, coalesceComputePasses);
      const instrumented = await run(recorded, coalesceComputePasses);
      expect(instrumented).toEqual(baseline);
      // A second explicitly untimed recorded run introduces no extra compute passes or drift.
      expect(await run(recorded, coalesceComputePasses)).toEqual(baseline);
    }
  } finally {
    production.destroy();
    recorded.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});
