// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, test} from 'vitest';
import {importGraphBuffer} from '../../examples/deck/spatial-analysis-showcase/src/engine/graph-buffers';
import {
  FrontierResidualDiffusion,
  getFrontierResidualDiffusionParameterValues
} from '../../examples/deck/spatial-analysis-showcase/src/scenes/raster/frontier-neural';

test('cloud repair converges without freezing infill or changing observed pixels', async () => {
  const device = await getWebGPUTestDevice('core');
  if (!device) return;
  const width = 8;
  const height = 8;
  const state = new Float32Array(width * height * 4);
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      const cell = row * width + column;
      if (column === 0 || row === 0 || column === width - 1 || row === height - 1) {
        state.set([0.2, -0.1, 0.05, 1], cell * 4);
      }
    }
  }
  const buffers: Buffer[] = [];
  const upload = (data: Float32Array | Uint32Array) => {
    const buffer = device.createBuffer({
      data,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    buffers.push(buffer);
    return buffer;
  };
  const states = [upload(state), upload(state)];
  const prior = upload(new Float32Array(width * height * 3).fill(0.5));
  const display = upload(new Uint32Array(width * height));
  const parameters = upload(getFrontierResidualDiffusionParameterValues({stepScale: 1}));
  const graphs = states.map((input, index) => {
    const graph = new GPUCommandGraph(device);
    graph.add(
      new FrontierResidualDiffusion({
        width,
        height,
        state: importGraphBuffer(graph, 'state', input, 'float32'),
        nextState: importGraphBuffer(graph, 'next', states[1 - index], 'float32'),
        prior: importGraphBuffer(graph, 'prior', prior, 'float32'),
        display: importGraphBuffer(graph, 'display', display, 'uint32'),
        parameters: importGraphBuffer(graph, 'parameters', parameters, 'float32')
      })
    );
    return graph.compile();
  });
  try {
    const commandEncoder = device.createCommandEncoder();
    for (let iteration = 0; iteration < 256; iteration++)
      graphs[iteration % 2].encode(commandEncoder, {parameters: undefined});
    device.submit(commandEncoder.finish());
    const bytes = await states[0].readAsync();
    const result = new Float32Array(bytes.buffer, bytes.byteOffset, state.length);
    for (let cell = 0; cell < width * height; cell++) {
      expect(result[cell * 4]).toBeCloseTo(0.2, 5);
      expect(result[cell * 4 + 1]).toBeCloseTo(-0.1, 5);
      expect(result[cell * 4 + 2]).toBeCloseTo(0.05, 5);
      expect(result[cell * 4 + 3]).toBe(state[cell * 4 + 3]);
    }
    // Converged infill must still respond when an observation changes.
    const updated = result.slice();
    updated.set([0.4, -0.1, 0.05, 1], 4);
    states[0].write(updated);
    const nextEncoder = device.createCommandEncoder();
    graphs[0].encode(nextEncoder, {parameters: undefined});
    device.submit(nextEncoder.finish());
    const nextBytes = await states[1].readAsync();
    const next = new Float32Array(nextBytes.buffer, nextBytes.byteOffset, state.length);
    expect(next[(width + 1) * 4]).toBeGreaterThan(0.24);
  } finally {
    for (const graph of graphs) graph.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}, 30_000);
