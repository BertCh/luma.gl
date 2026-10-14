// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPUNetworkCostMatrix} from '../../../src/gpu-network/network-accessibility';
import {
  createInputBuffer,
  createOutputBuffer,
  isSoftwareDevice,
  readUint32
} from '../../utils/gpu-contributor-test-utils';
import {buildCSR, createGridFixture} from './network-accessibility-oracle';

const MEASURED_FRAMES = 5;

function median(values: number[]): number {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.floor(sorted.length / 2)];
}

/** Builds a cost matrix graph, dense when `targetCount` is 0 and compact otherwise. */
function createGraph(
  device: Device,
  buffers: Buffer[],
  width: number,
  rowCount: number,
  targetCount: number,
  laneCount: number
) {
  const nodeCount = width * width;
  const csr = buildCSR(nodeCount, createGridFixture(13, width, width).edges);
  const graph = new GPUCommandGraph(device, {id: `matrix-bench-${targetCount}`});
  const input = (values: Uint32Array | Float32Array) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return buffer;
  };
  const columnCount = targetCount || nodeCount;
  const costs = createOutputBuffer(device, rowCount * columnCount);
  const converged = createOutputBuffer(device, 1);
  buffers.push(costs, converged);
  const starts = Uint32Array.from({length: rowCount}, (_, row) => (row * 389 + 17) % nodeCount);
  const targets = Uint32Array.from({length: targetCount}, (_, t) => (t * 97 + 3) % nodeCount);
  graph.add(
    new GPUNetworkCostMatrix({
      id: 'matrix',
      topology: {
        offsets: importGraphBuffer(graph, 'offsets', input(csr.offsets), 'uint32', nodeCount + 1),
        neighbors: importGraphBuffer(
          graph,
          'neighbors',
          input(csr.neighbors),
          'uint32',
          csr.neighbors.length
        ),
        weights: importGraphBuffer(
          graph,
          'weights',
          input(csr.weights),
          'float32',
          csr.weights.length
        )
      },
      startNodes: importGraphBuffer(graph, 'starts', input(starts), 'uint32', rowCount),
      targetNodes: targetCount
        ? importGraphBuffer(graph, 'targets', input(targets), 'uint32', targetCount)
        : undefined,
      laneCount,
      maximumIterations: 24,
      output: {
        costs: importGraphBuffer(graph, 'costs', costs, 'float32', rowCount * columnCount),
        status: {converged: importGraphBuffer(graph, 'converged', converged, 'uint32', 1)}
      }
    })
  );
  return {graph, converged};
}

/** A/B: the dense matrix against the compact `targetNodes` matrix; frame = submit to readback. */
it('GPUNetworkCostMatrix bench: dense matrix vs targetNodes', async () => {
  const device = await getWebGPUTestDevice();
  if (!device || isSoftwareDevice(device)) {
    return;
  }
  const width = 100;
  const rowCount = 1024;
  const laneCount = 64;
  const buffers: Buffer[] = [];
  const lines: string[] = [];
  for (const targetCount of [0, 100]) {
    const {graph, converged} = createGraph(
      device,
      buffers,
      width,
      rowCount,
      targetCount,
      laneCount
    );
    const compiled = graph.compile();
    const frameTimes: number[] = [];
    let flag = 0;
    for (let frame = 0; frame < MEASURED_FRAMES + 1; frame++) {
      const start = performance.now();
      const encoder = device.createCommandEncoder();
      compiled.encode(encoder, {parameters: undefined});
      device.submit(encoder.finish());
      [flag] = await readUint32(converged, 1);
      if (frame > 0) {
        frameTimes.push(performance.now() - start);
      }
    }
    expect(flag).toBe(1);
    const outputMegabytes = (rowCount * (targetCount || width * width) * 4) / 1e6;
    lines.push(
      `  ${targetCount ? `${targetCount} targets` : 'dense'}: output ${outputMegabytes.toFixed(1)} MB, ` +
        `${compiled.stats.nodeOrder.length} graph nodes, frame ${median(frameTimes).toFixed(2)} ms`
    );
    compiled.destroy();
  }
  // eslint-disable-next-line no-console
  console.warn(
    `cost matrix bench ${rowCount} rows, ${width}x${width} grid, laneCount ${laneCount}\n${lines.join('\n')}`
  );
  for (const buffer of buffers) {
    buffer.destroy();
  }
}, 180000);
