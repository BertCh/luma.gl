// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPUNetworkCentrality} from '../../../src/gpu-network/network-centrality/index';
import {
  createInputBuffer,
  createOutputBuffer,
  isSoftwareDevice,
  readFloat32,
  readUint32
} from '../../utils/gpu-contributor-test-utils';
import {
  buildCentralityCSRs,
  computeBrandesBetweenness,
  createCentralityGridEdges,
  getCentralityGridMaximumIterations
} from './network-centrality-oracle';

const MEASURED_FRAMES = 3;

function median(values: number[]): number {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.floor(sorted.length / 2)];
}

/** Times a compiled centrality graph (CPU encode, submit and readback of the converged flag). */
async function measureCentrality(
  device: Device,
  width: number,
  pivots: readonly number[],
  options: {laneCount?: number} = {}
): Promise<{
  frameMilliseconds: number;
  graphNodes: number;
  converged: number;
  betweenness: Float32Array;
}> {
  const nodeCount = width * width;
  const edges = createCentralityGridEdges(width, 1);
  const {forward, reverse} = buildCentralityCSRs(nodeCount, edges);
  const graph = new GPUCommandGraph(device, {id: `centrality-bench-${width}`});
  const buffers: Buffer[] = [];
  const input = (values: Uint32Array | Float32Array) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return buffer;
  };
  const topologyViews = (name: string, csr: typeof forward) => ({
    offsets: importGraphBuffer(
      graph,
      `${name}-offsets`,
      input(csr.offsets),
      'uint32',
      nodeCount + 1
    ),
    neighbors: importGraphBuffer(
      graph,
      `${name}-neighbors`,
      input(csr.neighbors),
      'uint32',
      csr.neighbors.length
    ),
    weights: importGraphBuffer(
      graph,
      `${name}-weights`,
      input(csr.weights),
      'float32',
      csr.weights.length
    )
  });
  const betweennessBuffer = createOutputBuffer(device, nodeCount);
  const convergedBuffer = createOutputBuffer(device, 1);
  buffers.push(betweennessBuffer, convergedBuffer);
  graph.add(
    new GPUNetworkCentrality({
      id: 'centrality',
      topology: topologyViews('forward', forward),
      reverseTopology: topologyViews('reverse', reverse),
      pivotNodes: importGraphBuffer(
        graph,
        'pivots',
        input(Uint32Array.from(pivots)),
        'uint32',
        pivots.length
      ),
      laneCount: options.laneCount,
      // Sigma and delta advance one tight edge per dispatch, independently of localIterations.
      maximumIterations: getCentralityGridMaximumIterations(width, edges),
      betweennessScale: nodeCount / pivots.length,
      output: {
        betweenness: importGraphBuffer(
          graph,
          'betweenness',
          betweennessBuffer,
          'float32',
          nodeCount
        ),
        status: {converged: importGraphBuffer(graph, 'converged', convergedBuffer, 'uint32', 1)}
      }
    })
  );
  const compiled = graph.compile();
  const frameTimes: number[] = [];
  let converged = 0;
  for (let frame = 0; frame < MEASURED_FRAMES + 1; frame++) {
    const start = performance.now();
    const encoder = device.createCommandEncoder();
    compiled.encode(encoder, {parameters: undefined});
    device.submit(encoder.finish());
    [converged] = await readUint32(convergedBuffer, 1);
    if (frame > 0) {
      frameTimes.push(performance.now() - start);
    }
  }
  const betweenness = Float32Array.from(await readFloat32(betweennessBuffer, nodeCount));
  const graphNodes = compiled.stats.nodeOrder.length;
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return {frameMilliseconds: median(frameTimes), graphNodes, converged, betweenness};
}

/** Times the CPU Brandes speed bar, best of two runs. */
function measureBrandes(width: number, pivots: readonly number[]): {milliseconds: number} {
  const nodeCount = width * width;
  const {forward} = buildCentralityCSRs(nodeCount, createCentralityGridEdges(width, 1));
  let best = Infinity;
  for (let run = 0; run < 2; run++) {
    const start = performance.now();
    computeBrandesBetweenness(forward, nodeCount, pivots);
    best = Math.min(best, performance.now() - start);
  }
  return {milliseconds: best};
}

it('GPUNetworkCentrality bench: all-pivot and sampled betweenness against CPU Brandes', async () => {
  const device = await getWebGPUTestDevice();
  if (!device || isSoftwareDevice(device)) {
    return;
  }
  const lines: string[] = ['GPUNetworkCentrality bench (frame = encode + submit + readback)'];
  for (const [label, width, pivotCount] of [
    ['all pivots, 64 x 64 grid', 64, 64 * 64],
    ['256 sampled pivots, 224 x 224 grid', 224, 256]
  ] as const) {
    const nodeCount = width * width;
    const stride = Math.floor(nodeCount / pivotCount);
    const pivots = Array.from({length: pivotCount}, (_, row) => row * stride);
    const gpu = await measureCentrality(device, width, pivots);
    const cpu = measureBrandes(width, pivots);
    lines.push(
      `  ${label}: GPU ${gpu.frameMilliseconds.toFixed(1)} ms (${gpu.graphNodes} graph nodes, converged ${gpu.converged}), ` +
        `CPU Brandes ${cpu.milliseconds.toFixed(1)} ms, speedup ${(cpu.milliseconds / gpu.frameMilliseconds).toFixed(2)}x`
    );
    expect(gpu.converged).toBe(1);
  }
  console.log(lines.join('\n'));
});
