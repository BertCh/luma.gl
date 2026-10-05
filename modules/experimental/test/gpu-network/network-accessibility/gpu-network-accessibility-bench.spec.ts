// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPUNetworkAccessibility,
  GPUNetworkCostMatrix,
  GPU_NETWORK_ACCESSIBILITY_PARAMETER_LENGTH,
  encodeGPUNetworkAccessibilityParameters
} from '../../../src/gpu-network/network-accessibility';
import {
  createInputBuffer,
  createOutputBuffer,
  isSoftwareDevice,
  readUint32
} from '../../utils/gpu-contributor-test-utils';
import {buildCSR, createGridFixture, type AccessibilityCSR} from './network-accessibility-oracle';

const MEASURED_FRAMES = 5;

function median(values: number[]): number {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.floor(sorted.length / 2)];
}

type BenchResult = {
  label: string;
  graphNodes: number;
  compileMilliseconds: number;
  encodeMilliseconds: number;
  frameMilliseconds: number;
  converged: number;
};

/** Times one compiled graph: CPU encode and submit-to-readback of a one-word sync buffer. */
async function measureGraph(
  device: Device,
  label: string,
  graph: GPUCommandGraph,
  syncBuffer: Buffer,
  beforeFrame: () => void = () => {}
): Promise<BenchResult> {
  const compileStart = performance.now();
  const compiled = graph.compile();
  const compileMilliseconds = performance.now() - compileStart;
  const encodeTimes: number[] = [];
  const frameTimes: number[] = [];
  let converged = 0;
  for (let frame = 0; frame < MEASURED_FRAMES + 1; frame++) {
    beforeFrame();
    const frameStart = performance.now();
    const encoder = device.createCommandEncoder();
    compiled.encode(encoder, {parameters: undefined});
    const encodeMilliseconds = performance.now() - frameStart;
    device.submit(encoder.finish());
    [converged] = await readUint32(syncBuffer, 1);
    if (frame > 0) {
      encodeTimes.push(encodeMilliseconds);
      frameTimes.push(performance.now() - frameStart);
    }
  }
  const graphNodes = compiled.stats.nodeOrder.length;
  compiled.destroy();
  return {
    label,
    graphNodes,
    compileMilliseconds,
    encodeMilliseconds: median(encodeTimes),
    frameMilliseconds: median(frameTimes),
    converged
  };
}

function createMatrixGraph(
  device: Device,
  csr: AccessibilityCSR,
  nodeCount: number,
  rowNodes: number[],
  laneCount: number,
  maxIterations: number,
  buffers: Buffer[]
): {
  graph: GPUCommandGraph;
  converged: Buffer;
  costs: Buffer;
  nodeCount: number;
} {
  const graph = new GPUCommandGraph(device, {id: `matrix-${laneCount}`});
  const input = (values: Uint32Array | Float32Array) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return buffer;
  };
  const costs = createOutputBuffer(device, rowNodes.length * nodeCount);
  const converged = createOutputBuffer(device, 1);
  buffers.push(costs, converged);
  const recipe = new GPUNetworkCostMatrix({
    id: 'matrix',
    offsets: importGraphBuffer(graph, 'offsets', input(csr.offsets), 'uint32', nodeCount + 1),
    neighbors: importGraphBuffer(
      graph,
      'neighbors',
      input(csr.neighbors),
      'uint32',
      csr.neighbors.length
    ),
    weights: importGraphBuffer(graph, 'weights', input(csr.weights), 'float32', csr.weights.length),
    seedNodes: importGraphBuffer(
      graph,
      'seeds',
      input(Uint32Array.from(rowNodes)),
      'uint32',
      rowNodes.length
    ),
    laneCount,
    maxIterations,
    costs: importGraphBuffer(graph, 'costs', costs, 'float32', rowNodes.length * nodeCount),
    converged: importGraphBuffer(graph, 'converged', converged, 'uint32', 1)
  });
  graph.add(recipe);
  return {graph, converged, costs, nodeCount};
}

function formatBench(title: string, results: BenchResult[]): string {
  return [
    title,
    ...results.map(
      result =>
        `  ${result.label}: ${result.graphNodes} graph nodes, compile ${result.compileMilliseconds.toFixed(1)} ms, ` +
        `encode ${result.encodeMilliseconds.toFixed(2)} ms, frame ${result.frameMilliseconds.toFixed(2)} ms, converged ${result.converged}`
    )
  ].join('\n');
}

it('GPUNetworkAccessibility bench: lane batching, search side, and per-frame scoring', async () => {
  const device = await getWebGPUTestDevice();
  if (!device || isSoftwareDevice(device)) {
    return;
  }
  const width = 48;
  const nodeCount = width * width;
  const csr = buildCSR(nodeCount, createGridFixture(13, width, width).edges);
  const maxIterations = 24;
  const buffers: Buffer[] = [];
  const results: BenchResult[] = [];

  // Opportunity side: 64 opportunity rows over the reverse (here symmetric) network.
  const opportunityRows = Array.from({length: 64}, (_, row) => (row * 389 + 17) % nodeCount);
  let opportunityMatrix: Buffer | undefined;
  for (const laneCount of [1, 8, 32, 64]) {
    const matrix = createMatrixGraph(
      device,
      csr,
      nodeCount,
      opportunityRows,
      laneCount,
      maxIterations,
      buffers
    );
    results.push(
      await measureGraph(
        device,
        `64 opportunity rows, laneCount ${laneCount}`,
        matrix.graph,
        matrix.converged
      )
    );
    opportunityMatrix = matrix.costs;
  }

  // Origin side: one search per node (every node is an origin).
  const originRows = Array.from({length: nodeCount}, (_, node) => node);
  const originMatrix = createMatrixGraph(
    device,
    csr,
    nodeCount,
    originRows,
    64,
    maxIterations,
    buffers
  );
  results.push(
    await measureGraph(
      device,
      `${nodeCount} origin rows, laneCount 64`,
      originMatrix.graph,
      originMatrix.converged
    )
  );
  for (const result of results) {
    expect(result.converged).toBe(1);
  }

  // Per-frame scoring of the retained 64-row matrix: only the parameters change.
  const scoringGraph = new GPUCommandGraph(device, {id: 'scoring'});
  const parameters = new GPUParameterBuffer(device, {
    id: 'parameters',
    format: 'float32',
    length: GPU_NETWORK_ACCESSIBILITY_PARAMETER_LENGTH
  });
  const gravity = createOutputBuffer(device, nodeCount);
  const sync = createOutputBuffer(device, 1);
  const weights = createInputBuffer(device, new Float32Array(64).fill(1));
  buffers.push(gravity, sync, weights);
  scoringGraph.add(
    new GPUNetworkAccessibility({
      id: 'accessibility',
      costs: importGraphBuffer(
        scoringGraph,
        'costs',
        opportunityMatrix!,
        'float32',
        64 * nodeCount
      ),
      opportunityWeights: importGraphBuffer(scoringGraph, 'weights', weights, 'float32', 64),
      parameters: parameters.importToGraph(scoringGraph),
      gravity: importGraphBuffer(scoringGraph, 'gravity', gravity, 'float32', nodeCount)
    })
  );
  let frame = 0;
  const scoring = await measureGraph(
    device,
    'scoring 64 x 2304, new beta every frame',
    scoringGraph,
    sync,
    () =>
      parameters.write(
        encodeGPUNetworkAccessibilityParameters({
          threshold: 40,
          decay: 'exponential',
          beta: 0.05 + 0.01 * frame++
        })
      )
  );
  scoring.converged = 1;
  results.push(scoring);
  // eslint-disable-next-line no-console
  console.warn(formatBench(`network-accessibility bench ${width}x${width} grid`, results));
  parameters.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
}, 120000);
