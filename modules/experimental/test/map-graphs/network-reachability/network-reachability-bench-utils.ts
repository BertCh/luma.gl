// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../../src/map-graphs';
import type {GPUNetworkReachabilityProps} from '../../../src/map-graphs/network-reachability';
import {createInputBuffer, createOutputBuffer, readUint32} from '../map-graph-test-utils';
import {buildCSR, type NetworkEdge} from './network-reachability-oracle';

/** Recipe class under measurement: the current one or a baseline copy. */
export type ReachabilityRecipeClass = new (
  props: GPUNetworkReachabilityProps
) => {
  getCommandNodes: (graph: GPUCommandGraph) => readonly unknown[];
};

/** One measured configuration. */
export type ReachabilityBenchResult = {
  label: string;
  graphNodes: number;
  compileMilliseconds: number;
  encodeMilliseconds: number;
  submitToReadbackMilliseconds: number;
  rounds: number;
  converged: number;
};

const WARMUP_FRAMES = 2;
const MEASURED_FRAMES = 10;

function median(values: number[]): number {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.floor(sorted.length / 2)];
}

/**
 * Compiles one reachability graph on `edges`, then times `MEASURED_FRAMES` encodes (CPU wall time of
 * `compiled.encode`) and submit-to-readback round trips. Reports the final round count and flag.
 */
export async function measureReachability(
  device: Device,
  RecipeClass: ReachabilityRecipeClass,
  options: {
    label: string;
    nodeCount: number;
    edges: readonly NetworkEdge[];
    sources: number[];
    maxIterations: number;
    localIterations?: number;
    /** Also compute predecessors, which adds the tie-level phase. */
    predecessors?: boolean;
    maxTieIterations?: number;
  }
): Promise<ReachabilityBenchResult> {
  const {nodeCount} = options;
  const csr = buildCSR(nodeCount, options.edges);
  const graph = new GPUCommandGraph(device, {id: `bench-${options.label}`});
  const buffers = [
    createInputBuffer(device, csr.offsets),
    createInputBuffer(device, csr.neighbors),
    createInputBuffer(device, csr.weights),
    createInputBuffer(device, Uint32Array.from(options.sources)),
    createOutputBuffer(device, nodeCount),
    createOutputBuffer(device, 1),
    createOutputBuffer(device, 1),
    createOutputBuffer(device, nodeCount)
  ];
  const [offsets, neighbors, weights, sources, costs, converged, iterationCount, predecessors] =
    buffers;
  const props: GPUNetworkReachabilityProps = {
    id: 'bench',
    offsets: importGraphBuffer(graph, 'offsets', offsets, 'uint32', nodeCount + 1),
    neighbors: importGraphBuffer(graph, 'neighbors', neighbors, 'uint32', csr.neighbors.length),
    weights: importGraphBuffer(graph, 'weights', weights, 'float32', csr.weights.length),
    sources: importGraphBuffer(graph, 'sources', sources, 'uint32', options.sources.length),
    costs: importGraphBuffer(graph, 'costs', costs, 'float32', nodeCount),
    converged: importGraphBuffer(graph, 'converged', converged, 'uint32', 1),
    iterationCount: importGraphBuffer(graph, 'iteration-count', iterationCount, 'uint32', 1),
    maxIterations: options.maxIterations,
    ...(options.predecessors
      ? {
          predecessors: importGraphBuffer(graph, 'predecessors', predecessors, 'uint32', nodeCount),
          ...(options.maxTieIterations === undefined
            ? {}
            : {maxTieIterations: options.maxTieIterations})
        }
      : {}),
    ...(options.localIterations === undefined ? {} : {localIterations: options.localIterations})
  };
  graph.add(new RecipeClass(props) as never);
  const compileStart = performance.now();
  const compiled = graph.compile();
  const compileMilliseconds = performance.now() - compileStart;
  const encodeTimes: number[] = [];
  const roundTripTimes: number[] = [];
  for (let frame = 0; frame < WARMUP_FRAMES + MEASURED_FRAMES; frame++) {
    const commandEncoder = device.createCommandEncoder({id: 'bench-encoder'});
    const encodeStart = performance.now();
    compiled.encode(commandEncoder, {parameters: undefined});
    const encodeEnd = performance.now();
    device.submit(commandEncoder.finish());
    await costs.readAsync(0, 4);
    const roundTripEnd = performance.now();
    if (frame >= WARMUP_FRAMES) {
      encodeTimes.push(encodeEnd - encodeStart);
      roundTripTimes.push(roundTripEnd - encodeEnd);
    }
  }
  const result: ReachabilityBenchResult = {
    label: options.label,
    graphNodes: compiled.stats.nodeOrder.length,
    compileMilliseconds,
    encodeMilliseconds: median(encodeTimes),
    submitToReadbackMilliseconds: median(roundTripTimes),
    rounds: (await readUint32(iterationCount, 1))[0],
    converged: (await readUint32(converged, 1))[0]
  };
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

/** Formats results as an aligned text table for the test log. */
export function formatReachabilityBench(title: string, results: ReachabilityBenchResult[]): string {
  const rows = results.map(
    result =>
      `${result.label.padEnd(28)} nodes ${String(result.graphNodes).padStart(5)}  compile ${result.compileMilliseconds.toFixed(0).padStart(6)} ms  encode ${result.encodeMilliseconds.toFixed(2).padStart(7)} ms  submit+read ${result.submitToReadbackMilliseconds.toFixed(2).padStart(8)} ms  rounds ${String(result.rounds).padStart(5)}  converged ${result.converged}`
  );
  return `${title}\n${rows.join('\n')}`;
}
