// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {decodeGPUNetworkStatistics} from '../../../src/gpu-network/network-statistics';
import fixture from '../../oracles/fixtures/network-statistics.json';
import type {NetworkEdge} from '../network-reachability/network-reachability-oracle';

/** One networkx statistics scene (see `test/oracles/network-statistics.py`). */
export type NetworkStatisticsReferenceScene = {
  directed: boolean;
  nodeCount: number;
  /** Edges listed once, `[u, v]`; `u === v` is a self-loop. */
  edges: [number, number][];
  communities: number[];
  resolution: number;
  /** networkx degree (undirected, a self-loop adds 2) or out-degree per node. */
  outDegrees: number[];
  /** networkx in-degree per node (equal to `outDegrees` when undirected). */
  inDegrees: number[];
  componentCount: number;
  largestComponentSize: number;
  isolatedVertexCount: number;
  edgeCount: number;
  selfLoopCount: number;
  modularity: number;
  tolerance: {modularityAbsolute: number; counts: 'exact'};
};

/** The whole fixture. */
export type NetworkStatisticsReferenceFixture = {
  generator: string;
  versions: Record<string, string>;
  scenes: Record<string, NetworkStatisticsReferenceScene>;
};

/** The committed networkx fixture. */
export const NETWORK_STATISTICS_REFERENCE = fixture as unknown as NetworkStatisticsReferenceFixture;

/**
 * CSR edge list of a scene: an undirected edge becomes two slots, a self-loop one slot (the
 * `GPUNetworkStatistics` convention), a directed edge one slot.
 */
export function getNetworkStatisticsReferenceEdges(
  scene: NetworkStatisticsReferenceScene
): NetworkEdge[] {
  return scene.edges.flatMap(([from, to]): NetworkEdge[] =>
    scene.directed || from === to
      ? [[from, to, 1]]
      : [
          [from, to, 1],
          [to, from, 1]
        ]
  );
}

/**
 * Compares a raw `GPUNetworkStatistics` summary (32 linear bins of width 1) with the networkx
 * values of a scene. Returns the first violation as text, or `undefined`.
 */
export function compareNetworkStatistics(
  scene: NetworkStatisticsReferenceScene,
  words: ArrayLike<number>
): string | undefined {
  const binCount = 32;
  const decoded = decodeGPUNetworkStatistics(words, {degreeBinCount: binCount});
  const histogram = (degrees: number[]) => {
    const bins = new Array<number>(binCount).fill(0);
    for (const degree of degrees) {
      bins[Math.min(degree, binCount - 1)]++;
    }
    return bins;
  };
  const totals = scene.outDegrees.map((degree, node) =>
    scene.directed ? degree + scene.inDegrees[node] : degree
  );
  const expected: Record<string, unknown> = {
    liveNodeCount: scene.nodeCount,
    liveEdgeCount: scene.edgeCount,
    selfLoopSlotCount: scene.selfLoopCount,
    componentCount: scene.componentCount,
    largestComponentSize: scene.largestComponentSize,
    componentsConverged: true,
    isolatedNodeCount: scene.isolatedVertexCount,
    maximumOutDegree: Math.max(...scene.outDegrees),
    maximumInDegree: Math.max(...scene.inDegrees),
    maximumTotalDegree: Math.max(...totals),
    modularityValid: true,
    outDegreeHistogram: histogram(scene.outDegrees),
    inDegreeHistogram: histogram(scene.inDegrees),
    totalDegreeHistogram: histogram(totals)
  };
  for (const [key, value] of Object.entries(expected)) {
    const actual = decoded[key as keyof typeof decoded];
    if (JSON.stringify(actual) !== JSON.stringify(value)) {
      return `${key}: expected ${JSON.stringify(value)}, got ${JSON.stringify(actual)}`;
    }
  }
  if (Math.abs(decoded.modularity - scene.modularity) > scene.tolerance.modularityAbsolute) {
    return `modularity: expected ${scene.modularity}, got ${decoded.modularity}`;
  }
  return undefined;
}
