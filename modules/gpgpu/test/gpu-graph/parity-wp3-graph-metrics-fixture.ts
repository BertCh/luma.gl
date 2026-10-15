// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/// <reference types="vite/client" />

import {GPUGraph, GPUGraphTopology} from '@luma.gl/gpgpu/gpu-graph';
import {createParityResources} from '../../../experimental/test/gpu-network/parity-wp3-round2-resources';
import {decodeOracleFixture} from '../../../experimental/test/oracles/oracle-fixture';
import syntheticDirected from '../../../experimental/test/oracles/fixtures/parity-wp3-graph-metrics-synthetic-directed.json?raw';
import syntheticUndirected from '../../../experimental/test/oracles/fixtures/parity-wp3-graph-metrics-synthetic-undirected.json?raw';
import flightsDirected from '../../../experimental/test/oracles/fixtures/parity-wp3-graph-metrics-openflights-directed.json?raw';
import flightsUndirected from '../../../experimental/test/oracles/fixtures/parity-wp3-graph-metrics-openflights-undirected.json?raw';
import isolates from '../../../experimental/test/oracles/fixtures/parity-wp3-graph-metrics-isolates.json?raw';

export type GraphMetricsFixture = {
  case: string;
  reference: string;
  versions: Record<string, string>;
  nodeCount: number;
  edges: [number, number, number][];
  directed: boolean;
  cores: number[];
  degeneracy: number;
  outgoing: number[];
  incoming: number[];
  coreIterations: number;
  rankCases: {damping: number; iterations: number; ranks: number[]}[];
};
// Native JSON-module transforms can perturb f64 literals by one ulp (Vite 8 observed).
// Parse source text to preserve the reference bits, especially exact f32 input coordinates.
export const graphMetricsFixtures = [
  syntheticDirected,
  syntheticUndirected,
  flightsDirected,
  flightsUndirected,
  isolates
].map(value => decodeOracleFixture<GraphMetricsFixture>(JSON.parse(value)));

/** Allocation only; topology is built on GPU from every original edge occurrence. */
export function createMetricsTopology(
  resources: ReturnType<typeof createParityResources>,
  fixture: Pick<GraphMetricsFixture, 'nodeCount' | 'edges' | 'directed'>,
  weighted = true
): GPUGraphTopology {
  const graph = new GPUGraph({
    vertexCount: fixture.nodeCount,
    directed: fixture.directed,
    sourceVertices: resources.vector(
      'sources',
      'uint32',
      fixture.edges.map(edge => edge[0])
    ),
    targetVertices: resources.vector(
      'targets',
      'uint32',
      fixture.edges.map(edge => edge[1])
    ),
    edgeWeights: weighted
      ? resources.vector(
          'weights',
          'float32',
          fixture.edges.map(edge => edge[2])
        )
      : undefined
  });
  const createAdjacency = (name: string) => ({
    offsets: resources.vector(
      `${name}-offsets`,
      'uint32',
      new Array(fixture.nodeCount + 1).fill(0)
    ),
    neighbors: resources.vector(
      `${name}-neighbors`,
      'uint32',
      new Array(2 * fixture.edges.length).fill(0)
    ),
    edgeIds: resources.vector(
      `${name}-edgeIds`,
      'uint32',
      new Array(2 * fixture.edges.length).fill(0)
    ),
    edgeWeights: weighted
      ? resources.vector(`${name}-weights`, 'float32', new Array(2 * fixture.edges.length).fill(0))
      : undefined,
    count: resources.vector(`${name}-count`, 'uint32', [0]),
    overflow: resources.vector(`${name}-overflow`, 'uint32', [0])
  });
  const topology = new GPUGraphTopology({
    graph,
    forward: createAdjacency('forward'),
    reverse: fixture.directed ? createAdjacency('reverse') : undefined,
    invalidEdgeCount: resources.vector('invalid', 'uint32', [0])
  });
  topology.addToGraph(resources.graph);
  return topology;
}

/** Stochastic-matrix definition; independent residual evaluation, not a PageRank solver. */
export function evaluateRankResidual(
  fixture: Pick<GraphMetricsFixture, 'nodeCount' | 'edges' | 'directed'>,
  ranks: readonly number[],
  damping: number
): number[] {
  const outgoing = Array.from({length: fixture.nodeCount}, () => [] as number[]);
  for (const [source, target] of fixture.edges) {
    outgoing[source].push(target);
    if (!fixture.directed && source !== target) outgoing[target].push(source);
  }
  const dangling = ranks.reduce((sum, value, node) => sum + (outgoing[node].length ? 0 : value), 0);
  const next = new Array<number>(fixture.nodeCount).fill(
    (1 - damping + damping * dangling) / fixture.nodeCount
  );
  outgoing.forEach((neighbors, source) => {
    for (const target of neighbors) next[target] += (damping * ranks[source]) / neighbors.length;
  });
  return next.map((value, node) => Math.abs(value - ranks[node]));
}
