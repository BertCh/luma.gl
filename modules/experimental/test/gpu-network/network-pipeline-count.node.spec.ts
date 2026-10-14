// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUEdgeBundling} from '../../src/gpu-network/edge-bundling/index';
import {GPUNetworkCostMatrix} from '../../src/gpu-network/network-accessibility/index';
import {GPUNetworkServiceAreas} from '../../src/gpu-network/network-analysis/index';
import {GPUNetworkKFunction} from '../../src/gpu-network/network-k-function/index';
import {GPUNetworkReachability} from '../../src/gpu-network/network-reachability/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources
} from '../utils/gpu-contributor-test-utils';

function getDistinctComputePipelineCount(nodes: readonly unknown[]): number {
  return getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}).size;
}

function getDistinctComputePipelineCountMatching(
  nodes: readonly unknown[],
  idPattern: RegExp
): number {
  return getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true, idPattern}).size;
}

// Library rule 12 (review E-01, E-02): a round, batch or block index never reaches WGSL text or an
// override constant. Each spec builds the same contributor at N and N + 1 rounds or batches and
// asserts that the number of distinct (source, constants, layout) pipelines does not change.

const NODE_COUNT = 9;
const EDGE_COUNT = 24;

function createGraph(id: string): GPUCommandGraph {
  return new GPUCommandGraph(createNullWebGPUDevice(), {id});
}

function buildReachability(maximumIterations: number, maximumTieIterations: number) {
  const graph = createGraph(`reachability-${maximumIterations}`);
  return new GPUNetworkReachability({
    id: 'reach',
    topology: {
      offsets: createTransientView(graph, 'offsets', 'uint32', NODE_COUNT + 1),
      neighbors: createTransientView(graph, 'neighbors', 'uint32', EDGE_COUNT),
      weights: createTransientView(graph, 'weights', 'float32', EDGE_COUNT)
    },
    startNodes: createTransientView(graph, 'sources', 'uint32', 2),
    output: {
      costs: createTransientView(graph, 'costs', 'float32', NODE_COUNT),
      predecessors: createTransientView(graph, 'predecessors', 'uint32', NODE_COUNT),
      status: {
        converged: createTransientView(graph, 'converged', 'uint32', 1)
      }
    },
    maximumIterations,
    maximumTieIterations
  }).getCommandNodes(graph);
}

it('GPUNetworkReachability compiles one relax and one tie-level pipeline for every round', () => {
  const small = buildReachability(64, 4);
  const large = buildReachability(65, 5);
  expect(small.filter(node => /-relax-\d+$/.test(node.id))).toHaveLength(64);
  // Before the round cursor, 64 rounds compiled 64 relax pipelines (review E-01).
  expect(getDistinctComputePipelineCountMatching(small, /-relax-\d+$/)).toBe(1);
  expect(getDistinctComputePipelineCountMatching(small, /-tie-level-\d+$/)).toBe(1);
  expect(getDistinctComputePipelineCount(large)).toBe(getDistinctComputePipelineCount(small));
});

function buildServiceAreas(labelIterations: number) {
  const graph = createGraph(`service-areas-${labelIterations}`);
  return new GPUNetworkServiceAreas({
    id: 'areas',
    topology: {
      offsets: createTransientView(graph, 'offsets', 'uint32', NODE_COUNT + 1),
      neighbors: createTransientView(graph, 'neighbors', 'uint32', EDGE_COUNT),
      weights: createTransientView(graph, 'weights', 'float32', EDGE_COUNT)
    },
    facilities: createTransientView(graph, 'facilities', 'uint32', 3),
    output: {
      assignments: createTransientView(graph, 'assignments', 'uint32', NODE_COUNT),
      status: {
        converged: createTransientView(graph, 'converged', 'uint32', 1)
      }
    },
    maximumIterations: labelIterations + 3,
    maximumLabelIterations: labelIterations
  }).getCommandNodes(graph);
}

it('GPUNetworkServiceAreas compiles one label-round pipeline for every round', () => {
  const small = buildServiceAreas(8);
  const large = buildServiceAreas(9);
  expect(getDistinctComputePipelineCountMatching(small, /-label-round-\d+$/)).toBe(1);
  expect(getDistinctComputePipelineCount(large)).toBe(getDistinctComputePipelineCount(small));
});

function buildCostMatrix(rowCount: number, laneCount: number, nodeCount = 16) {
  const graph = createGraph(`cost-matrix-${rowCount}-${laneCount}`);
  const contributor = new GPUNetworkCostMatrix({
    id: 'matrix',
    topology: {
      offsets: createTransientView(graph, 'offsets', 'uint32', nodeCount + 1),
      neighbors: createTransientView(graph, 'neighbors', 'uint32', EDGE_COUNT),
      weights: createTransientView(graph, 'weights', 'float32', EDGE_COUNT)
    },
    startNodes: createTransientView(graph, 'seeds', 'uint32', rowCount),
    output: {
      costs: createTransientView(graph, 'costs', 'float32', rowCount * nodeCount),
      status: {
        converged: createTransientView(graph, 'converged', 'uint32', 1)
      }
    },
    laneCount,
    maximumIterations: 8
  });
  return {contributor, nodes: contributor.getCommandNodes(graph)};
}

it('GPUNetworkCostMatrix batches share pipelines, including a short last batch', () => {
  // 16 nodes x 4 lanes = 64 elements per batch: full batches write in place, the short batch
  // searches into scratch at the same lane count and is copied.
  const reference = buildReachability(8, 4);
  const five = buildCostMatrix(5 * 4 - 1, 4);
  const six = buildCostMatrix(6 * 4 - 1, 4);
  expect(five.contributor.batchCount).toBe(5);
  expect(six.contributor.batchCount).toBe(6);
  const count = getDistinctComputePipelineCount(five.nodes);
  expect(getDistinctComputePipelineCount(six.nodes)).toBe(count);
  // About 400 pipelines at 5 batches before the aligned stride (review E-02); now one set of
  // reachability kernels (initialize, seed, relax, finalize) plus the matrix kernels.
  expect(count).toBeLessThanOrEqual(10);
  expect(getDistinctComputePipelineCountMatching(five.nodes, /-relax-\d+$/)).toBe(1);
  expect(five.nodes.filter(node => node.id.endsWith('-copy')).map(node => node.id)).toEqual([
    'matrix-batch-4-copy'
  ]);
  expect(reference.length).toBeGreaterThan(0);
});

it('GPUNetworkCostMatrix with an unaligned lane count copies every batch through one pipeline', () => {
  // 9 nodes x 3 lanes = 27 elements: no batch offset is 256-byte aligned, so every batch copies.
  const five = buildCostMatrix(5 * 3 - 2, 3, 9);
  const six = buildCostMatrix(6 * 3 - 2, 3, 9);
  expect(five.nodes.filter(node => node.id.endsWith('-copy'))).toHaveLength(5);
  expect(getDistinctComputePipelineCount(six.nodes)).toBe(
    getDistinctComputePipelineCount(five.nodes)
  );
  expect(getDistinctComputePipelineCountMatching(five.nodes, /-copy$/)).toBe(1);
});

function buildKFunction(eventCount: number) {
  const graph = createGraph(`k-function-${eventCount}`);
  const simulationCount = 2;
  const bandCount = 8;
  const contributor = new GPUNetworkKFunction({
    id: 'k',
    positions: createTransientView(graph, 'points', 'float32x2', eventCount),
    topology: {
      offsets: createTransientView(graph, 'offsets', 'uint32', NODE_COUNT + 1),
      neighbors: createTransientView(graph, 'neighbors', 'uint32', EDGE_COUNT),
      weights: createTransientView(graph, 'weights', 'float32', EDGE_COUNT),
      nodePositions: createTransientView(graph, 'positions', 'float32x2', NODE_COUNT)
    },
    maximumDistance: createTransientView(graph, 'maximum-distance', 'float32', 1),
    networkLength: createTransientView(graph, 'network-length', 'float32', 1),
    parameters: createTransientView(graph, 'parameters', 'uint32', 4),
    bandCount,
    simulationCount,
    rowsPerBlock: 7,
    laneCount: 3,
    maximumIterations: 8,
    output: {
      kValues: createTransientView(graph, 'k-values', 'float32', (1 + simulationCount) * bandCount),
      status: {
        converged: createTransientView(graph, 'converged', 'uint32', 1)
      }
    }
  });
  return {contributor, nodes: contributor.getCommandNodes(graph)};
}

it('GPUNetworkKFunction blocks and batches share pipelines', () => {
  // 3 patterns x 10 events = 30 rows in blocks of 7 (the last block short), 3 batches per block
  // of 3 lanes (the last batch short): the block index, batch offsets and short shapes are data.
  const small = buildKFunction(10);
  const large = buildKFunction(12);
  expect(small.contributor.blockCount).toBe(5);
  expect(large.contributor.blockCount).toBe(6);
  expect(getDistinctComputePipelineCountMatching(small.nodes, /-block-\d+-count$/)).toBe(1);
  expect(getDistinctComputePipelineCountMatching(small.nodes, /-relax-\d+$/)).toBe(1);
  expect(getDistinctComputePipelineCountMatching(small.nodes, /-batch-\d+-seed$/)).toBe(1);
  // Same pipeline count at 5 and 6 blocks, except kernels whose sizes depend on the event count
  // (snapping, event flags): those are compared kernel family by kernel family.
  const families = (nodes: readonly {id: string}[]) =>
    new Set(nodes.map(node => node.id.replace(/-(block|batch|relax|round)-\d+/g, '')));
  expect(families(large.nodes)).toEqual(families(small.nodes));
  expect(getDistinctComputePipelineCount(large.nodes)).toBe(
    getDistinctComputePipelineCount(small.nodes)
  );
});

function buildEdgeBundling(maximumIterations: number) {
  const graph = createGraph(`edge-bundling-${maximumIterations}`);
  return new GPUEdgeBundling({
    id: 'bundle',
    nodePositions: createTransientView(graph, 'positions', 'float32x2', NODE_COUNT),
    sourceNodes: createTransientView(graph, 'sources', 'uint32', 12),
    targetNodes: createTransientView(graph, 'targets', 'uint32', 12),
    parameters: createTransientView(graph, 'parameters', 'float32', 5),
    output: {paths: createTransientView(graph, 'paths', 'float32x2', 12 * 8)},
    pointsPerEdge: 8,
    maximumIterations,
    densityResolution: 32
  }).getCommandNodes(graph);
}

it('GPUEdgeBundling iterations share one splat and one update pipeline', () => {
  const small = buildEdgeBundling(6);
  const large = buildEdgeBundling(7);
  expect(getDistinctComputePipelineCountMatching(small, /-splat-\d+$/)).toBe(1);
  expect(getDistinctComputePipelineCountMatching(small, /-update-\d+$/)).toBe(1);
  expect(getDistinctComputePipelineCount(large)).toBe(getDistinctComputePipelineCount(small));
});

it('every kernel of the round-cursor and batched contributors parses', () => {
  for (const nodes of [
    buildReachability(3, 2),
    buildServiceAreas(3),
    buildCostMatrix(5 * 4 - 1, 4).nodes,
    buildCostMatrix(5 * 3 - 2, 3, 9).nodes,
    buildKFunction(10).nodes,
    buildEdgeBundling(3)
  ]) {
    expect(getParsedKernelSources(nodes).size).toBeGreaterThan(0);
  }
  // The round index is data: no relax kernel names its round.
  const relax = getParsedKernelSources(buildReachability(3, 2)).get('reach-relax-2') ?? '';
  expect(relax).toContain('frontierNextRound()');
  expect(relax).not.toMatch(/FRONTIER_(CURRENT|NEXT)_ROUND/);
});
