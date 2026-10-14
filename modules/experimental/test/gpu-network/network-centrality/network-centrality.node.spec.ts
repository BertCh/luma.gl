// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUNetworkCentrality,
  type GPUNetworkCentralityProps
} from '../../../src/gpu-network/network-centrality/index';
import {
  CORE_PRIVATE_MEMORY_LIMIT_BYTES,
  CORE_WORKGROUP_STORAGE_LIMIT_BYTES,
  createNullWebGPUDevice,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources,
  getPrivateMemoryBytes,
  getStorageBindingCount,
  getWorkgroupMemoryBytes
} from '../../utils/gpu-contributor-test-utils';
import {
  CENTRALITY_REFERENCE,
  compareCentralityValues,
  createCentralityGridEdges,
  getCentralityGridMaximumIterations,
  mirrorCentrality
} from './network-centrality-oracle';

const NODE_COUNT = 8;
const EDGE_COUNT = 12;

it('benchmark grids budget one dependency round per possible tight hop plus the root', () => {
  for (const width of [1, 4, 64, 224]) {
    const edges = createCentralityGridEdges(width, 1);
    const maximumIterations = getCentralityGridMaximumIterations(width, edges);
    expect(maximumIterations).toBeGreaterThanOrEqual(2 * (width - 1) + 1);
    expect(maximumIterations).toBeLessThanOrEqual(1024);
    // For unit weights the bound is exactly the corner-to-corner hop count plus the root.
    expect(
      getCentralityGridMaximumIterations(
        width,
        edges.map(([source, target]) => [source, target, 1])
      )
    ).toBe(2 * (width - 1) + 1);
  }
});

let viewCount = 0;

function createView<Format extends 'uint32' | 'float32'>(
  graph: GPUCommandGraph,
  format: Format,
  length: number
) {
  return createTransientView(graph, `view-${viewCount++}`, format, length);
}

function createProps(
  graph: GPUCommandGraph,
  pivotCount: number,
  overrides: Partial<GPUNetworkCentralityProps> = {}
): GPUNetworkCentralityProps {
  return {
    id: 'centrality',
    topology: {
      offsets: createView(graph, 'uint32', NODE_COUNT + 1),
      neighbors: createView(graph, 'uint32', EDGE_COUNT),
      weights: createView(graph, 'float32', EDGE_COUNT)
    },
    reverseTopology: {
      offsets: createView(graph, 'uint32', NODE_COUNT + 1),
      neighbors: createView(graph, 'uint32', EDGE_COUNT),
      weights: createView(graph, 'float32', EDGE_COUNT)
    },
    pivotNodes: createView(graph, 'uint32', pivotCount),
    costLimit: createView(graph, 'float32', 1),
    laneCount: 2,
    maximumIterations: 3,
    output: {
      betweenness: createView(graph, 'float32', NODE_COUNT),
      closeness: createView(graph, 'float32', pivotCount),
      harmonicCloseness: createView(graph, 'float32', pivotCount),
      reachedCounts: createView(graph, 'uint32', pivotCount),
      status: {converged: createView(graph, 'uint32', 1)}
    },
    ...overrides
  };
}

function build(pivotCount: number, overrides: Partial<GPUNetworkCentralityProps> = {}) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const contributor = new GPUNetworkCentrality(createProps(graph, pivotCount, overrides));
  return {graph, contributor, nodes: contributor.getCommandNodes(graph)};
}

it('GPUNetworkCentrality lists its nodes per batch', () => {
  const {contributor, nodes} = build(5);
  expect(contributor.laneCount).toBe(2);
  expect(contributor.batchCount).toBe(3);
  const ids = nodes.map(node => node.id);
  expect(ids.slice(0, 4)).toEqual([
    'centrality-expand-seeds',
    'centrality-batch-info',
    'centrality-pack-adjacency',
    'centrality-zero-betweenness'
  ]);
  for (const batch of [0, 1, 2]) {
    const prefix = `centrality-batch-${batch}`;
    const order = [
      `${prefix}-reachability-initialize`,
      `${prefix}-reachability-relax-2`,
      `${prefix}-closeness`,
      `${prefix}-reset`,
      `${prefix}-sigma-initialize`,
      `${prefix}-sigma-0`,
      `${prefix}-sigma-2`,
      `${prefix}-delta-initialize`,
      `${prefix}-delta-0`,
      `${prefix}-delta-2`,
      `${prefix}-accumulate`,
      `${prefix}-status`
    ].map(id => ids.indexOf(id));
    expect(
      order.every(position => position >= 0),
      prefix
    ).toBe(true);
    expect([...order].sort((left, right) => left - right)).toEqual(order);
  }
  expect(ids.at(-1)).toBe('centrality-converged');
  expect(ids.filter(id => /-sigma-\d+$/.test(id))).toHaveLength(9);
});

it('GPUNetworkCentrality builds only the nodes of the outputs it is given', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const props = createProps(graph, 5);
  const onlyBetweenness = new GPUNetworkCentrality({
    ...props,
    output: {betweenness: props.output.betweenness}
  }).getCommandNodes(graph);
  const ids = onlyBetweenness.map(node => node.id);
  expect(ids.some(id => id.endsWith('-closeness') || id.endsWith('-batch-info'))).toBe(false);
  expect(ids.some(id => id.endsWith('-status') || id.endsWith('-converged'))).toBe(false);
  expect(ids).toContain('centrality-batch-2-accumulate');

  const closenessGraph = new GPUCommandGraph(createNullWebGPUDevice());
  const closenessProps = createProps(closenessGraph, 5);
  const closenessIds = new GPUNetworkCentrality({
    ...closenessProps,
    output: {reachedCounts: closenessProps.output.reachedCounts}
  })
    .getCommandNodes(closenessGraph)
    .map(node => node.id);
  expect(closenessIds.some(id => /-sigma-|-delta-|-accumulate|pack-adjacency/.test(id))).toBe(
    false
  );
  expect(closenessIds).toContain('centrality-batch-0-closeness');
});

it('GPUNetworkCentrality validates its props', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const make = (overrides: Partial<GPUNetworkCentralityProps>, pivotCount = 4) =>
    new GPUNetworkCentrality({...createProps(graph, pivotCount), ...overrides});
  const props = createProps(graph, 4);
  expect(() => make({output: {}})).toThrow(/output must contain/);
  expect(() => make({output: {status: props.output.status}})).toThrow(/output must contain/);
  expect(() => make({pivotNodes: createView(graph, 'uint32', 0)})).toThrow(/at least one pivot/);
  expect(() => make({pivotNodes: createView(graph, 'float32', 4) as never})).toThrow(/pivotNodes/);
  expect(() => make({laneCount: 5})).toThrow(/laneCount/);
  expect(() => make({laneCount: 0})).toThrow(/laneCount/);
  expect(() => make({maximumIterations: 0})).toThrow(/maximumIterations/);
  expect(() => make({maximumIterations: 1025})).toThrow(/maximumIterations/);
  expect(() => make({localIterations: 65})).toThrow(/localIterations/);
  expect(() => make({betweennessScale: Infinity})).toThrow(/betweennessScale/);
  expect(() => make({costLimit: createView(graph, 'float32', 2)})).toThrow(/costLimit/);
  expect(() =>
    make({
      reverseTopology: {
        offsets: createView(graph, 'uint32', NODE_COUNT + 2),
        neighbors: createView(graph, 'uint32', EDGE_COUNT),
        weights: createView(graph, 'float32', EDGE_COUNT)
      }
    })
  ).toThrow(/node count/);
  expect(() => make({output: {betweenness: createView(graph, 'float32', NODE_COUNT + 1)}})).toThrow(
    /one row per node/
  );
  expect(() => make({output: {closeness: createView(graph, 'float32', 3)}})).toThrow(
    /one row per pivot/
  );
  expect(() => make({output: {reachedCounts: createView(graph, 'uint32', 5)}})).toThrow(
    /one row per pivot/
  );
  expect(() =>
    make({
      output: {
        closeness: createView(graph, 'float32', 4),
        status: {converged: createView(graph, 'uint32', 2)}
      }
    })
  ).toThrow(/exactly one row/);
  const aliasedWeights = graph.createDataView(props.topology.weights.buffer, {
    format: 'float32',
    length: NODE_COUNT,
    byteOffset: 0
  });
  expect(() => make({topology: props.topology, output: {betweenness: aliasedWeights}})).toThrow(
    /outputs must not/
  );
  const shared = createView(graph, 'float32', 4);
  expect(() => make({output: {closeness: shared, harmonicCloseness: shared}})).toThrow(
    /separate buffers/
  );
  // An undirected network may pass the same topology twice.
  expect(() =>
    make({reverseTopology: props.topology, output: {closeness: createView(graph, 'float32', 4)}})
  ).not.toThrow();
});

it('GPUNetworkCentrality resolves a default lane count from the pivot count', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const props = createProps(graph, 40, {laneCount: undefined});
  const contributor = new GPUNetworkCentrality(props);
  // Small networks run every pivot in one batch.
  expect(contributor.laneCount).toBe(40);
  expect(contributor.batchCount).toBe(1);
});

it('GPUNetworkCentrality kernels parse, bind at most eight buffers and fit core memory', () => {
  const sources = getParsedKernelSources(build(5).nodes);
  expect(sources.size).toBeGreaterThan(10);
  for (const [id, source] of sources) {
    expect(getStorageBindingCount(source), id).toBeLessThanOrEqual(8);
    expect(getPrivateMemoryBytes(source), id).toBeLessThanOrEqual(CORE_PRIVATE_MEMORY_LIMIT_BYTES);
    expect(getWorkgroupMemoryBytes(source), id).toBeLessThanOrEqual(
      CORE_WORKGROUP_STORAGE_LIMIT_BYTES
    );
  }
  // The sigma and delta rounds bind the adjacency, the costs, the scratch array and the four
  // frontier arrays.
  expect(getStorageBindingCount(sources.get('centrality-batch-0-sigma-1')!)).toBe(7);
  expect(getStorageBindingCount(sources.get('centrality-batch-0-delta-1')!)).toBe(7);
});

it('GPUNetworkCentrality compiles the same pipelines for every batch and round count', () => {
  const pipelineCount = (pivotCount: number, maximumIterations: number) =>
    getGPUCommandNodePipelineKeys(build(pivotCount, {maximumIterations}).nodes, {
      includeShaderLayout: true
    }).size;
  // Fixed laneCount 2: 6 pivots are 3 batches, 7 pivots are 4 batches, 2 pivots one batch.
  expect(build(6).contributor.batchCount).toBe(3);
  expect(build(7).contributor.batchCount).toBe(4);
  expect(pipelineCount(7, 3)).toBe(pipelineCount(6, 3));
  expect(pipelineCount(2, 3)).toBe(pipelineCount(6, 3));
  // One more round adds nodes, not pipelines.
  expect(pipelineCount(6, 4)).toBe(pipelineCount(6, 3));
  const idPattern = /-(sigma|delta)-\d+$/;
  const rounds = (maximumIterations: number) =>
    getGPUCommandNodePipelineKeys(build(4, {maximumIterations}).nodes, {
      includeShaderLayout: true,
      idPattern
    }).size;
  // One sigma and one delta pipeline serve every round of every batch.
  expect(rounds(3)).toBe(2);
  expect(rounds(4)).toBe(2);
});

it('the centrality fixture records its generator, versions and derived tolerances', () => {
  expect(CENTRALITY_REFERENCE.generator).toBe('test/oracles/network-centrality.py');
  expect(CENTRALITY_REFERENCE.versions['networkx']).toMatch(/^\d+\.\d+/);
  expect(Object.keys(CENTRALITY_REFERENCE.scenes)).toEqual(['hand', 'grid', 'random']);
  for (const scene of Object.values(CENTRALITY_REFERENCE.scenes)) {
    expect(scene.maximumSigma).toBeLessThan(2 ** 24);
    // h (7 + K) 2^-24, derived in the generator.
    expect(scene.tolerance.deltaRelative).toBeCloseTo(
      scene.maximumDepth * (7 + scene.maximumOutDegree) * 2 ** -24,
      20
    );
    expect(scene.tolerance.betweennessRelative).toBeCloseTo(
      scene.tolerance.deltaRelative + (scene.nodeCount + 2) * 2 ** -24,
      20
    );
    expect(scene.betweenness.exact).toHaveLength(scene.nodeCount);
    expect(scene.edges.every(([, , weight]) => Number.isInteger(weight) && weight >= 1)).toBe(true);
  }
});

it('the hand scene has equal-cost shortest paths and a one-way edge', () => {
  const hand = CENTRALITY_REFERENCE.scenes['hand'];
  expect(hand.maximumSigma).toBeGreaterThan(1);
  // Node 8 has no in-edges, so it is never an intermediate: betweenness 0. Node 7 is behind a
  // one-way edge from 6.
  expect(hand.betweenness.exact[8]).toBe(0);
  expect(hand.closeness.unbounded.reachedCounts[8]).toBe(9);
  expect(hand.closeness.unbounded.reachedCounts[0]).toBe(8);
});

it('the CPU mirror of the kernels reproduces networkx for every scene and variant', () => {
  for (const [name, scene] of Object.entries(CENTRALITY_REFERENCE.scenes)) {
    const {nodeCount, edges, tolerance} = scene;
    const allNodes = Array.from({length: nodeCount}, (_, node) => node);
    for (const laneCount of [1, 3, nodeCount]) {
      const exact = mirrorCentrality({nodeCount, edges, pivots: allNodes, laneCount});
      expect(
        compareCentralityValues(
          `${name} exact lanes ${laneCount}`,
          exact.betweenness,
          scene.betweenness.exact,
          tolerance.betweennessRelative
        )
      ).toBeUndefined();
      expect(Array.from(exact.reachedCounts)).toEqual(scene.closeness.unbounded.reachedCounts);
      expect(
        compareCentralityValues(
          `${name} closeness`,
          exact.closeness,
          scene.closeness.unbounded.closeness,
          tolerance.closenessRelative
        )
      ).toBeUndefined();
      expect(
        compareCentralityValues(
          `${name} harmonic`,
          exact.harmonicCloseness,
          scene.closeness.unbounded.harmonicCloseness,
          tolerance.harmonicRelative
        )
      ).toBeUndefined();

      const sampled = mirrorCentrality({
        nodeCount,
        edges,
        pivots: scene.sampledPivots,
        laneCount,
        scale: scene.sampledScale
      });
      expect(
        compareCentralityValues(
          `${name} sampled lanes ${laneCount}`,
          sampled.betweenness,
          scene.betweenness.sampled,
          tolerance.betweennessRelative,
          scene.sampledScale
        )
      ).toBeUndefined();

      const bounded = mirrorCentrality({
        nodeCount,
        edges,
        pivots: allNodes,
        laneCount,
        radius: scene.radius
      });
      expect(
        compareCentralityValues(
          `${name} radius lanes ${laneCount}`,
          bounded.betweenness,
          scene.betweenness.radius,
          tolerance.betweennessRelative
        )
      ).toBeUndefined();
      expect(Array.from(bounded.reachedCounts)).toEqual(scene.closeness.radius.reachedCounts);
      expect(
        compareCentralityValues(
          `${name} radius closeness`,
          bounded.closeness,
          scene.closeness.radius.closeness,
          tolerance.closenessRelative
        )
      ).toBeUndefined();
      expect(
        compareCentralityValues(
          `${name} radius harmonic`,
          bounded.harmonicCloseness,
          scene.closeness.radius.harmonicCloseness,
          tolerance.harmonicRelative
        )
      ).toBeUndefined();

      const boundedSampled = mirrorCentrality({
        nodeCount,
        edges,
        pivots: scene.sampledPivots,
        laneCount,
        radius: scene.radius
      });
      expect(
        compareCentralityValues(
          `${name} radius sampled`,
          boundedSampled.betweenness,
          scene.betweenness.radiusSampled,
          tolerance.betweennessRelative
        )
      ).toBeUndefined();
    }
  }
});

it('the CPU mirror ignores out-of-range pivots and reports zero rows for them', () => {
  const hand = CENTRALITY_REFERENCE.scenes['hand'];
  const result = mirrorCentrality({
    nodeCount: hand.nodeCount,
    edges: hand.edges,
    pivots: [0, 0xffffffff, 3],
    laneCount: 2
  });
  expect(result.reachedCounts[1]).toBe(0);
  expect(result.closeness[1]).toBe(0);
  expect(result.harmonicCloseness[1]).toBe(0);
  const without = mirrorCentrality({
    nodeCount: hand.nodeCount,
    edges: hand.edges,
    pivots: [0, 3],
    laneCount: 2
  });
  expect(Array.from(result.betweenness)).toEqual(Array.from(without.betweenness));
});
