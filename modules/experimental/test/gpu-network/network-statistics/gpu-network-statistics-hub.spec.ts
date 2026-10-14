// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {
  createTransientView,
  GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {getWorkgroupCounterKernelSource} from '../../../src/gpu-network/network-workgroup-counters';
import {
  getGPUNetworkStatisticsLength,
  GPU_NETWORK_STATISTICS_WORD,
  GPUNetworkStatistics
} from '../../../src/gpu-network/network-statistics';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../../src/utils/wgsl-kernel-nodes';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {createSymmetricEdges} from '../network-analysis/network-analytics-oracle';
import {buildCSR, type NetworkEdge} from '../network-reachability/network-reachability-oracle';
import {computeNetworkStatisticsOracle} from './network-statistics-oracle';

const NODE_COUNT = 120_000;
const HUB_DEGREE = 100_000;
const BIN_COUNT = 16;
const COMMUNITY_COUNT = 50;

function createSeededRandom(seed: number): () => number {
  let state = seed;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 2 ** 32;
  };
}

/** Hub vertex 0 linked to 100k leaves (including a self-loop) plus many small random rows. */
function createHubEdges(): NetworkEdge[] {
  const random = createSeededRandom(11);
  const edges: NetworkEdge[] = [[0, 0, 1]];
  for (let leaf = 1; leaf <= HUB_DEGREE; leaf++) edges.push([0, leaf, 1]);
  for (let count = 0; count < 30_000; count++) {
    const from = 1 + Math.floor(random() * (NODE_COUNT - 1));
    const to = 1 + Math.floor(random() * (NODE_COUNT - 1));
    edges.push([from, to, 1]);
  }
  return createSymmetricEdges(edges);
}

function createLabels(): Uint32Array {
  const random = createSeededRandom(7);
  return Uint32Array.from({length: NODE_COUNT}, () => Math.floor(random() * COMMUNITY_COUNT));
}

type Scene = {
  csr: ReturnType<typeof buildCSR>;
  communities: Uint32Array;
  vertexMask: Uint32Array;
};

function createScene(): Scene {
  const csr = buildCSR(NODE_COUNT, createHubEdges());
  const random = createSeededRandom(3);
  const vertexMask = Uint32Array.from({length: NODE_COUNT}, (_, vertex) =>
    vertex === 0 || random() >= 0.03 ? 1 : 0
  );
  return {csr, communities: createLabels(), vertexMask};
}

/** Words that must match exactly; modularity (word 10) is a float and compared separately. */
function expectSummaryMatches(actual: number[], expected: Uint32Array): void {
  const MODULARITY = GPU_NETWORK_STATISTICS_WORD.modularity;
  const actualWords = actual.map((value, index) => (index === MODULARITY ? 0 : value));
  const expectedWords = Array.from(expected, (value, index) => (index === MODULARITY ? 0 : value));
  expect(actualWords).toEqual(expectedWords);
  const bits = (words: ArrayLike<number>) =>
    new Float32Array(new Uint32Array([words[MODULARITY]]).buffer)[0];
  expect(Math.abs(bits(actual) - bits(expected))).toBeLessThan(1e-5);
}

type Views = {
  offsets: GraphDataView<'uint32'>;
  neighbors: GraphDataView<'uint32'>;
  communities: GraphDataView<'uint32'>;
  output: GraphDataView<'uint32'>;
  nodeMask?: GraphDataView<'uint32'>;
};

function createViews(
  device: Device,
  graph: GPUCommandGraph<undefined>,
  scene: Scene,
  withMask: boolean
): {views: Views; buffers: Buffer[]; outputBuffer: Buffer} {
  const {csr} = scene;
  const outputLength = getGPUNetworkStatisticsLength(BIN_COUNT);
  const buffers = [
    createInputBuffer(device, csr.offsets),
    createInputBuffer(device, csr.neighbors),
    createInputBuffer(device, scene.communities),
    createOutputBuffer(device, outputLength),
    createInputBuffer(device, scene.vertexMask)
  ];
  const views: Views = {
    offsets: importGraphBuffer(graph, 'offsets', buffers[0], 'uint32', NODE_COUNT + 1),
    neighbors: importGraphBuffer(graph, 'neighbors', buffers[1], 'uint32', csr.neighbors.length),
    communities: importGraphBuffer(graph, 'communities', buffers[2], 'uint32', NODE_COUNT),
    output: importGraphBuffer(graph, 'output', buffers[3], 'uint32', outputLength),
    nodeMask: withMask
      ? importGraphBuffer(graph, 'vertex-mask', buffers[4], 'uint32', NODE_COUNT)
      : undefined
  };
  return {views, buffers, outputBuffer: buffers[3]};
}

it('GPUNetworkStatistics matches the oracle with a 1e5-slot hub row and many small rows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = createScene();
  for (const directed of [false, true]) {
    const graph = new GPUCommandGraph<undefined>(device, {id: `hub-statistics-${directed}`});
    const {views, buffers, outputBuffer} = createViews(device, graph, scene, true);
    graph.add(
      new GPUNetworkStatistics({
        id: 'hub-statistics',
        topology: {offsets: views.offsets, neighbors: views.neighbors},
        directed,
        nodeMask: views.nodeMask,
        communities: views.communities,
        degreeBinning: 'log2',
        degreeBinCount: BIN_COUNT,
        output: views.output
      })
    );
    const compiled = await graph.compileAsync();
    submitGraph(device, compiled, undefined);
    const words = await readUint32(outputBuffer, getGPUNetworkStatisticsLength(BIN_COUNT));
    const expected = computeNetworkStatisticsOracle({
      nodeCount: NODE_COUNT,
      csr: scene.csr,
      directed,
      vertexMask: scene.vertexMask,
      communities: scene.communities,
      binCount: BIN_COUNT,
      binning: 'log2'
    });
    expectSummaryMatches(words, expected);
    // The hub row alone has 1e5 live-capable slots.
    expect(words[GPU_NETWORK_STATISTICS_WORD.maximumOutDegree]).toBeGreaterThan(HUB_DEGREE / 2);
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});

/**
 * The pre-migration thread-per-row `slots` and `intra-community` kernels (undirected, no masks),
 * kept here only as the A/B baseline. A hub row serializes on one thread.
 */
function getRowLoopProducer(views: Views): GPUCommandNodeProducer {
  const word = GPU_NETWORK_STATISTICS_WORD;
  return {
    getCommandNodes: <Parameters>(graph: GPUCommandGraph<Parameters>) => {
      const masked = createTransient(graph, 'row-loop-masked', views.neighbors.length);
      const outDegree = createTransient(graph, 'row-loop-out-degree', NODE_COUNT);
      const slotCounters = getWorkgroupCounterKernelSource({
        counters: [
          {target: 'summary', word: word.liveSlotCount},
          {target: 'summary', word: word.selfLoopSlotCount}
        ],
        rowCount: 'NODE_COUNT',
        rowSource: `let rowEnd = min(offsets[offsetsOffset + index + 1u], SLOT_COUNT);
  let rowBegin = min(offsets[offsetsOffset + index], rowEnd);
  var liveCount = 0u;
  var selfCount = 0u;
  for (var slot = rowBegin; slot < rowEnd; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    let slotLive = neighbor < NODE_COUNT;
    maskedOut[maskedOutOffset + slot] = select(0xffffffffu, neighbor, slotLive);
    if (slotLive) {
      liveCount++;
      if (neighbor == index) {
        selfCount++;
      }
    }
  }
  outDegree[outDegreeOffset + index] = liveCount + selfCount;
  addCounter(0u, liveCount);
  addCounter(1u, selfCount);`
      });
      const intraCounters = getWorkgroupCounterKernelSource({
        counters: [{target: 'summary', word: word.intraCommunitySlotCount}],
        rowCount: 'NODE_COUNT',
        rowSource: `let label = communities[communitiesOffset + index];
  let rowEnd = min(offsets[offsetsOffset + index + 1u], SLOT_COUNT);
  let rowBegin = min(offsets[offsetsOffset + index], rowEnd);
  var intraCount = 0u;
  for (var slot = rowBegin; slot < rowEnd; slot++) {
    let neighbor = maskedIn[maskedInOffset + slot];
    if (neighbor != 0xffffffffu && communities[communitiesOffset + neighbor] == label) {
      intraCount++;
    }
  }
  addCounter(0u, intraCount);`
      });
      const declarations = (counters: {
        declarations: string;
      }) => `const NODE_COUNT: u32 = ${NODE_COUNT}u;
const SLOT_COUNT: u32 = ${views.neighbors.length}u;
${counters.declarations}`;
      return [
        createWGSLKernelNode<Parameters>(graph, {
          id: 'row-loop-slots',
          operation: 'GPUNetworkStatisticsBaseline',
          bindings: [
            {name: 'offsets', view: views.offsets, type: 'u32', access: 'read'},
            {name: 'neighbors', view: views.neighbors, type: 'u32', access: 'read'},
            {name: 'maskedOut', view: masked, type: 'u32', access: 'read_write'},
            {name: 'outDegree', view: outDegree, type: 'u32', access: 'read_write'},
            {name: 'summary', view: views.output, type: 'atomic<u32>', access: 'read_write'}
          ],
          invocationCount: NODE_COUNT,
          workgroupSize: 256,
          guardIndex: false,
          declarations: declarations(slotCounters),
          body: slotCounters.body
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: 'row-loop-intra',
          operation: 'GPUNetworkStatisticsBaseline',
          bindings: [
            {name: 'offsets', view: views.offsets, type: 'u32', access: 'read'},
            {name: 'maskedIn', view: masked, type: 'u32', access: 'read'},
            {name: 'communities', view: views.communities, type: 'u32', access: 'read'},
            {name: 'summary', view: views.output, type: 'atomic<u32>', access: 'read_write'}
          ],
          invocationCount: NODE_COUNT,
          workgroupSize: 256,
          guardIndex: false,
          declarations: declarations(intraCounters),
          body: intraCounters.body
        })
      ] as GPUCommandNode<Parameters>[];
    }
  } as unknown as GPUCommandNodeProducer;
}

function createTransient<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  name: string,
  length: number
): GraphDataView<'uint32'> {
  return createTransientView(graph, name, 'uint32', length);
}

/** Same-run A/B of the slot passes (masking, counts, intra-community) on the hub scene. */
it('GPUNetworkStatistics slot passes: load-balanced vs the thread-per-row loop on a hub (A/B)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = createScene();
  const measure = async (implementation: 'load-balanced' | 'row-loop'): Promise<number> => {
    const graph = new GPUCommandGraph<undefined>(device, {id: `hub-ab-${implementation}`});
    const {views, buffers} = createViews(device, graph, scene, false);
    if (implementation === 'row-loop') {
      graph.add(getRowLoopProducer(views));
    } else {
      // Only the slot loops of the contributor, so both sides do the same work.
      const contributor = new GPUNetworkStatistics({
        id: 'hub-ab',
        topology: {offsets: views.offsets, neighbors: views.neighbors},
        communities: views.communities,
        degreeBinCount: BIN_COUNT,
        output: views.output
      });
      graph.add({
        getCommandNodes: <Parameters>(target: GPUCommandGraph<Parameters>) =>
          contributor
            .getCommandNodes(target)
            .filter(node => /^hub-ab-(slots|intra-community)/.test(node.id))
      } as unknown as GPUCommandNodeProducer);
    }
    const compiled = await graph.compileAsync();
    submitGraph(device, compiled, undefined);
    await readUint32(buffers[3], 1);
    const samples: number[] = [];
    for (let sample = 0; sample < 5; sample++) {
      const start = performance.now();
      submitGraph(device, compiled, undefined);
      await readUint32(buffers[3], 1);
      samples.push(performance.now() - start);
    }
    samples.sort((left, right) => left - right);
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
    return samples[2];
  };
  const balanced = await measure('load-balanced');
  const rowLoop = await measure('row-loop');
  console.info(
    `network statistics hub A/B (slots + intra-community): load-balanced ${balanced.toFixed(2)} ms, ` +
      `row loop ${rowLoop.toFixed(2)} ms`
  );
  expect(balanced).toBeGreaterThan(0);
}, 120_000);
