// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/gpgpu/gpu-data';
import {GPUGraph} from '@luma.gl/gpgpu/gpu-graph';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it, vi} from 'vitest';

import {GPUGraphDeckEffect} from '../src/gpu-graph/gpu-graph-effect';
import {
  getGPUGraphRecipeColumnsSkipReason,
  GPUGraphRecipeColumns
} from '../src/gpu-graph/gpu-graph-recipe-columns';
import {makeGraphExplorerDataset} from '../../../examples/experimental/gpu-graph-explorer/graph-data';
import {
  buildCSR,
  NONE,
  bandOracle,
  type NetworkCSR,
  type NetworkEdge
} from '../../experimental/test/map-graphs/network-reachability/network-reachability-oracle';
import {
  componentsOracle,
  coreNumberOracle,
  degreeOracle,
  normalizeOracle,
  pageRankOracle
} from '../../experimental/test/map-graphs/network-analysis/network-analytics-oracle';
import {neighborhoodOracle} from '../../experimental/test/map-graphs/network-analysis/network-neighborhood-oracle';

const BAND_THRESHOLDS = [0, 1, 2, 3, 4, 6, 8, 12];

async function readUint32(buffer: Buffer, length: number): Promise<number[]> {
  const bytes = await buffer.readAsync(0, length * 4);
  return Array.from(new Uint32Array(bytes.buffer, bytes.byteOffset, length));
}

async function readFloat32(buffer: Buffer, length: number): Promise<number[]> {
  const bytes = await buffer.readAsync(0, length * 4);
  return Array.from(new Float32Array(bytes.buffer, bytes.byteOffset, length));
}

function createGraph(
  device: Device,
  vertexCount: number,
  sources: Uint32Array,
  targets: Uint32Array
): {graph: GPUGraph; destroy: () => void} {
  const buffers: Buffer[] = [];
  const vector = (name: string, values: Uint32Array) => {
    const buffer = device.createBuffer({
      id: name,
      data: values,
      usage: Buffer.STORAGE | Buffer.COPY_DST
    });
    buffers.push(buffer);
    return new GPUVector<'uint32'>({
      type: 'data',
      name,
      format: 'uint32',
      data: [new GPUData<'uint32'>({buffer, format: 'uint32', length: values.length})],
      ownsData: false
    });
  };
  const graph = new GPUGraph({
    vertexCount,
    directed: true,
    sourceVertices: vector('test-sources', sources),
    targetVertices: vector('test-targets', targets)
  });
  return {graph, destroy: () => buffers.forEach(buffer => buffer.destroy())};
}

/** Symmetrized CSR with one slot per direction per edge, exactly like the undirected topology. */
function buildSymmetricCSR(vertexCount: number, sources: Uint32Array, targets: Uint32Array) {
  const edges: NetworkEdge[] = [];
  for (let edge = 0; edge < sources.length; edge++) {
    edges.push([sources[edge], targets[edge], 1], [targets[edge], sources[edge], 1]);
  }
  return buildCSR(vertexCount, edges);
}

function bfsDistances(csr: NetworkCSR, vertexCount: number, source: number): number[] {
  const distances = new Array<number>(vertexCount).fill(NONE);
  distances[source] = 0;
  const queue = [source];
  for (let head = 0; head < queue.length; head++) {
    const node = queue[head];
    for (let edge = csr.offsets[node]; edge < csr.offsets[node + 1]; edge++) {
      const neighbor = csr.neighbors[edge];
      if (distances[neighbor] === NONE) {
        distances[neighbor] = distances[node] + 1;
        queue.push(neighbor);
      }
    }
  }
  return distances;
}

async function encodeAndSubmit(device: Device, columns: GPUGraphRecipeColumns): Promise<void> {
  const encoder = device.createCommandEncoder({id: 'recipe-columns-test'});
  columns.encodeAnalytics(encoder);
  columns.encodeInteraction(encoder);
  device.submit(encoder.finish());
}

it('GPUGraphRecipeColumns produces oracle-exact GPU columns, masks, paths, and bands without recompiling', async () => {
  const device = await getWebGPUTestDevice();
  expect(device, 'a WebGPU device is required for the recipe-columns parity test').toBeTruthy();
  if (!device) return;

  const dataset = makeGraphExplorerDataset();
  const sources = Uint32Array.from(dataset.sourceChunks.flatMap(chunk => Array.from(chunk)));
  const targets = Uint32Array.from(dataset.targetChunks.flatMap(chunk => Array.from(chunk)));
  const vertexCount = dataset.vertexCount;
  const symmetric = buildSymmetricCSR(vertexCount, sources, targets);
  const {graph, destroy: destroyGraph} = createGraph(device, vertexCount, sources, targets);

  const submitSpy = vi.spyOn(device, 'submit');
  const columns = new GPUGraphRecipeColumns(device, graph);
  try {
    expect(submitSpy.mock.calls.length, 'construction never submits hidden GPU work').toBe(0);

    const computeSpy = vi.spyOn(device, 'createComputePipeline');
    const renderSpy = vi.spyOn(device, 'createRenderPipeline');
    const graphs = [columns.topologyGraph, ...columns.analyticsGraphs, columns.interactionGraph];

    // Initial encoding: topology, analytics, and the (empty) interaction state.
    columns.setPathEndpoints(null, null);
    await encodeAndSubmit(device, columns);
    expect(columns.isAnalyticsComplete).toBe(true);
    const initialInteractionCount = columns.stats.interactionEncodeCount;
    expect(initialInteractionCount).toBe(1);

    // ---- analytics columns
    const countRows = vertexCount;
    expect(await readUint32(columns.rawColumns.degree.buffer, countRows)).toEqual(
      Array.from(degreeOracle(symmetric.offsets))
    );
    const rankOracle = pageRankOracle(vertexCount, symmetric, symmetric, 0.85, 20);
    const rawPageRank = await readFloat32(columns.rawColumns.pageRank.buffer, countRows);
    rawPageRank.forEach((value, node) => expect(value).toBeCloseTo(rankOracle[node], 4));
    const normalizedDegree = await readFloat32(columns.columns.degree.buffer, countRows);
    const degreeExpected = normalizeOracle(degreeOracle(symmetric.offsets));
    normalizedDegree.forEach((value, node) => expect(value).toBeCloseTo(degreeExpected[node], 5));
    const normalizedPageRank = await readFloat32(columns.columns.pageRank.buffer, countRows);
    const pageRankExpected = normalizeOracle(rankOracle);
    normalizedPageRank.forEach((value, node) =>
      expect(value).toBeCloseTo(pageRankExpected[node], 3)
    );
    const normalizedCore = await readFloat32(columns.columns.coreNumber.buffer, countRows);
    const coreExpected = normalizeOracle(coreNumberOracle(vertexCount, symmetric));
    normalizedCore.forEach((value, node) => expect(value).toBeCloseTo(coreExpected[node], 5));
    expect(columns.columns.degree.format).toBe('float32');
    expect(columns.columns.component.format).toBe('uint32');

    const componentLabels = await readUint32(columns.columns.component.buffer, countRows);
    expect(componentLabels).toEqual(Array.from(componentsOracle(vertexCount, symmetric)));
    const communityLabels = await readUint32(columns.columns.community.buffer, countRows);
    communityLabels.forEach((label, node) => {
      expect(label, 'community labels are source vertex IDs').toBeLessThan(vertexCount);
      expect(componentLabels[label], 'label propagation never crosses a weak component').toBe(
        componentLabels[node]
      );
    });

    // ---- hover neighborhood
    const hoverNode = 5;
    for (const hops of [0, 1, 2, 3]) {
      columns.setHoverVertex(hoverNode);
      columns.setNeighborhoodHops(hops);
      await encodeAndSubmit(device, columns);
      const expected = neighborhoodOracle({
        csr: symmetric,
        nodeCount: vertexCount,
        seeds: [hoverNode],
        hops,
        maxHops: 8
      });
      const mask = await readUint32(columns.neighborhoodMask, countRows);
      expect(mask.map(Boolean), `neighborhood mask at ${hops} hops`).toEqual(
        expected.nodeMask.map(Boolean)
      );
      expect(await readUint32(columns.hopDistances, countRows)).toEqual(expected.hopDistances);
      expect(mask[hoverNode]).toBeGreaterThan(0);
    }
    columns.setHoverVertex(null);
    await encodeAndSubmit(device, columns);
    expect(
      (await readUint32(columns.neighborhoodMask, countRows)).every(value => value === 0),
      'seedCount 0 yields an all-zero mask'
    ).toBe(true);

    // ---- A to B path and reachability bands
    const pathSource = 0;
    const distances = bfsDistances(symmetric, vertexCount, pathSource);
    let pathTarget = pathSource;
    for (let node = 0; node < vertexCount; node++) {
      if (distances[node] !== NONE && distances[node] > distances[pathTarget]) pathTarget = node;
    }
    expect(distances[pathTarget], 'fixture has a multi-hop path').toBeGreaterThanOrEqual(2);
    columns.setPathEndpoints(pathSource, pathTarget);
    await encodeAndSubmit(device, columns);
    const ranks = await readUint32(columns.pathRanks, countRows);
    const pathNodes = ranks
      .map((rank, node) => ({rank, node}))
      .filter(entry => entry.rank > 0)
      .sort((left, right) => left.rank - right.rank);
    expect(pathNodes.length, 'path length equals BFS distance + 1').toBe(distances[pathTarget] + 1);
    pathNodes.forEach((entry, index) => expect(entry.rank).toBe(index + 1));
    expect(pathNodes[0].node, 'rank 1 is source A').toBe(pathSource);
    expect(pathNodes[pathNodes.length - 1].node, 'last rank is target B').toBe(pathTarget);
    for (let index = 1; index < pathNodes.length; index++) {
      const from = pathNodes[index - 1].node;
      const to = pathNodes[index].node;
      const adjacent = Array.from(
        symmetric.neighbors.slice(symmetric.offsets[from], symmetric.offsets[from + 1])
      ).includes(to);
      expect(adjacent, `path step ${from} -> ${to} follows an edge`).toBe(true);
    }
    const bands = await readUint32(columns.reachabilityBands, countRows);
    expect(bands).toEqual(
      Array.from(
        bandOracle(
          Float32Array.from(distances, distance => (distance === NONE ? Infinity : distance)),
          BAND_THRESHOLDS
        )
      )
    );

    // Unreachable target (the explorer graph ends with an isolated vertex) clears the path.
    const isolated = vertexCount - 1;
    expect(distances[isolated]).toBe(NONE);
    columns.setPathEndpoints(pathSource, isolated);
    await encodeAndSubmit(device, columns);
    expect(
      (await readUint32(columns.pathRanks, countRows)).every(value => value === 0),
      'unreachable target yields no path'
    ).toBe(true);

    // Null endpoints clear ranks and bands.
    columns.setPathEndpoints(pathSource, pathTarget);
    await encodeAndSubmit(device, columns);
    columns.setPathEndpoints(null, null);
    await encodeAndSubmit(device, columns);
    expect((await readUint32(columns.pathRanks, countRows)).every(value => value === 0)).toBe(true);
    expect(
      (await readUint32(columns.reachabilityBands, countRows)).every(value => value === NONE),
      'bands are all NONE without a source'
    ).toBe(true);

    // ---- no recompilation across all of the interactions above
    expect(computeSpy.mock.calls.length, 'interactions never create compute pipelines').toBe(0);
    expect(renderSpy.mock.calls.length, 'interactions never create render pipelines').toBe(0);
    expect(
      [columns.topologyGraph, ...columns.analyticsGraphs, columns.interactionGraph],
      'compiled graph objects are identical'
    ).toEqual(graphs);
    const encodeCount = columns.stats.interactionEncodeCount;
    const idleEncoder = device.createCommandEncoder({id: 'recipe-columns-idle'});
    expect(columns.encodeInteraction(idleEncoder), 'unchanged inputs encode nothing').toBe(false);
    expect(columns.encodeAnalytics(idleEncoder), 'analytics encode only once').toBe(false);
    idleEncoder.finish();
    expect(columns.stats.interactionEncodeCount).toBe(encodeCount);
    computeSpy.mockRestore();
    renderSpy.mockRestore();
  } finally {
    submitSpy.mockRestore();
    columns.destroy();
    destroyGraph();
  }
});

it('GPUGraphDeckEffect owns recipe columns by default and can opt out', async () => {
  const device = await getWebGPUTestDevice();
  expect(device, 'a WebGPU device is required for the effect recipe-columns test').toBeTruthy();
  if (!device) return;

  const dataset = makeGraphExplorerDataset();
  const submitSpy = vi.spyOn(device, 'submit');
  const effect = new GPUGraphDeckEffect(device, dataset);
  const disabled = new GPUGraphDeckEffect(device, dataset, {recipeColumns: false});
  try {
    expect(submitSpy.mock.calls.length, 'effect construction never submits').toBe(0);
    submitSpy.mockRestore();
    expect(effect.recipeColumns).toBeInstanceOf(GPUGraphRecipeColumns);
    expect(effect.recipeColumnsSkipReason).toBeNull();
    expect(disabled.recipeColumns).toBeUndefined();
    expect(disabled.recipeColumnsSkipReason).toContain('disabled');
    expect(getGPUGraphRecipeColumnsSkipReason(device, 128, 0)).toContain('no edges');
    expect(
      getGPUGraphRecipeColumnsSkipReason(
        device,
        1024,
        Math.ceil(device.limits.maxStorageBufferBindingSize / 4)
      ),
      'adjacency beyond the adapter binding limit is skipped'
    ).toContain('adapter limit');

    effect.setHoverVertex(5);
    effect.setNeighborhoodHops(2);
    effect.setPathEndpoints(0, 7);
    effect.preRender({viewports: [{}]} as unknown as Parameters<
      GPUGraphDeckEffect['preRender']
    >[0]);
    device.submit();
    const mask = await readUint32(effect.recipeColumns!.neighborhoodMask, dataset.vertexCount);
    expect(mask[5], 'the hovered vertex is inside its own neighborhood').toBeGreaterThan(0);
    expect(effect.recipeColumns!.stats.interactionEncodeCount).toBe(1);
    expect(effect.recipeColumns!.isAnalyticsComplete).toBe(true);
  } finally {
    effect.cleanup({} as never);
    disabled.cleanup({} as never);
  }
});
