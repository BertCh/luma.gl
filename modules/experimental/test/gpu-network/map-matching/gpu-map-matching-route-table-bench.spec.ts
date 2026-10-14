// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPUMapMatching,
  getGPUMapMatchingParameterValues,
  type GPUMapMatchingSettings
} from '../../../src/gpu-network/map-matching/index';
import {GPUNetworkCostTable} from '../../../src/gpu-network/network-cost-table/index';
import {
  createInputBuffer,
  createOutputBuffer,
  isSoftwareDevice,
  readUint32
} from '../../utils/gpu-contributor-test-utils';
import {buildCSR, type NetworkEdge} from '../network-reachability/network-reachability-oracle';

// A/B harness (not part of the correctness run): map matching 1000 tracks of 50 points about
// 200 m apart on a 100 x 100 street grid with 100 m blocks, once with the per-pair bounded A*
// search and once with a prebuilt UBODT-style cost table (the table build is timed separately,
// since it is built once per static network and reused by every matching encoding).

const WIDTH = 100;
const BLOCK = 100;
const NODE_COUNT = WIDTH * WIDTH;
const TRACK_COUNT = 1000;
const POINTS_PER_TRACK = 50;
const POINT_COUNT = TRACK_COUNT * POINTS_PER_TRACK;
const MEASURED_FRAMES = 5;
const PARAMETERS: GPUMapMatchingSettings = {
  sigma: 15,
  beta: 50,
  searchRadius: 50,
  routeFactor: 3,
  routeSlack: 50
};
// Default bound for 200 m samples is 3 * 200 + 50 = 650 m, less the tail of the source edge.
const TABLE_COST_LIMIT = 700;
// About 2 * 7 * 8 + 1 = 113 nodes lie within 700 m of a node on a 100 m grid.
const TABLE_CAPACITY = 1_500_000;

function median(values: number[]): number {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.floor(sorted.length / 2)];
}

/** Both directions of every grid link, 100 m long. */
function createStreetGrid(): NetworkEdge[] {
  const edges: NetworkEdge[] = [];
  for (let y = 0; y < WIDTH; y++) {
    for (let x = 0; x < WIDTH; x++) {
      const node = y * WIDTH + x;
      if (x + 1 < WIDTH) {
        edges.push([node, node + 1, BLOCK], [node + 1, node, BLOCK]);
      }
      if (y + 1 < WIDTH) {
        edges.push([node, node + WIDTH, BLOCK], [node + WIDTH, node, BLOCK]);
      }
    }
  }
  return edges;
}

/** Random walks of two blocks per sample that keep their heading, with 10 m Gaussian-ish noise. */
function createTracks(): Float32Array {
  let state = 17;
  const uniform = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 2 ** 32;
  };
  const points = new Float32Array(POINT_COUNT * 2);
  for (let track = 0; track < TRACK_COUNT; track++) {
    let x = 10 + Math.floor(uniform() * (WIDTH - 20));
    let y = 10 + Math.floor(uniform() * (WIDTH - 20));
    let direction = Math.floor(uniform() * 4);
    for (let step = 0; step < POINTS_PER_TRACK; step++) {
      const offset = (track * POINTS_PER_TRACK + step) * 2;
      points[offset] = x * BLOCK + (uniform() - 0.5) * 20;
      points[offset + 1] = y * BLOCK + (uniform() - 0.5) * 20;
      if (uniform() < 0.25) {
        direction = (direction + (uniform() < 0.5 ? 1 : 3)) % 4;
      }
      for (let block = 0; block < 2; block++) {
        const dx = [1, 0, -1, 0][direction];
        const dy = [0, 1, 0, -1][direction];
        if (x + dx < 0 || x + dx >= WIDTH || y + dy < 0 || y + dy >= WIDTH) {
          direction = (direction + 2) % 4;
        }
        x += [1, 0, -1, 0][direction];
        y += [0, 1, 0, -1][direction];
      }
    }
  }
  return points;
}

type Timing = {compileMilliseconds: number; encodeMilliseconds: number; frameMilliseconds: number};

/** Times one compiled graph: CPU encode and submit-to-readback of a one-word sync buffer. */
async function measureGraph(
  device: Device,
  graph: GPUCommandGraph,
  syncBuffer: Buffer,
  frames: number
): Promise<Timing> {
  const compileStart = performance.now();
  const compiled = graph.compile();
  const compileMilliseconds = performance.now() - compileStart;
  const encodeTimes: number[] = [];
  const frameTimes: number[] = [];
  for (let frame = 0; frame < frames + 1; frame++) {
    const frameStart = performance.now();
    const encoder = device.createCommandEncoder();
    compiled.encode(encoder, {parameters: undefined});
    const encodeMilliseconds = performance.now() - frameStart;
    device.submit(encoder.finish());
    await readUint32(syncBuffer, 1);
    if (frame > 0) {
      encodeTimes.push(encodeMilliseconds);
      frameTimes.push(performance.now() - frameStart);
    }
  }
  compiled.destroy();
  return {
    compileMilliseconds,
    encodeMilliseconds: median(encodeTimes),
    frameMilliseconds: median(frameTimes)
  };
}

it('GPUMapMatching route table bench: A* search against a prebuilt cost table', async () => {
  const device = await getWebGPUTestDevice();
  if (!device || isSoftwareDevice(device)) {
    return;
  }
  const csr = buildCSR(NODE_COUNT, createStreetGrid());
  const nodePositions = new Float32Array(NODE_COUNT * 2);
  for (let node = 0; node < NODE_COUNT; node++) {
    nodePositions[node * 2] = (node % WIDTH) * BLOCK;
    nodePositions[node * 2 + 1] = Math.floor(node / WIDTH) * BLOCK;
  }
  const points = createTracks();
  const trackOffsets = Uint32Array.from(
    {length: TRACK_COUNT + 1},
    (_, track) => track * POINTS_PER_TRACK
  );
  const buffers: Buffer[] = [];
  const input = (values: Uint32Array | Float32Array) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return buffer;
  };
  const output = (length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return buffer;
  };
  const sharedBuffers = {
    offsets: input(csr.offsets),
    neighbors: input(csr.neighbors),
    weights: input(csr.weights),
    nodePositions: input(nodePositions),
    points: input(points),
    trackOffsets: input(trackOffsets),
    tableCostLimit: input(Float32Array.of(TABLE_COST_LIMIT)),
    tableOffsets: output(NODE_COUNT + 1),
    tableTargets: output(TABLE_CAPACITY),
    tableCosts: output(TABLE_CAPACITY),
    tableOverflow: output(1),
    tableRequired: output(1),
    tableConverged: output(1)
  };
  const parameters = new GPUParameterBuffer(device, {
    id: 'bench-parameters',
    format: 'float32',
    length: 8,
    values: getGPUMapMatchingParameterValues(PARAMETERS)
  });

  const createMatchingGraph = (id: string, useTable: boolean) => {
    const graph = new GPUCommandGraph(device, {id});
    const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
      name: string,
      buffer: Buffer,
      format: Format,
      length: number
    ) => importGraphBuffer(graph, `${id}-${name}`, buffer, format, length);
    const edges = output(POINT_COUNT);
    const matchedCount = output(1);
    const overflow = output(1);
    const routeOverflow = output(1);
    graph.add(
      new GPUMapMatching({
        id: 'match',
        positions: view('points', sharedBuffers.points, 'float32x2', POINT_COUNT),
        trackOffsets: view('track-offsets', sharedBuffers.trackOffsets, 'uint32', TRACK_COUNT + 1),
        topology: {
          nodePositions: view(
            'nodes',
            sharedBuffers.nodePositions,
            'float32x2',
            NODE_COUNT
          ) as never,
          offsets: view('offsets', sharedBuffers.offsets, 'uint32', NODE_COUNT + 1),
          neighbors: view('neighbors', sharedBuffers.neighbors, 'uint32', csr.neighbors.length)
        },
        parameters: parameters.importToGraph(graph),
        routeTable: useTable
          ? {
              offsets: view('table-offsets', sharedBuffers.tableOffsets, 'uint32', NODE_COUNT + 1),
              targets: view('table-targets', sharedBuffers.tableTargets, 'uint32', TABLE_CAPACITY),
              costs: view('table-costs', sharedBuffers.tableCosts, 'float32', TABLE_CAPACITY),
              costLimit: view('table-limit', sharedBuffers.tableCostLimit, 'float32', 1),
              status: {
                overflow: view('table-overflow', sharedBuffers.tableOverflow, 'uint32', 1)
              }
            }
          : undefined,
        cellSize: 60,
        bounds: {minimum: [-100, -100], maximum: [WIDTH * BLOCK + 100, WIDTH * BLOCK + 100]},
        output: {
          matchedEdges: view('edges', edges, 'uint32', POINT_COUNT),
          matchedCount: view('matched-count', matchedCount, 'uint32', 1),
          status: {
            overflow: view('overflow', overflow, 'uint32', 1),
            routeOverflow: view('route-overflow', routeOverflow, 'uint32', 1)
          }
        }
      })
    );
    return {graph, edges, matchedCount, routeOverflow};
  };

  // Table build, once.
  const buildGraph = new GPUCommandGraph(device, {id: 'bench-table'});
  const buildView = <Format extends 'uint32' | 'float32'>(
    name: string,
    buffer: Buffer,
    format: Format,
    length: number
  ) => importGraphBuffer(buildGraph, `build-${name}`, buffer, format, length);
  buildGraph.add(
    new GPUNetworkCostTable({
      id: 'table',
      topology: {
        offsets: buildView('offsets', sharedBuffers.offsets, 'uint32', NODE_COUNT + 1),
        neighbors: buildView('neighbors', sharedBuffers.neighbors, 'uint32', csr.neighbors.length),
        weights: buildView('weights', sharedBuffers.weights, 'float32', csr.weights.length)
      },
      costLimit: buildView('limit', sharedBuffers.tableCostLimit, 'float32', 1),
      // Eight rounds of up to 16 hops cover the 7 hops of the table bound.
      maximumIterations: 8,
      output: {
        offsets: buildView('table-offsets', sharedBuffers.tableOffsets, 'uint32', NODE_COUNT + 1),
        targets: buildView('table-targets', sharedBuffers.tableTargets, 'uint32', TABLE_CAPACITY),
        costs: buildView('table-costs', sharedBuffers.tableCosts, 'float32', TABLE_CAPACITY),
        status: {
          overflow: buildView('table-overflow', sharedBuffers.tableOverflow, 'uint32', 1),
          requiredCount: buildView('table-required', sharedBuffers.tableRequired, 'uint32', 1),
          converged: buildView('table-converged', sharedBuffers.tableConverged, 'uint32', 1)
        }
      }
    })
  );
  const build = await measureGraph(device, buildGraph, sharedBuffers.tableConverged, 3);
  const [overflow] = await readUint32(sharedBuffers.tableOverflow, 1);
  const [converged] = await readUint32(sharedBuffers.tableConverged, 1);
  const [required] = await readUint32(sharedBuffers.tableRequired, 1);

  const aStar = createMatchingGraph('bench-astar', false);
  const aStarTiming = await measureGraph(device, aStar.graph, aStar.matchedCount, MEASURED_FRAMES);
  const aStarEdges = await readUint32(aStar.edges, POINT_COUNT);
  const [aStarRouteOverflow] = await readUint32(aStar.routeOverflow, 1);
  const table = createMatchingGraph('bench-table', true);
  const tableTiming = await measureGraph(device, table.graph, table.matchedCount, MEASURED_FRAMES);
  const tableEdges = await readUint32(table.edges, POINT_COUNT);
  const [tableRouteOverflow] = await readUint32(table.routeOverflow, 1);
  const differing = tableEdges.filter((edge, point) => edge !== aStarEdges[point]).length;

  const format = (label: string, timing: Timing) =>
    `  ${label}: compile ${timing.compileMilliseconds.toFixed(1)} ms, ` +
    `encode ${timing.encodeMilliseconds.toFixed(2)} ms, frame ${timing.frameMilliseconds.toFixed(2)} ms`;
  // eslint-disable-next-line no-console
  console.log(
    [
      `GPUMapMatching route table bench: ${TRACK_COUNT} tracks x ${POINTS_PER_TRACK} points, ${NODE_COUNT} nodes`,
      format('table build (once)', build),
      `    entries ${required}, overflow ${overflow}, converged ${converged}`,
      format('A* search', aStarTiming),
      `    routeOverflow ${aStarRouteOverflow}`,
      format('route table', tableTiming),
      `    routeOverflow ${tableRouteOverflow}`,
      `  matched edges differing from A*: ${differing} of ${POINT_COUNT}`
    ].join('\n')
  );

  expect(overflow).toBe(0);
  expect(converged).toBe(1);
  // The table is exact, A* is exact only while its node budget holds; only budget-limited searches
  // may differ.
  expect(differing).toBeLessThanOrEqual(Math.ceil(POINT_COUNT * 0.001) + aStarRouteOverflow);
  parameters.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
});
