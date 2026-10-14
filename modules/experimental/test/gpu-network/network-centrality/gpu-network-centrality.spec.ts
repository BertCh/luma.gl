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
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import type {NetworkEdge} from '../network-reachability/network-reachability-oracle';
import {
  buildCentralityCSRs,
  CENTRALITY_REFERENCE,
  compareCentralityValues,
  createCentralityGridEdges,
  mirrorCentrality,
  type CentralityReferenceScene
} from './network-centrality-oracle';

const NONE = 0xffffffff;

type CentralityRun = {
  betweenness: Float32Array;
  closeness: Float32Array;
  harmonicCloseness: Float32Array;
  reachedCounts: Uint32Array;
  converged: number;
};

type CentralityRunOptions = {
  laneCount?: number;
  radius?: number;
  scale?: number;
  maximumIterations?: number;
  localIterations?: number;
};

/** Builds, encodes once and reads one centrality computation with every output. */
async function runCentrality(
  device: Device,
  nodeCount: number,
  edges: readonly NetworkEdge[],
  pivots: readonly number[],
  options: CentralityRunOptions = {}
): Promise<CentralityRun> {
  const graph = new GPUCommandGraph(device, {id: 'centrality'});
  const buffers: Buffer[] = [];
  const input = (values: Uint32Array | Float32Array) => {
    const buffer = createInputBuffer(device, values.length ? values : new Uint32Array(1));
    buffers.push(buffer);
    return buffer;
  };
  const output = (length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return buffer;
  };
  const {forward, reverse} = buildCentralityCSRs(nodeCount, edges);
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
  const betweennessBuffer = output(nodeCount);
  const closenessBuffer = output(pivots.length);
  const harmonicBuffer = output(pivots.length);
  const reachedBuffer = output(pivots.length);
  const convergedBuffer = output(1);
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
      costLimit:
        options.radius !== undefined
          ? importGraphBuffer(
              graph,
              'cost-limit',
              input(Float32Array.from([options.radius])),
              'float32',
              1
            )
          : undefined,
      laneCount: options.laneCount,
      maximumIterations: options.maximumIterations ?? 32,
      localIterations: options.localIterations,
      betweennessScale: options.scale,
      output: {
        betweenness: importGraphBuffer(
          graph,
          'betweenness',
          betweennessBuffer,
          'float32',
          nodeCount
        ),
        closeness: importGraphBuffer(graph, 'closeness', closenessBuffer, 'float32', pivots.length),
        harmonicCloseness: importGraphBuffer(
          graph,
          'harmonic',
          harmonicBuffer,
          'float32',
          pivots.length
        ),
        reachedCounts: importGraphBuffer(graph, 'reached', reachedBuffer, 'uint32', pivots.length),
        status: {converged: importGraphBuffer(graph, 'converged', convergedBuffer, 'uint32', 1)}
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result: CentralityRun = {
    betweenness: Float32Array.from(await readFloat32(betweennessBuffer, nodeCount)),
    closeness: Float32Array.from(await readFloat32(closenessBuffer, pivots.length)),
    harmonicCloseness: Float32Array.from(await readFloat32(harmonicBuffer, pivots.length)),
    reachedCounts: Uint32Array.from(await readUint32(reachedBuffer, pivots.length)),
    converged: (await readUint32(convergedBuffer, 1))[0]
  };
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

function getAllNodes(scene: CentralityReferenceScene): number[] {
  return Array.from({length: scene.nodeCount}, (_, node) => node);
}

function getBits(values: Float32Array): number[] {
  return Array.from(new Uint32Array(values.buffer, values.byteOffset, values.length));
}

for (const [name, scene] of Object.entries(CENTRALITY_REFERENCE.scenes)) {
  const {nodeCount, edges, tolerance} = scene;

  it(`GPUNetworkCentrality matches networkx betweenness on ${name}: exact, sampled and radius`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const allNodes = getAllNodes(scene);
    // laneCount 1 (a batch per pivot), a lane count that leaves a short last batch, and one batch.
    for (const laneCount of [1, 4, nodeCount]) {
      const label = `${name} lanes ${laneCount}`;
      const exact = await runCentrality(device, nodeCount, edges, allNodes, {laneCount});
      expect(exact.converged, label).toBe(1);
      expect(
        compareCentralityValues(
          `${label} exact`,
          exact.betweenness,
          scene.betweenness.exact,
          tolerance.betweennessRelative
        )
      ).toBeUndefined();

      const sampled = await runCentrality(device, nodeCount, edges, scene.sampledPivots, {
        laneCount: Math.min(laneCount, scene.sampledPivots.length),
        scale: scene.sampledScale
      });
      expect(sampled.converged, label).toBe(1);
      expect(
        compareCentralityValues(
          `${label} sampled`,
          sampled.betweenness,
          scene.betweenness.sampled,
          tolerance.betweennessRelative,
          scene.sampledScale
        )
      ).toBeUndefined();

      const bounded = await runCentrality(device, nodeCount, edges, allNodes, {
        laneCount,
        radius: scene.radius
      });
      expect(bounded.converged, label).toBe(1);
      expect(
        compareCentralityValues(
          `${label} radius`,
          bounded.betweenness,
          scene.betweenness.radius,
          tolerance.betweennessRelative
        )
      ).toBeUndefined();

      const boundedSampled = await runCentrality(device, nodeCount, edges, scene.sampledPivots, {
        laneCount: Math.min(laneCount, scene.sampledPivots.length),
        radius: scene.radius
      });
      expect(
        compareCentralityValues(
          `${label} radius sampled`,
          boundedSampled.betweenness,
          scene.betweenness.radiusSampled,
          tolerance.betweennessRelative
        )
      ).toBeUndefined();
    }
  });

  it(`GPUNetworkCentrality matches networkx closeness and harmonic closeness on ${name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const allNodes = getAllNodes(scene);
    for (const [variant, rows, radius] of [
      ['unbounded', scene.closeness.unbounded, undefined],
      ['radius', scene.closeness.radius, scene.radius]
    ] as const) {
      const runs: CentralityRun[] = [];
      for (const laneCount of [1, 4, nodeCount]) {
        const label = `${name} ${variant} lanes ${laneCount}`;
        const run = await runCentrality(device, nodeCount, edges, allNodes, {laneCount, radius});
        expect(Array.from(run.reachedCounts), label).toEqual(rows.reachedCounts);
        expect(
          compareCentralityValues(
            `${label} closeness`,
            run.closeness,
            rows.closeness,
            tolerance.closenessRelative
          )
        ).toBeUndefined();
        expect(
          compareCentralityValues(
            `${label} harmonic`,
            run.harmonicCloseness,
            rows.harmonicCloseness,
            tolerance.harmonicRelative
          )
        ).toBeUndefined();
        runs.push(run);
      }
      // The costs are bit-identical for every lane count, and so are the per-lane reductions.
      for (const run of runs) {
        expect(Array.from(run.reachedCounts)).toEqual(Array.from(runs[0].reachedCounts));
        expect(getBits(run.closeness)).toEqual(getBits(runs[0].closeness));
        expect(getBits(run.harmonicCloseness)).toEqual(getBits(runs[0].harmonicCloseness));
      }
    }
  });
}

it('GPUNetworkCentrality is deterministic and agrees with the CPU mirror of its arithmetic', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = CENTRALITY_REFERENCE.scenes['grid'];
  const {nodeCount, edges} = scene;
  const allNodes = getAllNodes(scene);
  const first = await runCentrality(device, nodeCount, edges, allNodes, {laneCount: 6});
  const second = await runCentrality(device, nodeCount, edges, allNodes, {laneCount: 6});
  expect(getBits(second.betweenness)).toEqual(getBits(first.betweenness));
  const mirror = mirrorCentrality({nodeCount, edges, pivots: allNodes, laneCount: 6});
  // The mirror rounds like the kernels; fused multiply-add on the device may differ in the last bits.
  expect(
    compareCentralityValues(
      'grid mirror',
      first.betweenness,
      mirror.betweenness,
      scene.tolerance.betweennessRelative
    )
  ).toBeUndefined();
});

it('GPUNetworkCentrality ignores out-of-range pivots', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = CENTRALITY_REFERENCE.scenes['hand'];
  const {nodeCount, edges} = scene;
  const withInvalid = await runCentrality(device, nodeCount, edges, [0, NONE, 3, nodeCount], {
    laneCount: 3
  });
  const valid = await runCentrality(device, nodeCount, edges, [0, 3], {laneCount: 2});
  expect(Array.from(withInvalid.reachedCounts)).toEqual([
    valid.reachedCounts[0],
    0,
    valid.reachedCounts[1],
    0
  ]);
  expect(withInvalid.closeness[1]).toBe(0);
  expect(withInvalid.harmonicCloseness[3]).toBe(0);
  expect(
    compareCentralityValues(
      'invalid pivots',
      withInvalid.betweenness,
      valid.betweenness,
      scene.tolerance.betweennessRelative
    )
  ).toBeUndefined();
});

it('GPUNetworkCentrality gives the same betweenness for every localIterations and reports truncation', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // A 12 x 12 grid has chains of 22 hops: localIterations 1 needs a round per hop.
  const width = 12;
  const nodeCount = width * width;
  const edges = createCentralityGridEdges(width, 12);
  const pivots = Array.from({length: 24}, (_, row) => (row * 7) % nodeCount);
  const chained = await runCentrality(device, nodeCount, edges, pivots, {
    laneCount: 8,
    localIterations: 16
  });
  const strict = await runCentrality(device, nodeCount, edges, pivots, {
    laneCount: 8,
    localIterations: 1,
    maximumIterations: 64
  });
  expect(chained.converged).toBe(1);
  expect(strict.converged).toBe(1);
  const mirror = mirrorCentrality({nodeCount, edges, pivots, laneCount: 8});
  // At most 40 hops along a tight chain, tight out-degree at most 4: (7 + 4) * 40 + (pivots + 2)
  // units of 2^-24, the bound derived in test/oracles/network-centrality.py.
  const tolerance = ((7 + 4) * 40 + pivots.length + 2) * 2 ** -24;
  for (const run of [chained, strict]) {
    expect(
      compareCentralityValues('grid betweenness', run.betweenness, mirror.betweenness, tolerance)
    ).toBeUndefined();
  }
  const truncated = await runCentrality(device, nodeCount, edges, pivots, {
    laneCount: 8,
    localIterations: 1,
    maximumIterations: 4
  });
  expect(truncated.converged).toBe(0);
});
