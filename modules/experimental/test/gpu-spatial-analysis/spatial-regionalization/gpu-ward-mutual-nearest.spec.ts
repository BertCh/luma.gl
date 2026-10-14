// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import type {Device} from '@luma.gl/core';
import {expect, it} from 'vitest';
import {
  getGPUWardRegionsParameterValues,
  GPUWardRegions,
  GPU_WARD_STATUS
} from '../../../src/gpu-spatial-analysis/spatial-regionalization';
import {GPU_WARD_ROUND_COUNT_STATUS_WORD} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-ward-regions';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  createRandomValues,
  createRookLattice,
  findDisconnectedLabels
} from './large-lattice-invariants';
import {evaluatePartition} from './spatial-regionalization-oracle';
import {
  createCompleteAdjacency,
  createLatticeAdjacency,
  getCSR,
  runWardReference,
  WARD_FIXTURE,
  type WardFixtureScene
} from './ward-reference';

type WardRun = {
  labels: number[];
  objective: number;
  status: number[];
  milliseconds: number;
};

/** Runs `GPUWardRegions` (raw values) and returns labels, objective and the five status words. */
async function runWard(
  device: Device,
  props: {
    offsets: Uint32Array;
    neighbors: Uint32Array;
    values: Float32Array;
    columnCount: number;
    targetRegionCount: number;
    mergeSchedule: 'sequential' | 'mutual-nearest';
    maximumIterations: number;
    seed?: number;
  }
): Promise<WardRun> {
  const rowCount = props.offsets.length - 1;
  const edgeWeights = Float32Array.from(props.neighbors, () => 1);
  const statusLength = GPU_WARD_ROUND_COUNT_STATUS_WORD + 1;
  const graph = new GPUCommandGraph(device, {id: `ward-${props.mergeSchedule}`});
  const inputs = {
    offsets: createInputBuffer(device, props.offsets),
    neighbors: createInputBuffer(device, props.neighbors),
    weights: createInputBuffer(device, edgeWeights),
    values: createInputBuffer(device, props.values)
  };
  const outputs = {
    labels: createOutputBuffer(device, rowCount),
    objective: createOutputBuffer(device, 1),
    status: createOutputBuffer(device, statusLength)
  };
  const parametersBuffer = createInputBuffer(
    device,
    getGPUWardRegionsParameterValues({seed: props.seed ?? 0})
  );
  graph.add(
    new GPUWardRegions({
      weights: {
        offsets: importGraphBuffer(
          graph,
          'offsets',
          inputs.offsets,
          'uint32',
          props.offsets.length
        ),
        neighbors: importGraphBuffer(
          graph,
          'neighbors',
          inputs.neighbors,
          'uint32',
          props.neighbors.length
        ),
        weights: importGraphBuffer(graph, 'weights', inputs.weights, 'float32', edgeWeights.length)
      },
      values: importGraphBuffer(graph, 'values', inputs.values, 'float32', props.values.length),
      columnCount: props.columnCount,
      standardize: false,
      targetRegionCount: props.targetRegionCount,
      maximumIterations: props.maximumIterations,
      mergeSchedule: props.mergeSchedule,
      parameters: importGraphBuffer(graph, 'parameters', parametersBuffer, 'uint32', 1),
      output: {
        labels: importGraphBuffer(graph, 'labels', outputs.labels, 'uint32', rowCount),
        objective: importGraphBuffer(graph, 'objective', outputs.objective, 'float32', 1),
        status: importGraphBuffer(graph, 'status', outputs.status, 'uint32', statusLength)
      }
    })
  );
  const compiled = graph.compile();
  const start = performance.now();
  submitGraph(device, compiled, undefined);
  const labels = await readUint32(outputs.labels, rowCount);
  const milliseconds = performance.now() - start;
  const result = {
    labels,
    objective: (await readFloat32(outputs.objective, 1))[0],
    status: await readUint32(outputs.status, statusLength),
    milliseconds
  };
  compiled.destroy();
  Object.values(inputs).forEach(buffer => buffer.destroy());
  parametersBuffer.destroy();
  Object.values(outputs).forEach(buffer => buffer.destroy());
  return result;
}

function runScene(
  device: Device,
  scene: WardFixtureScene,
  adjacency: readonly (readonly number[])[],
  targetRegionCount: number,
  mergeSchedule: 'sequential' | 'mutual-nearest'
) {
  const {offsets, neighbors} = getCSR(adjacency);
  return runWard(device, {
    offsets,
    neighbors,
    values: Float32Array.from(scene.values),
    columnCount: scene.columnCount,
    targetRegionCount,
    mergeSchedule,
    maximumIterations: scene.rowCount - targetRegionCount
  });
}

it('GPUWardRegions mutual-nearest equals scipy ward + fcluster on the separated complete scene', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = WARD_FIXTURE.separated;
  const adjacency = createCompleteAdjacency(scene.rowCount);
  for (const cut of scene.cuts) {
    for (const mergeSchedule of ['sequential', 'mutual-nearest'] as const) {
      const result = await runScene(device, scene, adjacency, cut.targetRegionCount, mergeSchedule);
      const name = `${mergeSchedule} k=${cut.targetRegionCount}`;
      expect(result.labels, name).toEqual(cut.labels);
      // float32 moment sums over at most 28 rows: relative error well below the tolerance.
      expect(Math.abs(result.objective - cut.objective), name).toBeLessThan(
        WARD_FIXTURE.tolerances.objectiveRelative * cut.objective
      );
      expect(result.status[GPU_WARD_STATUS.iterationCount], name).toBe(
        scene.rowCount - cut.targetRegionCount
      );
      expect(result.status[GPU_WARD_STATUS.converged], name).toBe(1);
      expect(result.status[GPU_WARD_STATUS.iterationLimitReached], name).toBe(0);
      expect(result.status[GPU_WARD_STATUS.invalidCount], name).toBe(0);
    }
  }
});

it('GPUWardRegions sequential equals scipy (random complete) and scikit-learn (constrained lattice)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const [scene, adjacency] of [
    [WARD_FIXTURE.random, createCompleteAdjacency(WARD_FIXTURE.random.rowCount)],
    [
      WARD_FIXTURE.lattice,
      createLatticeAdjacency(WARD_FIXTURE.lattice.width!, WARD_FIXTURE.lattice.width!)
    ]
  ] as const) {
    const cut = scene.cuts[0];
    const result = await runScene(device, scene, adjacency, cut.targetRegionCount, 'sequential');
    expect(result.labels).toEqual(cut.labels);
    expect(Math.abs(result.objective - cut.objective)).toBeLessThan(
      WARD_FIXTURE.tolerances.objectiveRelative * cut.objective
    );
  }
});

it('GPUWardRegions mutual-nearest equals the CPU mirror on the random and lattice scenes', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // The mirror is the float64 schedule; the scenes' merge margins (fixture marginRatio) keep
  // float32 costs from flipping a choice, and mutual-nearest compares the same cost orders.
  for (const [scene, adjacency] of [
    [WARD_FIXTURE.random, createCompleteAdjacency(WARD_FIXTURE.random.rowCount)],
    [
      WARD_FIXTURE.lattice,
      createLatticeAdjacency(WARD_FIXTURE.lattice.width!, WARD_FIXTURE.lattice.width!)
    ]
  ] as const) {
    const targetRegionCount = scene.cuts[0].targetRegionCount;
    const expected = runWardReference({
      values: scene.values,
      columnCount: scene.columnCount,
      adjacency,
      targetRegionCount,
      mergeSchedule: 'mutual-nearest'
    });
    const result = await runScene(device, scene, adjacency, targetRegionCount, 'mutual-nearest');
    expect(result.status[GPU_WARD_STATUS.iterationCount]).toBe(expected.mergeCount);
    if (scene === WARD_FIXTURE.random) {
      // 40 rows with a 0.3% sequential merge margin: float32 costs keep every choice.
      expect(result.labels).toEqual(expected.labels);
      expect(result.status[GPU_WARD_ROUND_COUNT_STATUS_WORD]).toBe(expected.roundCount);
    } else {
      // Reciprocal choices on the lattice have no fixture margin, so a float32 flip may move a
      // merge into the next round: bound the round count and the objective, not the labels.
      expect(
        Math.abs(result.status[GPU_WARD_ROUND_COUNT_STATUS_WORD] - expected.roundCount)
      ).toBeLessThanOrEqual(3);
    }
    expect(Math.abs(result.objective - expected.objective)).toBeLessThan(
      (scene === WARD_FIXTURE.random ? 1e-4 : 2e-2) * Math.max(expected.objective, 1)
    );
  }
});

it('GPUWardRegions mutual-nearest needs about log2(n) rounds on a 64 x 64 lattice', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const width = 64;
  const rowCount = width * width;
  const targetRegionCount = 1;
  const {offsets, neighbors} = createRookLattice(width, width);
  const values = createRandomValues(rowCount, 77);
  const result = await runWard(device, {
    offsets,
    neighbors,
    values,
    columnCount: 1,
    targetRegionCount,
    mergeSchedule: 'mutual-nearest',
    maximumIterations: 256
  });
  const rounds = result.status[GPU_WARD_ROUND_COUNT_STATUS_WORD];
  // eslint-disable-next-line no-console
  console.log(`ward mutual-nearest 64x64: rounds=${rounds} ms=${result.milliseconds.toFixed(1)}`);
  expect(result.status[GPU_WARD_STATUS.converged]).toBe(1);
  expect(result.status[GPU_WARD_STATUS.iterationCount]).toBe(rowCount - 1);
  // Each round merges every reciprocal pair, at least a third of the regions on a lattice in
  // practice; 12 * log2(n) = 144 is a loose bound compared with 4095 sequential merges.
  expect(rounds).toBeGreaterThan(0);
  expect(rounds).toBeLessThanOrEqual(12 * Math.log2(rowCount));
  expect(new Set(result.labels).size).toBe(1);
}, 120_000);

it('GPUWardRegions mutual-nearest stops exactly at the target with a contiguous exact partition', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const width = 40;
  const rowCount = width * width;
  const {offsets, neighbors} = createRookLattice(width, width);
  const values = createRandomValues(rowCount * 2, 91);
  // Targets that fall in the middle of a round exercise the radix-select guard.
  for (const targetRegionCount of [700, 123, 7]) {
    const result = await runWard(device, {
      offsets,
      neighbors,
      values,
      columnCount: 2,
      targetRegionCount,
      mergeSchedule: 'mutual-nearest',
      maximumIterations: 128
    });
    expect(new Set(result.labels).size, `k=${targetRegionCount}`).toBe(targetRegionCount);
    expect(result.status[GPU_WARD_STATUS.converged]).toBe(1);
    expect(result.status[GPU_WARD_STATUS.iterationCount]).toBe(rowCount - targetRegionCount);
    expect(findDisconnectedLabels(offsets, neighbors, result.labels)).toEqual([]);
    const exact = evaluatePartition(values, 2, result.labels).withinSsd;
    expect(Math.abs(result.objective - exact)).toBeLessThan(1e-3 * Math.max(exact, 1));
  }
});

it('GPUWardRegions mutual-nearest reports exhausted merges on disconnected weights', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runWard(device, {
    offsets: Uint32Array.from([0, 1, 2, 3, 4]),
    neighbors: Uint32Array.from([1, 0, 3, 2]),
    values: Float32Array.from([0, 1, 10, 11]),
    columnCount: 1,
    targetRegionCount: 1,
    mergeSchedule: 'mutual-nearest',
    maximumIterations: 3
  });
  expect(result.labels).toEqual([0, 0, 2, 2]);
  expect(result.status.slice(0, 4)).toEqual([2, 0, 0, 0]);
  expect(result.status[GPU_WARD_ROUND_COUNT_STATUS_WORD]).toBe(2);
});

it('GPUWardRegions A/B: sequential vs mutual-nearest on a 4096-row lattice to 16 regions', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const width = 64;
  const rowCount = width * width;
  const targetRegionCount = 16;
  const {offsets, neighbors} = createRookLattice(width, width);
  const values = createRandomValues(rowCount * 2, 5);
  const runs = {} as Record<'sequential' | 'mutual-nearest', WardRun>;
  for (const mergeSchedule of ['sequential', 'mutual-nearest'] as const) {
    runs[mergeSchedule] = await runWard(device, {
      offsets,
      neighbors,
      values,
      columnCount: 2,
      targetRegionCount,
      mergeSchedule,
      // Sequential unrolls one iteration per merge; mutual-nearest needs far fewer rounds.
      maximumIterations: mergeSchedule === 'sequential' ? rowCount - targetRegionCount : 256
    });
    const run = runs[mergeSchedule];
    const exact = evaluatePartition(values, 2, run.labels).withinSsd;
    // eslint-disable-next-line no-console
    console.log(
      `ward A/B ${mergeSchedule}: ms=${run.milliseconds.toFixed(1)} merges=${run.status[GPU_WARD_STATUS.iterationCount]} rounds=${run.status[GPU_WARD_ROUND_COUNT_STATUS_WORD]} objective=${run.objective.toFixed(3)} exact=${exact.toFixed(3)}`
    );
    expect(new Set(run.labels).size).toBe(targetRegionCount);
    expect(findDisconnectedLabels(offsets, neighbors, run.labels)).toEqual([]);
  }
}, 600_000);
