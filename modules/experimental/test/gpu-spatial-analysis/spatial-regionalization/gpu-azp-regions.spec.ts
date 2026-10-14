// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, type GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  getGPUAZPRegionsParameterValues,
  GPUAZPRegions,
  GPU_AZP_STATUS,
  type GPUAZPRegionsProps
} from '../../../src/gpu-spatial-analysis/spatial-regionalization';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  createRandomValues,
  createRookLattice,
  findDisconnectedLabels
} from './large-lattice-invariants';
import {
  canonicalizePartition,
  evaluatePartition,
  standardizeColumns
} from './spatial-regionalization-oracle';
import {requireTestDevice} from '../../utils/require-test-device';
import statisticsAzpFixture from '../../oracles/fixtures/statistics-azp.json';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import {GPULegacyAZPRegions} from './azp-exact-reachability-legacy';
import {getAdjacency, getWithinSumOfSquares, runBatchedAzpMirror} from './azp-batched-mirror';

it('GPUAZPRegions reproducibly improves a contiguous partition and reports convergence', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const offsets = Uint32Array.from([0, 1, 3, 5, 6]);
  const neighbors = Uint32Array.from([1, 0, 2, 1, 3, 2]);
  const edgeWeights = Float32Array.from([1, 1, 1, 1, 1, 1]);
  const values = Float32Array.from([0, 10, 11, 20]);
  const initialLabels = Uint32Array.from([0, 0, 1, 1]);
  const graph = new GPUCommandGraph(device, {id: 'azp-oracle'});
  const inputs = {
    offsets: createInputBuffer(device, offsets),
    neighbors: createInputBuffer(device, neighbors),
    weights: createInputBuffer(device, edgeWeights),
    values: createInputBuffer(device, values),
    initialLabels: createInputBuffer(device, initialLabels)
  };
  const outputs = {
    labels: createOutputBuffer(device, 4),
    objective: createOutputBuffer(device, 1),
    status: createOutputBuffer(device, GPU_AZP_STATUS.length)
  };
  const seed = 17;
  const parametersBuffer = createInputBuffer(device, getGPUAZPRegionsParameterValues({seed}));
  graph.add(
    new GPUAZPRegions({
      weights: {
        offsets: importGraphBuffer(graph, 'offsets', inputs.offsets, 'uint32', offsets.length),
        neighbors: importGraphBuffer(
          graph,
          'neighbors',
          inputs.neighbors,
          'uint32',
          neighbors.length
        ),
        weights: importGraphBuffer(graph, 'weights', inputs.weights, 'float32', edgeWeights.length)
      },
      values: importGraphBuffer(graph, 'values', inputs.values, 'float32', values.length),
      // Pinned raw-unit expectations: the default standardization is covered separately.
      standardize: false,
      initialLabels: importGraphBuffer(
        graph,
        'initial-labels',
        inputs.initialLabels,
        'uint32',
        initialLabels.length
      ),
      regionCapacity: 2,
      maximumIterations: 8,
      parameters: importGraphBuffer(graph, 'parameters', parametersBuffer, 'uint32', 1),
      output: {
        labels: importGraphBuffer(graph, 'labels', outputs.labels, 'uint32', 4),
        objective: importGraphBuffer(graph, 'objective', outputs.objective, 'float32', 1),
        status: importGraphBuffer(graph, 'status', outputs.status, 'uint32', GPU_AZP_STATUS.length)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const labels = await readUint32(outputs.labels, 4);
  const [objective] = await readFloat32(outputs.objective, 1);
  const status = await readUint32(outputs.status, GPU_AZP_STATUS.length);
  expect(labels).toEqual([0, 1, 1, 1]);
  const mean = (10 + 11 + 20) / 3;
  const expectedObjective = (10 - mean) ** 2 + (11 - mean) ** 2 + (20 - mean) ** 2;
  expect(objective).toBeCloseTo(expectedObjective, 4);
  expect(status[GPU_AZP_STATUS.iterationCount]).toBe(1);
  expect(status[GPU_AZP_STATUS.converged]).toBe(1);
  expect(status[GPU_AZP_STATUS.iterationLimitReached]).toBe(0);
  expect(status[GPU_AZP_STATUS.invalidCount]).toBe(0);
  compiled.destroy();
  Object.values(inputs).forEach(buffer => buffer.destroy());
  parametersBuffer.destroy();
  Object.values(outputs).forEach(buffer => buffer.destroy());
});

it('GPUAZPRegions accepts a non-leaf move when exact source connectivity is preserved', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Rows 0, 1 and 2 form a triangle in region 0. Row 0 has two same-region neighbors, but it is
  // not an articulation point because rows 1 and 2 remain directly connected after it moves.
  const offsets = Uint32Array.from([0, 3, 5, 7, 8]);
  const neighbors = Uint32Array.from([1, 2, 3, 0, 2, 0, 1, 0]);
  const edgeWeights = Float32Array.from({length: neighbors.length}, () => 1);
  const values = Float32Array.from([10, 0, 0, 10]);
  const initialLabels = Uint32Array.from([0, 0, 0, 1]);
  const graph = new GPUCommandGraph(device, {id: 'azp-articulation-oracle'});
  const inputs = {
    offsets: createInputBuffer(device, offsets),
    neighbors: createInputBuffer(device, neighbors),
    weights: createInputBuffer(device, edgeWeights),
    values: createInputBuffer(device, values),
    initialLabels: createInputBuffer(device, initialLabels)
  };
  const outputs = {
    labels: createOutputBuffer(device, 4),
    objective: createOutputBuffer(device, 1),
    status: createOutputBuffer(device, GPU_AZP_STATUS.length)
  };
  const seed = 5;
  const parametersBuffer = createInputBuffer(device, getGPUAZPRegionsParameterValues({seed}));
  graph.add(
    new GPUAZPRegions({
      weights: {
        offsets: importGraphBuffer(
          graph,
          'cycle-offsets',
          inputs.offsets,
          'uint32',
          offsets.length
        ),
        neighbors: importGraphBuffer(
          graph,
          'cycle-neighbors',
          inputs.neighbors,
          'uint32',
          neighbors.length
        ),
        weights: importGraphBuffer(
          graph,
          'cycle-weights',
          inputs.weights,
          'float32',
          edgeWeights.length
        )
      },
      values: importGraphBuffer(graph, 'cycle-values', inputs.values, 'float32', values.length),
      // Pinned raw-unit expectations: the default standardization is covered separately.
      standardize: false,
      initialLabels: importGraphBuffer(
        graph,
        'cycle-initial-labels',
        inputs.initialLabels,
        'uint32',
        initialLabels.length
      ),
      regionCapacity: 2,
      maximumIterations: 4,
      parameters: importGraphBuffer(graph, 'parameters', parametersBuffer, 'uint32', 1),
      output: {
        labels: importGraphBuffer(graph, 'cycle-labels', outputs.labels, 'uint32', 4),
        objective: importGraphBuffer(graph, 'cycle-objective', outputs.objective, 'float32', 1),
        status: importGraphBuffer(
          graph,
          'cycle-status',
          outputs.status,
          'uint32',
          GPU_AZP_STATUS.length
        )
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  expect(await readUint32(outputs.labels, 4)).toEqual([1, 0, 0, 1]);
  expect((await readFloat32(outputs.objective, 1))[0]).toBeCloseTo(0, 4);
  expect(await readUint32(outputs.status, GPU_AZP_STATUS.length)).toEqual([1, 1, 0, 0]);
  compiled.destroy();
  Object.values(inputs).forEach(buffer => buffer.destroy());
  parametersBuffer.destroy();
  Object.values(outputs).forEach(buffer => buffer.destroy());
});

it('GPUAZPRegions keeps 36 contiguous regions on a 900-row lattice', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // 900 rows exceed the 256-lane workgroup, so the candidate argmin needs a second reduction pass.
  const width = 30;
  const tile = 5;
  const rowCount = width * width;
  const regionCount = (width / tile) ** 2;
  const {offsets, neighbors, weights: edgeWeights} = createRookLattice(width, width);
  const values = createRandomValues(rowCount, 303);
  const initialLabels = Uint32Array.from({length: rowCount}, (_, row) => {
    const x = row % width;
    const y = Math.floor(row / width);
    return Math.floor(y / tile) * (width / tile) + Math.floor(x / tile);
  });
  const initialObjective = evaluatePartition(values, 1, initialLabels).withinSsd;
  const graph = new GPUCommandGraph(device, {id: 'azp-large'});
  const inputs = {
    offsets: createInputBuffer(device, offsets),
    neighbors: createInputBuffer(device, neighbors),
    weights: createInputBuffer(device, edgeWeights),
    values: createInputBuffer(device, values),
    initialLabels: createInputBuffer(device, initialLabels)
  };
  const outputs = {
    labels: createOutputBuffer(device, rowCount),
    objective: createOutputBuffer(device, 1),
    status: createOutputBuffer(device, GPU_AZP_STATUS.length)
  };
  const seed = 11;
  const parametersBuffer = createInputBuffer(device, getGPUAZPRegionsParameterValues({seed}));
  graph.add(
    new GPUAZPRegions({
      weights: {
        offsets: importGraphBuffer(graph, 'offsets', inputs.offsets, 'uint32', offsets.length),
        neighbors: importGraphBuffer(
          graph,
          'neighbors',
          inputs.neighbors,
          'uint32',
          neighbors.length
        ),
        weights: importGraphBuffer(graph, 'weights', inputs.weights, 'float32', edgeWeights.length)
      },
      values: importGraphBuffer(graph, 'values', inputs.values, 'float32', rowCount),
      // Pinned raw-unit expectations: the default standardization is covered separately.
      standardize: false,
      initialLabels: importGraphBuffer(
        graph,
        'initial-labels',
        inputs.initialLabels,
        'uint32',
        rowCount
      ),
      regionCapacity: regionCount,
      maximumIterations: 16,
      parameters: importGraphBuffer(graph, 'parameters', parametersBuffer, 'uint32', 1),
      output: {
        labels: importGraphBuffer(graph, 'labels', outputs.labels, 'uint32', rowCount),
        objective: importGraphBuffer(graph, 'objective', outputs.objective, 'float32', 1),
        status: importGraphBuffer(graph, 'status', outputs.status, 'uint32', GPU_AZP_STATUS.length)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const labels = await readUint32(outputs.labels, rowCount);
  const [objective] = await readFloat32(outputs.objective, 1);
  const status = await readUint32(outputs.status, GPU_AZP_STATUS.length);
  expect(labels.every(label => label < regionCount)).toBe(true);
  expect(new Set(labels).size).toBe(regionCount);
  expect(status[GPU_AZP_STATUS.invalidCount]).toBe(0);
  expect(findDisconnectedLabels(offsets, neighbors, labels)).toEqual([]);
  // The reported objective is the within-region SSD of the returned labels, and AZP only improves.
  const finalObjective = evaluatePartition(values, 1, labels).withinSsd;
  expect(Math.abs(objective - finalObjective)).toBeLessThan(1e-3 * finalObjective);
  expect(finalObjective).toBeLessThanOrEqual(initialObjective * (1 + 1e-6));
  expect(status[GPU_AZP_STATUS.iterationCount]).toBeGreaterThan(0);
  compiled.destroy();
  Object.values(inputs).forEach(buffer => buffer.destroy());
  parametersBuffer.destroy();
  Object.values(outputs).forEach(buffer => buffer.destroy());
});

type AzpRun = {labels: number[]; objective: number; status: number[]};

/** Runs one `GPUAZPRegions` graph on a unit-weight CSR and reads everything back. */
async function runAzp(
  device: NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>,
  scene: {
    offsets: Uint32Array;
    neighbors: Uint32Array;
    values: Float32Array;
    columnCount?: number;
    initialLabels: Uint32Array;
    regionCapacity: number;
    extensiveValues?: Float32Array;
    extensiveThreshold?: number;
    standardize?: boolean;
    maximumIterations?: number;
    /** The contributor to run; defaults to the current `GPUAZPRegions`. */
    contributor?: new (
      props: GPUAZPRegionsProps
    ) => GPUCommandNodeProducer;
  }
): Promise<AzpRun> {
  const rowCount = scene.offsets.length - 1;
  const edgeWeights = Float32Array.from({length: scene.neighbors.length}, () => 1);
  const graph = new GPUCommandGraph(device, {id: 'azp-run'});
  const inputs = {
    offsets: createInputBuffer(device, scene.offsets),
    neighbors: createInputBuffer(device, scene.neighbors),
    weights: createInputBuffer(device, edgeWeights),
    values: createInputBuffer(device, scene.values),
    initialLabels: createInputBuffer(device, scene.initialLabels),
    ...(scene.extensiveValues
      ? {
          extensiveValues: createInputBuffer(device, scene.extensiveValues),
          threshold: createInputBuffer(device, Float32Array.from([scene.extensiveThreshold ?? 0]))
        }
      : {})
  };
  const outputs = {
    labels: createOutputBuffer(device, rowCount),
    objective: createOutputBuffer(device, 1),
    status: createOutputBuffer(device, GPU_AZP_STATUS.length)
  };
  const seed = 3;
  const parametersBuffer = createInputBuffer(device, getGPUAZPRegionsParameterValues({seed}));
  const Contributor = scene.contributor ?? GPUAZPRegions;
  graph.add(
    new Contributor({
      weights: {
        offsets: importGraphBuffer(
          graph,
          'offsets',
          inputs.offsets,
          'uint32',
          scene.offsets.length
        ),
        neighbors: importGraphBuffer(
          graph,
          'neighbors',
          inputs.neighbors,
          'uint32',
          scene.neighbors.length
        ),
        weights: importGraphBuffer(graph, 'weights', inputs.weights, 'float32', edgeWeights.length)
      },
      values: importGraphBuffer(graph, 'values', inputs.values, 'float32', scene.values.length),
      columnCount: scene.columnCount,
      standardize: scene.standardize,
      extensiveValues: inputs.extensiveValues
        ? importGraphBuffer(graph, 'extensive', inputs.extensiveValues, 'float32', rowCount)
        : undefined,
      extensiveThreshold: inputs.threshold
        ? importGraphBuffer(graph, 'extensive-threshold', inputs.threshold, 'float32', 1)
        : undefined,
      initialLabels: importGraphBuffer(
        graph,
        'initial-labels',
        inputs.initialLabels,
        'uint32',
        rowCount
      ),
      regionCapacity: scene.regionCapacity,
      maximumIterations: scene.maximumIterations ?? 8,
      parameters: importGraphBuffer(graph, 'parameters', parametersBuffer, 'uint32', 1),
      output: {
        labels: importGraphBuffer(graph, 'labels', outputs.labels, 'uint32', rowCount),
        objective: importGraphBuffer(graph, 'objective', outputs.objective, 'float32', 1),
        status: importGraphBuffer(graph, 'status', outputs.status, 'uint32', GPU_AZP_STATUS.length)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    labels: await readUint32(outputs.labels, rowCount),
    objective: (await readFloat32(outputs.objective, 1))[0],
    status: await readUint32(outputs.status, GPU_AZP_STATUS.length)
  };
  compiled.destroy();
  Object.values(inputs).forEach(buffer => buffer.destroy());
  parametersBuffer.destroy();
  Object.values(outputs).forEach(buffer => buffer.destroy());
  return result;
}

it('GPUAZPRegions reports a disconnected initial region through invalidCount', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Path 0-1-2-3 with alternating labels: each region is two isolated rows, one extra component
  // per region.
  const result = await runAzp(device, {
    offsets: Uint32Array.from([0, 1, 3, 5, 6]),
    neighbors: Uint32Array.from([1, 0, 2, 1, 3, 2]),
    values: Float32Array.from([0, 1, 2, 3]),
    initialLabels: Uint32Array.from([0, 1, 0, 1]),
    regionCapacity: 2,
    standardize: false
  });
  expect(result.status[GPU_AZP_STATUS.invalidCount]).toBe(2);
  expect(result.status[GPU_AZP_STATUS.iterationCount]).toBe(0);
});

it('GPUAZPRegions treats directed weights as undirected adjacency', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Each row lists only its lower neighbor (1 -> 0, 2 -> 1, 3 -> 2). Row 2 can move to region 1
  // only through the link that row 3 lists.
  const result = await runAzp(device, {
    offsets: Uint32Array.from([0, 0, 1, 2, 3]),
    neighbors: Uint32Array.from([0, 1, 2]),
    values: Float32Array.from([0, 0, 10, 10]),
    initialLabels: Uint32Array.from([0, 0, 0, 1]),
    regionCapacity: 2,
    standardize: false
  });
  expect(result.status[GPU_AZP_STATUS.invalidCount]).toBe(0);
  expect(canonicalizePartition(result.labels)).toEqual([0, 0, 1, 1]);
  expect(result.objective).toBeCloseTo(0, 4);
});

it('GPUAZPRegions keeps a region at or above the extensive threshold', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = {
    offsets: Uint32Array.from([0, 1, 3, 5, 6]),
    neighbors: Uint32Array.from([1, 0, 2, 1, 3, 2]),
    values: Float32Array.from([0, 0, 10, 10]),
    initialLabels: Uint32Array.from([0, 0, 0, 1]),
    regionCapacity: 2,
    extensiveValues: Float32Array.from([1, 1, 1, 1]),
    standardize: false
  };
  // Without the constraint row 2 moves next to its equal; with a floor of 3 region 0 may not shrink.
  const free = await runAzp(device, {...scene, extensiveThreshold: 0});
  expect(canonicalizePartition(free.labels)).toEqual([0, 0, 1, 1]);
  const bound = await runAzp(device, {...scene, extensiveThreshold: 3});
  expect(bound.labels).toEqual([0, 0, 0, 1]);
  expect(bound.status[GPU_AZP_STATUS.iterationCount]).toBe(0);
  expect(bound.status[GPU_AZP_STATUS.converged]).toBe(1);
  expect(bound.objective).toBeCloseTo(
    evaluatePartition(scene.values, 1, bound.labels).withinSsd,
    3
  );
});

it('GPUAZPRegions standardizes columns by default and reports the z-score objective', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const width = 12;
  const rowCount = width * width;
  const {offsets, neighbors} = createRookLattice(width, width);
  const small = createRandomValues(rowCount, 51, 1);
  const large = createRandomValues(rowCount, 52, 1000);
  const values = Float32Array.from({length: rowCount * 2}, (_, index) =>
    index % 2 === 0 ? small[index >> 1] : large[index >> 1]
  );
  const tile = 4;
  const initialLabels = Uint32Array.from({length: rowCount}, (_, row) => {
    const x = row % width;
    const y = Math.floor(row / width);
    return Math.floor(y / tile) * (width / tile) + Math.floor(x / tile);
  });
  const result = await runAzp(device, {
    offsets,
    neighbors,
    values,
    columnCount: 2,
    initialLabels,
    regionCapacity: (width / tile) ** 2,
    maximumIterations: 16
  });
  expect(result.status[GPU_AZP_STATUS.invalidCount]).toBe(0);
  const expected = evaluatePartition(standardizeColumns(values, 2), 2, result.labels).withinSsd;
  expect(Math.abs(result.objective - expected)).toBeLessThan(1e-3 * Math.max(expected, 1));
});

/** Tile labels of a `width x height` lattice cut into `tile x tile` blocks. */
function createTileLabels(width: number, height: number, tile: number): Uint32Array {
  return Uint32Array.from({length: width * height}, (_, row) => {
    const x = row % width;
    const y = Math.floor(row / width);
    return Math.floor(y / tile) * Math.ceil(width / tile) + Math.floor(x / tile);
  });
}

it('GPUAZPRegions keeps 1,024 contiguous regions on a 262,144-row lattice (no row cap)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // 8 times the former 32,768-row cap. Per-candidate work and private memory no longer depend on
  // the row count, so the same kernels run unchanged.
  const width = 512;
  const tile = 16;
  const rowCount = width * width;
  const regionCapacity = (width / tile) ** 2;
  const {offsets, neighbors} = createRookLattice(width, width);
  const values = createRandomValues(rowCount, 262144);
  const initialLabels = createTileLabels(width, width, tile);
  const initialObjective = evaluatePartition(values, 1, initialLabels).withinSsd;
  const result = await runAzp(device, {
    offsets,
    neighbors,
    values,
    initialLabels,
    regionCapacity,
    standardize: false,
    maximumIterations: 256
  });
  expect(result.status[GPU_AZP_STATUS.invalidCount]).toBe(0);
  expect(result.status[GPU_AZP_STATUS.iterationCount]).toBeGreaterThan(0);
  expect(result.labels.every(label => label < regionCapacity)).toBe(true);
  expect(new Set(result.labels).size).toBe(regionCapacity);
  // Contiguity is the invariant of the local articulation test.
  expect(findDisconnectedLabels(offsets, neighbors, result.labels)).toEqual([]);
  const finalObjective = evaluatePartition(values, 1, result.labels).withinSsd;
  expect(finalObjective).toBeLessThanOrEqual(initialObjective * (1 + 1e-6));
  expect(Math.abs(result.objective - finalObjective)).toBeLessThan(1e-3 * finalObjective);
}, 600_000);

it('GPUAZPRegions ends at a local optimum of the float64 CPU mirror and keeps contiguity', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const [width, height, tile] of [
    [18, 14, 3],
    [40, 25, 5]
  ]) {
    const rowCount = width * height;
    const {offsets, neighbors} = createRookLattice(width, height);
    const values = createRandomValues(rowCount, 900 + width);
    const initialLabels = createTileLabels(width, height, tile);
    const regionCapacity = Math.ceil(width / tile) * Math.ceil(height / tile);
    const result = await runAzp(device, {
      offsets,
      neighbors,
      values,
      initialLabels,
      regionCapacity,
      standardize: false,
      maximumIterations: 256
    });
    expect(result.status[GPU_AZP_STATUS.converged], `${width}x${height}`).toBe(1);
    expect(findDisconnectedLabels(offsets, neighbors, result.labels)).toEqual([]);
    const adjacency = getAdjacency(offsets, neighbors);
    const initialObjective = getWithinSumOfSquares(values, initialLabels, regionCapacity);
    const finalObjective = getWithinSumOfSquares(values, result.labels, regionCapacity);
    expect(finalObjective).toBeLessThanOrEqual(initialObjective * (1 + 1e-6));
    // One more mirror round from the GPU labels finds no move whose gain exceeds float32 noise
    // (the GPU's own threshold is 4e-6 of the terms' magnitude; the mirror applies the same one,
    // so any move it still finds is at most a boundary case of that threshold).
    const next = runBatchedAzpMirror({
      adjacency,
      values,
      labels: result.labels,
      regionCapacity,
      maximumRounds: 1
    });
    const largestGain = Math.max(0, ...next.rounds[0].moves.map(move => move.gain));
    expect(largestGain).toBeLessThan(1e-3 * Math.max(finalObjective, 1));
  }
}, 120_000);

it('GPUAZPRegions reaches the quality of spopt AZP from the same starts (spopt oracle)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = decodeOracleFixture<{
    instances: {
      name: string;
      regionCount: number;
      offsets: number[];
      neighbors: number[];
      values: number[];
      initialLabels: number[];
      initialObjective: number;
      spoptObjective: number;
    }[];
  }>(statisticsAzpFixture);
  for (const instance of fixture.instances) {
    const offsets = Uint32Array.from(instance.offsets);
    const neighbors = Uint32Array.from(instance.neighbors);
    const result = await runAzp(device, {
      offsets,
      neighbors,
      values: Float32Array.from(instance.values),
      initialLabels: Uint32Array.from(instance.initialLabels),
      regionCapacity: instance.regionCount,
      standardize: false,
      maximumIterations: 256
    });
    expect(result.status[GPU_AZP_STATUS.converged], instance.name).toBe(1);
    expect(findDisconnectedLabels(offsets, neighbors, result.labels), instance.name).toEqual([]);
    const objective = getWithinSumOfSquares(instance.values, result.labels, instance.regionCount);
    expect(objective, instance.name).toBeLessThanOrEqual(instance.initialObjective * (1 + 1e-6));
    // Same bound as the CPU mirror spec (azp-batched-moves.node.spec.ts): both are local searches.
    expect(objective, instance.name).toBeLessThanOrEqual(1.5 * instance.spoptObjective);
    console.log(
      `${instance.name}: GPU ${objective.toFixed(2)}, spopt ${instance.spoptObjective.toFixed(2)}, ` +
        `initial ${instance.initialObjective.toFixed(2)}, rounds ` +
        `${result.status[GPU_AZP_STATUS.iterationCount]}`
    );
  }
}, 120_000);

it('GPUAZPRegions A/B: local articulation batches vs the exact-reachability single move on 16,384 rows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const width = 128;
  const tile = 8;
  const rowCount = width * width;
  const regionCapacity = (width / tile) ** 2;
  const {offsets, neighbors} = createRookLattice(width, width);
  const values = createRandomValues(rowCount, 16384);
  const initialLabels = createTileLabels(width, width, tile);
  const initialObjective = evaluatePartition(values, 1, initialLabels).withinSsd;
  const scene = {
    offsets,
    neighbors,
    values,
    initialLabels,
    regionCapacity,
    standardize: false,
    maximumIterations: 256
  };
  const outcomes: Record<string, {milliseconds: number; objective: number; run: AzpRun}> = {};
  // Alternate once to warm both pipelines, then time each variant.
  for (const [name, contributor] of [
    ['legacy', GPULegacyAZPRegions],
    ['batched', GPUAZPRegions],
    ['legacy', GPULegacyAZPRegions],
    ['batched', GPUAZPRegions]
  ] as const) {
    const start = performance.now();
    const run = await runAzp(device, {...scene, contributor});
    outcomes[name] = {
      milliseconds: performance.now() - start,
      objective: evaluatePartition(values, 1, run.labels).withinSsd,
      run
    };
  }
  for (const [name, {milliseconds, objective, run}] of Object.entries(outcomes)) {
    console.log(
      `AZP A/B 16,384 rows ${name}: ${milliseconds.toFixed(0)} ms, objective ${objective.toFixed(2)} ` +
        `(initial ${initialObjective.toFixed(2)}), iterations ` +
        `${run.status[GPU_AZP_STATUS.iterationCount]}, converged ${run.status[GPU_AZP_STATUS.converged]}`
    );
    expect(findDisconnectedLabels(offsets, neighbors, run.labels), name).toEqual([]);
    expect(objective, name).toBeLessThanOrEqual(initialObjective * (1 + 1e-6));
  }
}, 600_000);

it('GPUAZPRegions checks a 16,384-row snake region for contiguity within bounded dispatches', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // One region winding through a 128 x 128 lattice as a single path of 16,384 rows (the worst case
  // for label propagation: a diameter of n - 1). The check must pass it, and flag a cut copy.
  const width = 128;
  const rowCount = width * width;
  const order: number[] = [];
  for (let y = 0; y < width; y++) {
    for (let x = 0; x < width; x++) order.push(y * width + (y % 2 ? width - 1 - x : x));
  }
  const lists: number[][] = Array.from({length: rowCount}, () => []);
  for (let i = 0; i + 1 < rowCount; i++) {
    lists[order[i]].push(order[i + 1]);
    lists[order[i + 1]].push(order[i]);
  }
  const offsets = new Uint32Array(rowCount + 1);
  lists.forEach((list, row) => (offsets[row + 1] = offsets[row] + list.length));
  const neighbors = Uint32Array.from(lists.flat());
  const scene = {
    offsets,
    neighbors,
    values: createRandomValues(rowCount, 7),
    regionCapacity: 2,
    standardize: false,
    maximumIterations: 1
  };
  const whole = await runAzp(device, {...scene, initialLabels: new Uint32Array(rowCount)});
  expect(whole.status[GPU_AZP_STATUS.invalidCount]).toBe(0);
  const cut = new Uint32Array(rowCount);
  cut[order[rowCount / 2]] = 1;
  const split = await runAzp(device, {...scene, initialLabels: cut});
  // Region 0 is cut in two: one extra component.
  expect(split.status[GPU_AZP_STATUS.invalidCount]).toBe(1);
  expect(split.status[GPU_AZP_STATUS.iterationCount]).toBe(0);
}, 120_000);
