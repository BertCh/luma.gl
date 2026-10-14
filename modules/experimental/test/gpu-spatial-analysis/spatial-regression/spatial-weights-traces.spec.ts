// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPUSpatialWeightsTraces} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-weights-traces';
import {
  getSpatialWeightsTracesOnCPU,
  type SpatialWeightsCSR,
  type GPUSpatialWeightsTracesMode
} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-weights-traces-parameters';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  createLatticeWeights,
  createNearestNeighborScene,
  multiply,
  toDenseWeights
} from './spatial-regression-diagnostics-oracle';
import {createLagScene} from './spatial-two-stage-least-squares-oracle';
import {
  SPATIAL_WEIGHTS_TRACES_REFERENCE,
  SPATIAL_WEIGHTS_TRACES_REFERENCE_MAXIMUM_POWER
} from './spatial-weights-traces-reference';

const Q = SPATIAL_WEIGHTS_TRACES_REFERENCE_MAXIMUM_POWER;

type Csr = {offsets: Uint32Array; neighbors: Uint32Array; weights: Float32Array};
type RunOptions = {
  maximumPower: number;
  mode?: GPUSpatialWeightsTracesMode;
  includeGramTraces?: boolean;
  probeCount?: number;
  seed?: number;
  tileRowCount?: number;
  probeBatchSize?: number;
};
type Result = {
  powerTraces: Float32Array;
  powerSums: Float32Array;
  gramTraces?: Float32Array;
  approximation: number;
  infinityNorm: number;
};

/** `tr(W^q)` of the unscaled `W` from the scaled `tr((W/s)^q)` the node writes (`q` is 1-based). */
function unscale(result: {infinityNorm: number}, value: number, q: number): number {
  return value * result.infinityNorm ** q;
}

async function runTraces(device: Device, weights: Csr, options: RunOptions): Promise<Result> {
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const graph = new GPUCommandGraph(device, {id: 'traces-graph'});
  const {maximumPower} = options;
  const traceBuffer = track(createOutputBuffer(device, maximumPower));
  const sumsBuffer = track(createOutputBuffer(device, maximumPower));
  const gramBuffer = track(createOutputBuffer(device, 4));
  const approximationBuffer = track(createOutputBuffer(device, 1));
  const normBuffer = track(createOutputBuffer(device, 1));
  graph.add(
    new GPUSpatialWeightsTraces({
      id: 'traces',
      weights: {
        offsets: importGraphBuffer(
          graph,
          'offsets',
          track(createInputBuffer(device, weights.offsets)),
          'uint32',
          weights.offsets.length
        ),
        neighbors: importGraphBuffer(
          graph,
          'neighbors',
          track(createInputBuffer(device, weights.neighbors)),
          'uint32',
          weights.neighbors.length
        ),
        weights: importGraphBuffer(
          graph,
          'weights',
          track(createInputBuffer(device, weights.weights)),
          'float32',
          weights.weights.length
        )
      },
      maximumPower,
      mode: options.mode,
      probeCount: options.probeCount,
      seed: options.seed,
      tileRowCount: options.tileRowCount,
      probeBatchSize: options.probeBatchSize,
      includeGramTraces: options.includeGramTraces ?? false,
      output: {
        powerTraces: importGraphBuffer(graph, 'traces-out', traceBuffer, 'float32', maximumPower),
        powerSums: importGraphBuffer(graph, 'sums-out', sumsBuffer, 'float32', maximumPower),
        ...(options.includeGramTraces
          ? {gramTraces: importGraphBuffer(graph, 'gram-out', gramBuffer, 'float32', 4)}
          : {}),
        approximation: importGraphBuffer(graph, 'approximation', approximationBuffer, 'uint32', 1),
        infinityNorm: importGraphBuffer(graph, 'infinity-norm', normBuffer, 'float32', 1)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result: Result = {
    powerTraces: Float32Array.from(await readFloat32(traceBuffer, maximumPower)),
    powerSums: Float32Array.from(await readFloat32(sumsBuffer, maximumPower)),
    ...(options.includeGramTraces
      ? {gramTraces: Float32Array.from(await readFloat32(gramBuffer, 4))}
      : {}),
    approximation: (await readUint32(approximationBuffer, 1))[0],
    infinityNorm: (await readFloat32(normBuffer, 1))[0]
  };
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

function expectClose(
  label: string,
  actual: number,
  expected: number,
  relative: number,
  floor: number
) {
  expect(Math.abs(actual - expected), `${label} ${actual} vs ${expected}`).toBeLessThanOrEqual(
    relative * Math.max(Math.abs(expected), floor)
  );
}

const scenes = [
  ['lattice', createLagScene(12, 21, 0.5, true).weights, SPATIAL_WEIGHTS_TRACES_REFERENCE.lattice],
  [
    'rawLattice',
    createLagScene(10, 8, 0.12, false).weights,
    SPATIAL_WEIGHTS_TRACES_REFERENCE.rawLattice
  ],
  [
    'knn60',
    createNearestNeighborScene(60, 3, 13, 0.6, false).weights,
    SPATIAL_WEIGHTS_TRACES_REFERENCE.knn60
  ],
  [
    'knn90',
    createNearestNeighborScene(90, 3, 29, 0.6, false).weights,
    SPATIAL_WEIGHTS_TRACES_REFERENCE.knn90
  ]
] as const;

// Float32 chains of up to 12 sparse matrix-vector products: relative error well below 1e-4 against
// the numpy float64 matrix powers; the floors cover traces that are exactly zero (bipartite lattice).
function expectExactParity(label: string, actual: Result, reference: (typeof scenes)[number][2]) {
  expect(actual.approximation).toBe(0);
  for (let power = 0; power < Q; power++) {
    expectClose(
      `${label} tr(W^${power + 1})`,
      unscale(actual, actual.powerTraces[power], power + 1),
      reference.powerTraces[power],
      1e-4,
      1e-2
    );
    expectClose(
      `${label} 1'W^${power + 1}1`,
      unscale(actual, actual.powerSums[power], power + 1),
      reference.powerSums[power],
      1e-4,
      1e-2
    );
  }
  reference.gramTraces.forEach((expected, index) => {
    expectClose(`${label} gram ${index}`, actual.gramTraces![index], expected, 1e-4, 1e-2);
  });
}

it('GPUSpatialWeightsTraces exact mode matches numpy matrix_power traces and sums', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  for (const [name, weights, reference] of scenes) {
    const actual = await runTraces(device, weights, {
      maximumPower: Q,
      mode: 'exact',
      includeGramTraces: true
    });
    expectExactParity(name, actual, reference);
  }
});

it('GPUSpatialWeightsTraces defaults to exact below 4097 rows and is independent of tiling and batching', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const [, weights, reference] = scenes[2];
  const baseline = await runTraces(device, weights, {maximumPower: Q, includeGramTraces: true});
  expectExactParity('default', baseline, reference);
  // Tile sizes change the float32 summation grouping inside a tile only through the tile count;
  // results stay within the parity tolerance, and probe batching changes nothing at all.
  for (const tileRowCount of [1, 7, 60]) {
    expectExactParity(
      `tile ${tileRowCount}`,
      await runTraces(device, weights, {maximumPower: Q, includeGramTraces: true, tileRowCount}),
      reference
    );
  }
  const batched = await runTraces(device, weights, {
    maximumPower: Q,
    includeGramTraces: true,
    probeBatchSize: 7
  });
  expect(Array.from(batched.powerTraces)).toEqual(Array.from(baseline.powerTraces));
  expect(Array.from(batched.gramTraces!)).toEqual(Array.from(baseline.gramTraces!));
  expect(Array.from(batched.powerSums)).toEqual(Array.from(baseline.powerSums));
});

it('GPUSpatialWeightsTraces computes the Gram traces when only one power is requested', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const [, weights, reference] = scenes[3];
  for (const mode of ['exact', 'stochastic'] as const) {
    const actual = await runTraces(device, weights, {
      maximumPower: 1,
      mode,
      probeCount: 2000,
      includeGramTraces: true
    });
    if (mode === 'exact') {
      expectClose(
        'tr(W) ',
        unscale(actual, actual.powerTraces[0], 1),
        reference.powerTraces[0],
        1e-4,
        1e-2
      );
    } else {
      // Hutchinson estimate of tr(W): its variance is (1/m) sum_{i != j} w_ij (w_ij + w_ji)
      // <= 2 ||W||_F^2 / m for Rademacher probes, so allow six standard errors (as the
      // stochastic replay test below does) instead of the exact-mode tolerance.
      const frobenius = Math.sqrt(reference.gramTraces[0]);
      expect(
        Math.abs(unscale(actual, actual.powerTraces[0], 1) - reference.powerTraces[0]),
        'stochastic tr(W)'
      ).toBeLessThanOrEqual(6 * Math.sqrt(2 / 2000) * frobenius);
    }
    // tr(W'W) and sum c^2 are exact in both modes.
    expectClose(`${mode} tr(W'W)`, actual.gramTraces![0], reference.gramTraces[0], 1e-4, 1e-2);
    expectClose(`${mode} sum c^2`, actual.gramTraces![3], reference.gramTraces[3], 1e-4, 1e-2);
    if (mode === 'exact') {
      expectClose("tr(W'WW)", actual.gramTraces![1], reference.gramTraces[1], 1e-4, 1e-2);
      expectClose("tr((W'W)^2)", actual.gramTraces![2], reference.gramTraces[2], 1e-4, 1e-2);
    }
  }
});

it('GPUSpatialWeightsTraces handles self-loops in W', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // W = [[0.5, 0.5], [1, 0]]: tr(W) = 0.5, tr(W^2) = 1.25 (W^2 = [[0.75, 0.25], [0.5, 0.5]]).
  const weights = {
    offsets: Uint32Array.of(0, 2, 3),
    neighbors: Uint32Array.of(0, 1, 0),
    weights: Float32Array.of(0.5, 0.5, 1)
  };
  const actual = await runTraces(device, weights, {
    maximumPower: 2,
    mode: 'exact',
    includeGramTraces: true
  });
  expect(actual.powerTraces[0]).toBeCloseTo(0.5, 5);
  expect(actual.powerTraces[1]).toBeCloseTo(1.25, 5);
  expect(actual.gramTraces![0]).toBeCloseTo(1.5, 5);
  const cpu = getSpatialWeightsTracesOnCPU(weights, 2, {includeGramTraces: true});
  expect(actual.gramTraces![1]).toBeCloseTo(cpu.gramTraces![1], 5);
  expect(actual.gramTraces![2]).toBeCloseTo(cpu.gramTraces![2], 5);
});

it('GPUSpatialWeightsTraces stochastic mode replays the CPU mirror, is seeded and unbiased', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const [, weights, reference] = scenes[2];
  const options = {
    maximumPower: 4,
    mode: 'stochastic',
    probeCount: 2000,
    seed: 5,
    includeGramTraces: true
  } as const;
  const actual = await runTraces(device, weights, options);
  expect(actual.approximation).toBe(1);
  // Same probes as the CPU mirror: only float32 arithmetic separates them.
  const cpu = getSpatialWeightsTracesOnCPU(weights as SpatialWeightsCSR, 4, options);
  for (let power = 0; power < 4; power++) {
    expectClose(
      `mirror tr ${power}`,
      actual.powerTraces[power],
      cpu.powerTraces[power],
      1e-3,
      5e-2
    );
    expectClose(`mirror sum ${power}`, actual.powerSums[power], cpu.powerSums[power], 1e-4, 1e-2);
  }
  for (let slot = 0; slot < 4; slot++) {
    expectClose(`mirror gram ${slot}`, actual.gramTraces![slot], cpu.gramTraces![slot], 1e-3, 5e-2);
  }
  // Hutchinson standard error is at most sqrt(2 / probes) ||W^q||_F; allow six.
  const dense = toDenseWeights(weights);
  let power = dense;
  for (let q = 1; q <= 4; q++) {
    if (q > 1) {
      power = multiply(power, dense);
    }
    const frobenius = Math.sqrt(
      power.reduce((sum, row) => sum + row.reduce((a, b) => a + b * b, 0), 0)
    );
    const bound = 6 * Math.sqrt(2 / options.probeCount) * frobenius;
    expect(
      Math.abs(unscale(actual, actual.powerTraces[q - 1], q) - reference.powerTraces[q - 1]),
      `tr(W^${q})`
    ).toBeLessThan(bound + 1e-3);
  }
  // Determinism: the same seed gives bit-identical results, another seed does not.
  const again = await runTraces(device, weights, options);
  expect(Array.from(again.powerTraces)).toEqual(Array.from(actual.powerTraces));
  expect(Array.from(again.gramTraces!)).toEqual(Array.from(actual.gramTraces!));
  const other = await runTraces(device, weights, {...options, seed: 6});
  expect(Array.from(other.powerTraces)).not.toEqual(Array.from(actual.powerTraces));
  // Batching the probes does not change the estimate.
  const batched = await runTraces(device, weights, {...options, probeBatchSize: 64});
  expect(Array.from(batched.powerTraces)).toEqual(Array.from(actual.powerTraces));
});

it('GPUSpatialWeightsTraces defaults to stochastic above 4096 rows', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // A 70 x 70 row-standardized lattice has 4900 rows, above the exact-mode default limit. Only
  // the weights are built: a lag scene would densely invert a 4900 x 4900 system on the CPU.
  const weights = createLatticeWeights(70, true);
  const actual = await runTraces(device, weights, {maximumPower: 2, probeCount: 256});
  expect(actual.approximation).toBe(1);
  const exact = getSpatialWeightsTracesOnCPU(weights, 2);
  // Six standard errors, sqrt(2 / probes) ||W^2||_F, with ||W^2||_F <= sqrt(n) (each row of W^2 sums to 1).
  expect(Math.abs(actual.powerTraces[1] - exact.powerTraces[1])).toBeLessThan(
    6 * Math.sqrt(2 / 256) * Math.sqrt(weights.offsets.length)
  );
});

it('GPUSpatialWeightsTraces keeps a raw binary lattice finite at the default Q = 128', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // Row sums reach 4, so the unscaled tr(W^q) would reach about n 3.8^q and overflow float32 near
  // q = 64; the scaled chains stay bounded by n.
  const [, weights] = scenes[1];
  const rows = weights.offsets.length - 1;
  const actual = await runTraces(device, weights, {maximumPower: 128, mode: 'exact'});
  const cpu = getSpatialWeightsTracesOnCPU(weights as SpatialWeightsCSR, 128);
  expect(actual.infinityNorm).toBeCloseTo(cpu.infinityNorm, 6);
  expect(actual.infinityNorm).toBeCloseTo(4, 6);
  for (let power = 0; power < 128; power++) {
    expect(Number.isFinite(actual.powerTraces[power]), `tr ${power}`).toBe(true);
    expect(Math.abs(actual.powerTraces[power])).toBeLessThanOrEqual(rows * (1 + 1e-4));
    expectClose(`raw tr ${power}`, actual.powerTraces[power], cpu.powerTraces[power], 1e-3, 5e-2);
    expectClose(`raw sum ${power}`, actual.powerSums[power], cpu.powerSums[power], 1e-3, 5e-2);
  }
});
