// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPURipley} from '../../../src/gpu-dataframe/pair-statistics/gpu-ripley';
import {GPURipleyEnvelope} from '../../../src/gpu-dataframe/pair-statistics/gpu-ripley-envelope';
import {
  getGPURipleyParameterValues,
  GPU_RIPLEY_ENVELOPE_STATISTICS_LENGTH,
  GPU_RIPLEY_PARAMETER_LENGTH
} from '../../../src/gpu-dataframe/pair-statistics/ripley-parameters';
import {createPairStatisticsHarness, getFloatBits} from './pair-statistics-harness';
import {createRandom} from './pair-statistics-oracle';
import {computeRipleyOnCPU} from './ripley-oracle';
import {computeRipleyEnvelopeOnCPU, simulateCSROnCPU} from './ripley-envelope-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

const BOUNDS = [0, 0, 100, 100] as const;
const RADIUS_COUNT = 4;
const SIMULATION_COUNT = 39;
const SEED = 5;
const MAXIMUM_DISTANCE = 12;

type Readback = Record<
  'k' | 'lower' | 'upper' | 'globalLower' | 'globalUpper' | 'theoretical' | 'statistics',
  number[]
>;

/** Observed pattern K and its envelopes in one graph: the Ripley pass feeds `observedK`. */
function createHarness(device: Device, positions: Float32Array) {
  const harness = createPairStatisticsHarness(device, {
    positions,
    mask: new Uint32Array(positions.length / 2).fill(1),
    parameterLength: GPU_RIPLEY_PARAMETER_LENGTH,
    outputs: {
      k: {format: 'float32', length: RADIUS_COUNT},
      lower: {format: 'float32', length: RADIUS_COUNT},
      upper: {format: 'float32', length: RADIUS_COUNT},
      globalLower: {format: 'float32', length: RADIUS_COUNT},
      globalUpper: {format: 'float32', length: RADIUS_COUNT},
      theoretical: {format: 'float32', length: RADIUS_COUNT},
      statistics: {format: 'float32', length: GPU_RIPLEY_ENVELOPE_STATISTICS_LENGTH}
    },
    createContributor: views => {
      const ripley = new GPURipley({
        positions: views.positions,
        mask: views.mask,
        parameters: views.parameters,
        gridSize: [16, 16],
        radiusCount: RADIUS_COUNT,
        output: {k: views.outputs.k as never}
      });
      const envelope = new GPURipleyEnvelope({
        positions: views.positions,
        mask: views.mask,
        parameters: views.parameters,
        gridSize: [16, 16],
        radiusCount: RADIUS_COUNT,
        simulationCount: SIMULATION_COUNT,
        statistic: 'k',
        observedK: views.outputs.k as never,
        output: {
          lower: views.outputs.lower as never,
          upper: views.outputs.upper as never,
          globalLower: views.outputs.globalLower as never,
          globalUpper: views.outputs.globalUpper as never,
          theoretical: views.outputs.theoretical as never,
          statistics: views.outputs.statistics as never
        }
      });
      return {
        getCommandNodes: graph => [
          ...ripley.getCommandNodes(graph),
          ...envelope.getCommandNodes(graph)
        ]
      };
    }
  });
  return {
    run: async (parameters: Float32Array) => (await harness.run(parameters)) as Readback,
    destroy: () => harness.destroy()
  };
}

const FRAME = {
  bounds: BOUNDS,
  maximumDistance: MAXIMUM_DISTANCE,
  edgeCorrection: 'isotropic',
  seed: SEED
} as const;
const RADII = Array.from(
  {length: RADIUS_COUNT},
  (_, b) => (MAXIMUM_DISTANCE * (b + 1)) / RADIUS_COUNT
);

/** The same 200-point CSR and 14 x 15 near-lattice patterns as the node spec's CPU check. */
function createCSRPattern(): Float32Array {
  const random = createRandom(99);
  return new Float32Array(400).map(() => random() * 100);
}

function createLatticePattern(): Float32Array {
  const lattice = new Float32Array(400);
  for (let index = 0; index < 200; index++) {
    lattice[index * 2] = (index % 14) * 7 + 1.5 + ((index * 3) % 5) * 0.2;
    lattice[index * 2 + 1] = Math.floor(index / 14) * 7 + 1.5;
  }
  return lattice;
}

it('GPURipleyEnvelope matches the CPU mirror of its simulations and ranking', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const positions = createCSRPattern();
  const harness = createHarness(device, positions);
  try {
    const result = await harness.run(getGPURipleyParameterValues(FRAME));
    const simulatedK = Array.from(
      {length: SIMULATION_COUNT},
      (_, simulation) =>
        computeRipleyOnCPU(
          {positions: simulateCSROnCPU(SEED, simulation, 200, BOUNDS)},
          FRAME,
          RADIUS_COUNT
        ).k
    );
    const observed = computeRipleyOnCPU({positions}, FRAME, RADIUS_COUNT).k;
    const oracle = computeRipleyEnvelopeOnCPU(simulatedK, observed, RADII, 1, 'k');
    // The simulations are bit-identical Philox draws; a one-ulp difference in a coordinate
    // (fused multiply-add) can move a pair across a radius, so compare in K units with a margin.
    const margin = (value: number) => 0.03 * Math.abs(value) + 1;
    for (let b = 0; b < RADIUS_COUNT; b++) {
      expect(Math.abs(result.k[b] - observed[b]), `k ${b}`).toBeLessThan(margin(observed[b]));
      expect(Math.abs(result.lower[b] - oracle.lower[b]), `lower ${b}`).toBeLessThan(
        margin(oracle.lower[b])
      );
      expect(Math.abs(result.upper[b] - oracle.upper[b]), `upper ${b}`).toBeLessThan(
        margin(oracle.upper[b])
      );
      expect(result.theoretical[b]).toBeCloseTo(Math.PI * RADII[b] ** 2, 2);
      expect(result.lower[b]).toBeLessThanOrEqual(result.upper[b]);
    }
    const [n, valid, observedDeviation, critical, pValue, rank] = result.statistics;
    expect([n, valid, rank]).toEqual([200, SIMULATION_COUNT, 1]);
    expect(Math.abs(critical - oracle.critical)).toBeLessThan(0.3);
    expect(Math.abs(observedDeviation - oracle.observedDeviation)).toBeLessThan(0.3);
    expect(pValue).toBeGreaterThanOrEqual(1 / (SIMULATION_COUNT + 1));
    // Deterministic: the same seed reproduces every output bit for bit; another seed does not.
    const again = await harness.run(getGPURipleyParameterValues(FRAME));
    expect(getFloatBits(again.upper)).toEqual(getFloatBits(result.upper));
    expect(getFloatBits(again.statistics)).toEqual(getFloatBits(result.statistics));
    const other = await harness.run(getGPURipleyParameterValues({...FRAME, seed: SEED + 1}));
    expect(getFloatBits(other.upper)).not.toEqual(getFloatBits(result.upper));
    // The seed is a uint32 word: 2^24 and 2^24 + 1 round to one float32, but are two streams.
    const low = await harness.run(getGPURipleyParameterValues({...FRAME, seed: 2 ** 24}));
    const high = await harness.run(getGPURipleyParameterValues({...FRAME, seed: 2 ** 24 + 1}));
    expect(getFloatBits(high.statistics)).not.toEqual(getFloatBits(low.statistics));
  } finally {
    harness.destroy();
  }
}, 240000);

it('GPURipleyEnvelope keeps a CSR pattern inside the global envelope and rejects a regular one', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const [name, positions, inside] of [
    ['csr', createCSRPattern(), true],
    ['lattice', createLatticePattern(), false]
  ] as const) {
    const harness = createHarness(device, positions);
    try {
      const result = await harness.run(getGPURipleyParameterValues(FRAME));
      const [, , observedDeviation, critical, pValue] = result.statistics;
      expect(observedDeviation <= critical, `${name} inside global envelope`).toBe(inside);
      if (inside) {
        expect(pValue, `${name} p`).toBeGreaterThan(0.05);
      } else {
        expect(pValue, `${name} p`).toBeLessThanOrEqual(1 / (SIMULATION_COUNT + 1) + 1e-6);
      }
      for (let b = 0; b < RADIUS_COUNT; b++) {
        expect(result.globalLower[b]).toBeLessThan(result.theoretical[b]);
        expect(result.globalUpper[b]).toBeGreaterThan(result.theoretical[b]);
      }
    } finally {
      harness.destroy();
    }
  }
}, 240000);
