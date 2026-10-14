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

import {GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode} from '../../../src/utils/wgsl-kernel-nodes';
import {createPairStatisticsWindowHarness} from './pair-statistics-harness';
import {
  ORACLE_L_SHAPE,
  ORACLE_RECTANGLE,
  getOracleQuadratAreas,
  getOracleWindowArea,
  isOracleInsideWindow,
  type OracleStudyWindow
} from '../../analysis-core/study-window-oracle';

it('polygon envelope draws are inside, spatially uniform and bitwise deterministic', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rowCount = 4096;
  const simulationCount = 2;
  // A small pair radius keeps the uniformity test inexpensive; two rows fail inclusion.
  const positions = new Float32Array(rowCount * 2).map((_, index) => (index % 2 ? 100 : 200));
  positions.set([1500, 1500, NaN, 100], (rowCount - 2) * 2);
  const includedCount = rowCount - 2;
  const harness = createPairStatisticsWindowHarness(device, {
    positions,
    window: ORACLE_L_SHAPE,
    parameterLength: GPU_RIPLEY_PARAMETER_LENGTH,
    outputs: {
      samples: {format: 'float32', length: rowCount * simulationCount * 2},
      masks: {format: 'uint32', length: rowCount * simulationCount},
      unplacedCount: {format: 'uint32', length: 1},
      statistics: {format: 'float32', length: 6}
    },
    createContributor: (views, studyWindow) => ({
      getCommandNodes(graph) {
        const nodes = new GPURipleyEnvelope({
          positions: views.positions,
          parameters: views.parameters,
          studyWindow,
          gridSize: [64, 64],
          radiusCount: 2,
          simulationCount,
          output: {
            statistics: views.outputs.statistics as never,
            unplacedCount: views.outputs.unplacedCount as never
          }
        }).getCommandNodes(graph);
        // Inspect the graph's declared simulation outputs immediately after each draw, before
        // their transient allocations can be reused. This adds no runtime public API.
        return nodes.flatMap(node => {
          const match = /^ripley-envelope-sim-(\d+)-points$/.exec(node.id);
          if (!match) return [node];
          const simulation = Number(match[1]);
          const written = node.resources!.flatMap(resource =>
            'buffer' in resource &&
            resource.buffer instanceof GraphDataView &&
            resource.usage === 'storage-read-write'
              ? [resource.buffer]
              : []
          );
          const points = written.find(view => view.format === 'float32x2')!;
          const mask = written.find(view => view.format === 'uint32' && view.length === rowCount)!;
          return [
            node,
            createWGSLKernelNode(graph, {
              id: `inspect-simulation-${simulation}`,
              operation: 'EnvelopeTest',
              variant: 'inspect',
              invocationCount: rowCount,
              bindings: [
                {name: 'points', view: points, type: 'f32', access: 'read'},
                {name: 'mask', view: mask, type: 'u32', access: 'read'},
                {name: 'samples', view: views.outputs.samples, type: 'f32', access: 'read_write'},
                {name: 'masks', view: views.outputs.masks, type: 'u32', access: 'read_write'}
              ],
              body: `let row = ${simulation * rowCount}u + index;
  samples[samplesOffset + row * 2u] = points[pointsOffset + index * 2u];
  samples[samplesOffset + row * 2u + 1u] = points[pointsOffset + index * 2u + 1u];
  masks[masksOffset + row] = mask[maskOffset + index];`
            })
          ];
        });
      }
    })
  });
  try {
    const parameters = getGPURipleyParameterValues({
      bounds: 'study-window',
      maximumDistance: 10,
      edgeCorrection: 'none',
      seed: 731
    });
    const result = await harness.run(parameters);
    expect(result.unplacedCount).toEqual([0]);
    expect(result.statistics.slice(0, 2)).toEqual([includedCount, simulationCount]);
    expect(result.masks.filter(value => value === 1)).toHaveLength(includedCount * simulationCount);
    const counts = new Array<number>(16).fill(0);
    for (let row = 0; row < rowCount * simulationCount; row++) {
      if (!result.masks[row]) continue;
      const point = [result.samples[row * 2], result.samples[row * 2 + 1]] as const;
      expect(isOracleInsideWindow(ORACLE_L_SHAPE, point), `simulation row ${row}`).toBe(true);
      counts[
        Math.min(Math.floor(point[1] / 500), 3) * 4 + Math.min(Math.floor(point[0] / 500), 3)
      ]++;
    }
    const areas = getOracleQuadratAreas(ORACLE_L_SHAPE, [0, 0, 2000, 2000], [4, 4], 100);
    let chiSquare = 0;
    let nonempty = 0;
    areas.forEach((area, index) => {
      if (!area) {
        expect(counts[index]).toBe(0);
        return;
      }
      nonempty++;
      const expected =
        (includedCount * simulationCount * area) / getOracleWindowArea(ORACLE_L_SHAPE);
      chiSquare += (counts[index] - expected) ** 2 / expected;
    });
    expect(nonempty).toBe(12);
    // chi-square(11) 0.999 quantile: p > 0.001, fixed seed.
    expect(chiSquare).toBeLessThan(31.264134);
    const again = await harness.run(parameters);
    expect(getFloatBits(again.samples)).toEqual(getFloatBits(result.samples));
    expect(getFloatBits(again.statistics)).toEqual(getFloatBits(result.statistics));
    expect(again.unplacedCount).toEqual([0]);
  } finally {
    harness.destroy();
  }
}, 240000);

it('polygon-as-rectangle envelope preserves observed K and agrees within Monte Carlo tolerance', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const window: OracleStudyWindow = {
    ...ORACLE_RECTANGLE,
    positions: new Float64Array([0, 0, 100, 0, 100, 100, 0, 100])
  };
  const positions = createCSRPattern();
  const rectangle = createHarness(device, positions);
  const polygon = createPairStatisticsWindowHarness(device, {
    positions,
    window,
    parameterLength: GPU_RIPLEY_PARAMETER_LENGTH,
    outputs: {
      k: {format: 'float32', length: RADIUS_COUNT},
      lower: {format: 'float32', length: RADIUS_COUNT},
      upper: {format: 'float32', length: RADIUS_COUNT},
      unplacedCount: {format: 'uint32', length: 1}
    },
    createContributor: (views, studyWindow) => ({
      getCommandNodes: graph => [
        ...new GPURipley({
          positions: views.positions,
          parameters: views.parameters,
          studyWindow,
          gridSize: [16, 16],
          radiusCount: RADIUS_COUNT,
          output: {k: views.outputs.k as never}
        }).getCommandNodes(graph),
        ...new GPURipleyEnvelope({
          positions: views.positions,
          parameters: views.parameters,
          studyWindow,
          gridSize: [16, 16],
          radiusCount: RADIUS_COUNT,
          simulationCount: SIMULATION_COUNT,
          statistic: 'k',
          observedK: views.outputs.k as never,
          output: {
            lower: views.outputs.lower as never,
            upper: views.outputs.upper as never,
            unplacedCount: views.outputs.unplacedCount as never
          }
        }).getCommandNodes(graph)
      ]
    })
  });
  try {
    // Border uses exact counts in both modes, so the observed curve is bit-identical.
    const frame = {...FRAME, edgeCorrection: 'border'} as const;
    const expected = await rectangle.run(getGPURipleyParameterValues(frame));
    const actual = await polygon.run(
      getGPURipleyParameterValues({...frame, bounds: 'study-window'})
    );
    expect(getFloatBits(actual.k)).toEqual(getFloatBits(expected.k));
    expect(actual.unplacedCount).toEqual([0]);
    for (const name of ['lower', 'upper'] as const) {
      actual[name].forEach((value, index) => {
        expect(Math.abs(value - expected[name][index])).toBeLessThan(
          0.3 * Math.PI * RADII[index] ** 2
        );
      });
    }
  } finally {
    rectangle.destroy();
    polygon.destroy();
  }
}, 240000);

import {getStudyWindowWGSLOptions} from '../../../src/utils/study-window-wgsl';

it('polygon envelope masks empty or unusable samplers and clears failed counts each encoding', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const invalidWindow of [false, true]) {
    const harness = createPairStatisticsWindowHarness(device, {
      positions: new Float32Array([100, 100, 200, 200, 300, 300]),
      window: ORACLE_L_SHAPE,
      parameterLength: GPU_RIPLEY_PARAMETER_LENGTH,
      outputs: {
        lower: {format: 'float32', length: 2},
        unplacedCount: {format: 'uint32', length: 1}
      },
      createContributor: (views, studyWindow) => ({
        getCommandNodes(graph) {
          const windowViews = studyWindow.getViews(graph);
          const layout = getStudyWindowWGSLOptions(windowViews).layout;
          // Fault injection models the prepared window's unusable/empty GPU result.
          const invalidate = createWGSLKernelNode(graph, {
            id: 'empty-window',
            operation: 'EnvelopeTest',
            variant: 'empty-window',
            invocationCount: 1,
            bindings: [
              {name: 'packed', view: windowViews.packed, type: 'u32', access: 'read_write'}
            ],
            body: invalidWindow
              ? `packed[packedOffset + ${layout.summary + 4}u] = 0u;`
              : `packed[packedOffset + ${layout.sampleCellCount}u] = 0u;`
          });
          return [
            invalidate,
            ...new GPURipleyEnvelope({
              positions: views.positions,
              parameters: views.parameters,
              studyWindow,
              radiusCount: 2,
              gridSize: [8, 8],
              simulationCount: 2,
              output: {
                lower: views.outputs.lower as never,
                unplacedCount: views.outputs.unplacedCount as never
              }
            }).getCommandNodes(graph)
          ];
        }
      })
    });
    try {
      const parameters = getGPURipleyParameterValues({
        bounds: 'study-window',
        maximumDistance: 100
      });
      for (let encoding = 0; encoding < 2; encoding++) {
        const result = await harness.run(parameters);
        expect(result.unplacedCount).toEqual([6]);
        expect(result.lower.every(Number.isNaN)).toBe(true);
      }
    } finally {
      harness.destroy();
    }
  }
}, 120000);
