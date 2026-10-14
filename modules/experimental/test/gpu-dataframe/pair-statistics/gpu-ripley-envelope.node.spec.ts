// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPURipleyEnvelope,
  GPU_RIPLEY_ENVELOPE_MAXIMUM_SIMULATION_COUNT,
  type GPURipleyEnvelopeOutput,
  type GPURipleyEnvelopeProps
} from '../../../src/gpu-dataframe/pair-statistics/gpu-ripley-envelope';
import {
  getGPURipleyParameterValues,
  GPU_RIPLEY_ENVELOPE_STATISTICS_LENGTH,
  GPU_RIPLEY_PARAMETER_LENGTH
} from '../../../src/gpu-dataframe/pair-statistics/ripley-parameters';
import {WgslReflect} from 'wgsl_reflect';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';
import {createRandom} from './pair-statistics-oracle';
import {computeRipleyOnCPU} from './ripley-oracle';
import {
  computeRipleyEnvelopeOnCPU,
  simulateCSROnCPU,
  type RipleyEnvelopeOracleStatistic
} from './ripley-envelope-oracle';
import {
  ENVELOPE_FIXTURE_CASES,
  ENVELOPE_FIXTURE_K_OBSERVED,
  ENVELOPE_FIXTURE_K_SIMULATED,
  ENVELOPE_FIXTURE_RADII
} from './ripley-envelope-fixture';

let serial = 0;

type Overrides = Omit<Partial<GPURipleyEnvelopeProps>, 'output'> & {
  output?: Partial<GPURipleyEnvelopeOutput>;
};

function createProps(graph: GPUCommandGraph, overrides: Overrides = {}): GPURipleyEnvelopeProps {
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  return {
    positions: view('float32x2', 10),
    parameters: view('float32', GPU_RIPLEY_PARAMETER_LENGTH),
    gridSize: [4, 4],
    radiusCount: 5,
    simulationCount: 4,
    ...overrides,
    output: {lower: view('float32', 5), upper: view('float32', 5), ...overrides.output}
  };
}

function expectThrows(overrides: (graph: GPUCommandGraph) => Overrides, message: RegExp): void {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  expect(() => new GPURipleyEnvelope(createProps(graph, overrides(graph)))).toThrow(message);
  device.destroy();
}

it('getGPURipleyParameterValues packs and validates the Monte Carlo seed', () => {
  const bounds = [0, 0, 1, 1] as const;
  const getSeedWord = (values: Float32Array): number => new Uint32Array(values.buffer)[6];
  expect(getSeedWord(getGPURipleyParameterValues({bounds, maximumDistance: 1}))).toBe(0);
  // The seed is a uint32 word (B-22): values past 2^24, including NaN bit patterns, survive.
  for (const seed of [12345, 2 ** 24 + 1, 0x7fc00001, 0xffffffff]) {
    expect(getSeedWord(getGPURipleyParameterValues({bounds, maximumDistance: 1, seed}))).toBe(seed);
  }
  for (const seed of [-1, 1.5, 2 ** 32, NaN]) {
    expect(() => getGPURipleyParameterValues({bounds, maximumDistance: 1, seed})).toThrow(/seed/);
  }
});

it('GPURipleyEnvelope validates its props', () => {
  const view = (graph: GPUCommandGraph, length: number) =>
    createTransientView(graph, `bad-${serial++}`, 'float32', length);
  expectThrows(() => ({simulationCount: 1}), /simulationCount/);
  expectThrows(
    () => ({simulationCount: GPU_RIPLEY_ENVELOPE_MAXIMUM_SIMULATION_COUNT + 1}),
    /simulationCount/
  );
  expectThrows(() => ({simulationCount: 3, rank: 2}), /simulationCount/);
  expectThrows(() => ({rank: 0}), /rank/);
  expectThrows(() => ({radiusCount: 0}), /radiusCount/);
  expectThrows(() => ({statistic: 'f' as never}), /statistic/);
  expectThrows(graph => ({output: {lower: view(graph, 4)}}), /output\.lower must hold/);
  expectThrows(
    graph => ({output: {statistics: view(graph, GPU_RIPLEY_ENVELOPE_STATISTICS_LENGTH - 1)}}),
    /output\.statistics/
  );
  expectThrows(
    () => ({output: {lower: undefined, upper: undefined}}),
    /output needs at least one view/
  );
  expectThrows(
    () => ({spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}}),
    /project positions to a planar frame/
  );
  expectThrows(graph => {
    const shared = view(graph, 8);
    return {parameters: shared, output: {lower: shared}};
  }, /share buffers|alias|overlap|disjoint/i);
});

it('GPURipleyEnvelope emits one Ripley pass per simulation with deterministic node IDs', () => {
  const device = createNullWebGPUDevice();
  const ids = () => {
    const graph = new GPUCommandGraph(device);
    const view = (length: number) => createTransientView(graph, `e-${serial++}`, 'float32', length);
    return new GPURipleyEnvelope({
      ...createProps(graph),
      id: 'e',
      simulationCount: 3,
      output: {
        ...createProps(graph).output,
        globalLower: view(5),
        globalUpper: view(5),
        theoretical: view(5),
        statistics: view(GPU_RIPLEY_ENVELOPE_STATISTICS_LENGTH)
      }
    })
      .getCommandNodes(graph)
      .map(node => node.id);
  };
  const first = ids();
  expect(ids()).toEqual(first);
  expect(new Set(first).size).toBe(first.length);
  expect(first.every(id => id.startsWith('e-'))).toBe(true);
  for (const simulation of [0, 1, 2]) {
    expect(first).toContain(`e-sim-${simulation}-points`);
    expect(first).toContain(`e-sim-${simulation}-finish`);
    expect(first).toContain(`e-sim-${simulation}-gather`);
  }
  for (const id of ['e-count', 'e-deviations', 'e-summary', 'e-envelope']) {
    expect(first).toContain(id);
  }
  // No envelope-per-radius node when only the summary is requested.
  const graph = new GPUCommandGraph(device);
  const summaryOnly = new GPURipleyEnvelope({
    ...createProps(graph),
    id: 'e',
    output: {
      statistics: createTransientView(
        graph,
        'stats',
        'float32',
        GPU_RIPLEY_ENVELOPE_STATISTICS_LENGTH
      )
    }
  })
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(summaryOnly).not.toContain('e-envelope');
  device.destroy();
});

it('GPURipleyEnvelope rejects views from another graph', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const other = new GPUCommandGraph(device);
  const envelope = new GPURipleyEnvelope(
    createProps(graph, {output: {lower: createTransientView(other, 'foreign', 'float32', 5)}})
  );
  expect(() => envelope.getCommandNodes(graph)).toThrow(/target graph/);
  device.destroy();
});

// Oracle: <scratchpad>/fix/work/D/point-patterns/envelope_ranking.py (numpy 2.5.4), embedded in
// ripley-envelope-fixture.ts.
for (const rank of [1, 2]) {
  for (const statistic of ['k', 'l', 'l-minus-r'] as RipleyEnvelopeOracleStatistic[]) {
    it(`computeRipleyEnvelopeOnCPU matches numpy for rank ${rank}, ${statistic}`, () => {
      const expected = ENVELOPE_FIXTURE_CASES[`${rank}:${statistic}`];
      const result = computeRipleyEnvelopeOnCPU(
        ENVELOPE_FIXTURE_K_SIMULATED,
        ENVELOPE_FIXTURE_K_OBSERVED,
        ENVELOPE_FIXTURE_RADII,
        rank,
        statistic
      );
      const scale = (value: number) => 1e-9 * Math.max(1, Math.abs(value));
      for (const name of ['lower', 'upper', 'globalLower', 'globalUpper', 'theoretical'] as const) {
        expected[name].forEach((value, b) => {
          expect(Math.abs(result[name][b] - value), `${name} ${b}`).toBeLessThan(scale(value));
        });
      }
      expect(result.critical).toBeCloseTo(expected.critical, 12);
      expect(result.observedDeviation).toBeCloseTo(expected.observedDeviation, 12);
      expect(result.pValue).toBeCloseTo(expected.pValue, 12);
      result.deviations.forEach((value, index) => {
        expect(value).toBeCloseTo(expected.deviations[index], 12);
      });
    });
  }
}

it('the global envelope contains the observed curve exactly when its deviation is at most critical', () => {
  const result = computeRipleyEnvelopeOnCPU(
    ENVELOPE_FIXTURE_K_SIMULATED,
    ENVELOPE_FIXTURE_K_OBSERVED,
    ENVELOPE_FIXTURE_RADII,
    1,
    'l-minus-r'
  );
  const observedL = ENVELOPE_FIXTURE_K_OBSERVED.map(kValue => Math.sqrt(kValue / Math.PI));
  const inside = observedL.every(
    (lValue, b) =>
      lValue - ENVELOPE_FIXTURE_RADII[b] >= result.globalLower[b] &&
      lValue - ENVELOPE_FIXTURE_RADII[b] <= result.globalUpper[b]
  );
  expect(inside).toBe(result.observedDeviation <= result.critical);
  // Monte Carlo p-value of rank 1: the observed maximum is exceeded by at least one simulation.
  expect(result.pValue).toBeGreaterThanOrEqual(1 / (ENVELOPE_FIXTURE_K_SIMULATED.length + 1));
});

it('simulateCSROnCPU is a reproducible uniform draw in the window', () => {
  const bounds = [10, -4, 30, 6] as const;
  const first = simulateCSROnCPU(7, 3, 4000, bounds);
  expect(Array.from(simulateCSROnCPU(7, 3, 4000, bounds))).toEqual(Array.from(first));
  expect(Array.from(simulateCSROnCPU(7, 4, 4000, bounds))).not.toEqual(Array.from(first));
  expect(Array.from(simulateCSROnCPU(8, 3, 4000, bounds))).not.toEqual(Array.from(first));
  let sumX = 0;
  let sumY = 0;
  for (let row = 0; row < 4000; row++) {
    const x = first[row * 2];
    const y = first[row * 2 + 1];
    expect(x >= 10 && x <= 30 && y >= -4 && y <= 6).toBe(true);
    sumX += x;
    sumY += y;
  }
  expect(sumX / 4000).toBeGreaterThan(19);
  expect(sumX / 4000).toBeLessThan(21);
  expect(sumY / 4000).toBeGreaterThan(0.8);
  expect(sumY / 4000).toBeLessThan(1.2);
});

it('isotropic K of simulated CSR patterns averages to pi r^2 (the theoretical value)', () => {
  const bounds = [0, 0, 100, 100] as const;
  const radiusCount = 4;
  const sums = new Array<number>(radiusCount).fill(0);
  const simulationCount = 60;
  for (let simulation = 0; simulation < simulationCount; simulation++) {
    const k = computeRipleyOnCPU(
      {positions: simulateCSROnCPU(11, simulation, 200, bounds)},
      {bounds, maximumDistance: 12, edgeCorrection: 'isotropic'},
      radiusCount
    ).k;
    k.forEach((value, b) => {
      sums[b] += value / simulationCount;
    });
  }
  sums.forEach((mean, b) => {
    const radius = 3 * (b + 1);
    expect(Math.abs(mean / (Math.PI * radius * radius) - 1), `radius ${radius}`).toBeLessThan(0.1);
  });
});

it('a regular pattern leaves the global envelope of CSR simulations, a CSR one stays inside', () => {
  const bounds = [0, 0, 100, 100] as const;
  const radiusCount = 4;
  const options = {bounds, maximumDistance: 12, edgeCorrection: 'isotropic'} as const;
  const radii = [3, 6, 9, 12];
  const simulated = Array.from(
    {length: 39},
    (_, simulation) =>
      computeRipleyOnCPU(
        {positions: simulateCSROnCPU(5, simulation, 200, bounds)},
        options,
        radiusCount
      ).k
  );
  const random = createRandom(99);
  const csr = new Float32Array(400).map(() => random() * 100);
  const lattice = new Float32Array(400);
  for (let index = 0; index < 200; index++) {
    lattice[index * 2] = (index % 14) * 7 + 1.5 + ((index * 3) % 5) * 0.2;
    lattice[index * 2 + 1] = Math.floor(index / 14) * 7 + 1.5;
  }
  const test = (positions: Float32Array) =>
    computeRipleyEnvelopeOnCPU(
      simulated,
      computeRipleyOnCPU({positions}, options, radiusCount).k,
      radii,
      1,
      'l-minus-r'
    );
  expect(test(lattice).pValue).toBeLessThanOrEqual(1 / 40 + 1e-12);
  expect(test(csr).pValue).toBeGreaterThan(0.05);
});

it('GPURipleyEnvelope pipelines do not grow with simulationCount (rule 12, E-07)', () => {
  // Before: a SIMULATION_INDEX override gave two pipelines per simulation (19 at 2, 21 at 3).
  const build = (simulationCount: number) => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const nodes = new GPURipleyEnvelope(createProps(graph, {simulationCount})).getCommandNodes(
      graph
    );
    device.destroy();
    return nodes;
  };
  const counts = [2, 3, 10].map(
    simulationCount => getGPUCommandNodePipelineKeys(build(simulationCount)).size
  );
  expect(counts[1]).toBe(counts[0]);
  expect(counts[2]).toBe(counts[0]);
  for (const {id, source} of getGPUCommandNodeKernelSources(build(3))) {
    expect(() => new WgslReflect(source), id).not.toThrow();
    expect(source, id).not.toContain('SIMULATION_INDEX');
  }
});

import {createRipleyTestWindow} from './ripley-envelope-fixture';
import {simulatePolygonCSROnCPU} from './ripley-envelope-oracle';
import {ORACLE_L_SHAPE, isOracleInsideWindow} from '../../analysis-core/study-window-oracle';

it('GPURipleyEnvelope validates polygon sampling controls', () => {
  for (const maximumAttempts of [0, 257, 1.5, NaN, Infinity]) {
    expectThrows(
      () => ({maximumAttempts}),
      /ripley-envelope maximumAttempts must be an integer in \[1, 256\]/
    );
  }
  expectThrows(
    graph => ({output: {unplacedCount: createTransientView(graph, 'empty-unplaced', 'uint32', 0)}}),
    /unplacedCount must hold 1 row/
  );
  const device = createNullWebGPUDevice();
  for (const maximumAttempts of [1, 32, 256]) {
    const graph = new GPUCommandGraph(device);
    expect(new GPURipleyEnvelope(createProps(graph, {maximumAttempts}))).toBeDefined();
  }
  device.destroy();
});

it('GPURipleyEnvelope polygon kernels reuse effective parameters and fit eight bindings', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const studyWindow = createRipleyTestWindow(graph);
  const nodes = new GPURipleyEnvelope(
    createProps(graph, {
      studyWindow,
      maximumAttempts: 17,
      output: {unplacedCount: createTransientView(graph, 'failed-draws', 'uint32', 1)}
    })
  ).getCommandNodes(graph);
  const sources = getGPUCommandNodeKernelSources(nodes);
  expect(nodes.some(node => node.id === 'ripley-envelope-window-parameters')).toBe(true);
  expect(nodes.some(node => node.id === 'ripley-envelope-unplaced-clear')).toBe(true);
  for (const {id, source} of sources) {
    expect(() => new WgslReflect(source), id).not.toThrow();
    expect(new WgslReflect(source).storage.length, id).toBeLessThanOrEqual(8);
  }
  const simulation = sources.find(kernel => kernel.id === 'ripley-envelope-sim-0-points')!.source;
  expect(simulation).toContain('attempt < 17u');
  expect(simulation).toContain('simulationIndex[simulationIndexOffset], attempt, 0u');
  expect(simulation).toContain('getStudyWindowSampleCell(cell)');
  expect(simulation).toContain('atomicAdd(&unplacedCount[unplacedCountOffset], 1u)');
  expect(simulation.match(/isInsideStudyWindow\(/g)).toHaveLength(2); // declaration + one call
  expect(sources.find(kernel => kernel.id === 'ripley-envelope-sim-0-finish')!.source).toContain(
    'studyWindowSummaryValue(4u)'
  );
  const packed = getGPURipleyParameterValues({
    bounds: 'study-window',
    maximumDistance: 1,
    seed: 0xffffffff
  });
  expect(Array.from(packed.slice(0, 4)).every(Number.isNaN)).toBe(true);
  expect(new Uint32Array(packed.buffer)[6]).toBe(0xffffffff);
  device.destroy();
});

it('GPURipleyEnvelope explicit rectangle keeps every generated shader unchanged', () => {
  const device = createNullWebGPUDevice();
  const build = (studyWindow?: 'rectangle') => {
    const graph = new GPUCommandGraph(device);
    return getGPUCommandNodeKernelSources(
      new GPURipleyEnvelope(createProps(graph, {studyWindow})).getCommandNodes(graph)
    ).map(kernel => kernel.source);
  };
  expect(build('rectangle')).toEqual(build());
  expect(build().join('\n')).not.toContain('studyWindow');
  device.destroy();
});

it('polygon CSR oracle is deterministic, inside, uniform, and records exhausted draws', () => {
  const result = simulatePolygonCSROnCPU(19, 2, 6000, ORACLE_L_SHAPE, [16, 16]);
  expect(result.unplacedCount).toBe(0);
  expect(simulatePolygonCSROnCPU(19, 2, 6000, ORACLE_L_SHAPE, [16, 16])).toEqual(result);
  let horizontalCount = 0;
  for (let row = 0; row < result.mask.length; row++) {
    const point = [result.positions[row * 2], result.positions[row * 2 + 1]] as const;
    expect(isOracleInsideWindow(ORACLE_L_SHAPE, point)).toBe(true);
    if (point[0] > 700) horizontalCount++;
  }
  expect(Math.abs(horizontalCount / 6000 - 910000 / 2310000)).toBeLessThan(0.025);
  expect(
    simulatePolygonCSROnCPU(19, 2, 10, ORACLE_L_SHAPE, [16, 16], 1, new Uint32Array()).unplacedCount
  ).toBe(10);
});
