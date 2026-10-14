// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  GPUNetworkCostWeights,
  getGPUNetworkCostWeightsParameterValues,
  type GPUNetworkCostWeightsSettings
} from '../../../src/gpu-network/network-cost-weights';
import {GPUCatchmentAccessibility} from '../../../src/gpu-spatial-analysis/catchment-accessibility/gpu-catchment-accessibility';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import fixture from '../../oracles/fixtures/network-cost-weights-2sfca.json';
import {getNetworkCostWeightsReference} from './network-cost-weights-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

/**
 * Weight tolerance. Weights are at most 1 for these decays (power uses `minimumCost` 1). WGSL `exp`
 * is accurate to `3 + 2|x|` ulp and the fixtures keep `|x| <= 4`, so an exponential weight is within
 * 11 ulp; the Gaussian adds the subtraction of `exp(-1/2)` and the division by `1 - exp(-1/2)`
 * (about 2.5x amplification of an absolute error of 12 ulp at 1). `pow` is `exp2(y log2 x)` with the
 * same bound on `|y log2 x| <= 4`. 32 ulp at 1 is `32 * 2^-24`, below 2e-6.
 */
const WEIGHT_TOLERANCE = 2e-6;

type WeightsRun = {
  offsets: number[];
  neighbors: number[];
  weights: number[];
  distances: number[];
  overflow: number;
  requiredCount: number;
};

/** Builds, encodes once and reads back one cost-weights contributor. */
async function runCostWeights(
  device: Device,
  props: {
    costs: Float32Array;
    rowCount: number;
    columns?: Uint32Array;
    orientation: 'rows' | 'columns';
    settings: GPUNetworkCostWeightsSettings;
    capacity: number;
  }
): Promise<WeightsRun> {
  const graph = new GPUCommandGraph(device, {id: 'cost-weights'});
  const buffers: Buffer[] = [];
  const input = (values: Float32Array | Uint32Array) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return buffer;
  };
  const output = (length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return buffer;
  };
  const columnCount = props.costs.length / props.rowCount;
  const locationCount = props.columns?.length ?? columnCount;
  const rows = props.orientation === 'rows' ? props.rowCount : locationCount;
  const offsetsBuffer = output(rows + 1);
  const neighborsBuffer = output(props.capacity);
  const weightsBuffer = output(props.capacity);
  const distancesBuffer = output(props.capacity);
  const overflowBuffer = output(1);
  const requiredCountBuffer = output(1);
  const parameters = getGPUNetworkCostWeightsParameterValues(props.settings);
  graph.add(
    new GPUNetworkCostWeights({
      id: 'weights',
      costs: importGraphBuffer(graph, 'costs', input(props.costs), 'float32', props.costs.length),
      rowCount: props.rowCount,
      columns: props.columns
        ? importGraphBuffer(graph, 'columns', input(props.columns), 'uint32', props.columns.length)
        : undefined,
      orientation: props.orientation,
      parameters: importGraphBuffer(graph, 'parameters', input(parameters), 'float32', 4),
      output: {
        offsets: importGraphBuffer(graph, 'offsets', offsetsBuffer, 'uint32', rows + 1),
        neighbors: importGraphBuffer(graph, 'neighbors', neighborsBuffer, 'uint32', props.capacity),
        weights: importGraphBuffer(graph, 'weights', weightsBuffer, 'float32', props.capacity),
        distances: importGraphBuffer(
          graph,
          'distances',
          distancesBuffer,
          'float32',
          props.capacity
        ),
        status: {
          overflow: importGraphBuffer(graph, 'overflow', overflowBuffer, 'uint32', 1),
          requiredCount: importGraphBuffer(
            graph,
            'required-count',
            requiredCountBuffer,
            'uint32',
            1
          )
        }
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const offsets = await readUint32(offsetsBuffer, rows + 1);
  const used = offsets[rows];
  const run: WeightsRun = {
    offsets,
    neighbors: (await readUint32(neighborsBuffer, props.capacity)).slice(0, used),
    weights: (await readFloat32(weightsBuffer, props.capacity)).slice(0, used),
    distances: (await readFloat32(distancesBuffer, props.capacity)).slice(0, used),
    overflow: (await readUint32(overflowBuffer, 1))[0],
    requiredCount: (await readUint32(requiredCountBuffer, 1))[0]
  };
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return run;
}

/** A 6 x 37 matrix with unreached, NaN, negative and zero costs. */
function createMatrix(): Float32Array {
  const rowCount = 6;
  const columnCount = 37;
  const costs = new Float32Array(rowCount * columnCount);
  let state = 12345;
  for (let index = 0; index < costs.length; index++) {
    state = (Math.imul(state, 1103515245) + 12345) >>> 0;
    costs[index] = Math.fround(((state >>> 8) / 2 ** 24) * 8);
  }
  costs[3] = Infinity;
  costs[40] = Number.NaN;
  costs[41] = -2;
  costs[77] = 0;
  costs[100] = 4;
  return costs;
}

it('GPUNetworkCostWeights matches the CPU reference for every orientation and decay', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const costs = createMatrix();
  const columns = Uint32Array.from([36, 0, 5, 99, 12, 12, 3, 20]);
  const cases: {
    orientation: 'rows' | 'columns';
    columns?: Uint32Array;
    settings: GPUNetworkCostWeightsSettings;
  }[] = [
    {orientation: 'rows', settings: {costLimit: 4}},
    {orientation: 'columns', settings: {costLimit: 4, decay: 'exponential', beta: 0.5}},
    {orientation: 'rows', columns, settings: {costLimit: 6, decay: 'gaussian'}},
    {orientation: 'columns', columns, settings: {costLimit: Infinity, decay: 'power', beta: 1.2}}
  ];
  for (const testCase of cases) {
    const capacity = 6 * 37;
    const expected = getNetworkCostWeightsReference({
      costs,
      rowCount: 6,
      ...testCase,
      capacity
    });
    const actual = await runCostWeights(device, {costs, rowCount: 6, ...testCase, capacity});
    const label = `${testCase.orientation} ${testCase.settings.decay ?? 'none'}`;
    expect(actual.offsets, label).toEqual(expected.offsets);
    expect(actual.neighbors, label).toEqual(expected.neighbors);
    expect(actual.distances, label).toEqual(expected.distances.map(Math.fround));
    expect(actual.requiredCount, label).toBe(expected.requiredCount);
    expect(actual.overflow, label).toBe(0);
    actual.weights.forEach((weight, slot) => {
      expect(
        Math.abs(weight - expected.weights[slot]),
        `${label} slot ${slot}`
      ).toBeLessThanOrEqual(WEIGHT_TOLERANCE);
    });
  }
});

it('GPUNetworkCostWeights truncates at the capacity with consistent offsets', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const costs = createMatrix();
  const settings = {costLimit: 4};
  const expected = getNetworkCostWeightsReference({
    costs,
    rowCount: 6,
    orientation: 'columns',
    settings,
    capacity: 50
  });
  expect(expected.requiredCount).toBeGreaterThan(50);
  const actual = await runCostWeights(device, {
    costs,
    rowCount: 6,
    orientation: 'columns',
    settings,
    capacity: 50
  });
  expect(actual.offsets).toEqual(expected.offsets);
  expect(actual.neighbors).toEqual(expected.neighbors);
  expect(actual.overflow).toBe(1);
  expect(actual.requiredCount).toBe(expected.requiredCount);
});

it('GPUNetworkCostWeights feeds GPUCatchmentAccessibility, matching PySAL access', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {facilityCount, demandCount, costLimit, beta} = fixture;
  const costs = Float32Array.from(fixture.costs, cost => (cost === null ? Infinity : cost));
  const decays = {
    none: {costLimit},
    exponential: {costLimit, decay: 'exponential', beta},
    gaussian: {costLimit, decay: 'gaussian'}
  } as const satisfies Record<keyof typeof fixture.accessibility, GPUNetworkCostWeightsSettings>;
  // 2SFCA and 3SFCA (Wan et al. 2012, selection weight times decay, G_ij W_ij) for every decay.
  const runs = (['2sfca', '3sfca'] as const).flatMap(method =>
    Object.keys(decays).map(name => ({name, method}))
  );
  for (const {name, method} of runs) {
    const settings: GPUNetworkCostWeightsSettings = decays[name as keyof typeof decays];
    const graph = new GPUCommandGraph(device, {id: `catchment-${method}-${name}`});
    const buffers: Buffer[] = [];
    const input = (values: Float32Array) => {
      const buffer = createInputBuffer(device, values);
      buffers.push(buffer);
      return buffer;
    };
    const capacity = facilityCount * demandCount;
    const parameters = importGraphBuffer(
      graph,
      'parameters',
      input(getGPUNetworkCostWeightsParameterValues(settings)),
      'float32',
      4
    );
    const costsView = importGraphBuffer(graph, 'costs', input(costs), 'float32', costs.length);
    const createWeights = (orientation: 'rows' | 'columns', rows: number) => {
      const contributor = new GPUNetworkCostWeights({
        id: `${orientation}-weights`,
        costs: costsView,
        rowCount: facilityCount,
        orientation,
        parameters,
        output: {
          offsets: createTransientView(graph, `${orientation}-offsets`, 'uint32', rows + 1),
          neighbors: createTransientView(graph, `${orientation}-neighbors`, 'uint32', capacity),
          weights: createTransientView(graph, `${orientation}-weights`, 'float32', capacity)
        }
      });
      graph.add(contributor);
      return contributor.outputWeights;
    };
    const facilityWeights = createWeights('rows', facilityCount);
    const demandWeights = createWeights('columns', demandCount);
    const accessibilityBuffer = createOutputBuffer(device, demandCount);
    buffers.push(accessibilityBuffer);
    graph.add(
      new GPUCatchmentAccessibility({
        id: 'catchment',
        method,
        supply: importGraphBuffer(
          graph,
          'supply',
          input(Float32Array.from(fixture.supply)),
          'float32',
          facilityCount
        ),
        demand: importGraphBuffer(
          graph,
          'demand',
          input(Float32Array.from(fixture.demand)),
          'float32',
          demandCount
        ),
        facilityWeights,
        demandWeights,
        output: {
          accessibility: importGraphBuffer(
            graph,
            'accessibility',
            accessibilityBuffer,
            'float32',
            demandCount
          )
        }
      })
    );
    const compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    const actual = await readFloat32(accessibilityBuffer, demandCount);
    compiled.destroy();
    for (const buffer of buffers) {
      buffer.destroy();
    }
    const references = method === '2sfca' ? fixture.accessibility : fixture.threeStageAccessibility;
    const expected = references[name as keyof typeof references];
    actual.forEach((score, demand) => {
      expect(
        Math.abs(score - expected[demand]),
        `${method} ${name} demand ${demand}: ${score} vs ${expected[demand]}`
      ).toBeLessThanOrEqual(fixture.relativeTolerance * Math.abs(expected[demand]));
    });
  }
});
