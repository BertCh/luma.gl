// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  GPUCatchmentAccessibility,
  GPUHuffTradeAreas
} from '../../../src/gpu-spatial-analysis/catchment-accessibility/index';
import {computeCatchmentOracle, computeHuffOracle} from './catchment-oracle';
import {AnalysisRig, buildKernelWeights, createSeededRandom, transposeWeights} from './rig';

function expectClose(actual: number[], expected: number[], label: string, relative = 5e-5): void {
  expect(actual.length, `${label} length`).toBe(expected.length);
  for (let index = 0; index < expected.length; index++) {
    if (Math.abs(actual[index] - expected[index]) > 1e-6 + relative * Math.abs(expected[index])) {
      throw new Error(`${label}: [${index}] ${actual[index]} != ${expected[index]}`);
    }
  }
}

function randomPoints(count: number, seed: number): number[] {
  const random = createSeededRandom(seed);
  return Array.from({length: count * 2}, () => random() * 10);
}

function randomValues(count: number, seed: number, scale: number): number[] {
  const random = createSeededRandom(seed);
  return Array.from({length: count}, () => Math.fround(1 + Math.floor(random() * scale)));
}

for (const method of ['2sfca', '3sfca'] as const) {
  it(`GPUCatchmentAccessibility ${method} matches the oracle on symmetric self-join weights`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) return;
    const count = 60;
    const points = randomPoints(count, 11);
    const csr = buildKernelWeights(points, points, 2.5, {selfJoin: false});
    const supply = randomValues(count, 5, 20);
    const demand = randomValues(count, 6, 200);
    const expected = computeCatchmentOracle(method, supply, demand, csr, csr);
    const rig = new AnalysisRig(device);
    const accessibility = rig.output('float32', count);
    const ratios = rig.output('float32', count);
    const reachable = rig.output('uint32', count);
    rig.run(
      new GPUCatchmentAccessibility({
        method,
        supply: rig.input(Float32Array.from(supply), 'float32'),
        demand: rig.input(Float32Array.from(demand), 'float32'),
        facilityWeights: rig.weights(csr),
        accessibility,
        ratios,
        reachableFacilities: reachable
      })
    );
    const actual = await rig.readFloat(accessibility);
    expect(actual.some(value => value > 0)).toBe(true);
    expectClose(actual, expected.accessibility, `${method} accessibility`);
    expectClose(await rig.readFloat(ratios), expected.ratios, `${method} ratios`);
    expect(await rig.readUint(reachable)).toEqual(expected.reachable);
    rig.destroy();
  });
}

it('GPUCatchmentAccessibility 3sfca conserves supply and handles cross weights', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const facilityCount = 12;
  const demandCount = 50;
  const facilityPoints = randomPoints(facilityCount, 21);
  const demandPoints = randomPoints(demandCount, 22);
  const facilityWeights = buildKernelWeights(facilityPoints, demandPoints, 4, {selfJoin: false});
  const demandWeights = transposeWeights(facilityWeights, demandCount);
  const supply = randomValues(facilityCount, 7, 10);
  const demand = randomValues(demandCount, 8, 100);
  const expected = computeCatchmentOracle('3sfca', supply, demand, facilityWeights, demandWeights);
  const rig = new AnalysisRig(device);
  const accessibility = rig.output('float32', demandCount);
  rig.run(
    new GPUCatchmentAccessibility({
      method: '3sfca',
      supply: rig.input(Float32Array.from(supply), 'float32'),
      demand: rig.input(Float32Array.from(demand), 'float32'),
      facilityWeights: rig.weights(facilityWeights),
      demandWeights: rig.weights(demandWeights),
      accessibility
    })
  );
  const actual = await rig.readFloat(accessibility);
  expectClose(actual, expected.accessibility, '3sfca cross accessibility');
  // Conservation: sum_i P_i A_i equals the supply of facilities with demand in range.
  const reachableSupply = supply.reduce(
    (sum, value, j) =>
      sum + (facilityWeights.offsets[j + 1] > facilityWeights.offsets[j] ? value : 0),
    0
  );
  const total = actual.reduce((sum, value, i) => sum + value * demand[i], 0);
  expect(Math.abs(total - reachableSupply) / reachableSupply).toBeLessThan(1e-3);
  rig.destroy();
});

it('GPUHuffTradeAreas matches the oracle with an exponent and keeps rows normalized', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const facilityCount = 10;
  const demandCount = 45;
  const demandPoints = randomPoints(demandCount, 31);
  const facilityPoints = randomPoints(facilityCount, 32);
  const demandWeights = buildKernelWeights(demandPoints, facilityPoints, 5, {selfJoin: false});
  const facilityWeights = transposeWeights(demandWeights, facilityCount);
  const attractiveness = randomValues(facilityCount, 9, 30);
  attractiveness[3] = 0;
  const demand = randomValues(demandCount, 10, 50);
  const alpha = 1.5;
  const expected = computeHuffOracle(attractiveness, demand, alpha, demandWeights, facilityWeights);
  const rig = new AnalysisRig(device);
  const probabilities = rig.output('float32', demandWeights.neighbors.length + 3);
  const tradeArea = rig.output('uint32', demandCount);
  const tradeAreaProbability = rig.output('float32', demandCount);
  const expectedDemand = rig.output('float32', facilityCount);
  rig.run(
    new GPUHuffTradeAreas({
      attractiveness: rig.input(Float32Array.from(attractiveness), 'float32'),
      demand: rig.input(Float32Array.from(demand), 'float32'),
      demandWeights: rig.weights(demandWeights),
      facilityWeights: rig.weights(facilityWeights),
      parameters: rig.input(Float32Array.of(alpha), 'float32'),
      probabilities,
      tradeArea,
      tradeAreaProbability,
      expectedDemand
    })
  );
  const probabilityValues = (await rig.readFloat(probabilities)).slice(
    0,
    demandWeights.neighbors.length
  );
  expect(probabilityValues.some(value => value > 0)).toBe(true);
  expectClose(probabilityValues, expected.probabilities, 'huff probabilities');
  for (let i = 0; i < demandCount; i++) {
    let sum = 0;
    for (let s = demandWeights.offsets[i]; s < demandWeights.offsets[i + 1]; s++) {
      sum += probabilityValues[s];
    }
    if (demandWeights.offsets[i + 1] > demandWeights.offsets[i]) {
      expect(Math.abs(sum - 1)).toBeLessThan(1e-4);
    }
  }
  expect(await rig.readUint(tradeArea)).toEqual(expected.tradeArea);
  expectClose(await rig.readFloat(tradeAreaProbability), expected.tradeAreaProbability, 'trade p');
  const expectedDemandValues = await rig.readFloat(expectedDemand);
  expectClose(expectedDemandValues, expected.expectedDemand, 'expected demand');
  expect(expectedDemandValues[3]).toBe(0);
  // Demand is conserved over demand locations that reach any facility.
  const served = demand.reduce(
    (sum, value, i) => sum + (expected.tradeArea[i] !== 0xffffffff ? value : 0),
    0
  );
  const captured = expectedDemandValues.reduce((sum, value) => sum + value, 0);
  expect(Math.abs(captured - served) / served).toBeLessThan(1e-3);
  rig.destroy();
});
