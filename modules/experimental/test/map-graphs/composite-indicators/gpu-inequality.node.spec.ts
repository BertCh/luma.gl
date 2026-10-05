// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUInequality,
  type GPUInequalityProps
} from '../../../src/map-graphs/composite-indicators/gpu-inequality';
import {
  getGPUInequalityParameterValues,
  GPU_INEQUALITY_PARAMETER_LENGTH
} from '../../../src/map-graphs/composite-indicators/inequality-parameters';
import {createNullWebGPUDevice} from '../map-graph-test-utils';
import {computeDiscreteGini, computeInequalityOnCPU, evaluateLorenz} from './inequality-oracle';

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUInequalityProps> = {}
): GPUInequalityProps {
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  return {
    values: view('float32', 10),
    zoneIds: view('uint32', 10),
    zoneCount: 3,
    parameters: view('float32', GPU_INEQUALITY_PARAMETER_LENGTH),
    output: {gini: view('float32', 3)},
    ...overrides
  };
}

function expectThrows(
  overrides: (graph: GPUCommandGraph) => Partial<GPUInequalityProps>,
  message: RegExp
): void {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  expect(() => new GPUInequality(createProps(graph, overrides(graph)))).toThrow(message);
  device.destroy();
}

it('getGPUInequalityParameterValues packs epsilon and the Palma cuts', () => {
  expect(Array.from(getGPUInequalityParameterValues())).toEqual([
    1,
    Math.fround(0.1),
    Math.fround(0.4),
    0
  ]);
  expect(Array.from(getGPUInequalityParameterValues({epsilon: 2.5, palmaTopShare: 0.2}))).toEqual([
    2.5,
    Math.fround(0.2),
    Math.fround(0.4),
    0
  ]);
  expect(() => getGPUInequalityParameterValues({epsilon: -1})).toThrow(/epsilon/);
  expect(() => getGPUInequalityParameterValues({epsilon: NaN})).toThrow(/epsilon/);
  expect(() => getGPUInequalityParameterValues({palmaTopShare: 0})).toThrow(/palmaTopShare/);
  expect(() => getGPUInequalityParameterValues({palmaBottomShare: 1.5})).toThrow(
    /palmaBottomShare/
  );
  expect(() => getGPUInequalityParameterValues({}, new Float32Array(2))).toThrow(/hold/);
});

it('GPUInequality validates props', () => {
  expectThrows(() => ({zoneCount: 0}), /zoneCount/);
  expectThrows(() => ({lorenzKnotCount: 1}), /lorenzKnotCount/);
  expectThrows(
    graph => ({zoneIds: createTransientView(graph, `z-${serial++}`, 'uint32', 9)}),
    /zoneIds length/
  );
  expectThrows(
    graph => ({weights: createTransientView(graph, `w-${serial++}`, 'float32', 9)}),
    /weights length/
  );
  expectThrows(
    graph => ({mask: createTransientView(graph, `m-${serial++}`, 'uint32', 9)}),
    /mask length/
  );
  expectThrows(
    graph => ({parameters: createTransientView(graph, `p-${serial++}`, 'float32', 2)}),
    /parameters/
  );
  expectThrows(
    graph => ({output: {gini: createTransientView(graph, `g-${serial++}`, 'float32', 2)}}),
    /gini/
  );
  expectThrows(
    graph => ({
      output: {
        count: createTransientView(graph, `c-${serial++}`, 'float32', 3) as never
      }
    }),
    /count/
  );
  expectThrows(
    graph => ({
      output: {lorenzKnots: createTransientView(graph, `l-${serial++}`, 'float32', 32)}
    }),
    /lorenzKnots/
  );
  expectThrows(
    graph => ({
      output: {globalSummary: createTransientView(graph, `s-${serial++}`, 'float32', 7)}
    }),
    /globalSummary/
  );
  expectThrows(() => ({output: {}}), /at least one output/);
  expectThrows(graph => {
    const values = createTransientView(graph, `v-${serial++}`, 'float32', 10);
    return {values, output: {gini: values}};
  }, /share buffers|hold at least/);
});

it('GPUInequality emits deterministic node IDs and only the passes its outputs need', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `n-${serial++}`, format, length);
  const basic = new GPUInequality(createProps(graph, {id: 'basic'}));
  const basicIds = basic.getCommandNodes(graph).map(node => node.id);
  expect(basicIds[0]).toBe('basic-keys');
  expect(basicIds).toContain('basic-zone-keys');
  expect(basicIds).toContain('basic-zone-stats');
  expect(basicIds).toContain('basic-publish-indices');
  expect(basicIds).not.toContain('basic-publish-moments');
  expect(basicIds).not.toContain('basic-global-sums');
  const full = new GPUInequality(
    createProps(graph, {
      id: 'full',
      weights: view('float32', 10),
      mask: view('uint32', 10),
      lorenzKnotCount: 5,
      output: {
        gini: view('float32', 3),
        theilT: view('float32', 3),
        theilL: view('float32', 3),
        atkinson: view('float32', 3),
        hoover: view('float32', 3),
        palma: view('float32', 3),
        mean: view('float32', 3),
        count: view('uint32', 3),
        lorenzKnots: view('float32', 15),
        globalSummary: view('float32', 8)
      }
    })
  );
  const fullIds = full.getCommandNodes(graph).map(node => node.id);
  expect(fullIds.slice(-4)).toEqual([
    'full-publish-indices',
    'full-publish-moments',
    'full-global-sums',
    'full-global-gini'
  ]);
  expect(new Set(fullIds).size).toBe(fullIds.length);
  device.destroy();
});

it('computeInequalityOnCPU matches known answers', () => {
  // Zone 0: perfect equality. Zone 1: one holder of everything. Zone 2: 1, 2, 3, 4 (textbook).
  const values = [5, 5, 5, 5, 0, 0, 0, 12, 1, 2, 3, 4];
  const zoneIds = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
  const result = computeInequalityOnCPU({values, zoneIds, zoneCount: 3});
  for (const index of [result.gini, result.theilT, result.hoover]) {
    expect(index[0]).toBeCloseTo(0, 12);
  }
  expect(result.theilL[0]).toBeCloseTo(0, 12);
  expect(result.atkinson[0]).toBeCloseTo(0, 12);
  expect(result.palma[0]).toBeCloseTo(0.1 / 0.4, 6);
  expect(result.gini[1]).toBeCloseTo(3 / 4, 12);
  expect(result.theilT[1]).toBeCloseTo(Math.log(4), 12);
  expect(result.theilL[1]).toBeNaN();
  expect(result.atkinson[1]).toBeNaN();
  expect(result.hoover[1]).toBeCloseTo(3 / 4, 12);
  // 1, 2, 3, 4: Gini = 2 * (1 + 4 + 9 + 16) / (4 * 10) - 5 / 4 = 0.25.
  expect(result.gini[2]).toBeCloseTo(0.25, 12);
  expect(result.mean[2]).toBe(2.5);
  expect(result.hoover[2]).toBeCloseTo((1.5 + 0.5 + 0.5 + 1.5) / 20, 12);
  // Lorenz knots of 1, 2, 3, 4 at p = 0.25 and 0.5 (every 0.1 interpolates inside a row).
  const knots = result.lorenzKnots.slice(22, 33);
  expect(knots[0]).toBe(0);
  expect(knots[10]).toBeCloseTo(1, 12);
  expect(knots[5]).toBeCloseTo(0.3, 12);
  expect(
    evaluateLorenz(
      [1, 2, 3, 4].map(x => ({x, w: 1})),
      0.25
    )
  ).toBeCloseTo(0.1, 12);
  // Theil T of 1, 2, 3, 4 by definition.
  const theil = [1, 2, 3, 4].reduce((sum, x) => sum + (x / 2.5) * Math.log(x / 2.5), 0) / 4;
  expect(result.theilT[2]).toBeCloseTo(theil, 12);
  expect(result.count).toEqual([4, 4, 4]);
});

it('computeInequalityOnCPU agrees with the discrete Gini and decomposes Theil T', () => {
  const values: number[] = [];
  const zoneIds: number[] = [];
  let state = 99;
  const random = () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
  for (let row = 0; row < 500; row++) {
    zoneIds.push(row % 4);
    values.push(Math.exp(random() * 3) * (1 + (row % 4)));
  }
  const result = computeInequalityOnCPU({values, zoneIds, zoneCount: 4});
  for (let zone = 0; zone < 4; zone++) {
    const zoneValues = values.filter((_, row) => zoneIds[row] === zone);
    expect(result.gini[zone]).toBeCloseTo(computeDiscreteGini(zoneValues), 10);
  }
  expect(result.globalSummary[3]).toBeCloseTo(computeDiscreteGini(values), 10);
  const [total, between, within] = result.globalSummary;
  expect(total).toBeCloseTo(between + within, 10);
  expect(between).toBeGreaterThan(0);
});

it('computeInequalityOnCPU treats weights as population and applies the inclusion rules', () => {
  // Weight 2 on a value equals listing it twice.
  const weighted = computeInequalityOnCPU({
    values: [1, 3],
    zoneIds: [0, 0],
    zoneCount: 1,
    weights: [2, 1]
  });
  const repeated = computeInequalityOnCPU({values: [1, 1, 3], zoneIds: [0, 0, 0], zoneCount: 1});
  expect(weighted.gini[0]).toBeCloseTo(repeated.gini[0], 12);
  expect(weighted.theilT[0]).toBeCloseTo(repeated.theilT[0], 12);
  expect(weighted.hoover[0]).toBeCloseTo(repeated.hoover[0], 12);
  expect(weighted.palma[0]).toBeCloseTo(repeated.palma[0], 12);
  // Negative, NaN, masked, unassigned and out-of-range rows are excluded.
  const filtered = computeInequalityOnCPU({
    values: [1, -1, NaN, 2, 3, 4, 5],
    zoneIds: [0, 0, 0, 0xffffffff, 7, 0, 0],
    mask: [1, 1, 1, 1, 1, 0, 1],
    zoneCount: 2
  });
  expect(filtered.count).toEqual([2, 0]);
  expect(filtered.mean[1]).toBeNaN();
  expect(filtered.gini[1]).toBeNaN();
  expect(filtered.globalSummary[4]).toBe(2);
  // Atkinson with a zero: defined for epsilon < 1, NaN for epsilon >= 1.
  const withZero = (epsilon: number) =>
    computeInequalityOnCPU({
      values: [0, 2, 4],
      zoneIds: [0, 0, 0],
      zoneCount: 1,
      settings: {epsilon}
    });
  expect(withZero(0.5).atkinson[0]).toBeGreaterThan(0);
  expect(withZero(1).atkinson[0]).toBeNaN();
  expect(withZero(2).atkinson[0]).toBeNaN();
  expect(withZero(0).atkinson[0]).toBeCloseTo(0, 12);
  // All-zero zone: mean 0, indices undefined.
  const zero = computeInequalityOnCPU({values: [0, 0], zoneIds: [0, 0], zoneCount: 1});
  expect(zero.mean[0]).toBe(0);
  expect(zero.gini[0]).toBeNaN();
});
