// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getMultipleTestingAdjustedPValues,
  GPUMultipleTestingAdjustment,
  resolveGPUMultipleTestingMethod,
  type GPUMultipleTestingAdjustmentOutput,
  type GPUMultipleTestingAdjustmentProps
} from '../../../src/gpu-spatial-analysis/multiple-testing';
import type {GPUMultipleTestingMethod} from '../../../src/analysis-core/contracts';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {MULTIPLE_TESTING_FIXTURES} from './multiple-testing-fixtures';

const ADJUSTING_METHODS = [
  'bonferroni',
  'holm',
  'benjamini-hochberg',
  'benjamini-yekutieli'
] as const;

function expectClose(actual: ArrayLike<number>, expected: ArrayLike<number>, label: string): void {
  expect(actual.length, label).toBe(expected.length);
  for (let row = 0; row < expected.length; row++) {
    if (Number.isNaN(expected[row])) {
      expect(actual[row], `${label} row ${row}`).toBeNaN();
    } else {
      expect(Math.abs(actual[row] - expected[row]), `${label} row ${row}`).toBeLessThanOrEqual(
        1e-12 + 1e-12 * Math.abs(expected[row])
      );
    }
  }
}

for (const fixture of MULTIPLE_TESTING_FIXTURES) {
  it(`getMultipleTestingAdjustedPValues matches statsmodels multipletests on ${fixture.name}`, () => {
    for (const method of ADJUSTING_METHODS) {
      expectClose(
        getMultipleTestingAdjustedPValues(fixture.pValues, method, fixture.minimumFamilySize),
        fixture.expected[method],
        `${fixture.name} ${method}`
      );
    }
  });
}

it('getMultipleTestingAdjustedPValues none copies tested p-values and gives ties identical values', () => {
  const fixture = MULTIPLE_TESTING_FIXTURES.find(entry => entry.name === 'missing')!;
  expectClose(getMultipleTestingAdjustedPValues(fixture.pValues, 'none'), fixture.pValues, 'none');
  // Values outside [0, 1] and infinities are untested.
  const adjusted = getMultipleTestingAdjustedPValues(
    [0.5, -0.1, 1.5, Infinity, 0.25],
    'bonferroni'
  );
  expectClose(adjusted, [1, Number.NaN, Number.NaN, Number.NaN, 0.5], 'out of range');
  const ties = MULTIPLE_TESTING_FIXTURES.find(entry => entry.name === 'ties')!;
  for (const method of ADJUSTING_METHODS) {
    const result = getMultipleTestingAdjustedPValues(ties.pValues, method);
    for (let a = 0; a < ties.pValues.length; a++) {
      for (let b = 0; b < ties.pValues.length; b++) {
        if (ties.pValues[a] === ties.pValues[b]) {
          expect(result[a], `${method} ${a} ${b}`).toBe(result[b]);
        }
      }
    }
  }
});

it('getMultipleTestingAdjustedPValues is monotone in p and bounded by [p, 1]', () => {
  let state = 12345;
  const random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
  const pValues = Float64Array.from({length: 500}, () => random() ** 3);
  for (const method of ADJUSTING_METHODS) {
    const adjusted = getMultipleTestingAdjustedPValues(pValues, method);
    const order = Array.from(pValues.keys()).sort((a, b) => pValues[a] - pValues[b]);
    for (let rank = 0; rank < order.length; rank++) {
      const value = adjusted[order[rank]];
      expect(value).toBeGreaterThanOrEqual(pValues[order[rank]]);
      expect(value).toBeLessThanOrEqual(1);
      if (rank > 0) {
        expect(value).toBeGreaterThanOrEqual(adjusted[order[rank - 1]]);
      }
    }
  }
});

it('resolveGPUMultipleTestingMethod and the CPU reference validate their arguments', () => {
  expect(resolveGPUMultipleTestingMethod('t', undefined)).toBe('none');
  expect(resolveGPUMultipleTestingMethod('t', 'holm')).toBe('holm');
  expect(() => resolveGPUMultipleTestingMethod('t', 'sidak' as GPUMultipleTestingMethod)).toThrow(
    /method/
  );
  expect(() => getMultipleTestingAdjustedPValues([0.1], 'holm', -1)).toThrow(/minimumFamilySize/);
  expect(() => getMultipleTestingAdjustedPValues([0.1], 'holm', 1.5)).toThrow(/minimumFamilySize/);
  expect(() => getMultipleTestingAdjustedPValues([0.1], 'holm', 2 ** 24)).toThrow(
    /minimumFamilySize/
  );
});

let serial = 0;

function createViews(graph: GPUCommandGraph) {
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  return {
    view,
    common: {pValues: view('float32', 10), output: {adjustedPValues: view('float32', 10)}}
  };
}

it('GPUMultipleTestingAdjustment validates its props and emits deterministic node IDs', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'multiple-testing-validation'});
  const {view, common} = createViews(graph);
  const create = (
    overrides: Partial<GPUMultipleTestingAdjustmentProps> = {},
    output: Partial<GPUMultipleTestingAdjustmentOutput> = {}
  ) =>
    new GPUMultipleTestingAdjustment({
      ...common,
      method: 'holm',
      ...overrides,
      output: {...common.output, ...output}
    });
  expect(() => create()).not.toThrow();
  expect(() => create({method: 'sidak' as GPUMultipleTestingMethod})).toThrow(/method/);
  expect(() => create({}, {adjustedPValues: view('float32', 9)})).toThrow(/adjustedPValues length/);
  expect(() => create({}, {adjustedPValues: common.pValues})).toThrow(/share buffers/);
  expect(() => create({}, {adjustedPValues: view('uint32', 10) as never})).toThrow();
  expect(() => create({minimumFamilySize: -1})).toThrow(/minimumFamilySize/);
  expect(() => create({minimumFamilySize: 2 ** 24})).toThrow(/minimumFamilySize/);
  expect(() => create({minimumFamilySize: 2.5})).toThrow(/minimumFamilySize/);
  expect(() => create({}, {testedCount: view('uint32', 0)})).toThrow(/testedCount/);
  expect(() => create({}, {testedCount: common.pValues as never})).toThrow();
  expect(() =>
    create({}, {testedCount: view('uint32', 1), adjustedPValues: view('float32', 10)})
  ).not.toThrow();

  const otherGraph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'other-graph'});
  expect(() => create({id: 'foreign'}).getCommandNodes(otherGraph)).toThrow(/belong to the target/);

  const ids = (method: GPUMultipleTestingMethod, withTestedCount = false) => {
    const idsGraph = new GPUCommandGraph(createNullWebGPUDevice(), {id: `ids-${serial++}`});
    return new GPUMultipleTestingAdjustment({
      id: 'mt',
      method,
      pValues: createTransientView(idsGraph, 'p', 'float32', 10),
      output: {
        adjustedPValues: createTransientView(idsGraph, 'a', 'float32', 10),
        testedCount: withTestedCount ? createTransientView(idsGraph, 'n', 'uint32', 1) : undefined
      }
    })
      .getCommandNodes(idsGraph)
      .map(node => node.id);
  };
  expect(ids('none')).toEqual(['mt-apply']);
  expect(ids('none', true)).toEqual(['mt-counters-clear', 'mt-count', 'mt-family', 'mt-apply']);
  expect(ids('bonferroni')).toEqual(['mt-counters-clear', 'mt-count', 'mt-family', 'mt-apply']);
  const holm = ids('holm');
  expect(holm).toContain('mt-block-scan');
  expect(holm).not.toContain('mt-total-scan');
  expect(holm.at(-1)).toBe('mt-scatter');
  expect(new Set(holm).size).toBe(holm.length);
  // More than 256 rows need the scan of block totals.
  const largeGraph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'large'});
  const large = new GPUMultipleTestingAdjustment({
    id: 'large',
    method: 'benjamini-yekutieli',
    pValues: createTransientView(largeGraph, 'p', 'float32', 70000),
    output: {adjustedPValues: createTransientView(largeGraph, 'a', 'float32', 70000)}
  })
    .getCommandNodes(largeGraph)
    .map(node => node.id);
  expect(large).toContain('large-total-scan');
});
