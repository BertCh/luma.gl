// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUEmpiricalBayesRates} from '../../../src/gpu-spatial-analysis/rate-smoothing';
import {computeEmpiricalBayesOracle} from './empirical-bayes-oracle';
import {EB_LOW_DISPERSION} from './eb-low-dispersion-fixture';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('GPUEmpiricalBayesRates validates its props', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'eb-validation'});
  const events = createTransientView(graph, 'events', 'float32', 10);
  const populations = createTransientView(graph, 'populations', 'float32', 10);
  expect(() => new GPUEmpiricalBayesRates({events, populations})).toThrow(/no output/);
  expect(
    () =>
      new GPUEmpiricalBayesRates({
        events,
        populations: createTransientView(graph, 'short', 'float32', 9),
        rawRates: createTransientView(graph, 'raw', 'float32', 10)
      })
  ).toThrow(/populations length/);
  const nodes = new GPUEmpiricalBayesRates({
    events,
    populations,
    standardizedRates: createTransientView(graph, 'z', 'float32', 10)
  }).getCommandNodes(graph);
  // partials-a, totals-a, partials-b, summary, rates
  expect(nodes.length).toBe(5);
});

it('the empirical-Bayes oracle reproduces esda unclamped and the numpy Marshall clamp', () => {
  const input = {events: EB_LOW_DISPERSION.events, populations: EB_LOW_DISPERSION.populations};
  const esda = computeEmpiricalBayesOracle(input);
  const clamped = computeEmpiricalBayesOracle({...input, clampPriorVariance: true});
  expect(esda.priorVariance).toBeCloseTo(EB_LOW_DISPERSION.priorVariance, 12);
  for (const [row, expected] of EB_LOW_DISPERSION.unclamped.entries()) {
    expect(esda.smoothed[row]).toBeCloseTo(expected, 12);
  }
  for (const [row, expected] of EB_LOW_DISPERSION.clamped.entries()) {
    expect(clamped.smoothed[row]).toBeCloseTo(expected, 12);
  }
  expect(esda.smoothed[0]).toBeLessThan(0);
});

it('GPUEmpiricalBayesRates builds the same nodes with and without the clamp', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'eb-clamp'});
  const events = createTransientView(graph, 'events', 'float32', 10);
  const populations = createTransientView(graph, 'populations', 'float32', 10);
  for (const clampPriorVariance of [true, false, undefined]) {
    const nodes = new GPUEmpiricalBayesRates({
      id: `eb-${clampPriorVariance}`,
      events,
      populations,
      clampPriorVariance,
      smoothedRates: createTransientView(graph, `s-${clampPriorVariance}`, 'float32', 10)
    }).getCommandNodes(graph);
    expect(nodes.length).toBe(5);
  }
});
