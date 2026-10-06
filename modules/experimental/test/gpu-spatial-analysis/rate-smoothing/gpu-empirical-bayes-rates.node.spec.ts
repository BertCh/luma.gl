// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUEmpiricalBayesRates} from '../../../src/gpu-spatial-analysis/rate-smoothing';
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
  expect(nodes.length).toBeGreaterThan(5);
});
