// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUSpatialEmpiricalBayesRates} from '../../../src/gpu-spatial-analysis/rate-smoothing/gpu-spatial-empirical-bayes-rates';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('GPUSpatialEmpiricalBayesRates validates its props', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'seb-validation'});
  const events = createTransientView(graph, 'events', 'float32', 4);
  const populations = createTransientView(graph, 'populations', 'float32', 4);
  const weights = {
    offsets: createTransientView(graph, 'offsets', 'uint32', 5),
    neighbors: createTransientView(graph, 'neighbors', 'uint32', 6),
    weights: createTransientView(graph, 'weights', 'float32', 6)
  };
  const output = createTransientView(graph, 'out', 'float32', 4);
  expect(() => new GPUSpatialEmpiricalBayesRates({events, populations, weights})).toThrow(
    /no output/
  );
  expect(
    () =>
      new GPUSpatialEmpiricalBayesRates({
        events,
        populations,
        weights: {...weights, offsets: createTransientView(graph, 'short', 'uint32', 4)},
        spatialRates: output
      })
  ).toThrow(/row count/);
  expect(
    () =>
      new GPUSpatialEmpiricalBayesRates({
        events,
        populations,
        weights,
        smoothedRates: createTransientView(graph, 'bad', 'float32', 3)
      })
  ).toThrow(/length/);
  const nodes = new GPUSpatialEmpiricalBayesRates({
    events,
    populations,
    weights,
    smoothedRates: output
  }).getCommandNodes(graph);
  expect(nodes.length).toBe(1);
});
