// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {addSpaceTimeHotSpotsRecipe} from '../../../src/gpu-spatial-analysis/recipes/space-time-hot-spots-recipe';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('addSpaceTimeHotSpotsRecipe rejects cyclic fields and invalid slices, and forwards the options', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = <Format extends 'uint32' | 'float32' | 'sint32' | 'float32x2' | 'uint32x2'>(
    format: Format,
    length: number,
    name: string
  ) => createTransientView(graph, name, format, length);
  let serial = 0;
  const props = (overrides: Record<string, unknown>) => {
    const key = `${Object.keys(overrides).join('-')}-${serial++}`;
    return {
      id: `rejects-${key}`,
      timestamps: view('uint32x2', 4, `ts-${key}`),
      calendarParameters: view('sint32', 2, `cp-${key}`),
      slices: {step: 'month', origin: {year: 2023, month: 1}, count: 6},
      cells: {
        kind: 'lattice',
        positions: view('float32x2', 4, `pos-${key}`),
        width: 2,
        height: 2,
        bounds: [0, 0, 2, 2]
      },
      parameters: view('float32', 8, `par-${key}`),
      ...overrides
    } as never;
  };
  // Cyclic calendar fields are no longer accepted.
  expect(() =>
    addSpaceTimeHotSpotsRecipe(graph, props({slices: {field: 'month', firstValue: 1, count: 6}}))
  ).toThrow(/cyclic calendar fields/);
  expect(() =>
    addSpaceTimeHotSpotsRecipe(
      graph,
      props({slices: {step: 'weekday', origin: {year: 2023}, count: 6}})
    )
  ).toThrow(/slices.step/);
  expect(() =>
    addSpaceTimeHotSpotsRecipe(
      graph,
      props({slices: {step: 'day', interval: 0, origin: {year: 2023}, count: 6}})
    )
  ).toThrow(/slices.interval/);
  expect(() =>
    addSpaceTimeHotSpotsRecipe(
      graph,
      props({slices: {step: 'day', origin: {year: 2023, month: 13}, count: 6}})
    )
  ).toThrow(/slices.origin.month/);
  // multipleTesting reaches the result's plain inference object.
  const recipe = addSpaceTimeHotSpotsRecipe(
    graph,
    props({multipleTesting: 'benjamini-yekutieli', finalSlice: 'last-valid'})
  );
  expect(recipe.inference.multipleTesting).toBe('benjamini-yekutieli');
  expect(recipe.inference.alternative).toBe('two-sided');
  device.destroy();
});
