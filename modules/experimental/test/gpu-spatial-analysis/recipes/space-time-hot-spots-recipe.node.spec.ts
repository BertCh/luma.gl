// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUCalendarBuckets} from '../../../src/gpu-dataframe/calendar-buckets/index';
import {addSpaceTimeHotSpotsRecipe} from '../../../src/gpu-spatial-analysis/recipes/space-time-hot-spots-recipe';
import {GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH} from '../../../src/gpu-spatial-analysis/emerging-hot-spots/index';
import {GPUParameterBuffer} from '../../../src/utils/gpu-contributor-utils';
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
      settings: {temporalWindow: 1},
      parameters: new GPUParameterBuffer(device, {
        id: `par-${key}`,
        format: 'float32',
        length: GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH
      }),
      ...overrides
    } as never;
  };
  const timeZone = {
    transitions: view('uint32x2', 1, 'zone-transitions'),
    offsets: view('sint32', 2, 'zone-offsets')
  };
  const utcOffsets = view('sint32', 4, 'row-offsets');
  expect(() => addSpaceTimeHotSpotsRecipe(graph, props({timeZone, utcOffsets}))).toThrow(
    'addSpaceTimeHotSpotsRecipe pass utcOffsets or timeZone, not both'
  );
  for (const options of [{timeZone}, {utcOffsets}]) {
    const calendarRecipe = addSpaceTimeHotSpotsRecipe(graph, props(options));
    const calendar = calendarRecipe.contributors.find(
      contributor => contributor instanceof GPUCalendarBuckets
    );
    expect(calendar).toBeInstanceOf(GPUCalendarBuckets);
    if (calendar instanceof GPUCalendarBuckets) {
      expect(calendar.props.timeZone).toBe(options.timeZone);
      expect(calendar.props.utcOffsets).toBe(options.utcOffsets);
    }
  }
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
  expect(recipe.spatialContext).toEqual({
    coordinateSpace: 'planar',
    metric: 'native',
    units: 'native'
  });
  const spatialContext = {
    coordinateSpace: 'planar',
    metric: 'native',
    units: 'meters',
    frame: {id: 'area-frame', kind: 'local-lambert-equal-area'}
  } as const;
  const lattice = {
    kind: 'lattice',
    positions: view('float32x2', 4, 'context-positions'),
    width: 2,
    height: 2,
    bounds: [0, 0, 2, 2],
    spatialContext
  };
  expect(addSpaceTimeHotSpotsRecipe(graph, props({cells: lattice})).spatialContext).toBe(
    spatialContext
  );
  for (const metric of ['none', 'great-circle']) {
    expect(() =>
      addSpaceTimeHotSpotsRecipe(
        graph,
        props({
          cells: {
            ...lattice,
            spatialContext: {coordinateSpace: 'longitude-latitude', metric}
          }
        })
      )
    ).toThrow(/project positions to a planar frame first/);
  }
  const geographic = {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const;
  const weights = {
    offsets: view('uint32', 5, 'context-offsets'),
    neighbors: view('uint32', 4, 'context-neighbors'),
    weights: view('float32', 4, 'context-weights'),
    spatialContext: geographic
  };
  const cells = {kind: 'ids', cellIds: view('uint32', 4, 'context-ids'), cellCount: 4, weights};
  expect(addSpaceTimeHotSpotsRecipe(graph, props({cells})).spatialContext).toBe(geographic);
  expect(
    addSpaceTimeHotSpotsRecipe(
      graph,
      props({
        cells: {
          ...cells,
          weights: {...weights, spatialContext: undefined}
        }
      })
    ).spatialContext
  ).toBeUndefined();
  device.destroy();
});
