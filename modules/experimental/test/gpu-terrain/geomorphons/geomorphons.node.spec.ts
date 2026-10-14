// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {getRasterGridGeoreference} from '../../../src/analysis-core/raster-grid';
import {makeGPURasterMetadata} from '../../../src/gpu-raster';
import {
  GPU_GEOMORPHON_FORMS,
  GPU_GEOMORPHONS_PARAMETER_LENGTH,
  GPUGeomorphons,
  getGPUGeomorphonsParameterValues,
  type GPUGeomorphonsProps
} from '../../../src/gpu-terrain/geomorphons';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {computeGeomorphons, getGeomorphonForm, getRotatedTernaryCode} from './geomorphons-oracle';
import {createBand} from '../terrain-test-utils';

const GRID = makeGPURasterMetadata({width: 12, height: 10, cellSize: [2, 3], origin: [0, 5]});

it('GPUGeomorphons schedules elevation canonicalization and one classify kernel', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const create = (overrides: Partial<GPUGeomorphonsProps> = {}) => {
    instance++;
    return new GPUGeomorphons({
      grid: GRID,
      searchRadius: 4,
      elevation: createBand(graph, `elevation-${instance}`, 120),
      parameters: createTransientView(graph, `settings-${instance}`, 'float32', 8),
      output: {forms: createTransientView(graph, `forms-${instance}`, 'uint32', 120)},
      ...overrides
    });
  };
  const contributor = create();
  expect(contributor.requiredHalo).toBe(4);
  const ids = contributor.getCommandNodes(graph).map(node => node.id);
  expect(ids[ids.length - 1]).toBe('geomorphons-classify');
  expect(ids.length).toBe(2);
  expect(
    create({id: 'landforms', skipRadius: 1, searchRadius: 9})
      .getCommandNodes(graph)
      .every(node => node.id.startsWith('landforms-'))
  ).toBe(true);
  expect(GPU_GEOMORPHON_FORMS).toEqual({
    flat: 1,
    peak: 2,
    ridge: 3,
    shoulder: 4,
    spur: 5,
    slope: 6,
    hollow: 7,
    footslope: 8,
    valley: 9,
    pit: 10
  });
  device.destroy();
});

it('GPUGeomorphons validates radii, outputs, and buffers', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const create = (overrides: Partial<GPUGeomorphonsProps> = {}) => {
    instance++;
    return new GPUGeomorphons({
      grid: GRID,
      searchRadius: 4,
      elevation: createBand(graph, `elevation-${instance}`, 120),
      parameters: createTransientView(graph, `settings-${instance}`, 'float32', 8),
      output: {forms: createTransientView(graph, `forms-${instance}`, 'uint32', 120)},
      ...overrides
    });
  };
  expect(() => create({searchRadius: 0})).toThrow(/searchRadius/);
  expect(() => create({searchRadius: 2.5})).toThrow(/searchRadius/);
  expect(() => create({skipRadius: -1})).toThrow(/skipRadius/);
  expect(() => create({skipRadius: 1.5})).toThrow(/skipRadius/);
  expect(() => create({searchRadius: 3, skipRadius: 2})).toThrow(/skipRadius \+ 2/);
  expect(() => create({searchRadius: 4, skipRadius: 2})).not.toThrow();
  expect(() => create({comparison: 'angle' as never})).toThrow(/comparison/);
  expect(() => create({output: {}})).toThrow(/at least one output/);
  expect(() =>
    create({output: {forms: createTransientView(graph, 'short-forms', 'uint32', 119)}})
  ).toThrow(/one value per pixel/);
  expect(() => create({parameters: createTransientView(graph, 'settings7', 'float32', 7)})).toThrow(
    /parameters/
  );
  expect(() => create({cellSizeMode: 'polar' as never})).toThrow(/cellSizeMode/);
  expect(() => create({rowDirection: 'east' as never})).toThrow(/rowDirection/);
  expect(() => create({grid: {...GRID, width: 0}})).toThrow(/dimensions/);
  const shared = createTransientView(graph, 'shared', 'uint32', 120);
  expect(() => create({output: {forms: shared, ternary: shared}})).toThrow(/share buffers/);
  const otherGraph = new GPUCommandGraph(device);
  expect(() =>
    new GPUGeomorphons({
      grid: GRID,
      searchRadius: 4,
      elevation: createBand(otherGraph, 'foreign', 120),
      parameters: createTransientView(otherGraph, 'foreign-settings', 'float32', 8),
      output: {forms: createTransientView(otherGraph, 'foreign-forms', 'uint32', 120)}
    }).getCommandNodes(graph)
  ).toThrow(/belong to the target graph/);
  device.destroy();
});

it('getGPUGeomorphonsParameterValues packs the 8-float layout with the grid georeference', () => {
  expect(GPU_GEOMORPHONS_PARAMETER_LENGTH).toBe(8);
  // Rows of 3 units advance south from y = 5 over 10 rows: edges 5 and -25.
  expect(Array.from(getGPUGeomorphonsParameterValues({}, GRID))).toEqual([
    2, 3, 1, 5, -25, 1, 0, 0
  ]);
  const georeference = getRasterGridGeoreference(GRID);
  expect(Array.from(getGPUGeomorphonsParameterValues({}, GRID).subarray(0, 5))).toEqual([
    ...georeference.cellSize,
    1,
    georeference.northEdge,
    georeference.southEdge
  ]);
  expect(
    Array.from(
      getGPUGeomorphonsParameterValues(
        {zFactor: 4, flatThresholdDegrees: 2.5, flatDistance: 30},
        GRID
      )
    )
  ).toEqual([2, 3, 4, 5, -25, 2.5, 30, 0]);
  const target = new Float32Array(10).fill(9);
  const unit = makeGPURasterMetadata({width: 4, height: 4});
  getGPUGeomorphonsParameterValues({}, unit, target);
  expect(Array.from(target.slice(0, 8))).toEqual([1, 1, 1, 0, -4, 1, 0, 0]);
  expect(target[8]).toBe(9);
  expect(() => getGPUGeomorphonsParameterValues({}, unit, new Float32Array(7))).toThrow(
    /getGPUGeomorphonsParameterValues/
  );
});

it('the oracle ternary rotation has 498 classes and the form table is consistent', () => {
  const classes = new Set<number>();
  for (let code = 0; code < 6561; code++) {
    classes.add(getRotatedTernaryCode(code));
  }
  expect(classes.size).toBe(498);
  expect(getGeomorphonForm(8, 0)).toBe(GPU_GEOMORPHON_FORMS.peak);
  expect(getGeomorphonForm(0, 8)).toBe(GPU_GEOMORPHON_FORMS.pit);
  expect(getGeomorphonForm(3, 3)).toBe(GPU_GEOMORPHON_FORMS.slope);
  expect(getGeomorphonForm(6, 0)).toBe(GPU_GEOMORPHON_FORMS.ridge);
  expect(getGeomorphonForm(0, 6)).toBe(GPU_GEOMORPHON_FORMS.valley);
  expect(getGeomorphonForm(8, 8)).toBe(0);
});

it('review C-13: flatDistance inside the skip distance or beyond the search distance is ignored (GRASS)', () => {
  const width = 24;
  const height = 20;
  let state = 7;
  const elevation = Array.from({length: width * height}, (_, index) => {
    state = (state * 1103515245 + 12345) % 2147483648;
    return (index % width) * 0.3 + (state / 2147483648) * 4;
  });
  const run = (flatDistance: number, skipRadius: number) =>
    computeGeomorphons(
      elevation,
      undefined,
      width,
      height,
      getGPUGeomorphonsParameterValues(
        {flatDistance, flatThresholdDegrees: 3},
        makeGPURasterMetadata({width, height, cellSize: 10})
      ),
      {searchRadius: 6, skipRadius}
    ).forms;
  // Skip distance 2 * 10 m = 20 m, search distance 60 m.
  expect(run(15, 2)).toEqual(run(0, 2));
  expect(run(20, 2)).toEqual(run(0, 2));
  expect(run(60, 0)).toEqual(run(0, 0));
  // Inside the window the rule applies and changes some forms on this terrain.
  expect(run(25, 2)).not.toEqual(run(0, 2));
});
