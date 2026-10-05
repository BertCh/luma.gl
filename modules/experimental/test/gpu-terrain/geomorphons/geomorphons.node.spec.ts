// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPU_GEOMORPHON_FORMS,
  GPU_GEOMORPHONS_PARAMETER_LENGTH,
  GPUGeomorphons,
  getGPUGeomorphonsParameterValues,
  type GPUGeomorphonsProps
} from '../../../src/gpu-terrain/geomorphons';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {getGeomorphonForm, getRotatedTernaryCode} from './geomorphons-oracle';

function createBand(graph: GPUCommandGraph, id: string, length: number) {
  return {
    id,
    format: 'float32' as const,
    storage: {kind: 'buffer' as const, values: createTransientView(graph, id, 'float32', length)}
  };
}

it('GPUGeomorphons schedules elevation canonicalization and one classify kernel', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const create = (overrides: Partial<GPUGeomorphonsProps> = {}) => {
    instance++;
    return new GPUGeomorphons({
      width: 12,
      height: 10,
      searchRadius: 4,
      elevation: createBand(graph, `elevation-${instance}`, 120),
      settings: createTransientView(graph, `settings-${instance}`, 'float32', 8),
      forms: createTransientView(graph, `forms-${instance}`, 'uint32', 120),
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
      width: 12,
      height: 10,
      searchRadius: 4,
      elevation: createBand(graph, `elevation-${instance}`, 120),
      settings: createTransientView(graph, `settings-${instance}`, 'float32', 8),
      forms: createTransientView(graph, `forms-${instance}`, 'uint32', 120),
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
  expect(() => create({forms: undefined})).toThrow(/at least one output/);
  expect(() => create({forms: createTransientView(graph, 'short-forms', 'uint32', 119)})).toThrow(
    /one value per pixel/
  );
  expect(() => create({settings: createTransientView(graph, 'settings7', 'float32', 7)})).toThrow(
    /settings/
  );
  expect(() => create({cellSizeMode: 'polar' as never})).toThrow(/cellSizeMode/);
  expect(() => create({rowDirection: 'east' as never})).toThrow(/rowDirection/);
  expect(() => create({width: 0})).toThrow(/dimensions/);
  const shared = createTransientView(graph, 'shared', 'uint32', 120);
  expect(() => create({forms: shared, ternary: shared})).toThrow(/share buffers/);
  const otherGraph = new GPUCommandGraph(device);
  expect(() =>
    new GPUGeomorphons({
      width: 12,
      height: 10,
      searchRadius: 4,
      elevation: createBand(otherGraph, 'foreign', 120),
      settings: createTransientView(otherGraph, 'foreign-settings', 'float32', 8),
      forms: createTransientView(otherGraph, 'foreign-forms', 'uint32', 120)
    }).getCommandNodes(graph)
  ).toThrow(/belong to the target graph/);
  device.destroy();
});

it('getGPUGeomorphonsParameterValues packs the 8-float layout', () => {
  expect(GPU_GEOMORPHONS_PARAMETER_LENGTH).toBe(8);
  expect(Array.from(getGPUGeomorphonsParameterValues({cellSize: [2, 3]}))).toEqual([
    2, 3, 1, 0, 0, 1, 0, 0
  ]);
  expect(
    Array.from(
      getGPUGeomorphonsParameterValues({
        cellSize: [2, 3],
        zFactor: 4,
        northEdge: 0.25,
        southEdge: 0.5,
        flatThresholdDegrees: 2.5,
        flatDistance: 30
      })
    )
  ).toEqual([2, 3, 4, 0.25, 0.5, 2.5, 30, 0]);
  const target = new Float32Array(10).fill(9);
  getGPUGeomorphonsParameterValues({cellSize: [1, 1]}, target);
  expect(Array.from(target.slice(0, 8))).toEqual([1, 1, 1, 0, 0, 1, 0, 0]);
  expect(target[8]).toBe(9);
  expect(() => getGPUGeomorphonsParameterValues({cellSize: [1, 1]}, new Float32Array(7))).toThrow(
    /8 values/
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
