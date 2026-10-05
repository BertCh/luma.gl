// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPULocalDominanceParameterValues,
  GPULocalDominance,
  type GPULocalDominanceProps
} from '../../../src/gpu-terrain/relief-visualization/gpu-local-dominance';
import {
  getGPUMultiScaleReliefParameterValues,
  getGPUMultiScaleReliefRadii,
  GPUMultiScaleRelief,
  type GPUMultiScaleReliefProps
} from '../../../src/gpu-terrain/relief-visualization/gpu-multi-scale-relief';
import {
  getGPUReliefBlendParameterValues,
  GPU_RELIEF_BLEND_VAT_ARCHAEOLOGICAL,
  GPU_RELIEF_BLEND_VAT_FLAT,
  GPUReliefBlend,
  type GPUReliefBlendProps
} from '../../../src/gpu-terrain/relief-visualization/gpu-relief-blend';
import {
  getGPUSimpleLocalReliefParameterValues,
  GPUSimpleLocalRelief,
  type GPUSimpleLocalReliefProps
} from '../../../src/gpu-terrain/relief-visualization/gpu-simple-local-relief';
import {roundHalfEven} from '../../../src/gpu-terrain/relief-visualization/relief-visualization-utils';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {pythonRound} from './relief-visualization-oracle';

function createBand(graph: GPUCommandGraph, id: string, length: number) {
  return {
    id,
    format: 'float32' as const,
    storage: {
      kind: 'buffer' as const,
      values: createTransientView(graph, id, 'float32', length)
    }
  };
}

it('roundHalfEven matches Python rounding', () => {
  for (const value of [-3.5, -2.5, -1.5, -0.5, 0, 0.4999999, 0.5, 1.5, 2.5, 3.5, 7.0000001, 9.9]) {
    expect(Math.abs(roundHalfEven(value))).toBe(Math.abs(pythonRound(value)));
  }
});

it('GPUSimpleLocalRelief validates props and schedules two mean filter passes', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let instance = 0;
  const create = (overrides: Partial<GPUSimpleLocalReliefProps> = {}) => {
    instance++;
    return new GPUSimpleLocalRelief({
      width: 6,
      height: 5,
      radius: 3,
      elevation: createBand(graph, `elevation-${instance}`, 30),
      settings: createTransientView(graph, `settings-${instance}`, 'float32', 4),
      relief: createTransientView(graph, `relief-${instance}`, 'float32', 30),
      ...overrides
    });
  };
  const contributor = create();
  expect(contributor.id).toBe('simple-local-relief');
  expect(contributor.requiredHalo).toBe(3);
  const nodeIds = contributor.getCommandNodes(graph).map(node => node.id);
  expect(nodeIds.slice(-2)).toEqual([
    'simple-local-relief-mean-rows',
    'simple-local-relief-mean-columns'
  ]);
  expect(() => create({radius: 0})).toThrow(/radius/);
  expect(() => create({radius: 2.5})).toThrow(/radius/);
  expect(() => create({relief: undefined})).toThrow(/at least one output/);
  expect(() => create({width: 0})).toThrow(/dimensions/);
  expect(() =>
    create({settings: createTransientView(graph, 'short-settings', 'float32', 3)})
  ).toThrow(/settings/);
  expect(() => create({relief: createTransientView(graph, 'short-relief', 'float32', 29)})).toThrow(
    /relief/
  );
  const other = new GPUCommandGraph(createNullWebGPUDevice());
  expect(() => create().getCommandNodes(other)).toThrow(/graph/);
  expect(Array.from(getGPUSimpleLocalReliefParameterValues({verticalExaggeration: 3}))).toEqual([
    3, 0, 0, 0
  ]);
  expect(Array.from(getGPUSimpleLocalReliefParameterValues())).toEqual([1, 0, 0, 0]);
  expect(() => getGPUSimpleLocalReliefParameterValues({verticalExaggeration: NaN})).toThrow(
    /finite/
  );
  expect(() => getGPUSimpleLocalReliefParameterValues({}, new Float32Array(3))).toThrow(/4 values/);
});

it('GPUMultiScaleRelief validates scales and schedules one or two filters', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let instance = 0;
  const create = (overrides: Partial<GPUMultiScaleReliefProps> = {}) => {
    instance++;
    return new GPUMultiScaleRelief({
      width: 6,
      height: 5,
      resolution: 1,
      featureMinimum: 3,
      featureMaximum: 40,
      scalingFactor: 2,
      elevation: createBand(graph, `elevation-${instance}`, 30),
      settings: createTransientView(graph, `settings-${instance}`, 'float32', 4),
      relief: createTransientView(graph, `relief-${instance}`, 'float32', 30),
      ...overrides
    });
  };
  const contributor = create();
  expect(contributor.id).toBe('multi-scale-relief');
  expect(contributor.requiredHalo).toBe(25);
  expect(contributor.radii.firstRadius).toBe(1);
  const withFine = contributor.getCommandNodes(graph).map(node => node.id);
  expect(withFine.filter(id => /-(first|last)-(rows|columns)$/.test(id))).toHaveLength(4);
  // i = 0: the finest surface is the elevation itself, so one filter suffices.
  const linear = create({
    id: 'linear-relief',
    featureMinimum: 1,
    featureMaximum: 20,
    scalingFactor: 1
  });
  expect(
    linear
      .getCommandNodes(graph)
      .map(node => node.id)
      .filter(id => /-(first|last)-(rows|columns)$/.test(id))
  ).toHaveLength(2);
  // n <= i throws when the feature range spans no filter step.
  expect(() => create({featureMinimum: 21, featureMaximum: 21, scalingFactor: 1})).toThrow(
    /too narrow/
  );
  expect(() => create({featureMinimum: 1, featureMaximum: 1})).toThrow(/too narrow/);
  expect(() => create({featureMinimum: 1, featureMaximum: 0.5, scalingFactor: 1})).toThrow(
    /too narrow/
  );
  expect(() => create({resolution: 0})).toThrow(/resolution/);
  expect(() => create({scalingFactor: 0.5})).toThrow(/scalingFactor/);
  expect(() => create({featureMaximum: Infinity})).toThrow(/finite/);
  expect(() => create({relief: undefined})).toThrow(/at least one output/);
  expect(() =>
    create({settings: createTransientView(graph, 'short-settings', 'float32', 2)})
  ).toThrow(/settings/);
  expect(
    getGPUMultiScaleReliefRadii({
      resolution: 1,
      featureMinimum: 1,
      featureMaximum: 20,
      scalingFactor: 1
    }).lastIndex
  ).toBe(10);
  expect(Array.from(getGPUMultiScaleReliefParameterValues({verticalExaggeration: 2}))).toEqual([
    2, 0, 0, 0
  ]);
});

it('GPULocalDominance validates props and schedules one kernel', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let instance = 0;
  const create = (overrides: Partial<GPULocalDominanceProps> = {}) => {
    instance++;
    return new GPULocalDominance({
      width: 6,
      height: 5,
      elevation: createBand(graph, `elevation-${instance}`, 30),
      settings: createTransientView(graph, `settings-${instance}`, 'float32', 4),
      dominance: createTransientView(graph, `dominance-${instance}`, 'float32', 30),
      ...overrides
    });
  };
  const contributor = create();
  expect(contributor.id).toBe('local-dominance');
  expect(contributor.requiredHalo).toBe(20);
  expect(contributor.shifts.count).toBe(264);
  expect(contributor.getCommandNodes(graph).map(node => node.id)).toContain(
    'local-dominance-sample'
  );
  expect(create({maximumRadius: 12}).requiredHalo).toBe(12);
  expect(() => create({minimumRadius: 0})).toThrow(/minimumRadius/);
  expect(() => create({minimumRadius: 25})).toThrow(/minimumRadius/);
  expect(() => create({radiusIncrement: 1.5})).toThrow(/radiusIncrement/);
  expect(() => create({dominance: undefined})).toThrow(/at least one output/);
  expect(() =>
    create({settings: createTransientView(graph, 'short-settings', 'float32', 3)})
  ).toThrow(/settings/);
  expect(Array.from(getGPULocalDominanceParameterValues())).toEqual([Math.fround(1.7), 1, 0, 0]);
  expect(() => getGPULocalDominanceParameterValues({observerHeight: 0})).toThrow(/positive/);
  expect(() => getGPULocalDominanceParameterValues({verticalExaggeration: Infinity})).toThrow(
    /finite/
  );
});

it('GPUReliefBlend validates layer count, settings and presets', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let instance = 0;
  const createLayers = (count: number) =>
    Array.from({length: count}, () =>
      createTransientView(graph, `layer-${instance++}`, 'float32', 30)
    );
  const create = (overrides: Partial<GPUReliefBlendProps> = {}) => {
    instance++;
    return new GPUReliefBlend({
      width: 6,
      height: 5,
      layers: createLayers(4),
      settings: createTransientView(graph, `settings-${instance}`, 'float32', 20),
      blend: createTransientView(graph, `blend-${instance}`, 'float32', 30),
      ...overrides
    });
  };
  const contributor = create();
  expect(contributor.id).toBe('relief-blend');
  expect(contributor.getCommandNodes(graph).map(node => node.id)).toEqual(['relief-blend-blend']);
  // Color or validity adds one encode node even without a blend output.
  const encoded = create({
    blend: undefined,
    color: createTransientView(graph, 'color', 'uint32', 30)
  });
  expect(encoded.getCommandNodes(graph).map(node => node.id)).toEqual([
    'relief-blend-blend',
    'relief-blend-encode'
  ]);
  // Five layers plus settings and blend is 7 bindings, within the 8 limit.
  expect(
    create({
      layers: createLayers(5),
      settings: createTransientView(graph, 'settings-five', 'float32', 25)
    }).getCommandNodes(graph)
  ).toHaveLength(1);
  expect(() => create({layers: createLayers(6)})).toThrow(/1 to 5/);
  expect(() => create({layers: []})).toThrow(/1 to 5/);
  expect(() => create({blend: undefined})).toThrow(/at least one output/);
  expect(() =>
    create({settings: createTransientView(graph, 'short-settings', 'float32', 19)})
  ).toThrow(/settings/);
  expect(() =>
    create({layers: [createTransientView(graph, 'short-layer', 'float32', 29), ...createLayers(1)]})
  ).toThrow(/layer 0/);
  expect(getGPUReliefBlendParameterValues(GPU_RELIEF_BLEND_VAT_ARCHAEOLOGICAL)).toHaveLength(20);
  expect(
    Array.from(getGPUReliefBlendParameterValues(GPU_RELIEF_BLEND_VAT_FLAT).subarray(5, 10))
  ).toEqual([0, 15, 1, 5, 0.5]);
  expect(
    Array.from(
      getGPUReliefBlendParameterValues(GPU_RELIEF_BLEND_VAT_ARCHAEOLOGICAL).subarray(10, 15)
    )
  ).toEqual([68, 93, 0, 3, 0.5]);
  expect(() => getGPUReliefBlendParameterValues([{minimum: 1, maximum: 1}])).toThrow(/greater/);
  expect(() => getGPUReliefBlendParameterValues([{minimum: 0, maximum: 1, opacity: 25}])).toThrow(
    /opacity/
  );
  expect(() => getGPUReliefBlendParameterValues([])).toThrow(/1 to 5/);
  expect(() =>
    getGPUReliefBlendParameterValues(GPU_RELIEF_BLEND_VAT_ARCHAEOLOGICAL, new Float32Array(10))
  ).toThrow(/5 values per layer/);
  expect(() => getGPUReliefBlendParameterValues([{minimum: NaN, maximum: 1}])).toThrow(/finite/);
});
