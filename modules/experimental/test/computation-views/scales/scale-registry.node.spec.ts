// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {
  ScaleRegistry,
  createDivergingScaleSpec,
  createQuantileScaleSpec,
  directionFor,
  getQuantileBreaks,
  rgbToHex,
  sampleRamp,
  type ScaleSpec
} from '../../../src/computation-views/scales';

function makeSpec(): ScaleSpec {
  return createQuantileScaleSpec({
    id: 'scan.output',
    variable: 'output',
    variableKind: {level: 'ratio'},
    finalStateValues: [0, 10, 20, 30, 40],
    classCount: 4,
    ground: 'dark',
    ramp: 'magma',
    noDataColor: '#888888',
    unit: 'items'
  });
}

describe('ScaleRegistry', () => {
  test('identical registration is idempotent and shares one frozen object', () => {
    const registry = new ScaleRegistry({runId: 'run'});
    const spec = makeSpec();
    const reference = registry.register(spec);
    expect(registry.register({...structuredClone(spec), variable: spec.variable})).toBe(reference);
    expect(registry.getByVariable('output')).toBe(reference);
    expect(registry.getByVariable('missing')).toBeUndefined();
    expect(registry.get(reference)).toBe(registry.list()[0]);
    expect(Object.isFrozen(reference)).toBe(true);
    const scale = registry.get(reference);
    expect(Object.isFrozen(scale)).toBe(true);
    expect(Object.isFrozen(scale.domain)).toBe(true);
    expect(Object.isFrozen(scale.classification)).toBe(true);
    expect(Object.isFrozen(scale.variableKind)).toBe(true);
    if (scale.classification.kind === 'classed')
      expect(Object.isFrozen(scale.classification.breaks)).toBe(true);
    spec.domain = [-1, 100];
    spec.variableKind.level = 'nominal';
    expect(scale.domain).toEqual([0, 40]);
    expect(scale.variableKind.level).toBe('ratio');
    registry.list().pop();
    expect(registry.list()).toHaveLength(1);
  });

  test('rejects conflicts, duplicate ids, and foreign or missing references', () => {
    const registry = new ScaleRegistry({runId: 'run'});
    const spec = makeSpec();
    registry.register(spec);
    expect(() => registry.register({...spec, noDataColor: '#ffffff'})).toThrow();
    expect(() => registry.register({...spec, id: 'another'})).toThrow();
    expect(() => registry.register({...spec, variable: 'another'})).toThrow();
    expect(() => registry.get({runId: 'other-run', scaleId: spec.id})).toThrow();
    expect(() => registry.get({runId: 'run', scaleId: 'missing'})).toThrow();
  });

  test('freeze permits only identical existing registrations', () => {
    const registry = new ScaleRegistry({runId: 'run'});
    const spec = makeSpec();
    const reference = registry.register(spec);
    registry.freeze();
    expect(registry.register(spec)).toBe(reference);
    expect(() => registry.register({...spec, id: 'new', variable: 'new'})).toThrow();
    expect(() => registry.register({...spec, unit: 'changed'})).toThrow();
  });

  test('missing values use no-data and overshoots clamp colour while retaining class flags', () => {
    const registry = new ScaleRegistry({runId: 'run'});
    const scale = registry.get(registry.register(makeSpec()));
    for (const value of [NaN, null, undefined]) expect(scale.colorFor(value)).toBe('#888888');
    expect(scale.classFor(-1)).toBe(-1);
    expect(scale.classFor(10)).toBe(1);
    expect(scale.classFor(40)).toBe(3);
    expect(scale.classFor(41)).toBe(4);
    expect(scale.colorFor(-1)).toBe(scale.colorFor(0));
    expect(scale.colorFor(41)).toBe(scale.colorFor(40));
    expect(scale.legend().unit).toBe('items');
    expect(scale.legend().classes.filter(entry => entry.beyondRange)).toHaveLength(2);
    expect(scale.legend().classes.find(entry => entry.index === 1)?.label).toBe('10–20');
    expect(Object.isFrozen(scale.legend().classes)).toBe(true);
  });

  test('quantiles ignore non-finite input and use the copied break calculation', () => {
    const values = new Float64Array([0, 10, NaN, 20, 30, 40, Infinity]);
    const spec = createQuantileScaleSpec({...makeSpec(), finalStateValues: values, classCount: 4});
    expect(spec.domain).toEqual([0, 40]);
    expect(spec.classification).toEqual({
      kind: 'classed',
      method: 'quantile',
      source: 'computed',
      breaks: getQuantileBreaks(values, 4)
    });
    expect(spec.domainSource).toBe('final-state');
    expect(() => createQuantileScaleSpec({...makeSpec(), finalStateValues: [NaN]})).toThrow();
    const constant = createQuantileScaleSpec({...makeSpec(), finalStateValues: [7, 7]});
    const registry = new ScaleRegistry({runId: 'constant'});
    expect(registry.get(registry.register(constant)).colorFor(7)).toMatch(/^#[a-f0-9]{6}$/);
  });

  test.each(['light', 'dark'] as const)('direction follows %s ground', ground => {
    const spec = createQuantileScaleSpec({...makeSpec(), ground, finalStateValues: [0, 1]});
    expect(spec.reverse).toBe(directionFor(ground, 'magma').reverse);
    const registry = new ScaleRegistry({runId: ground});
    const scale = registry.get(registry.register(spec));
    expect(scale.colorFor(0)).toBe(rgbToHex(sampleRamp('magma', 0, spec.reverse)));
    expect(scale.colorFor(1)).toBe(rgbToHex(sampleRamp('magma', 1, spec.reverse)));
  });

  test('diverging helper retains midpoint and a neutral midpoint class', () => {
    const spec = createDivergingScaleSpec({
      ...makeSpec(),
      ramp: 'rdbu',
      finalStateValues: [-10, -2, 0, 1, 50],
      midpoint: 0
    });
    const registry = new ScaleRegistry({runId: 'diverging'});
    const scale = registry.get(registry.register(spec));
    expect(scale.kind).toBe('diverging');
    expect(scale.colorFor(0)).toBe(rgbToHex(sampleRamp('rdbu', 0.5)));
    expect(scale.legend().midpointText).toBe('Midpoint: 0 items');
    expect(spec.reverse).toBe(directionFor(spec.ground, 'rdbu').reverse);
    const continuous = registry.get(
      registry.register({
        ...spec,
        id: 'continuous',
        variable: 'continuous',
        classification: {kind: 'unclassed', purpose: 'continuous-inspection'}
      })
    );
    expect(continuous.colorFor(0)).toBe(rgbToHex(sampleRamp('rdbu', 0.5)));
    expect(continuous.colorFor(-10)).toBe(rgbToHex(sampleRamp('rdbu', 0)));
    expect(continuous.colorFor(50)).toBe(rgbToHex(sampleRamp('rdbu', 1)));
  });

  test('qualitative class colours come from the supplied palette', () => {
    const registry = new ScaleRegistry({runId: 'nominal'});
    const spec = makeSpec();
    const scale = registry.get(
      registry.register({
        ...spec,
        kind: 'qualitative',
        variableKind: {level: 'nominal'},
        ramp: undefined,
        palette: ['#123456', '#abcdef']
      })
    );
    expect(scale.colorFor(0)).toBe('#123456');
    expect(scale.colorFor(10)).toBe('#abcdef');
  });
});
