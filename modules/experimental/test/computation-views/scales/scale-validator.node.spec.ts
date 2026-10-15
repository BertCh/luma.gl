// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test, vi} from 'vitest';
import {
  ScaleRegistry,
  SCALE_RULES,
  createQuantileScaleSpec,
  directionFor,
  isAchromatic,
  validateScaleSpec,
  validateScaleSpecs,
  validateViewEncodings,
  type ScaleRuleId,
  type ScaleSpec,
  type ViewEncodingSpec
} from '../../../src/computation-views/scales';

function makeSpec(overrides: Partial<ScaleSpec> = {}): ScaleSpec {
  return {
    ...createQuantileScaleSpec({
      id: 'output',
      variable: 'output',
      variableKind: {level: 'ratio'},
      finalStateValues: [0, 10, 20, 30, 40],
      ground: 'dark',
      ramp: 'magma',
      noDataColor: '#888888'
    }),
    ...overrides
  };
}

function makeEncoding(overrides: Partial<ViewEncodingSpec> = {}): ViewEncodingSpec {
  return {
    viewId: 'view',
    color: {runId: 'run', scaleId: 'output'},
    provenanceChannel: 'texture',
    provisionalChannel: 'crispness',
    uncertaintyChannel: 'none',
    iterationChannel: 'position',
    freeAxes: ['iteration'],
    structureColor: '#777777',
    geometry: {spatial: true, supportDrawn: true},
    labelPlacement: 'above-data',
    animated: {staticCompanion: 'chart', reducedMotion: 'stepped'},
    ...overrides
  };
}

function validateEncoding(
  overrides: Partial<ViewEncodingSpec> = {},
  specOverrides: Partial<ScaleSpec> = {}
) {
  const registry = new ScaleRegistry({runId: 'run'});
  registry.register(makeSpec(specOverrides));
  return validateViewEncodings(registry, [makeEncoding(overrides)]);
}

describe('scale rules', () => {
  test('all fourteen rules are named and the reference page satisfies every rule', () => {
    expect(Object.keys(SCALE_RULES)).toHaveLength(14);
    expect(validateEncoding()).toEqual([]);
  });

  test('one-scale-per-variable: identical scale versus a second id', () => {
    const spec = makeSpec();
    expect(validateScaleSpecs([spec, {...spec}])).toEqual([]);
    expect(validateScaleSpecs([spec, {...spec, id: 'duplicate'}])).toContainEqual(
      expect.objectContaining({rule: 'one-scale-per-variable', severity: 'error'})
    );
  });

  test('iteration-domain: fixed domain versus per-frame data and unlabelled overshoots', () => {
    expect(validateEncoding({}, {domainSource: 'declared'})).toEqual([]);
    expect(validateEncoding({}, {domainSource: 'data'})).toContainEqual(
      expect.objectContaining({rule: 'iteration-domain', severity: 'error'})
    );
    expect(validateEncoding({freeAxes: ['entity']}, {domainSource: 'data'})).toEqual([]);
    expect(validateScaleSpec(makeSpec({beyondRange: undefined}))).toContainEqual(
      expect.objectContaining({rule: 'iteration-domain', severity: 'warning'})
    );
  });

  test('classed-default: quantiles and continuous inspection versus both Jenks spellings', () => {
    expect(validateScaleSpec(makeSpec())).toEqual([]);
    expect(
      validateScaleSpec(
        makeSpec({classification: {kind: 'unclassed', purpose: 'continuous-inspection'}})
      )
    ).toEqual([]);
    for (const method of ['natural', 'natural-breaks'] as const) {
      expect(
        validateScaleSpec(
          makeSpec({classification: {kind: 'classed', method, breaks: [10], source: 'computed'}})
        )
      ).toEqual(
        expect.arrayContaining([
          expect.objectContaining({rule: 'classed-default', severity: 'warning'}),
          expect.objectContaining({rule: 'classed-default', severity: 'info'})
        ])
      );
    }
    expect(
      validateScaleSpec(
        makeSpec({
          classification: {
            kind: 'classed',
            method: 'equal-interval',
            breaks: [20],
            source: 'computed'
          }
        })
      )
    ).toEqual([]);
  });

  test('hue-only-nominal: correct levels and midpoint versus mismatches', () => {
    expect(
      validateScaleSpec(
        makeSpec({
          kind: 'qualitative',
          ramp: undefined,
          palette: ['#aa0000'],
          variableKind: {level: 'nominal'}
        })
      )
    ).toEqual([]);
    expect(validateScaleSpec(makeSpec({kind: 'diverging', ramp: 'rdbu', midpoint: 0}))).toEqual([]);
    for (const overrides of [
      {kind: 'qualitative'},
      {variableKind: {level: 'nominal'}},
      {kind: 'diverging', ramp: 'rdbu'}
    ] satisfies Partial<ScaleSpec>[]) {
      expect(validateScaleSpec(makeSpec(overrides))).toContainEqual(
        expect.objectContaining({rule: 'hue-only-nominal', severity: 'error'})
      );
    }
  });

  test('no-rainbow-ordered: sequential/periodic match versus rainbow and periodic mismatches', () => {
    expect(validateScaleSpec(makeSpec())).toEqual([]);
    expect(
      validateScaleSpec(makeSpec({kind: 'cyclic', ramp: 'romao', variableKind: {level: 'cyclic'}}))
    ).toEqual([]);
    for (const ramp of ['spectral', 'rdylbu', 'romao'] as const) {
      expect(validateScaleSpec(makeSpec({ramp}))).toContainEqual(
        expect.objectContaining({rule: 'no-rainbow-ordered', severity: 'error'})
      );
    }
    expect(validateScaleSpec(makeSpec({variableKind: {level: 'cyclic'}}))).toContainEqual(
      expect.objectContaining({rule: 'no-rainbow-ordered', severity: 'warning'})
    );
  });

  test('ramp-direction-ground: both grounds use their required reversal', () => {
    for (const ground of ['light', 'dark'] as const) {
      const reverse = directionFor(ground, 'magma').reverse;
      expect(validateScaleSpec(makeSpec({ground, reverse}))).toEqual([]);
      expect(validateScaleSpec(makeSpec({ground, reverse: !reverse}))).toContainEqual(
        expect.objectContaining({rule: 'ramp-direction-ground', severity: 'error'})
      );
    }
  });

  const encodingCases: {
    rule: ScaleRuleId;
    valid: Partial<ViewEncodingSpec>;
    invalid: Partial<ViewEncodingSpec>;
    severity: 'error' | 'warning';
  }[] = [
    {
      rule: 'iteration-own-channel',
      valid: {iterationChannel: 'small-multiple'},
      invalid: {iterationChannel: 'color'},
      severity: 'error'
    },
    {
      rule: 'separate-variables',
      valid: {provenanceChannel: 'texture', provisionalChannel: 'none'},
      invalid: {provenanceChannel: 'saturation'},
      severity: 'error'
    },
    {
      rule: 'spatial-support-drawn',
      valid: {geometry: {spatial: true, supportDrawn: true}},
      invalid: {geometry: {spatial: true, supportDrawn: false}},
      severity: 'error'
    },
    {
      rule: 'structure-no-hue',
      valid: {structureColor: '#888888'},
      invalid: {structureColor: '#ff0000'},
      severity: 'error'
    },
    {
      rule: 'labels-above-data',
      valid: {labelPlacement: 'above-data'},
      invalid: {labelPlacement: 'below-data'},
      severity: 'warning'
    },
    {
      rule: 'animation-not-evidence',
      valid: {animated: {staticCompanion: 'key-frame-strip', reducedMotion: 'stepped'}},
      invalid: {animated: {staticCompanion: 'none', reducedMotion: 'stepped'}},
      severity: 'error'
    }
  ];
  test.each(encodingCases)('$rule: positive, negative and declared exception', ({
    rule,
    valid,
    invalid,
    severity
  }) => {
    expect(validateEncoding(valid).filter(issue => issue.rule === rule)).toEqual([]);
    expect(validateEncoding(invalid)).toContainEqual(
      expect.objectContaining({rule, severity, viewId: 'view'})
    );
    const excepted = validateEncoding({
      ...invalid,
      exceptions: [{rule, reason: 'Explicit pedagogical comparison.'}]
    });
    expect(excepted).toContainEqual(
      expect.objectContaining({
        rule,
        severity: 'info',
        exceptionReason: 'Explicit pedagogical comparison.'
      })
    );
  });

  test('separate-variables rejects each saturation channel and alpha provisional state', () => {
    for (const overrides of [
      {provenanceChannel: 'none'},
      {provenanceChannel: 'alpha'},
      {provisionalChannel: 'saturation'},
      {provisionalChannel: 'alpha'},
      {uncertaintyChannel: 'saturation'}
    ] satisfies Partial<ViewEncodingSpec>[]) {
      expect(validateEncoding(overrides)).toContainEqual(
        expect.objectContaining({rule: 'separate-variables', severity: 'error'})
      );
    }
    expect(validateEncoding({uncertaintyChannel: 'alpha'})).toContainEqual(
      expect.objectContaining({rule: 'separate-variables', severity: 'warning'})
    );
  });

  test('animation-not-evidence rejects continuous reduced motion even with a companion', () => {
    expect(
      validateEncoding({animated: {staticCompanion: 'chart', reducedMotion: 'continuous'}})
    ).toContainEqual(expect.objectContaining({rule: 'animation-not-evidence', severity: 'error'}));
  });

  test('structure-no-hue uses HSL saturation, accepting short neutral hex and rejecting coloured ink', () => {
    expect(isAchromatic('#abc')).toBe(false);
    expect(isAchromatic('#777')).toBe(true);
    expect(isAchromatic('#000000')).toBe(true);
    expect(isAchromatic('#ffffff')).toBe(true);
    expect(isAchromatic('#888788')).toBe(true);
    expect(isAchromatic('#111827')).toBe(false);
    expect(isAchromatic('invalid')).toBe(false);
  });

  test('consistent-mapping: shared reference versus a foreign run and a missing scale', () => {
    const registry = new ScaleRegistry({runId: 'run'});
    const reference = registry.register(makeSpec());
    expect(
      validateViewEncodings(registry, [
        makeEncoding(),
        makeEncoding({viewId: 'second', color: {...reference}})
      ])
    ).toEqual([]);
    for (const color of [
      {...reference, runId: 'other'},
      {...reference, scaleId: 'absent'}
    ]) {
      expect(
        validateViewEncodings(registry, [makeEncoding(), makeEncoding({viewId: 'second', color})])
      ).toContainEqual(expect.objectContaining({rule: 'consistent-mapping', severity: 'error'}));
    }
    // Also defend against an external registry implementation returning two mappings of one variable.
    const alternateRegistry = new ScaleRegistry({runId: 'run'});
    const alternate = alternateRegistry.get(
      alternateRegistry.register(makeSpec({id: 'alternate'}))
    );
    const original = registry.get(reference);
    const getter = vi
      .spyOn(registry, 'get')
      .mockImplementation(ref => (ref.scaleId === 'alternate' ? alternate : original));
    try {
      expect(
        validateViewEncodings(registry, [
          makeEncoding(),
          makeEncoding({viewId: 'second', color: {runId: 'run', scaleId: 'alternate'}})
        ])
      ).toContainEqual(expect.objectContaining({rule: 'consistent-mapping', severity: 'error'}));
    } finally {
      getter.mockRestore();
    }
  });

  test('distinct-variables-distinct-palettes: separate palettes versus shared palette and domain', () => {
    const registry = new ScaleRegistry({runId: 'run'});
    const first = registry.register(makeSpec());
    const second = registry.register(makeSpec({id: 'second', variable: 'second'}));
    const third = registry.register(makeSpec({id: 'third', variable: 'third', ramp: 'cividis'}));
    const encoding = makeEncoding({color: first});
    expect(
      validateViewEncodings(registry, [encoding, makeEncoding({viewId: 'third', color: third})])
    ).toEqual([]);
    expect(
      validateViewEncodings(registry, [encoding, makeEncoding({viewId: 'second', color: second})])
    ).toContainEqual(
      expect.objectContaining({rule: 'distinct-variables-distinct-palettes', severity: 'warning'})
    );
    expect(
      validateViewEncodings(registry, [
        encoding,
        makeEncoding({
          viewId: 'second',
          color: second,
          exceptions: [
            {
              rule: 'distinct-variables-distinct-palettes',
              reason: 'Comparison teaching the ambiguity.'
            }
          ]
        })
      ])
    ).toContainEqual(
      expect.objectContaining({
        rule: 'distinct-variables-distinct-palettes',
        severity: 'info',
        exceptionReason: 'Comparison teaching the ambiguity.'
      })
    );
  });

  test('scale exceptions downgrade errors and warnings, but empty reasons do not', () => {
    for (const reason of ['Documented exception', '']) {
      const issues = validateScaleSpec(
        makeSpec({reverse: true, exceptions: [{rule: 'ramp-direction-ground', reason}]})
      );
      expect(issues).toContainEqual(
        expect.objectContaining({
          rule: 'ramp-direction-ground',
          severity: reason ? 'info' : 'error'
        })
      );
    }
    expect(
      validateEncoding(
        {},
        {
          domainSource: 'data',
          exceptions: [{rule: 'iteration-domain', reason: 'Comparison of changing domains.'}]
        }
      )
    ).toContainEqual(
      expect.objectContaining({
        rule: 'iteration-domain',
        severity: 'info',
        exceptionReason: 'Comparison of changing domains.'
      })
    );
  });
});
