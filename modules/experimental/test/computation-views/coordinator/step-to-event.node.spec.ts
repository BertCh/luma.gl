// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {
  stepToEvent,
  type DatumStamps
} from '../../../src/computation-views/coordinator/step-to-event';

const stamps: DatumStamps = {
  name: 'Settled',
  values: [3, 1, 5, 0xffffffff],
  neverValue: 0xffffffff,
  provenance: 'readback'
};

describe('stepToEvent', () => {
  test('queries latest, earliest and count thresholds from actual stamps', () => {
    expect(stepToEvent(stamps, {kind: 'first-settled', keys: [0, 1, 2]})).toMatchObject({
      status: 'ok',
      iteration: 5
    });
    expect(stepToEvent(stamps, {kind: 'last-change', keys: [0, 1]})).toMatchObject({
      status: 'ok',
      iteration: 3
    });
    expect(stepToEvent(stamps, {kind: 'first-any-settled', keys: [0, 1, 3]})).toMatchObject({
      status: 'ok',
      iteration: 1
    });
    expect(
      stepToEvent(stamps, {kind: 'iteration-where', predicate: 'count-settled-at-least', count: 2})
    ).toMatchObject({status: 'ok', iteration: 3});
    expect(
      stepToEvent(stamps, {kind: 'iteration-where', predicate: 'count-settled-at-least', count: 0})
    ).toMatchObject({status: 'ok', iteration: 0});
  });
  test('distinguishes never from absent or invalid evidence', () => {
    expect(stepToEvent(stamps, {kind: 'first-settled', keys: [0, 3]}).status).toBe('never');
    expect(stepToEvent(stamps, {kind: 'first-any-settled', keys: [3]}).status).toBe('never');
    expect(
      stepToEvent(stamps, {kind: 'iteration-where', predicate: 'count-settled-at-least', count: 4})
        .status
    ).toBe('never');
    for (const keys of [[], [99], [-1], [0.5]])
      expect(stepToEvent(stamps, {kind: 'last-change', keys}).status).toBe('unknown');
    expect(stepToEvent(undefined, {kind: 'last-change', keys: [0]}).status).toBe('unknown');
    expect(stepToEvent({...stamps, values: []}, {kind: 'last-change', keys: [0]}).status).toBe(
      'unknown'
    );
    expect(
      stepToEvent({...stamps, provenance: 'unknown'}, {kind: 'last-change', keys: [0]}).status
    ).toBe('unknown');
    expect(stepToEvent({...stamps, values: [NaN]}, {kind: 'last-change', keys: [0]}).status).toBe(
      'unknown'
    );
  });
});
