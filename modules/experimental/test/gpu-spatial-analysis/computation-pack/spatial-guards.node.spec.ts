// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {
  SPATIAL_GUARDS,
  SPATIAL_GUARD_MISCONCEPTIONS,
  getSpatialGuardMetadata,
  getSpatialGuards,
  type SpatialGuardId
} from '../../../src/gpu-spatial-analysis/computation-pack/guards';
import {SPATIAL_GUARD_CHOICE_NAMES} from '../../../src/gpu-spatial-analysis/computation-pack/choices';
import {makeScanRecord} from '../../computation-views/fixtures/scan-record';

type Metadata = Record<string, string | number | boolean>;
const CASES: [SpatialGuardId, Metadata, Metadata | undefined][] = [
  ['coordinate-model', {coordinateModel: 'projected'}, {coordinateModel: 'geographic'}],
  ['count-vs-rate', {mappedVariableExtent: 'intensive'}, {mappedVariableExtent: 'extensive'}],
  ['conservation', {sumBefore: 12, sumAfter: 12}, {sumBefore: 12, sumAfter: 14}],
  ['uncorrected-p', {pCorrection: 'BH'}, {pCorrection: 'none'}],
  ['classification-needs-inference', {significanceAvailable: true}, {significanceAvailable: false}],
  ['row-standardisation', {weightsStandardisation: 'row'}, {weightsStandardisation: 'binary'}],
  ['pseudo-p-grid', {permutations: 99}, undefined],
  ['islands-disclosed', {islandCount: 2}, undefined],
  ['eb-reference-rate', {ebReferenceRate: 'pooled'}, {ebReferenceRate: 'mean-of-rates'}]
];

describe('spatial guards', () => {
  for (const [identifier, passing, failing] of CASES) {
    test(`${identifier}: pass, unknown and defined failure; record fallback and explicit override`, () => {
      const guard = SPATIAL_GUARDS[identifier];
      expect(guard.evaluate({metadata: passing}).status).toBe('pass');
      expect(guard.evaluate({metadata: {}}).status).toBe('unknown');
      if (failing) expect(guard.evaluate({metadata: failing}).status).toBe('fail');
      const record = makeScanRecord();
      record.choices = Object.entries(passing).map(([name, value]) => ({
        name,
        value,
        stage: 'inference',
        sweepable: false
      }));
      expect(guard.evaluate({record, metadata: {}}).status).toBe('pass');
      expect(
        guard.evaluate({
          record,
          metadata: Object.fromEntries(Object.keys(passing).map(name => [name, undefined]))
        }).status
      ).toBe('unknown');
      if (failing) expect(guard.evaluate({record, metadata: failing}).status).toBe('fail');
      expect(SPATIAL_GUARD_MISCONCEPTIONS[identifier]).toMatch(/^M\d+$/);
      expect(getSpatialGuards([identifier])).toEqual([guard]);
    });
  }

  test('grid and islands use unknown for invalid data; they do not invent a failing premise', () => {
    for (const permutations of [-1, 0, 2.5, NaN, Infinity, '99'])
      expect(SPATIAL_GUARDS['pseudo-p-grid'].evaluate({metadata: {permutations}}).status).toBe(
        'unknown'
      );
    for (const islandCount of [-1, NaN, Infinity, '2'])
      expect(SPATIAL_GUARDS['islands-disclosed'].evaluate({metadata: {islandCount}}).status).toBe(
        'unknown'
      );
    expect(
      SPATIAL_GUARDS['pseudo-p-grid'].evaluate({metadata: {permutations: 99}}).reason
    ).toContain('k/100; floor 1/100 = 0.01');
    expect(SPATIAL_GUARDS['islands-disclosed'].evaluate({metadata: {islandCount: 0}}).reason).toBe(
      '0 islands, hatched'
    );
    expect(
      SPATIAL_GUARDS['eb-reference-rate'].evaluate({metadata: {ebReferenceRate: 'pooled'}}).reason
    ).toContain('μ = ΣO/ΣP');
  });

  test('extracts only scalar choices, preserves last choice and documents guard names', () => {
    const record = makeScanRecord();
    record.choices = [
      {name: 'permutations', value: 9, stage: 'inference', sweepable: false},
      {name: 'permutations', value: 99, stage: 'inference', sweepable: false},
      // @ts-expect-error Malformed imported choices must not become guard metadata.
      {name: 'array', value: [1, 2], stage: 'inference', sweepable: false}
    ];
    expect(getSpatialGuardMetadata(record)).toEqual({permutations: 99});
    expect(getSpatialGuardMetadata()).toEqual({});
    for (const [, passing] of CASES)
      for (const name of Object.keys(passing)) expect(SPATIAL_GUARD_CHOICE_NAMES).toContain(name);
  });
});

describe('spatial pack guardsFor', () => {
  test('returns each element guards with its fixed metadata as fallback', async () => {
    const {createSpatialComputationPack, getSpatialOperatorEntries} = await import(
      '../../../src/gpu-spatial-analysis/computation-pack/index'
    );
    const pack = createSpatialComputationPack();
    const record = makeScanRecord();
    expect(pack.guardsFor?.({record})).toEqual([]);
    for (const [element, entry] of Object.entries(getSpatialOperatorEntries())) {
      const guards = pack.guardsFor?.({record, element}) ?? [];
      expect(guards.map(guard => guard.id)).toEqual([...entry.guards]);
      if (entry.guards.includes('coordinate-model') && entry.metadata?.['metric'] === 'geodesic') {
        const coordinateModel = guards.find(guard => guard.id === 'coordinate-model')!;
        // Geodesic operators are metric on lon/lat (M8 exception).
        expect(
          coordinateModel.evaluate({record, metadata: {coordinateModel: 'geographic'}}).status
        ).toBe('pass');
      }
    }
  });
});
