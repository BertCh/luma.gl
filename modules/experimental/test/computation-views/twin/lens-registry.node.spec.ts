// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  getLensScenario,
  isLensScenarioId,
  listLensScenarios,
  registerLensScenario,
  toRecordedLensFile,
  fromRecordedLensFile
} from '../../../src/computation-lenses/registry';
import type {LensScenarioDescriptor} from '../../../src/computation-lenses/types';
import {makeScanRecord} from '../fixtures/scan-record';

const descriptor: LensScenarioDescriptor = {
  id: 'registry-test.scan',
  entry: 'registry-test',
  element: 'GPUScan',
  title: 'test',
  audience: 'engineer',
  rungs: ['table'],
  defaultRung: 'table',
  load: async () => ({run: async () => makeScanRecord()})
};

test('validates scenario ids, coverage contract, duplicates and sorted lookup', () => {
  expect(isLensScenarioId('gpu-core.scan.small-data')).toBe(true);
  for (const id of ['scan', 'GPU.scan', 'gpu-.scan', 'gpu.scan.a.b', '.scan'])
    expect(isLensScenarioId(id)).toBe(false);
  expect(() => registerLensScenario({...descriptor, entry: 'wrong'})).toThrow('entry');
  expect(() => registerLensScenario({...descriptor, defaultRung: 'kernel'})).toThrow('defaultRung');
  expect(() => registerLensScenario({...descriptor, element: 'local-moran'})).toThrow('element');
  expect(() => registerLensScenario({...descriptor, id: 'bad'})).toThrow('id');
  registerLensScenario(descriptor);
  expect(getLensScenario(descriptor.id)).toBe(descriptor);
  expect(() => registerLensScenario(descriptor)).toThrow('Duplicate');
  expect(listLensScenarios().map(scenario => scenario.id)).toEqual(
    listLensScenarios()
      .map(scenario => scenario.id)
      .sort()
  );
});

test('JSON round trip preserves typed capture arrays and non-finite tokens', () => {
  const record = makeScanRecord();
  const capture = record.captures!.snapshots[0];
  record.captures = {
    ...record.captures!,
    snapshots: [
      {...capture, elementType: 'f32', values: new Float32Array([NaN, Infinity, -Infinity, 1.25])},
      {...capture, elementType: 'u32', values: new Uint32Array([0, 4294967295])},
      {...capture, elementType: 'i32', values: new Int32Array([-2147483648, 2147483647])}
    ]
  };
  const serialized = toRecordedLensFile(record, {
    scenarioId: 'gpu-core.scan',
    recordedAt: record.createdAt,
    adapterLabel: 'Apple M3'
  });
  const json = JSON.stringify(serialized);
  // Typed captures travel as base64 of their little-endian bytes (NaN and infinities included).
  expect(json).toContain('"values":"b64:AADAfwAAgH8AAID/AACgPw=="');
  expect(json).not.toContain('null');
  const restored = fromRecordedLensFile(JSON.parse(json));
  expect(restored.record.captures!.snapshots.map(snapshot => snapshot.values)).toEqual(
    record.captures.snapshots.map(snapshot => snapshot.values)
  );
  expect(restored.record.captures!.snapshots[0].values).toBeInstanceOf(Float32Array);
  expect(restored.record.captures!.snapshots[1].values).toBeInstanceOf(Uint32Array);
  expect(restored.record.captures!.snapshots[2].values).toBeInstanceOf(Int32Array);
  expect(capture.values).toBeInstanceOf(Uint32Array);
  expect(() => fromRecordedLensFile({schema: 'wrong'})).toThrow('schema');
  expect(() => fromRecordedLensFile({...serialized, recordedAt: 'bad'})).toThrow('recordedAt');
  const invalid = JSON.parse(json);
  invalid.record.captures.snapshots[0].values = [null];
  expect(() => fromRecordedLensFile(invalid)).toThrow('values');
});

test('series values and unrelated attributes survive deep conversion', () => {
  const record = makeScanRecord();
  record.graphs = record.graphs.map(graph => ({...graph, attrs: {values: 'a named attribute'}}));
  record.metrics = record.metrics!.map(metrics => ({
    ...metrics,
    series: [{name: 'residual', axis: 'iteration', values: [NaN, Infinity, 0]}]
  }));
  const serialized = toRecordedLensFile(record, {
    scenarioId: 'gpu-core.scan',
    recordedAt: record.createdAt,
    adapterLabel: 'Apple M3'
  });
  const restored = fromRecordedLensFile(JSON.parse(JSON.stringify(serialized)));
  expect(restored.record.graphs[0].attrs?.['values']).toBe('a named attribute');
  expect(restored.record.metrics![0].series![0].values).toEqual([NaN, Infinity, 0]);
});
