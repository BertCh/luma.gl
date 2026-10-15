// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {createGPUComputationRecord, createGPUComputationCaptures} from '@luma.gl/gpgpu';
import {toRecordedLensFile} from '../../../src/computation-lenses/registry';
import {createViewContext} from '../../../src/computation-views/registry/view-registry';
import {buildLensTwin, validateLensTwin} from '../../../src/computation-views/twin/static-twin';
import {
  buildMoranScatterModel,
  buildFdrStaircaseModel
} from '../../../src/gpu-spatial-analysis/computation-pack';

function makeDenseRecord() {
  const count = 4096;
  const record = createGPUComputationRecord({
    id: 'spatial-budget',
    adapter: {backend: 'webgpu', description: 'Fixture adapter'},
    domains: ['gpu-spatial-analysis'],
    createdAt: '2026-10-15T00:00:00Z',
    graphs: [],
    axes: {keySpaces: [{id: 'row', label: 'row', size: count}], axes: []},
    choices: Object.entries({
      weightsStandardisation: 'row',
      permutations: 9999,
      pCorrection: 'BH',
      significanceLevel: 0.05
    }).map(([name, value]) => ({name, value, stage: 'inference', sweepable: false}))
  });
  const columns: Record<string, number[]> = {
    values: Array.from({length: count}, (_, index) => index),
    spatialLag: Array.from({length: count}, (_, index) => index - 2047.5),
    globalStatistics: [count, 2047.5, 1, 1],
    adjustedPValues: Array.from({length: count}, () => 0.01),
    pValues: Array.from({length: count}, (_, index) => (index < 2000 ? 0.0001 : 0.075)),
    quadrants: Array.from({length: count}, (_, index) => (index < 2048 ? 3 : 1)),
    weightsRowOffsets: Array.from({length: count + 1}, (_, index) => (index <= 2345 ? 0 : 512)),
    weightsNeighbors: Array.from({length: 512}, (_, index) => index),
    weightsValues: Array.from({length: 512}, () => 1 / 512)
  };
  return {
    ...record,
    captures: createGPUComputationCaptures({
      runId: record.runId,
      snapshots: Object.entries(columns).map(([name, values]) => ({
        id: name,
        name,
        keySpace: 'row',
        values,
        elementType: 'f32',
        provenance: 'gpu-readback'
      }))
    })
  };
}

test.each([
  'moran-scatter',
  'fdr-staircase'
])('%s bounds full and thumbnail twins for dense in-window evidence', view => {
  const record = makeDenseRecord();
  const evidence = JSON.stringify(record);
  const file = toRecordedLensFile(record, {
    scenarioId: 'gpu-spatial-analysis.fixture',
    recordedAt: record.createdAt,
    adapterLabel: 'Fixture adapter'
  });
  for (const size of ['full', 'thumbnail'] as const) {
    const twin = buildLensTwin(
      file,
      {
        scenario: file.scenarioId,
        rung: 'datum',
        view,
        datum: '2345'
      },
      {size}
    );
    expect(twin.viewId).toBe(view);
    expect(twin.disabled).toBeUndefined();
    expect(validateLensTwin(twin)).toEqual([]);
    expect(Buffer.byteLength(JSON.stringify(twin, null, 2))).toBeLessThanOrEqual(
      (size === 'thumbnail' ? 48 : 256) * 1024
    );
    expect(twin.figure.html).toContain('data-key="2345"');
    const marks = twin.figure.html.match(/data-action="pick"/g) ?? [];
    expect(marks.length).toBeLessThanOrEqual(size === 'thumbnail' ? 29 : 165);
    if (view === 'moran-scatter') {
      expect(twin.figure.html).toContain('of 4096 scatter points shown');
      expect(twin.keyNumber.value).toBe('1');
      for (let neighbor = 0; neighbor < 16; neighbor++) {
        expect(twin.figure.html).toContain(`data-key="${neighbor}"`);
      }
    } else {
      expect(twin.figure.html).toContain('sampled ranks; crossing and pinned rank retained');
      expect(twin.figure.html).toContain('data-key="1999"');
      expect(twin.keyNumber.value).toBe('2,000');
    }
  }
  expect(JSON.stringify(record)).toBe(evidence);
});

test('display decimation leaves full-family values and the exact BH crossing intact', () => {
  const record = makeDenseRecord();
  for (const size of ['full', 'thumbnail'] as const) {
    const context = createViewContext(record, {bindings: {entity: 2345}, options: {size}});
    const scatter = buildMoranScatterModel(context);
    expect(scatter.points).toHaveLength(4096);
    expect(scatter.significantCount).toBe(4096);
    expect(scatter.slope).toBe(1);
    const staircase = buildFdrStaircaseModel(context);
    expect(staircase.familySize).toBe(4096);
    expect(staircase.rejectionCount).toBe(2000);
    expect(staircase.points.some(point => point.rank === 2000)).toBe(true);
    expect(staircase.points.some(point => point.key === 2345)).toBe(true);
    expect(staircase.points.at(-1)?.rank).toBe(4096);
  }
});
