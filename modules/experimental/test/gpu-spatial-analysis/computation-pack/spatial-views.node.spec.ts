// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import type {LensRecord} from '../../../src/computation-views/registry/view-registration';
import {createDefaultViewRegistry} from '../../../src/computation-views/registry/default-registry';
import {createViewContext} from '../../../src/computation-views/registry/view-registry';
import {toRecordedLensFile} from '../../../src/computation-lenses/registry';
import {
  buildLensTwin,
  validateLensTwin,
  renderLensTwinHtml
} from '../../../src/computation-views/twin/static-twin';
import {viewNodeToHtml} from '../../../src/computation-views/view-node';
import {
  NEIGHBOURHOOD_VIEW,
  MORAN_SCATTER_VIEW,
  FDR_STAIRCASE_VIEW,
  buildNeighbourhoodViewModel,
  buildMoranScatterModel,
  buildFdrStaircaseModel
} from '../../../src/gpu-spatial-analysis/computation-pack';

function makeRecord(standardisation = 'row'): LensRecord {
  const captures: Record<string, number[]> = {
    values: [1, 2, 3, 4, 5, 6, 7, 8],
    localI: [1, 1, 1, 0, 0, 1, 1, 1],
    spatialLag: [-2.5, -2, -1.5, 0, 0, 1.5, 2, 2.5],
    zScores: [-2, -1, 0, 0, 0, 1, 2, 3],
    pValues: [0.001, 0.01, 0.02, 0.04, 0.3, 0.5, 0.7, 0.9],
    adjustedPValues: [0.008, 0.04, 0.053333, 0.08, 0.48, 0.6667, 0.8, 0.9],
    quadrants: [3, 3, 3, 0, 0, 1, 1, 1],
    neighborCounts: [1, 2, 1, 0, 0, 1, 2, 1],
    weightsRowOffsets: [0, 1, 3, 4, 4, 4, 5, 7, 8],
    weightsNeighbors: [1, 0, 2, 1, 6, 5, 7, 6],
    weightsValues: [1, 0.5, 0.5, 1, 1, 0.5, 0.5, 1],
    globalStatistics: [8, 4.5, 5.25, Math.sqrt(5.25), 2]
  };
  return {
    schema: 'luma.computation/1',
    id: 'spatial-fixture',
    runId: 'spatial-run',
    createdAt: '2026-10-14T00:00:00Z',
    adapter: {backend: 'webgpu', description: 'Fixture adapter'},
    domains: ['gpu-spatial-analysis'],
    graphs: [],
    axes: {
      axes: [{id: 'entity', label: 'row', kind: 'data', keySpace: 'row', size: 8}],
      keySpaces: [{id: 'row', label: 'row', size: 8}],
      relations: []
    },
    refs: {formulas: [], sources: []},
    decisions: [],
    choices: Object.entries({
      weightsStandardisation: standardisation,
      permutations: 99,
      multipleTesting: 'BH',
      pCorrection: 'BH',
      significanceLevel: 0.05,
      islandCount: 2
    }).map(([name, value]) => ({name, value, stage: 'inference', sweepable: false})),
    captures: {
      schema: 'luma.computation-captures/1',
      runId: 'spatial-run',
      snapshots: Object.entries(captures).map(([name, values]) => ({
        id: name,
        name,
        keySpace: 'row',
        values,
        elementType: 'f32',
        provenance: 'gpu-readback'
      })),
      stamps: [],
      replays: []
    }
  };
}

function renderTwin(record: LensRecord, view: string) {
  return buildLensTwin(
    toRecordedLensFile(record, {
      scenarioId: 'gpu-spatial-analysis.fixture',
      recordedAt: '2026-10-14',
      adapterLabel: 'Fixture adapter'
    }),
    {scenario: 'gpu-spatial-analysis.fixture', rung: 'datum', view, datum: '1'},
    {registry: createDefaultViewRegistry()}
  );
}

describe('spatial views', () => {
  for (const view of [NEIGHBOURHOOD_VIEW, MORAN_SCATTER_VIEW, FDR_STAIRCASE_VIEW]) {
    test(`${view.id} satisfies named captures, disables specifically and builds deterministic twins`, () => {
      const record = makeRecord();
      expect(view.satisfies(createViewContext(record))).toEqual({ok: true});
      const twin = renderTwin(record, view.id);
      expect(twin.disabled).toBeUndefined();
      expect(twin.viewId).toBe(view.id);
      expect(validateLensTwin(twin)).toEqual([]);
      expect(twin).toEqual(renderTwin(record, view.id));
      expect(renderLensTwinHtml(twin)).not.toMatch(/<td[^>]*><\/td>|undefined|NaN/);
      record.captures!.snapshots = [];
      expect(view.satisfies(createViewContext(record))).toEqual({
        ok: false,
        reason: `missing or empty ${view.id === 'neighbourhood-view' ? 'neighborCounts' : view.id === 'moran-scatter' ? 'values' : 'pValues'} capture`
      });
      expect(view.order).toBeGreaterThanOrEqual(60);
      expect(view.order).toBeLessThan(70);
    });
  }

  test('Moran labels the slope only with row weights; missing inference never colours quadrants', () => {
    expect(renderLensTwinHtml(renderTwin(makeRecord(), 'moran-scatter'))).toContain(
      '<p>slope = I:'
    );
    expect(renderLensTwinHtml(renderTwin(makeRecord('binary'), 'moran-scatter'))).not.toContain(
      '<p>slope = I:'
    );
    const record = makeRecord();
    record.choices = record.choices.filter(choice => choice.name !== 'significanceLevel');
    const context = createViewContext(record);
    const model = buildMoranScatterModel(context);
    expect(model.significantCount).toBe(0);
    expect(viewNodeToHtml(MORAN_SCATTER_VIEW.render(model, context))).not.toContain('cv-nom-');
    expect(renderLensTwinHtml(renderTwin(record, 'moran-scatter'))).toContain('unknown');
    record.captures!.snapshots = record.captures!.snapshots.filter(
      capture => capture.name !== 'globalStatistics'
    );
    expect(buildMoranScatterModel(createViewContext(record)).standardised).toBe(false);
    expect(renderLensTwinHtml(renderTwin(record, 'moran-scatter'))).toContain(
      'centred, not standardised'
    );
  });

  test('uses the run reference statistics and divides both scatter axes by its deviation', () => {
    const model = buildMoranScatterModel(createViewContext(makeRecord()));
    expect(model.points[0].horizontal).toBeCloseTo(-3.5 / Math.sqrt(5.25));
    expect(model.points[0].vertical).toBeCloseTo(-2.5 / Math.sqrt(5.25));
    expect(model.significantCount).toBe(2);
  });

  test('BH uses the full family: rank 2 passes 0.0125; rank 3 fails 0.01875', () => {
    const model = buildFdrStaircaseModel(createViewContext(makeRecord()));
    expect(model.rejectionCount).toBe(2);
    expect(model.floor).toBe(0.01);
    expect(model.grid[0]).toBe(0.01);
  });

  test('large FDR families retain the exact crossing and pinned rank while decimating', () => {
    const record = makeRecord();
    record.captures!.snapshots.find(capture => capture.name === 'pValues')!.values = Array.from(
      {length: 2000},
      (_, index) => (index < 123 ? 0.0001 : 0.8)
    );
    const model = buildFdrStaircaseModel({...createViewContext(record), bindings: {entity: 777}});
    expect(model.rejectionCount).toBe(123);
    expect(model.points.length).toBeLessThanOrEqual(512);
    expect(model.points.some(point => point.rank === 123)).toBe(true);
    expect(model.points.some(point => point.key === 777)).toBe(true);
  });

  test('counts islands and shows the pinned weighted star without defaulting to row zero', () => {
    const context = {...createViewContext(makeRecord()), bindings: {entity: 1}};
    const model = buildNeighbourhoodViewModel(context);
    expect(model.islands).toEqual([3, 4]);
    expect(model.rowSum).toBe(1);
    expect(model.neighbors).toEqual([
      {key: 0, weight: 0.5},
      {key: 2, weight: 0.5}
    ]);
    expect(model.verdict.reason).toBe('2 islands, hatched');
    expect(buildNeighbourhoodViewModel(createViewContext(makeRecord())).core).toBeUndefined();
  });
});

test('all-island CSR arrays are valid, overflow is a flag, and histogram bars stay bounded', () => {
  const record = makeRecord();
  const captures = record.captures!.snapshots;
  captures.find(capture => capture.name === 'neighborCounts')!.values = [0, 0, 0, 0, 0, 0, 0, 0];
  captures.find(capture => capture.name === 'weightsRowOffsets')!.values = new Uint32Array(9);
  captures.find(capture => capture.name === 'weightsNeighbors')!.values = [];
  captures.find(capture => capture.name === 'weightsValues')!.values = [];
  let context = createViewContext(record);
  expect(NEIGHBOURHOOD_VIEW.satisfies(context)).toEqual({ok: true});
  expect(buildNeighbourhoodViewModel(context).islands).toHaveLength(8);
  expect(
    viewNodeToHtml(NEIGHBOURHOOD_VIEW.render(buildNeighbourhoodViewModel(context), context))
  ).not.toContain('Edge truncation');
  record.captures!.snapshots = [
    ...captures,
    {
      id: 'overflow',
      name: 'weightsOverflow',
      keySpace: 'row',
      values: [1],
      elementType: 'u32',
      provenance: 'gpu-readback'
    }
  ];
  captures.find(capture => capture.name === 'neighborCounts')!.values = [
    0, 1, 2, 15, 16, 17, 80, 100
  ];
  context = createViewContext(record);
  const model = buildNeighbourhoodViewModel(context);
  expect(model.histogram).toHaveLength(17);
  expect(model.histogram.at(-1)).toEqual({label: '≥16', count: 4});
  expect(viewNodeToHtml(NEIGHBOURHOOD_VIEW.render(model, context))).toContain(
    'capacity exceeded; edges truncated'
  );
});

test('missing correction is unknown, none is disclosed, and high pinned p remains visible', () => {
  const record = makeRecord();
  record.choices = record.choices.map(choice =>
    choice.name === 'pCorrection' ? {...choice, value: 'none'} : choice
  );
  const context = {...createViewContext(record), bindings: {entity: 7}};
  const model = buildFdrStaircaseModel(context);
  const html = viewNodeToHtml(FDR_STAIRCASE_VIEW.render(model, context));
  expect(html).toContain('uncorrected');
  expect(html).toContain('↑ pinned p 0.9');
  expect(model.rejectionCount).toBe(2);
  record.choices = record.choices.filter(
    choice => !['pCorrection', 'multipleTesting'].includes(choice.name)
  );
  const unknownContext = createViewContext(record);
  expect(
    viewNodeToHtml(
      FDR_STAIRCASE_VIEW.render(buildFdrStaircaseModel(unknownContext), unknownContext)
    )
  ).toContain('unknown — pCorrection missing');
});
