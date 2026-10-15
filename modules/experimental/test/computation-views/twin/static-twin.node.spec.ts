// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import {
  buildLensTwin,
  parseLensDirective,
  renderLensTwinHtml,
  validateLensTwin
} from '../../../src/computation-views/twin/static-twin';
import {
  getLensScenario,
  registerLensScenario,
  toRecordedLensFile
} from '../../../src/computation-lenses/registry';
import {
  COORDINATE_MODEL_GUARD,
  CONSERVATION_GUARD
} from '../../../src/computation-views/guards/standard-guards';
import {withGuardMetadata} from '../../../src/computation-views/guards/guards';
import {ViewRegistry} from '../../../src/computation-views/registry/view-registry';
import {TABLE_VIEW} from '../../../src/computation-views/views/table-view';
import {h} from '../../../src/computation-views/view-node';
import {makeScanRecord} from '../fixtures/scan-record';

const recorded = toRecordedLensFile(makeScanRecord(), {
  scenarioId: 'gpu-core.scan',
  recordedAt: '2026-10-14T00:00:00.000Z',
  adapterLabel: 'Apple M3',
  variant: 'metal-3'
});

test('pure deterministic twin and required contract', () => {
  const directive = {
    scenario: recorded.scenarioId,
    rung: 'table',
    datum: '3',
    iteration: '2',
    caption: 'Scan <result>'
  };
  const twin = buildLensTwin(recorded, directive);
  expect(twin).toEqual(buildLensTwin(recorded, directive));
  expect(validateLensTwin(twin)).toEqual([]);
  expect(twin).toMatchObject({
    viewId: 'table',
    keyNumber: {label: 'nodes', value: '3'},
    figure: {width: 640, height: 0},
    state: {runId: 'scan-run-1', bindings: {entity: 3, iteration: 2}}
  });
  expect(twin.provenance.observerEffects).toEqual([
    'timestamps disabled compute-pass coalescing, so passes differ from production',
    'lens twin: transients partial-a, partial-b retained instead of aliased'
  ]);
  const html = renderLensTwinHtml(twin);
  for (const text of [
    '<figcaption>',
    'Scan &lt;result&gt;',
    'class="lens-key-number"',
    'class="lens-receipt"',
    '<pre class="lens-tex">',
    'Recorded on Apple M3',
    'class="lens-ascii"'
  ])
    expect(html).toContain(text);
  expect(
    renderLensTwinHtml(twin, {
      renderTeX: (_tex, options) => (options.displayMode ? '<math>formula</math>' : '')
    })
  ).toContain('<math>formula</math>');
  expect(
    validateLensTwin({
      ...twin,
      caption: '',
      keyNumber: {label: 'nodes', value: ''},
      figure: {...twin.figure, html: ''},
      state: {...twin.state, runId: 'wrong'}
    })
  ).toHaveLength(4);
});

test('disabled rung falls back visibly without changing requested state', () => {
  const twin = buildLensTwin(recorded, {scenario: recorded.scenarioId, rung: 'kernel'});
  expect(twin).toMatchObject({
    viewId: 'table',
    rung: 'kernel',
    disabled: {reason: 'no compute node with a recorded dispatch shape and workgroup size'},
    state: {rung: 'kernel'}
  });
  expect(twin.figure.html).toContain('lens-table');
  expect(twin.description).toContain('kernel unavailable');
  expect(renderLensTwinHtml(twin)).toContain(
    'no compute node with a recorded dispatch shape and workgroup size'
  );
});

test('custom registry and root SVG dimensions', () => {
  const registry = new ViewRegistry();
  registry.register({
    ...TABLE_VIEW,
    id: 'custom',
    rung: 'kernel',
    render: () => h('svg', {width: 320, height: '120'})
  });
  expect(
    buildLensTwin(recorded, {scenario: recorded.scenarioId, rung: 'kernel'}, {registry}).figure
  ).toMatchObject({width: 320, height: 120});
});

test('directive validation names invalid attributes', () => {
  const directive = {scenario: 'gpu-core.scan', rung: 'table'};
  expect(parseLensDirective({...directive, bind: 'page', datum: '0', view: 'table'})).toMatchObject(
    {bindings: {entity: 0}, bind: 'page', viewId: 'table'}
  );
  for (const [attribute, value] of [
    ['rung', 'receipt'],
    ['scenario', 'oops'],
    ['datum', '1.5'],
    ['datum', ''],
    ['datum', '9007199254740992'],
    ['iteration', 'NaN'],
    ['iteration', '2x']
  ])
    expect(() => parseLensDirective({...directive, [attribute]: value})).toThrow(attribute);
  expect(() => buildLensTwin(recorded, {...directive, scenario: 'other.scan'})).toThrow('scenario');
});

test('evaluates the element guards of the record packs and prints verdict chips', () => {
  const registry = new ViewRegistry();
  const guardsFor = vi.fn(({element}: {element?: string}) =>
    element === 'GPUScan'
      ? [
          COORDINATE_MODEL_GUARD,
          withGuardMetadata(CONSERVATION_GUARD, {sumBefore: 10, sumAfter: 10})
        ]
      : []
  );
  registry.registerPack({id: 'gpu-core', metadata: {coordinateModel: 'geographic'}, guardsFor});
  const scenario = {
    id: 'gpu-core.scan',
    entry: 'gpu-core',
    element: 'GPUScan',
    title: 'Scan',
    audience: 'engineer' as const,
    rungs: ['table' as const],
    defaultRung: 'table' as const,
    load: async () => ({run: async () => makeScanRecord()})
  };
  if (!getLensScenario(scenario.id)) registerLensScenario(scenario);
  const twin = buildLensTwin(recorded, {scenario: recorded.scenarioId, rung: 'table'}, {registry});
  expect(guardsFor).toHaveBeenCalledWith(
    expect.objectContaining({element: getLensScenario('gpu-core.scan')!.element})
  );
  expect(twin.verdicts?.map(verdict => [verdict.guardId, verdict.status])).toEqual([
    ['coordinate-model', 'fail'],
    ['conservation', 'pass']
  ]);
  const html = renderLensTwinHtml(twin);
  expect(html).toContain('class="lens-verdicts"');
  expect(html).toContain('✕ Coordinate model: metric glyphs');
  expect(
    buildLensTwin(recorded, {scenario: recorded.scenarioId, rung: 'table'}, {registry, guards: []})
      .verdicts
  ).toBeUndefined();
  expect(
    buildLensTwin(
      recorded,
      {scenario: recorded.scenarioId, rung: 'table'},
      {registry, size: 'thumbnail'}
    ).verdicts
  ).toBeUndefined();
});

test('states COPY_SRC added for observation as an observer effect', () => {
  const record = makeScanRecord();
  const graph = record.graphs[0];
  record.graphs = [
    {
      ...graph,
      resources: graph.resources!.map(resource =>
        resource.id === 'partial-a' ? {...resource, attrs: {observerUsage: 'copy-src'}} : resource
      )
    }
  ];
  const file = toRecordedLensFile(record, {
    scenarioId: 'gpu-core.scan',
    recordedAt: '2026-10-14T00:00:00.000Z',
    adapterLabel: 'Apple M3'
  });
  expect(
    buildLensTwin(file, {scenario: 'gpu-core.scan', rung: 'table'}).provenance.observerEffects
  ).toContain('transients partial-a allocated with COPY_SRC for observation');
});
