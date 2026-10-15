// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {TABLE_VIEW, buildTypedTable} from '../../../src/computation-views/views/table-view';
import {createViewContext} from '../../../src/computation-views/registry/view-registry';
import {viewNodeToHtml} from '../../../src/computation-views/view-node';
import {makeScanRecord} from '../fixtures/scan-record';

test('complete sections, numeric cells, receipt and no blank cells', () => {
  const context = createViewContext(makeScanRecord());
  const model = TABLE_VIEW.build(context);
  expect(model.sections.map(section => section.id)).toEqual(
    expect.arrayContaining([
      'graph:compiled',
      'resources',
      'captures',
      'decisions',
      'choices',
      'metrics',
      'sources',
      'formulas',
      'edges'
    ])
  );
  expect(model.sections.flatMap(section => section.table.rows.flat()).every(Boolean)).toBe(true);
  const html = viewNodeToHtml(TABLE_VIEW.render(model, context));
  expect(html).toContain('1×1×1');
  expect(html).toContain('GPU readback');
  expect(html).toContain('lens-table-number');
  expect(html).toContain('unsupported: missing subgroups');
  expect(html).not.toContain('<td></td>');
  expect(TABLE_VIEW.receipt?.(model, context)?.rows).toHaveLength(3);
  expect(TABLE_VIEW.describe(model, context).keyNumber).toEqual({label: 'nodes', value: 3});
});

test('bound node attributes and entity capture values', () => {
  const context = createViewContext(makeScanRecord(), {
    bindings: {node: 'scan/add-partials', entity: 3}
  });
  const model = TABLE_VIEW.build(context);
  expect(model.sections.find(section => section.id === 'graph:compiled')?.table.rows).toHaveLength(
    1
  );
  expect(
    model.sections.find(section => section.id === 'attributes:compiled')?.table.rows
  ).toContainEqual(['scan/add-partials', 'level', '2']);
  const captures = model.sections.find(section => section.id === 'captures')!.table;
  expect(captures.columns.at(-1)?.name).toBe('value at 3');
  expect(captures.rows[1].at(-1)).toBe('6');
});

test('column inference considers truncated rows and missing cells', () => {
  const table = buildTypedTable(
    [{value: 1, flag: true, array: [1, 2]}, {value: 'two'}, {other: 3}],
    {maxRows: 1}
  );
  expect(table.columns).toEqual([
    {name: 'value', type: 'mixed'},
    {name: 'flag', type: 'boolean'},
    {name: 'array', type: 'array'},
    {name: 'other', type: 'number'}
  ]);
  expect(table.rows).toEqual([['1', 'true', '1, 2', '—']]);
  expect(table.truncatedCount).toBe(2);
  expect(buildTypedTable(Array.from({length: 205}, (_, value) => ({value}))).truncatedCount).toBe(
    5
  );
  expect(buildTypedTable([])).toEqual({columns: [], rows: [], truncatedCount: 0});
});

test('replay badges, non-finite statistics and visible truncation', () => {
  const record = makeScanRecord();
  const capture = record.captures!.snapshots[0];
  record.captures = {
    ...record.captures!,
    replays: [
      {
        ...capture,
        id: 'replay',
        elementType: 'f32',
        values: new Float32Array([NaN, Infinity, -Infinity, 3, -2]),
        provenance: 'gpu-replay'
      }
    ]
  };
  const context = createViewContext(record);
  const model = TABLE_VIEW.build(context);
  const captures = model.sections.find(section => section.id === 'captures')!.table;
  const replay = Object.fromEntries(
    captures.columns.map((column, index) => [column.name, captures.rows[3][index]])
  );
  expect(replay).toMatchObject({
    min: '-2',
    max: '3',
    'NaN count': '1',
    'Inf count': '2',
    provenance: 'GPU replay'
  });
  expect(viewNodeToHtml(TABLE_VIEW.render(model, context))).toContain('data-texture="hatched"');
  model.sections[0].table.truncatedCount = 5;
  expect(viewNodeToHtml(TABLE_VIEW.render(model, context))).toContain('5 more rows');
});

test('metadata-only records use most concrete graph for the key number', () => {
  const record = makeScanRecord();
  record.captures = undefined;
  record.graphs = [
    {...record.graphs[0], id: 'intent', form: 'intent', nodes: []},
    ...record.graphs
  ];
  const context = createViewContext(record);
  expect(TABLE_VIEW.describe(TABLE_VIEW.build(context), context)).toMatchObject({
    keyNumber: {value: 3},
    provenance: 'metadata'
  });
});
