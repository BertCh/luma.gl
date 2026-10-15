// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  buildDefaultAxisHeader,
  getAxisHeader
} from '../../../src/computation-views/registry/default-axis-header';
import {createViewContext} from '../../../src/computation-views/registry/view-registry';
import {renderAxisHeader} from '../../../src/computation-views/views/axis-header-view';
import {TABLE_VIEW} from '../../../src/computation-views/views/table-view';
import {getViewNodeText, viewNodeToHtml} from '../../../src/computation-views/view-node';
import {makeScanRecord} from '../fixtures/scan-record';

test('explicit pin, wildcard, adapter and delegated button', () => {
  const view = {
    ...TABLE_VIEW,
    signature: {
      binds: ['entity'],
      free: ['iteration'],
      layout: {iteration: 'small-multiple' as const}
    }
  };
  const context = createViewContext(makeScanRecord(), {bindings: {entity: 17}});
  const model = buildDefaultAxisHeader(view, context);
  const tree = renderAxisHeader(model);
  expect(getViewNodeText(tree)).toBe(
    'row 17 pinned · iteration free (all 64 iterations, small multiple) · Apple M3 · metal-3'
  );
  expect(viewNodeToHtml(tree)).toContain('type="button"');
  expect(viewNodeToHtml(tree)).toContain(
    'data-action="free" data-axis="entity" aria-label="Free entity"'
  );
  expect(
    buildDefaultAxisHeader(view, {...context, bindings: {entity: [1, 2, 3]}}).chips[0].valueText
  ).toBe('{3 keys}');
  expect(getAxisHeader(view, TABLE_VIEW.build(context), context)).toEqual(model);
  expect(
    getAxisHeader(
      {...view, axisHeader: () => ({chips: [], adapterLabel: 'override'})},
      TABLE_VIEW.build(context),
      context
    ).adapterLabel
  ).toBe('override');
});

test('unknown size and adapter fallbacks remain explicit', () => {
  const record = makeScanRecord();
  record.adapter = {backend: 'webgpu', vendor: 'vendor'};
  const context = createViewContext(record);
  expect(buildDefaultAxisHeader(TABLE_VIEW, context)).toMatchObject({
    adapterLabel: 'vendor',
    chips: [{valueText: 'all nodes, position'}, {valueText: 'all buffers, position'}]
  });
  record.adapter = {backend: 'webgpu'};
  expect(buildDefaultAxisHeader(TABLE_VIEW, context).adapterLabel).toBe('webgpu');
});
