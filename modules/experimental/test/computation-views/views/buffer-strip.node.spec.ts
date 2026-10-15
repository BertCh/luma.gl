// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {
  BUFFER_STRIP_VIEW,
  buildBufferStripModel,
  renderBufferStrip
} from '../../../src/computation-views/views/buffer-strip';
import {viewNodeToHtml} from '../../../src/computation-views/view-node';
import {makeContext, makeCapture} from '../fixtures/v4-record';

describe('BufferStrip', () => {
  test('selects final capture of last snapshot name and computes statistics, histogram, envelope and diff', () => {
    const context = makeContext();
    expect(BUFFER_STRIP_VIEW.satisfies(context)).toEqual({ok: true});
    const model = buildBufferStripModel(context);
    expect(model).toMatchObject({
      captureId: 'result-1',
      count: 9,
      indexSpace: '2d',
      stats: {min: 2, max: 18, finiteCount: 9, nanCount: 0, infCount: 0},
      diff: {againstCaptureId: 'result-0', changedCount: 9, maxAbsDelta: 1}
    });
    expect(model.histogram.counts.reduce((total, value) => total + value, 0)).toBe(9);
    expect(model.histogram.edges).toHaveLength(17);
    expect(model.envelope?.min).toEqual([1, 3, 5, 7, 9, 11, 13, 15, 17]);
    expect(model.envelope?.max).toEqual([2, 4, 6, 8, 10, 12, 14, 16, 18]);
    expect(model.table[1]).toEqual({key: 1, value: 4, delta: 1});
    expect(BUFFER_STRIP_VIEW.describe(model, context).keyNumber).toEqual({label: 'max', value: 18});
    expect(buildBufferStripModel({...context, bindings: {buffer: 'result-0'}}).captureId).toBe(
      'result-0'
    );
    context.record.captures = undefined;
    expect(BUFFER_STRIP_VIEW.satisfies(context)).toEqual({
      ok: false,
      reason: 'no captured buffers in this record'
    });
  });
  test('handles nonfinite values and vector components, and centres a bounded table on the pin', () => {
    const context = makeContext();
    context.record.captures!.snapshots = [
      makeCapture({components: 2, values: [NaN, Infinity, -Infinity, 4, 8, 6]})
    ];
    let model = buildBufferStripModel(context);
    expect(model).toMatchObject({
      count: 3,
      stats: {nanCount: 1, infCount: 2, finiteCount: 3, min: 4, max: 8}
    });
    expect(model.table[1].value).toBe('-Infinity, 4');
    let html = viewNodeToHtml(renderBufferStrip(model, context));
    expect(html).toContain('lens-badge--nan');
    expect(html).toContain('2 Inf');
    context.record.captures!.snapshots = [makeCapture({values: [NaN, NaN]})];
    model = buildBufferStripModel(context);
    expect(BUFFER_STRIP_VIEW.describe(model, context).keyNumber).toEqual({
      label: 'NaN count',
      value: 2
    });
    context.record.captures!.snapshots = [
      makeCapture({values: Array.from({length: 200}, (_, index) => index)})
    ];
    model = buildBufferStripModel({...context, bindings: {entity: 100}});
    expect(model.table).toHaveLength(64);
    expect(model.table[0].key).toBe(68);
    context.record.axes.keySpaces = [{id: 'cells', label: 'Cell', indexSpace: 'csr'}];
    model = buildBufferStripModel(context);
    html = viewNodeToHtml(renderBufferStrip(model, context));
    expect(html).toContain('CSR offsets not captured');
    context.record.captures!.stamps = [
      makeCapture({id: 'offsets', name: 'result-offsets', values: [0, 3, 200]})
    ];
    expect(viewNodeToHtml(renderBufferStrip(model, context))).toContain('lens-csr-brackets');
  });
  test('renders deterministic lineage, pickable cells, typed table and block aggregation', () => {
    const context = makeContext();
    const model = buildBufferStripModel(context);
    const html = viewNodeToHtml(renderBufferStrip(model, context));
    expect(html).toContain('data-provenance="readback"');
    expect(html).toContain('data-provenance="derived"');
    expect(html).toContain('data-action="pick"');
    expect(html).toContain('lens-buffer-table');
    expect(html).not.toContain('<td></td>');
    expect(html).toBe(viewNodeToHtml(renderBufferStrip(model, context)));
    const provisional = {...context, bindings: {buffer: 'result-0'}};
    expect(
      viewNodeToHtml(renderBufferStrip(buildBufferStripModel(provisional), provisional))
    ).toContain('data-provisional="provisional"');
    context.record.axes.keySpaces = [
      {id: 'cells', label: 'Cell', indexSpace: '2d', shape: [257, 1]}
    ];
    context.record.captures!.snapshots = [makeCapture({values: new Array(257).fill(2)})];
    expect(viewNodeToHtml(renderBufferStrip(buildBufferStripModel(context), context))).toContain(
      'data-sampling="mean"'
    );
  });
});
