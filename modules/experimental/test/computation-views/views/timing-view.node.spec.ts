// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {makeHost} from '../fixtures/v5-host';
import {
  TIMING_VIEW,
  mountTimingView,
  buildTimingViewModel,
  renderTimingView
} from '../../../src/computation-views/views/timing-view';
import {viewNodeToHtml} from '../../../src/computation-views/view-node';
import {makeContext} from '../fixtures/v5-record';

describe('timing view', () => {
  test('converts ns, derives namespace spans, and states encoding observer effects', () => {
    const context = makeContext();
    expect(TIMING_VIEW.satisfies(context)).toEqual({ok: true});
    const model = buildTimingViewModel(context);
    expect(model).toMatchObject({
      timebase: 'ns',
      unitLabel: 'µs',
      total: {begin: 0, end: 5},
      untimed: ['finish']
    });
    expect(model.cpuEncodeMsTotal).toBeCloseTo(0.6);
    expect(model.tracks.map(track => track.level)).toEqual([0, 1, 2]);
    expect(model.tracks[0].spans[0]).toMatchObject({
      label: 'solver',
      begin: 0,
      end: 5,
      provenance: 'derived',
      nodeIds: ['relax-0', 'relax-1']
    });
    expect(model.passes).toHaveLength(3);
    const html = viewNodeToHtml(renderTimingView(model, context));
    for (const text of [
      'coalesced: no (timestamps on)',
      'taps 1',
      'lens twin: yes',
      'retained: state',
      'timestamps disabled compute-pass coalescing',
      'data-provenance="derived"',
      'data-provenance="readback"',
      'data-action="bind"',
      'Pass index',
      'Test adapter'
    ])
      expect(html).toContain(text);
    expect(html).not.toContain('<td></td>');
    expect(viewNodeToHtml(renderTimingView(buildTimingViewModel(context), context))).toBe(html);
    expect(TIMING_VIEW.describe(model, context).keyNumber).toEqual({
      label: 'GPU time',
      value: 5,
      unit: 'µs'
    });
    expect(TIMING_VIEW.describe(model, context).facts[0]).toContain('3 µs');
  });
  test('allows CPU-only timing without claiming zero GPU duration and disables missing metrics', () => {
    const context = makeContext();
    context.record.metrics = [
      {...context.record.metrics![0], timebase: 'ms', nodes: {'relax-0': {cpuEncodeMs: 2}}}
    ];
    expect(TIMING_VIEW.satisfies(context)).toEqual({ok: true});
    const model = buildTimingViewModel(context);
    expect(model.unitLabel).toBe('ms');
    expect(model.tracks).toEqual([]);
    expect(model.untimed).toHaveLength(3);
    expect(TIMING_VIEW.describe(model, context).keyNumber.value).toBe('unknown');
    expect(viewNodeToHtml(renderTimingView(model, context))).toContain('GPU timestamps unknown');
    context.record.metrics = undefined;
    expect(TIMING_VIEW.satisfies(context)).toEqual({
      ok: false,
      reason: 'no timing sidecar (timestamps need timestamp-query)'
    });
  });
  test('prefers the bound node graph and keeps ms unchanged', () => {
    const context = makeContext({bindings: {node: 'other:relax-0'}});
    context.record.graphs = [
      ...context.record.graphs,
      {...context.record.graphs[0], id: 'other', form: 'lowered'}
    ];
    context.record.metrics = [
      ...context.record.metrics!,
      {...context.record.metrics![0], graph: 'other', timebase: 'ms'}
    ];
    const model = buildTimingViewModel(context);
    expect(model.graphId).toBe('other');
    expect(model.total.end).toBe(5000);
  });
});

test('timing mount replaces records and stops updating after destruction', () => {
  const host = makeHost();
  const record = makeContext().record;
  const mounted = mountTimingView(host.element, record);
  expect(host.getChildCount()).toBe(1);
  const original = host.getText();
  mounted.update({...record, graphs: [], metrics: undefined, captures: undefined});
  expect(host.getText()).not.toBe(original);
  expect(host.getText()).toContain('no ');
  mounted.update(record);
  expect(host.getText()).toBe(original);
  mounted.destroy();
  expect(host.getChildCount()).toBe(0);
  mounted.update(record);
  expect(host.getChildCount()).toBe(0);
});
