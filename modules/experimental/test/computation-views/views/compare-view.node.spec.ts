// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {
  COMPARE_VIEW,
  buildCompareViewModel,
  renderCompareView
} from '../../../src/computation-views/views/compare-view';
import {getRunScaleRegistry} from '../../../src/computation-views/views/value-strip';
import {viewNodeToHtml, findViewNodes} from '../../../src/computation-views/view-node';
import {makeContext, makeRecord, makeCapture} from '../fixtures/v4-record';

describe('CompareView', () => {
  test('handles unchanged and nonfinite references without inventing finite deltas', () => {
    const context = makeContext();
    context.record.captures!.snapshots = [
      makeCapture({id: 'first', iteration: 0, values: [2, NaN, Infinity]}),
      makeCapture({values: [2, NaN, Infinity]})
    ];
    const model = buildCompareViewModel(context);
    expect(model).toMatchObject({changedCount: 0, equalCount: 3, maxAbsDelta: 0});
    expect(model.delta[0]).toBe(0);
    expect(model.delta[1]).toBeNaN();
    expect(model.delta[2]).toBeNaN();
    expect(viewNodeToHtml(renderCompareView(model, context))).toContain('midpoint 0 = unchanged');
  });
  test('selects last two iterations, computes B minus A and honours explicit iterations', () => {
    const context = makeContext();
    expect(COMPARE_VIEW.satisfies(context)).toEqual({ok: true});
    const model = buildCompareViewModel(context);
    expect(model).toMatchObject({
      name: 'result',
      labelA: 'iteration 0',
      labelB: 'iteration 1',
      mode: 'delta',
      count: 9,
      changedCount: 9,
      equalCount: 0,
      maxAbsDelta: 1,
      midpoint: 0,
      provenance: 'derived'
    });
    expect(model.delta).toEqual(new Array(9).fill(1));
    expect(
      buildCompareViewModel({...context, options: {iterationA: 1, iterationB: 0}}).delta
    ).toEqual(new Array(9).fill(-1));
    expect(COMPARE_VIEW.describe(model, context).keyNumber).toEqual({
      label: 'keys changed',
      value: 9
    });
    context.record.captures!.snapshots = [makeCapture()];
    expect(COMPARE_VIEW.satisfies(context)).toEqual({
      ok: false,
      reason: 'needs two references (k vs k′, run A vs B, or GPU vs replay)'
    });
  });
  test('prefers external references, then replay versus readback', () => {
    const context = makeContext({bindings: {buffer: 'result'}});
    context.record.captures!.replays = [
      makeCapture({id: 'replay', provenance: 'gpu-replay', values: new Array(9).fill(0)})
    ];
    expect(buildCompareViewModel(context)).toMatchObject({labelA: 'GPU replay', labelB: 'run'});
    const reference = makeRecord();
    reference.captures!.snapshots = [makeCapture({values: new Array(9).fill(100)})];
    expect(
      buildCompareViewModel({
        ...context,
        compare: {record: reference, label: 'A', referenceLabel: 'B'}
      })
    ).toMatchObject({labelA: 'A', labelB: 'B', delta: new Array(9).fill(100)});
  });
  test('registers a diverging zero-centred scale and renders deterministic derived marks', () => {
    const context = makeContext();
    const model = buildCompareViewModel(context);
    const html = viewNodeToHtml(renderCompareView(model, context));
    const scale = getRunScaleRegistry(context)
      .list()
      .find(candidate => candidate.variable === 'delta:result');
    expect(scale).toMatchObject({kind: 'diverging', midpoint: 0, ramp: 'rdbu', domain: [-1, 1]});
    expect(html).toContain('midpoint 0 = unchanged');
    expect(html).toContain('data-provenance="derived"');
    expect(html).toContain('data-action="pick"');
    expect(html).not.toContain('<td></td>');
    expect(html).toBe(viewNodeToHtml(renderCompareView(model, context)));
  });
  test('juxtaposes references on one shared value scale', () => {
    const context = makeContext({options: {mode: 'juxtapose'}});
    const model = buildCompareViewModel(context);
    const tree = renderCompareView(model, context);
    expect(
      findViewNodes(tree, node => node.attributes['class'] === 'lens-value-strip')
    ).toHaveLength(2);
    expect(
      getRunScaleRegistry(context)
        .list()
        .map(scale => scale.variable)
    ).toEqual(['result']);
  });
});
