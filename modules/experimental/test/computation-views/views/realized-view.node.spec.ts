// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {makeHost} from '../fixtures/v5-host';
import {
  REALIZED_VIEW,
  mountRealizedView,
  buildRealizedViewModel,
  renderRealizedView
} from '../../../src/computation-views/views/realized-view';
import {findViewNodes, viewNodeToHtml} from '../../../src/computation-views/view-node';
import {resolveValueScale} from '../../../src/computation-views/views/value-strip';
import {makeContext, makeCapture} from '../fixtures/v5-record';

describe('realized view', () => {
  test('uses recorded execution, log-spaced frames, one scale and pinned stamps', () => {
    const context = makeContext({bindings: {entity: 1, iteration: 4}});
    expect(REALIZED_VIEW.satisfies(context)).toEqual({ok: true});
    const model = buildRealizedViewModel(context);
    expect(model.loops[0]).toMatchObject({executedIterations: 2, gated: true});
    expect(model.series[0].logScale).toBe(true);
    expect(model.keyFrames.map(frame => frame.iteration)).toEqual([0, 1, 2, 4, 8, 9]);
    expect(model.keyFrames.at(-1)?.provisional).toBe('final');
    expect(model.pinned?.settledAt).toEqual({value: 4, meaning: 'settledAt round'});
    expect(model.pinned?.trajectory).toHaveLength(7);
    const scale = resolveValueScale(context, 'state')?.scale;
    const tree = renderRealizedView(model, context);
    const html = viewNodeToHtml(tree);
    for (const text of [
      'type="range"',
      'data-action="scrub"',
      'data-bound-iteration="4"',
      'data-provisional="provisional"',
      'data-provenance="readback"',
      'lens-pinned-trajectory',
      'Test adapter'
    ])
      expect(html).toContain(text);
    expect(findViewNodes(tree, node => node.attributes['data-state'] === 'selection')).toHaveLength(
      6
    );
    expect(html).not.toContain('<td></td>');
    expect(viewNodeToHtml(renderRealizedView(buildRealizedViewModel(context), context))).toBe(html);
    expect(resolveValueScale(context, 'state')?.scale).toBe(scale);
    expect(REALIZED_VIEW.describe(model, context).keyNumber).toEqual({
      label: 'iterations executed',
      value: 2
    });
  });
  test('keeps final frame within eight, selects requested capture, and states chaotic prefix', () => {
    const context = makeContext({
      options: {capture: 'chosen'},
      packs: [{id: 'pack', metadata: {executionOrder: 'chaotic'}}]
    });
    context.record.captures = {
      ...context.record.captures!,
      snapshots: [
        ...context.record.captures!.snapshots,
        ...Array.from({length: 1025}, (_, iteration) =>
          makeCapture({id: `chosen-${iteration}`, name: 'chosen', iteration})
        )
      ]
    };
    const model = buildRealizedViewModel(context);
    expect(model.keyFrames).toHaveLength(8);
    expect(model.keyFrames.at(-1)).toMatchObject({iteration: 1024, provisional: 'final'});
    expect(model.caption).toBe(
      'Each frame is a valid execution prefix on Test adapter, not the only possible one.'
    );
  });
  test('labels key-frame captures that skip executed iterations', () => {
    const context = makeContext({options: {capture: 'chosen'}});
    context.record.captures = {
      ...context.record.captures!,
      snapshots: [0, 1, 2, 4, 8, 11].map(iteration =>
        makeCapture({id: `chosen-${iteration}`, name: 'chosen', iteration})
      )
    };
    expect(buildRealizedViewModel(context).caption).toContain(
      'Key frames only: iterations 0, 1, 2, 4, 8, 11 were captured'
    );
    context.record.captures.snapshots = context.record.captures.snapshots.slice(0, 3);
    expect(buildRealizedViewModel(context).caption).not.toContain('Key frames only');
  });
  test('does not count skipped nodes or invent stamps, and disables missing traces', () => {
    const context = makeContext({bindings: {entity: 1}, options: {keySpace: 'other'}});
    context.record.metrics = [
      {
        ...context.record.metrics![0],
        nodes: {'relax-0': {cpuEncodeMs: 1, gpuBegin: 0, gpuEnd: 2, outcome: 'skipped'}}
      }
    ];
    let model = buildRealizedViewModel(context);
    expect(model.loops[0].executedIterations).toBe(0);
    expect(model.pinned?.settledAt).toHaveProperty('unknown');
    expect(model.pinned?.trajectory).toEqual([]);
    context.record.metrics = undefined;
    model = buildRealizedViewModel(context);
    expect(model.loops[0].executedIterations).toBeUndefined();
    context.record.captures = undefined;
    expect(REALIZED_VIEW.satisfies(context)).toEqual({
      ok: false,
      reason:
        'no realized iterations recorded (needs a loop trace: series, per-iteration snapshots or stamps)'
    });
  });
  test('keeps nonpositive convergence linear and preserves replay texture', () => {
    const context = makeContext();
    context.record.metrics = [
      {
        ...context.record.metrics![0],
        series: [{name: 'residual', axis: 'iteration', values: [1000, 0, -1]}]
      }
    ];
    context.record.captures = {
      ...context.record.captures!,
      snapshots: [makeCapture({provenance: 'gpu-replay'})]
    };
    const model = buildRealizedViewModel(context);
    expect(model.series[0].logScale).toBe(false);
    const html = viewNodeToHtml(renderRealizedView(model, context));
    expect(html).toContain('data-provenance="replay"');
    expect(html).toContain('id="v5-frame-0-hatched"');
  });
});

test('reads a published GPU iteration counter as executed iterations of its loop', () => {
  const context = makeContext();
  context.record.metrics = undefined;
  expect(buildRealizedViewModel(context).loops[0].executedIterations).toBeUndefined();
  context.record.captures = {
    ...context.record.captures!,
    snapshots: [
      ...context.record.captures!.snapshots,
      makeCapture({
        id: 'iteration-count',
        name: 'iterationCount',
        iteration: undefined,
        elementType: 'u32',
        values: [7],
        meaning: 'executed iterations: solve'
      })
    ]
  };
  const model = buildRealizedViewModel(context);
  expect(model.loops.find(loop => loop.loopId === 'solve')?.executedIterations).toBe(7);
  expect(REALIZED_VIEW.describe(model, context).keyNumber).toEqual({
    label: 'iterations executed',
    value: 7
  });
});

test('realized range excludes declared but unexecuted loop copies', () => {
  const context = makeContext();
  const graph = context.record.graphs[0];
  context.record.graphs = [
    {
      ...graph,
      nodes: [
        ...graph.nodes,
        {...graph.nodes[0], id: 'unexecuted', loop: {id: 'solve', iteration: 99}}
      ]
    }
  ];
  expect(buildRealizedViewModel(context).iterationRange).toEqual({min: 0, max: 9});
});

test('realized mount replaces records and stops updating after destruction', () => {
  const host = makeHost();
  const record = makeContext().record;
  const mounted = mountRealizedView(host.element, record);
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
