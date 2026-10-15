// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {makeHost} from '../fixtures/v5-host';
import {
  SKELETON_VIEW,
  mountSkeletonView,
  buildSkeletonViewModel,
  renderSkeletonView
} from '../../../src/computation-views/views/skeleton-view';
import {findViewNodes, viewNodeToHtml} from '../../../src/computation-views/view-node';
import {makeContext, makeNode} from '../fixtures/v5-record';

describe('skeleton view', () => {
  test('folds loops and renders deterministic outline-only declared structure', () => {
    const context = makeContext();
    expect(SKELETON_VIEW.satisfies(context)).toEqual({ok: true});
    const model = buildSkeletonViewModel(context);
    expect(model).toMatchObject({layers: 2, depth: 2, work: 3, workUnit: 'nodes'});
    expect(model.nodes[0].loop).toEqual({iterationCount: 2, maximumIterations: 16});
    const tree = renderSkeletonView(model, context);
    expect(
      findViewNodes(tree, node => node.tag === 'rect').every(
        node =>
          node.attributes['fill'] === 'none' && node.attributes['data-provenance'] === 'metadata'
      )
    ).toBe(true);
    const html = viewNodeToHtml(tree);
    expect(html).toContain('data-action="bind"');
    expect(html).toContain('×2');
    expect(html).toContain('Declared structure, not this run&#39;s schedule');
    expect(viewNodeToHtml(renderSkeletonView(buildSkeletonViewModel(context), context))).toBe(html);
    expect(SKELETON_VIEW.describe(model, context).keyNumber).toEqual({label: 'depth', value: 2});
    context.record.graphs = [];
    expect(SKELETON_VIEW.satisfies(context)).toEqual({
      ok: false,
      reason: 'no graph form to draw a declared structure from'
    });
  });
  test('prefers intent, sums complete workloads, and summarizes more than forty nodes', () => {
    const context = makeContext();
    const graph = context.record.graphs[0];
    context.record.graphs = [
      graph,
      {
        ...graph,
        id: 'intent',
        form: 'intent',
        nodes: Array.from({length: 41}, (_, index) =>
          makeNode({id: `node-${index}`, loop: undefined, workload: {maximumInvocationCount: 10}})
        ),
        edges: []
      }
    ];
    const model = buildSkeletonViewModel(context);
    expect(model).toMatchObject({graphId: 'intent', work: 410, workUnit: 'invocations'});
    expect(viewNodeToHtml(renderSkeletonView(model, context))).toContain('41 nodes');
    expect(
      findViewNodes(renderSkeletonView(model, context), node => node.tag === 'rect')
    ).toHaveLength(1);
  });
  test('shows a declared loop bound without implying execution', () => {
    const context = makeContext();
    context.record.graphs = [
      {
        ...context.record.graphs[0],
        nodes: [makeNode({kind: 'loop', loop: undefined, attrs: {maximumIterations: 32}})],
        edges: []
      }
    ];
    expect(viewNodeToHtml(renderSkeletonView(buildSkeletonViewModel(context), context))).toContain(
      '≤ 32'
    );
  });
});

test('skeleton mount replaces records and stops updating after destruction', () => {
  const host = makeHost();
  const record = makeContext().record;
  const mounted = mountSkeletonView(host.element, record);
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
