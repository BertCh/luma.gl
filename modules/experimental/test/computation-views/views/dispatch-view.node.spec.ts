// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {makeHost} from '../fixtures/v5-host';
import {
  DISPATCH_VIEW,
  mountDispatchView,
  buildDispatchViewModel,
  renderDispatchView
} from '../../../src/computation-views/views/dispatch-view';
import {findViewNodes, viewNodeToHtml} from '../../../src/computation-views/view-node';
import {makeContext, makeNode} from '../fixtures/v5-record';

describe('dispatch view', () => {
  test('expands one pinned workgroup and distinguishes guard, selection, and probe', () => {
    const context = makeContext({bindings: {entity: 9}});
    expect(DISPATCH_VIEW.satisfies(context)).toEqual({ok: true});
    const model = buildDispatchViewModel(context);
    expect(model).toMatchObject({
      totalInvocations: 12,
      idleLaneCount: 2,
      expandedWorkgroup: 2,
      expandedReason: 'pinned-entity',
      pinnedLane: {globalIndex: 9, workgroup: 2, lane: 1},
      idProbe: {values: [108, 109, 110, 111], provenance: 'readback'}
    });
    expect(model.lanes).toHaveLength(4);
    expect(model.lanes.map(lane => lane.active)).toEqual([true, true, false, false]);
    const html = viewNodeToHtml(renderDispatchView(model, context));
    for (const text of [
      'data-provenance="readback"',
      'data-axis="workgroup"',
      'data-axis="lane" data-key="9"',
      'selection-halo',
      'idle: beyond the invocation guard',
      'lens-kernel-source'
    ])
      expect(html).toContain(text);
    expect(html).not.toContain('<td></td>');
    expect(viewNodeToHtml(renderDispatchView(buildDispatchViewModel(context), context))).toBe(html);
    expect(DISPATCH_VIEW.describe(model, context).keyNumber).toEqual({
      label: 'invocations',
      value: 12
    });
    expect(buildDispatchViewModel({...context, bindings: {entity: 9, workgroup: 0}})).toMatchObject(
      {expandedWorkgroup: 0, expandedReason: 'bound'}
    );
  });
  test('large workgroups use one image and bounded labels while retaining the pinned lane', () => {
    const context = makeContext({bindings: {entity: 200}});
    context.record.graphs[0].nodes = [
      makeNode({
        workgroupSize: [256, 1, 1],
        dispatch: {workgroups: [1, 1, 1], invocationCount: 230}
      })
    ];
    const tree = renderDispatchView(buildDispatchViewModel(context), context);
    expect(findViewNodes(tree, node => node.tag === 'image')).toHaveLength(1);
    expect(findViewNodes(tree, node => node.attributes['data-axis'] === 'lane')).toHaveLength(16);
    expect(viewNodeToHtml(tree)).toContain('data-axis="lane" data-key="200"');
    expect(viewNodeToHtml(tree)).toContain('256 lanes; first 15 and pinned lane labelled');
  });
  test('evaluates only explicit identity/divide relations and reports unknown mapping', () => {
    const context = makeContext({bindings: {entity: 9}});
    const graph = context.record.graphs[0];
    context.record.graphs = [
      {...graph, nodes: [makeNode({datum: {keySpace: 'rows', relation: 'mapping'}})]}
    ];
    context.record.axes = {
      ...context.record.axes,
      relations: [
        {
          id: 'mapping',
          from: 'rows',
          to: 'lanes',
          cardinality: 'N:1',
          kind: 'function',
          ref: {function: 'divide', params: {divisor: 3}},
          timeDependent: false
        }
      ]
    };
    expect(buildDispatchViewModel(context).pinnedLane).toEqual({
      globalIndex: 3,
      workgroup: 0,
      lane: 3
    });
    context.record.axes = {
      ...context.record.axes,
      relations: [{...context.record.axes.relations[0], kind: 'unknown'}]
    };
    expect(buildDispatchViewModel(context).pinnedLane).toEqual({
      unknown: 'relation mapping (unknown) not evaluable'
    });
    context.record.graphs = [{...graph, nodes: [makeNode({datum: undefined})]}];
    expect(buildDispatchViewModel(context).pinnedLane).toHaveProperty('unknown');
  });
  test('caps grids, labels missing guards and probes, and disables absent shapes', () => {
    const context = makeContext();
    context.record.captures = undefined;
    context.record.graphs = [
      {
        ...context.record.graphs[0],
        nodes: [makeNode({dispatch: {workgroups: [1025, 1, 1], indirect: true}})]
      }
    ];
    const model = buildDispatchViewModel(context);
    expect(model.workgroupGrid).toEqual({drawn: 1, aggregated: true});
    expect(model.lanes.every(lane => lane.active)).toBe(true);
    const html = viewNodeToHtml(renderDispatchView(model, context));
    expect(html).toContain('1025 workgroups');
    expect(html).toContain('invocation guard unknown');
    expect(html).toContain('id probe not recorded; lane ids are computed from the dispatch shape');
    expect(DISPATCH_VIEW.describe(model, context).provenance).toBe('metadata');
    context.record.graphs = [];
    expect(DISPATCH_VIEW.satisfies(context)).toEqual({
      ok: false,
      reason: 'no compute node with a recorded dispatch shape and workgroup size'
    });
  });
});

test('dispatch lane indices flatten global x/y/z rather than concatenating workgroups', () => {
  const context = makeContext({bindings: {entity: 13}});
  context.record.graphs = [
    {
      ...context.record.graphs[0],
      nodes: [
        makeNode({dispatch: {workgroups: [2, 2, 2], invocationCount: 14}, workgroupSize: [2, 2, 2]})
      ]
    }
  ];
  const model = buildDispatchViewModel(context);
  expect(model.pinnedLane).toEqual({globalIndex: 13, workgroup: 2, lane: 3});
  expect(model.expandedWorkgroup).toBe(2);
  expect(model.lanes.map(lane => lane.globalIndex)).toEqual([8, 9, 12, 13, 24, 25, 28, 29]);
  expect(model.lanes.filter(lane => lane.active)).toHaveLength(4);
});

test('dispatch rejects probes belonging to another graph or node', () => {
  const context = makeContext();
  context.record.captures = {
    ...context.record.captures!,
    snapshots: context.record.captures!.snapshots.map(capture => ({...capture, graph: 'other'}))
  };
  expect(buildDispatchViewModel(context).idProbe).toEqual({status: 'not-recorded'});
});

test('dispatch mount replaces records and stops updating after destruction', () => {
  const host = makeHost();
  const record = makeContext().record;
  const mounted = mountDispatchView(host.element, record);
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
