// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {
  COMMAND_GRAPH_VIEW,
  buildCommandGraphViewModel,
  renderCommandGraphView
} from '../../../src/computation-views/views/command-graph-view';
import {findViewNodes, viewNodeToHtml} from '../../../src/computation-views/view-node';
import {makeV6Context, makeV6Record} from '../fixtures/v6-command-graph';

describe('CommandGraphView', () => {
  test('registers the graph rung and explains missing graphs', () => {
    const context = makeV6Context();
    expect(COMMAND_GRAPH_VIEW).toMatchObject({
      rung: 'graph',
      recordType: 'graph',
      order: 50,
      level: 'execution',
      grain: 'contributor',
      signature: {binds: [], free: ['node'], layout: {node: 'position'}}
    });
    expect(COMMAND_GRAPH_VIEW.satisfies(context)).toEqual({ok: true});
    expect(
      COMMAND_GRAPH_VIEW.satisfies({...context, record: {...context.record, graphs: []}})
    ).toEqual({ok: false, reason: 'record has no graph'});
  });

  test('builds compiled defaults, tiers, scalar previews and only one focused thumbnail', () => {
    const context = makeV6Context();
    const model = buildCommandGraphViewModel(context);
    expect(model).toMatchObject({
      graphId: 'compiled',
      form: 'compiled',
      level: 'execution',
      preset: 'algorithm',
      hasLogical: true,
      hasCompiled: true,
      counts: {nodes: 40, edges: 45, resources: 5},
      focus: {unitId: 'node:node-2', derivedFrom: ['intent:scan'], realizedBy: []}
    });
    expect(
      model.previews
        .filter(preview => preview.thumbnail)
        .map(preview => preview.thumbnail?.captureId)
    ).toEqual(['focus-capture']);
    expect(model.previews.find(preview => preview.unitId === 'node:node-2')?.scalars).toEqual([
      {name: 'size', value: 256},
      {name: 'stride', value: 2},
      {name: 'maximumInvocationCount', value: 256}
    ]);
    expect(model.previews.every(preview => preview.scalars.length <= 3)).toBe(true);
    expect(model.previews.filter(preview => preview.hasCaptures)).toHaveLength(2);
    expect(COMMAND_GRAPH_VIEW.describe(model, context)).toMatchObject({
      subject: 'compiled command graph of Scan and relaxation',
      keyNumber: {label: 'nodes', value: 40},
      provenance: 'metadata'
    });
    expect(model).toEqual(buildCommandGraphViewModel(context));
  });

  test('renders typed arcs, achromatic tiers, group affordances and capture inspectors deterministically', () => {
    const context = makeV6Context();
    const model = buildCommandGraphViewModel(context);
    const tree = renderCommandGraphView(model, context);
    const html = viewNodeToHtml(tree);
    expect(html).toContain('lens-graph-toolbar');
    expect(html).toContain('Compiled schedule on Fixture adapter (retrospective).');
    expect(html).toContain('data-provenance="metadata"');
    expect(html).toContain('data-provenance="readback"');
    expect(html).toContain('data-action="bind" data-axis="node" data-key="node-2"');
    expect(html).toContain('data-action="toggle" data-axis="node"');
    expect(html).toContain('opacity:0.55');
    expect(html).toContain('opacity:0.2');
    expect(html).toContain('height="12"');
    expect(html).not.toContain('<td></td>');
    const paths = findViewNodes(tree, node => node.tag === 'path');
    expect(paths.filter(path => path.attributes['data-kind'] === 'indirect')).toHaveLength(2);
    expect(paths.find(path => path.attributes['data-kind'] === 'war')?.attributes).toMatchObject({
      'stroke-dasharray': '5 3',
      'data-count': 2,
      'stroke-width': 2,
      stroke: 'currentColor'
    });
    expect(
      paths.find(path => path.attributes['data-kind'] === 'explicit')?.attributes[
        'stroke-dasharray'
      ]
    ).toBe('1 3');
    expect(
      findViewNodes(tree, node => node.attributes['data-action'] === 'pin-inspector')
    ).toHaveLength(2);
    expect(html).toBe(
      viewNodeToHtml(renderCommandGraphView(buildCommandGraphViewModel(context), context))
    );
  });

  test('logical form uses intent before lowered and preserves all forty realizations', () => {
    const context = makeV6Context({bindings: {}, options: {form: 'logical', focus: 'scan'}});
    const model = buildCommandGraphViewModel(context);
    expect(model).toMatchObject({form: 'logical', graphId: 'intent', level: 'skeleton'});
    expect(model.focus?.realizedBy).toHaveLength(40);
    expect(model.focus?.realizedBy[39]).toBe('compiled:node-39');
    expect(viewNodeToHtml(renderCommandGraphView(model, context))).toContain(
      'Logical graph (prospective): what was declared, before scheduling.'
    );
    const record = makeV6Record();
    record.graphs = [{...record.graphs[0], form: 'lowered'}];
    expect(buildCommandGraphViewModel({...context, record})).toMatchObject({
      form: 'logical',
      hasLogical: true,
      hasCompiled: false
    });
  });

  test('missing logical form disables the toggle and keeps the displayed form truthful', () => {
    const context = makeV6Context({options: {form: 'logical'}});
    context.record.graphs = context.record.graphs.filter(graph => graph.form === 'compiled');
    const model = buildCommandGraphViewModel(context);
    expect(model.form).toBe('compiled');
    const button = findViewNodes(
      renderCommandGraphView(model, context),
      node => node.attributes['data-value'] === 'logical'
    )[0];
    expect(button.attributes).toMatchObject({
      disabled: true,
      title: 'No logical graph is recorded',
      'aria-pressed': 'false'
    });
  });

  test('ladder renders accesses and corresponding receipt without empty cells', () => {
    const context = makeV6Context({options: {preset: 'pass-resource'}});
    const model = buildCommandGraphViewModel(context);
    const html = viewNodeToHtml(renderCommandGraphView(model, context));
    expect(model.ladder?.passes).toHaveLength(40);
    expect(html).toContain('data-access="read"');
    expect(html).toContain('data-access="write"');
    expect(html).toContain('lens-provenance--metadata');
    expect(html).not.toContain('<td></td>');
    expect(COMMAND_GRAPH_VIEW.receipt?.(model, context)?.rows).toHaveLength(40);
  });

  test('aliasing renders shared rows, retained hatching and savings with a matching receipt', () => {
    const context = makeV6Context({options: {preset: 'memory-aliasing'}});
    const model = buildCommandGraphViewModel(context);
    const tree = renderCommandGraphView(model, context);
    const html = viewNodeToHtml(tree);
    expect(
      findViewNodes(tree, node => node.attributes['data-allocation'] === 'shared')
    ).toHaveLength(1);
    expect(html).toContain('retained for observation (lens twin)');
    expect(html).toContain('fill="url(#v6-hatched)"');
    expect(html).toContain('Aliased savings: 128 bytes');
    expect(html).not.toContain('<td></td>');
    expect(COMMAND_GRAPH_VIEW.receipt?.(model, context)?.rows).toHaveLength(5);
    expect(COMMAND_GRAPH_VIEW.describe(model, context).facts).toContain(
      'Aliased savings: 128 bytes'
    );
  });

  test('honestly renders unknown lifetimes and scopes captures to the selected graph', () => {
    const context = makeV6Context({options: {preset: 'memory-aliasing'}});
    const graph = context.record.graphs[1];
    graph.resources = graph.resources!.map(resource => ({...resource, firstNode: undefined}));
    const html = viewNodeToHtml(
      renderCommandGraphView(buildCommandGraphViewModel(context), context)
    );
    expect(html).toContain('unknown lifetime');
    expect(html).toContain('Peak transient bytes: unknown');
    const logicalContext = {...context, options: {form: 'logical'}, bindings: {node: 'scan'}};
    expect(
      buildCommandGraphViewModel(logicalContext).previews.every(preview => !preview.hasCaptures)
    ).toBe(true);
  });
});
