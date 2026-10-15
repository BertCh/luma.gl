// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {
  SCOPE_DIAGRAM_VIEW,
  buildScopeDiagramModel,
  renderScopeDiagram
} from '../../../src/computation-views/views/scope-diagram';
import type {OperatorScope} from '../../../src/computation-views/types';
import {viewNodeToHtml} from '../../../src/computation-views/view-node';
import {makeContext} from '../fixtures/v4-record';

describe('ScopeDiagram', () => {
  test('requires an operator descriptor and respects a bound node', () => {
    const context = makeContext();
    expect(SCOPE_DIAGRAM_VIEW.satisfies(context)).toEqual({ok: true});
    expect(SCOPE_DIAGRAM_VIEW.satisfies({...context, bindings: {node: 'start'}})).toMatchObject({
      ok: false
    });
    expect(SCOPE_DIAGRAM_VIEW.satisfies({...context, packs: []})).toEqual({
      ok: false,
      reason: 'no operator with a scope × reduction descriptor (domain pack operators table)'
    });
    expect(buildScopeDiagramModel(context)).toMatchObject({
      op: 'double',
      scope: 'focal',
      reduction: 'sum',
      grid: {size: 7}
    });
  });
  test('generates every scope from metadata alone', () => {
    const expectedCounts: Record<OperatorScope, number> = {
      local: 0,
      focal: 8,
      block: 8,
      zonal: 27,
      global: 48,
      frontier: 24,
      pairwise: 12,
      solver: 48
    };
    for (const [scope, count] of Object.entries(expectedCounts)) {
      const context = makeContext({
        packs: [
          {id: 'test', operators: {double: {scope: scope as OperatorScope, reduction: 'sum'}}}
        ]
      });
      const model = buildScopeDiagramModel(context);
      expect(model.grid.cells).toHaveLength(49);
      expect(model.grid.cells.filter(cell => cell.role === 'subject')).toEqual([
        {dx: 0, dy: 0, role: 'subject'}
      ]);
      expect(SCOPE_DIAGRAM_VIEW.describe(model, context).keyNumber).toEqual({
        label: 'input cells',
        value: count
      });
      if (scope === 'zonal') expect(model.caption).toContain('schematic zones');
      if (scope === 'solver') expect(model.caption).toContain('iterates to a fixed point');
      if (scope === 'frontier')
        expect(viewNodeToHtml(renderScopeDiagram(model, context))).toContain('data-ring="2"');
    }
  });
  test('renders deterministic achromatic metadata tiers, glyph and pinned cell location', () => {
    const context = makeContext();
    const model = buildScopeDiagramModel(context);
    const html = viewNodeToHtml(renderScopeDiagram(model, context));
    expect(html).toContain('data-provenance="metadata"');
    expect(html).toContain('scope focal × reduction sum');
    expect(html).toContain('Σ');
    expect(html).toContain('opacity="0.55"');
    expect(html).toContain('Applies to cell (1, 0)');
    expect(html).toContain('Schematic generated from scope × reduction, not this run&#39;s data.');
    expect(html).not.toContain('<td></td>');
    expect(html).toBe(viewNodeToHtml(renderScopeDiagram(model, context)));
    expect(SCOPE_DIAGRAM_VIEW.describe(model, context).provenance).toBe('metadata');
  });
});
