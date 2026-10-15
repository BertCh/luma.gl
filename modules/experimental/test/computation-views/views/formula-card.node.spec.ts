// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {
  FORMULA_CARD_VIEW,
  buildFormulaCardModel,
  renderFormulaCard
} from '../../../src/computation-views/views/formula-card';
import {viewNodeToHtml} from '../../../src/computation-views/view-node';
import {makeContext, makeCapture} from '../fixtures/v4-record';

describe('FormulaCard', () => {
  test('prioritizes the bound node and fades intermediate steps before parameters', () => {
    const context = makeContext({
      bindings: {node: 'compiled:finish', entity: 1},
      options: {formula: 'other', fadeStep: 3}
    });
    const formula = context.record.refs.formulas[0];
    context.record.refs.formulas = [
      {
        ...formula,
        tex: `${formula.tex}+\\term{middle}{m}`,
        symbols: [...formula.symbols, {id: 'middle', role: 'intermediate'}]
      },
      {...formula, id: 'other'}
    ];
    context.record.captures!.replays = [
      makeCapture({id: 'middle', name: 'middle', resource: undefined, provenance: 'gpu-replay'})
    ];
    const model = buildFormulaCardModel(context);
    expect(model.formulaId).toBe('double');
    expect(model.fadedSymbolIds).toEqual(['output', 'middle', 'factor']);
    const complete = buildFormulaCardModel({...context, options: {fadeStep: 0}});
    expect(FORMULA_CARD_VIEW.describe(complete, context).provenance).toBe('replay');
    expect(
      buildFormulaCardModel({...context, bindings: {}, options: {formula: 'other'}}).formulaId
    ).toBe('other');
  });
  test('selects a formula and explains a missing closed form', () => {
    const context = makeContext();
    expect(FORMULA_CARD_VIEW.satisfies(context)).toEqual({ok: true});
    expect(buildFormulaCardModel(context)).toMatchObject({
      formulaId: 'double',
      entity: 1,
      workedTex:
        '\\term{output}{4}=\\term{factor}{\\left(2\\right)}\\term{input}{\\left(2\\right)}',
      nodes: [{id: 'compiled:finish', label: 'Double'}]
    });
    expect(FORMULA_CARD_VIEW.satisfies({...context, options: {formula: 'none'}})).toMatchObject({
      ok: false,
      reason: expect.stringContaining('no formula ref')
    });
    context.record.refs.formulas = [];
    expect(FORMULA_CARD_VIEW.satisfies(context)).toMatchObject({ok: false});
  });
  test('fades output, intermediate, parameter, then input; describes substituted outputs', () => {
    const context = makeContext();
    const model = buildFormulaCardModel(context);
    expect(FORMULA_CARD_VIEW.describe(model, context).keyNumber).toEqual({label: 'y', value: 4});
    expect(FORMULA_CARD_VIEW.describe(model, context).provenance).toBe('readback');
    const faded = buildFormulaCardModel({...context, options: {fadeStep: 2}});
    expect(faded.fadedSymbolIds).toEqual(['output', 'factor']);
    expect(faded.workedTex).toContain('\\htmlClass{lens-formula-placeholder}{y\\;?}');
    expect(faded.workedTex).toContain('\\term{input}{\\left(2\\right)}');
    expect(FORMULA_CARD_VIEW.describe(faded, context).keyNumber).toEqual({
      label: 'bound symbols',
      value: '3/5'
    });
    const unpinned = buildFormulaCardModel({...context, bindings: {}});
    expect(unpinned.workedTex).toBeUndefined();
    expect(FORMULA_CARD_VIEW.describe(unpinned, context).provenance).toBe('metadata');
    expect(buildFormulaCardModel({...context, options: {fadeStep: -1}}).fadeStep).toBe(0);
  });
  test('renders deterministic provenance, navigation and host TeX; exposes twin payloads', () => {
    const context = makeContext();
    const model = buildFormulaCardModel(context);
    const html = viewNodeToHtml(renderFormulaCard(model, context));
    expect(html).toContain('lens-view lens-formula-card');
    expect(html).toContain('curated formula');
    expect(html).toContain('data-provenance="derived"');
    expect(html).toContain('data-action="bind" data-axis="node" data-key="compiled:finish"');
    expect(html).toContain('lens-symbol-chip--unknown');
    expect(html).toContain('lens-symbol-chip--unbound');
    expect(html).not.toContain('<td></td>');
    expect(html).toBe(viewNodeToHtml(renderFormulaCard(model, context)));
    expect(FORMULA_CARD_VIEW.formula!(model, context)).toMatchObject({
      ascii: 'y = a x',
      derivation: 'curated'
    });
    expect(FORMULA_CARD_VIEW.receipt!(model, context)?.rows[2]).toEqual([
      'y',
      'output',
      'output',
      '4',
      'readback'
    ]);
    expect(
      viewNodeToHtml(
        renderFormulaCard(model, {...context, renderTeX: () => '<math>host math</math>'})
      )
    ).toContain('<math>host math</math>');
  });
});

describe('FormulaCard pack formulas', () => {
  test('an operator declared formula: none disables the rung with the pack reason', () => {
    const context = makeContext({
      bindings: {},
      options: {op: 'GPUIterativeThing'},
      packs: [
        {
          id: 'test-pack',
          formulas: {GPUIterativeThing: {formula: 'none', reason: 'iterative solver'}}
        }
      ]
    });
    expect(FORMULA_CARD_VIEW.satisfies(context)).toEqual({
      ok: false,
      reason: 'no closed-form formula: iterative solver (see the iterations rung)'
    });
  });
});
