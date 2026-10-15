// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {
  DATUM_RECEIPT_VIEW,
  buildDatumReceiptModel,
  renderDatumReceipt
} from '../../../src/computation-views/views/datum-receipt';
import {viewNodeToHtml, findViewNodes} from '../../../src/computation-views/view-node';
import {makeContext, makeCapture} from '../fixtures/v4-record';

describe('DatumReceipt', () => {
  test('requires a pin and captured value, orders stages, and substitutes captured symbols', () => {
    const context = makeContext();
    expect(DATUM_RECEIPT_VIEW.satisfies(context)).toEqual({ok: true});
    expect(DATUM_RECEIPT_VIEW.satisfies({...context, bindings: {}})).toEqual({
      ok: false,
      reason: 'pin a datum (datum=<n>) to open its receipt'
    });
    expect(DATUM_RECEIPT_VIEW.satisfies({...context, bindings: {entity: 90}})).toEqual({
      ok: false,
      reason: 'no captured values for cells 90'
    });
    const model = buildDatumReceiptModel(context);
    expect(model.rows.map(row => [row.name, row.stage, row.value])).toEqual([
      ['input', 'prepare', '2'],
      ['factor', 'prepare', '2'],
      ['result', 'compute', '3'],
      ['result', 'compute', '4']
    ]);
    expect(model.formula).toMatchObject({
      formulaId: 'double',
      workedTex: '\\term{output}{4}=\\term{factor}{\\left(2\\right)}\\term{input}{\\left(2\\right)}'
    });
    expect(DATUM_RECEIPT_VIEW.describe(model, context)).toMatchObject({
      subject: 'Receipt of Cell 1',
      keyNumber: {label: 'result', value: '4'},
      facts: ['prepare: 2 quantities', 'compute: 2 quantities']
    });
    expect(DATUM_RECEIPT_VIEW.receipt!(model, context)?.rows).toHaveLength(4);
  });
  test('coarsens provisional vectors and retains unknown stages in first-seen order', () => {
    const context = makeContext({
      packs: [
        {
          id: 'test',
          analysisStages: ['known'],
          captureStages: {first: 'zeta', second: 'alpha', third: 'known'}
        }
      ]
    });
    context.record.captures!.snapshots = [
      makeCapture({
        id: 'first-0',
        name: 'first',
        values: [0, 0, 1.23456, 9.87654],
        components: 2,
        iteration: 0
      }),
      makeCapture({id: 'second', name: 'second'}),
      makeCapture({id: 'third', name: 'third'}),
      makeCapture({id: 'first-1', name: 'first'})
    ];
    const model = buildDatumReceiptModel(context);
    expect(model.rows.map(row => row.stage)).toEqual(['known', 'zeta', 'zeta', 'alpha']);
    const html = viewNodeToHtml(renderDatumReceipt(model, context));
    expect(html).toContain('1.2, 9.9');
    expect(html).toContain('data-provisional="provisional"');
    expect(html).toContain('no neighbourhood declared by the domain pack');
  });
  test('renders edge truncation and diagonal cost only when declared', () => {
    const context = makeContext({bindings: {entity: 0}});
    const model = buildDatumReceiptModel(context);
    const html = viewNodeToHtml(renderDatumReceipt(model, context));
    expect(html).toContain('outside raster (edge truncated)');
    expect(html).not.toContain('1.414');
    expect(model.neighbourhoods[0].cells?.filter(cell => cell.key === undefined)).toHaveLength(5);
    expect(html).toContain('data-action="pick"');
    expect(html).toContain('data-provenance="readback"');
    expect(html).not.toContain('<td></td>');
    expect(html).toBe(viewNodeToHtml(renderDatumReceipt(model, context)));
    const declared = {
      ...context,
      packs: context.packs!.map(pack => ({...pack, metadata: {stencilDiagonalCost: 1.414}}))
    };
    expect(
      viewNodeToHtml(renderDatumReceipt(buildDatumReceiptModel(declared), declared))
    ).toContain('1.414');
  });
  test('renders bounded CSR neighbours and reports uncaptured or unknown relations', () => {
    const context = makeContext({
      bindings: {entity: 0},
      packs: [{id: 'test', neighbourhoods: {cells: {kind: 'csr-star', relation: 'neighbours'}}}]
    });
    context.record.axes.relations = [
      {
        id: 'neighbours',
        from: 'cells',
        to: 'cells',
        cardinality: 'N:M',
        kind: 'buffer',
        timeDependent: false,
        ref: {layout: 'csr', resource: 'targets', offsetsResource: 'offsets', capture: 'targets'}
      }
    ];
    expect(buildDatumReceiptModel(context).neighbourhoods[0]).toMatchObject({
      kind: 'unknown',
      reason: 'relation values not captured'
    });
    context.record.captures!.stamps = [
      makeCapture({
        id: 'targets',
        name: 'targets',
        keySpace: 'edges',
        values: new Array(20).fill(1)
      }),
      makeCapture({
        id: 'targets-offsets',
        name: 'targets-offsets',
        keySpace: 'offsets',
        values: [0, 20]
      })
    ];
    const model = buildDatumReceiptModel(context);
    expect(model.neighbourhoods[0].neighbours).toHaveLength(20);
    const tree = renderDatumReceipt(model, context);
    expect(
      findViewNodes(
        tree,
        node => node.tag === 'circle' && node.attributes['data-action'] === 'pick'
      )
    ).toHaveLength(16);
    expect(viewNodeToHtml(tree)).toContain('+4 more');
    context.record.axes.relations = context.record.axes.relations.map(relation => ({
      ...relation,
      kind: 'unknown'
    }));
    expect(viewNodeToHtml(renderDatumReceipt(buildDatumReceiptModel(context), context))).toContain(
      'unknown — relation is unknown'
    );
  });
  test('supports candidate lists and flag/offset/slot capture values', () => {
    const context = makeContext({
      packs: [
        {id: 'test', neighbourhoods: {cells: {kind: 'candidate-list', relation: 'candidate'}}}
      ]
    });
    context.record.axes.relations = [
      {
        id: 'candidate',
        from: 'cells',
        to: 'cells',
        cardinality: '1:1',
        kind: 'buffer',
        timeDependent: false,
        ref: {layout: 'index', resource: 'targets', capture: 'targets'}
      }
    ];
    context.record.captures!.stamps = [
      makeCapture({id: 'targets', name: 'targets', keySpace: 'other', values: [1, 2, 3]})
    ];
    expect(buildDatumReceiptModel(context).neighbourhoods[0].neighbours).toEqual([
      {key: 2, value: 3}
    ]);
    expect(viewNodeToHtml(renderDatumReceipt(buildDatumReceiptModel(context), context))).toContain(
      'lens-candidate-list'
    );
    context.packs = [
      {
        id: 'test',
        neighbourhoods: {
          cells: {kind: 'flag-offset-slot', captures: ['input', 'factor', 'missing']}
        }
      }
    ];
    const html = viewNodeToHtml(renderDatumReceipt(buildDatumReceiptModel(context), context));
    expect(html).toContain('lens-flag-offset-slot');
    expect(html).toContain('unknown — value not captured');
    expect(html).not.toContain('<td></td>');
  });
});
