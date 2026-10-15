// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {
  buildKernelSourceModel,
  renderKernelSource
} from '../../../src/computation-views/views/kernel-source';
import {viewNodeToHtml} from '../../../src/computation-views/view-node';
import {makeRecord} from '../fixtures/v5-record';

describe('kernel source', () => {
  test('marks declarations and entry point, escapes source, and links resources', () => {
    const record = makeRecord();
    const model = buildKernelSourceModel(record, record.graphs[0].nodes[0]);
    expect(model).toMatchObject({
      lines: [{marks: ['binding']}, {marks: ['workgroup-size']}, {marks: ['entry-point']}],
      truncatedLineCount: 0
    });
    const html = viewNodeToHtml(renderKernelSource(model));
    expect(html).toContain('data-line="1"');
    expect(html).toContain('var&lt;storage');
    expect(html).toContain('data-axis="buffer" data-key="state"');
    expect(html).toContain('lens-provenance--metadata');
    expect(html).not.toContain('<td></td>');
    expect(viewNodeToHtml(renderKernelSource(model))).toBe(html);
  });
  test('reports hash-only source and caps retained lines without substring binding matches', () => {
    const record = makeRecord();
    const source = record.refs.sources[0];
    record.refs = {...record.refs, sources: [{...source, text: undefined}]};
    expect(buildKernelSourceModel(record, record.graphs[0].nodes[0])).toEqual({
      status: 'unavailable',
      reason: 'source text not retained (only hash abc123)'
    });
    record.refs = {
      ...record.refs,
      sources: [
        {
          ...source,
          text: Array.from({length: 203}, () => 'var<uniform> valuesExtra: f32;').join('\n')
        }
      ]
    };
    const model = buildKernelSourceModel(record, record.graphs[0].nodes[0]);
    expect(model).toMatchObject({truncatedLineCount: 3});
    if ('lines' in model) {
      expect(model.lines).toHaveLength(200);
      expect(model.lines[0].marks).toEqual([]);
    }
    expect(
      buildKernelSourceModel(
        {...record, refs: {sources: [], formulas: []}},
        record.graphs[0].nodes[0]
      )
    ).toMatchObject({status: 'unavailable'});
  });
});
