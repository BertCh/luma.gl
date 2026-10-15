// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {
  FORMULA_TEX_MACROS,
  isTrustedFormulaCommand,
  listTexTerms,
  substituteTexTerms,
  stripTexTerms,
  resolveSymbolBinding,
  getSymbolValueAt
} from '../../../src/computation-views/views/formula-tex';
import {makeRecord, makeCapture} from '../fixtures/v4-record';

describe('formula TeX', () => {
  test('parses balanced nested groups and preserves offsets and wrappers', () => {
    const tex = 'a+\\term{x}{\\frac{1}{\\term{y}{y_i}}}+\\term{z}{\\{z\\}}';
    const terms = listTexTerms(tex);
    expect(terms.map(term => term.symbolId)).toEqual(['x', 'y', 'z']);
    expect(terms.map(term => tex.slice(term.start, term.end))).toEqual([
      '\\term{x}{\\frac{1}{\\term{y}{y_i}}}',
      '\\term{y}{y_i}',
      '\\term{z}{\\{z\\}}'
    ]);
    expect(substituteTexTerms(tex, new Map([['y', '4']]))).toContain('\\frac{1}{\\term{y}{4}}');
    expect(stripTexTerms(tex)).toBe('a+\\frac{1}{y_i}+\\{z\\}');
    expect(substituteTexTerms(tex, new Map([['x', '9']]))).toContain('\\term{x}{9}');
    expect(listTexTerms('\\term{x}{oops')).toEqual([]);
    expect(listTexTerms('\\\\term{x}{not a command}')).toEqual([]);
    expect(stripTexTerms('\\term{x}{oops')).toBe('\\term{x}{oops');
  });
  test('limits trusted KaTeX commands', () => {
    expect(FORMULA_TEX_MACROS['\\term']).toBe('\\htmlData{term=#1}{#2}');
    expect(isTrustedFormulaCommand({command: '\\htmlClass'})).toBe(true);
    expect(isTrustedFormulaCommand({command: '\\htmlData'})).toBe(true);
    expect(isTrustedFormulaCommand({command: '\\href'})).toBe(false);
    expect(isTrustedFormulaCommand({command: '\\includegraphics'})).toBe(false);
  });
  test('resolves resources, WGSL words, scalar and missing bindings honestly', () => {
    const record = makeRecord();
    const resolve = (binding?: string) =>
      resolveSymbolBinding(record, {id: 'test', role: 'input', binding});
    expect(resolve('resource:output')).toEqual({
      status: 'bound',
      kind: 'resource',
      target: 'output',
      nodeIds: ['compiled:finish']
    });
    expect(resolve('resource:unused')).toMatchObject({status: 'bound', nodeIds: []});
    expect(resolve('resource:missing')).toMatchObject({status: 'unknown'});
    expect(resolve('wgsl:weights')).toMatchObject({status: 'bound', nodeIds: ['compiled:finish']});
    expect(resolve('wgsl:sourceValue')).toMatchObject({status: 'bound'});
    expect(resolve('wgsl:source')).toMatchObject({status: 'unknown'});
    record.refs.sources = [
      ...record.refs.sources,
      {
        id: 'unused-source',
        language: 'wgsl',
        moduleId: 'unused',
        hash: 'unused',
        text: 'let orphan = 1;'
      }
    ];
    expect(resolve('wgsl:orphan')).toMatchObject({status: 'bound', nodeIds: []});
    expect(resolve('2')).toMatchObject({status: 'bound', kind: 'scalar', target: '2'});
    expect(resolve()).toEqual({status: 'unbound'});
  });
  test('prefers highest-iteration resource capture, then name, with first-component indexing', () => {
    const record = makeRecord();
    expect(
      getSymbolValueAt(record, {id: 'output', role: 'output', binding: 'resource:output'}, 1)
    ).toEqual({value: 4, provenance: 'readback', captureId: 'result-1'});
    record.captures!.replays = [
      makeCapture({
        id: 'vector',
        name: 'vector',
        resource: undefined,
        components: 2,
        values: [10, 11, 20, 21],
        provenance: 'gpu-replay'
      })
    ];
    expect(
      getSymbolValueAt(record, {id: 'vector', role: 'input', binding: 'resource:absent'}, 1)
    ).toEqual({value: 20, provenance: 'replay', captureId: 'vector'});
    expect(getSymbolValueAt(record, {id: 'vector', role: 'input'}, 2)).toBeUndefined();
  });
});
