// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {
  encodeLensUrlState,
  decodeLensUrlState,
  getLensHistoryAction,
  readLensStateFromSearch,
  writeLensStateToSearch
} from '../../../src/computation-views/hosts/url-state';
import type {LensUrlState} from '../../../src/computation-views/types';

const state: LensUrlState = {
  runId: 'run-123',
  rung: 'kernel',
  bindings: {entity: [1, 2], iteration: 3},
  clauses: [
    {source: 'map', keySpace: 'rows', predicate: {kind: 'keys', keys: [1, 2]}, label: 'Δοκιμή 🌎'}
  ],
  params: {iteration: 3, enabled: true, label: '東京'},
  focus: 'scan/up'
};
const encodeRaw = (value: unknown) => Buffer.from(JSON.stringify(value)).toString('base64url');

describe('lens URL state', () => {
  test('preserves finite numeric axis bindings without assuming integer entity indices', () => {
    const numericState: LensUrlState = {
      ...state,
      bindings: {time: 0.25, parameter: -2, choice: [0.5, 1.5]}
    };
    expect(decodeLensUrlState(encodeLensUrlState(numericState))).toEqual({
      status: 'ok',
      state: numericState
    });
  });
  test('round trips Unicode and encodes deterministically regardless of object insertion order', () => {
    const encoded = encodeLensUrlState(state);
    expect(encoded).toMatch(/^[A-Za-z0-9_-]+$/);
    expect(decodeLensUrlState(encoded)).toEqual({status: 'ok', state});
    const reordered: LensUrlState = {
      focus: state.focus,
      params: {label: '東京', enabled: true, iteration: 3},
      bindings: {iteration: 3, entity: [1, 2]},
      clauses: state.clauses,
      rung: 'kernel',
      runId: state.runId
    };
    expect(encodeLensUrlState(reordered)).toBe(encoded);
  });
  test('reports run mismatch visibly with both the state and expected run', () => {
    const encoded = encodeLensUrlState(state);
    expect(decodeLensUrlState(encoded, {expectedRunId: 'run-new'})).toEqual({
      status: 'run-mismatch',
      state,
      expectedRunId: 'run-new'
    });
    expect(decodeLensUrlState(encoded, {expectedRunId: state.runId}).status).toBe('ok');
  });
  test('rejects missing run, unknown rung, malformed shape, clauses and encodings', () => {
    for (const invalid of [
      null,
      [],
      {...state, runId: undefined},
      {...state, runId: ''},
      {...state, rung: 'receipt'},
      {...state, bindings: null},
      {...state, clauses: [{}]},
      {...state, params: {bad: []}},
      {...state, focus: 17},
      {
        ...state,
        clauses: [{...state.clauses[0], predicate: {kind: 'range', field: 'value', range: [5, 1]}}]
      }
    ]) {
      expect(decodeLensUrlState(encodeRaw(invalid)).status).toBe('invalid');
    }
    for (const invalid of ['', '%', 'a', 'e30=', Buffer.from('{').toString('base64url'), '_w'])
      expect(decodeLensUrlState(invalid).status).toBe('invalid');
  });
  test('pushes navigation and replaces scrubs or clause edits', () => {
    expect(getLensHistoryAction(undefined, state)).toBe('push');
    expect(getLensHistoryAction(state, {...state})).toBe('none');
    expect(getLensHistoryAction(state, {...state, rung: 'table'})).toBe('push');
    expect(getLensHistoryAction(state, {...state, bindings: {entity: 4}})).toBe('push');
    expect(getLensHistoryAction(state, {...state, params: {...state.params, iteration: 4}})).toBe(
      'replace'
    );
    expect(getLensHistoryAction(state, {...state, clauses: []})).toBe('replace');
  });
  test('preserves unrelated and repeated search parameters', () => {
    const search = writeLensStateToSearch('?tag=a&tag=b&name=hello+world&l=old', 'new');
    expect(readLensStateFromSearch(search)).toBe('new');
    expect(new URLSearchParams(search).getAll('tag')).toEqual(['a', 'b']);
    expect(new URLSearchParams(search).get('name')).toBe('hello world');
    expect(readLensStateFromSearch('?other=1')).toBeUndefined();
    expect(readLensStateFromSearch(writeLensStateToSearch('', 'custom', 'lens'), 'lens')).toBe(
      'custom'
    );
  });
});
