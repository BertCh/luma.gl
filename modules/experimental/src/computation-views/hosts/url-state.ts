// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {LENS_RUNG_IDS, type LensUrlState, type SelectionClause} from '../types';

/** Encodes canonical JSON as unpadded base64url, including non-ASCII labels. */
export function encodeLensUrlState(state: LensUrlState): string {
  const bytes = new TextEncoder().encode(canonicalJson(state));
  let binary = '';
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

/** Validates untrusted permalink data, making a different captured run visible to the host. */
export function decodeLensUrlState(
  encoded: string,
  options: {expectedRunId?: string} = {}
):
  | {status: 'ok'; state: LensUrlState}
  | {status: 'run-mismatch'; state: LensUrlState; expectedRunId: string}
  | {status: 'invalid'; reason: string} {
  try {
    if (!encoded || !/^[A-Za-z0-9_-]+$/.test(encoded) || encoded.length % 4 === 1)
      return {status: 'invalid', reason: 'Invalid base64url state'};
    const binary = atob(encoded.replace(/-/g, '+').replace(/_/g, '/'));
    const state: unknown = JSON.parse(
      new TextDecoder('utf-8', {fatal: true}).decode(
        Uint8Array.from(binary, character => character.charCodeAt(0))
      )
    );
    if (!isLensUrlState(state))
      return {
        status: 'invalid',
        reason: 'Invalid lens state: runId, rung, bindings, clauses and params are required'
      };
    if (options.expectedRunId !== undefined && state.runId !== options.expectedRunId)
      return {status: 'run-mismatch', state, expectedRunId: options.expectedRunId};
    return {status: 'ok', state};
  } catch {
    return {status: 'invalid', reason: 'Malformed lens state encoding or JSON'};
  }
}

/** Rung/binding navigation pushes history; scrubs and other state edits replace it. */
export function getLensHistoryAction(
  previous: LensUrlState | undefined,
  next: LensUrlState
): 'push' | 'replace' | 'none' {
  if (!previous) return 'push';
  if (canonicalJson(previous) === canonicalJson(next)) return 'none';
  if (
    previous.rung !== next.rung ||
    previous.runId !== next.runId ||
    canonicalJson(previous.bindings) !== canonicalJson(next.bindings)
  )
    return 'push';
  return 'replace';
}

/** Reads an encoded lens state without interpreting unrelated query parameters. */
export function readLensStateFromSearch(search: string, parameterName = 'l'): string | undefined {
  return new URLSearchParams(search).get(parameterName) ?? undefined;
}

/** Writes one lens parameter, preserving other query parameters and a leading question mark. */
export function writeLensStateToSearch(
  search: string,
  encoded: string,
  parameterName = 'l'
): string {
  const parameters = new URLSearchParams(search);
  parameters.set(parameterName, encoded);
  return `?${parameters.toString()}`;
}

function canonicalJson(value: unknown): string {
  return JSON.stringify(value, (_key, entry: unknown) => {
    if (entry && typeof entry === 'object' && !Array.isArray(entry)) {
      return Object.fromEntries(
        Object.entries(entry).sort(([left], [right]) => (left < right ? -1 : left > right ? 1 : 0))
      );
    }
    return entry;
  });
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function isKey(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value >= 0;
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

function isClause(value: unknown): value is SelectionClause {
  if (
    !isRecord(value) ||
    typeof value['source'] !== 'string' ||
    typeof value['keySpace'] !== 'string' ||
    typeof value['label'] !== 'string'
  )
    return false;
  if (
    value['clients'] !== undefined &&
    (!Array.isArray(value['clients']) ||
      !value['clients'].every(client => typeof client === 'string'))
  )
    return false;
  const predicate = value['predicate'];
  if (!isRecord(predicate)) return false;
  switch (predicate['kind']) {
    case 'all':
      return true;
    case 'keys':
      return Array.isArray(predicate['keys']) && predicate['keys'].every(isKey);
    case 'range':
      return (
        typeof predicate['field'] === 'string' &&
        Array.isArray(predicate['range']) &&
        predicate['range'].length === 2 &&
        predicate['range'].every(bound => typeof bound === 'number' && Number.isFinite(bound)) &&
        predicate['range'][0] <= predicate['range'][1]
      );
    default:
      return false;
  }
}

function isLensUrlState(value: unknown): value is LensUrlState {
  if (!isRecord(value)) return false;
  return (
    typeof value['runId'] === 'string' &&
    value['runId'].length > 0 &&
    LENS_RUNG_IDS.some(rung => rung === value['rung']) &&
    isRecord(value['bindings']) &&
    Object.values(value['bindings']).every(
      binding =>
        typeof binding === 'string' ||
        isFiniteNumber(binding) ||
        (Array.isArray(binding) && binding.every(isFiniteNumber))
    ) &&
    Array.isArray(value['clauses']) &&
    value['clauses'].every(isClause) &&
    isRecord(value['params']) &&
    Object.values(value['params']).every(
      parameter =>
        typeof parameter === 'string' ||
        typeof parameter === 'boolean' ||
        (typeof parameter === 'number' && Number.isFinite(parameter))
    ) &&
    (value['focus'] === undefined || typeof value['focus'] === 'string')
  );
}
