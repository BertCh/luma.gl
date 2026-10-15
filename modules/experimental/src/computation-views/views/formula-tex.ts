// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationFormulaSymbol} from '@luma.gl/gpgpu';
import type {LensRecord} from '../registry/view-registration';
import type {MarkProvenance} from '../types';
import {getAllCaptures, getCaptureProvenance, getCaptureValue} from './record-queries';

/** Pass to KaTeX's `macros` option to retain symbol ids as HTML data attributes. */
export const FORMULA_TEX_MACROS: Readonly<Record<string, string>> = {
  '\\term': '\\htmlData{term=#1}{#2}'
};

/** KaTeX `trust` callback permitting only the class and data commands used by formula chips. */
export function isTrustedFormulaCommand(context: {command: string}): boolean {
  return context.command === '\\htmlClass' || context.command === '\\htmlData';
}

function readGroup(tex: string, start: number): {body: string; end: number} | undefined {
  if (tex[start] !== '{') return undefined;
  let depth = 1;
  for (let index = start + 1; index < tex.length; index++) {
    if (tex[index] === '\\') {
      index++;
    } else if (tex[index] === '{') {
      depth++;
    } else if (tex[index] === '}' && --depth === 0) {
      return {body: tex.slice(start + 1, index), end: index + 1};
    }
  }
  return undefined;
}

/** Lists tagged terms, including nested terms, with half-open offsets into the original TeX. */
export function listTexTerms(
  tex: string
): Array<{symbolId: string; body: string; start: number; end: number}> {
  const terms: ReturnType<typeof listTexTerms> = [];
  for (let start = 0; start < tex.length; start++) {
    if (tex[start] !== '\\') continue;
    if (!tex.startsWith('\\term{', start)) {
      start++;
      continue;
    }
    const identifier = readGroup(tex, start + 5);
    const body = identifier && readGroup(tex, identifier.end);
    if (!identifier || !body) continue;
    terms.push({symbolId: identifier.body, body: body.body, start, end: body.end});
    for (const nested of listTexTerms(body.body)) {
      terms.push({
        ...nested,
        start: nested.start + identifier.end + 1,
        end: nested.end + identifier.end + 1
      });
    }
    start = body.end - 1;
  }
  return terms;
}

function transformTerms(tex: string, values?: ReadonlyMap<string, string>): string {
  let output = '';
  let offset = 0;
  for (const term of listTexTerms(tex)) {
    if (term.start < offset) continue;
    const body = values?.get(term.symbolId) ?? transformTerms(term.body, values);
    output += tex.slice(offset, term.start) + (values ? `\\term{${term.symbolId}}{${body}}` : body);
    offset = term.end;
  }
  return output + tex.slice(offset);
}

/**
 * TeX for a substituted readback value. Outputs print bare; every other role is parenthesised so
 * juxtaposed products (`a x` → `(2)(3)`) and negative values stay unambiguous in the worked
 * example. Non-finite values print as words, never as a number.
 */
export function formatTexValue(value: number, role: GPUComputationFormulaSymbol['role']): string {
  const text = Number.isNaN(value)
    ? '\\text{NaN}'
    : !Number.isFinite(value)
      ? value > 0
        ? '\\infty'
        : '-\\infty'
      : Number.isInteger(value)
        ? String(value)
        : String(Number(value.toPrecision(6)));
  return role === 'output' ? text : `\\left(${text}\\right)`;
}

/** Substitutes term bodies while retaining wrappers for interactive symbol binding. */
export function substituteTexTerms(tex: string, values: ReadonlyMap<string, string>): string {
  return transformTerms(tex, values);
}

/** Removes term wrappers recursively for hosts without macro support. */
export function stripTexTerms(tex: string): string {
  return transformTerms(tex);
}

/** Resolves a declared symbol against graph resources and retained shader identifiers. */
export function resolveSymbolBinding(
  record: LensRecord,
  symbol: GPUComputationFormulaSymbol
):
  | {status: 'bound'; kind: 'resource' | 'wgsl' | 'scalar'; target: string; nodeIds: string[]}
  | {status: 'unbound'}
  | {status: 'unknown'; reason: string} {
  const binding = symbol.binding;
  if (!binding) return {status: 'unbound'};
  const kind = binding.startsWith('resource:')
    ? 'resource'
    : binding.startsWith('wgsl:')
      ? 'wgsl'
      : 'scalar';
  const target = kind === 'scalar' ? binding : binding.slice(binding.indexOf(':') + 1);
  const nodeIds: string[] = [];
  const sourceIds = new Set(
    kind === 'wgsl'
      ? record.refs.sources
          .filter(source => source.text?.match(/[A-Za-z_][A-Za-z_0-9]*/g)?.includes(target))
          .map(source => source.id)
      : []
  );
  let exists = kind === 'scalar' || sourceIds.size > 0;
  for (const graph of record.graphs) {
    if (kind === 'resource' && graph.resources?.some(resource => resource.id === target))
      exists = true;
    for (const node of graph.nodes) {
      const matches =
        kind === 'resource'
          ? node.reads.includes(target) || node.writes.includes(target)
          : kind === 'wgsl' &&
            (node.bindings?.some(candidate => candidate.name === target) ||
              (node.source !== undefined && sourceIds.has(node.source)));
      if (matches) {
        exists = true;
        nodeIds.push(`${graph.id}:${node.id}`);
      }
    }
  }
  return exists
    ? {status: 'bound', kind, target, nodeIds}
    : {status: 'unknown', reason: `${kind} ${target} not found in this record`};
}

/** Finds the latest resource capture (then a symbol-named capture) and its first component. */
export function getSymbolValueAt(
  record: LensRecord,
  symbol: GPUComputationFormulaSymbol,
  entity: number
): {value: number; provenance: MarkProvenance; captureId: string} | undefined {
  const captures = getAllCaptures(record);
  const resource = symbol.binding?.startsWith('resource:') ? symbol.binding.slice(9) : undefined;
  const resourceCaptures = resource
    ? captures.filter(capture => capture.resource === resource)
    : [];
  const candidates = resourceCaptures.length
    ? resourceCaptures
    : captures.filter(capture => capture.name === symbol.id);
  const capture = candidates.sort(
    (first, second) => (second.iteration ?? Infinity) - (first.iteration ?? Infinity)
  )[0];
  if (!capture) return undefined;
  const value = getCaptureValue(capture, entity);
  return value === undefined
    ? undefined
    : {value, provenance: getCaptureProvenance(capture), captureId: capture.id};
}
