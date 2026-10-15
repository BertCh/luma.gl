// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {MarkProvenance} from '../types';

/** Per-datum event stamps captured from the real run, with an optional never sentinel. */
export type DatumStamps = {
  name: string;
  values: ArrayLike<number>;
  neverValue?: number;
  provenance: MarkProvenance;
};

/** Queries over captured stamps; all-settled and last-change use the latest stamp. */
export type StepToEventQuery =
  | {kind: 'first-settled'; keys: readonly number[]}
  | {kind: 'last-change'; keys: readonly number[]}
  | {kind: 'first-any-settled'; keys: readonly number[]}
  | {kind: 'iteration-where'; predicate: 'count-settled-at-least'; count: number};

/** Answers using only captured evidence; missing or invalid stamps remain unknown. */
export function stepToEvent(
  stamps: DatumStamps | undefined,
  query: StepToEventQuery
):
  | {status: 'ok'; iteration: number; caption: string}
  | {status: 'never'; caption: string}
  | {status: 'unknown'; caption: string} {
  const unknown = {status: 'unknown', caption: 'Event stamps are unknown'} as const;
  if (!stamps?.values.length || stamps.provenance === 'unknown') return unknown;
  const never = {
    status: 'never',
    caption: `${stamps.name}: requested event never occurred`
  } as const;
  const keys =
    query.kind === 'iteration-where'
      ? Array.from({length: stamps.values.length}, (_, index) => index)
      : query.keys;
  if (!keys.length) return unknown;
  const values: number[] = [];
  let neverCount = 0;
  for (const key of new Set(keys)) {
    if (!Number.isInteger(key) || key < 0 || key >= stamps.values.length) return unknown;
    const value = stamps.values[key];
    if (value === stamps.neverValue) neverCount++;
    else if (!Number.isInteger(value) || value < 0) return unknown;
    else values.push(value);
  }
  let iteration: number;
  if (query.kind === 'iteration-where') {
    if (!Number.isInteger(query.count) || query.count < 0) return unknown;
    if (query.count > values.length) return never;
    values.sort((left, right) => left - right);
    iteration = query.count === 0 ? 0 : values[query.count - 1];
  } else if (query.kind === 'first-any-settled') {
    if (!values.length) return never;
    iteration = values.reduce((earliest, value) => Math.min(earliest, value), Infinity);
  } else {
    if (neverCount) return never;
    iteration = values.reduce((latest, value) => Math.max(latest, value), 0);
  }
  return {
    status: 'ok',
    iteration,
    caption: `${stamps.name}: ${query.kind} at iteration ${iteration}`
  };
}
