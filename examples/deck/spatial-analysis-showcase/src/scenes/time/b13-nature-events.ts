// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LoadedDataset} from '../../data/catalog';
import {getInt64TimeWords} from '@luma.gl/experimental/gpu-dataframe';

/** Epoch milliseconds of 2023-01-01 00:00 (the dataset's clock is local time stored as UTC). */
export const NATURE_YEAR_START_MS = Date.UTC(2023, 0, 1);
/** Seconds in 2023. */
export const NATURE_YEAR_SECONDS = 365 * 86400;

/** Chicago nature observations as the typed arrays the space-time contributors read. */
export type B13NatureEvents = {
  count: number;
  origin: [number, number];
  /** Planar meters around `origin`. */
  positions: Float32Array;
  /** Seconds since 2023-01-01 00:00. */
  seconds: Float32Array;
  /** Epoch milliseconds as Int64 `(low, high)` words, ready for `uint32x2` views. */
  timeWords: Uint32Array;
  category: Uint8Array;
  categoryNames: readonly string[];
};

/** Reads the nature dataset once. */
export function readNatureEvents(observations: LoadedDataset): B13NatureEvents {
  const origin = observations.defaultOrigin;
  const raw = observations.column<Uint32Array>('timestamp');
  const count = raw.length;
  const seconds = new Float32Array(count);
  const milliseconds = new BigInt64Array(count);
  for (let index = 0; index < count; index++) {
    seconds[index] = raw[index];
    milliseconds[index] = BigInt(NATURE_YEAR_START_MS + raw[index] * 1000);
  }
  return {
    count,
    origin,
    positions: observations.projectColumn('position', origin),
    seconds,
    timeWords: getInt64TimeWords(milliseconds),
    category: observations.column<Uint8Array>('category'),
    categoryNames: observations.categories('category')
  };
}

/** Select options of the group filter: all observations plus every group (categories are display names). */
export function getNatureGroupOptions(names: readonly string[]): {value: string; label: string}[] {
  return [
    {value: 'all', label: 'All observations'},
    ...names.map((name, index) => ({value: String(index), label: name}))
  ];
}

/** 1 for observations of the selected category (or every event for `'all'`). */
export function fillCategoryMask(
  events: B13NatureEvents,
  selection: string,
  target: Uint32Array
): number {
  const wanted = selection === 'all' ? -1 : Number(selection);
  let included = 0;
  for (let index = 0; index < events.count; index++) {
    const inside = wanted < 0 || events.category[index] === wanted ? 1 : 0;
    target[index] = inside;
    included += inside;
  }
  return included;
}
