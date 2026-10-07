// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LoadedDataset} from '../../data/catalog';

/** Camera that frames the City of Chicago. */
export const CHICAGO_VIEW = {longitude: -87.68, latitude: 41.835, zoom: 9.9} as const;

/** `COLUMN` category option list: every iNaturalist group in the dataset plus "all". */
export const CATEGORY_SELECT_OPTIONS = [
  {value: 'all', label: 'All observations'},
  {value: 'Plants', label: 'Plants'},
  {value: 'Birds', label: 'Birds'},
  {value: 'Insects', label: 'Insects'},
  {value: 'Fungi', label: 'Fungi'},
  {value: 'Mammals', label: 'Mammals'},
  {value: 'Spiders and kin', label: 'Spiders and kin'},
  {value: 'Amphibians and reptiles', label: 'Amphibians and reptiles'},
  {value: 'Snails and mussels', label: 'Snails and mussels'},
  {value: 'Fish', label: 'Fish'},
  {value: 'Other life', label: 'Other life'}
] as const;

/** Observation columns shared by both cell scenes, decoded once. */
export type NatureCells = {
  count: number;
  /** Longitude/latitude degrees, interleaved. The cell contributors key degrees directly. */
  lngLat: Float32Array;
  /** Month 1-12 of each observation (local clock). */
  month: Uint8Array;
  /** Index into `categoryNames`. */
  category: Uint8Array;
  categoryNames: readonly string[];
  /** 1 when the community confirmed the identification (research grade). */
  researchGrade: Float32Array;
  /** Community area 1-77, 0 for none. */
  communityArea: Uint8Array;
  origin: [number, number];
};

/** Seconds since 2023-01-01 of the first day of each month, plus the end of the year. */
function getMonthStarts(): number[] {
  const starts: number[] = [];
  const base = Date.UTC(2023, 0, 1);
  for (let month = 0; month <= 12; month++) {
    starts.push((Date.UTC(2023, month, 1) - base) / 1000);
  }
  return starts;
}

/** Reads the Chicago nature dataset into the arrays the cell scenes need. */
export function readNatureCells(observations: LoadedDataset): NatureCells {
  const timestamps = observations.column<Uint32Array>('timestamp');
  const count = timestamps.length;
  const starts = getMonthStarts();
  const month = new Uint8Array(count);
  const researchGradeRaw = observations.column<Uint8Array>('researchGrade');
  const researchGrade = new Float32Array(count);
  for (let index = 0; index < count; index++) {
    const seconds = timestamps[index];
    let value = 1;
    while (value < 12 && seconds >= starts[value]) value++;
    month[index] = value;
    researchGrade[index] = researchGradeRaw[index] ? 1 : 0;
  }
  return {
    count,
    lngLat: Float32Array.from(observations.column<Float32Array>('position')),
    month,
    category: observations.column<Uint8Array>('category'),
    categoryNames: observations.categories('category'),
    researchGrade,
    communityArea: observations.column<Uint8Array>('communityArea'),
    origin: observations.defaultOrigin
  };
}

/** Deterministic per-row hash in [0, 1) used to thin a sample reproducibly. */
export function hashToUnit(index: number): number {
  let hash = Math.imul(index ^ 0x9e3779b9, 0x85ebca6b);
  hash ^= hash >>> 13;
  hash = Math.imul(hash, 0xc2b2ae35);
  hash ^= hash >>> 16;
  return (hash >>> 0) / 4294967296;
}

/** Index of a category name, or -1 for `'all'`. */
export function getCategoryIndex(columns: NatureCells, name: string): number {
  return name === 'all' ? -1 : columns.categoryNames.indexOf(name);
}

/**
 * Writes the 0/1 inclusion mask for a category, a month window and a sampled share. Returns the
 * number of rows kept. Runs on the CPU over 43.6 thousand rows, only when an option changes.
 */
export function fillObservationMask(
  columns: NatureCells,
  mask: Uint32Array,
  filter: {
    category: string;
    months: readonly [number, number];
    sampledPercent: number;
  }
): number {
  const categoryIndex = getCategoryIndex(columns, filter.category);
  const [firstMonth, lastMonth] = filter.months;
  let kept = 0;
  for (let index = 0; index < columns.count; index++) {
    const month = columns.month[index];
    let inside = month >= firstMonth && month <= lastMonth;
    if (inside && categoryIndex >= 0) inside = columns.category[index] === categoryIndex;
    if (inside && filter.sampledPercent < 100) {
      inside = hashToUnit(index) * 100 < filter.sampledPercent;
    }
    mask[index] = inside ? 1 : 0;
    kept += inside ? 1 : 0;
  }
  return kept;
}

/** Formats a 64-bit cell key given as little-endian words. */
export function formatCellKey(low: number, high: number): string {
  return `0x${((BigInt(high) << 32n) | BigInt(low)).toString(16)}`;
}

/** Decodes the tile column, row and zoom of a Quadbin key given as little-endian words. */
export function decodeQuadbin(low: number, high: number): {x: number; y: number; z: number} {
  const key = (BigInt(high) << 32n) | BigInt(low);
  const z = Number((key >> 52n) & 0x1fn);
  const path = (key & 0xfffffffffffffn) >> BigInt(52 - 2 * z);
  let x = 0;
  let y = 0;
  for (let bit = 0; bit < z; bit++) {
    x |= Number((path >> BigInt(2 * bit)) & 1n) << bit;
    y |= Number((path >> BigInt(2 * bit + 1)) & 1n) << bit;
  }
  return {x, y, z};
}

/** Center longitude/latitude of a Quadbin tile. */
export function getQuadbinCenter(low: number, high: number): [number, number] {
  const {x, y, z} = decodeQuadbin(low, high);
  const tiles = 2 ** z;
  const longitude = ((x + 0.5) / tiles) * 360 - 180;
  const latitude = (Math.atan(Math.sinh(Math.PI * (1 - (2 * (y + 0.5)) / tiles))) * 180) / Math.PI;
  return [longitude, latitude];
}

/** Rows of a `uint32x2` word buffer as BigInt keys are slow: compare by words instead. */
export function packKeyWords(low: number, high: number): string {
  return `${high}:${low}`;
}

/** Month names used by readouts and legends. */
export const MONTH_NAMES = [
  'Jan',
  'Feb',
  'Mar',
  'Apr',
  'May',
  'Jun',
  'Jul',
  'Aug',
  'Sep',
  'Oct',
  'Nov',
  'Dec'
] as const;

/** Formats a `[first, last]` month window such as `Jan-Jun`. */
export function formatMonthWindow(months: readonly [number, number]): string {
  const [first, last] = months;
  return first === last
    ? MONTH_NAMES[first - 1]
    : `${MONTH_NAMES[first - 1]}-${MONTH_NAMES[last - 1]}`;
}
