// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LoadedDataset} from '../../data/catalog';

/** Camera that frames the City of Chicago. */
export const CHICAGO_VIEW = {longitude: -87.68, latitude: 41.835, zoom: 9.8} as const;

/** Iconic groups that are not wildlife (used for the "animal share" weight). */
const NON_ANIMAL_CATEGORIES = new Set(['Plants', 'Fungi', 'Other life']);

/** Display name of an observation group. The dataset already stores display names. */
export function formatCategory(name: string): string {
  return name;
}

/** Chicago nature-observation columns decoded once and shared by the points scenes. */
export type NatureColumns = {
  count: number;
  /** Planar meters around `origin`. */
  positions: Float32Array;
  origin: [number, number];
  /** Hour of day 0-23 (portal local clock). */
  hour: Uint8Array;
  /** Day of week, 0 = Sunday. */
  weekday: Uint8Array;
  /** Index into `categoryNames`. */
  category: Uint8Array;
  categoryNames: readonly string[];
  /** 1 when the community confirmed the identification (iNaturalist research grade). */
  researchGrade: Float32Array;
  /** 1 when the taxon is introduced (non-native) in Chicago. */
  introduced: Float32Array;
  /** 1 for wildlife groups (anything except plants, fungi and other life). */
  animal: Float32Array;
  /** Dense taxon index, for richness counts. */
  species: Uint32Array;
};

/** Reads the nature dataset into scene-friendly typed arrays. */
export function readNatureColumns(
  observations: LoadedDataset,
  origin: [number, number] = observations.defaultOrigin
): NatureColumns {
  const timestamps = observations.column<Uint32Array>('timestamp');
  const category = observations.column<Uint8Array>('category');
  const categoryNames = observations.categories('category');
  const researchGradeRaw = observations.column<Uint8Array>('researchGrade');
  const introducedRaw = observations.column<Uint8Array>('introduced');
  const count = timestamps.length;
  const hour = new Uint8Array(count);
  const weekday = new Uint8Array(count);
  const researchGrade = new Float32Array(count);
  const introduced = new Float32Array(count);
  const animal = new Float32Array(count);
  const animalIndexes = new Set<number>();
  categoryNames.forEach((name, index) => {
    if (!NON_ANIMAL_CATEGORIES.has(name)) animalIndexes.add(index);
  });
  for (let index = 0; index < count; index++) {
    const seconds = timestamps[index];
    hour[index] = Math.floor((seconds % 86400) / 3600);
    // 2023-01-01 was a Sunday.
    weekday[index] = Math.floor(seconds / 86400) % 7;
    researchGrade[index] = researchGradeRaw[index] ? 1 : 0;
    introduced[index] = introducedRaw[index] ? 1 : 0;
    animal[index] = animalIndexes.has(category[index]) ? 1 : 0;
  }
  return {
    count,
    positions: observations.projectColumn('position', origin),
    origin,
    hour,
    weekday,
    category,
    categoryNames,
    researchGrade,
    introduced,
    animal,
    species: observations.column<Uint32Array>('species')
  };
}

/** Hour-of-day, weekday and category filters that every nature scene shares. */
export type NatureFilter = {
  /** `[from, to)` hours; `[0, 24]` keeps every hour. */
  hours: readonly [number, number];
  /** Keep the hours OUTSIDE the window (for example overnight). */
  invertHours: boolean;
  dayType: 'all' | 'weekdays' | 'weekends';
  /** Category index, or -1 for all categories. */
  category: number;
};

/** Writes the 0/1 inclusion mask for a filter and returns how many rows pass. */
export function fillNatureMask(
  columns: NatureColumns,
  filter: NatureFilter,
  mask: Uint32Array
): number {
  const [from, to] = filter.hours;
  let included = 0;
  for (let index = 0; index < columns.count; index++) {
    const hour = columns.hour[index];
    let inside = hour >= from && hour < to;
    if (filter.invertHours) inside = !inside;
    if (inside && filter.dayType !== 'all') {
      const weekend = columns.weekday[index] === 0 || columns.weekday[index] === 6;
      inside = filter.dayType === 'weekends' ? weekend : !weekend;
    }
    if (inside && filter.category >= 0) inside = columns.category[index] === filter.category;
    mask[index] = inside ? 1 : 0;
    included += inside ? 1 : 0;
  }
  return included;
}

/** Formats an hour window such as `18:00 to 03:00`. */
export function formatHourWindow(hours: readonly [number, number], invert: boolean): string {
  const pad = (hour: number) => `${String(hour % 24).padStart(2, '0')}:00`;
  if (hours[0] <= 0 && hours[1] >= 24) return invert ? 'no hours' : 'all hours';
  return invert ? `${pad(hours[1])} to ${pad(hours[0])}` : `${pad(hours[0])} to ${pad(hours[1])}`;
}

/** Categorical colors for the eight most useful observation groups; used by the layer palette and legend. */
export const NATURE_PALETTE: readonly (readonly [number, number, number, number])[] = [
  [78, 168, 222, 255],
  [255, 148, 72, 255],
  [189, 122, 255, 255],
  [87, 205, 140, 255],
  [255, 105, 168, 255],
  [236, 200, 60, 255],
  [120, 140, 255, 255],
  [235, 85, 85, 255]
];
