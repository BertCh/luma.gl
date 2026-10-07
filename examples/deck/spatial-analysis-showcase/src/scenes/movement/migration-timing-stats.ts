// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Pure statistics of the migration-timing story: everything the CPU derives from the read-back
 * `GPUTemporalReduction` cube (counts and maxima per species, latitude band and time bucket), so
 * the charts, the crossing dates and the tooltips come from the same numbers the matrix shows.
 * No GPU and no DOM here.
 */

import {DAYS_IN_YEAR} from './migration-shared';

/** Species choices of the `species` option. */
export type SpeciesChoice = 'all' | 'marsh' | 'montagu' | 'spoonbill';

/** Bit mask of the species (manifest order: marsh harrier, Montagu's harrier, spoonbill). */
export const SPECIES_MASKS: Readonly<Record<SpeciesChoice, number>> = {
  all: 7,
  marsh: 1,
  montagu: 2,
  spoonbill: 4
};

/** Species in the dataset. */
export const SPECIES_COUNT = 3;
/** Time buckets the reduction is compiled for (366 / 128 = 2.86, so the narrowest width is 3 days). */
export const BUCKET_COUNT = 128;
/** South edge of the first latitude band, in degrees north. */
export const LATITUDE_START = 6;
/** Height of one latitude band in degrees. */
export const BAND_DEGREES = 3;
/** Cells with fewer fixes than this are hatched ("fewer than 20 fixes"). */
export const THIN_FIXES = 20;
/** Latitude bands per 12-degree group of the band chart. */
export const ROWS_PER_GROUP = 4;
/** Narrowest and widest bucket the slider offers, in days. */
export const BUCKET_DAYS_RANGE = [3, 14] as const;

/** The CPU copy of the reduction, read back once per bucket width. */
export type TimingCube = {
  /** Fixes per slot, `(species * rowCount + row) * BUCKET_COUNT + bucket`. */
  counts: Uint32Array;
  /** Maximum step speed in km/h per slot (NaN where the slot is empty). */
  fastest: Float32Array;
  /** Occupied slots reported by the contributor. */
  occupied: number;
  /** Latitude bands per species (the first band starts at {@link LATITUDE_START}). */
  rowCount: number;
  /** The bucket width, in days, this cube was reduced with. */
  bucketDays: number;
};

/** Buckets that hold the folded year at a width (the last one may be short). */
export function getValidBuckets(bucketDays: number): number {
  return Math.ceil(DAYS_IN_YEAR / bucketDays);
}

/** Bucket holding a day of the folded year (0-based). */
export function getBucketOfDay(day: number, bucketDays: number): number {
  return Math.min(getValidBuckets(bucketDays) - 1, Math.max(0, Math.floor(day / bucketDays)));
}

/** Southern edge of a latitude row, in degrees north. */
export function getRowLatitude(row: number): number {
  return LATITUDE_START + row * BAND_DEGREES;
}

/**
 * The latitude row containing a latitude. A latitude exactly on a band edge belongs to the band
 * below it (36 N is the top of the 33-36 band), so a probe at a round number reads the band the
 * line closes.
 */
export function getRowOfLatitude(latitude: number, rowCount: number): number {
  return Math.min(
    rowCount - 1,
    Math.max(0, Math.ceil((latitude - LATITUDE_START) / BAND_DEGREES - 1e-9) - 1)
  );
}

function getSlot(cube: TimingCube, species: number, row: number, bucket: number): number {
  return (species * cube.rowCount + row) * BUCKET_COUNT + bucket;
}

/** Fixes of the chosen species in one cell (row, bucket) and the fastest step ending there. */
export function getCell(
  cube: TimingCube,
  mask: number,
  row: number,
  bucket: number
): {count: number; fastest: number} {
  let count = 0;
  let fastest = Number.NaN;
  for (let species = 0; species < SPECIES_COUNT; species++) {
    if (!((mask >> species) & 1)) continue;
    const slot = getSlot(cube, species, row, bucket);
    const slotCount = cube.counts[slot];
    if (slotCount === 0) continue;
    count += slotCount;
    fastest = Number.isNaN(fastest) ? cube.fastest[slot] : Math.max(fastest, cube.fastest[slot]);
  }
  return {count, fastest};
}

/** Fixes of the chosen species per latitude row in one bucket, and their total. */
export function getColumn(
  cube: TimingCube,
  mask: number,
  bucket: number
): {perRow: Float64Array; total: number} {
  const perRow = new Float64Array(cube.rowCount);
  let total = 0;
  for (let species = 0; species < SPECIES_COUNT; species++) {
    if (!((mask >> species) & 1)) continue;
    for (let row = 0; row < cube.rowCount; row++) {
      const count = cube.counts[getSlot(cube, species, row, bucket)];
      perRow[row] += count;
      total += count;
    }
  }
  return {perRow, total};
}

/** Fixes of the chosen species in each valid bucket (the denominator of every share). */
export function getFixesPerBucket(cube: TimingCube, mask: number): Float64Array {
  const buckets = getValidBuckets(cube.bucketDays);
  const result = new Float64Array(buckets);
  for (let bucket = 0; bucket < buckets; bucket++)
    result[bucket] = getColumn(cube, mask, bucket).total;
  return result;
}

/**
 * Median latitude of the chosen species' fixes in each valid bucket, in degrees north (NaN for a
 * bucket without fixes). The band counts are cumulated and interpolated linearly inside the band
 * that holds the middle fix, so the result is good to a fraction of a band.
 */
export function getMedianLatitudes(cube: TimingCube, mask: number): Float64Array {
  const buckets = getValidBuckets(cube.bucketDays);
  const result = new Float64Array(buckets).fill(Number.NaN);
  for (let bucket = 0; bucket < buckets; bucket++) {
    const {perRow, total} = getColumn(cube, mask, bucket);
    if (total === 0) continue;
    const half = total / 2;
    let before = 0;
    for (let row = 0; row < perRow.length; row++) {
      if (before + perRow[row] >= half && perRow[row] > 0) {
        result[bucket] = getRowLatitude(row) + (BAND_DEGREES * (half - before)) / perRow[row];
        break;
      }
      before += perRow[row];
    }
  }
  return result;
}

/** Day of the folded year at which a series of medians crosses a latitude, or `null`. */
export type Crossings = {northbound: number | null; southbound: number | null};

/**
 * When the median latitude crosses `latitude`: the first time it rises through it (northbound)
 * and the last time it falls through it (southbound), interpolated between bucket centres. A
 * southbound crossing that is not after the northbound one is dropped, and a bucket without fixes
 * breaks the line (no crossing is invented across a gap).
 */
export function findCrossings(
  medians: ArrayLike<number>,
  bucketDays: number,
  latitude: number
): Crossings {
  const centre = (bucket: number) => (bucket + 0.5) * bucketDays;
  let northbound: number | null = null;
  for (let bucket = 1; bucket < medians.length; bucket++) {
    const before = medians[bucket - 1];
    const after = medians[bucket];
    if (
      Number.isFinite(before) &&
      Number.isFinite(after) &&
      before < latitude &&
      after >= latitude
    ) {
      northbound = centre(bucket - 1) + ((latitude - before) / (after - before)) * bucketDays;
      break;
    }
  }
  let southbound: number | null = null;
  for (let bucket = medians.length - 1; bucket >= 1; bucket--) {
    const before = medians[bucket - 1];
    const after = medians[bucket];
    if (
      Number.isFinite(before) &&
      Number.isFinite(after) &&
      before >= latitude &&
      after < latitude
    ) {
      southbound = centre(bucket - 1) + ((before - latitude) / (before - after)) * bucketDays;
      break;
    }
  }
  if (northbound !== null && southbound !== null && southbound <= northbound) southbound = null;
  return {northbound, southbound};
}

/**
 * Share of the chosen species' fixes in each 12-degree latitude group (south to north) per valid
 * bucket, in percent. A bucket without fixes reads 0 in every group.
 */
export function getGroupShares(cube: TimingCube, mask: number): number[][] {
  const groupCount = Math.ceil(cube.rowCount / ROWS_PER_GROUP);
  const buckets = getValidBuckets(cube.bucketDays);
  const groups: number[][] = Array.from({length: groupCount}, () => new Array(buckets).fill(0));
  for (let bucket = 0; bucket < buckets; bucket++) {
    const {perRow, total} = getColumn(cube, mask, bucket);
    if (total === 0) continue;
    for (let row = 0; row < perRow.length; row++) {
      groups[Math.floor(row / ROWS_PER_GROUP)][bucket] += (perRow[row] / total) * 100;
    }
  }
  return groups;
}

/** The label of a 12-degree group, `18-30` (degrees north; the last group stops at the data). */
export function getGroupLabel(group: number, rowCount: number): string {
  const south = LATITUDE_START + group * ROWS_PER_GROUP * BAND_DEGREES;
  const north = Math.min(rowCount, (group + 1) * ROWS_PER_GROUP) * BAND_DEGREES + LATITUDE_START;
  return `${south}-${north}`;
}

/** Counts of the occupied cells of the chosen species: all of them and the thin ones. */
export function getOccupiedCells(
  cube: TimingCube,
  mask: number
): {values: {count: number; fastest: number; share: number}[]; thin: number} {
  const buckets = getValidBuckets(cube.bucketDays);
  const values: {count: number; fastest: number; share: number}[] = [];
  let thin = 0;
  for (let bucket = 0; bucket < buckets; bucket++) {
    const {perRow, total} = getColumn(cube, mask, bucket);
    for (let row = 0; row < cube.rowCount; row++) {
      if (perRow[row] === 0) continue;
      const cell = getCell(cube, mask, row, bucket);
      values.push({...cell, share: cell.count / total});
      if (cell.count < THIN_FIXES) thin++;
    }
  }
  return {values, thin};
}

/**
 * Six class breaks that cut the occupied cells' fix counts into seven equal-sized groups
 * (quantiles), rounded to whole fixes and de-duplicated. Fewer than six breaks come back when
 * many cells share a count.
 */
export function getCountBreaks(counts: readonly number[]): number[] {
  if (counts.length === 0) return [];
  const sorted = [...counts].sort((a, b) => a - b);
  const breaks: number[] = [];
  for (let index = 1; index < 7; index++) {
    const value = Math.round(
      sorted[Math.min(sorted.length - 1, Math.floor((sorted.length * index) / 7))]
    );
    if (value > 0 && (breaks.length === 0 || value > breaks[breaks.length - 1])) breaks.push(value);
  }
  return breaks;
}

/** The cell with the fastest step (the step ends in it), or `null` without fixes. */
export function getFastestCell(
  cube: TimingCube,
  mask: number
): {row: number; bucket: number; speed: number; count: number} | null {
  const buckets = getValidBuckets(cube.bucketDays);
  let best: {row: number; bucket: number; speed: number; count: number} | null = null;
  for (let bucket = 0; bucket < buckets; bucket++) {
    for (let row = 0; row < cube.rowCount; row++) {
      const cell = getCell(cube, mask, row, bucket);
      if (cell.count > 0 && (!best || cell.fastest > best.speed)) {
        best = {row, bucket, speed: cell.fastest, count: cell.count};
      }
    }
  }
  return best;
}

/** Class index of `value` in ascending `breaks`: the number of breaks at or below it. */
export function getClassOf(value: number, breaks: readonly number[]): number {
  let index = 0;
  while (index < breaks.length && value >= breaks[index]) index++;
  return index;
}
