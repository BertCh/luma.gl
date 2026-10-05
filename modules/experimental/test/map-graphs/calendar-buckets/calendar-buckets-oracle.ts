// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPU_CALENDAR_BUCKETS_INVALID_UINT32,
  GPU_CALENDAR_BUCKETS_INVALID_YEAR,
  GPU_CALENDAR_BUCKETS_MATRIX_LENGTH,
  GPU_CALENDAR_BUCKETS_MAXIMUM_DAYS,
  GPU_CALENDAR_BUCKETS_MAXIMUM_OFFSET_MINUTES
} from '../../../src/map-graphs/calendar-buckets';

/** Calendar fields of one row. Invalid rows hold the documented sentinels and `valid: false`. */
export type CalendarFields = {
  valid: boolean;
  year: number;
  month: number;
  day: number;
  hour: number;
  minute: number;
  weekday: number;
  dayOfYear: number;
  isoWeek: number;
  isoWeekYear: number;
  quarter: number;
};

const INT64_MINIMUM = -(2n ** 63n);
const INT64_MAXIMUM = 2n ** 63n - 1n;
const MILLISECONDS_PER_DAY = 86_400_000n;

/** Fields of an invalid row. */
export function getInvalidCalendarFields(): CalendarFields {
  const invalid = GPU_CALENDAR_BUCKETS_INVALID_UINT32;
  const year = GPU_CALENDAR_BUCKETS_INVALID_YEAR;
  return {
    valid: false,
    year,
    month: invalid,
    day: invalid,
    hour: invalid,
    minute: invalid,
    weekday: invalid,
    dayOfYear: invalid,
    isoWeek: invalid,
    isoWeekYear: year,
    quarter: invalid
  };
}

function floorDivide(numerator: bigint, denominator: bigint): bigint {
  const quotient = numerator / denominator;
  return numerator % denominator !== 0n && numerator < 0n !== denominator < 0n
    ? quotient - 1n
    : quotient;
}

/** Days since 1970-01-01 of a proleptic Gregorian date (Hinnant `days_from_civil`) in BigInt. */
function getDaysFromCivil(year: bigint, month: bigint, day: bigint): bigint {
  const shiftedYear = month <= 2n ? year - 1n : year;
  const era = floorDivide(shiftedYear, 400n);
  const yearOfEra = shiftedYear - era * 400n;
  const monthPrime = month > 2n ? month - 3n : month + 9n;
  const dayOfYear = (153n * monthPrime + 2n) / 5n + day - 1n;
  const dayOfEra = yearOfEra * 365n + yearOfEra / 4n - yearOfEra / 100n + dayOfYear;
  return era * 146097n + dayOfEra - 719468n;
}

/** Year, month and day of a day number (Hinnant `civil_from_days`) in BigInt. */
function getCivilFromDays(days: bigint): [bigint, bigint, bigint] {
  const z = days + 719468n;
  const era = floorDivide(z, 146097n);
  const dayOfEra = z - era * 146097n;
  const yearOfEra = (dayOfEra - dayOfEra / 1460n + dayOfEra / 36524n - dayOfEra / 146096n) / 365n;
  const dayOfMarchYear = dayOfEra - (365n * yearOfEra + yearOfEra / 4n - yearOfEra / 100n);
  const monthPrime = (5n * dayOfMarchYear + 2n) / 153n;
  const day = dayOfMarchYear - (153n * monthPrime + 2n) / 5n + 1n;
  const month = monthPrime < 10n ? monthPrime + 3n : monthPrime - 9n;
  return [yearOfEra + era * 400n + (month <= 2n ? 1n : 0n), month, day];
}

/**
 * CPU oracle in exact BigInt arithmetic.
 *
 * @param milliseconds Signed 64-bit epoch milliseconds.
 * @param offsetMinutes UTC offset in minutes.
 * @param firstDayOfWeek 0 = Monday to 6 = Sunday, for `weekday`.
 */
export function decodeCalendarOnCPU(
  milliseconds: bigint,
  offsetMinutes: number,
  firstDayOfWeek: number = 0
): CalendarFields {
  if (Math.abs(offsetMinutes) > GPU_CALENDAR_BUCKETS_MAXIMUM_OFFSET_MINUTES) {
    return getInvalidCalendarFields();
  }
  const local = milliseconds + BigInt(offsetMinutes) * 60_000n;
  if (local < INT64_MINIMUM || local > INT64_MAXIMUM) {
    return getInvalidCalendarFields();
  }
  const days = floorDivide(local, MILLISECONDS_PER_DAY);
  if (
    days > BigInt(GPU_CALENDAR_BUCKETS_MAXIMUM_DAYS) ||
    days < -BigInt(GPU_CALENDAR_BUCKETS_MAXIMUM_DAYS)
  ) {
    return getInvalidCalendarFields();
  }
  const millisecondsOfDay = Number(local - days * MILLISECONDS_PER_DAY);
  const [year, month, day] = getCivilFromDays(days);
  const isoWeekday = Number(floorDivide(days + 3n, 1n) - floorDivide(days + 3n, 7n) * 7n);
  const [isoWeekYear, thursdayMonth, thursdayDay] = getCivilFromDays(
    days - BigInt(isoWeekday) + 3n
  );
  const thursdayDayOfYear =
    getDaysFromCivil(isoWeekYear, thursdayMonth, thursdayDay) -
    getDaysFromCivil(isoWeekYear, 1n, 1n) +
    1n;
  return {
    valid: true,
    year: Number(year),
    month: Number(month),
    day: Number(day),
    hour: Math.floor(millisecondsOfDay / 3_600_000),
    minute: Math.floor((millisecondsOfDay % 3_600_000) / 60_000),
    weekday: (isoWeekday - firstDayOfWeek + 7) % 7,
    dayOfYear: Number(days - getDaysFromCivil(year, 1n, 1n) + 1n),
    isoWeek: Number((thursdayDayOfYear - 1n) / 7n + 1n),
    isoWeekYear: Number(isoWeekYear),
    quarter: Math.floor((Number(month) - 1) / 3) + 1
  };
}

/** Builds a UTC `Date` for any year, including those below 100. */
function createUtcDate(year: number, month: number, day: number): Date {
  const date = new Date(0);
  date.setUTCFullYear(year, month, day);
  return date;
}

/**
 * Independent reference using JS `Date` UTC getters. Only valid while `milliseconds + offset`
 * is within the `Date` range of +-8.64e15 ms.
 */
export function decodeCalendarWithDate(
  milliseconds: number,
  offsetMinutes: number,
  firstDayOfWeek: number = 0
): CalendarFields {
  const date = new Date(milliseconds + offsetMinutes * 60_000);
  const year = date.getUTCFullYear();
  const month = date.getUTCMonth();
  const isoWeekday = (date.getUTCDay() + 6) % 7;
  const dayOfYear =
    Math.round(
      (createUtcDate(year, month, date.getUTCDate()).getTime() -
        createUtcDate(year, 0, 1).getTime()) /
        Number(MILLISECONDS_PER_DAY)
    ) + 1;
  const thursday = new Date(date.getTime() + (3 - isoWeekday) * Number(MILLISECONDS_PER_DAY));
  const isoWeekYear = thursday.getUTCFullYear();
  const thursdayOffset =
    createUtcDate(isoWeekYear, thursday.getUTCMonth(), thursday.getUTCDate()).getTime() -
    createUtcDate(isoWeekYear, 0, 1).getTime();
  return {
    valid: true,
    year,
    month: month + 1,
    day: date.getUTCDate(),
    hour: date.getUTCHours(),
    minute: date.getUTCMinutes(),
    weekday: (isoWeekday - firstDayOfWeek + 7) % 7,
    dayOfYear,
    isoWeek: Math.floor(Math.round(thursdayOffset / Number(MILLISECONDS_PER_DAY)) / 7) + 1,
    isoWeekYear,
    quarter: Math.floor(month / 3) + 1
  };
}

/** Counts valid rows into a `weekday * 24 + hour` matrix. */
export function countHourWeekdayOnCPU(rows: readonly CalendarFields[]): Uint32Array {
  const counts = new Uint32Array(GPU_CALENDAR_BUCKETS_MATRIX_LENGTH);
  for (const row of rows) {
    if (row.valid) {
      counts[row.weekday * 24 + row.hour]++;
    }
  }
  return counts;
}

/** Deterministic xorshift in [0, 2^32). */
export function createRandomWords(seed: number): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return state;
  };
}

/**
 * Random epoch milliseconds over many magnitudes, from sub-second to the full Int64 range, with
 * both signs. About a third are within the `Date` range.
 */
export function createRandomTimestamps(seed: number, count: number): BigInt64Array {
  const next = createRandomWords(seed);
  const timestamps = new BigInt64Array(count);
  for (let row = 0; row < count; row++) {
    const bits = 10 + (next() % 54);
    const word = (BigInt(next()) << 32n) | BigInt(next());
    const magnitude = word & ((1n << BigInt(bits)) - 1n);
    timestamps[row] = next() % 2 === 0 ? magnitude : -magnitude;
  }
  return timestamps;
}
