// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Number of `sint32` elements in a calendar bucket parameter view. */
export const GPU_CALENDAR_BUCKETS_PARAMETER_LENGTH = 2;

/** Number of elements of the `hourWeekdayCounts` matrix: 7 weekday rows by 24 hour columns. */
export const GPU_CALENDAR_BUCKETS_MATRIX_LENGTH = 7 * 24;

/** Sentinel written to every `uint32` output of a masked or invalid row. */
export const GPU_CALENDAR_BUCKETS_INVALID_UINT32 = 0xffffffff;

/** Sentinel written to `year` and `isoWeekYear` of a masked or invalid row (`-2^31`). */
export const GPU_CALENDAR_BUCKETS_INVALID_YEAR = -2147483648;

/** Largest UTC offset magnitude in minutes. A larger per-row offset makes the row invalid. */
export const GPU_CALENDAR_BUCKETS_MAXIMUM_OFFSET_MINUTES = 1440;

/**
 * Largest absolute day number (days since 1970-01-01) a row may reach after the offset. Rows
 * beyond it, roughly years -2.7 million to +2.7 million, are invalid.
 */
export const GPU_CALENDAR_BUCKETS_MAXIMUM_DAYS = 1_000_000_000;

/**
 * Packs per-frame calendar parameters for `GPUCalendarBuckets`.
 *
 * Layout (`sint32`): `[utcOffsetMinutes, firstDayOfWeek]`. The offset is a fixed offset applied to
 * every row that has no per-row offset column. Daylight saving time is only handled through the
 * per-row offset column, which the caller precomputes.
 *
 * @param utcOffsetMinutes Fixed offset east of UTC in minutes, an integer in `[-1440, 1440]`.
 * @param firstDayOfWeek First day of the week for the `weekday` output and the hour by weekday
 * matrix: 0 = Monday (ISO, default) to 6 = Sunday. ISO week numbers always use Monday.
 * @param target Optional destination of at least 2 elements.
 * @throws If a value is out of range or not an integer, or `target` is too short.
 */
export function getGPUCalendarBucketsParameterValues(
  utcOffsetMinutes: number,
  firstDayOfWeek: number = 0,
  target: Int32Array = new Int32Array(GPU_CALENDAR_BUCKETS_PARAMETER_LENGTH)
): Int32Array {
  if (target.length < GPU_CALENDAR_BUCKETS_PARAMETER_LENGTH) {
    throw new Error(
      `Calendar bucket parameter target must hold ${GPU_CALENDAR_BUCKETS_PARAMETER_LENGTH} elements`
    );
  }
  if (
    !Number.isInteger(utcOffsetMinutes) ||
    Math.abs(utcOffsetMinutes) > GPU_CALENDAR_BUCKETS_MAXIMUM_OFFSET_MINUTES
  ) {
    throw new Error(
      `Calendar bucket UTC offset must be an integer in [-${GPU_CALENDAR_BUCKETS_MAXIMUM_OFFSET_MINUTES}, ${GPU_CALENDAR_BUCKETS_MAXIMUM_OFFSET_MINUTES}] minutes`
    );
  }
  if (!Number.isInteger(firstDayOfWeek) || firstDayOfWeek < 0 || firstDayOfWeek > 6) {
    throw new Error('Calendar bucket first day of week must be an integer in [0, 6]');
  }
  target[0] = utcOffsetMinutes;
  target[1] = firstDayOfWeek;
  return target;
}
