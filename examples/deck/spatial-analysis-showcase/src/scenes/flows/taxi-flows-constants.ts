// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Numbers the taxi-flows scene file and its compute share. This module imports nothing, so the
 * scene file (loaded by the gallery) can use it without pulling in luma.gl.
 */

/** Rows of the top-K flow list (compile-time capacity of the aggregation). */
export const TOP_FLOW_COUNT = 512;

/** Interior circles drawn (the largest same-zone flows). */
export const INTERIOR_CIRCLE_COUNT = 8;

/** Width in pixels of the heaviest flow of the dataset. */
export const MAXIMUM_FLOW_WIDTH = 10;

/** Radius in pixels of the largest interior circle of the dataset. */
export const MAXIMUM_INTERIOR_RADIUS = 26;

/**
 * Clock hour at which the taxi day starts. The quietest pickup hour of the year is 03:00, so a
 * day that runs 04:00 to 04:00 keeps a late-night window (22:00 to 02:00) in one piece.
 */
export const TAXI_DAY_START_HOUR = 4;

/** Clock label (`04:00`) of an hour counted from the start of the taxi day; wraps past 24. */
export function formatTaxiDayHour(hour: number): string {
  const wrapped = (((Math.round(hour) + TAXI_DAY_START_HOUR) % 24) + 24) % 24;
  return `${String(wrapped).padStart(2, '0')}:00`;
}
