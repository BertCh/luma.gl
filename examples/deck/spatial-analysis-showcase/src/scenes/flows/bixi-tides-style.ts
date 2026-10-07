// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {makeClassTable} from '../../cartography/class-table';
import type {MapGround} from '../../cartography/hue-registry';
import {formatSigned} from '../../cartography/live-text';
import type {ClassTable} from '../../cartography/types';
import {getNetFlowPalette, getSymmetricBreaks} from './flows-style';

/**
 * Symbolisation of the bixi-tides scene that is not a shared table: the fixed class ladder of net
 * bikes per station, the two things the "failing" states change (the unit and the scale mode), the
 * disc sizes and the class table of one ground. Pure data and strings: no luma.gl, so the scene
 * file can import it for its legends.
 *
 * Net is `arrivals - departures` per station: orange means more bikes leave than arrive (the dock
 * drains), purple more arrive than leave (the dock fills), with a real neutral class around zero.
 */

/** Unit of the shown net: a rate per average day-type hour, or the raw August total. */
export type TidesUnits = 'rate' | 'total';

/** `fixed`: one scale for every hour of the day. `perHour`: each hour rescales to its own maximum. */
export type TidesScaleMode = 'fixed' | 'perHour';

/** Positive steps of the fixed ladder, in rides per average weekday hour: 1, 3, 6. */
export const TIDES_NET_STEPS = [1, 3, 6] as const;

/** A station this far out of balance (rides per average hour) draws the largest disc. */
export const TIDES_SIZE_REFERENCE = 40;

/** Disc radius range in CSS pixels: `3 + k sqrt(|net|)` up to 16. */
export const TIDES_DISC_MIN_PIXELS = 3;
export const TIDES_DISC_MAX_PIXELS = 16;

/** Alpha (0-255) of the discs, and of the balanced class so quiet stations stay as pale dots. */
export const TIDES_DISC_ALPHA = 235;
export const TIDES_BALANCED_ALPHA = 102;

/** Share of the hour's own maximum at the three break steps of the per-hour (failing) scale. */
const PER_HOUR_SHARES = [1 / 12, 1 / 4, 1 / 2] as const;

const MINUS = '−';

/** What the two scale choices of one view are. */
export type TidesScale = {
  /** Interior class breaks, ascending, in the shown unit. */
  breaks: number[];
  /** Value drawn with the full disc radius. */
  sizeMaximum: number;
  /** Legend unit, for example `rides per average weekday hour`. */
  unit: string;
  /** One line under the legend: which scale this is. */
  note: string;
};

/** Inputs of {@link getTidesScale}. */
export type TidesScaleInput = {
  units: TidesUnits;
  scaleMode: TidesScaleMode;
  dayType: 'weekday' | 'weekend';
  /** Largest absolute net of the hour in the shown unit (used by the per-hour scale only). */
  hourMaximum: number;
  weekdayCount: number;
};

/** `1`, `1.5`, `12`: no trailing zero below 10, whole numbers from 10. */
export function formatNetNumber(value: number): string {
  const magnitude = Math.abs(value);
  const text =
    magnitude >= 10 ? String(Math.round(magnitude)) : String(Number(magnitude.toFixed(1)));
  return value < 0 ? `${MINUS}${text}` : text;
}

/**
 * Class breaks, disc size reference, unit and note of one view. The fixed scale is the same for
 * every hour and both day types; in totals units the same ladder is multiplied by the number of
 * weekdays, so a weekend (nine days instead of 22) looks weaker for no real reason. The per-hour
 * scale (the failing state) stretches the ladder to the hour's own maximum.
 */
export function getTidesScale(input: TidesScaleInput): TidesScale {
  const {units, scaleMode, dayType, hourMaximum, weekdayCount} = input;
  const factor = units === 'total' ? weekdayCount : 1;
  const dayWord = dayType === 'weekday' ? 'weekday' : 'weekend day';
  const unit = units === 'total' ? 'rides in August' : `rides per average ${dayWord} hour`;
  if (scaleMode === 'perHour') {
    const maximum = Math.max(hourMaximum, 1e-6);
    return {
      breaks: getSymmetricBreaks(PER_HOUR_SHARES.map(share => share * maximum)),
      sizeMaximum: maximum,
      unit,
      note: 'Scale per hour: the biggest station of every hour looks equally big. Compare two hours and you compare nothing.'
    };
  }
  return {
    breaks: getSymmetricBreaks(TIDES_NET_STEPS.map(step => step * factor)),
    sizeMaximum: TIDES_SIZE_REFERENCE * factor,
    unit,
    note:
      units === 'total'
        ? 'One scale for every view, in August totals: weekdays have 22 days, weekends nine.'
        : 'Scale fixed for the whole day and both day types.'
  };
}

/** Labels of the seven net classes for a break ladder, orange arm first. */
export function getNetLabels(breaks: readonly number[]): string[] {
  const [a, b, c] = breaks.slice(3).map(formatNetNumber);
  return [
    `Below ${MINUS}${c}`,
    `${MINUS}${c} to ${MINUS}${b}`,
    `${MINUS}${b} to ${MINUS}${a}`,
    `About balanced (${MINUS}${a} to ${a})`,
    `${a} to ${b}`,
    `${b} to ${c}`,
    `Above ${c}`
  ];
}

/**
 * The seven-class net table of a ground: symmetric about zero, orange where more leave, purple
 * where more arrive. The balanced class is drawn pale (alpha {@link TIDES_BALANCED_ALPHA}), so a
 * quiet station stays on the map as a small pale disc without competing with the movers.
 */
export function getNetTable(scale: TidesScale, ground: MapGround): ClassTable {
  const colors = getNetFlowPalette(7, ground).map(
    (color, index) =>
      [color[0], color[1], color[2], index === 3 ? TIDES_BALANCED_ALPHA : TIDES_DISC_ALPHA] as [
        number,
        number,
        number,
        number
      ]
  );
  return makeClassTable({
    breaks: scale.breaks,
    colors,
    labels: getNetLabels(scale.breaks),
    unit: scale.unit,
    extent: [-scale.sizeMaximum, scale.sizeMaximum],
    method: scale.note,
    noData: {label: 'No station', color: [0, 0, 0, 0]}
  });
}

/** `+3.2` or `−3.2` with a true minus and one digit. */
export function formatNetRate(value: number): string {
  return formatSigned(value, 1);
}
