// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getClassIndexOf, getClassPalette, makeClassTable} from '../../cartography/class-table';
import {formatCount, formatSigned} from '../../cartography/live-text';
import type {ClassTable} from '../../cartography/types';
import type {MapGround} from '../../cartography/hue-registry';
import type {PaletteColor} from '../../engine/ramps';
import type {TooltipContent} from '../scene';
import {formatClockHour, getNetFlowPalette, getSymmetricBreaks} from './flows-style';

/**
 * Symbolisation of the nyc-taxi-tides scene that is not a shared table: the three class tables
 * (net trips, imbalance share, size of the net), the break ladders that stay fixed across every
 * hour and zone size, the window labels and the structured tooltip. Pure data and strings: no
 * luma.gl, so the scene file can import it for its legends.
 *
 * The net is `arrivals - departures` per hexagon. Orange is more leave than arrive, purple more
 * arrive than leave (`getNetFlowPalette`), with a real neutral class in the middle.
 */

/** Positive steps of the net-trips classes: 5, 15, 40, 100 trips per window. */
export const NET_STEPS = [5, 15, 40, 100] as const;

/** Positive steps of the imbalance-share classes: 5, 15, 30, 50 percent of the traffic. */
export const SHARE_STEPS = [0.05, 0.15, 0.3, 0.5] as const;

/** Largest net that still counts as "about balanced". */
export const BALANCED_TRIPS = NET_STEPS[0];

/**
 * Value a hexagon below the minimum volume takes in the share buffer: under the first (masked)
 * class break of {@link getShareTable}, so one class table draws it hatched.
 */
export const MASKED_SHARE = -9;

/** Break between the masked class and the first share class. */
const MASK_BREAK = -2;

/**
 * Symmetric break ladder with the positive breaks nudged up by a hair, so a net of exactly +5
 * falls in the middle class like -5 does (classes are `value >= break`, and nets are integers).
 */
function getTiedBreaks(steps: readonly number[], hair: number): number[] {
  return getSymmetricBreaks(steps).map(value => (value > 0 ? value + hair : value));
}

/** Net-trips breaks, the same for every window and every hexagon size. */
export const NET_BREAKS: readonly number[] = getTiedBreaks(NET_STEPS, 0.001);

/** Imbalance-share breaks. */
export const SHARE_BREAKS: readonly number[] = getTiedBreaks(SHARE_STEPS, 0.00001);

/** Class breaks of the size-of-net ("wrong") map: |net| in 0-5, 6-15, 16-40, 41-100, over 100. */
export const MAGNITUDE_BREAKS: readonly number[] = NET_STEPS.map(step => step + 0.001);

const MINUS = '−';

/** Labels of the nine net-trips classes, orange arm first. */
const NET_LABELS = (() => {
  const [a, b, c, d] = NET_STEPS;
  return [
    `Beyond ${MINUS}${d}`,
    `${MINUS}${d} to ${MINUS}${c + 1}`,
    `${MINUS}${c} to ${MINUS}${b + 1}`,
    `${MINUS}${b} to ${MINUS}${a + 1}`,
    `About balanced (${MINUS}${a} to +${a})`,
    `+${a + 1} to +${b}`,
    `+${b + 1} to +${c}`,
    `+${c + 1} to +${d}`,
    `Beyond +${d}`
  ];
})();

const percent = (share: number): string => `${Math.round(share * 100)}%`;

/** Labels of the nine share classes. */
const SHARE_LABELS = (() => {
  const [a, b, c, d] = SHARE_STEPS;
  return [
    `${MINUS}${percent(d)} or more`,
    `${MINUS}${percent(c)} to ${MINUS}${percent(d)}`,
    `${MINUS}${percent(b)} to ${MINUS}${percent(c)}`,
    `${MINUS}${percent(a)} to ${MINUS}${percent(b)}`,
    `About balanced (within ${percent(a)})`,
    `+${percent(a)} to +${percent(b)}`,
    `+${percent(b)} to +${percent(c)}`,
    `+${percent(c)} to +${percent(d)}`,
    `+${percent(d)} or more`
  ];
})();

/**
 * The net-trips class table of a ground: nine classes of arrivals minus departures, symmetric
 * about zero, orange where more leave and purple where more arrive. Hexagons without any trip are
 * no data (transparent), which is not the same as balanced.
 */
export function getNetTable(ground: MapGround): ClassTable {
  return makeClassTable({
    breaks: NET_BREAKS,
    colors: getNetFlowPalette(9, ground),
    labels: NET_LABELS,
    unit: 'trips',
    method: 'Fixed symmetric classes, the same for every hour and hexagon size',
    noData: {label: 'No trips in the window', color: [0, 0, 0, 0]}
  });
}

/**
 * The imbalance-share class table: `(arrivals - departures) / (arrivals + departures)` in the
 * same nine PuOr classes, plus a first hatched class for hexagons under `minVolume` trips.
 */
export function getShareTable(ground: MapGround, minVolume: number): ClassTable {
  const colors = getNetFlowPalette(9, ground);
  return makeClassTable({
    breaks: [MASK_BREAK, ...SHARE_BREAKS],
    colors: [[0, 0, 0, 0], ...colors],
    labels: [`Fewer than ${formatCount(minVolume)} trips (hidden)`, ...SHARE_LABELS],
    unit: 'share of trips',
    hatched: [0],
    method: 'Fixed symmetric classes; small hexagons are hatched, not coloured',
    noData: {label: 'No trips in the window', color: [0, 0, 0, 0]}
  });
}

/**
 * The deliberately wrong map: the size of the net on one sequential ramp. It shows where
 * something happens and loses the direction.
 */
export function getMagnitudeTable(ground: MapGround): ClassTable {
  const colors = getClassPalette('Greys', 5, {ground}).map(
    (color, index) => [color[0], color[1], color[2], index === 0 ? 120 : 255] as PaletteColor
  );
  return makeClassTable({
    breaks: MAGNITUDE_BREAKS,
    colors,
    labels: [
      `Up to ${NET_STEPS[0]}`,
      `${NET_STEPS[0] + 1} to ${NET_STEPS[1]}`,
      `${NET_STEPS[1] + 1} to ${NET_STEPS[2]}`,
      `${NET_STEPS[2] + 1} to ${NET_STEPS[3]}`,
      `Over ${NET_STEPS[3]}`
    ],
    unit: 'trips',
    method: 'Sequential on |net|: direction is lost',
    noData: {label: 'No trips in the window', color: [0, 0, 0, 0]}
  });
}

/** Stripe colour of the hatched (small-number) class on a ground. */
export function getHatchColor(ground: MapGround): PaletteColor {
  return ground === 'dark' ? [214, 220, 230, 120] : [31, 41, 51, 110];
}

/** Hairline between hexagons: white on paper, the ground ink on dark. */
export function getZoneOutlineColor(ground: MapGround): PaletteColor {
  return ground === 'dark' ? [14, 17, 22, 128] : [255, 255, 255, 153];
}

const WEEKDAYS = ['Thu', 'Fri', 'Sat', 'Sun'];

/** `Fri 08:00-09:00` from hours since Thu 1 Jan 00:00 local. */
export function formatShortWindow(start: number, end: number): string {
  const weekday = WEEKDAYS[Math.min(WEEKDAYS.length - 1, Math.floor(start / 24))];
  return `${weekday} ${formatClockHour(start)}-${formatClockHour(end)}`;
}

/** `Fri 2 Jan 2015, 08:00-09:00` (or both days when the window crosses midnight). */
export function formatLongWindow(start: number, end: number): string {
  const startDay = Math.floor(start / 24);
  const endDay = Math.floor((end - 1e-6) / 24);
  const name = (day: number) => `${WEEKDAYS[Math.min(WEEKDAYS.length - 1, day)]} ${1 + day} Jan`;
  if (startDay === endDay) {
    return `${name(startDay)} 2015, ${formatClockHour(start)}-${formatClockHour(end)}`;
  }
  return `${name(startDay)} ${formatClockHour(start)} to ${name(endDay)} ${formatClockHour(end)}, 2015`;
}

/** A signed share as `+27%` or `−85%`. */
export function formatShare(share: number): string {
  const rounded = Math.round(share * 100);
  return rounded === 0 ? '0%' : `${rounded > 0 ? '+' : MINUS}${Math.abs(rounded)}%`;
}

/** Input of {@link getZoneTooltip}. */
export type ZoneTooltipInput = {
  /** `near Midtown` style phrase for the zone, or `null`. */
  place: string | null;
  zoneNoun: 'Hexagon' | 'Cell';
  arrivals: number;
  departures: number;
  mode: 'net' | 'share';
  netTable: ClassTable;
  shareTable: ClassTable;
  minVolume: number;
  windowLabel: string;
};

/** The structured tooltip of one zone: the mapped value with its class swatch first. */
export function getZoneTooltip(input: ZoneTooltipInput): TooltipContent {
  const {arrivals, departures, mode, netTable, shareTable, minVolume} = input;
  const net = arrivals - departures;
  const volume = arrivals + departures;
  const share = volume > 0 ? net / volume : 0;
  const masked = volume < minVolume;
  const netRow = {
    label: 'Net',
    value: formatSigned(net),
    unit: 'trips',
    swatch: netTable.colors[getClassIndexOf(netTable, net)]
  };
  const shareRow = {
    label: 'Imbalance share',
    value: volume > 0 ? formatShare(share) : '–',
    swatch: shareTable.colors[masked ? 0 : getClassIndexOf(shareTable, share)]
  };
  const rows =
    mode === 'net'
      ? [
          {...netRow, emphasis: true},
          {...shareRow, swatch: undefined},
          {label: 'Arrivals', value: formatCount(arrivals), unit: 'trips'},
          {label: 'Departures', value: formatCount(departures), unit: 'trips'}
        ]
      : [
          {...shareRow, emphasis: true, swatch: masked ? undefined : shareRow.swatch},
          {...netRow, swatch: undefined},
          {label: 'Arrivals', value: formatCount(arrivals), unit: 'trips'},
          {label: 'Departures', value: formatCount(departures), unit: 'trips'}
        ];
  return {
    title: `${input.zoneNoun} ${input.place ?? 'in the city'}`,
    subtitle: input.windowLabel,
    rows,
    note:
      mode === 'share' && masked
        ? `Hidden: fewer than ${formatCount(minVolume)} trips, too few to call a share`
        : undefined
  };
}
