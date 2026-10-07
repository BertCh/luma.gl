// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * The charts of the migration-timing story, built from the read-back cube: the band chart (a
 * stacked area in the same colours as the group key beside the matrix), the occupancy curve of
 * the bucket-width sweep, the fixes-per-week strip (the denominator) and the latitude profile at
 * the cursor. Pure functions that return `ChartData`; the scene decides when to call
 * `ctx.setChart`.
 */

import type {MapGround} from '../../cartography/hue-registry';
import {formatCount} from '../../cartography/live-text';
import type {PaletteColor} from '../../engine/ramps';
import type {
  BarChartData,
  ChartBand,
  DiagramChartData,
  LineChartData,
  TimelineChartData
} from '../chart-types';
import {DAYS_IN_YEAR, MIGRATION_SEASONS} from './migration-shared';
import {getGroupColors} from './migration-timing-panel';
import {BAND_DEGREES, getGroupLabel, getRowLatitude} from './migration-timing-stats';
import {FOLDED_MONTH_TICKS, formatFoldedDay, toHex} from './movement-style';

const BAND_WIDTH = 320;
const BAND_HEIGHT = 168;
const PLOT_LEFT = 30;
const PLOT_RIGHT = 312;
const PLOT_TOP = 8;
const PLOT_BOTTOM = 104;

/** Inputs of {@link buildBandChart}. */
export type BandChartInput = {
  /** Share in percent per 12-degree group (south to north), one value per bucket. */
  groups: readonly (readonly number[])[];
  /** Day of the folded year at the centre of each bucket. */
  days: readonly number[];
  /** Playhead, day of the folded year. */
  playhead: number;
  rowCount: number;
  ground: MapGround;
  /** What the shares are of ("selected fixes"), for the description. */
  subject: string;
};

const x = (day: number) => PLOT_LEFT + ((PLOT_RIGHT - PLOT_LEFT) * day) / DAYS_IN_YEAR;
const y = (percent: number) => PLOT_BOTTOM - ((PLOT_BOTTOM - PLOT_TOP) * percent) / 100;

/**
 * The share of the fixes in each 12-degree latitude group through the year, as a stacked area
 * (the groups add up to 100 %, so the picture is a hand-over from one band to the next). The
 * colours are the group key drawn beside the matrix on the map, so a colour names a latitude on
 * both. Built as a diagram because the shell's line charts take theme colours, not a class table.
 */
export function buildBandChart(input: BandChartInput): DiagramChartData {
  const colors: PaletteColor[] = getGroupColors(input.ground);
  const parts: string[] = [];
  for (const percent of [0, 50, 100]) {
    parts.push(
      `<line class="diagram-muted" x1="${PLOT_LEFT}" x2="${PLOT_RIGHT}" y1="${y(percent)}" y2="${y(percent)}" stroke-opacity="0.35" stroke-width="0.6"/>`,
      `<text class="diagram-muted" x="${PLOT_LEFT - 4}" y="${y(percent) + 3.5}" text-anchor="end">${percent}</text>`
    );
  }
  // Cumulative stack, south at the bottom. Each group is the polygon between two running totals.
  const count = input.days.length;
  const lower = new Array<number>(count).fill(0);
  input.groups.forEach((values, group) => {
    const upper = lower.map((base, index) => base + (values[index] ?? 0));
    const forward = input.days
      .map(
        (day, index) =>
          `${index === 0 ? 'M' : 'L'}${x(day).toFixed(1)} ${y(upper[index]).toFixed(1)}`
      )
      .join('');
    const backward = [...input.days]
      .map((day, index) => `L${x(day).toFixed(1)} ${y(lower[index]).toFixed(1)}`)
      .reverse()
      .join('');
    parts.push(
      `<path d="${forward}${backward}Z" fill="${toHex(colors[group % colors.length])}" stroke="#ffffff" stroke-opacity="0.55" stroke-width="0.5"/>`
    );
    upper.forEach((value, index) => {
      lower[index] = value;
    });
  });
  FOLDED_MONTH_TICKS.forEach((tick, index) => {
    const next = FOLDED_MONTH_TICKS[index + 1]?.at ?? DAYS_IN_YEAR;
    parts.push(
      `<line class="diagram-muted" x1="${x(tick.at)}" x2="${x(tick.at)}" y1="${PLOT_BOTTOM}" y2="${PLOT_BOTTOM + 3}" stroke-width="0.8"/>`,
      `<text class="diagram-muted" x="${x((tick.at + next) / 2)}" y="${PLOT_BOTTOM + 14}" text-anchor="middle">${tick.label}</text>`
    );
  });
  parts.push(
    `<line class="diagram-ink" x1="${x(input.playhead + 0.5)}" x2="${x(input.playhead + 0.5)}" y1="${PLOT_TOP - 3}" y2="${PLOT_BOTTOM}" stroke-width="1.2"/>`
  );
  // The key: one swatch per group, north first (the order of the stack, top to bottom).
  const itemWidth = 54;
  [...input.groups.keys()].reverse().forEach((group, index) => {
    const left = PLOT_LEFT + index * itemWidth;
    parts.push(
      `<rect x="${left}" y="138" width="10" height="10" fill="${toHex(colors[group % colors.length])}"/>`,
      `<text class="diagram-ink" x="${left + 14}" y="147">${getGroupLabel(group, input.rowCount)}</text>`
    );
  });
  parts.push(
    `<text class="diagram-muted" x="${PLOT_LEFT}" y="${BAND_HEIGHT - 4}">Latitude group, ° N. Share in %.</text>`
  );
  return {
    kind: 'diagram',
    width: BAND_WIDTH,
    height: BAND_HEIGHT,
    svg: parts.join(''),
    description: `Stacked areas of the share of the ${input.subject} in each 12-degree latitude group through the folded year. The groups add up to 100 percent, so a hand-over from one area to the next is the population moving through.`
  };
}

/**
 * Occupied slots as a share of all slots, for every bucket width of the sweep. A click or drag
 * writes the bucket width. `occupied[i]` belongs to `widths[i]`.
 */
export function buildOccupancyChart(
  widths: readonly number[],
  share: readonly number[]
): LineChartData {
  return {
    kind: 'line',
    series: [{label: 'Slots with at least one fix', x: widths, y: share, points: true, color: 2}],
    xLabel: 'Bucket width (days)',
    yLabel: '% of slots occupied',
    xDomain: [widths[0], widths[widths.length - 1]],
    yDomain: [0, Math.max(10, Math.ceil(Math.max(...share) / 10) * 10)],
    height: 118,
    formatX: value => `${Math.round(value)}`,
    formatY: value => `${Math.round(value)}`,
    link: {option: 'bucketDays', label: value => `${value} days`},
    description:
      'The share of the (species, latitude band, bucket) slots that hold at least one fix, for each bucket width from 3 to 14 days. Narrow buckets leave most slots empty.'
  };
}

/** The seasons of the folded year as shaded bands of a time chart. */
export function getSeasonBands(): ChartBand[] {
  return [
    {from: 0, to: MIGRATION_SEASONS.spring.days[0], label: 'Winter', tone: 'muted'},
    {
      from: MIGRATION_SEASONS.spring.days[0],
      to: MIGRATION_SEASONS.spring.days[1],
      label: 'Spring',
      tone: 0
    },
    {
      from: MIGRATION_SEASONS.breeding.days[0],
      to: MIGRATION_SEASONS.breeding.days[1],
      label: 'Breeding',
      tone: 'muted'
    },
    {
      from: MIGRATION_SEASONS.autumn.days[0],
      to: MIGRATION_SEASONS.autumn.days[1],
      label: 'Autumn',
      tone: 1
    },
    {from: MIGRATION_SEASONS.winter.days[0], to: DAYS_IN_YEAR, label: 'Winter', tone: 'muted'}
  ];
}

/**
 * Fixes per bucket across the year: the denominator of every share, as a strip in the units of
 * the matrix' columns. The last bucket is left out when it is shorter than the others.
 */
export function buildFixesChart(
  days: readonly number[],
  fixes: readonly number[],
  playhead: number,
  subject: string
): TimelineChartData {
  return {
    kind: 'timeline',
    x: days,
    y: fixes,
    mode: 'bars',
    playhead,
    xDomain: [0, DAYS_IN_YEAR],
    bands: getSeasonBands(),
    xLabel: 'Day of the folded year',
    yLabel: 'GPS fixes per bucket',
    height: 120,
    formatX: value => formatFoldedDay(Math.min(DAYS_IN_YEAR - 1, value)),
    formatY: value => formatCount(value),
    description: `Number of GPS fixes of the ${subject} in each time bucket of the folded year: the denominator behind every share. A bucket shorter than the rest, at the end of the year, is left out.`
  };
}

/**
 * The latitude profile at the cursor: the share of the fixes of the cursor's bucket in each
 * latitude band, north at the top so the bars line up with the rows of the matrix.
 */
export function buildProfileChart(shares: readonly number[], highlightRow: number): BarChartData {
  const rowCount = shares.length;
  const rows = Array.from({length: rowCount}, (_, index) => rowCount - 1 - index);
  return {
    kind: 'bars',
    values: rows.map(row => shares[row] * 100),
    labels: rows.map(row => `${getRowLatitude(row)}-${getRowLatitude(row) + BAND_DEGREES}`),
    highlight: highlightRow >= 0 ? [rowCount - 1 - highlightRow] : [],
    horizontal: true,
    height: 190,
    xLabel: "% of the bucket's fixes, by band (° N)",
    formatX: value => `${Math.round(value)}`,
    formatY: value => `${Math.round(value)}`,
    description:
      "Where the selected birds are at the cursor: the share of the cursor's time bucket in each 3-degree latitude band, north at the top."
  };
}
