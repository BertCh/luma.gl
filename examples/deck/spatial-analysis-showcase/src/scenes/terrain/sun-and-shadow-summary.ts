// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * What sun-and-shadow reads back from the map and how it says it: the structured tooltip of a
 * hovered cell, the histogram of a displayed product with the class breaks of the map, and the
 * share of cells where the two shadow methods disagree.
 */

import {getClassIndexOf, getClassLabel} from '../../cartography/class-table';
import type {ClassTable} from '../../cartography/types';
import type {BarChartData, TooltipContent, TooltipRow} from '../scene';
import {getCompassName} from './b14b-terrain';
import type {DemProbe} from './cpu-dem';
import {
  DIFFERENCE_THRESHOLD,
  formatDay,
  getShadowColors,
  SUN_GOLD,
  type SunOptions,
  type SunTables
} from './sun-and-shadow.style';
import type {ValueStats} from './b14a-session';

/** A cell under the pointer: the displayed raster's value and the elevation. */
export type HoverCell = {value: number; elevation: number; column: number; row: number};

/** What the tooltip needs besides the cell. */
export type SunTooltipInput = {
  state: SunOptions;
  tables: SunTables;
  cell: HoverCell;
  probe: DemProbe;
  /** Sun now at the village: azimuth and altitude in degrees. */
  sun: {azimuth: number; altitude: number};
  onGlacier: boolean;
};

const formatHours = (hours: number) => `${hours.toFixed(1)}`;

/**
 * The structured tooltip: the mapped value first with its class swatch, then elevation, slope and
 * aspect from the 3 x 3 window, and where useful the sun now.
 */
export function getSunTooltip(input: SunTooltipInput): TooltipContent | null {
  const {state, tables, cell, probe, sun, onGlacier} = input;
  const {value, elevation} = cell;
  const rows: TooltipRow[] = [];
  let title = 'Relief';
  const colors = getShadowColors(tables.ground);
  if (state.product === 'shadow') {
    if (!Number.isFinite(value)) return null;
    if (state.method === 'difference') {
      title = 'Two shadow methods';
      const index = getClassIndexOf(tables.difference, value);
      rows.push({
        label: 'Difference',
        value: (value * 100).toFixed(1),
        unit: '% of full sun',
        swatch: tables.difference.colors[Math.max(0, index)],
        emphasis: true
      });
      rows.push({label: 'Class', value: getClassLabel(tables.difference, index)});
    } else {
      title = state.method === 'horizon' ? 'Sun, horizon map' : 'Sun, exact cast';
      const lit = value >= 0.99;
      const dark = value <= 0.01;
      rows.push({
        label: 'Direct sun',
        value: lit
          ? 'In sun'
          : dark
            ? 'In shadow'
            : `Penumbra, ${(value * 100).toFixed(0)}% of the disc visible`,
        swatch: lit ? SUN_GOLD : dark ? colors.shadow : colors.penumbra,
        emphasis: true
      });
    }
  } else if (state.product === 'light') {
    title = state.light === 'sun' ? 'Real sun' : 'Relief';
    if (state.light === 'sun' && Number.isFinite(value)) {
      rows.push({
        label: 'Illumination',
        value: value.toFixed(2),
        unit: 'of full sun on a flat surface',
        emphasis: true
      });
    }
  } else if (state.product === 'sun-hours') {
    if (!Number.isFinite(value)) return null;
    title = 'Direct sun';
    const index = getClassIndexOf(tables.hours, value);
    rows.push({
      label: formatDay(state.dayOfYear),
      value: formatHours(value),
      unit: 'h of direct sun',
      swatch: tables.hours.colors[Math.max(0, index)],
      emphasis: true
    });
    rows.push({label: 'Class', value: getClassLabel(tables.hours, index)});
  } else {
    if (!Number.isFinite(value)) return null;
    title = 'Clear-sky insolation';
    const kilowattHours = value / 1000;
    const index = getClassIndexOf(tables.insolation, kilowattHours);
    rows.push({
      label: formatDay(state.dayOfYear),
      value: kilowattHours.toFixed(2),
      unit: 'kWh/m² per day',
      swatch: tables.insolation.colors[Math.max(0, index)],
      emphasis: true
    });
    rows.push({label: 'Class', value: getClassLabel(tables.insolation, index)});
  }
  if (Number.isFinite(elevation)) {
    rows.push({
      label: 'Elevation',
      value: Math.round(elevation).toLocaleString('en-US'),
      unit: 'm'
    });
  }
  const horn = probe.hornAt(cell.column, cell.row);
  rows.push({label: 'Slope', value: horn.slopeDeg.toFixed(0), unit: '°'});
  if (Number.isFinite(horn.aspectDeg) && horn.slopeDeg >= 5) {
    rows.push({
      label: 'Faces',
      value: `${getCompassName(horn.aspectDeg)} (${Math.round(horn.aspectDeg)}°)`
    });
  }
  if (state.product === 'shadow' || (state.product === 'light' && state.light === 'sun')) {
    rows.push({
      label: 'Sun now',
      value: `${sun.altitude.toFixed(1)}° high, from ${getCompassName(sun.azimuth)}`
    });
  }
  return {
    title,
    ...(onGlacier ? {subtitle: 'On a glacier'} : {}),
    rows
  };
}

/**
 * The histogram of the displayed product: `bins` equal bins over `[0, domainMaximum]` in the
 * table's unit, bars coloured by the map's class table, with a marker at one value (the village).
 *
 * @param stats Histogram of the raw raster (1,024 bins between its minimum and maximum).
 * @param scale Multiplier from the raster's unit to the table's (Wh to kWh is 0.001).
 */
export function getValueHistogramChart(
  stats: Extract<ValueStats, {kind: 'float'}>,
  table: ClassTable,
  options: {
    domainMaximum: number;
    scale?: number;
    bins?: number;
    xLabel: string;
    unit: string;
    marker?: {value: number; label: string};
  }
): BarChartData {
  const bins = options.bins ?? 32;
  const scale = options.scale ?? 1;
  const counts = new Array<number>(bins).fill(0);
  const source = stats.histogram;
  const span = stats.max - stats.min;
  for (let index = 0; index < source.length; index++) {
    if (source[index] === 0) continue;
    const raw = span > 0 ? stats.min + ((index + 0.5) / source.length) * span : stats.min;
    const position = Math.floor(((raw * scale) / options.domainMaximum) * bins);
    counts[Math.min(bins - 1, Math.max(0, position))] += source[index];
  }
  return {
    kind: 'histogram',
    values: counts,
    xDomain: [0, options.domainMaximum],
    breaks: table.breaks,
    classColors: table.colors,
    xLabel: options.xLabel,
    yLabel: 'cells',
    formatY: value =>
      value >= 1e6 ? `${(value / 1e6).toFixed(1)} M` : `${Math.round(value / 1e3)} k`,
    description: `Cells by ${options.unit}, bars coloured by the map classes.`,
    ...(options.marker ? {now: options.marker.value, nowLabel: options.marker.label} : {})
  };
}

/** The share of finite cells whose difference exceeds {@link DIFFERENCE_THRESHOLD}. */
export function getDifferenceShare(values: Float32Array): number {
  let finite = 0;
  let different = 0;
  for (const value of values) {
    if (!Number.isFinite(value)) continue;
    finite++;
    if (value > DIFFERENCE_THRESHOLD) different++;
  }
  return finite > 0 ? different / finite : Number.NaN;
}

/** `"4.2%"` of a share, with `<0.1%` for a tiny non-zero one. */
export function formatShare(share: number): string {
  if (!Number.isFinite(share)) return '-';
  const percent = share * 100;
  if (percent > 0 && percent < 0.1) return '<0.1%';
  return `${percent.toFixed(percent < 10 ? 1 : 0)}%`;
}
