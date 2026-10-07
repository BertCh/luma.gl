// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {makeClassTable} from '../../cartography/class-table';
import {MAP_INK, hexToRgba, type MapGround} from '../../cartography/hue-registry';
import type {ClassTable} from '../../cartography/types';
import type {PaletteColor} from '../../engine/ramps';
import type {LegendSpec} from '../scene';
import {getWaitingClasses, ZONE_EVENT_INK, inkFor} from './movement-style';
import type {OutlineStyleName} from './zone-dwell-geometry';
import {
  formatBreak,
  getUnitBreakFactor,
  getZoneValueLabels,
  type ZoneDefinition,
  type ZoneMetric,
  type ZoneUnit
} from './zone-dwell-values';

/**
 * The look of the zone-dwell plate, as data: the one class table (layer, legend, tooltip and
 * chart read it), the three outline line styles, the event and stop inks and their legends.
 * Pure TypeScript.
 */

/** Label of the "nothing to say" swatch: zero is a value, not missing data. */
export const NO_TIME_LABEL = 'No vessel time (outline only)';

/** Everything `makeZoneClassTable` needs. */
export type ZoneTableOptions = {
  /** Interior class breaks in vessel-hours (or hours, or visits), before the unit factor. */
  breaks: readonly number[];
  ground: MapGround;
  metric: ZoneMetric;
  unit: ZoneUnit;
  definition: ZoneDefinition;
  /** `[min, max]` of the zones with time, in the unit of the map. */
  extent: readonly [number, number];
  /** How the breaks were chosen, for the legend note. */
  method: string;
};

/**
 * The class table of the zone fill: the registry's `waiting` row (YlOrRd on paper, magma-like on
 * dark ground), exact breaks, a swatch for zones with no vessel time.
 */
export function makeZoneClassTable(options: ZoneTableOptions): ClassTable {
  const factor = getUnitBreakFactor(options.metric, options.unit);
  const breaks = options.breaks.map(value => value * factor);
  const classCount = breaks.length + 1;
  const palette = getWaitingClasses(options.ground, Math.max(classCount, 3)).slice(0, classCount);
  const labels = getZoneValueLabels(options.metric, options.unit, options.definition);
  return makeClassTable({
    breaks,
    colors: palette,
    unit: labels.unit,
    extent: options.extent,
    method: options.method,
    noData: {label: NO_TIME_LABEL},
    format: formatBreak
  });
}

/** One outline line style. */
export type OutlineStyle = {
  color: PaletteColor;
  widthPixels: number;
  /** Ground-coloured casing on both sides of the line (the hairline that separates fills). */
  casing?: PaletteColor;
  casingPixels?: number;
  dashArray?: readonly [number, number];
};

/** Polygon hairline: white at 0.63 on light grounds, ground ink at 0.5 on dark. */
function getHairline(ground: MapGround): PaletteColor {
  return ground === 'dark' ? [14, 17, 22, 128] : [255, 255, 255, 161];
}

/**
 * The three line styles of the zone boundaries: official anchorage solid, maintained channel
 * lighter, hand-drawn approximations dashed. Kind is told by line style, never by hue.
 */
export function getOutlineStyles(ground: MapGround): Record<OutlineStyleName, OutlineStyle> {
  const ink = hexToRgba(MAP_INK[ground].ink);
  const withAlpha = (alpha: number): PaletteColor => [ink[0], ink[1], ink[2], alpha];
  const casing = getHairline(ground);
  return {
    official: {color: withAlpha(184), widthPixels: 0.8, casing, casingPixels: 0.5},
    channel: {color: withAlpha(96), widthPixels: 0.8, casing, casingPixels: 0.5},
    approximate: {color: withAlpha(214), widthPixels: 1.1, dashArray: [5, 3]}
  };
}

/** Context track ink on this ground (one neutral ink, normal blending on paper). */
export function getTrackInk(ground: MapGround): PaletteColor {
  return ground === 'dark' ? [200, 210, 235, 40] : [40, 50, 70, 40];
}

/** The ground-coloured ring around stop discs. */
export function getStopRing(ground: MapGround): PaletteColor {
  return ground === 'dark' ? [10, 13, 18, 235] : [255, 255, 255, 235];
}

/** Enter and exit ink of the event pulses (sky and orange). */
export function getEventInks(ground: MapGround): {enter: PaletteColor; exit: PaletteColor} {
  return {enter: inkFor(ZONE_EVENT_INK.enter, ground), exit: inkFor(ZONE_EVENT_INK.exit, ground)};
}

/** Class breaks of a stop's duration in seconds (the stop discs share the waiting colours). */
export const STOP_BREAKS_SECONDS: readonly number[] = [1800, 7200, 21600, 43200];

/** Labels of the five stop classes. */
export const STOP_CLASS_LABELS: readonly string[] = [
  'Under 30 min',
  '30 min to 2 h',
  '2 to 6 h',
  '6 to 12 h',
  'Over 12 h'
];

/** The legend entry of the outline line styles. */
export function getOutlineLegend(ground: MapGround): LegendSpec {
  const styles = getOutlineStyles(ground);
  return {
    kind: 'line',
    title: 'Zone outline',
    entries: [
      {
        color: styles.official.color,
        widthPixels: styles.official.widthPixels + 0.4,
        label: 'Official anchorage'
      },
      {
        color: styles.channel.color,
        widthPixels: styles.channel.widthPixels + 0.4,
        label: 'Maintained channel'
      },
      {
        color: styles.approximate.color,
        widthPixels: styles.approximate.widthPixels + 0.2,
        label: 'Approximate, hand-drawn',
        dashed: true
      }
    ],
    note: 'Kind is told by line style, not colour.'
  };
}

/** The legend entry of the enter and exit pulses. */
export function getEventLegend(ground: MapGround): LegendSpec {
  const inks = getEventInks(ground);
  return {
    kind: 'categories',
    title: 'Where a track crosses a zone edge',
    entries: [
      {color: inks.enter, label: 'Enters the zone', shape: 'ring'},
      {color: inks.exit, label: 'Exits the zone', shape: 'ring'}
    ],
    note: 'Rings pulse at the interpolated crossing, as the clock passes it.'
  };
}

/** The legend entry of the stop discs, with the waiting colours of the fill. */
export function getStopLegend(ground: MapGround): LegendSpec {
  const colors = getWaitingClasses(ground, 5);
  return {
    kind: 'categories',
    title: 'Stops inside the zones',
    entries: STOP_CLASS_LABELS.map((label, index) => ({
      color: colors[index],
      label,
      shape: 'dot' as const
    })),
    note: 'Disc size and colour grow with the time stopped.'
  };
}
