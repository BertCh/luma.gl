// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getClassBreaks} from '../../cartography/breaks';
import {formatDwell} from './movement-style';
import type {ZoneGroups} from './zone-dwell-names';

/**
 * The numbers of the zone-dwell story: how the per-zone table of `GPUGroupStatistics` (count,
 * sum, maximum of the visit dwell, in seconds) becomes the value drawn on the map, in which unit,
 * and with which class breaks. Pure TypeScript.
 */

/** The statistic of a zone that fills the map. */
export type ZoneMetric = 'total' | 'mean' | 'longest' | 'visits';

/**
 * The unit of the `total` metric: vessel-hours in the zone, vessel-hours per km2, or vessels
 * present on average (vessel-hours over the 24 hours of the day).
 */
export type ZoneUnit = 'total' | 'density' | 'present';

/** Which definition of dwell a table or a number belongs to. */
export type ZoneDefinition = 'inside' | 'stopped' | 'both';

/** Seconds in the day of the AIS sample. */
export const SECONDS_PER_DAY = 86400;

/** Manual class breaks of the waiting metrics (mean and longest stay), in hours. */
export const WAITING_BREAKS_HOURS: readonly number[] = [0.5, 2, 6, 12];

/** The per-zone table the GPU reduces the visits (or the stops) to. */
export type ZoneStatistics = {
  /** Visits, or stops. */
  counts: Uint32Array;
  /** Summed dwell in seconds. */
  sums: Float32Array;
  /** Longest single dwell in seconds. */
  maximums: Float32Array;
};

/** The same table added up per named zone (pieces of one anchorage are one zone). */
export type GroupStatistics = {
  counts: Float64Array;
  sums: Float64Array;
  maximums: Float64Array;
};

/** Adds the per-zone table up per named zone. */
export function reduceToGroups(stats: ZoneStatistics, groups: ZoneGroups): GroupStatistics {
  const counts = new Float64Array(groups.groupCount);
  const sums = new Float64Array(groups.groupCount);
  const maximums = new Float64Array(groups.groupCount);
  for (let zone = 0; zone < groups.groupOfZone.length; zone++) {
    const group = groups.groupOfZone[zone];
    counts[group] += stats.counts[zone];
    sums[group] += stats.sums[zone];
    maximums[group] = Math.max(maximums[group], stats.maximums[zone]);
  }
  return {counts, sums, maximums};
}

/**
 * The value of every named zone in the unit of the map, or `NaN` where the zone has nothing to
 * say (no vessel time): total in vessel-hours, per km2 or vessels present; mean and longest stay
 * in hours; visits as a count.
 */
export function getGroupValues(
  stats: GroupStatistics,
  metric: ZoneMetric,
  unit: ZoneUnit,
  areasKm2: ArrayLike<number>
): Float64Array {
  const values = new Float64Array(stats.counts.length);
  for (let group = 0; group < values.length; group++) {
    let value = Number.NaN;
    if (stats.counts[group] > 0 && stats.sums[group] > 0) {
      const hours = stats.sums[group] / 3600;
      if (metric === 'total') {
        value =
          unit === 'density'
            ? hours / Math.max(areasKm2[group], 1e-9)
            : unit === 'present'
              ? hours / (SECONDS_PER_DAY / 3600)
              : hours;
      } else if (metric === 'mean') {
        value = hours / stats.counts[group];
      } else if (metric === 'longest') {
        value = stats.maximums[group] / 3600;
      } else {
        value = stats.counts[group];
      }
    }
    values[group] = value;
  }
  return values;
}

/**
 * Class breaks of a metric from the values of the zones that have time. The waiting metrics use
 * manual breaks; everything else quantiles, so every class holds about as many zones.
 */
export function computeClassBreaks(values: ArrayLike<number>, metric: ZoneMetric): number[] {
  if (metric === 'mean' || metric === 'longest') return [...WAITING_BREAKS_HOURS];
  return getClassBreaks(values, 5, 'quantile');
}

/**
 * The key a frozen class table is kept under: the breaks of vessel-hours are shared by vessels
 * present (a constant factor apart), per-km2 values get their own.
 */
export function getBreaksKey(metric: ZoneMetric, unit: ZoneUnit): string {
  if (metric !== 'total') return metric;
  return unit === 'density' ? 'total:density' : 'total:amount';
}

/** The factor between the breaks of vessel-hours and those of `unit`. */
export function getUnitBreakFactor(metric: ZoneMetric, unit: ZoneUnit): number {
  return metric === 'total' && unit === 'present' ? 3600 / SECONDS_PER_DAY : 1;
}

/** A number with about three significant figures and thousands separators. */
export function formatQuantity(value: number): string {
  if (!Number.isFinite(value)) return '–';
  const magnitude = Math.abs(value);
  const digits = magnitude >= 100 ? 0 : magnitude >= 1 ? 1 : 2;
  return value.toLocaleString('en-US', {minimumFractionDigits: 0, maximumFractionDigits: digits});
}

/** A class break label: whole numbers when large, trimmed decimals when small. */
export function formatBreak(value: number): string {
  const magnitude = Math.abs(value);
  const digits = magnitude >= 100 ? 0 : magnitude >= 10 ? 1 : 2;
  return value.toLocaleString('en-US', {minimumFractionDigits: 0, maximumFractionDigits: digits});
}

/** The words of a unit for the title, unit and basis of a legend. */
export type ZoneValueLabels = {
  /** Legend title, for example "Time inside the zone". */
  title: string;
  /** Value unit, for example "vessel-hours". */
  unit: string;
  /** The denominator, for example "per km²". */
  basis?: string;
};

/** The legend words of a metric and unit. */
export function getZoneValueLabels(
  metric: ZoneMetric,
  unit: ZoneUnit,
  definition: ZoneDefinition
): ZoneValueLabels {
  const where =
    definition === 'stopped'
      ? 'Time stopped in the zone'
      : definition === 'both'
        ? 'Time in the zone'
        : 'Time inside the zone';
  if (metric === 'total') {
    if (unit === 'density') return {title: where, unit: 'vessel-hours', basis: 'per km²'};
    if (unit === 'present') {
      return {title: 'Vessels present on average', unit: 'vessels'};
    }
    return {title: where, unit: 'vessel-hours'};
  }
  if (metric === 'mean') return {title: 'Mean stay per visit', unit: 'h'};
  if (metric === 'longest') return {title: 'Longest single stay', unit: 'h'};
  return {
    title: definition === 'stopped' ? 'Stops in the zone' : 'Visits to the zone',
    unit: definition === 'stopped' ? 'stops' : 'visits'
  };
}

/** A value of the map's unit as text, for tooltips, notes and readouts. */
export function formatZoneValue(value: number, metric: ZoneMetric, unit: ZoneUnit): string {
  if (!Number.isFinite(value)) return 'none';
  if (metric === 'total') {
    if (unit === 'density') return `${formatQuantity(value)} vessel-hours per km²`;
    if (unit === 'present') return `${formatQuantity(value)} vessels`;
    return `${formatQuantity(value)} vessel-hours`;
  }
  if (metric === 'visits') return `${formatQuantity(value)} visits`;
  return formatDwell(value * 3600);
}
