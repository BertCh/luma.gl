// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {labelsFor, RANDSTAD} from '../../cartography/gazetteer';
import {hexToRgba} from '../../cartography/class-table';
import type {MapAnnotation} from '../scene';

/** The time window and source wording repeated in the three Randstad cartouches. */
export const RANDSTAD_SCHEDULE_CREDIT =
  'Schedules: OVapi / NDOV GTFS, CC0 · timetable, not observed vehicles';

/** Separate rail-graph vintage; it must not inherit the playback scenes’ July trip date. */
export const RANDSTAD_RAIL_CREDIT =
  'Rail graph: OVapi / NDOV GTFS, CC0 · service date 9 October 2026';

/** Orientation labels used above the timetable layers rather than below them. */
export const RANDSTAD_ORIENTATION = labelsFor(
  RANDSTAD,
  ['amsterdam', 'rotterdam', 'den-haag', 'utrecht', 'schiphol', 'haarlem'],
  {
    amsterdam: {minZoom: 8, priority: 4},
    rotterdam: {minZoom: 8, priority: 4},
    'den-haag': {minZoom: 8, priority: 4},
    utrecht: {minZoom: 8, priority: 4},
    schiphol: {minZoom: 9, tone: 'muted'},
    haarlem: {minZoom: 9.5, tone: 'muted'},
    leiden: {minZoom: 9.5, tone: 'muted'}
  }
);

/** A compact national label set for the rail-reach story. */
export const RAIL_ORIENTATION = labelsFor(
  RANDSTAD,
  ['amsterdam', 'rotterdam', 'den-haag', 'utrecht', 'schiphol', 'groningen'],
  {groningen: {minZoom: 6.9, priority: 3}}
);

/** The GTFS clipping extent, explicitly drawn and labelled in the frequency and playback scenes. */
export const RANDSTAD_DATA_FRAME: readonly MapAnnotation[] = [
  {
    id: 'randstad-data-edge',
    kind: 'note',
    coordinate: [5.16, 52.43],
    title: 'Timetable data ends here',
    text: 'Trips are clipped to this Randstad extent.',
    tone: 'muted',
    maxZoom: 10,
    priority: 5
  }
];

/** Class thresholds in both-direction vehicles/hour and their rider-facing approximation. */
export const SERVICE_BREAKS = [2, 4, 8, 12, 20] as const;
export const SERVICE_LABELS = [
  '< 2 veh/h · over 60 min',
  '2–4 veh/h · 30–60 min',
  '4–8 veh/h · 15–30 min',
  '8–12 veh/h · 10–15 min',
  '12–20 veh/h · 6–10 min',
  '20+ veh/h · under 6 min'
] as const;

/** Shared class lookup for renderer evidence, legend, tooltip and class histogram. */
export function getServiceClassIndex(vehiclesPerHour: number): number {
  const index = SERVICE_BREAKS.findIndex(breakpoint => vehiclesPerHour < breakpoint);
  return index < 0 ? SERVICE_LABELS.length - 1 : index;
}

/** Full rider-facing class label for one both-direction service value. */
export function getServiceClassLabel(vehiclesPerHour: number): string {
  return SERVICE_LABELS[getServiceClassIndex(vehiclesPerHour)];
}

/** Fixed scheduled-speed classes shared by the playback markers, legend and histogram. */
export const TRANSIT_SPEED_BREAKS_KILOMETERS_PER_HOUR = [15, 30, 60, 100] as const;
export const TRANSIT_SPEED_LABELS = ['0–15', '15–30', '30–60', '60–100', '100+'] as const;

/** YlOrRd on paper and a lifted magma equivalent on dark ground, low class first. */
export function getServiceClassColors(dark: boolean) {
  const colors = dark
    ? ['#2d1160', '#721f81', '#b63679', '#f1605d', '#fec287', '#fcfdbf']
    : ['#ffffb2', '#fed976', '#feb24c', '#fd8d3c', '#f03b20', '#bd0026'];
  return colors.map((color, index) => hexToRgba(color, index === 0 ? 150 : 218));
}

/** Trimmed magma classes for scheduled speed, low to high, on either map ground. */
export function getTransitSpeedClassColors(dark: boolean) {
  const colors = dark
    ? ['#36106b', '#7c2382', '#c63d73', '#f66e5b', '#fdd08c']
    : ['#4a0c6b', '#8c2981', '#ca3f70', '#f46d5a', '#fbc17d'];
  return colors.map(color => hexToRgba(color, 235));
}

/** Approximate per-direction headway, explicitly avoiding a false route-level promise. */
export function formatServiceHeadway(vehiclesPerHour: number): string {
  if (vehiclesPerHour <= 0) return 'no scheduled crossings';
  const minutes = 120 / vehiclesPerHour;
  return `about every ${minutes >= 20 ? Math.round(minutes / 5) * 5 : Math.round(minutes)} min each way`;
}
