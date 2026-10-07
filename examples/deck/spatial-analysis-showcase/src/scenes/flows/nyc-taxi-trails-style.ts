// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {makeClassTable} from '../../cartography/class-table';
import type {ClassTable} from '../../cartography/types';
import {getClassColors, sampleRamp, type PaletteColor} from '../../engine/ramps';
import type {DiagramChartData} from '../scene';

/**
 * Symbolisation of the nyc-taxi-trails scene that is not a shared table: the marks of the night
 * ground (white heads, taxi-yellow trails), the fare class table with its matched and mismatched
 * ramps, the heading rose colours and the fade diagram. Pure data and strings: no luma.gl, so the
 * scene file can import it for its legends.
 */

/** Taxi yellow of the trails when they are not coloured by anything. */
export const TRAIL_INK: PaletteColor = [255, 201, 51, 215];

/** The one bright mark: a white head with a ground-colour halo. */
export const HEAD_INK: PaletteColor = [255, 255, 255, 255];

/** Halo of the heads: the night ground. */
export const HEAD_HALO: PaletteColor = [10, 13, 18, 235];

/** Every route of the sample as a faint backdrop (the context tier of `CONTEXT_INK.dark`). */
export const BACKDROP_INK: PaletteColor = [138, 148, 163, 28];

/** The straight chord from pickup to drop-off: dashed white at 0.35. */
export const CHORD_INK: PaletteColor = [255, 255, 255, 89];

/** The full route of a chosen cab, under its chord. */
export const ROUTE_INK: PaletteColor = [255, 255, 255, 120];

/** The inspected cab: its route in grey, the vertex ticks lighter. */
export const INSPECT_ROUTE_INK: PaletteColor = [176, 184, 198, 230];
export const INSPECT_TICK_INK: PaletteColor = [214, 220, 230, 255];

/** Trail width in CSS pixels by zoom: 1.5 px up to the city view, 2.25 px at street level. */
export const TRAIL_WIDTH_STOPS: readonly (readonly [number, number])[] = [
  [11, 1.5],
  [13, 2.25]
];

/** Head radius in CSS pixels by zoom: 3.5 px, 5 px from zoom 13.5. */
export const HEAD_RADIUS_STOPS: readonly (readonly [number, number])[] = [
  [13, 3.5],
  [13.5, 5]
];

/** Sectors of the heading rose: 10 degrees each, so the avenue grid shows as four spikes. */
export const HEADING_BINS = 36;

/** Most cabs the chords step draws, and the shortest chord (metres) worth drawing. */
export const MAXIMUM_CHORDS = 40;
export const MINIMUM_CHORD_METERS = 800;

/** The seed of the cyclic ramp: heading colours are `romao` sampled at `degrees / 360`. */
export const HEADING_RAMP = 'romao' as const;

const WINDS = [
  'north',
  'north-north-east',
  'north-east',
  'east-north-east',
  'east',
  'east-south-east',
  'south-east',
  'south-south-east',
  'south',
  'south-south-west',
  'south-west',
  'west-south-west',
  'west',
  'west-north-west',
  'north-west',
  'north-north-west'
] as const;

/** Name of the 16-point compass direction of a heading in degrees clockwise from north. */
export function getCompassName(degrees: number): string {
  const wrapped = ((degrees % 360) + 360) % 360;
  return WINDS[Math.round(wrapped / 22.5) % 16];
}

/** Colours of the heading rose: the cyclic ramp at the centre of each sector. */
export function getHeadingColors(): PaletteColor[] {
  return Array.from({length: HEADING_BINS}, (_, bin) => {
    const [r, g, b] = sampleRamp(HEADING_RAMP, (bin + 0.5) / HEADING_BINS);
    return [r, g, b, 255] as PaletteColor;
  });
}

/** How a fare ramp sits on the night ground. */
export type RampMatch = 'matched' | 'mismatched';

/**
 * Five class colours for the fare, low class first. Matched: inferno trimmed to 0.3-1, so the
 * dearest trips are the brightest marks and the cheapest still clear the ground. Mismatched: the
 * same ramp reversed, the habit of a light page, so the dearest trips sink into the dark.
 */
export function getFareColors(match: RampMatch, classCount = 5, alpha = 255): PaletteColor[] {
  return getClassColors('inferno', classCount, match === 'mismatched', alpha, [0.3, 1]);
}

/** Input of {@link getFareTable}. */
export type FareTableInput = {
  /** Four quintile breaks in USD. */
  breaks: readonly number[];
  /** Cheapest and dearest fare, USD. */
  extent: readonly [number, number];
  match: RampMatch;
};

/** The fare class table of the trails and its legend: five quantile classes, one table. */
export function getFareTable({breaks, extent, match}: FareTableInput): ClassTable {
  return makeClassTable({
    breaks,
    colors: getFareColors(match, breaks.length + 1),
    unit: 'USD',
    extent,
    format: value => `$${value % 1 ? value.toFixed(2) : value}`,
    method:
      match === 'matched'
        ? 'Quintiles of the fare; brightest is dearest, toward contrast with the ground'
        : 'Quintiles of the fare; the ramp runs against the ground, dearest is darkest',
    noData: {label: 'No fare recorded'}
  });
}

/**
 * The fade of a trail as a small diagram: alpha against age, `(1 - age / tail)^2`, opaque at the
 * head and gone at the tail of the window.
 */
export function getFadeDiagram(): DiagramChartData {
  const left = 24;
  const right = 304;
  const top = 12;
  const bottom = 84;
  const points = Array.from({length: 21}, (_, step) => {
    const age = step / 20;
    const x = left + age * (right - left);
    const y = bottom - (1 - age) ** 2 * (bottom - top);
    return `${step === 0 ? 'M' : 'L'}${x.toFixed(1)} ${y.toFixed(1)}`;
  }).join(' ');
  return {
    kind: 'diagram',
    width: 320,
    height: 112,
    description:
      'A curve of trail opacity against age: fully opaque at the cab and fading to nothing at the end of the trail.',
    svg: [
      `<line class="diagram-muted" x1="${left}" y1="${bottom}" x2="${right}" y2="${bottom}"/>`,
      `<line class="diagram-muted" x1="${left}" y1="${top}" x2="${left}" y2="${bottom}"/>`,
      `<path class="diagram-fill" d="${points} L${right} ${bottom} L${left} ${bottom} Z"/>`,
      `<path class="diagram-signal" d="${points}"/>`,
      `<text class="diagram-muted" x="${left + 6}" y="${top + 10}">opaque</text>`,
      `<text class="diagram-ink" x="${left}" y="104">at the cab</text>`,
      `<text class="diagram-ink" x="${right}" y="104" text-anchor="end">a trail length behind</text>`
    ].join('')
  };
}
