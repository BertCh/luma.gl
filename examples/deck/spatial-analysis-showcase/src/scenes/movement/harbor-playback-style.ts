// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * The look of harbor-playback: stop symbol sizes and classes, the inks of the focus-vessel layers
 * and the legends that go with them. The layers (compute file) and the legends (scene file) read
 * the same constants and functions, so what the map draws and what the key says cannot drift.
 */

import {MAP_INK, type MapGround} from '../../cartography/hue-registry';
import {hexToRgba} from '../../cartography/class-table';
import type {PaletteColor} from '../../engine/ramps';
import type {LegendSpec} from '../scene';
import {getWaitingClasses, SUBJECT_INK} from './movement-style';

// ---------------------------------------------------------------------------------------------
// Stops: four dwell classes, radius by the square root of the dwell
// ---------------------------------------------------------------------------------------------

/** Dwell class breaks in seconds: 30 minutes, 2 hours and 6 hours (the layer's `classBreaks`). */
export const STOP_BREAKS_SECONDS: readonly number[] = [1800, 7200, 21600];

/** The same breaks in minutes, for the dwell histogram (`breaks`) and the classes legend. */
export const STOP_BREAKS_MINUTES: readonly number[] = STOP_BREAKS_SECONDS.map(
  seconds => seconds / 60
);

/** Labels of the four dwell classes. */
export const STOP_CLASS_LABELS: readonly string[] = [
  'Under 30 min',
  '30 min to 2 h',
  '2 to 6 h',
  'Over 6 h'
];

/** Radius in CSS pixels of a zero-duration stop. */
export const STOP_BASE_RADIUS_PIXELS = 4;
/** Extra radius per square root of a second of dwell, so disc area grows with the dwell. */
export const STOP_RADIUS_PER_SQRT_SECOND = 0.08;
/** Upper bound of the stop radius in CSS pixels. */
export const STOP_MAXIMUM_RADIUS_PIXELS = 16;

/**
 * Radius in CSS pixels of a stop of `seconds`: the formula `StopMarkerLayer` evaluates per stop,
 * so the nested size legend and the map use one rule.
 */
export function getStopRadius(seconds: number): number {
  return Math.min(
    STOP_BASE_RADIUS_PIXELS + STOP_RADIUS_PER_SQRT_SECOND * Math.sqrt(Math.max(seconds, 0)),
    STOP_MAXIMUM_RADIUS_PIXELS
  );
}

/**
 * The four dwell class colours, alpha included: the registry's `waiting` row at five classes
 * without its first class. That class (deep purple on night, pale yellow on paper) sits too close
 * to the ground to read as a disc, and the four that remain keep the row's order and hues.
 */
export function getStopClassColors(ground: MapGround, alpha = 235): PaletteColor[] {
  return getWaitingClasses(ground, 5)
    .slice(1)
    .map(color => [color[0], color[1], color[2], alpha] as PaletteColor);
}

/**
 * The stops key: a classes legend (with the number of stops per class when known) and the nested
 * size legend of the same three breaks. Both come from {@link STOP_BREAKS_SECONDS}.
 */
export function getStopLegends(ground: MapGround, classCounts?: readonly number[]): LegendSpec[] {
  const colors = getStopClassColors(ground, 255);
  return [
    {
      kind: 'classes',
      id: 'stop-classes',
      title: 'Time stopped',
      unit: 'min',
      breaks: STOP_BREAKS_MINUTES,
      colors,
      labels: STOP_CLASS_LABELS,
      ...(classCounts ? {counts: classCounts} : {}),
      layout: 'list',
      noData: {label: 'Not a stop'},
      note: 'Slower than the stop speed for at least the minimum duration.'
    },
    {
      kind: 'size',
      title: 'Disc radius grows with the dwell',
      layout: 'nested',
      entries: STOP_BREAKS_SECONDS.map((seconds, index) => ({
        radiusPixels: getStopRadius(seconds),
        label: ['30 min', '2 h', '6 h'][index]
      })),
      color: colors[1],
      unit: 'time stopped',
      note: 'Area follows the square root of the dwell, capped at 6 hours.'
    }
  ];
}

// ---------------------------------------------------------------------------------------------
// Focus-vessel inks (steps 3 and 5)
// ---------------------------------------------------------------------------------------------

/** Inks of the layers that draw one vessel's fixes, chord and resampled points. */
export type FocusInks = {
  /** A real AIS fix: a small hollow ring. */
  fix: PaletteColor;
  /** The two fixes either side of the playhead: filled. */
  bracketFix: PaletteColor;
  /** The straight chord the arrow is placed on. */
  chord: PaletteColor;
  /** The raw track between fixes, thin. */
  rawTrack: PaletteColor;
  /** A resampled point or route: the reader's parameter, so the signal colour. */
  sample: PaletteColor;
  /** The resampled polyline of the focus vessel. */
  sampleRoute: PaletteColor;
  /** Every track resampled (the fingerprint), neutral. */
  fingerprint: PaletteColor;
  /** Ground-coloured halo under the dots. */
  halo: PaletteColor;
};

/** Focus inks for a ground. Rings and chords are achromatic, samples use the signal colour. */
export function getFocusInks(ground: MapGround): FocusInks {
  const ink = hexToRgba(MAP_INK[ground].ink);
  const signal = hexToRgba(MAP_INK[ground].signal);
  const halo = hexToRgba(MAP_INK[ground].halo, 235);
  return {
    fix: [ink[0], ink[1], ink[2], 235],
    bracketFix: [ink[0], ink[1], ink[2], 255],
    chord: [ink[0], ink[1], ink[2], 255],
    rawTrack: [ink[0], ink[1], ink[2], 90],
    sample: [signal[0], signal[1], signal[2], 255],
    sampleRoute: [signal[0], signal[1], signal[2], 150],
    fingerprint: ground === 'dark' ? [200, 210, 235, 40] : [40, 50, 70, 60],
    halo: [halo[0], halo[1], halo[2], 235]
  };
}

// ---------------------------------------------------------------------------------------------
// Legends of the vessel layers
// ---------------------------------------------------------------------------------------------

/** The key of the single-colour view: one amber, one meaning ("a vessel with a position now"). */
export function getUniformVesselLegend(ground: MapGround): LegendSpec {
  return {
    kind: 'categories',
    title: 'Vessels',
    entries: [
      {
        color: SUBJECT_INK[ground],
        label: 'A vessel with a position now',
        detail: 'estimated between fixes',
        shape: 'dot'
      },
      {
        color: getContextSwatch(ground),
        label: 'Every track of the day',
        shape: 'line'
      }
    ],
    note: 'Arrows point along the heading.'
  };
}

/** A neutral swatch for context tracks in legends (the layer ink at a visible alpha). */
function getContextSwatch(ground: MapGround): PaletteColor {
  return ground === 'dark' ? [200, 210, 235, 110] : [40, 50, 70, 130];
}

/** The key of the fix layers of the focus vessel, by focus mode. */
export function getFocusLegend(mode: 'ferry' | 'transit', ground: MapGround): LegendSpec {
  const inks = getFocusInks(ground);
  if (mode === 'ferry') {
    return {
      kind: 'categories',
      title: 'One vessel between its fixes',
      layout: 'list',
      entries: [
        {color: inks.fix, label: 'An AIS fix (measured)', shape: 'ring'},
        {color: inks.bracketFix, label: 'The two fixes around the playhead', shape: 'dot'},
        {color: inks.chord, label: 'The chord between them (assumed)', shape: 'line'},
        {color: SUBJECT_INK[ground], label: 'The arrow (estimated)', shape: 'dot'}
      ],
      note: 'Position = fix A + fraction x (fix B - fix A).'
    };
  }
  return {
    kind: 'categories',
    title: 'One route, resampled',
    layout: 'list',
    entries: [
      {color: inks.fix, label: 'An AIS fix (measured)', shape: 'ring'},
      {color: inks.sample, label: 'A resampled point', shape: 'dot'},
      {color: inks.sampleRoute, label: 'The resampled route', shape: 'line'},
      {color: inks.fingerprint, label: 'Every route resampled', shape: 'line'}
    ],
    note: 'Every track gets the same number of points.'
  };
}
