// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationCapture} from '@luma.gl/gpgpu';
import type {ViewContext} from '../registry/view-registration';
import {getDeclaredVariableKind, getRunScaleRegistry} from '../views/value-strip';
import type {ViewGround} from '../types';
import {getClassColors, type RampName} from '../scales/ramps';

/**
 * Element-card design tokens (CARD-DESIGN): the library-wide role colours, class palettes, stroke
 * weights and sizes of result figures, twins and catalog thumbnails. Hosts load the same values as
 * CSS custom properties (`website/src/css/lens-card.css`); a node spec keeps both in sync.
 */

/** Drawing role of a mark in a result figure. */
export type CardRole = 'context' | 'input' | 'output' | 'highlight';

/** Class palette of a valued mark: ordered (`seq`), signed with a stated midpoint (`div`), iteration time (`time`), nominal (`nom`). */
export type CardPalette = 'seq' | 'div' | 'time' | 'nom';

/** Shape family of a mark, which decides fill versus stroke. */
export type CardMarkShape = 'point' | 'line' | 'area' | 'cell';

/** Number of classes of every card palette. */
export const CARD_CLASS_COUNT = 5;

/** Colour tokens per ground (CSS `--cv-<name>`). */
export const CARD_COLORS: Record<ViewGround, Readonly<Record<string, string>>> = {
  light: {
    paper: '#ffffff',
    ink: '#1f2328',
    muted: '#5c636e',
    'frame-border': '#e3e3e0',
    'context-line': '#d4d4d0',
    'context-fill': '#f1f1ef',
    input: '#5f6672',
    output: '#0072b2',
    highlight: '#d55e00',
    'output-wash': 'rgb(0 114 178 / 0.1)',
    'no-data': '#d9d9d9'
  },
  dark: {
    paper: '#13171c',
    ink: '#e8eaed',
    muted: '#9aa3ae',
    'frame-border': '#2a313a',
    'context-line': '#343b45',
    'context-fill': '#1c2128',
    input: '#a3abb7',
    output: '#56b4e9',
    highlight: '#f0782a',
    'output-wash': 'rgb(86 180 233 / 0.12)',
    'no-data': '#3a3f46'
  }
};

/** Library ramp, direction and trim each class palette is sampled from, per ground. */
export const CARD_PALETTE_SOURCES: Record<
  Exclude<CardPalette, 'nom'>,
  Record<ViewGround, {ramp: RampName; reverse: boolean; range: readonly [number, number]}>
> = {
  seq: {
    light: {ramp: 'blues', reverse: false, range: [0.25, 0.95]},
    dark: {ramp: 'blues', reverse: true, range: [0.05, 0.8]}
  },
  div: {
    light: {ramp: 'rdbu', reverse: false, range: [0, 1]},
    dark: {ramp: 'berlin', reverse: false, range: [0, 1]}
  },
  time: {
    light: {ramp: 'purples', reverse: false, range: [0.25, 1]},
    dark: {ramp: 'purples', reverse: true, range: [0, 0.75]}
  }
};

/** Okabe-Ito without black and without the two role hues (blue = output, vermillion = highlight). */
export const CARD_NOMINAL_COLORS: readonly string[] = [
  '#e69f00',
  '#56b4e9',
  '#009e73',
  '#cc79a7',
  '#f0e442'
];

function toHex(color: readonly number[]): string {
  return `#${color
    .slice(0, 3)
    .map(channel => channel.toString(16).padStart(2, '0'))
    .join('')}`;
}

/** The five class colours of a card palette on a ground, low class first. */
export function getCardPaletteColors(palette: CardPalette, ground: ViewGround): string[] {
  if (palette === 'nom') {
    return [...CARD_NOMINAL_COLORS];
  }
  const {ramp, reverse, range} = CARD_PALETTE_SOURCES[palette][ground];
  return getClassColors(ramp, CARD_CLASS_COUNT, reverse, 255, range).map(toHex);
}

/** Stroke widths, fill opacities and sizes (CSS `--cv-<name>`). */
export const CARD_METRICS = {
  'stroke-context': '0.75px',
  'stroke-input': '1px',
  'stroke-output': '1.5px',
  'stroke-highlight': '2.5px',
  'fill-input': '0.12',
  'fill-output': '0.22',
  'label-size': '11px',
  'radius-card': '10px',
  'radius-tile': '8px',
  'figure-max': '720px',
  'thumb-width': '160px',
  'thumb-height': '96px',
  'motion-swap': '150ms'
} as const;

/** Figure geometry of result figures, in SVG user units (CSS px). */
export const CARD_FIGURE = {
  width: 640,
  height: 360,
  padding: 24,
  panelGap: 16,
  stripRowHeight: 28,
  stripRowGap: 12,
  stripPadding: 16,
  maxStripCells: 16,
  thumbnailWidth: 160,
  thumbnailHeight: 96,
  thumbnailPadding: 6,
  thumbnailMarkScale: 0.6,
  maxFrames: 8,
  playMilliseconds: 5000
} as const;

/** Mark radii and stroke widths per role, in SVG user units, before the thumbnail scale. */
export const CARD_MARKS: Record<
  CardRole,
  {radius: number; strokeWidth: number; fillOpacity: number}
> = {
  context: {radius: 1.5, strokeWidth: 0.75, fillOpacity: 1},
  input: {radius: 2.5, strokeWidth: 1, fillOpacity: 0.12},
  output: {radius: 3, strokeWidth: 1.5, fillOpacity: 0.22},
  highlight: {radius: 4.5, strokeWidth: 2.5, fillOpacity: 1}
};

/**
 * SVG attributes of one result-figure mark: the role and class names the host CSS styles, plus
 * light-ground presentation attributes so a figure stays legible without the stylesheet.
 */
export function getCardMarkAttributes(
  role: CardRole,
  shape: CardMarkShape,
  options: {palette?: CardPalette; classIndex?: number; noData?: boolean; scale?: number} = {}
): Record<string, string> {
  const scale = options.scale ?? 1;
  const marks = CARD_MARKS[role];
  const roleColor =
    role === 'context' ? CARD_COLORS.light['context-line'] : CARD_COLORS.light[role];
  const classNames = ['cv-mark', `cv-mark--${role}`, `cv-${shape}`];
  let fill = role === 'context' ? CARD_COLORS.light['context-fill'] : roleColor;
  if (options.noData) {
    classNames.push('cv-v', 'cv-no-data');
    fill = CARD_COLORS.light['no-data'];
  } else if (options.palette && options.classIndex !== undefined) {
    const classIndex = Math.min(CARD_CLASS_COUNT - 1, Math.max(0, options.classIndex));
    classNames.push('cv-v', `cv-${options.palette}-${classIndex + 1}`);
    fill = getCardPaletteColors(options.palette, 'light')[classIndex];
  }
  const valued = classNames.includes('cv-v');
  const attributes: Record<string, string> = {
    class: classNames.join(' '),
    'data-role': role,
    'stroke-width': String(marks.strokeWidth * scale)
  };
  if (shape === 'line') {
    attributes['fill'] = 'none';
    attributes['stroke'] = valued ? fill : roleColor;
  } else {
    attributes['fill'] = fill;
    attributes['stroke'] = valued || shape === 'cell' ? CARD_COLORS.light['paper'] : roleColor;
    if (!valued && shape === 'area') {
      attributes['fill-opacity'] = String(marks.fillOpacity);
    }
  }
  if (shape === 'point') {
    attributes['r'] = String(marks.radius * scale);
    attributes['stroke'] = CARD_COLORS.light['paper'];
    attributes['stroke-width'] = String((role === 'highlight' ? 2 : 0.75) * scale);
    if (role === 'highlight') attributes['paint-order'] = 'stroke';
  }
  if (shape === 'cell') {
    attributes['stroke-width'] = String(scale);
    if (role === 'input') attributes['fill-opacity'] = '0.35';
  }
  if (valued && shape === 'area') {
    attributes['fill-opacity'] = '1';
    attributes['stroke-width'] = String(0.5 * scale);
  }
  if (role === 'context') {
    attributes['aria-hidden'] = 'true';
  }
  return attributes;
}

/** CSS custom properties of every token for one ground, as declarations. */
export function getCardTokenDeclarations(ground: ViewGround): string[] {
  const declarations = Object.entries(CARD_COLORS[ground]).map(
    ([name, value]) => `--cv-${name}: ${value};`
  );
  for (const palette of ['seq', 'div', 'time', 'nom'] as const) {
    if (ground === 'dark' && palette === 'nom') continue;
    getCardPaletteColors(palette, ground).forEach((color, index) =>
      declarations.push(`--cv-${palette}-${index + 1}: ${color};`)
    );
  }
  if (ground === 'light') {
    for (const [name, value] of Object.entries(CARD_METRICS)) {
      declarations.push(`--cv-${name}: ${value};`);
    }
  }
  return declarations;
}

/** Theme-aware token stylesheet, matching the host's explicit-theme precedence. */
export function getCardTokenCss(): string {
  const light = getCardTokenDeclarations('light').join('\n  ');
  const dark = getCardTokenDeclarations('dark').join('\n  ');
  return `:root {\n  ${light}\n}\n:root[data-theme='dark'] {\n  ${dark}\n}\n@media (prefers-color-scheme: dark) {\n  :root:not([data-theme]) {\n  ${dark}\n  }\n}`;
}

/** Selects a palette from declared measurement metadata, never from the sign of values. */
export function getCardPaletteForVariable(
  context: ViewContext,
  captureName: string,
  capture?: GPUComputationCapture
): CardPalette {
  const kind = getDeclaredVariableKind(context, captureName);
  if (kind?.level === 'nominal') return 'nom';
  // VariableKind has no midpoint; only a registered scale can currently declare one.
  const registry = getRunScaleRegistry(context);
  const reference = registry.getByVariable(captureName);
  const scale = reference && registry.get(reference);
  if (scale?.kind === 'diverging' && scale.midpoint !== undefined) return 'div';
  if (/round|iteration/i.test(capture?.meaning ?? '')) return 'time';
  return 'seq';
}
