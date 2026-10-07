// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * The hue registry of the showcase: one colour scheme per QUESTION, so a reader learns the colours
 * once (SYNTHESIS 1.2, rule 3.1.1). A coordinator who wants a different ramp for a registry
 * question writes the reason in the design sheet.
 *
 * - {@link HUE_REGISTRY}: the questions (harm, access, deviation, z-scores, party ...) with their
 *   light-ground and dark-ground tables, copied from the decision table.
 * - {@link getRegistryColors}: the colours of a question on a ground, optionally at another class
 *   count.
 * - {@link MAP_INK}: the ink tokens (text, halo, water, context, signal, rule, no-data) as hex for
 *   canvas and WGSL, mirroring the `--map-*` CSS tokens.
 * - Categorical sets ({@link OKABE_ITO_LIGHT}, {@link RACE_GROUP_COLORS}, {@link HURRICANE_CLASS}),
 *   selection and signal colours, and {@link EFFORT_CAVEAT}.
 *
 * Hue guards: blue means water only in terrain, hydrology and earth; red never means "you" or
 * "error" and "the fire" at once; "brightest = highest" holds on dark grounds and is reversed on
 * light grounds; "between groups / other" is neutral grey, never a saturated hue. Selection is
 * achromatic (ink core plus ground halo), reader-placed inputs are a ring-and-dot in the signal
 * colour, the terrain eye is gold.
 *
 * Classed colour comes from exact ColorBrewer tables ({@link getClassPalette}), not from resampled
 * 9-stop ramps.
 */

import {type PaletteColor, type RampName, sampleRamp} from '../engine/ramps';
import {
  type ClassSchemeName,
  deriveDarkDiverging,
  deriveDarkSequential,
  getClassPalette,
  getClassPaletteHex,
  hexToRgba,
  liftLightnessFloor,
  resampleHexTable,
  rgbToHex
} from './class-table';

export {hexToRgba, rgbToHex};

/** A ground the map is drawn on: the page theme's light or dark basemap. */
export type MapGround = 'light' | 'dark';

/** A value for each ground. */
export type GroundPair<T> = {light: T; dark: T};

/**
 * Picks the light or the dark member of a `{light, dark}` pair.
 *
 * @example
 * ```ts
 * const color = getGroundColor(NO_DATA_COLOR, ground);
 * ```
 */
export function getGroundColor<T>(color: GroundPair<T>, ground: MapGround): T {
  return color[ground];
}

// ---------------------------------------------------------------------------------------------
// Ink tokens
// ---------------------------------------------------------------------------------------------

/** One ground's ink tokens (hex or CSS colour, plus alphas where the token is drawn translucent). */
export type MapInkTokens = {
  /** Text and strong lines (`--map-ink`). */
  ink: string;
  /** Secondary text (`--map-ink-muted`). */
  inkMuted: string;
  /** Label halo colour (`--map-halo`) and its default alpha 0-1. */
  halo: string;
  haloAlpha: number;
  /** Water fill / water label colour. */
  water: string;
  waterLabel: string;
  /** Ghost data (`--map-context`) and its alpha 0-1 (light 0.12-0.18, dark 0.10-0.15). */
  context: string;
  contextAlpha: number;
  /** "The thing you are controlling": window, brush, probe, observed marker (`--map-signal`). */
  signal: string;
  /** Furniture lines are the ink at this alpha (`--map-rule`). */
  rule: string;
  ruleAlpha: number;
  /** No-data fill and alpha. */
  noData: string;
  noDataAlpha: number;
  /** Not-significant ghost fill and alpha. */
  notSignificant: string;
  notSignificantAlpha: number;
};

/** The ink tokens of SYNTHESIS 1.2 for each ground, as hex for canvas and WGSL use. */
export const MAP_INK: GroundPair<MapInkTokens> = {
  light: {
    ink: '#1f2933',
    inkMuted: '#4a5663',
    halo: '#ffffff',
    haloAlpha: 0.92,
    water: '#2c6a99',
    waterLabel: '#35729f',
    context: '#8a949e',
    contextAlpha: 0.15,
    signal: '#d95f0e',
    rule: '#1f2933',
    ruleAlpha: 0.55,
    noData: '#d9d9d9',
    noDataAlpha: 0.7,
    notSignificant: '#eeeeee',
    notSignificantAlpha: 0.5
  },
  dark: {
    ink: '#e8edf2',
    inkMuted: '#aab6c3',
    halo: '#090c10',
    haloAlpha: 0.92,
    water: '#7fb3d9',
    waterLabel: '#7fb3d9',
    context: '#8a94a3',
    contextAlpha: 0.12,
    signal: '#ff9e45',
    rule: '#e8edf2',
    ruleAlpha: 0.55,
    noData: '#3a3f46',
    noDataAlpha: 1,
    notSignificant: '#2b2f38',
    notSignificantAlpha: 0.25
  }
};

/** No-data fill per ground (RGBA 0-255; light `#D9D9D9` at 0.7, dark `#3A3F46`). */
export const NO_DATA_COLOR: GroundPair<PaletteColor> = {
  light: hexToRgba(MAP_INK.light.noData, Math.round(MAP_INK.light.noDataAlpha * 255)),
  dark: hexToRgba(MAP_INK.dark.noData, Math.round(MAP_INK.dark.noDataAlpha * 255))
};

/** Not-significant ghost per ground (light `#EEEEEE` at 0.5, dark `#2B2F38` at 0.25). */
export const NOT_SIGNIFICANT_COLOR: GroundPair<PaletteColor> = {
  light: hexToRgba(
    MAP_INK.light.notSignificant,
    Math.round(MAP_INK.light.notSignificantAlpha * 255)
  ),
  dark: hexToRgba(MAP_INK.dark.notSignificant, Math.round(MAP_INK.dark.notSignificantAlpha * 255))
};

/** Selection is achromatic: an ink core with a ground-colour halo (SYNTHESIS 1.2). */
export const SELECTION_INK = {
  /** Core stroke colour: near-black on light grounds, white on dark. */
  core: {light: '#111827', dark: '#ffffff'} as GroundPair<string>,
  /** Halo colour: the ground colour. */
  halo: {light: '#ffffff', dark: '#13171c'} as GroundPair<string>,
  coreWidth: 2,
  haloWidth: 4,
  note: 'Ink core 2 px with a 4 px ground-colour halo. Never a hue.'
} as const;

/** The terrain observer's gold eye: orange is the "marginal" class of the visibility triad. */
export const TERRAIN_EYE_GOLD = '#f2b134';

/** The shared observer-effort chip text (rule 3.1.15), stated once per chapter step 1. */
export const EFFORT_CAVEAT = 'Counts measure where people look, not how much wildlife there is.';

/** "Other / between groups": neutral grey, never a saturated hue (hollow ring for points). */
export const OTHER_GREY: GroundPair<string> = {light: '#9aa0a6', dark: '#7c8590'};

// ---------------------------------------------------------------------------------------------
// Categorical sets
// ---------------------------------------------------------------------------------------------

function toPalette(hexes: readonly string[], alpha = 255): PaletteColor[] {
  return hexes.map(hex => hexToRgba(hex, alpha));
}

const OKABE_ITO_LIGHT_HEX = [
  '#0072b2',
  '#d55e00',
  '#009e73',
  '#cc79a7',
  '#e69f00',
  '#56b4e9',
  '#7a5195',
  '#8c6d31'
] as const;

const OKABE_ITO_DARK_HEX = [
  '#3fa0e0',
  '#ff8a3d',
  '#2fd0a0',
  '#e39bc6',
  '#ffc247',
  '#7ccbf5',
  '#a98bd0',
  '#d2a85a'
] as const;

/** Categorical hex, light ground: at most 7 hues plus {@link OTHER_GREY}; no yellow `#F0E442` on paper. */
export const OKABE_ITO_LIGHT_HEXES: readonly string[] = OKABE_ITO_LIGHT_HEX;

/** Categorical hex, dark ground (each hue lifted; same order as {@link OKABE_ITO_LIGHT_HEXES}). */
export const OKABE_ITO_DARK_HEXES: readonly string[] = OKABE_ITO_DARK_HEX;

/** Okabe-Ito categorical set for light grounds (blue, vermillion, green, magenta, orange, sky, violet, olive). */
export const OKABE_ITO_LIGHT: readonly PaletteColor[] = toPalette(OKABE_ITO_LIGHT_HEX);

/** The lifted Okabe-Ito set for dark grounds, same hue order as {@link OKABE_ITO_LIGHT}. */
export const OKABE_ITO_DARK: readonly PaletteColor[] = toPalette(OKABE_ITO_DARK_HEX);

/** Race-group names in {@link RACE_GROUP_COLORS} order. */
export const RACE_GROUP_LABELS = ['White', 'Black', 'Hispanic', 'Asian', 'Other'] as const;

const RACE_HEX: GroundPair<readonly string[]> = {
  light: ['#0072b2', '#d55e00', '#009e73', '#ac3c8c', '#787c84'],
  dark: ['#56b4e9', '#ff9f1c', '#00c896', '#f064c8', '#969ba5']
};

/**
 * Race-group dot colours chapter-wide, in {@link RACE_GROUP_LABELS} order: White blue, Black
 * orange, Hispanic green, Asian purple-magenta, Other grey. Dominance tiers use the same hue
 * families as 3-step ramps.
 */
export const RACE_GROUP_COLORS: GroundPair<readonly PaletteColor[]> = {
  light: toPalette(RACE_HEX.light),
  dark: toPalette(RACE_HEX.dark)
};

/** The {@link RACE_GROUP_COLORS} as hex. */
export const RACE_GROUP_HEXES: GroundPair<readonly string[]> = RACE_HEX;

/** Hurricane class names, weakest first (tropical depression ... category 5). */
export const HURRICANE_CLASS_LABELS = ['TD', 'TS', 'C1', 'C2', 'C3', 'C4', 'C5'] as const;

const HURRICANE_HEX: GroundPair<readonly string[]> = {
  light: ['#9db4c3', '#5aa7b8', '#fed976', '#feb24c', '#fd8d3c', '#e31a1c', '#7a0177'],
  dark: ['#6f8fb8', '#4fb3b0', '#ffe28a', '#ffb347', '#ff7a3d', '#f0394a', '#ff4fd8']
};

/** Shared hurricane-class colours TD, TS, C1 ... C5 (dark ground lifted); one table for season, landfall, families. */
export const HURRICANE_CLASS: GroundPair<readonly PaletteColor[]> = {
  light: toPalette(HURRICANE_HEX.light),
  dark: toPalette(HURRICANE_HEX.dark)
};

/** The {@link HURRICANE_CLASS} as hex. */
export const HURRICANE_CLASS_HEXES: GroundPair<readonly string[]> = HURRICANE_HEX;

// ---------------------------------------------------------------------------------------------
// The registry
// ---------------------------------------------------------------------------------------------

/** Measurement level of a registry question. */
export type HueRegistryKind = 'sequential' | 'diverging' | 'cyclic' | 'qualitative';

/** One question of the registry. */
export type HueRegistryEntry = {
  /** The question the hue answers, in a sentence. */
  question: string;
  kind: HueRegistryKind;
  /** Light-ground table as hex, low class first (8-digit hex carries alpha). */
  light: readonly string[];
  /** Dark-ground table as hex, low class first ("brightest = highest"). */
  dark: readonly string[];
  /** ColorBrewer scheme this entry is built on (used to rebuild at another class count). */
  scheme?: ClassSchemeName;
  /** True when the scheme is used reversed (near = darkest in `access`). */
  reverse?: boolean;
  /** Continuous counterpart in `RAMP_STOPS`, for surfaces of the same question. */
  ramp?: RampName;
  /** For diverging entries: what the centre means. */
  midpointMeaning?: string;
  /** Where it is used, caveats, alternates. */
  note: string;
  /** Companion tables of the question (hex), such as the 7-class variant. */
  extras?: Readonly<Record<string, readonly string[]>>;
};

/** Ids of the registry questions. */
export type HueRegistryId =
  | 'harm'
  | 'fireEffect'
  | 'people'
  | 'nature'
  | 'natureWeights'
  | 'share'
  | 'socioEconomic'
  | 'access'
  | 'remoteness'
  | 'load'
  | 'similarity'
  | 'inequality'
  | 'rainfall'
  | 'elapsedTime'
  | 'waiting'
  | 'vegetationChange'
  | 'deviation'
  | 'zScore'
  | 'coefficient'
  | 'direction'
  | 'party'
  | 'windSpeed'
  | 'hurricane'
  | 'slope'
  | 'slopeGentle'
  | 'visibility'
  | 'warnings';

/** A classed sample of a perceptual ramp, trimmed: SYNTHESIS "inferno t 0.15-1" on dark grounds. */
function sampleHexes(ramp: RampName, count: number, range: readonly [number, number]): string[] {
  return Array.from({length: count}, (_, index) =>
    rgbToHex(sampleRamp(ramp, count <= 1 ? 1 : index / (count - 1), false, range))
  );
}

/** A cyclic ramp sampled at `count` evenly spaced hues around the circle (no repeated seam colour). */
function sampleCyclicHexes(ramp: RampName, count: number): string[] {
  return Array.from({length: count}, (_, index) => rgbToHex(sampleRamp(ramp, index / count)));
}

const YLORRD_7 = ['#ffffb2', '#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#b10026'];
const YLORRD_5 = ['#ffffb2', '#fecc5c', '#fd8d3c', '#f03b20', '#bd0026'];
const MAGMA_DARK_6 = ['#2d1160', '#721f81', '#b63679', '#f1605d', '#fec287', '#fcfdbf'];
const MAGMA_DARK_5 = ['#51127c', '#8c2981', '#de4968', '#fe9f6d', '#fcfdbf'];
const YLGNBU_5 = ['#ffffcc', '#a1dab4', '#41b6c4', '#2c7fb8', '#253494'];
const YLGNBU_7 = ['#ffffcc', '#c7e9b4', '#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#0c2c84'];
const YLGNBU_8 = [
  '#ffffd9',
  '#edf8b1',
  '#c7e9b4',
  '#7fcdbb',
  '#41b6c4',
  '#1d91c0',
  '#225ea8',
  '#0c2c84'
];
const RDBU_7 = ['#2166ac', '#67a9cf', '#d1e5f0', '#f7f7f7', '#fddbc7', '#ef8a62', '#b2182b'];
const DARK_RDBU_7 = ['#4da3ff', '#2a6fbf', '#1b3a66', '#2b3240', '#66361f', '#c8661f', '#ffb347'];
const SLOPE = ['#fee391', '#fdae61', '#f46d43', '#c51b7d', '#762a83', '#1b1b3a'];

/**
 * The hue registry. Light tables are the exact hexes of the SYNTHESIS 1.2 decision table; dark
 * tables are the authored ones where the table gives them, otherwise derived (reverse plus a
 * lightness floor for sequential, dark-centre arms for diverging) so the strongest class is the
 * brightest and the lowest class keeps 3:1 against `#13171C`. Lengths of the light and dark table
 * may differ (harm: 7 and 6); ask {@link getRegistryColors} with a `classCount` for parity.
 *
 * Hex is lowercase; an 8-digit hex carries alpha.
 */
export const HUE_REGISTRY: Record<HueRegistryId, HueRegistryEntry> = {
  harm: {
    question: 'Harm, hazard, intensity, burden, heat',
    kind: 'sequential',
    light: YLORRD_7,
    dark: MAGMA_DARK_6,
    scheme: 'YlOrRd',
    ramp: 'ylorrd',
    note: 'Rate-smoothing, vulnerability, outage, frequency, centrality, event counts. YlOrRd-7 on paper, magma t 0.25-1 classed on dark (6 classes).',
    extras: {light5: YLORRD_5, dark5: MAGMA_DARK_5}
  },
  fireEffect: {
    question: 'Effect of fire on the ground',
    kind: 'sequential',
    light: ['#fdcc8a', '#fc8d59', '#e34a33', '#7f0000'],
    dark: ['#fdcc8a', '#fc8d59', '#e34a33', '#7f0000'],
    ramp: 'orrd',
    note: 'Burn, site and terrain stories. Unburned is transparent; regrowth is cool (extras.regrowth). The same table on both grounds.',
    extras: {regrowth: ['#92c5de', '#2166ac']}
  },
  people: {
    question: 'People: density, size',
    kind: 'sequential',
    light: ['#ffffd4', '#fed98e', '#fe9929', '#d95f0e', '#993404'],
    dark: sampleHexes('inferno', 5, [0.15, 1]),
    scheme: 'YlOrBr',
    ramp: 'ylorbr',
    note: 'Population density; diabetes in weights and regression. Not for choropleth-classes, where each variable gets its own hue on purpose (RdPu for diabetes).'
  },
  nature: {
    question: 'Nature counts and density',
    kind: 'sequential',
    light: YLGNBU_5,
    dark: sampleHexes('inferno', 5, [0.15, 1]),
    scheme: 'YlGnBu',
    ramp: 'ylgnbu',
    note: 'Points, cells, joins, time, statistics. YlGnBu-7 in extras.light7. Dark: inferno t 0.15-1, additive; dots amber (extras.dot).',
    extras: {light7: YLGNBU_7, dot: ['#fec44f']}
  },
  natureWeights: {
    question: 'Nature counts where the park fill is green (weights chapter)',
    kind: 'sequential',
    light: ['#edf8fb', '#b3cde3', '#8c96c6', '#8856a7', '#810f7c'],
    dark: sampleHexes('inferno', 5, [0.15, 1]),
    scheme: 'BuPu',
    ramp: 'bupu',
    note: 'The one exception to the nature hue: BuPu because the park fill is green.'
  },
  share: {
    question: 'Share or rate of a subset',
    kind: 'sequential',
    light: ['#edf8e9', '#bae4b3', '#74c476', '#31a354', '#006d2c'],
    dark: deriveDarkSequential(['#edf8e9', '#bae4b3', '#74c476', '#31a354', '#006d2c']),
    scheme: 'Greens',
    ramp: 'greens',
    note: 'Research-grade share, NDVI. Vegetation level uses YlGn-6 (see getClassPalette). Dark: reversed Greens with the floor lift.'
  },
  socioEconomic: {
    question: 'Socio-economic ingredient',
    kind: 'sequential',
    light: ['#eff3ff', '#bdd7e7', '#6baed6', '#3182bd', '#08519c'],
    dark: deriveDarkSequential(['#eff3ff', '#bdd7e7', '#6baed6', '#3182bd', '#08519c']),
    scheme: 'Blues',
    ramp: 'blues',
    note: 'Indicators, income. PuBu-5 is the alternate (extras.pubu).',
    extras: {pubu: ['#f1eef6', '#bdc9e1', '#74a9cf', '#2b8cbe', '#045a8d']}
  },
  access: {
    question: 'Access: time from an origin, reach',
    kind: 'sequential',
    light: ['#0c2c84', '#225ea8', '#1d91c0', '#41b6c4', '#7fcdbb', '#c7e9b4'],
    dark: deriveDarkSequential(['#0c2c84', '#225ea8', '#1d91c0', '#41b6c4', '#7fcdbb', '#c7e9b4']),
    scheme: 'YlGnBu',
    reverse: true,
    ramp: 'ylgnbu',
    note: 'Routing, isochrones, job-access, reachability. Near is the strongest class (class 0): darkest on light, brightest on dark. Decision-threshold classes.'
  },
  remoteness: {
    question: 'Remoteness: distance to the nearest (inverse of access)',
    kind: 'sequential',
    light: YLORRD_5,
    dark: MAGMA_DARK_5,
    scheme: 'YlOrRd',
    ramp: 'ylorrd',
    note: 'Nearest-facility. Far is the strongest class.'
  },
  load: {
    question: 'Load, frequency, importance',
    kind: 'sequential',
    light: YLORRD_5,
    dark: MAGMA_DARK_5,
    scheme: 'YlOrRd',
    ramp: 'ylorrd',
    note: 'YlOrRd / magma; draw the lowest class as a grey hairline (extras.hairline).',
    extras: {hairline: ['#c9ccd3']}
  },
  similarity: {
    question: 'Similarity',
    kind: 'sequential',
    light: ['#edf8fb', '#b2e2e2', '#66c2a4', '#2ca25f', '#006d2c'],
    dark: deriveDarkSequential(['#edf8fb', '#b2e2e2', '#66c2a4', '#2ca25f', '#006d2c']),
    scheme: 'BuGn',
    note: 'Similar places. "Unlike" is RdPu-5 (extras.unlike).',
    extras: {unlike: getClassPaletteHex('RdPu', 5)}
  },
  inequality: {
    question: 'Inequality within a unit',
    kind: 'sequential',
    light: ['#f2f0f7', '#cbc9e2', '#9e9ac8', '#756bb1', '#54278f'],
    dark: deriveDarkSequential(['#f2f0f7', '#cbc9e2', '#9e9ac8', '#756bb1', '#54278f']),
    scheme: 'Purples',
    ramp: 'purples',
    note: 'Vulnerability.'
  },
  rainfall: {
    question: 'Rainfall',
    kind: 'sequential',
    light: YLGNBU_8,
    dark: deriveDarkSequential(YLGNBU_8),
    scheme: 'YlGnBu',
    ramp: 'ylgnbu',
    note: 'NWS-like classes in mm: <10, 10-25, 25-50, 50-75, 75-100, 100-150, 150-200, >=200. Dark: reversed, wetter is brighter.'
  },
  elapsedTime: {
    question: 'Elapsed time, duration (earth)',
    kind: 'sequential',
    light: ['#d9f0e5', '#a6dcc9', '#6fc2c5', '#4a9cc4', '#5a6fb8', '#6a3f9a'],
    dark: ['#d9f0e5', '#a6dcc9', '#6fc2c5', '#4a9cc4', '#5a6fb8', '#8d6ad0'],
    note: 'Teal to violet. Drifter age, swath first-reach, lead time. The violet end is lifted on dark.'
  },
  waiting: {
    question: 'Waiting, dwell',
    kind: 'sequential',
    light: YLORRD_5,
    dark: MAGMA_DARK_5,
    scheme: 'YlOrRd',
    ramp: 'ylorrd',
    note: 'Movement stops, plus a hollow grey ring for the home range (extras.homeRing).',
    extras: {homeRing: ['#9a9890']}
  },
  vegetationChange: {
    question: 'Vegetation change',
    kind: 'diverging',
    light: ['#8c510a', '#d8b365', '#f6e8c3', '#f5f5f5', '#c7eae5', '#5ab4ac', '#01665e'],
    dark: deriveDarkDiverging([
      '#8c510a',
      '#d8b365',
      '#f6e8c3',
      '#f5f5f5',
      '#c7eae5',
      '#5ab4ac',
      '#01665e'
    ]),
    scheme: 'BrBG',
    ramp: 'brbg',
    midpointMeaning: '0 = no change; brown = loss, teal = gain',
    note: 'Veg and cell-pyramid change. BrBG-5 (extras.brbg5) for coarse change.',
    extras: {brbg5: ['#a6611a', '#dfc27d', '#f5f5f5', '#80cdc1', '#018571']}
  },
  deviation: {
    question: 'Deviation from expected (residual, observed minus expected, difference B minus A)',
    kind: 'diverging',
    light: ['#542788', '#998ec3', '#d8daeb', '#f7f7f7', '#fee0b6', '#f1a340', '#b35806'],
    dark: deriveDarkDiverging([
      '#542788',
      '#998ec3',
      '#d8daeb',
      '#f7f7f7',
      '#fee0b6',
      '#f1a340',
      '#b35806'
    ]),
    scheme: 'PuOr',
    ramp: 'puor',
    midpointMeaning: 'expected / reference; orange = above',
    note: 'Health residuals, nature share vs city, near-transit lift, method difference, raster minus exact. Name the midpoint in words. Flows exception: net flow, orange = more depart than arrive.'
  },
  zScore: {
    question: 'Z-scores, hot and cold, significance',
    kind: 'diverging',
    light: RDBU_7,
    dark: DARK_RDBU_7,
    scheme: 'RdBu',
    ramp: 'rdbu',
    midpointMeaning: 'not distinguishable from zero; red = high',
    note: 'Gi*, trend z, LISA. Esri bins at +-1.65/1.96/2.58 (GI_STAR_BREAKS). Dark: the authored dark-centre table, orange for high.'
  },
  coefficient: {
    question: 'Effect around a global estimate (GWR coefficient)',
    kind: 'diverging',
    light: RDBU_7,
    dark: DARK_RDBU_7,
    scheme: 'RdBu',
    ramp: 'rdbu',
    midpointMeaning: 'the global estimate (the claim); the legend says what blue means in words',
    note: 'Local relationships only.'
  },
  direction: {
    question: 'Direction, time of day, day of year',
    kind: 'cyclic',
    light: sampleCyclicHexes('romao', 8),
    dark: sampleCyclicHexes('romao', 8),
    ramp: 'romao',
    note: '8 classes here; ask for 12 with getRegistryColors. Use a ring legend. Aspect, hours, migration timing.'
  },
  party: {
    question: 'Party',
    kind: 'qualitative',
    light: ['#2166ac', '#b2182b'],
    dark: ['#2166ac', '#b2182b'],
    note: 'Election only: [Democratic blue, Republican red], never reversed, never in a frame that also uses RdBu z-scores.'
  },
  windSpeed: {
    question: 'Wind speed',
    kind: 'sequential',
    light: ['#c7e9b4', '#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#0c2c84'],
    dark: ['#3a2a6e', '#7a2a82', '#c43c75', '#f2704f', '#fdae61', '#fcf3b0'],
    note: 'NHC thresholds 17.5 / 32.9 / 42.7 / 49.4 m/s (+58.1). Wind-flow, jet.'
  },
  hurricane: {
    question: 'Hurricane class',
    kind: 'qualitative',
    light: HURRICANE_HEX.light,
    dark: HURRICANE_HEX.dark,
    note: 'TD, TS, C1 ... C5, ordered. Shared table HURRICANE_CLASS. Season, landfall, families.'
  },
  slope: {
    question: 'Slope (steep terrain)',
    kind: 'sequential',
    light: SLOPE,
    dark: liftLightnessFloor(SLOPE, 42.5),
    note: 'Classed at 25/30/35/40/45/50 degrees, transparent below 25. The avalanche hues are the convention, so dark keeps them and only lifts the darkest classes.'
  },
  slopeGentle: {
    question: 'Slope (gentle ground, fire terrain and site criteria)',
    kind: 'sequential',
    light: ['#dadaeb', '#bcbddc', '#9e9ac8', '#6a51a3', '#3f007d'],
    dark: deriveDarkSequential(['#dadaeb', '#bcbddc', '#9e9ac8', '#6a51a3', '#3f007d']),
    ramp: 'purples',
    note: 'Purples for gentle ground; never inferno for slope.'
  },
  visibility: {
    question: 'Visibility',
    kind: 'qualitative',
    light: ['#00000000', '#e69f00', '#1f2a4d96'],
    dark: ['#00000000', '#e69f00', '#1f2a4d96'],
    note: 'Visible (clear), marginal (orange hatch), hidden (indigo veil, alpha 150). Viewshed, horizon, sun.'
  },
  warnings: {
    question: 'Weather warnings',
    kind: 'qualitative',
    light: ['#d73027', '#f49628', '#26a69a'],
    dark: ['#d73027', '#f49628', '#26a69a'],
    note: 'NWS convention: tornado red, severe thunderstorm orange, flash flood teal. Storm stories.'
  }
};

/**
 * The colours of a registry question on a ground.
 *
 * Without `classCount` (or when it equals the entry's table length) the entry's own table is
 * returned. Otherwise the table is rebuilt at that count: from the published ColorBrewer table of
 * the entry's `scheme` ({@link getClassPalette}, honouring `reverse` and the ground's dark
 * variant), or, for entries without a scheme, by CIELAB interpolation of the entry's table
 * (cyclic entries are sampled around the circle from the ramp).
 *
 * @example
 * ```ts
 * const colors = getRegistryColors('access', ground); // near = strongest
 * const five = getRegistryColors('harm', ground, 5);
 * ```
 */
export function getRegistryColors(
  id: HueRegistryId,
  ground: MapGround,
  classCount?: number
): PaletteColor[] {
  const entry = HUE_REGISTRY[id];
  const table = entry[ground];
  if (classCount === undefined || classCount === table.length) {
    return table.map(hex => hexToRgba(hex));
  }
  if (entry.kind === 'cyclic' && entry.ramp) {
    return sampleCyclicHexes(entry.ramp, classCount).map(hex => hexToRgba(hex));
  }
  if (entry.scheme) {
    return getClassPalette(entry.scheme, classCount, {ground, reverse: entry.reverse});
  }
  return resampleHexTable(table, classCount).map(hex => hexToRgba(hex));
}
