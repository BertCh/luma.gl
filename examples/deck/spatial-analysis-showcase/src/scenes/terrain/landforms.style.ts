// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Pure (luma-free) pieces of the landforms story: the option state, the class tables of every
 * product (one object per product that the layer, the legend, the tooltip and the charts all read),
 * the legends, the sweep groups and the small formulas the card quotes.
 *
 * The questions and their tables:
 * - which landform: the "quiet ground" geomorphon table (the GRASS standard as the labelled
 *   "before"), nominal, ridge-warm to valley-cool;
 * - which way does the surface bend, how far above the window mean: the registry `deviation`
 *   PuOr table, middle class not drawn, orange = convex or above;
 * - at which size is the landform strongest: eight sequential classes (batlow, reversed);
 * - where two classifiers disagree: agree not drawn, disagree purple, opposite orange.
 */

import {
  getClassTableLegend,
  hexToRgba,
  makeClassTable,
  type ClassesLegendSpec
} from '../../cartography/class-table';
import {MAP_INK, NO_DATA_COLOR} from '../../cartography/hue-registry';
import type {ClassTable} from '../../cartography/types';
import {getClassColors} from '../../engine/ramps';
import type {LegendSpec} from '../scene';
import {
  GEOMORPHON_CODES,
  GEOMORPHON_LEGEND_ORDER,
  getCurvatureBreaks,
  getSignedLegend,
  getWeissLegend,
  makeAgreementTable,
  makeGeomorphonTable,
  makeSignedClassTable,
  makeWeissTable,
  DEV_BREAKS_SD,
  type TerrainGroundTone
} from './terrain-palettes';

export {GEOMORPHON_CODES, GEOMORPHON_LEGEND_ORDER};

/** RGB, 0-255. */
export type ClassColor = readonly [number, number, number];

/** Curvature kinds of `GPUTerrainCurvature`, plus the multi-radius ring product. */
export type CurvatureChoice =
  | 'profile'
  | 'plan'
  | 'tangential'
  | 'mean'
  | 'gaussian'
  | 'minimal'
  | 'maximal'
  | 'unsphericity'
  | 'difference'
  | 'horizontal-excess'
  | 'vertical-excess'
  | 'accumulation'
  | 'ring'
  | 'rotor'
  | 'laplacian'
  | 'ring-multi-radius';

/** Products of the scene. */
export type LandformProduct = 'geomorphons' | 'curvature' | 'position' | 'weiss';

/** What the position product shows: the signed height, the standardised height or the strongest scale. */
export type PositionProduct = 'tpi' | 'dev' | 'scale';

/** What the Weiss product shows: Weiss, geomorphons, or where they disagree. */
export type ClassifierView = 'weiss' | 'geomorphons' | 'agreement';

/** Option state of the landforms scene. */
export type LandformOptions = {
  product: LandformProduct;
  // Geomorphons
  palette: 'quiet' | 'grass';
  geomorphonRadius: number;
  geomorphonFlatAngle: number;
  geomorphonSkip: number;
  geomorphonComparison: 'anglev1' | 'anglev2' | 'anglev2-distance';
  geomorphonFlatDistance: number;
  showRays: boolean;
  compareRadius: boolean;
  // Curvature
  curvatureKind: 'profile' | 'plan';
  curvatureMoreKinds: 'none' | Exclude<CurvatureChoice, 'profile' | 'plan'>;
  curvatureMethod: 'evans-young' | 'zevenbergen-thorne' | 'florinsky';
  curvatureBorder: 'clamp' | 'nodata';
  flatGradient: number;
  zFactor: number;
  ringRadiusInner: number;
  ringRadiusOuter: number;
  ringGainInner: number;
  ringGainOuter: number;
  ringSquash: boolean;
  // Topographic position
  positionProduct: PositionProduct;
  scalePreset: 'fine' | 'landscape' | 'broad';
  scaleIndex: number;
  innerFraction: number;
  quantum: '4' | '64' | '256';
  // Weiss and agreement
  view: ClassifierView;
  weissSmall: number;
  weissLarge: number;
  weissStandardization: 'global' | 'local';
  weissThreshold: number;
  weissSlope: number;
  // Display
  underlay: boolean;
  opacity: number;
};

/** Ground meters per pixel at the central row of the alps-dem tile (display only). */
export const GROUND_PIXEL_METERS = 6.64;

/** Scale sets of the multi-scale topographic position, window radii in pixels (6.6 m each). */
export const SCALE_PRESETS: Record<LandformOptions['scalePreset'], readonly number[]> = {
  fine: [1, 2, 3, 4, 6, 8, 12, 16],
  landscape: [2, 4, 8, 16, 32, 64, 128, 256],
  broad: [8, 16, 32, 64, 128, 256, 384, 512]
};

/** The radii of the sweep: each doubles the last, from a rib to a whole mountain. */
export const SWEEP_RADII: readonly number[] = [6, 12, 24, 48, 96];

/** The short look-out the swipe of the scale step compares against, in pixels. */
export const COMPARE_RADIUS = 8;

// ---------------------------------------------------------------------------------------------
// Geomorphon classes
// ---------------------------------------------------------------------------------------------

type GeomorphonName = keyof typeof GEOMORPHON_CODES;

/** GRASS landform codes by minus count (row) and plus count (column); 0 marks impossible pairs. */
const FORM_TABLE: readonly (readonly number[])[] = [
  [1, 1, 1, 8, 8, 9, 9, 9, 10],
  [1, 1, 8, 8, 8, 9, 9, 9, 0],
  [1, 4, 6, 6, 7, 7, 9, 0, 0],
  [4, 4, 6, 6, 6, 7, 0, 0, 0],
  [4, 4, 5, 6, 6, 0, 0, 0, 0],
  [3, 3, 5, 5, 0, 0, 0, 0, 0],
  [3, 3, 3, 0, 0, 0, 0, 0, 0],
  [3, 3, 0, 0, 0, 0, 0, 0, 0],
  [2, 0, 0, 0, 0, 0, 0, 0, 0]
];

/** The display names of the ten forms by code (index 0 is the border). */
export const GEOMORPHON_NAMES: readonly string[] = [
  'Border',
  'Flat',
  'Peak',
  'Ridge',
  'Shoulder',
  'Spur',
  'Slope',
  'Hollow',
  'Footslope',
  'Valley',
  'Pit'
];

const rangeWords = (low: number, high: number) => (low === high ? `${low}` : `${low}-${high}`);

/**
 * How many of the eight rays fall away from the cell and how many rise above it, as the GRASS
 * table assigns each form: `falls` and `rises` are the ranges over every (falls, rises) pair that
 * gives the form. A bounding range, not a recipe: not every combination inside it is the form.
 */
export function getFormRayCounts(code: number): {
  falls: readonly [number, number];
  rises: readonly [number, number];
} {
  let fallsLow = 8;
  let fallsHigh = 0;
  let risesLow = 8;
  let risesHigh = 0;
  FORM_TABLE.forEach((row, falls) => {
    row.forEach((form, rises) => {
      if (form !== code) return;
      fallsLow = Math.min(fallsLow, falls);
      fallsHigh = Math.max(fallsHigh, falls);
      risesLow = Math.min(risesLow, rises);
      risesHigh = Math.max(risesHigh, rises);
    });
  });
  return {falls: [fallsLow, fallsHigh], rises: [risesLow, risesHigh]};
}

/** The form of a (falls, rises) pair of rays, 0 for an impossible pair. */
export function getFormCode(falls: number, rises: number): number {
  return FORM_TABLE[falls]?.[rises] ?? 0;
}

/** `5-7 fall, 0-2 rise`: the mini-profile words of a form for the legend. */
export function getFormProfileWords(code: number): string {
  const {falls, rises} = getFormRayCounts(code);
  if (falls[1] === 8) return 'all fall';
  if (rises[1] === 8) return 'all rise';
  return `${rangeWords(falls[0], falls[1])} fall, ${rangeWords(rises[0], rises[1])} rise`;
}

/** Groups of the sweep chart: which forms count as ridge-like, slope, valley-like and flat. */
export const SWEEP_GROUPS: readonly {id: string; label: string; codes: readonly number[]}[] = [
  {id: 'ridge', label: 'Ridge-like', codes: [2, 3, 4, 5]},
  {id: 'slope', label: 'Slope', codes: [6]},
  {id: 'valley', label: 'Valley-like', codes: [7, 8, 9, 10]},
  {id: 'flat', label: 'Flat', codes: [1]}
];

/** Peak and pit codes, which are single cells and are enlarged at low zoom. */
export const PEAK_CODE = GEOMORPHON_CODES.peak;
export const PIT_CODE = GEOMORPHON_CODES.pit;

/**
 * Cells around a peak or pit that take its colour at a zoom, so a single 6.6 m cell stays visible:
 * two below zoom 12.3 (a 5 x 5 block), one below 13 (3 x 3), none above.
 */
export function getDilationRadius(zoom: number): number {
  return zoom < 12.3 ? 2 : zoom < 13 ? 1 : 0;
}

/**
 * Rays read per cell by `GPUGeomorphons` at a search radius: the four straight rays take up to
 * `L - 1` cells; the four diagonal rays take steps of 1.41 cells, so they reach the same ground
 * distance in about 0.71 L steps. Same loop bounds as the shader.
 */
export function getReadsPerCell(searchRadius: number): number {
  const straight = Math.max(0, Math.ceil(searchRadius * 0.99999) - 1);
  const diagonal = Math.max(0, Math.ceil((searchRadius * 0.99999) / Math.SQRT2) - 1);
  return 4 * straight + 4 * diagonal;
}

// ---------------------------------------------------------------------------------------------
// Class tables
// ---------------------------------------------------------------------------------------------

/** The scale classes of the position product: the radii in pixels, one class each (code 1-8). */
export function getScaleLabels(radii: readonly number[]): string[] {
  return radii.map(radius => {
    const meters = radius * GROUND_PIXEL_METERS;
    return meters >= 1000 ? `${(meters / 1000).toFixed(1)} km` : `${Math.round(meters)} m`;
  });
}

/** Every nominal and fixed table of the story for one ground. */
export type LandformTables = {
  quiet: ClassTable;
  grass: ClassTable;
  weiss: ClassTable;
  agreement: ClassTable;
  scale: ClassTable;
};

/**
 * The characteristic-scale table: class = code, code 0 (no dominant scale) not drawn, codes 1-8 the
 * eight window radii on `batlow` reversed (small windows dark, large light) so the largest
 * landforms read as the lightest.
 */
export function makeScaleTable(radii: readonly number[], ground: TerrainGroundTone): ClassTable {
  const colors = getClassColors('batlow', radii.length, true, 190);
  return makeClassTable({
    breaks: radii.map((_, index) => index + 1),
    colors: [[0, 0, 0, 0], ...colors],
    transparent: [0],
    labels: ['No dominant scale', ...getScaleLabels(radii)],
    unit: 'm window radius',
    method: 'Radius at which the cell stands out most (DEVmax)',
    noData: {label: 'No dominant scale', color: NO_DATA_COLOR[ground]}
  });
}

/** The fixed tables of the story for a ground (the signed tables depend on the data, see below). */
export function makeLandformTables(
  ground: TerrainGroundTone,
  radii: readonly number[] = SCALE_PRESETS.landscape
): LandformTables {
  return {
    quiet: makeGeomorphonTable('quiet', ground),
    grass: makeGeomorphonTable('grass', ground),
    weiss: makeWeissTable(ground),
    agreement: makeAgreementTable(ground),
    scale: makeScaleTable(radii, ground)
  };
}

/** The curvature table at frozen breaks: PuOr, middle class not drawn, orange = convex. */
export function makeCurvatureTable(p98: number, ground: TerrainGroundTone): ClassTable {
  return makeSignedClassTable(getCurvatureBreaks(p98), ground, '1/m', {
    method: 'Breaks at fixed shares of the 98th percentile of |curvature|',
    format: value => value.toPrecision(2)
  });
}

/** The TPI table in metres at frozen breaks. */
export function makePositionTable(p98: number, ground: TerrainGroundTone): ClassTable {
  const rounded = Number(p98.toPrecision(1));
  return makeSignedClassTable(getCurvatureBreaks(rounded), ground, 'm', {
    method: 'Metres above or below the window mean; breaks frozen at the first scale shown',
    format: value => value.toFixed(value !== 0 && Math.abs(value) < 10 ? 1 : 0)
  });
}

/** The DEV table: standard deviations of the cell against its window. */
export function makeDeviationTable(ground: TerrainGroundTone): ClassTable {
  return makeSignedClassTable(DEV_BREAKS_SD, ground, 'standard deviations', {
    method: 'Height against the window mean in standard deviations of the window',
    format: value => value.toFixed(1)
  });
}

// ---------------------------------------------------------------------------------------------
// Swatches and legends
// ---------------------------------------------------------------------------------------------

type Rgba = readonly [number, number, number, number?];

/**
 * The colour a class swatch, chart segment or tooltip row shows: the table colour at full alpha,
 * except for classes the map draws nearly clear (slope, flat), which are mixed toward the muted
 * ink so the swatch can still be seen on a card.
 */
export function getSwatchColor(
  color: Rgba,
  ground: TerrainGroundTone
): [number, number, number, number] {
  const alpha = color[3] ?? 255;
  if (alpha > 140) return [color[0], color[1], color[2], 255];
  const ink = hexToRgba(MAP_INK[ground].inkMuted);
  const mix = 0.35;
  return [
    Math.round(color[0] * (1 - mix) + ink[0] * mix),
    Math.round(color[1] * (1 - mix) + ink[1] * mix),
    Math.round(color[2] * (1 - mix) + ink[2] * mix),
    255
  ];
}

/**
 * The key a signed table is frozen and stored under: the curvature kind, or the topographic
 * position measure (TPI breaks depend on the scale set, DEV breaks are fixed). Shared by the
 * scene's legends and the compute module.
 */
export function getSignedKey(
  state: Pick<
    LandformOptions,
    'product' | 'curvatureKind' | 'curvatureMoreKinds' | 'positionProduct' | 'scalePreset'
  >
): string {
  if (state.product === 'curvature') return `curvature:${getCurvatureKind(state)}`;
  return state.positionProduct === 'tpi' ? `position:tpi:${state.scalePreset}` : 'position:dev';
}

/** The note of the geomorphon legend about the single-cell classes. */
export const DILATION_NOTE = 'Peaks and pits are enlarged below zoom 13';

/**
 * The geomorphon legend in ridge-to-valley order with the share of cells and the mini-profile
 * words of each form, the border as the no-data entry. Same table as the layer. `shares` is by form
 * code (index 0 unused).
 */
export function getLandformLegend(
  table: ClassTable,
  ground: TerrainGroundTone,
  options: {shares?: readonly number[]; note?: string} = {}
): LegendSpec {
  const entries = GEOMORPHON_LEGEND_ORDER.map(name => {
    const code = GEOMORPHON_CODES[name as GeomorphonName];
    const share = options.shares?.[code];
    const words = getFormProfileWords(code);
    const shareText =
      share === undefined ? '' : `${(share * 100).toFixed(share < 0.1 ? 1 : 0)} % · `;
    return {
      color: getSwatchColor(table.colors[code], ground),
      label: GEOMORPHON_NAMES[code],
      detail: `${shareText}${words}`
    };
  });
  entries.push({
    color: [...NO_DATA_COLOR[ground]] as unknown as [number, number, number, number],
    label: table.noData?.label ?? 'Border (no full window)',
    detail: 'not drawn'
  });
  return {
    kind: 'categories',
    id: 'landform',
    title: 'Landform class (geomorphon)',
    layout: 'list',
    interactive: true,
    entries,
    note: options.note ?? `${table.method}. ${DILATION_NOTE}.`
  };
}

/** The legend of the scale classes (code per window radius), with the no-data entry last. */
export function getScaleLegend(
  table: ClassTable,
  ground: TerrainGroundTone,
  shares?: readonly number[]
): LegendSpec {
  const labels = table.labels ?? [];
  const entries = table.colors.slice(1).map((color, index) => ({
    color: getSwatchColor(color, ground),
    label: labels[index + 1] ?? '',
    ...(shares?.[index + 1] !== undefined
      ? {detail: `${((shares[index + 1] ?? 0) * 100).toFixed(1)} %`}
      : {})
  }));
  entries.push({
    color: [...NO_DATA_COLOR[ground]] as unknown as [number, number, number, number],
    label: table.noData?.label ?? 'No dominant scale',
    detail: 'not drawn'
  } as (typeof entries)[number]);
  return {
    kind: 'categories',
    id: 'scale',
    title: 'Window radius of the strongest landform (m)',
    layout: 'list',
    interactive: true,
    entries,
    note: table.method
  };
}

/** A signed legend (curvature, TPI, DEV) with the words for both ends and the neutral class. */
export function getSignedProductLegend(
  table: ClassTable,
  title: string,
  words: {high: string; low: string}
): ClassesLegendSpec {
  return getSignedLegend(table, title, {
    id: 'signed',
    interactive: true,
    neutralNote: `Orange = ${words.high}; purple = ${words.low}; the middle class (no clear departure) is not drawn`
  });
}

/** The agreement legend: three classes, shares when known. */
export function getAgreementLegend(
  table: ClassTable,
  shares?: readonly number[]
): ClassesLegendSpec {
  return getClassTableLegend(table, {
    title: 'Do Weiss and geomorphons agree?',
    id: 'agreement',
    layout: 'list',
    ...(shares ? {counts: shares.map(share => Math.round(share * 1000) / 10)} : {}),
    note: 'Both classifiers read as convex, neutral or concave. Opposite = one says ridge-like, the other valley-like. Percent of cells.'
  });
}

/** The Weiss legend (3 x 3 key). */
export function getWeissKey(ground: TerrainGroundTone): LegendSpec {
  return getWeissLegend(ground);
}

// ---------------------------------------------------------------------------------------------
// Curvature kinds
// ---------------------------------------------------------------------------------------------

/** One-line meaning of the other curvature kinds (option help and tooltips). */
export const OTHER_CURVATURE_KINDS: readonly {
  value: Exclude<CurvatureChoice, 'profile' | 'plan'>;
  label: string;
  help: string;
}[] = [
  {
    value: 'tangential',
    label: 'Tangential',
    help: 'Plan curvature scaled by the slope; zero on flats.'
  },
  {
    value: 'mean',
    label: 'Mean',
    help: 'Average of the two principal curvatures; independent of direction.'
  },
  {
    value: 'gaussian',
    label: 'Gaussian',
    help: 'Product of the principal curvatures: positive on bowls and domes, negative on saddles.'
  },
  {
    value: 'minimal',
    label: 'Minimal',
    help: 'The smaller principal curvature (the most concave direction).'
  },
  {
    value: 'maximal',
    label: 'Maximal',
    help: 'The larger principal curvature (the most convex direction).'
  },
  {
    value: 'unsphericity',
    label: 'Unsphericity',
    help: 'How far the surface is from a sphere (0 at a dome or bowl).'
  },
  {
    value: 'difference',
    label: 'Difference',
    help: 'Half the difference between profile and tangential curvature.'
  },
  {
    value: 'horizontal-excess',
    label: 'Horizontal excess',
    help: 'Plan minus mean curvature; the departure from a sphere in the horizontal.'
  },
  {
    value: 'vertical-excess',
    label: 'Vertical excess',
    help: 'Profile minus mean curvature; the departure from a sphere in the vertical.'
  },
  {
    value: 'accumulation',
    label: 'Accumulation',
    help: 'Product of profile and plan curvature: where flow both converges and decelerates.'
  },
  {value: 'ring', label: 'Ring (excess product)', help: 'Product of the two excess curvatures.'},
  {value: 'rotor', label: 'Rotor', help: 'Twisting of the flow lines.'},
  {
    value: 'laplacian',
    label: 'Laplacian',
    help: 'Sum of the second derivatives; the classic convexity measure.'
  },
  {
    value: 'ring-multi-radius',
    label: 'Ring curvature, multi-radius (mt-image)',
    help: 'Weighted sum over rings of samples at the chosen radii: emphasises ridge and valley lines at the ring scales.'
  }
];

/** The name of a curvature kind for titles and tooltips. */
export function getCurvatureName(kind: CurvatureChoice): string {
  if (kind === 'profile') return 'Profile curvature';
  if (kind === 'plan') return 'Plan curvature';
  return OTHER_CURVATURE_KINDS.find(entry => entry.value === kind)?.label ?? 'Curvature';
}

/** The curvature kind the options select: an expert kind wins over the profile / plan chips. */
export function getCurvatureKind(
  state: Pick<LandformOptions, 'curvatureKind' | 'curvatureMoreKinds'>
): CurvatureChoice {
  return state.curvatureMoreKinds !== 'none' ? state.curvatureMoreKinds : state.curvatureKind;
}
