// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Every class table of the terrain chapter (chapter sheet section 4), each as one object that the
 * layer, the legend, the tooltip and the histogram all read. A layer spreads
 * `getClassTableLayerProps(table)`; the legend is `getClassTableLegend(table, ...)` or one of the
 * `get*Legend` helpers here, so the two cannot drift apart. Ground is always the tone that is
 * actually shown (`ctx.ground()`), never the page theme.
 *
 * The questions and their hues (one hue answers one question, as in `HUE_REGISTRY`):
 * - elevation: the pale Alpine hypsometric tint of the relief ground, plus glacier;
 * - how steep: registry `slope`, classed at the decision thresholds, transparent below 25 degrees;
 * - which way it faces: cyclic `romao`, faded out on flat ground;
 * - above or below its surroundings: registry `deviation` (PuOr), middle class transparent;
 * - which landform: a quiet nominal table (and the GRASS standard as the labelled "before");
 * - how rugged: PuRd, deliberately not the slope hues;
 * - what can be seen: the visibility triad (clear, marginal hatch, indigo veil);
 * - how many lookouts see a cell: YlOrRd with the veil as class 0;
 * - how long the sun stays, how much it delivers, how high the skyline stands.
 */

import {
  DARK_GROUND_MINIMUM_LIGHTNESS,
  getClassPalette,
  getClassTableLegend,
  hexToRgba,
  liftLightnessFloor,
  makeClassTable,
  resampleHexTable,
  type ClassesLegendSpec
} from '../../cartography/class-table';
import {
  getRegistryColors,
  HUE_REGISTRY,
  MAP_INK,
  NO_DATA_COLOR,
  TERRAIN_EYE_GOLD,
  type MapGround
} from '../../cartography/hue-registry';
import type {ClassColor, ClassTable} from '../../cartography/types';
import type {PaletteColor} from '../../engine/ramps';
import {sampleRamp} from '../../engine/ramps';
import {ALPINE_TINTS, CONTOUR_STYLE, getReliefTintColor} from '../../engine/relief';
import type {LegendSpec} from '../scene';

/** The tone of the ground a table is drawn on: `ctx.ground()`. */
export type TerrainGroundTone = MapGround;

type LegendOptions = {
  id?: string;
  counts?: readonly number[];
  interactive?: boolean;
  layout?: 'bar' | 'list';
  note?: string;
  histogram?: readonly number[];
};

const withAlpha = (
  color: readonly [number, number, number, number?],
  alpha: number
): PaletteColor => [color[0], color[1], color[2], alpha];

// ---------------------------------------------------------------------------------------------
// Elevation tint (the ground itself)
// ---------------------------------------------------------------------------------------------

/** Class breaks of the elevation legend in metres: the ALPINE_TINTS stops up to 3,800 m. */
export const ELEVATION_BREAKS: readonly number[] = ALPINE_TINTS.stops
  .slice(0, -1)
  .map(stop => stop.elevation);

/** The glacier colour of the relief ground, `#D5E6EF`. */
export const GLACIER_COLOR = '#D5E6EF';

/** Elevation at which each legend class samples the tint (the middle of its band). */
const ELEVATION_SAMPLES: readonly number[] = [1400, 1700, 2100, 2550, 3050, 3550, 4200];

const FLAT_TONE: Record<TerrainGroundTone, readonly [number, number, number]> = {
  light: [0xf3, 0xef, 0xe6],
  dark: [0x27, 0x2d, 0x38]
};

/**
 * The elevation tint classes as the reader sees them on flat ground: the paper (or dark) tone
 * multiplied by the tint at the relief's `tintStrength`, so the swatches match the ground rather
 * than the saturated tint table. Seven classes: below 1,500 m, then the bands between the
 * ALPINE_TINTS stops, and 3,800 m and above.
 */
export function makeElevationTable(ground: TerrainGroundTone, tintStrength = 0.3): ClassTable {
  const flat = FLAT_TONE[ground];
  const colors: ClassColor[] = ELEVATION_SAMPLES.map(elevation => {
    const tint = getReliefTintColor('alpine', elevation);
    return [0, 1, 2].map(channel =>
      Math.round(flat[channel] * (1 - tintStrength + (tintStrength * tint[channel]) / 255))
    ) as unknown as ClassColor;
  });
  return makeClassTable({
    breaks: ELEVATION_BREAKS,
    colors: colors.map(color => [color[0], color[1], color[2], 255] as const),
    unit: 'm',
    method: 'Pale hypsometric tint under the hillshade',
    noData: {color: hexToRgba(GLACIER_COLOR), label: 'Glacier'}
  });
}

/** The elevation tint legend (`classes`, unit metres, glacier swatch in the no-data slot). */
export function getElevationLegend(
  ground: TerrainGroundTone,
  options: LegendOptions = {}
): ClassesLegendSpec {
  return getClassTableLegend(makeElevationTable(ground), {
    title: 'Elevation (m above sea level)',
    layout: 'bar',
    ...options
  });
}

/** `[{color, label}]` of the elevation tint and the glacier, for a custom swatch list. */
export function getElevationLegendEntries(
  ground: TerrainGroundTone
): {color: readonly [number, number, number, number?]; label: string}[] {
  const table = makeElevationTable(ground);
  const labels = table.labels ?? [
    '< 1,500',
    '1,500-1,900',
    '1,900-2,300',
    '2,300-2,800',
    '2,800-3,300',
    '3,300-3,800',
    '3,800 +'
  ];
  return [
    ...table.colors.map((color, index) => ({color, label: `${labels[index]} m`})),
    {color: hexToRgba(GLACIER_COLOR), label: 'Glacier'}
  ];
}

// ---------------------------------------------------------------------------------------------
// Slope (steep terrain)
// ---------------------------------------------------------------------------------------------

/** Slope class breaks in degrees (avalanche practice: 30 degrees is the decision threshold). */
export const SLOPE_BREAKS: readonly number[] = [25, 30, 35, 40, 45, 50];

/**
 * The slope table: registry `slope` (`#FEE391 ... #1B1B3A`), transparent below 25 degrees, drawn
 * at alpha 0.72 over the relief. The method says why the breaks are where they are.
 */
export function makeSlopeTable(ground: TerrainGroundTone, alpha = 184): ClassTable {
  const steep = getRegistryColors('slope', ground);
  return makeClassTable({
    breaks: SLOPE_BREAKS,
    colors: [withAlpha(steep[0], 0), ...steep.map(color => withAlpha(color, alpha))],
    transparent: [0],
    unit: 'degrees',
    extent: [0, 90],
    method: 'Decision thresholds (avalanche practice)',
    noData: {label: 'No data', color: NO_DATA_COLOR[ground]},
    format: value => `${value}°`
  });
}

/** The slope legend; pass `counts` (cells per class) to show shares. */
export function getSlopeLegend(table: ClassTable, options: LegendOptions = {}): ClassesLegendSpec {
  return getClassTableLegend(table, {title: 'Slope', id: 'slope', ...options});
}

/** The continuous slope "before" ramp: `colormap`, range and the transparent floor. */
export const SLOPE_CONTINUOUS = {
  ramp: 'ylorbr',
  domain: [0, 70],
  transparentBelowDegrees: 15
} as const;

// ---------------------------------------------------------------------------------------------
// Aspect (which way a slope faces)
// ---------------------------------------------------------------------------------------------

/** The cyclic ramp of aspect (north at the top of the ring, clockwise). */
export const ASPECT_RAMP = 'romao';

/** Aspect fades with the slope: alpha 0 at or below `flatDegrees`, 1 at or above `fullDegrees`. */
export const ASPECT_SLOPE_ALPHA = {flatDegrees: 5, fullDegrees: 25} as const;

/** Opacity of an aspect colour on a slope of `slopeDegrees` (the rule above, linear between). */
export function getAspectAlpha(slopeDegrees: number): number {
  const {flatDegrees, fullDegrees} = ASPECT_SLOPE_ALPHA;
  return Math.min(1, Math.max(0, (slopeDegrees - flatDegrees) / (fullDegrees - flatDegrees)));
}

/** The `romao` colour of an aspect (degrees clockwise from north), for tooltips and the rose. */
export function getAspectColor(aspectDegrees: number): [number, number, number] {
  return sampleRamp(ASPECT_RAMP, (((aspectDegrees % 360) + 360) % 360) / 360);
}

/** Aspect legend: a cyclic ring N E S W, with the flat-ground rule in the note. */
export function getAspectLegend(): LegendSpec {
  return {
    kind: 'cyclic',
    title: 'Aspect (direction the slope faces)',
    ramp: ASPECT_RAMP,
    labels: ['N', 'E', 'S', 'W'],
    note: `Flat ground (under ${ASPECT_SLOPE_ALPHA.flatDegrees}°) is not coloured: its aspect is noise.`
  };
}

// ---------------------------------------------------------------------------------------------
// Signed height against the surroundings (TPI, DEV, SLRM, MSRM, curvature)
// ---------------------------------------------------------------------------------------------

/** TPI classes in metres for the explore product: 5 classes, middle transparent. */
export const TPI_BREAKS_METERS: readonly number[] = [-6, -2, 2, 6];

/** DEV classes in standard deviations (7 classes). */
export const DEV_BREAKS_SD: readonly number[] = [-2, -1, -0.5, 0.5, 1, 2];

/**
 * Curvature breaks at +-0.25, 0.6 and 1.0 of the p98 of |value| (7 classes), to be computed once
 * at step entry and frozen.
 */
export function getCurvatureBreaks(p98: number): number[] {
  return [-1, -0.6, -0.25, 0.25, 0.6, 1].map(share => Number((share * p98).toPrecision(6)));
}

/**
 * A signed class table: registry `deviation` PuOr (purple below the surroundings, orange above,
 * "orange = convex / higher"), the middle class transparent so the relief shows through.
 * `breaks` must give 5 or 7 classes (4 or 6 breaks, symmetric about the neutral class).
 *
 * @param unit The legend unit ("m", "standard deviations", "1/m").
 */
export function makeSignedClassTable(
  breaks: readonly number[],
  ground: TerrainGroundTone,
  unit: string,
  options: {
    alpha?: number;
    method?: string;
    labels?: readonly string[];
    format?: (value: number) => string;
  } = {}
): ClassTable {
  const classCount = breaks.length + 1;
  const colors = getRegistryColors('deviation', ground, classCount);
  const middle = Math.floor(classCount / 2);
  const alpha = options.alpha ?? 204;
  return makeClassTable({
    breaks,
    colors: colors.map((color, index) => withAlpha(color, index === middle ? 0 : alpha)),
    transparent: [middle],
    unit,
    method: options.method ?? 'Purple below, orange above; middle class not drawn',
    ...(options.labels ? {labels: options.labels} : {}),
    ...(options.format ? {format: options.format} : {}),
    noData: {label: 'No data', color: NO_DATA_COLOR[ground]}
  });
}

/** Legend of a signed table; the neutral class is named by `HUE_REGISTRY.deviation.midpointMeaning`. */
export function getSignedLegend(
  table: ClassTable,
  title: string,
  options: LegendOptions & {neutralNote?: string} = {}
): ClassesLegendSpec {
  const {neutralNote, ...rest} = options;
  return getClassTableLegend(table, {
    title,
    note: neutralNote ?? `${HUE_REGISTRY.deviation.midpointMeaning}; middle class not drawn`,
    ...rest
  });
}

// ---------------------------------------------------------------------------------------------
// Landforms
// ---------------------------------------------------------------------------------------------

/** Geomorphon codes written by `GPUGeomorphons` (0 = no full window). */
export const GEOMORPHON_CODES = {
  flat: 1,
  peak: 2,
  ridge: 3,
  shoulder: 4,
  spur: 5,
  slope: 6,
  hollow: 7,
  footslope: 8,
  valley: 9,
  pit: 10
} as const;

/** Legend order of the ten geomorphons: ridge-warm to valley-cool, then flat. */
export const GEOMORPHON_LEGEND_ORDER: readonly (keyof typeof GEOMORPHON_CODES)[] = [
  'peak',
  'ridge',
  'shoulder',
  'spur',
  'slope',
  'flat',
  'hollow',
  'footslope',
  'valley',
  'pit'
];

const GEOMORPHON_LABELS: Record<keyof typeof GEOMORPHON_CODES, string> = {
  flat: 'Flat',
  peak: 'Peak',
  ridge: 'Ridge',
  shoulder: 'Shoulder',
  spur: 'Spur',
  slope: 'Slope',
  hollow: 'Hollow',
  footslope: 'Footslope',
  valley: 'Valley',
  pit: 'Pit'
};

/** The "quiet ground" geomorphon colours (`[hex, alpha 0-255]`), light ground. */
const QUIET_GEOMORPHONS_LIGHT: Record<keyof typeof GEOMORPHON_CODES, readonly [string, number]> = {
  peak: ['#67001F', 230],
  ridge: ['#B2182B', 255],
  shoulder: ['#EF8A62', 255],
  spur: ['#FDDBC7', 255],
  slope: ['#F4EFE6', 64],
  flat: ['#D9D4C4', 128],
  hollow: ['#D1E5F0', 255],
  footslope: ['#92C5DE', 255],
  valley: ['#2166AC', 255],
  pit: ['#053061', 230]
};

/** The same classes lifted for the dark ground (peaks and pits bright, slopes nearly clear). */
const QUIET_GEOMORPHONS_DARK: Record<keyof typeof GEOMORPHON_CODES, readonly [string, number]> = {
  peak: ['#FF8A94', 230],
  ridge: ['#E8505F', 255],
  shoulder: ['#F4936B', 255],
  spur: ['#F6C5A8', 230],
  slope: ['#C9CFD8', 51],
  flat: ['#8E96A3', 102],
  hollow: ['#A9CBE3', 230],
  footslope: ['#6FB0DB', 255],
  valley: ['#3F8FD9', 255],
  pit: ['#8CC4FF', 230]
};

/** The GRASS r.geomorphon standard colours, the labelled "before" of the quiet table. */
const GRASS_GEOMORPHONS: Record<keyof typeof GEOMORPHON_CODES, readonly [string, number]> = {
  flat: ['#DCDCDC', 255],
  peak: ['#380000', 255],
  ridge: ['#C80000', 255],
  shoulder: ['#FF5014', 255],
  spur: ['#FAD23C', 255],
  slope: ['#FFFF3C', 255],
  hollow: ['#B4E614', 255],
  footslope: ['#3CFA96', 255],
  valley: ['#0000FF', 255],
  pit: ['#000038', 255]
};

/**
 * The geomorphon table. The class index is the form code itself (`breaks` 1 to 10, so class 0 is
 * "no full window", transparent) and the layer needs no remap. `quiet` is the chapter table
 * (warm ridges, cool valleys, flat and slope nearly clear so the relief reads through); `grass`
 * is the GRASS standard.
 */
export function makeGeomorphonTable(
  kind: 'quiet' | 'grass',
  ground: TerrainGroundTone
): ClassTable {
  const source =
    kind === 'grass'
      ? GRASS_GEOMORPHONS
      : ground === 'dark'
        ? QUIET_GEOMORPHONS_DARK
        : QUIET_GEOMORPHONS_LIGHT;
  const names = Object.keys(GEOMORPHON_CODES) as (keyof typeof GEOMORPHON_CODES)[];
  const byCode = [...names].sort((a, b) => GEOMORPHON_CODES[a] - GEOMORPHON_CODES[b]);
  return makeClassTable({
    breaks: byCode.map(name => GEOMORPHON_CODES[name]),
    colors: [[0, 0, 0, 0], ...byCode.map(name => hexToRgba(source[name][0], source[name][1]))],
    transparent: [0],
    labels: ['No full window', ...byCode.map(name => GEOMORPHON_LABELS[name])],
    unit: 'landform class',
    method: kind === 'grass' ? 'GRASS standard colours' : 'Quiet ground: warm ridges, cool valleys',
    noData: {label: 'Border (no full window)', color: NO_DATA_COLOR[ground]}
  });
}

/**
 * The geomorphon legend in ridge-to-valley order (the table itself is in code order). Pass `shares`
 * (0-1, by form code 1-10) to show the share of cells per class.
 */
export function getGeomorphonLegend(
  table: ClassTable,
  options: {shares?: readonly number[]; note?: string} = {}
): LegendSpec {
  return {
    kind: 'categories',
    title: 'Landform (geomorphon)',
    layout: 'list',
    entries: GEOMORPHON_LEGEND_ORDER.map(name => {
      const code = GEOMORPHON_CODES[name];
      const share = options.shares?.[code];
      return {
        color: table.colors[code],
        label: GEOMORPHON_LABELS[name],
        ...(share !== undefined ? {detail: `${(share * 100).toFixed(share < 0.1 ? 1 : 0)} %`} : {})
      };
    }),
    note: options.note ?? table.method
  };
}

/** Weiss class names in `GPU_TERRAIN_WEISS_LANDFORMS` code order (1-10). */
export const WEISS_LABELS: readonly string[] = [
  'Canyon',
  'Midslope drainage',
  'Upland drainage',
  'U-shaped valley',
  'Plain',
  'Open slope',
  'Upper slope',
  'Local ridge',
  'Midslope ridge',
  'Mountain top'
];

const WEISS_LIGHT: readonly [string, number][] = [
  ['#08519C', 255],
  ['#4A90C2', 255],
  ['#9ECAE1', 255],
  ['#3A9A8F', 255],
  ['#ECE6C8', 255],
  ['#D9C590', 89],
  ['#C9954A', 255],
  ['#F4A582', 255],
  ['#D6604D', 255],
  ['#8C1D18', 255]
];

const WEISS_DARK: readonly [string, number][] = [
  ['#5BA5E8', 255],
  ['#79B6E6', 255],
  ['#B5D8F0', 255],
  ['#4DBBAA', 255],
  ['#D8D2B0', 128],
  ['#CDB77F', 89],
  ['#E0AE5C', 255],
  ['#F7B396', 255],
  ['#EE7B66', 255],
  ['#FF5A48', 255]
];

/** The Weiss (2001) table: class index = class code 1-10, class 0 (invalid) transparent. */
export function makeWeissTable(ground: TerrainGroundTone): ClassTable {
  const source = ground === 'dark' ? WEISS_DARK : WEISS_LIGHT;
  return makeClassTable({
    breaks: source.map((_, index) => index + 1),
    colors: [[0, 0, 0, 0], ...source.map(([hex, alpha]) => hexToRgba(hex, alpha))],
    transparent: [0],
    labels: ['No data', ...WEISS_LABELS],
    unit: 'landform class',
    method: 'Weiss (2001): position at a small and a large scale',
    noData: {label: 'No data', color: NO_DATA_COLOR[ground]}
  });
}

/** The 3 x 3 Weiss key: rows the small-scale position (low, mid, high), columns the large scale. */
export function getWeissLegend(ground: TerrainGroundTone): LegendSpec {
  const source = ground === 'dark' ? WEISS_DARK : WEISS_LIGHT;
  // Open slope shares the middle cell with plain: the plain colour is the key's.
  const order = [0, 1, 2, 3, 4, 6, 7, 8, 9];
  return {
    kind: 'matrix',
    title: 'Landform (Weiss)',
    rows: ['Low', 'Mid', 'High'],
    columns: ['Low', 'Mid', 'High'],
    rowTitle: 'Position, small scale',
    columnTitle: 'Position, large scale',
    colors: order.map(index => hexToRgba(source[index][0])),
    note: 'Middle cell: plain, or open slope where steeper than the threshold'
  };
}

/** Agreement of two classifiers: agree transparent, disagree purple, opposite (convex vs concave) orange. */
export function makeAgreementTable(ground: TerrainGroundTone): ClassTable {
  const dark = ground === 'dark';
  return makeClassTable({
    breaks: [1, 2],
    colors: [
      [0, 0, 0, 0],
      hexToRgba(dark ? '#B58AD6' : '#7B3294', 204),
      hexToRgba(dark ? '#FF9B45' : '#E66101', 230)
    ],
    transparent: [0],
    labels: ['Agree', 'Disagree', 'Opposite'],
    unit: 'agreement',
    method: 'Opposite: one says ridge-like, the other valley-like',
    noData: {label: 'No data', color: NO_DATA_COLOR[ground]}
  });
}

// ---------------------------------------------------------------------------------------------
// Ruggedness
// ---------------------------------------------------------------------------------------------

/**
 * Quantile breaks of a value array: the value at each probability (default p50, p75, p90, p98),
 * ignoring non-finite values. Large arrays are sampled evenly (at most about 500,000 values) so
 * the sort stays cheap; compute once per product and radius and freeze the result for the step.
 */
export function getQuantileBreaks(
  values: ArrayLike<number>,
  probabilities: readonly number[] = [0.5, 0.75, 0.9, 0.98]
): number[] {
  const stride = Math.max(1, Math.floor(values.length / 500_000));
  const sample: number[] = [];
  for (let index = 0; index < values.length; index += stride) {
    const value = values[index];
    if (Number.isFinite(value)) sample.push(value);
  }
  sample.sort((a, b) => a - b);
  if (!sample.length) return probabilities.map(() => 0);
  return probabilities.map(probability => {
    const position = Math.min(sample.length - 1, Math.max(0, probability * (sample.length - 1)));
    const low = Math.floor(position);
    const high = Math.min(sample.length - 1, low + 1);
    return sample[low] + (sample[high] - sample[low]) * (position - low);
  });
}

/**
 * The ruggedness table (TRI, VRM): PuRd-5, lowest class transparent. PuRd rather than the slope
 * hues, because a magnitude that must not read as slope: yellow-red-purple belongs to avalanche
 * terrain. `breaks` are four quantile breaks (see {@link getQuantileBreaks}).
 *
 * @param unit `'m'` for TRI, `'index 0-1'` for VRM.
 */
export function makeRuggednessTable(
  breaks: readonly number[],
  ground: TerrainGroundTone,
  unit: string,
  options: {alpha?: number; format?: (value: number) => string} = {}
): ClassTable {
  const colors = getClassPalette('PuRd', 5, {ground});
  return makeClassTable({
    breaks,
    colors: colors.map((color, index) =>
      withAlpha(color, index === 0 ? 0 : (options.alpha ?? 184))
    ),
    transparent: [0],
    unit,
    method: 'Quantile classes (median, 75th, 90th and 98th percentile of this tile)',
    ...(options.format ? {format: options.format} : {}),
    noData: {label: 'No data', color: NO_DATA_COLOR[ground]}
  });
}

// ---------------------------------------------------------------------------------------------
// Visibility, observers and lookouts
// ---------------------------------------------------------------------------------------------

/** Class indices of the visibility raster: the value a viewshed layer carries per cell. */
export const VISIBILITY_CLASS = {
  visible: 0,
  marginal: 1,
  hidden: 2,
  outOfRange: 3
} as const;

/** The hatch colour of the marginal class and its geometry. */
export const VISIBILITY_HATCH = {
  color: [230, 159, 0, 255],
  spacingPixels: 4,
  widthPixels: 1.2
} as const;

/**
 * The visibility triad as a raster class table (`breaks` 1, 2, 3): visible is transparent, marginal
 * is `#E69F00` and hatched, hidden is the indigo veil (`#1F2A4D` alpha 150; `#05070F` alpha 175 on
 * the dark ground) and out of range is transparent. Veil what is hidden, leave what you see clear.
 */
export function makeVisibilityTable(ground: TerrainGroundTone): ClassTable {
  const veil = ground === 'dark' ? hexToRgba('#05070F', 175) : hexToRgba('#1F2A4D', 150);
  return makeClassTable({
    breaks: [1, 2, 3],
    colors: [[0, 0, 0, 0], hexToRgba('#E69F00', 40), veil, [0, 0, 0, 0]],
    transparent: [0, 3],
    hatched: [1],
    labels: ['Visible', 'Marginal', 'Hidden', 'Out of range'],
    unit: 'visibility class',
    method: 'Marginal: within the vertical tolerance of the sight line',
    noData: {label: 'No data', color: NO_DATA_COLOR[ground], hatched: true}
  });
}

/**
 * Layer props for a viewshed raster layer: the table's classes, the marginal hatch and the 1 px
 * visible/hidden class edge (`#0E1530` at 0.8).
 */
export function getVisibilityLayerProps(table: ClassTable) {
  return {
    classBreaks: [...table.breaks],
    classColors: table.colors.map(
      color => [color[0], color[1], color[2], color[3] ?? 255] as PaletteColor
    ),
    hatchClasses: [...(table.hatched ?? [])],
    hatchColor: VISIBILITY_HATCH.color,
    hatchSpacingPixels: VISIBILITY_HATCH.spacingPixels,
    hatchWidthPixels: VISIBILITY_HATCH.widthPixels,
    outlineClasses: {color: [14, 21, 48, 204] as const, widthPixels: 1}
  };
}

/** The visibility legend: classes with the hatched marginal swatch and an out-of-range note. */
export function getVisibilityLegend(
  table: ClassTable,
  options: LegendOptions = {}
): ClassesLegendSpec {
  return getClassTableLegend(table, {
    title: 'Visibility',
    id: 'visibility',
    layout: 'list',
    ...options
  });
}

/**
 * "Mark visible instead": the wrong-way comparison that paints what is seen gold (alpha 0.45) and
 * clears the hidden class. Same class indices as {@link makeVisibilityTable}.
 */
export function makeMarkVisibleTable(ground: TerrainGroundTone): ClassTable {
  return makeClassTable({
    breaks: [1, 2, 3],
    colors: [
      hexToRgba(TERRAIN_EYE_GOLD, 115),
      hexToRgba('#E69F00', 40),
      [0, 0, 0, 0],
      [0, 0, 0, 0]
    ],
    transparent: [2, 3],
    hatched: [1],
    labels: ['Visible', 'Marginal', 'Hidden', 'Out of range'],
    unit: 'visibility class',
    method: 'The paint-the-visible map: what you see is painted, the rest is left bare',
    noData: {label: 'No data', color: NO_DATA_COLOR[ground], hatched: true}
  });
}

/** Colours of the visibility symbols (peaks, sight-line samples): visible, marginal, hidden. */
export function getVisibilitySymbolColors(ground: TerrainGroundTone) {
  const dark = ground === 'dark';
  const hex = {
    visible: dark ? '#56B4E9' : '#0072B2',
    marginal: '#E69F00',
    hidden: dark ? '#9AA3AF' : '#6B7280'
  } as const;
  return {
    hex,
    visible: hexToRgba(hex.visible),
    marginal: hexToRgba(hex.marginal),
    hidden: hexToRgba(hex.hidden),
    /** Class edge ink for the symbols' outlines. */
    outline: hexToRgba(dark ? '#05070F' : '#0E1530')
  };
}

/** The legend of the visibility symbols: filled, half-filled ring, hollow ring. */
export function getVisibilitySymbolLegend(ground: TerrainGroundTone): LegendSpec {
  const colors = getVisibilitySymbolColors(ground);
  return {
    kind: 'categories',
    title: 'Peak visibility',
    entries: [
      {color: colors.visible, label: 'Visible', shape: 'dot'},
      {color: colors.marginal, label: 'Marginal', shape: 'ring'},
      {color: colors.hidden, label: 'Hidden', shape: 'ring'}
    ]
  };
}

/** The observer's gold eye, the target's teal and the outline and halo around them. */
export function getObserverColors(ground: TerrainGroundTone) {
  const dark = ground === 'dark';
  return {
    eye: hexToRgba(TERRAIN_EYE_GOLD),
    eyeHex: TERRAIN_EYE_GOLD,
    outline: hexToRgba(dark ? '#0B0D12' : '#1F2A4D'),
    halo: hexToRgba(dark ? '#14171C' : '#F3EFE6'),
    target: hexToRgba(dark ? '#4FC3CF' : '#00838F'),
    lookout: hexToRgba(TERRAIN_EYE_GOLD),
    /** Cells that flip class between two models (curvature step). */
    changed: hexToRgba(dark ? '#B39DDB' : '#5E3C99')
  };
}

/**
 * Seen-from-how-many-lookouts: class 0 is the indigo veil ("seen from no station"), classes 1 to
 * `observerCount` are YlOrRd (3 to 6 observers use the published table of that size) at alpha 0.75.
 */
export function makeCumulativeTable(ground: TerrainGroundTone, observerCount = 6): ClassTable {
  const count = Math.min(6, Math.max(1, Math.round(observerCount)));
  const palette = getClassPalette('YlOrRd', Math.max(3, count), {ground}).slice(-count);
  const veil = ground === 'dark' ? hexToRgba('#05070F', 175) : hexToRgba('#1F2A4D', 150);
  return makeClassTable({
    breaks: Array.from({length: count}, (_, index) => index + 1),
    colors: [veil, ...palette.map(color => withAlpha(color, 191))],
    labels: ['None', ...Array.from({length: count}, (_, index) => String(index + 1))],
    unit: 'lookouts',
    method: 'Number of lookouts that see the cell',
    noData: {label: 'No data', color: NO_DATA_COLOR[ground]}
  });
}

// ---------------------------------------------------------------------------------------------
// Sun and shadow
// ---------------------------------------------------------------------------------------------

/** The cold-to-warm sun-hours colours, 0 h to 8 h or more (December scale). */
export const SUN_HOURS_HEXES: readonly string[] = [
  '#2B2D42',
  '#5C4B73',
  '#A5527A',
  '#E4707A',
  '#F9A65A',
  '#FFE08A'
];

/**
 * Sun hours as classes. `december` is the 6-class scale (0, 0-2, 2-4, 4-6, 6-8, 8 h or more);
 * `june` extends it in 2 h bins to 16 h (10 classes, the same colours resampled in CIELAB), the
 * common scale on which every date is drawn so days can be compared. Alpha 0.88; "no sun" is a
 * class of its own (a value under 0.01 h), never no data.
 */
export function makeSunHoursTable(
  scale: 'december' | 'june',
  ground: TerrainGroundTone
): ClassTable {
  const bins = scale === 'june' ? [2, 4, 6, 8, 10, 12, 14, 16] : [2, 4, 6, 8];
  const hexes =
    scale === 'june' ? resampleHexTable(SUN_HOURS_HEXES, bins.length + 2) : [...SUN_HOURS_HEXES];
  const floorLifted =
    ground === 'dark' ? liftLightnessFloor(hexes, DARK_GROUND_MINIMUM_LIGHTNESS) : hexes;
  return makeClassTable({
    breaks: [0.01, ...bins],
    colors: floorLifted.map(hex => hexToRgba(hex, 224)),
    labels: [
      'No direct sun',
      `0-${bins[0]} h`,
      ...bins.slice(1).map((value, index) => `${bins[index]}-${value} h`),
      `${bins[bins.length - 1]} h or more`
    ],
    unit: 'hours of direct sun',
    method:
      scale === 'june'
        ? 'Common scale of every date (2 h bins)'
        : 'Hours of direct sun, 21 December',
    noData: {label: 'No data', color: NO_DATA_COLOR[ground]}
  });
}

/** Clear-sky insolation, YlOrBr-7 (`kWh/m² per day` by default); breaks are the scene's. */
export function makeInsolationTable(
  breaks: readonly number[],
  ground: TerrainGroundTone,
  unit = 'kWh/m² per day'
): ClassTable {
  return makeClassTable({
    breaks,
    colors: getClassPalette('YlOrBr', 7, {ground, alpha: 224}),
    unit,
    method: 'Clear-sky insolation on the slope',
    noData: {label: 'No data', color: NO_DATA_COLOR[ground]}
  });
}

/** The cast-shadow veil (the hidden indigo of the triad) and the sun's gold. */
export function getShadowColors(ground: TerrainGroundTone) {
  return {
    shadow: ground === 'dark' ? hexToRgba('#05070F', 175) : hexToRgba('#1B2340', 217),
    penumbra: ground === 'dark' ? hexToRgba('#05070F', 90) : hexToRgba('#1B2340', 100),
    sun: hexToRgba(TERRAIN_EYE_GOLD),
    sunHex: TERRAIN_EYE_GOLD
  };
}

/** The shadow legend: in sun (clear), penumbra, in shadow. */
export function getShadowLegend(ground: TerrainGroundTone): LegendSpec {
  const colors = getShadowColors(ground);
  return {
    kind: 'categories',
    title: 'Direct sun',
    entries: [
      {color: [0, 0, 0, 0], label: 'In sun (clear)'},
      {color: colors.penumbra, label: 'Penumbra'},
      {color: colors.shadow, label: 'In shadow'}
    ]
  };
}

/**
 * Skyline angle (degrees above the horizontal): YlGnBu from the 9-class ColorBrewer table without
 * its two palest classes, so the thinnest line on paper is still visible; `binDegrees` wide bins
 * starting at `firstBreakDegrees` (default 3-degree bins, breaks 3 to 18).
 */
export function makeSkylineTable(
  ground: TerrainGroundTone,
  options: {binDegrees?: number; firstBreakDegrees?: number} = {}
): ClassTable {
  const bin = options.binDegrees ?? 3;
  const first = options.firstBreakDegrees ?? bin;
  const colors = getClassPalette('YlGnBu', 9, {ground}).slice(2);
  return makeClassTable({
    breaks: Array.from({length: colors.length - 1}, (_, index) => first + index * bin),
    colors,
    unit: 'degrees above horizontal',
    method: `${bin}-degree bins of the elevation angle`,
    noData: {label: 'No data', color: NO_DATA_COLOR[ground]}
  });
}

// ---------------------------------------------------------------------------------------------
// Summits
// ---------------------------------------------------------------------------------------------

/** Snap outcomes of a peak catalogue on the DEM (Okabe-Ito), in legend order. */
export const SNAP_STATUS = [
  {id: 'snapped', label: 'Snapped to the summit', light: '#009E73', dark: '#2FCB9C'},
  {id: 'unchanged', label: 'Already on the summit', light: '#969696', dark: '#B0B0B0'},
  {id: 'flank', label: 'On a flank (moved a little)', light: '#E69F00', dark: '#F2B23A'},
  {id: 'too-far', label: 'Too far from a summit', light: '#D55E00', dark: '#F27C3A'},
  {id: 'height-mismatch', label: 'Height does not match', light: '#CC79A7', dark: '#E69BC4'}
] as const;

/** One snap outcome id of {@link SNAP_STATUS}. */
export type SnapStatusId = (typeof SNAP_STATUS)[number]['id'];

/** The colour of a snap outcome on a ground. */
export function getSnapStatusColor(id: SnapStatusId, ground: TerrainGroundTone): PaletteColor {
  const entry = SNAP_STATUS.find(status => status.id === id) ?? SNAP_STATUS[1];
  return hexToRgba(entry[ground]);
}

/** The snap-status legend (filled circles), optionally with counts. */
export function getSnapStatusLegend(
  ground: TerrainGroundTone,
  counts?: Partial<Record<SnapStatusId, number>>
): LegendSpec {
  return {
    kind: 'categories',
    title: 'Catalogue position, after snapping',
    layout: 'list',
    entries: SNAP_STATUS.map(status => ({
      color: hexToRgba(status[ground]),
      label: status.label,
      shape: 'dot' as const,
      ...(counts?.[status.id] !== undefined ? {count: counts[status.id]} : {})
    }))
  };
}

/**
 * Summit symbol sizes by drop: four classes, one visual variable (size), no colour. `breaks` are
 * the drop thresholds in metres above the 100 m minimum; `sizesPixels` are the triangle sizes.
 */
export const SUMMIT_DROP_CLASSES = {
  minimumDropMeters: 100,
  breaks: [200, 400, 700],
  sizesPixels: [8, 11, 15, 20],
  labels: ['100-200', '200-400', '400-700', 'over 700']
} as const;

/** Triangle size in CSS pixels of a summit with a drop in metres. */
export function getSummitSizePixels(dropMeters: number): number {
  const index = SUMMIT_DROP_CLASSES.breaks.filter(limit => dropMeters >= limit).length;
  return SUMMIT_DROP_CLASSES.sizesPixels[index];
}

/** Ink of summit symbols, rejected candidates and the paper outline. */
export function getSummitInk(ground: TerrainGroundTone) {
  const dark = ground === 'dark';
  return {
    summit: hexToRgba(dark ? '#E8EDF2' : '#1E2230'),
    outline: hexToRgba(dark ? '#14171C' : '#F3EFE6'),
    rejected: hexToRgba('#8A949E'),
    saddle: hexToRgba('#FDAE61'),
    pit: hexToRgba(dark ? '#7FB3D9' : '#2166AC')
  };
}

/** The nested triangle-size legend of the summit symbols (unit: metres of drop). */
export function getSummitSizeLegend(ground: TerrainGroundTone): LegendSpec {
  return {
    kind: 'size',
    title: 'Summit drop',
    unit: 'm drop',
    layout: 'nested',
    color: getSummitInk(ground).summit,
    entries: SUMMIT_DROP_CLASSES.sizesPixels.map((size, index) => ({
      radiusPixels: size / 2,
      label: SUMMIT_DROP_CLASSES.labels[index]
    })),
    note: 'Highest within the radius and at least this far above its ring'
  };
}

/** The contour legend (`line` entries) from the shared USGS contour style. */
export function getContourLegend(ground: TerrainGroundTone, intervalMeters = 100): LegendSpec {
  const style = CONTOUR_STYLE[ground];
  const toColor = (hex: string, opacity: number) => hexToRgba(hex, Math.round(opacity * 255));
  return {
    kind: 'line',
    title: 'Contours',
    entries: [
      {
        color: toColor(style.intermediate.color, style.intermediate.opacity),
        widthPixels: style.intermediate.widthPixels,
        label: `${intervalMeters} m contour`
      },
      {
        color: toColor(style.index.color, style.index.opacity),
        widthPixels: style.index.widthPixels,
        label: `${intervalMeters * CONTOUR_STYLE.indexEvery} m index contour`
      },
      {
        color: toColor(style.glacier.color, style.glacier.opacity),
        widthPixels: style.glacier.widthPixels,
        label: 'Contour on a glacier'
      }
    ]
  };
}

/** Ink tokens of the ground for halos and outlines (re-exported so a scene needs one import). */
export function getTerrainInk(ground: TerrainGroundTone) {
  return MAP_INK[ground];
}
