// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Classed colour from exact published tables, and the one `ClassTable` object that the layer, the
 * `classes` legend, tooltips and histogram ticks all read.
 *
 * - {@link getClassPalette}: the published ColorBrewer n-class table (sequential 3-9, diverging
 *   3-11, qualitative at their published sizes), NOT a resample of the 9-stop ramps, so the lowest
 *   class is not near-white-or-black by accident. Tables are generated from palettable (see
 *   `class-schemes.ts`), never typed. Colours: Apache 2.0, Cynthia Brewer, Mark Harrower and
 *   The Pennsylvania State University (credit them in the "About colours" line).
 * - {@link makeClassTable}: breaks + colours + labels in one object.
 * - {@link getClassTableLayerProps} / {@link getClassTableLegend}: spread into a layer and a scene
 *   legend so a map and its legend cannot drift apart.
 *
 * Orientation: every table runs LOW class first (the app convention of `RAMP_STOPS`). Sequential
 * tables run pale to dark as published. Diverging tables run in the order of the matching ramp:
 * `RdBu`, `RdYlBu`, `Spectral`, `RdGy`, `RdYlGn` and `PuOr` are published warm/orange first and
 * are flipped here (blue, cool or purple low, red or orange high); `BrBG`, `PiYG` and `PRGn` are
 * as published (brown, pink, purple low). Pass `reverse: true` to flip any table.
 *
 * Break convention (shared with the layers): the class of a value is the number of breaks `<=` it.
 */

import {type PaletteColor, sampleRamp} from '../engine/ramps';
import type {LegendSpec} from '../scenes/scene';
import {formatBreakLabels, getClassIndex} from './breaks';
import {CLASS_SCHEME_DATA, type ClassSchemeKind, type ClassSchemeName} from './class-schemes';
import type {ClassColor, ClassTable} from './types';

export {CLASS_SCHEME_DATA};
export type {ClassSchemeKind, ClassSchemeName};

// ---------------------------------------------------------------------------------------------
// Colour helpers
// ---------------------------------------------------------------------------------------------

/**
 * Converts `#rgb`, `#rrggbb` or `#rrggbbaa` to a 0-255 RGBA colour. `alpha` (0-255) overrides the
 * alpha of the hex (default 255, or the hex's own alpha).
 */
export function hexToRgba(hex: string, alpha?: number): PaletteColor {
  let digits = hex.trim().replace(/^#/, '');
  if (digits.length === 3 || digits.length === 4) {
    digits = digits
      .split('')
      .map(digit => digit + digit)
      .join('');
  }
  const value = (start: number) => Number.parseInt(digits.slice(start, start + 2), 16);
  const ownAlpha = digits.length >= 8 ? value(6) : 255;
  return [value(0), value(2), value(4), alpha ?? ownAlpha];
}

/** Converts a 0-255 colour to lowercase `#rrggbb` (alpha is dropped). */
export function rgbToHex(color: readonly [number, number, number, number?]): string {
  const channel = (value: number) =>
    Math.round(Math.min(Math.max(value, 0), 255))
      .toString(16)
      .padStart(2, '0');
  return `#${channel(color[0])}${channel(color[1])}${channel(color[2])}`;
}

type Lab = [number, number, number];

// D65 white point.
const WHITE_X = 0.95047;
const WHITE_Z = 1.08883;

function toLinear(channel: number): number {
  const value = channel / 255;
  return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
}

function fromLinear(value: number): number {
  const clamped = value <= 0.0031308 ? value * 12.92 : 1.055 * value ** (1 / 2.4) - 0.055;
  return clamped * 255;
}

/** sRGB (0-255) to CIELAB (D65). */
function rgbToLab(red: number, green: number, blue: number): Lab {
  const r = toLinear(red);
  const g = toLinear(green);
  const b = toLinear(blue);
  const x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / WHITE_X;
  const y = 0.2126729 * r + 0.7151522 * g + 0.072175 * b;
  const z = (0.0193339 * r + 0.119192 * g + 0.9503041 * b) / WHITE_Z;
  const f = (t: number) => (t > 216 / 24389 ? Math.cbrt(t) : ((24389 / 27) * t + 16) / 116);
  const fx = f(x);
  const fy = f(y);
  const fz = f(z);
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}

/** Linear sRGB of a Lab colour (unclamped). */
function labToLinear(lightness: number, a: number, b: number): [number, number, number] {
  const fy = (lightness + 16) / 116;
  const fx = fy + a / 500;
  const fz = fy - b / 200;
  const inverse = (t: number) => (t ** 3 > 216 / 24389 ? t ** 3 : (116 * t - 16) / (24389 / 27));
  const x = inverse(fx) * WHITE_X;
  const y = lightness > 8 ? fy ** 3 : lightness / (24389 / 27);
  const z = inverse(fz) * WHITE_Z;
  return [
    3.2404542 * x - 1.5371385 * y - 0.4985314 * z,
    -0.969266 * x + 1.8760108 * y + 0.041556 * z,
    0.0556434 * x - 0.2040259 * y + 1.0572252 * z
  ];
}

/** Lab to `#rrggbb`; when out of gamut the chroma is reduced (lightness and hue are kept). */
function labToHex(lightness: number, a: number, b: number): string {
  let scale = 1;
  for (let step = 0; step < 24; step++) {
    const linear = labToLinear(lightness, a * scale, b * scale);
    if (linear.every(value => value >= -0.0005 && value <= 1.0005)) break;
    scale *= 0.93;
  }
  const [r, g, bl] = labToLinear(lightness, a * scale, b * scale).map(fromLinear);
  return rgbToHex([r, g, bl]);
}

function hexToLab(hex: string): Lab {
  const [r, g, b] = hexToRgba(hex);
  return rgbToLab(r, g, b);
}

/** Interpolates a list of hex colours to `count` colours, evenly spaced, in CIELAB. */
function resampleHexes(hexes: readonly string[], count: number): string[] {
  if (count <= 1) return [hexes[Math.floor((hexes.length - 1) / 2)]];
  const labs = hexes.map(hexToLab);
  return Array.from({length: count}, (_, index) => {
    const position = (index / (count - 1)) * (labs.length - 1);
    const lower = Math.min(Math.floor(position), labs.length - 2);
    const fraction = position - lower;
    const from = labs[lower];
    const to = labs[lower + 1];
    return labToHex(
      from[0] + (to[0] - from[0]) * fraction,
      from[1] + (to[1] - from[1]) * fraction,
      from[2] + (to[2] - from[2]) * fraction
    );
  });
}

/**
 * Resamples a colour table to `count` colours by interpolating in CIELAB (not sRGB), used where
 * no published n-class table exists. Returns lowercase hex.
 */
export function resampleHexTable(hexes: readonly string[], count: number): string[] {
  return resampleHexes(hexes, count);
}

// ---------------------------------------------------------------------------------------------
// Dark-ground variants
// ---------------------------------------------------------------------------------------------

/** The map ground behind data on a dark page (`--map-ground`), hex. */
export const DARK_GROUND_HEX = '#13171c';

/**
 * CIE L* at which a colour reaches 3:1 WCAG contrast against {@link DARK_GROUND_HEX} (with a small
 * margin): the floor for the lowest visible class on a dark ground.
 */
export const DARK_GROUND_MINIMUM_LIGHTNESS = 42.5;

/** Neutral centre class of a diverging table on a dark ground (ground-tinted, drawn at alpha 0.15-0.35). */
export const DARK_NEUTRAL_HEX = '#2b3240';

/** Neutral centre class of a diverging table on paper (`#f7f7f7`). */
export const LIGHT_NEUTRAL_HEX = '#f7f7f7';

/**
 * Dark-ground diverging tables authored in SYNTHESIS 1.2 (dark neutral centre, both arms
 * brightening outward), keyed `<scheme>-<classCount>`. Blue low, orange high.
 */
export const AUTHORED_DARK_DIVERGING: Readonly<Record<string, readonly string[]>> = {
  'RdBu-7': ['#4da3ff', '#2a6fbf', '#1b3a66', '#2b3240', '#66361f', '#c8661f', '#ffb347']
};

/**
 * Dark-ground sequential tables authored in SYNTHESIS 1.2 (magma-sampled), keyed
 * `<scheme>-<classCount>`. They start well above black.
 */
export const AUTHORED_DARK_SEQUENTIAL: Readonly<Record<string, readonly string[]>> = {
  'YlOrRd-6': ['#2d1160', '#721f81', '#b63679', '#f1605d', '#fec287', '#fcfdbf'],
  'YlOrRd-5': ['#51127c', '#8c2981', '#de4968', '#fe9f6d', '#fcfdbf']
};

/**
 * Schemes whose dark-ground variant is a classed sample of a perceptual ramp trimmed to `range`
 * (SYNTHESIS 1.2: harm is magma t 0.25-1, people and nature are inferno t 0.15-1) instead of the
 * reversed table. Used for class counts that have no entry in {@link AUTHORED_DARK_SEQUENTIAL}.
 */
export const DARK_FAMILY_RAMPS: Readonly<
  Partial<Record<ClassSchemeName, {ramp: 'magma' | 'inferno'; range: readonly [number, number]}>>
> = {
  YlOrRd: {ramp: 'magma', range: [0.25, 1]},
  YlOrBr: {ramp: 'inferno', range: [0.15, 1]},
  YlGnBu: {ramp: 'inferno', range: [0.15, 1]}
};

/**
 * Dark-ground variant of a LIGHT-ground sequential table (strongest class darkest): reversed so
 * the strongest class is the brightest, then the lightness of the lowest classes is lifted
 * (linear remap of L*, hue and chroma kept) so the lowest class reaches
 * {@link DARK_GROUND_MINIMUM_LIGHTNESS}, i.e. at least 3:1 against `#13171C`. The input must be
 * ordered by lightness (a published sequential table is).
 */
export function deriveDarkSequential(lightHexes: readonly string[]): string[] {
  const reversed = [...lightHexes].reverse();
  return liftLightnessFloor(reversed, DARK_GROUND_MINIMUM_LIGHTNESS);
}

/**
 * Lifts the darkest colour of a table to L* `floor` and remaps the rest linearly above it, so the
 * order and the spacing of lightness is kept while nothing is darker than `floor`. Colours of a
 * table already above the floor are returned unchanged.
 */
export function liftLightnessFloor(hexes: readonly string[], floor: number): string[] {
  const labs = hexes.map(hexToLab);
  const minimum = Math.min(...labs.map(lab => lab[0]));
  const maximum = Math.max(...labs.map(lab => lab[0]));
  if (minimum >= floor) return [...hexes];
  const top = Math.max(maximum, floor + 12);
  const span = maximum - minimum;
  return labs.map(([lightness, a, b]) => {
    const remapped = span > 1e-6 ? floor + ((lightness - minimum) / span) * (top - floor) : floor;
    return labToHex(remapped, a, b);
  });
}

/**
 * Dark-ground variant of a LIGHT-ground diverging table (low class first): a ground-tinted
 * neutral centre (`#2b3240`, a real class for odd counts) and both arms brightening outward, with
 * the hue of each arm taken from its published end colour. Lightness runs 22 (centre) to about 74
 * (ends), chroma grows with distance. For `RdBu` with 7 classes the authored table
 * ({@link AUTHORED_DARK_DIVERGING}) is used instead.
 */
export function deriveDarkDiverging(lightHexes: readonly string[]): string[] {
  const count = lightHexes.length;
  const middle = (count - 1) / 2;
  const lowEnd = hexToLab(lightHexes[0]);
  const highEnd = hexToLab(lightHexes[count - 1]);
  return lightHexes.map((_, index) => {
    const distance = Math.abs(index - middle);
    if (distance === 0) return DARK_NEUTRAL_HEX;
    const t = distance / middle;
    const end = index < middle ? lowEnd : highEnd;
    const endChroma = Math.hypot(end[1], end[2]);
    const hueScale = Math.min(1, 70 / Math.max(endChroma, 1)) * (0.5 + 0.5 * t);
    const lightness = 22 + 52 * t ** 1.6;
    return labToHex(lightness, end[1] * hueScale, end[2] * hueScale);
  });
}

/**
 * Dark-ground variant of a qualitative table: every colour darker than L* 70 is lifted by 15
 * (capped at 85), so dark hues (Dark2, Set1) stay luminous; light hues are unchanged.
 */
export function deriveDarkQualitative(lightHexes: readonly string[]): string[] {
  return lightHexes.map(hex => {
    const [lightness, a, b] = hexToLab(hex);
    return lightness >= 70 ? hex : labToHex(Math.min(lightness + 15, 85), a, b);
  });
}

// ---------------------------------------------------------------------------------------------
// Class palettes
// ---------------------------------------------------------------------------------------------

/** Options of {@link getClassPalette}. */
export type ClassPaletteOptions = {
  /** Flip the table (high class first). */
  reverse?: boolean;
  /** Alpha 0-255 of every colour (default 255). */
  alpha?: number;
  /** `'dark'` returns the dark-ground variant of the scheme (default `'light'`). */
  ground?: 'light' | 'dark';
};

/** Every ColorBrewer scheme name. */
export const CLASS_SCHEME_NAMES = Object.keys(CLASS_SCHEME_DATA) as ClassSchemeName[];

/** Scheme names of one measurement level. */
export function getClassSchemeNames(kind: ClassSchemeKind): ClassSchemeName[] {
  return CLASS_SCHEME_NAMES.filter(name => CLASS_SCHEME_DATA[name].kind === kind);
}

/** Measurement level and published class-count range of a scheme. */
export function getClassSchemeInfo(scheme: ClassSchemeName): {
  kind: ClassSchemeKind;
  minimumClasses: number;
  maximumClasses: number;
} {
  const data = CLASS_SCHEME_DATA[scheme];
  const sizes = Object.keys(data.tables).map(Number);
  return {
    kind: data.kind,
    minimumClasses: data.kind === 'qualitative' ? 3 : Math.min(...sizes),
    maximumClasses: Math.max(...sizes)
  };
}

/** The light-ground table of `scheme` with `classCount` classes, as published (low class first). */
function getLightTable(scheme: ClassSchemeName, classCount: number): string[] {
  const data = CLASS_SCHEME_DATA[scheme];
  const tables = data.tables as Readonly<Record<number, readonly string[]>>;
  const count = Math.max(1, Math.round(classCount));
  if (data.kind === 'qualitative') {
    const full = tables[Math.max(...Object.keys(tables).map(Number))];
    return Array.from({length: count}, (_, index) => full[index % full.length]);
  }
  if (tables[count]) return [...tables[count]];
  const sizes = Object.keys(tables).map(Number);
  const maximum = Math.max(...sizes);
  if (count > maximum) return resampleHexes(tables[maximum], count);
  // Fewer than 3 classes: the ends (2) or the middle (1) of the 3-class table.
  return resampleHexes(tables[Math.min(...sizes)], count);
}

/** The dark-ground table of `scheme` for `classCount` classes. */
function getDarkTable(scheme: ClassSchemeName, classCount: number): string[] {
  const data = CLASS_SCHEME_DATA[scheme];
  const light = getLightTable(scheme, classCount);
  if (data.kind === 'qualitative') return deriveDarkQualitative(light);
  if (data.kind === 'diverging') {
    return [...(AUTHORED_DARK_DIVERGING[`${scheme}-${classCount}`] ?? deriveDarkDiverging(light))];
  }
  const authored = AUTHORED_DARK_SEQUENTIAL[`${scheme}-${classCount}`];
  if (authored) return [...authored];
  const family = DARK_FAMILY_RAMPS[scheme];
  if (family) {
    return Array.from({length: classCount}, (_, index) =>
      rgbToHex(
        sampleRamp(family.ramp, classCount <= 1 ? 1 : index / (classCount - 1), false, family.range)
      )
    );
  }
  return deriveDarkSequential(light);
}

/**
 * The published n-class table of a ColorBrewer scheme as lowercase hex, low class first (see the
 * orientation note at the top of the module). Same arguments as {@link getClassPalette}.
 */
export function getClassPaletteHex(
  scheme: ClassSchemeName,
  classCount: number,
  options: Pick<ClassPaletteOptions, 'reverse' | 'ground'> = {}
): string[] {
  const table =
    options.ground === 'dark'
      ? getDarkTable(scheme, classCount)
      : getLightTable(scheme, classCount);
  return options.reverse ? table.reverse() : table;
}

/**
 * The published n-class ColorBrewer table as RGBA, low class first.
 *
 * Counts: sequential 3-9, diverging 3-11, qualitative up to the published size (3-8, Paired and
 * Set3 12, Set1 and Pastel1 9; more cycles the table). Outside that range the table is resampled
 * in CIELAB from the nearest published one (fewer than 3 classes keep the ends or the middle) and
 * the sample is no longer a published table.
 *
 * Colour-vision caveat: `RdYlGn`, `Spectral`, `Set2`, `Pastel1`, `Paired` and `Set3` are not colour-blind
 * safe (`node scripts/check-colours.mjs` prints the numbers); the rules also forbid a red-green pair.
 *
 * `ground: 'dark'` returns the dark-ground variant:
 * - sequential `YlOrRd` (5 and 6 classes) uses the authored magma tables of the hue registry,
 *   other counts a classed sample of magma t 0.25-1; `YlOrBr` and `YlGnBu` sample inferno t 0.15-1;
 * - every other sequential scheme is reversed (strongest class brightest, same hue family) and its
 *   lowest classes lifted to L* 42.5 so the lowest class keeps 3:1 against `#13171C`
 *   ({@link deriveDarkSequential});
 * - diverging: `RdBu` with 7 classes is the authored table (`#4da3ff ... #ffb347`), others get a
 *   dark neutral centre and arms brightening outward ({@link deriveDarkDiverging});
 * - qualitative: dark hues lifted by 15 L* ({@link deriveDarkQualitative}).
 *
 * @example
 * ```ts
 * getClassPalette('YlOrRd', 5); // pale yellow ... dark red, the published 5-class table
 * getClassPalette('PuOr', 7, {ground: 'dark', alpha: 200});
 * ```
 */
export function getClassPalette(
  scheme: ClassSchemeName,
  classCount: number,
  options: ClassPaletteOptions = {}
): PaletteColor[] {
  return getClassPaletteHex(scheme, classCount, options).map(hex =>
    hexToRgba(hex, options.alpha ?? 255)
  );
}

// ---------------------------------------------------------------------------------------------
// Class tables
// ---------------------------------------------------------------------------------------------

/** Options of {@link makeClassTable}. */
export type MakeClassTableOptions = {
  /** Interior class breaks, ascending. The class count is `breaks.length + 1`. */
  breaks: readonly number[];
  /** ColorBrewer scheme the class colours come from (default `'YlOrRd'`). */
  scheme?: ClassSchemeName;
  /** Explicit colours, one per class, low first. Overrides `scheme`. */
  colors?: readonly ClassColor[];
  /** Flip the colours. */
  reverse?: boolean;
  /** Ground the table is drawn on (default `'light'`). */
  ground?: 'light' | 'dark';
  /** Alpha 0-255 of every class (default 255). */
  alpha?: number;
  /** Classes drawn fully transparent (zero, "nothing to say"), by index. */
  transparent?: readonly number[];
  /** Classes drawn hatched in the legend (suppressed, low n, not significant), by index. */
  hatched?: readonly number[];
  /** One label per class, replacing the generated ranges. */
  labels?: readonly string[];
  /** Value unit for the legend title and tooltips. */
  unit?: string;
  /** Data `[min, max]` for the outer labels. */
  extent?: readonly [number, number];
  /** Classification method shown under the legend. */
  method?: string;
  /** The no-data swatch of the legend. */
  noData?: ClassTable['noData'];
  /** Value formatter for generated labels. */
  format?: (value: number) => string;
};

/**
 * Builds the one {@link ClassTable} that the layer (`getClassTableLayerProps`), the `classes`
 * legend (`getClassTableLegend`), tooltips (`getClassLabel`) and histogram ticks read.
 *
 * @throws When `colors` does not have `breaks.length + 1` entries.
 *
 * @example
 * ```ts
 * const table = makeClassTable({
 *   breaks: [10, 25, 50, 100],
 *   scheme: 'YlGnBu',
 *   unit: 'observations / km²',
 *   method: 'Quantile'
 * });
 * ```
 */
export function makeClassTable(options: MakeClassTableOptions): ClassTable {
  const classCount = options.breaks.length + 1;
  const base: readonly ClassColor[] =
    options.colors ??
    getClassPalette(options.scheme ?? 'YlOrRd', classCount, {
      ground: options.ground,
      alpha: options.alpha
    });
  if (base.length !== classCount) {
    throw new Error(`makeClassTable: ${base.length} colours for ${classCount} classes`);
  }
  const ordered = options.reverse ? [...base].reverse() : base;
  const transparent = new Set(options.transparent ?? []);
  const colors: ClassColor[] = ordered.map((color, index) => [
    color[0],
    color[1],
    color[2],
    transparent.has(index) ? 0 : (color[3] ?? options.alpha ?? 255)
  ]);
  const table: ClassTable = {breaks: [...options.breaks], colors};
  if (options.labels) table.labels = options.labels;
  if (options.unit !== undefined) table.unit = options.unit;
  if (options.extent) table.extent = options.extent;
  if (options.method !== undefined) table.method = options.method;
  if (options.hatched) table.hatched = options.hatched;
  if (options.noData) table.noData = options.noData;
  if (options.format) table.format = options.format;
  return table;
}

/** Props to spread into any `SpatialAnalysis*Layer` that takes `classBreaks` and `classColors`. */
export type ClassTableLayerProps = {
  classBreaks: number[];
  classColors: PaletteColor[];
  /** Present only when the table has hatched classes. */
  hatchClasses?: number[];
};

/**
 * Layer props of a class table: `new SpatialAnalysisPolygonLayer({...getClassTableLayerProps(t)})`.
 * Colours without alpha get 255.
 */
export function getClassTableLayerProps(table: ClassTable): ClassTableLayerProps {
  const props: ClassTableLayerProps = {
    classBreaks: [...table.breaks],
    classColors: table.colors.map(color => [color[0], color[1], color[2], color[3] ?? 255])
  };
  if (table.hatched?.length) props.hatchClasses = [...table.hatched];
  return props;
}

/** The `classes` legend spec. */
export type ClassesLegendSpec = Extract<LegendSpec, {kind: 'classes'}>;

/** Options of {@link getClassTableLegend}. */
export type ClassTableLegendOptions = {
  /** Legend title: say what is measured and the denominator ("Observations per km²"). */
  title: string;
  /** Stable id (needed for `interactive` legends). */
  id?: string;
  /** Normalisation basis, shown after the unit. */
  basis?: string;
  /** Features per class (list layout shows them next to the swatches). */
  counts?: readonly number[];
  /** `'bar'` or `'list'`; see the `classes` legend. */
  layout?: 'bar' | 'list';
  /** Distribution under the classes: bin counts spanning the table's `extent`. */
  histogram?: readonly number[];
  /** Swatches filter the map (hover isolates a class, click locks it). */
  interactive?: boolean;
  /** Note under the classes; defaults to the table's `method`. */
  note?: string;
};

/**
 * The `classes` legend of a class table, so the legend shows exactly what the layer draws.
 *
 * @example
 * ```ts
 * legend: [getClassTableLegend(table, {title: 'Observations per km²', counts, interactive: true})]
 * ```
 */
export function getClassTableLegend(
  table: ClassTable,
  options: ClassTableLegendOptions
): ClassesLegendSpec {
  const legend: ClassesLegendSpec = {
    kind: 'classes',
    title: options.title,
    breaks: table.breaks,
    colors: table.colors,
    table
  };
  if (options.id !== undefined) legend.id = options.id;
  if (options.basis !== undefined) legend.basis = options.basis;
  if (table.unit !== undefined) legend.unit = table.unit;
  if (table.extent) legend.extent = table.extent;
  if (table.labels) legend.labels = table.labels;
  if (table.format) legend.format = table.format;
  if (table.hatched) legend.hatched = table.hatched;
  if (table.noData) legend.noData = table.noData;
  if (options.counts) legend.counts = options.counts;
  if (options.layout) legend.layout = options.layout;
  if (options.histogram) legend.histogram = options.histogram;
  if (options.interactive) legend.interactive = true;
  const note = options.note ?? table.method;
  if (note !== undefined) legend.note = note;
  return legend;
}

/** Class index of `value` in `table` (the number of breaks `<= value`); `-1` for NaN. */
export function getClassIndexOf(table: ClassTable, value: number): number {
  return Number.isNaN(value) ? -1 : getClassIndex(value, table.breaks);
}

/** Default break formatter: up to two decimals, thousands separators. */
function formatBreak(value: number): string {
  return value.toLocaleString('en-US', {maximumFractionDigits: 2});
}

/**
 * Label of class `index`: the table's own label, else the generated range ("12-30", "< 12",
 * "≥ 400"). Returns `''` for an index outside the table.
 */
export function getClassLabel(table: ClassTable, index: number): string {
  if (index < 0 || index >= table.breaks.length + 1) return '';
  return (
    table.labels?.[index] ??
    formatBreakLabels(table.breaks, table.extent, table.format ?? formatBreak)[index]
  );
}

// ---------------------------------------------------------------------------------------------
// Diverging and significance breaks
// ---------------------------------------------------------------------------------------------

/**
 * Symmetric breaks about `center`: each step is used on both sides, so `getDivergingBreaks(0,
 * [1.65, 1.96, 2.58])` is `[-2.58, -1.96, -1.65, 1.65, 1.96, 2.58]` (seven classes, the middle one
 * is the neutral "not distinguishable from the centre" class).
 */
export function getDivergingBreaks(center: number, steps: readonly number[]): number[] {
  const sorted = [...new Set(steps.map(Math.abs))].sort((a, b) => a - b);
  const tidy = (value: number) => Number(value.toPrecision(12));
  return [
    ...sorted.map(step => tidy(center - step)).reverse(),
    ...sorted.map(step => tidy(center + step))
  ];
}

/** Getis-Ord Gi* z-score breaks (Esri hot-spot convention): 90, 95 and 99 % confidence. */
export const GI_STAR_BREAKS: readonly number[] = getDivergingBreaks(0, [1.65, 1.96, 2.58]);

/** The seven Gi* class labels, cold 99 % first. */
export const GI_STAR_LABELS: readonly string[] = [
  'Cold spot 99%',
  'Cold spot 95%',
  'Cold spot 90%',
  'Not significant',
  'Hot spot 90%',
  'Hot spot 95%',
  'Hot spot 99%'
];

/**
 * The canonical Gi* class table: RdBu-7 (blue cold, red hot), breaks at +-1.65/1.96/2.58 and the
 * neutral class drawn as a ghost (alpha 0.5 on light, 0.25 on dark). Dark grounds use the authored
 * dark RdBu table.
 */
export function makeGiStarClassTable(
  options: {ground?: 'light' | 'dark'; unit?: string; method?: string} = {}
): ClassTable {
  const ground = options.ground ?? 'light';
  const colors = getClassPalette('RdBu', 7, {ground}).map((color, index) =>
    index === 3 ? ([color[0], color[1], color[2], ground === 'dark' ? 64 : 128] as const) : color
  );
  return makeClassTable({
    breaks: GI_STAR_BREAKS,
    colors,
    labels: GI_STAR_LABELS,
    unit: options.unit ?? 'z-score',
    method: options.method ?? 'Gi* z-score, 90 / 95 / 99 % confidence'
  });
}
