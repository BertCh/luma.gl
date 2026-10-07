// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import type {SpatialAnalysisRasterLayerProps} from './layers';

/**
 * The relief ground helper (SYNTHESIS G16, section 1.2 "Relief"): one place that composites the
 * ground a terrain, hydrology or raster story draws on. A DEM becomes a packed RGBA8 image (a
 * tinted, multi-directional hillshade) for a {@link SpatialAnalysisRasterLayer} with
 * `colormap: 'rgba'`; analysis layers are then drawn above it.
 *
 * **Compositing rule.** The relief is a layer in the deck canvas, drawn first. Analysis layers go
 * above it either at alpha 0.62-0.88 with normal blending (stepped classes with a transparent
 * "nothing to say" class), or with `blending: 'multiply'`, which multiplies what the deck canvas
 * already shows. Neither can reach the basemap canvas, which is a different canvas under the deck
 * canvas, so a relief story uses a `style: 'none'` basemap (the relief is its ground) and puts
 * relief and analysis in the same deck pass.
 *
 * Typical use:
 *
 * ```ts
 * const image = buildReliefImage(dem, {tints: 'alpine', ground: 'light'});
 * const relief = new SpatialAnalysisRasterLayer({
 *   id: 'relief',
 *   ...createReliefLayerProps(device, image, metricBounds)
 * });
 * // then analysis layers, for example opacity: 0.8, or blending: 'multiply'
 * ```
 *
 * Colours are mixed in sRGB (not linear light), which is what the published tints assume.
 */

/** One hypsometric tint stop: the colour from `elevation` metres up to the next stop. */
export type ReliefTintStop = {
  /** Lower elevation limit in metres. */
  readonly elevation: number;
  /** CSS hex colour `#RRGGBB` (or `#RGB`). */
  readonly color: string;
};

/** A hypsometric tint table: stops ordered by ascending elevation. */
export type ReliefTintTable = {
  /** Stops by ascending `elevation`. Below the first stop the first colour is used. */
  readonly stops: readonly ReliefTintStop[];
  /** `true` blends linearly between stops (absolute tint); `false` draws hard bands. */
  readonly interpolate: boolean;
};

/**
 * Alpine absolute tint (SYNTHESIS 1.2): blended between 1500 m `#8F9F7A` and 4476 m `#FAFAFC`.
 * Meant for the Alps; other ranges should name their own table.
 */
export const ALPINE_TINTS: ReliefTintTable = {
  interpolate: true,
  stops: [
    {elevation: 1500, color: '#8F9F7A'},
    {elevation: 1900, color: '#AEAD86'},
    {elevation: 2300, color: '#C4B896'},
    {elevation: 2800, color: '#D6CDBB'},
    {elevation: 3300, color: '#E4DFD6'},
    {elevation: 3800, color: '#F0EEEB'},
    {elevation: 4476, color: '#FAFAFC'}
  ]
};

/**
 * Arid canyon tint (SYNTHESIS 1.2): hard bands 680-1000 m through 2500-2650 m; the last band
 * continues above 2650 m.
 */
export const CANYON_TINTS: ReliefTintTable = {
  interpolate: false,
  stops: [
    {elevation: 680, color: '#8C8F6E'},
    {elevation: 1000, color: '#B7A878'},
    {elevation: 1250, color: '#D4B685'},
    {elevation: 1500, color: '#D99F6C'},
    {elevation: 1750, color: '#CB8460'},
    {elevation: 2000, color: '#E0C3A0'},
    {elevation: 2250, color: '#EFE0C6'},
    {elevation: 2500, color: '#F7F1E3'}
  ]
};

/** No tint: the shade alone (white tint). */
const NEUTRAL_TINTS: ReliefTintTable = {
  interpolate: false,
  stops: [{elevation: 0, color: '#FFFFFF'}]
};

/** The named tint tables accepted by `ReliefOptions.tints`. */
export const RELIEF_TINT_TABLES = {
  alpine: ALPINE_TINTS,
  canyon: CANYON_TINTS,
  neutral: NEUTRAL_TINTS
} as const;

/** Colour, width in CSS pixels and alpha of one contour class. */
export type ReliefContourClass = {
  readonly color: string;
  readonly widthPixels: number;
  readonly opacity: number;
};

/** Contour classes of one ground. */
export type ReliefContourPalette = {
  /** Intermediate contours. */
  readonly intermediate: ReliefContourClass;
  /** Index contours (every {@link CONTOUR_STYLE}`.indexEvery` th). */
  readonly index: ReliefContourClass;
  /** Contours over glaciers (blue, as on USGS maps). */
  readonly glacier: ReliefContourClass;
};

/**
 * Contour styling (USGS convention, SYNTHESIS 1.2): brown intermediate contours 0.6 px at 0.75
 * and index contours 1.3 px, every fifth; blue on glaciers; a lighter brown pair on dark grounds.
 * The helper does not draw contours (use a segment layer); this is the shared style.
 */
export const CONTOUR_STYLE = {
  indexEvery: 5,
  light: {
    intermediate: {color: '#8A5A33', widthPixels: 0.6, opacity: 0.75},
    index: {color: '#6B3E1B', widthPixels: 1.3, opacity: 1},
    glacier: {color: '#4F86B0', widthPixels: 0.6, opacity: 0.8}
  },
  dark: {
    intermediate: {color: '#C9A27A', widthPixels: 0.6, opacity: 0.75},
    index: {color: '#E8C9A0', widthPixels: 1.3, opacity: 1},
    glacier: {color: '#8DBEE0', widthPixels: 0.6, opacity: 0.8}
  }
} as const satisfies {
  indexEvery: number;
  light: ReliefContourPalette;
  dark: ReliefContourPalette;
};

/** The DEM a relief is built from (a `DecodedRaster` from `data/loaders` fits). */
export type ReliefDem = {
  /** Columns. */
  width: number;
  /** Rows; row 0 is the north edge (the loaders' order). */
  height: number;
  /** Elevations in metres, `width * height`, row-major. Non-finite values are no data. */
  values: ArrayLike<number>;
  /**
   * `[west, south, east, north]` in degrees. Used only to derive the ground size of a cell (at
   * the centre latitude) when `cellSizeMeters` is not given.
   */
  bounds?: readonly [number, number, number, number];
  /** Elevation that means no data, in addition to NaN and infinity. */
  noData?: number;
};

/** Options of {@link buildReliefImage}. */
export type ReliefOptions = {
  /** Hypsometric tints: a table, or `'alpine'`, `'canyon'`, `'neutral'` (no tint). */
  tints: ReliefTintTable | keyof typeof RELIEF_TINT_TABLES;
  /** Light azimuth in degrees clockwise from north. Defaults to 315. */
  azimuthDegrees?: number;
  /** Light altitude above the horizon in degrees. Defaults to 40. */
  altitudeDegrees?: number;
  /**
   * Average five lights spread 60 degrees either side of the azimuth (weights 1-2-3-2-1) so no
   * ridge line disappears. Defaults to `true`; `false` is the classic single light.
   */
  multidirectional?: boolean;
  /** Vertical exaggeration. Defaults to 1. */
  zFactor?: number;
  /** Shade scale: above 1 darkens shadows and brightens lit slopes. Defaults to 1.2. */
  contrast?: number;
  /** Cool shadow colour (light ground). Defaults to `#5C6B8A`, never grey. */
  shadeColor?: string;
  /** Warm lit colour (light ground). Defaults to `#FFF4DE`. */
  lightColor?: string;
  /** How much of the tint multiplies the shade, 0-1. Defaults to 0.30. */
  tintStrength?: number;
  /**
   * Contrast by height: low ground is hazier. Between `from` and `to` metres the shade contrast
   * rises from `1 - strength` to 1. Defaults to `{from: 1800, to: 4200, strength: 0.35}` (the
   * Alps); pass `null` to turn it off for other ranges.
   */
  aerialPerspective?: {from: number; to: number; strength: number} | null;
  /** Non-zero where a cell is glacier (`width * height`). */
  glacierMask?: Uint8Array;
  /** Glacier colour. Defaults to `#D5E6EF`. */
  glacierColor?: string;
  /**
   * The ground the relief sits on. `'light'` (default) is the paper `#F3EFE6` ground. `'dark'`
   * uses lifted cool tones whose shade never goes below `#1B1F27` (paper `#14171C`), so the
   * relief stays readable but quiet. `shadeColor` and `lightColor` apply to the light ground.
   */
  ground?: 'light' | 'dark';
  /**
   * Ground size of one cell in metres, `[x, y]` or one number for both. Defaults to the size
   * derived from `dem.bounds` at the centre latitude (30 m without bounds).
   */
  cellSizeMeters?: number | readonly [number, number];
  /**
   * Cells over which alpha ramps from 0 at the image edge to 1, so the relief does not end in a
   * hard line. Defaults to 8; 0 turns the feather off.
   */
  featherCells?: number;
};

/** A relief as packed RGBA8 pixels (red in the low byte), row 0 at the north edge. */
export type ReliefImage = {
  /** One word per cell, ready for `valueFormat: 'uint32'` with `colormap: 'rgba'`. */
  pixels: Uint32Array;
  width: number;
  height: number;
};

const LIGHT_PAPER = '#F3EFE6';
const DARK_TONES = {shade: '#1B1F27', flat: '#272D38', light: '#5A6577'} as const;
/** The darkest value a dark-ground relief may reach, per channel (`#1B1F27`). */
const DARK_FLOOR: readonly [number, number, number] = [0x1b, 0x1f, 0x27];
/** Ramp position of flat ground between the shade end (0) and the lit end (1). */
const FLAT_POSITION = 0.62;
const MULTIDIRECTIONAL_OFFSETS = [-60, -30, 0, 30, 60] as const;
const MULTIDIRECTIONAL_WEIGHTS = [1, 2, 3, 2, 1] as const;
const METERS_PER_DEGREE_LATITUDE = 110574;
const METERS_PER_DEGREE_EQUATOR = 111320;

type Rgb = [number, number, number];

/** Parses `#RGB` or `#RRGGBB` to 0-255 channels. */
function parseHex(hex: string): Rgb {
  let digits = hex.trim().replace(/^#/, '');
  if (digits.length === 3) digits = digits.replace(/./g, digit => digit + digit);
  const value = Number.parseInt(digits, 16);
  if (digits.length !== 6 || !Number.isFinite(value)) {
    throw new Error(`Relief: bad colour "${hex}"`);
  }
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
}

function mixRgb(from: Rgb, to: Rgb, t: number, out: Rgb): Rgb {
  out[0] = from[0] + (to[0] - from[0]) * t;
  out[1] = from[1] + (to[1] - from[1]) * t;
  out[2] = from[2] + (to[2] - from[2]) * t;
  return out;
}

function smoothStep(low: number, high: number, value: number): number {
  const t = Math.min(1, Math.max(0, (value - low) / (high - low || 1)));
  return t * t * (3 - 2 * t);
}

/** A tint table with parsed colours, for per-cell lookups. */
type ParsedTints = {interpolate: boolean; elevations: number[]; colors: Rgb[]};

function parseTints(table: ReliefTintTable): ParsedTints {
  const stops = [...table.stops].sort((first, second) => first.elevation - second.elevation);
  if (!stops.length) throw new Error('Relief: empty tint table');
  return {
    interpolate: table.interpolate,
    elevations: stops.map(stop => stop.elevation),
    colors: stops.map(stop => parseHex(stop.color))
  };
}

/** Writes the tint colour of an elevation into `out` (0-255). */
function sampleParsedTints(tints: ParsedTints, elevation: number, out: Rgb): Rgb {
  const {elevations, colors} = tints;
  const last = elevations.length - 1;
  if (elevation <= elevations[0]) return mixRgb(colors[0], colors[0], 0, out);
  if (elevation >= elevations[last]) return mixRgb(colors[last], colors[last], 0, out);
  let index = 0;
  while (index < last && elevation >= elevations[index + 1]) index++;
  if (!tints.interpolate) return mixRgb(colors[index], colors[index], 0, out);
  const t = (elevation - elevations[index]) / (elevations[index + 1] - elevations[index]);
  return mixRgb(colors[index], colors[index + 1], t, out);
}

/**
 * The tint colour of an elevation in a table, as `[r, g, b]` 0-255 (for legends and tests).
 * Elevations below the first stop take the first colour, above the last the last.
 */
export function getReliefTintColor(
  tints: ReliefTintTable | keyof typeof RELIEF_TINT_TABLES,
  elevation: number
): [number, number, number] {
  const table = typeof tints === 'string' ? RELIEF_TINT_TABLES[tints] : tints;
  return sampleParsedTints(parseTints(table), elevation, [0, 0, 0]);
}

/** Ground size `[x, y]` in metres of one DEM cell. */
function getCellSizeMeters(dem: ReliefDem, option: ReliefOptions['cellSizeMeters']) {
  if (typeof option === 'number') return [option, option] as const;
  if (option) return [option[0], option[1]] as const;
  if (!dem.bounds) return [30, 30] as const;
  const [west, south, east, north] = dem.bounds;
  const centerLatitude = ((south + north) / 2) * (Math.PI / 180);
  return [
    (Math.abs(east - west) / dem.width) * METERS_PER_DEGREE_EQUATOR * Math.cos(centerLatitude),
    (Math.abs(north - south) / dem.height) * METERS_PER_DEGREE_LATITUDE
  ] as const;
}

/**
 * Composites a DEM into a relief image: a multi-directional Horn hillshade, toned from a cool
 * shade colour (`#5C6B8A`) through the paper to a warm lit colour (`#FFF4DE`), multiplied with the
 * stepped hypsometric tint at `tintStrength`, with contrast eased at low elevations (aerial
 * perspective), glaciers painted `#D5E6EF` (shaded), and the image edge feathered. Cells with no
 * data are transparent. The result is meant to be drawn first, in the deck canvas (see the
 * module doc for the compositing rule).
 *
 * The hillshade uses Horn's method on the 3x3 neighbourhood with the cell size in ground metres
 * (`cellSizeMeters`, or from `dem.bounds`), so slopes are true slopes at the raster's latitude.
 * Pixels are little-endian RGBA8, as the shader's `unpack4x8unorm` expects.
 */
export function buildReliefImage(dem: ReliefDem, options: ReliefOptions): ReliefImage {
  const {width, height, values} = dem;
  if (values.length < width * height) {
    throw new Error(`Relief: expected ${width}x${height} values, got ${values.length}`);
  }
  const dark = options.ground === 'dark';
  const tints = parseTints(
    typeof options.tints === 'string' ? RELIEF_TINT_TABLES[options.tints] : options.tints
  );
  const tintStrength = Math.min(1, Math.max(0, options.tintStrength ?? 0.3));
  const aerial =
    options.aerialPerspective === null
      ? null
      : (options.aerialPerspective ?? {from: 1800, to: 4200, strength: 0.35});
  const contrast = options.contrast ?? 1.2;
  const zFactor = options.zFactor ?? 1;
  const [cellX, cellY] = getCellSizeMeters(dem, options.cellSizeMeters);
  const altitude = ((options.altitudeDegrees ?? 40) * Math.PI) / 180;
  const cosZenith = Math.sin(altitude);
  const sinZenith = Math.cos(altitude);
  // Light directions as math angles (counter-clockwise from east), the Esri convention.
  const offsets = options.multidirectional === false ? [0] : [...MULTIDIRECTIONAL_OFFSETS];
  const weights = options.multidirectional === false ? [1] : [...MULTIDIRECTIONAL_WEIGHTS];
  const weightSum = weights.reduce((sum, weight) => sum + weight, 0);
  const lightCosines: number[] = [];
  const lightSines: number[] = [];
  for (const offset of offsets) {
    const math = ((360 - ((options.azimuthDegrees ?? 315) + offset) + 90) * Math.PI) / 180;
    lightCosines.push(Math.cos(math));
    lightSines.push(Math.sin(math));
  }
  const toneShade = parseHex(dark ? DARK_TONES.shade : (options.shadeColor ?? '#5C6B8A'));
  const toneFlat = parseHex(dark ? DARK_TONES.flat : LIGHT_PAPER);
  const toneLit = parseHex(dark ? DARK_TONES.light : (options.lightColor ?? '#FFF4DE'));
  const glacierColor = parseHex(options.glacierColor ?? '#D5E6EF');
  const glacierMask = options.glacierMask;
  const feather = Math.max(0, options.featherCells ?? 8);
  const noData = dem.noData;

  const pixels = new Uint32Array(width * height);
  const tone: Rgb = [0, 0, 0];
  const tint: Rgb = [0, 0, 0];
  const mixed: Rgb = [0, 0, 0];
  const isValid = (elevation: number) =>
    Number.isFinite(elevation) && (noData === undefined || elevation !== noData);

  for (let row = 0; row < height; row++) {
    const rowAbove = Math.max(0, row - 1) * width;
    const rowHere = row * width;
    const rowBelow = Math.min(height - 1, row + 1) * width;
    for (let column = 0; column < width; column++) {
      const centerElevation = values[rowHere + column];
      if (!isValid(centerElevation)) continue;
      const left = Math.max(0, column - 1);
      const right = Math.min(width - 1, column + 1);
      // Neighbours without data take the centre value, so a coast does not make a cliff.
      const read = (index: number) => {
        const elevation = values[index];
        return isValid(elevation) ? elevation : centerElevation;
      };
      const a = read(rowAbove + left);
      const b = read(rowAbove + column);
      const c = read(rowAbove + right);
      const d = read(rowHere + left);
      const f = read(rowHere + right);
      const g = read(rowBelow + left);
      const h = read(rowBelow + column);
      const i = read(rowBelow + right);
      // Horn: east-positive x, row-down y; spans of two cells (or one at the edges).
      const spanX = (right - left || 1) * cellX;
      const spanY = (Math.min(height - 1, row + 1) - Math.max(0, row - 1) || 1) * cellY;
      const slopeX = (c + 2 * f + i - (a + 2 * d + g)) / (4 * spanX);
      const slopeY = (g + 2 * h + i - (a + 2 * b + c)) / (4 * spanY);
      const gradient = zFactor * Math.hypot(slopeX, slopeY);
      const cosSlope = 1 / Math.sqrt(1 + gradient * gradient);
      const sinSlope = gradient * cosSlope;
      // Unit vector of the aspect (the downslope direction): cos = -dz/dx, sin = dz/dy (row-down).
      const cosAspect = gradient > 0 ? (-zFactor * slopeX) / gradient : 0;
      const sinAspect = gradient > 0 ? (zFactor * slopeY) / gradient : 0;
      let lit = 0;
      for (let k = 0; k < offsets.length; k++) {
        const shade =
          cosZenith * cosSlope +
          sinZenith * sinSlope * (lightCosines[k] * cosAspect + lightSines[k] * sinAspect);
        lit += weights[k] * Math.max(0, shade);
      }
      lit /= weightSum;

      // Contrast by height: haze lowers the shade contrast of low ground.
      const heightFactor = aerial
        ? 1 - aerial.strength * (1 - smoothStep(aerial.from, aerial.to, centerElevation))
        : 1;
      const delta = lit - cosZenith;
      const scaled =
        delta >= 0
          ? Math.min(1, (delta / (1 - cosZenith)) * contrast * heightFactor)
          : Math.max(-1, (delta / cosZenith) * contrast * heightFactor);
      const position =
        scaled >= 0 ? FLAT_POSITION + (1 - FLAT_POSITION) * scaled : FLAT_POSITION * (1 + scaled);

      // Tone ramp: shade -> flat (paper) -> lit.
      if (position < FLAT_POSITION) mixRgb(toneShade, toneFlat, position / FLAT_POSITION, tone);
      else mixRgb(toneFlat, toneLit, (position - FLAT_POSITION) / (1 - FLAT_POSITION), tone);

      // Multiply the tint in at its strength (tint 255 means no change).
      sampleParsedTints(tints, centerElevation, tint);
      for (let channel = 0; channel < 3; channel++) {
        tone[channel] *= 1 - tintStrength + (tintStrength * tint[channel]) / 255;
      }

      if (glacierMask && glacierMask[rowHere + column]) {
        const glacierShade = dark ? 0.35 + 0.3 * position : 0.78 + 0.22 * position;
        mixed[0] = glacierColor[0] * glacierShade;
        mixed[1] = glacierColor[1] * glacierShade;
        mixed[2] = glacierColor[2] * glacierShade;
        mixRgb(tone, mixed, 0.85, tone);
      }

      const red = dark ? Math.max(tone[0], DARK_FLOOR[0]) : tone[0];
      const green = dark ? Math.max(tone[1], DARK_FLOOR[1]) : tone[1];
      const blue = dark ? Math.max(tone[2], DARK_FLOOR[2]) : tone[2];
      let alpha = 255;
      if (feather > 0) {
        const edgeDistance = Math.min(column, row, width - 1 - column, height - 1 - row);
        alpha = Math.round(255 * smoothStep(0, feather, edgeDistance + 0.5));
      }
      pixels[rowHere + column] =
        (Math.min(255, Math.round(red)) |
          (Math.min(255, Math.round(green)) << 8) |
          (Math.min(255, Math.round(blue)) << 16) |
          (alpha << 24)) >>>
        0;
    }
  }
  return {pixels, width, height};
}

/** Raster layer props that draw a relief: spread them into a `SpatialAnalysisRasterLayer`. */
export type ReliefLayerProps = Pick<
  SpatialAnalysisRasterLayerProps,
  | 'gridSize'
  | 'bounds'
  | 'values'
  | 'valueFormat'
  | 'colormap'
  | 'rowOrigin'
  | 'opacity'
  | 'blending'
>;

/**
 * Uploads a relief image to a storage buffer and returns the props of the raster layer that draws
 * it as the underlay (draw it first): `values` (the buffer, which the caller owns and destroys
 * with the layer's resources), `valueFormat: 'uint32'`, `colormap: 'rgba'`, `gridSize`,
 * `rowOrigin: 'north'` (the DEM's row order), `opacity: 1` and `blending: 'normal'`. Add the layer
 * `id` and any `tessellation` (use 32-64 for rasters wider than about 100 km) yourself.
 *
 * @param bounds `[minX, minY, maxX, maxY]` of the image in the layers' planar metres.
 */
export function createReliefLayerProps(
  device: Device,
  image: ReliefImage,
  bounds: readonly [number, number, number, number],
  options: {id?: string} = {}
): ReliefLayerProps {
  const values = device.createBuffer({
    id: options.id ?? 'relief-pixels',
    usage: Buffer.STORAGE | Buffer.COPY_DST,
    data: image.pixels
  });
  return {
    values,
    valueFormat: 'uint32',
    colormap: 'rgba',
    gridSize: [image.width, image.height],
    bounds,
    rowOrigin: 'north',
    opacity: 1,
    blending: 'normal'
  };
}
