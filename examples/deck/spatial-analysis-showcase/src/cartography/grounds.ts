// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Ground presets (SYNTHESIS 1.1) and the pure resolution of a {@link BasemapSpec}. A scene writes
 * `basemap: ground('paperCity')` and both page themes are defined once. No DOM, no MapLibre: the
 * Deck host turns a {@link ResolvedBasemap} into styles (`engine/basemap-style.ts`).
 *
 * The data representation chooses the ground, not the page theme: a classed map sits on paper, an
 * additive glow on night, an ocean story on the abyss, an orbit on space and a terrain story on
 * relief. `ctx.ground()` reports the resulting luminance so ramps and ink follow it.
 */

import type {
  BasemapSpec,
  BasemapStyleName,
  CartographyTheme,
  GraticuleSpec,
  GroundName,
  GroundPalette,
  ResolvedBasemap
} from './types';

/** Default cross-fade length between grounds, in milliseconds. */
export const DEFAULT_GROUND_TRANSITION_MS = 600;

/** Graticule defaults: 10 degree lines, full strength, hairline. */
export const DEFAULT_GRATICULE: Required<GraticuleSpec> = {
  stepDegrees: 10,
  opacity: 1,
  widthPixels: 0.5
};

/** Background luminance below which a `none` ground counts as dark. */
const DARK_LUMINANCE_LIMIT = 0.2;

/** The label size range scenes may ask for. */
const LABEL_SCALE_RANGE: readonly [number, number] = [0.6, 1.4];

/** A parsed CSS colour: 0-255 channels, alpha 0-1. */
export type ParsedColor = {r: number; g: number; b: number; a: number};

/**
 * Parses `#rgb`, `#rrggbb`, `#rrggbbaa`, `rgb()` and `rgba()` (comma or space separated, optional
 * `/ alpha`) into channels. Returns `null` for anything else (named colours, `hsl()`, functions).
 */
export function parseCssColor(css: string | undefined): ParsedColor | null {
  if (!css) return null;
  const text = css.trim().toLowerCase();
  if (text.startsWith('#')) {
    const hex = text.slice(1);
    if (!/^[0-9a-f]+$/.test(hex)) return null;
    if (hex.length === 3 || hex.length === 4) {
      const [r, g, b, a] = [...hex].map(digit => parseInt(digit + digit, 16));
      return {r, g, b, a: a === undefined ? 1 : a / 255};
    }
    if (hex.length === 6 || hex.length === 8) {
      const channel = (index: number) => parseInt(hex.slice(index * 2, index * 2 + 2), 16);
      return {
        r: channel(0),
        g: channel(1),
        b: channel(2),
        a: hex.length === 8 ? channel(3) / 255 : 1
      };
    }
    return null;
  }
  const match = /^rgba?\(([^)]+)\)$/.exec(text);
  if (!match) return null;
  const parts = match[1]
    .split(/[\s,/]+/)
    .filter(Boolean)
    .map(Number);
  if (parts.length < 3 || parts.some(Number.isNaN)) return null;
  return {r: parts[0], g: parts[1], b: parts[2], a: parts.length > 3 ? parts[3] : 1};
}

/** Relative luminance (WCAG, 0 black to 1 white) of a CSS colour; `null` when it cannot be parsed. */
export function getRelativeLuminance(css: string | undefined): number | null {
  const color = parseCssColor(css);
  if (!color) return null;
  const linear = (value: number) => {
    const unit = value / 255;
    return unit <= 0.04045 ? unit / 12.92 : ((unit + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * linear(color.r) + 0.7152 * linear(color.g) + 0.0722 * linear(color.b);
}

/** Formats a parsed colour as `rgba(r, g, b, a)` with an optional alpha override. */
export function formatCssColor(color: ParsedColor, alpha: number = color.a): string {
  const round = (value: number) => Math.round(Math.min(Math.max(value, 0), 255));
  return `rgba(${round(color.r)}, ${round(color.g)}, ${round(color.b)}, ${Number(alpha.toFixed(3))})`;
}

/** Splits a `palette` field into its light and dark halves (a flat palette serves both). */
function splitPalette(palette: BasemapSpec['palette']): {
  light: GroundPalette;
  dark: GroundPalette;
} {
  if (!palette) return {light: {}, dark: {}};
  if ('light' in palette || 'dark' in palette) {
    const perTheme = palette as {light?: GroundPalette; dark?: GroundPalette};
    return {light: perTheme.light ?? {}, dark: perTheme.dark ?? {}};
  }
  return {light: palette as GroundPalette, dark: palette as GroundPalette};
}

/**
 * Resolves a basemap spec for a page theme into a complete {@link ResolvedBasemap}; `null` means
 * the defaults (today's look: positron or dark-matter by theme, labels in the basemap, no
 * recolouring). Every new field is defaulted: `palette` is the one for `theme`, `graticule: true`
 * becomes 10 degree hairlines, `transitionMs` is 600.
 */
export function resolveBasemap(spec: BasemapSpec | null, theme: CartographyTheme): ResolvedBasemap {
  const style = spec?.style ?? 'auto';
  const name: BasemapStyleName =
    style === 'auto'
      ? theme === 'dark'
        ? 'dark-matter'
        : 'positron'
      : typeof style === 'string'
        ? style
        : style[theme];
  const graticule = spec?.graticule;
  const [minimumScale, maximumScale] = LABEL_SCALE_RANGE;
  return {
    style: name,
    labels: spec?.labels ?? 'basemap',
    dim: Math.min(Math.max(spec?.dim ?? 0, 0), 1),
    desaturate: Math.min(Math.max(spec?.desaturate ?? 0, 0), 1),
    palette: splitPalette(spec?.palette)[theme],
    labelPreset: spec?.labelPreset ?? 'all',
    labelScale: Math.min(Math.max(spec?.labelScale ?? 1, minimumScale), maximumScale),
    suppressNames: spec?.suppressNames ?? [],
    world: spec?.world ?? false,
    graticule: !graticule
      ? null
      : graticule === true
        ? {...DEFAULT_GRATICULE}
        : {
            stepDegrees: Math.max(graticule.stepDegrees ?? DEFAULT_GRATICULE.stepDegrees, 1),
            opacity: Math.min(Math.max(graticule.opacity ?? DEFAULT_GRATICULE.opacity, 0), 1),
            widthPixels: Math.max(graticule.widthPixels ?? DEFAULT_GRATICULE.widthPixels, 0)
          },
    referenceLines: spec?.referenceLines ?? [],
    transitionMs: Math.max(spec?.transitionMs ?? DEFAULT_GROUND_TRANSITION_MS, 0)
  };
}

/**
 * Luminance of the ground a resolved basemap shows in a theme: `dark` for dark-matter, `light`
 * for positron and voyager. A `none` style follows its palette background (or land) colour: a
 * relative luminance below 0.2 is `dark` (so `abyss` and `space` are dark in the light theme), any
 * lighter colour is `light`; without a palette it follows the page theme (paper or ink).
 */
export function getBasemapGround(
  basemap: ResolvedBasemap,
  theme: CartographyTheme
): CartographyTheme {
  if (basemap.style === 'dark-matter') return 'dark';
  if (basemap.style !== 'none') return 'light';
  const luminance = getRelativeLuminance(basemap.palette.background ?? basemap.palette.land);
  if (luminance === null) return theme;
  return luminance < DARK_LUMINANCE_LIMIT ? 'dark' : 'light';
}

/** The ground luminance (`'light'` or `'dark'`) a spec gives in a page theme, as `ctx.ground()` reports. */
export function getGroundTone(spec: BasemapSpec | null, theme: CartographyTheme): CartographyTheme {
  return getBasemapGround(resolveBasemap(spec, theme), theme);
}

/**
 * The named ground presets, with the exact colours of SYNTHESIS 1.1 in both page themes.
 * Use through {@link ground}.
 */
export const GROUNDS: Record<GroundName, BasemapSpec> = {
  /**
   * Classed, diverging and significance maps at city scale (Chicago, NYC, Montreal). Positron or
   * dark-matter recoloured to paper: land `#F2F0EA` / `#14181C`, water `#D6E0E6` / `#0A0E13`,
   * parks `#E5EBDF` / `#1A2420`, roads off below z12 then a hairline `#E4E1DA` / `#222830` at
   * 0.6, quiet boundaries. Orientation labels above the data, basemap dimmed 0.3 and
   * desaturated 0.4. Draw the data fill at about 0.88 opacity so parks and the lake read through.
   */
  paperCity: {
    style: 'auto',
    labels: 'above',
    labelPreset: 'orientation',
    dim: 0.3,
    desaturate: 0.4,
    palette: {
      light: {
        background: '#F2F0EA',
        land: '#F2F0EA',
        water: '#D6E0E6',
        park: '#E5EBDF',
        road: '#E4E1DA',
        roadOpacity: 1,
        boundary: '#D9D5CC',
        boundaryOpacity: 0.8
      },
      dark: {
        background: '#14181C',
        land: '#14181C',
        water: '#0A0E13',
        park: '#1A2420',
        road: '#222830',
        roadOpacity: 0.6,
        boundary: '#2A313B',
        boundaryOpacity: 0.8
      }
    }
  },
  /**
   * National choropleths (US counties), matrices, diagrams and any map where tiles add only
   * noise. No tiles: a flat sheet `#F4F1EA` (dark `#14171C`, with land `#1B1F26` when `world` is
   * set), orientation labels from the theme's style above the data. Draw fills opaque (1.0) and
   * your own state lines. For the sheet-on-outside look of a national map override
   * `{world: 'land', palette: {light: {background: '#E4E8EA'}}}`.
   */
  paperSheet: {
    style: 'none',
    labels: 'above',
    labelPreset: 'orientation',
    palette: {
      light: {background: '#F4F1EA', land: '#F4F1EA', boundary: '#C9C4B9'},
      dark: {background: '#14171C', land: '#1B1F26', boundary: '#2A313B'}
    }
  },
  /**
   * Additive points, flows, trails and density glow (nature density, taxis, OSM history, AIS,
   * flights, wind). Dark-matter in BOTH themes (the shell frames it with a paper cartouche in the
   * light theme), land `#13171C`, water `#0A0E13`, boundary `#2A313B`, roads at 20 %. No basemap
   * labels (name places with annotations, or switch to `labels: 'above'` with the `orientation`
   * preset for five to seven names). Basemap dimmed 0.2 (set `dim: 0` for "dots meet the parks").
   */
  night: {
    style: 'dark-matter',
    labels: 'none',
    labelPreset: 'none',
    dim: 0.2,
    palette: {
      light: {
        background: '#13171C',
        land: '#13171C',
        water: '#0A0E13',
        park: '#13171C',
        road: '#2A313B',
        roadOpacity: 0.2,
        boundary: '#2A313B',
        boundaryOpacity: 1
      },
      dark: {
        background: '#13171C',
        land: '#13171C',
        water: '#0A0E13',
        park: '#13171C',
        road: '#2A313B',
        roadOpacity: 0.2,
        boundary: '#2A313B',
        boundaryOpacity: 1
      }
    }
  },
  /**
   * Oceans, storms, drifters, hurricanes, birds over sea and great circles. Forced dark, no
   * tiles: ocean `#0A1322`, Natural Earth land `#1A2231` with a coast stroke `#3B475E`, a 10
   * degree graticule in white at 0.06. No basemap labels (letter-spaced italic ocean names
   * `#6F7D96` are annotations). Needs the `natural-earth` dataset files; without them the map
   * shows the ocean colour and the graticule only.
   */
  abyss: {
    style: 'none',
    labels: 'none',
    labelPreset: 'none',
    world: 'land',
    graticule: true,
    palette: {
      light: {
        background: '#0A1322',
        land: '#1A2231',
        boundary: '#3B475E',
        boundaryOpacity: 1,
        graticule: 'rgba(255, 255, 255, 0.06)'
      },
      dark: {
        background: '#0A1322',
        land: '#1A2231',
        boundary: '#3B475E',
        boundaryOpacity: 1,
        graticule: 'rgba(255, 255, 255, 0.06)'
      }
    }
  },
  /**
   * Satellite stories. Forced dark, no tiles: `#04070D`, land `#10151E`, hairline coast
   * `#27303F`, a 30 degree graticule at 0.15, the equator at 0.30 and the tropics and polar
   * circles dashed at 0.20. No basemap labels. Needs the `natural-earth` dataset files; without
   * them the map shows the background and the graticule only.
   */
  space: {
    style: 'none',
    labels: 'none',
    labelPreset: 'none',
    world: 'land',
    graticule: {stepDegrees: 30, opacity: 1, widthPixels: 0.4},
    referenceLines: ['equator', 'tropics', 'polar-circles'],
    palette: {
      light: {
        background: '#04070D',
        land: '#10151E',
        boundary: '#27303F',
        boundaryOpacity: 1,
        graticule: 'rgba(255, 255, 255, 0.15)'
      },
      dark: {
        background: '#04070D',
        land: '#10151E',
        boundary: '#27303F',
        boundaryOpacity: 1,
        graticule: 'rgba(255, 255, 255, 0.15)'
      }
    }
  },
  /**
   * Terrain, hydrology, raster-with-DEM and Dixie stories, for scenes that draw the relief
   * underlay themselves. No tiles and no basemap labels (they are the scene's): paper `#F3EFE6`
   * in the light theme, `#14171C` in the dark theme. Draw the analysis raster at 0.62-0.88 over
   * the underlay.
   */
  relief: {
    style: 'none',
    labels: 'none',
    labelPreset: 'none',
    palette: {
      light: {background: '#F3EFE6', land: '#F3EFE6'},
      dark: {background: '#14171C', land: '#14171C'}
    }
  }
};

/**
 * A ground preset, optionally adjusted: `ground('paperCity', {dim: 0.5})`. The merge is shallow
 * (an override replaces the preset's field), except `palette`, which is merged per theme and per
 * field, so `{palette: {light: {water: '#CFE0EC'}}}` changes one colour in one theme.
 */
export function ground(name: GroundName, overrides?: BasemapSpec): BasemapSpec {
  const preset = GROUNDS[name];
  if (!overrides) return {...preset};
  const base = splitPalette(preset.palette);
  const extra = splitPalette(overrides.palette);
  const merged: BasemapSpec = {...preset, ...stripUndefined(overrides)};
  if (overrides.palette) {
    merged.palette = {
      light: {...base.light, ...extra.light},
      dark: {...base.dark, ...extra.dark}
    };
  }
  return merged;
}

/** Drops `undefined` fields so an override like `{dim: undefined}` keeps the preset's value. */
function stripUndefined<T extends object>(value: T): T {
  return Object.fromEntries(Object.entries(value).filter(([, field]) => field !== undefined)) as T;
}
