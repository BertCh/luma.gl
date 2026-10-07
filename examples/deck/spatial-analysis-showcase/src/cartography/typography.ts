// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Map typography: the text roles of the cartography guide as constants, for SVG, canvas and
 * legend code that cannot read CSS custom properties directly. The same values back the CSS in
 * `styles.css` (`--font`, `--map-font-serif`, `--fs-*`). One sans (Source Sans 3) serves the app
 * chrome and the map furniture; Source Serif 4 is reserved for the cartouche title and for
 * hydrography and landform names.
 *
 * Rules: 11 px minimum for anything the reader must read (10 px only for credits and the
 * elevation figure of a landform); weight carries emphasis, tracking carries extent, case carries
 * class, italic carries hydrography and landform only; never italic uppercase.
 */

/** Font family key of a role: Source Sans 3 or Source Serif 4. */
export type MapFontFamily = 'sans' | 'serif';

/** Typographic recipe of one map text role. Sizes are CSS pixels. */
export type MapTextRole = {
  family: MapFontFamily;
  /** Font size in CSS pixels. */
  size: number;
  /** CSS font weight (400 to 700). */
  weight: number;
  style?: 'italic';
  transform?: 'uppercase';
  /** CSS letter spacing, for example `'0.14em'`. */
  letterSpacing?: string;
};

/** The names of the map text roles. */
export type MapTextRoleName =
  | 'cartoucheTitle'
  | 'cartoucheTitlePhone'
  | 'subtitle'
  | 'areaSmall'
  | 'areaMedium'
  | 'areaLarge'
  | 'settlementSubject'
  | 'settlementContext'
  | 'valueDetail'
  | 'waterSmall'
  | 'waterMedium'
  | 'waterLarge'
  | 'landform'
  | 'landformElevation'
  | 'noteHeadline'
  | 'noteBody'
  | 'legendTitle'
  | 'legendLabel'
  | 'scaleBar'
  | 'credit';

/** Font stacks, identical to `--font`, `--map-font` and `--map-font-serif` in `styles.css`. */
export const FONT_STACKS: Record<MapFontFamily, string> = {
  sans: "'Source Sans 3', 'Source Sans Pro', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif",
  serif: "'Source Serif 4', 'Source Serif Pro', Georgia, 'Times New Roman', serif"
};

/** The UI type scale in pixels, the values of `--fs-xs` to `--fs-hero`. */
export const TYPE_SCALE = {
  xs: 11,
  sm: 12,
  md: 13,
  base: 14,
  lg: 16,
  xl: 20,
  '2xl': 28,
  hero: 40
} as const;

/** Every map text role with its family, size, weight, style, case and tracking. */
export const MAP_TEXT_ROLES: Record<MapTextRoleName, MapTextRole> = {
  cartoucheTitle: {family: 'serif', size: 18, weight: 600},
  cartoucheTitlePhone: {family: 'serif', size: 15, weight: 600},
  subtitle: {family: 'sans', size: 12, weight: 400},
  // Area and region names: weight carries emphasis, tracking carries extent.
  areaSmall: {
    family: 'sans',
    size: 10.5,
    weight: 600,
    transform: 'uppercase',
    letterSpacing: '0.14em'
  },
  areaMedium: {
    family: 'sans',
    size: 12,
    weight: 600,
    transform: 'uppercase',
    letterSpacing: '0.14em'
  },
  areaLarge: {
    family: 'sans',
    size: 14.5,
    weight: 600,
    transform: 'uppercase',
    letterSpacing: '0.14em'
  },
  settlementSubject: {family: 'sans', size: 12.5, weight: 600},
  settlementContext: {family: 'sans', size: 11, weight: 500},
  valueDetail: {family: 'sans', size: 11, weight: 400},
  // Hydrography: serif italic in the water colour; the largest bodies are tracked out.
  waterSmall: {family: 'serif', size: 11.5, weight: 400, style: 'italic'},
  waterMedium: {family: 'serif', size: 13, weight: 400, style: 'italic'},
  waterLarge: {family: 'serif', size: 16, weight: 400, style: 'italic', letterSpacing: '0.12em'},
  landform: {family: 'serif', size: 12, weight: 400, style: 'italic'},
  landformElevation: {family: 'sans', size: 10, weight: 400},
  noteHeadline: {family: 'sans', size: 12.5, weight: 700},
  noteBody: {family: 'sans', size: 11.5, weight: 400},
  legendTitle: {
    family: 'sans',
    size: 11,
    weight: 700,
    transform: 'uppercase',
    letterSpacing: '0.06em'
  },
  legendLabel: {family: 'sans', size: 11, weight: 400},
  scaleBar: {family: 'sans', size: 10.5, weight: 600},
  credit: {family: 'sans', size: 10, weight: 400}
};

/**
 * CSS `font` shorthand of a role (style, weight, size, family), for canvas `ctx.font` and for
 * text measurement. Letter spacing and case are not part of `font`; apply them separately.
 */
export function getRoleFont(role: MapTextRoleName): string {
  const {family, size, weight, style} = MAP_TEXT_ROLES[role];
  return `${style ?? 'normal'} ${weight} ${size}px ${FONT_STACKS[family]}`;
}

/**
 * Resolves when the web fonts have loaded (`document.fonts.ready`), immediately outside a DOM.
 * Await it before measuring text for annotation boxes, and measure again on the `loadingdone`
 * event of `document.fonts` when a late face arrives.
 */
export function whenFontsReady(): Promise<void> {
  if (typeof document === 'undefined' || !document.fonts?.ready) return Promise.resolve();
  return document.fonts.ready.then(() => undefined);
}
