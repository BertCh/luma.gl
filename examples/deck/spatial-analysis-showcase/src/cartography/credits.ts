// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Source credit constants with their licences. Use them in `furniture.credit` (a string replaces
 * the joined dataset attributions) and join several with {@link joinCredits}. The basemap's own
 * attribution stays in the map corner control.
 *
 * Licence reminders for any public deployment:
 * - iNaturalist observations are CC0, CC BY or CC BY-NC per observation, so a public deployment
 *   must stay non-commercial.
 * - OpenSky is non-commercial research data.
 * - OpenStreetMap is ODbL: attribution, and share-alike on derived databases.
 * - The CTA licence is limited (to promote public transportation) and must be credited.
 * - Copernicus and ESA WorldCover are CC BY 4.0 and must carry the notice.
 * `EFFORT_CAVEAT` lives in `hue-registry.ts`; it is not repeated here.
 */

/** Credit strings by source. Exact wording matters for the licences that require it. */
export const CREDITS = {
  openStreetMap: '© OpenStreetMap contributors (ODbL)',
  iNaturalist: 'iNaturalist contributors; CC0, CC BY, CC BY-NC (non-commercial use)',
  openSky: 'OpenSky Network, non-commercial research data',
  cta: 'Chicago Transit Authority GTFS (CTA licence)',
  cityOfChicago: 'City of Chicago Data Portal',
  usCensus: 'US Census Bureau (public domain)',
  naturalEarth: 'Made with Natural Earth',
  noaaNhc: 'NOAA National Hurricane Center (public domain)',
  usgs: 'US Geological Survey (public domain)',
  copernicus: 'Contains modified Copernicus data (CC BY 4.0)',
  esaWorldCover: '© ESA WorldCover project (CC BY 4.0)',
  carto: 'Basemap © CARTO © OpenStreetMap contributors',
  colorBrewer: 'Colours: ColorBrewer (Brewer, Harrower, Penn State), Apache 2.0',
  crameri: 'Scientific colour maps: Crameri (MIT)',
  kovesiCet: 'Perceptually uniform colour maps: Kovesi CET (CC BY 4.0)',
  okabeIto: 'Colour-blind-safe palette: Okabe and Ito'
} as const;

/** Key of {@link CREDITS}. */
export type CreditId = keyof typeof CREDITS;

/** One credit part: a string, a {@link CREDITS} key, a list of parts, or nothing (skipped). */
export type CreditPart = string | CreditId | readonly CreditPart[] | false | null | undefined;

/**
 * Joins credit parts with " · ", dropping empty parts and exact duplicates (first
 * occurrence wins). A part that is a key of {@link CREDITS} is replaced by its string, so
 * `joinCredits('usCensus', CREDITS.naturalEarth, show && 'extra')` works.
 *
 * @example
 * joinCredits(CREDITS.usCensus, 'usCensus', CREDITS.naturalEarth);
 * // 'US Census Bureau (public domain) · Made with Natural Earth'
 */
export function joinCredits(...parts: CreditPart[]): string {
  const seen = new Set<string>();
  const visit = (part: CreditPart) => {
    if (!part) return;
    if (typeof part !== 'string') {
      for (const nested of part) visit(nested);
      return;
    }
    const text = (CREDITS as Record<string, string>)[part] ?? part;
    const trimmed = text.trim();
    if (trimmed) seen.add(trimmed);
  };
  for (const part of parts) visit(part);
  return [...seen].join(' · ');
}
