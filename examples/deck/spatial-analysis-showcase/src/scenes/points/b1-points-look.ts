// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * The visual language the five `points` stories share (design sheet `FID/design/points-chapter.md`):
 * one warm amber for the observations a story is about, one neutral ghost for "all the rest", one
 * cool sky blue for a second population, the cartouche with the observer-effort chip, credits and
 * the one-line hand-off to the next story. Layer, legend and chart read these constants so a
 * colour means the same thing in every story of the chapter.
 */

import {CREDITS, joinCredits} from '../../cartography/credits';
import type {CartoucheSpec} from '../../cartography/types';

/** An RGBA colour, 0-255. */
export type PointsColor = readonly [number, number, number, number];

/** A ground: what `ctx.ground()` returns. */
export type PointsGround = 'light' | 'dark';

/**
 * The observations a story is about, as dots on the night ground: the registry `nature` dot
 * `#fec44f` (warm light that adds up under additive blending). Alpha is chosen per layer.
 */
export const SUBJECT_DOT_RGB = [254, 196, 79] as const;

/**
 * The same observations as ink on a paper ground: YlGnBu-7 class 6 `#225ea8`, the registry
 * `nature` hue on light grounds (darkest reads as "more").
 */
export const SUBJECT_INK_RGB = [34, 94, 168] as const;

/** A second population (Overture places, a CSR realisation): Okabe-Ito sky blue, lifted on dark. */
export const SECOND_DOT_RGB: Readonly<Record<PointsGround, readonly [number, number, number]>> = {
  dark: [124, 203, 245],
  light: [0, 114, 178]
};

/** The subject colour on a ground, with an alpha (0-255). */
export function getSubjectColor(ground: PointsGround, alpha = 255): PointsColor {
  const rgb = ground === 'dark' ? SUBJECT_DOT_RGB : SUBJECT_INK_RGB;
  return [rgb[0], rgb[1], rgb[2], alpha];
}

/** The second-population colour on a ground, with an alpha (0-255). */
export function getSecondColor(ground: PointsGround, alpha = 255): PointsColor {
  const rgb = SECOND_DOT_RGB[ground];
  return [rgb[0], rgb[1], rgb[2], alpha];
}

/**
 * "All the rest": the context tier of rule 7 (`#8a94a3` / `#6b7480`, alpha about 0.15), drawn
 * smaller than the subject and never in a hue.
 */
export function getGhostColor(ground: PointsGround, alpha?: number): PointsColor {
  return ground === 'dark' ? [138, 148, 163, alpha ?? 38] : [107, 116, 128, alpha ?? 46];
}

/**
 * Parameter geometry drawn by layers (window frames, r-circles, lassos): achromatic ink of the
 * ground, so it never competes with the data hue (consistency rule 3b).
 */
export function getParameterInk(ground: PointsGround, alpha = 230): PointsColor {
  return ground === 'dark' ? [232, 237, 242, alpha] : [31, 41, 51, alpha];
}

/** The halo under parameter geometry: the opposite luminance at half strength. */
export function getParameterHalo(ground: PointsGround, alpha = 128): PointsColor {
  return ground === 'dark' ? [9, 12, 16, alpha] : [255, 255, 255, alpha];
}

/** Sample line of the nature dataset (a standing cartouche line; the count is data, not prose). */
export const NATURE_SAMPLE = 'iNaturalist records, Chicago, 2023';

/**
 * The cartouche of one step: the claim or question (nine words or fewer), the variable line and,
 * for the observation stories, the observer-effort chip (consistency rule 15: stated once per
 * chapter, as a chip everywhere else).
 */
export function pointsCartouche(
  title: string,
  subtitle: string,
  options: {effort?: boolean; sample?: string} = {}
): CartoucheSpec {
  return {
    title,
    subtitle,
    ...(options.sample ? {sample: options.sample} : {}),
    ...(options.effort === false ? {} : {chips: ['Observer effort']})
  };
}

/** Overture Maps credit (not in the shared `CREDITS` table). */
export const OVERTURE_CREDIT = 'Overture Maps Foundation (CDLA-Permissive-2.0)';

/** CDC/ATSDR Social Vulnerability Index credit (tract population). */
export const SVI_CREDIT = 'CDC/ATSDR SVI 2022 (public domain)';

/** Credit lines of the chapter's datasets. */
export const POINTS_CREDITS = {
  nature: joinCredits(CREDITS.iNaturalist, CREDITS.cityOfChicago, CREDITS.naturalEarth),
  natureAndPlaces: joinCredits(
    CREDITS.iNaturalist,
    OVERTURE_CREDIT,
    CREDITS.cityOfChicago,
    CREDITS.naturalEarth
  ),
  roads: joinCredits(CREDITS.openStreetMap, CREDITS.cityOfChicago, SVI_CREDIT)
} as const;

/** The chapter order, for the hand-off line at the end of each story. */
export const POINTS_STORIES = [
  {id: 'nature-density', label: 'see the density'},
  {id: 'nature-clusters', label: 'name the hot spots'},
  {id: 'point-patterns', label: 'test them against chance'},
  {id: 'lasso-explorer', label: 'ask your own question'},
  {id: 'street-density', label: 'carry the method to lines'}
] as const;

/**
 * The closing hand-off of a story ("see it, name it, doubt it, ask it, carry it"): one markdown
 * line that links the next story in the chapter, or nothing after the last.
 */
export function nextStoryLine(currentId: string): string {
  const index = POINTS_STORIES.findIndex(story => story.id === currentId);
  const next = index >= 0 ? POINTS_STORIES[index + 1] : undefined;
  return next ? `Next: [${next.label}](#/story/${next.id}).` : '';
}
