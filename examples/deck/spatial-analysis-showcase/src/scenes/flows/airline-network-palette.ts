// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {MapGround} from '../../cartography/hue-registry';
import type {PaletteColor} from '../../engine/ramps';
import {BETWEEN_GROUPS_INK, getGroupPalette} from './flows-style';

/**
 * Categorical colour of the airline-network scene: communities, continents and the "between
 * groups" ink. The layers, the legends and the CPU colouring code all read these tables. This
 * module has no GPU imports so the scene file can use it without loading them.
 *
 * Palette slots: `0` to `6` are the seven Okabe-Ito hues of the chapter palette, `7` is the neutral
 * grey "other" and `8` is the between-groups ink (a non-hue).
 */

/** Number of group hues (slots `0` to `6`). */
export const GROUP_HUE_COUNT = 7;

/** Palette slot of every group beyond the seven that own a hue. */
export const OTHER_GROUP_INDEX = 7;

/** Palette slot of a route whose endpoints are in different groups. */
export const BETWEEN_GROUPS_INDEX = 8;

/** Number of palette slots (seven hues, other, between). */
export const NETWORK_PALETTE_SIZE = 9;

/** Within-group routes: the share of the group colour kept as line alpha (additive light). */
export const WITHIN_ROUTE_ALPHA = 0.16;

/** Between-group routes: off-white at this alpha, drawn after the within-group routes. */
export const BETWEEN_ROUTE_ALPHA = 0.3;

/** Routes of a selected airport: their group colour at this alpha. */
export const EGO_ROUTE_ALPHA = 0.9;

/** Alpha multiplier of routes that are not the selected airport's. */
export const EGO_DIM_ALPHA = 0.1;

/** Airport disc outline: the night ground colour, 0.75 px. */
export const DISC_OUTLINE_COLOR: PaletteColor = [11, 14, 19, 255];

/** Width of the airport disc outline in CSS pixels. */
export const DISC_OUTLINE_PIXELS = 0.75;

/** Neutral airport disc of the unanalysed first step. */
export const NEUTRAL_NODE_INK: Record<MapGround, PaletteColor> = {
  dark: [222, 228, 238, 255],
  light: [52, 62, 76, 255]
};

/** Neutral route ink of the unanalysed first step (one colour, additive on the dark ground). */
export const NEUTRAL_ROUTE_INK: Record<MapGround, PaletteColor> = {
  dark: [188, 202, 224, 34],
  light: [60, 70, 84, 40]
};

/** Airport disc colours by palette slot (full alpha): hues, grey other, between ink. */
export function getNodePalette(ground: MapGround): PaletteColor[] {
  const between = BETWEEN_GROUPS_INK[ground];
  return [...getGroupPalette(ground, 255), [between[0], between[1], between[2], 255]];
}

/**
 * Route colours by palette slot: the group hue at {@link WITHIN_ROUTE_ALPHA} for slots `0` to `7`
 * and the between-groups ink at {@link BETWEEN_ROUTE_ALPHA} for slot `8`.
 */
export function getRoutePalette(ground: MapGround): PaletteColor[] {
  const between = BETWEEN_GROUPS_INK[ground];
  return [
    ...getGroupPalette(ground, Math.round(WITHIN_ROUTE_ALPHA * 255)),
    [between[0], between[1], between[2], Math.round(BETWEEN_ROUTE_ALPHA * 255)]
  ];
}
