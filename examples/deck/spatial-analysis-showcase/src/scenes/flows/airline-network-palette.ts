// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {B3_PALETTE} from '../geometry/b3-palette';

/**
 * Categorical colour of the airline-network scene: communities, continents and the "between
 * groups" colour. The layers, the legends and the CPU colouring code all read this one list. This
 * module has no GPU imports so the scene file can use it without loading them.
 */
export const NETWORK_PALETTE: readonly (readonly [number, number, number, number])[] =
  B3_PALETTE.map(([red, green, blue]) => [red, green, blue, 255] as const);

/** Palette index of a route whose endpoints are in different groups. */
export const BETWEEN_GROUPS_INDEX = 7;

/** Palette index shared by every community outside the largest six. */
export const OTHER_GROUP_INDEX = 6;
