// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ViewGround} from '../types';
import {hexToRgba, rgbToHex} from '../scales/class-table';
import {SELECTION_INK} from '../scales/hue-registry';
import {contrastRatio, VIEW_GROUND_COLOR} from './provenance';

export {SELECTION_INK};

/**
 * Neutral selection tokens derived from the showcase ink. Its blue-grey tokens are desaturated;
 * its ground-coloured halo becomes a contrasting grey edge so every returned stroke meets 3:1.
 */
export function getSelectionInk(ground: ViewGround): {core: string; halo: string; hover: string} {
  const neutralize = (hex: string) => {
    const [red, green, blue] = hexToRgba(hex);
    const gray = Math.round(red * 0.2126 + green * 0.7152 + blue * 0.0722);
    const neutral = rgbToHex([gray, gray, gray]);
    return contrastRatio(neutral, VIEW_GROUND_COLOR[ground]) >= 3 ? neutral : '#777777';
  };
  const core = neutralize(SELECTION_INK.core[ground]);
  return {core, halo: neutralize(SELECTION_INK.halo[ground]), hover: core};
}
