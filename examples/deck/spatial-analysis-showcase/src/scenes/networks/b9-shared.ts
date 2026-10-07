// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {sampleRamp, type RampName} from '../../engine/ramps';
import type {SceneTheme} from '../scene';

/** Splits a summary readback into consecutive sections of the given byte lengths. */
export function sliceSections(bytes: ArrayBuffer, byteLengths: readonly number[]): ArrayBuffer[] {
  const sections: ArrayBuffer[] = [];
  let offset = 0;
  for (const byteLength of byteLengths) {
    sections.push(bytes.slice(offset, offset + byteLength));
    offset += byteLength;
  }
  return sections;
}

/** Formats seconds as minutes with one decimal. */
export function formatMinutes(seconds: number): string {
  return Number.isFinite(seconds) ? `${(seconds / 60).toFixed(1)} min` : 'unreachable';
}

/** Formats an integer with thousands separators. */
export function formatInteger(value: number): string {
  return Math.round(value).toLocaleString('en-US');
}

/** Basemap-friendly road colors for the two themes. */
export function getRoadColors(theme: SceneTheme) {
  return theme === 'dark'
    ? {
        minor: [150, 160, 185, 55] as const,
        major: [190, 198, 220, 110] as const,
        halo: [10, 12, 20, 200] as const,
        text: [235, 238, 245, 255] as const
      }
    : {
        minor: [70, 80, 100, 55] as const,
        major: [50, 58, 80, 120] as const,
        halo: [255, 255, 255, 230] as const,
        text: [20, 24, 36, 255] as const
      };
}

/** One categorical color per facility hue (the layer palette holds eight). */
export const CATEGORY_PALETTE: readonly (readonly [number, number, number, number])[] = [
  [78, 201, 255, 255],
  [255, 148, 72, 255],
  [189, 122, 255, 255],
  [87, 235, 168, 255],
  [255, 105, 168, 255],
  [245, 220, 87, 255],
  [107, 158, 255, 255],
  [255, 92, 92, 255]
];

/** Named places used as defaults in the network stories (`[longitude, latitude]`). */
export const CHICAGO_PLACES = {
  willisTower: [-87.6359, 41.8789],
  ohare: [-87.9048, 41.9786],
  midway: [-87.7524, 41.7868],
  wrigleyField: [-87.6553, 41.9484],
  soldierField: [-87.6167, 41.8623],
  unitedCenter: [-87.6742, 41.8807],
  scienceIndustry: [-87.5831, 41.7906],
  loop: [-87.6298, 41.8819],
  southShore: [-87.5736, 41.7586],
  austin: [-87.7657, 41.8934],
  englewood: [-87.6446, 41.7798]
} as const satisfies Record<string, readonly [number, number]>;

/**
 * Packs a ramp into `size` rgba8 words (`r | g << 8 | b << 16 | alpha << 24`), the palette format
 * of the isoband triangle layer.
 */
export function createPackedPalette(ramp: RampName, size: number, alpha: number): Uint32Array {
  const palette = new Uint32Array(size);
  for (let index = 0; index < size; index++) {
    const [r, g, b] = sampleRamp(ramp, index / (size - 1));
    palette[index] = (r | (g << 8) | (b << 16) | (alpha << 24)) >>> 0;
  }
  return palette;
}

/** Color of isoband `band` of `breakCount + 1`, exactly as the isoband layer samples its palette. */
export function getBandColor(
  ramp: RampName,
  band: number,
  breakCount: number,
  alpha = 255
): [number, number, number, number] {
  const t = Math.min((band + 0.5) / (breakCount + 1), 0.9999);
  const [r, g, b] = sampleRamp(ramp, t);
  return [r, g, b, alpha];
}
