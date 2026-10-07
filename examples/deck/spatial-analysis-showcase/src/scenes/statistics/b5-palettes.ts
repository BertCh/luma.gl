// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Packs 0-255 channels into the rgba8 layout of the classification contributors (r in the low byte). */
export function packColor(r: number, g: number, b: number, a = 255): number {
  return ((r & 255) | ((g & 255) << 8) | ((b & 255) << 16) | ((a & 255) << 24)) >>> 0;
}

/** Piecewise-linear interpolation through `[r, g, b]` stops. */
export function sampleStops(
  stops: readonly (readonly number[])[],
  t: number
): [number, number, number] {
  const scaled = Math.min(Math.max(t, 0), 1) * (stops.length - 1);
  const index = Math.min(Math.floor(scaled), stops.length - 2);
  const fraction = scaled - index;
  const from = stops[index];
  const to = stops[index + 1];
  return [
    from[0] + (to[0] - from[0]) * fraction,
    from[1] + (to[1] - from[1]) * fraction,
    from[2] + (to[2] - from[2]) * fraction
  ];
}

/** `count` packed colors sampled evenly along stops. */
export function getStopPalette(
  stops: readonly (readonly number[])[],
  count: number,
  alpha = 255
): Uint32Array {
  const palette = new Uint32Array(count);
  for (let index = 0; index < count; index++) {
    const [r, g, b] = sampleStops(stops, count === 1 ? 0.5 : index / (count - 1));
    palette[index] = packColor(Math.round(r), Math.round(g), Math.round(b), alpha);
  }
  return palette;
}

/** CSS color of a packed rgba8. */
export function getPackedColorCss(packed: number): string {
  return `rgba(${packed & 255},${(packed >>> 8) & 255},${(packed >>> 16) & 255},${((packed >>> 24) & 255) / 255})`;
}

/** Named class-ramp palettes (5 stops) for the choropleth classes. */
export const CLASS_PALETTES: Record<
  string,
  {label: string; stops: readonly (readonly number[])[]}
> = {
  ylorrd: {
    label: 'Yellow-orange-red',
    stops: [
      [255, 255, 178],
      [254, 204, 92],
      [253, 141, 60],
      [240, 59, 32],
      [189, 0, 38]
    ]
  },
  viridis: {
    label: 'Viridis',
    stops: [
      [68, 1, 84],
      [59, 82, 139],
      [33, 145, 140],
      [94, 201, 98],
      [253, 231, 37]
    ]
  },
  blues: {
    label: 'Blues',
    stops: [
      [222, 235, 247],
      [158, 202, 225],
      [66, 146, 198],
      [8, 81, 156],
      [8, 48, 107]
    ]
  },
  spectral: {
    label: 'Spectral',
    stops: [
      [94, 79, 162],
      [102, 194, 165],
      [254, 224, 139],
      [244, 109, 67],
      [158, 1, 66]
    ]
  },
  magma: {
    label: 'Magma',
    stops: [
      [0, 0, 4],
      [81, 18, 124],
      [183, 55, 121],
      [252, 137, 97],
      [252, 253, 191]
    ]
  }
};

/** Palette select entries shared by the scenes. */
export const CLASS_PALETTE_OPTIONS = Object.entries(CLASS_PALETTES).map(([value, {label}]) => ({
  value,
  label
}));

/** Colors of the 10 iNaturalist groups, in dataset category order (colour-blind-aware set). */
export const NATURE_CATEGORY_COLORS: readonly (readonly [number, number, number, number])[] = [
  [0, 158, 115, 235], // Plants
  [0, 114, 178, 235], // Birds
  [240, 228, 66, 235], // Insects
  [213, 94, 0, 235], // Fungi
  [230, 159, 0, 235], // Mammals
  [204, 121, 167, 235], // Spiders and kin
  [120, 94, 240, 235], // Amphibians and reptiles
  [153, 153, 153, 235], // Snails and mussels
  [86, 180, 233, 235], // Fish
  [90, 90, 90, 235] // Other life
];
