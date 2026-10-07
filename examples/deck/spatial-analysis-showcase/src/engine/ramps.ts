// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * The single color-ramp table of the showcase. The WGSL in `layers.ts` is generated from
 * {@link RAMP_STOPS} and every legend reads the same table through {@link getRampCssGradient}, so
 * a map and its legend cannot drift apart.
 *
 * Stops are evenly spaced sRGB colors on `t` in `[0, 1]`, interpolated linearly. `viridis`,
 * `magma`, `inferno` and `cividis` are perceptually uniform and color-blind safe; `diverging` is a
 * ColorBrewer RdBu ramp (blue for low, near-white at the middle, red for high) that is also
 * color-blind safe. Use `diverging` with a symmetric `valueRange` so zero lands on the white stop.
 */

/** sRGB color with 0-255 channels. */
export type RampColor = readonly [number, number, number];

/** Names of the scalar color ramps. */
export type RampName = 'grayscale' | 'viridis' | 'magma' | 'inferno' | 'cividis' | 'diverging';

/** Stops of every scalar ramp, low value first. */
export const RAMP_STOPS: Record<RampName, readonly RampColor[]> = {
  grayscale: [
    [0, 0, 0],
    [255, 255, 255]
  ],
  viridis: [
    [71, 1, 85],
    [72, 24, 106],
    [71, 45, 123],
    [67, 64, 134],
    [61, 82, 140],
    [52, 99, 142],
    [43, 114, 142],
    [35, 129, 141],
    [31, 144, 139],
    [33, 159, 135],
    [42, 174, 128],
    [61, 188, 116],
    [90, 200, 97],
    [128, 211, 73],
    [172, 220, 48],
    [216, 226, 29],
    [252, 231, 33]
  ],
  magma: [
    [0, 0, 4],
    [24, 15, 61],
    [68, 15, 118],
    [114, 31, 129],
    [158, 47, 127],
    [205, 64, 113],
    [241, 96, 93],
    [253, 149, 103],
    [252, 253, 191]
  ],
  inferno: [
    [0, 0, 0],
    [11, 6, 44],
    [33, 9, 74],
    [59, 12, 93],
    [86, 17, 104],
    [112, 23, 108],
    [138, 31, 105],
    [162, 41, 96],
    [186, 54, 82],
    [207, 69, 62],
    [226, 88, 42],
    [241, 111, 24],
    [249, 138, 15],
    [250, 169, 19],
    [247, 203, 44],
    [243, 234, 93],
    [250, 255, 168]
  ],
  cividis: [
    [0, 34, 78],
    [18, 53, 112],
    [59, 73, 108],
    [87, 93, 109],
    [112, 113, 115],
    [138, 134, 120],
    [166, 157, 117],
    [196, 181, 108],
    [228, 207, 91],
    [254, 232, 56]
  ],
  diverging: [
    [5, 48, 97],
    [33, 102, 172],
    [67, 147, 195],
    [146, 197, 222],
    [209, 229, 240],
    [247, 247, 247],
    [253, 219, 199],
    [244, 165, 130],
    [214, 96, 77],
    [178, 24, 43],
    [103, 0, 31]
  ]
};

/** Every scalar ramp name. */
export const RAMP_NAMES = Object.keys(RAMP_STOPS) as RampName[];

/** Style-uniform colormap index of each non-scalar mode and ramp. Must match the WGSL dispatcher. */
export const COLORMAP_INDEXES = {
  uniform: 0,
  viridis: 1,
  inferno: 2,
  grayscale: 3,
  category: 4,
  mask: 5,
  magma: 6,
  cividis: 7,
  diverging: 8
} as const;

/** Samples a ramp at `t` in `[0, 1]`. Returns 0-255 channels. */
export function sampleRamp(name: RampName, t: number): [number, number, number] {
  const stops = RAMP_STOPS[name];
  const position = Math.min(Math.max(t, 0), 1) * (stops.length - 1);
  const index = Math.min(Math.floor(position), stops.length - 2);
  const fraction = position - index;
  const from = stops[index];
  const to = stops[index + 1];
  return [0, 1, 2].map(channel =>
    Math.round(from[channel] + (to[channel] - from[channel]) * fraction)
  ) as [number, number, number];
}

/**
 * CSS `linear-gradient` of a ramp, left to right.
 *
 * @param sqrtScale Matches `sqrtScale` on the layer: the color at normalized value `x` is the ramp
 *   color at `sqrt(x)`, so the gradient is sampled that way.
 */
export function getRampCssGradient(
  name: RampName,
  options: {sqrtScale?: boolean; direction?: string} = {}
): string {
  const samples = 24;
  const parts: string[] = [];
  for (let index = 0; index <= samples; index++) {
    const x = index / samples;
    const [r, g, b] = sampleRamp(name, options.sqrtScale ? Math.sqrt(x) : x);
    parts.push(`rgb(${r} ${g} ${b}) ${(x * 100).toFixed(1)}%`);
  }
  return `linear-gradient(${options.direction ?? 'to right'}, ${parts.join(', ')})`;
}

const formatChannel = (value: number) => (value / 255).toFixed(4);

/** Generates the WGSL ramp functions and the `colormap` dispatcher from {@link RAMP_STOPS}. */
export function getRampWgsl(): string {
  const functions = RAMP_NAMES.map(name => {
    const stops = RAMP_STOPS[name];
    const entries = stops
      .map(
        ([r, g, b]) => `vec3<f32>(${formatChannel(r)}, ${formatChannel(g)}, ${formatChannel(b)})`
      )
      .join(', ');
    return `
fn spatialAnalysisRamp_${name}(t: f32) -> vec3<f32> {
  var stops = array<vec3<f32>, ${stops.length}>(${entries});
  let x = clamp(t, 0.0, 1.0) * ${(stops.length - 1).toFixed(1)};
  let i = min(u32(floor(x)), ${stops.length - 2}u);
  return mix(stops[i], stops[i + 1u], x - f32(i));
}`;
  });
  const branches = RAMP_NAMES.map(
    name =>
      `  if (colormap == ${COLORMAP_INDEXES[name]}u) { return spatialAnalysisRamp_${name}(t); }`
  );
  return `${functions.join('\n')}

// Maps a normalized value to a ramp color; unknown colormaps fall back to grayscale.
fn spatialAnalysisSampleRamp(colormap: u32, t: f32) -> vec3<f32> {
${branches.join('\n')}
  return vec3<f32>(t);
}
`;
}
