// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Pure (luma-free) pieces of the landforms scene: option state, the landform class palettes and
 * the scale presets, shared by the scene file and its compute module.
 */

import type {RampName} from '../../engine/ramps';

/** RGB, 0-255. */
export type ClassColor = readonly [number, number, number];

/** Curvature kinds of `GPUTerrainCurvature`, plus the multi-radius ring product. */
export type CurvatureChoice =
  | 'profile'
  | 'plan'
  | 'tangential'
  | 'mean'
  | 'gaussian'
  | 'minimal'
  | 'maximal'
  | 'unsphericity'
  | 'difference'
  | 'horizontal-excess'
  | 'vertical-excess'
  | 'accumulation'
  | 'ring'
  | 'rotor'
  | 'laplacian'
  | 'ring-multi-radius';

/** Products of the scene. */
export type LandformProduct =
  | 'curvature'
  | 'geomorphons'
  | 'tpi'
  | 'dev'
  | 'devmax'
  | 'scale'
  | 'weiss';

/** Option state of the landforms scene. */
export type LandformOptions = {
  product: LandformProduct;
  // Curvature
  curvatureKind: CurvatureChoice;
  curvatureMethod: 'evans-young' | 'zevenbergen-thorne' | 'florinsky';
  curvatureBorder: 'clamp' | 'nodata';
  flatGradient: number;
  ringRadiusInner: number;
  ringRadiusOuter: number;
  ringGainInner: number;
  ringGainOuter: number;
  ringSquash: boolean;
  zFactor: number;
  // Geomorphons
  geomorphonView: 'forms' | 'ternary';
  geomorphonRadius: number;
  geomorphonSkip: number;
  geomorphonComparison: 'anglev1' | 'anglev2' | 'anglev2-distance';
  geomorphonFlatAngle: number;
  geomorphonFlatDistance: number;
  // Topographic position
  scalePreset: 'fine' | 'landscape' | 'broad';
  scaleIndex: number;
  innerFraction: number;
  quantum: '4' | '64' | '256';
  // Weiss
  weissSmall: number;
  weissLarge: number;
  weissStandardization: 'global' | 'local';
  weissThreshold: number;
  weissSlope: number;
  // Display
  ramp: RampName;
  autoStretch: boolean;
  clipPercent: number;
  rangeScale: number;
  opacity: number;
  underlay: boolean;
};

/** GRASS `r.geomorphon` colors, class 1 to 10. */
export const GEOMORPHON_CLASSES: readonly {label: string; color: ClassColor; help: string}[] = [
  {label: 'Flat', color: [220, 220, 220], help: 'no significant relief in any direction'},
  {label: 'Peak', color: [56, 0, 0], help: 'higher than everything around'},
  {label: 'Ridge', color: [200, 0, 0], help: 'higher on both sides along a line'},
  {label: 'Shoulder', color: [255, 80, 20], help: 'convex break before a steep drop'},
  {label: 'Spur', color: [250, 210, 60], help: 'a ridge-like slope descending'},
  {label: 'Slope', color: [255, 255, 60], help: 'planar inclined surface'},
  {label: 'Hollow', color: [180, 230, 20], help: 'a valley-like slope descending'},
  {label: 'Footslope', color: [60, 250, 150], help: 'concave break at the base of a slope'},
  {label: 'Valley', color: [0, 0, 255], help: 'lower on both sides along a line'},
  {label: 'Pit', color: [0, 0, 56], help: 'lower than everything around'}
];

/** Weiss (2001) classes in `GPU_TERRAIN_WEISS_LANDFORMS` order. */
export const WEISS_CLASSES: readonly {label: string; color: ClassColor}[] = [
  {label: 'Canyon, deeply incised stream', color: [36, 0, 120]},
  {label: 'Midslope drainage, shallow valley', color: [40, 90, 200]},
  {label: 'Upland drainage, headwater', color: [110, 170, 230]},
  {label: 'U-shaped valley', color: [40, 170, 150]},
  {label: 'Plain', color: [240, 235, 170]},
  {label: 'Open slope', color: [220, 190, 110]},
  {label: 'Upper slope, mesa', color: [190, 140, 60]},
  {label: 'Local ridge, hill in valley', color: [240, 140, 90]},
  {label: 'Midslope ridge, small hill', color: [210, 70, 60]},
  {label: 'Mountain top, high ridge', color: [130, 0, 10]}
];

/** Scale sets of the multi-scale topographic position, window radii in pixels (6.6 m each). */
export const SCALE_PRESETS: Record<LandformOptions['scalePreset'], readonly number[]> = {
  fine: [1, 2, 3, 4, 6, 8, 12, 16],
  landscape: [2, 4, 8, 16, 32, 64, 128, 256],
  broad: [8, 16, 32, 64, 128, 256, 384, 512]
};

/** Ground meters per pixel at the central row of the alps-dem tile (display only). */
export const GROUND_PIXEL_METERS = 6.64;

/** One-line meaning of each curvature kind (for option help and tooltips). */
export const CURVATURE_KINDS: readonly {value: CurvatureChoice; label: string; help: string}[] = [
  {
    value: 'profile',
    label: 'Profile (down-slope)',
    help: 'Curvature in the direction of steepest descent: positive where the slope steepens downhill (convex), negative where it flattens (concave). Controls flow acceleration and erosion.'
  },
  {
    value: 'plan',
    label: 'Plan (across-slope)',
    help: 'Curvature of the contour lines: ridges diverge flow (positive), valleys converge it (negative).'
  },
  {
    value: 'tangential',
    label: 'Tangential',
    help: 'Plan curvature scaled by the slope; zero on flats.'
  },
  {
    value: 'mean',
    label: 'Mean',
    help: 'Average of the two principal curvatures; independent of direction.'
  },
  {
    value: 'gaussian',
    label: 'Gaussian',
    help: 'Product of the principal curvatures: positive on bowls and domes, negative on saddles.'
  },
  {
    value: 'minimal',
    label: 'Minimal',
    help: 'The smaller principal curvature (the most concave direction).'
  },
  {
    value: 'maximal',
    label: 'Maximal',
    help: 'The larger principal curvature (the most convex direction).'
  },
  {
    value: 'unsphericity',
    label: 'Unsphericity',
    help: 'How far the surface is from a sphere (0 at a dome or bowl).'
  },
  {
    value: 'difference',
    label: 'Difference',
    help: 'Half the difference between profile and tangential curvature.'
  },
  {
    value: 'horizontal-excess',
    label: 'Horizontal excess',
    help: 'Plan minus mean curvature; the departure from a sphere in the horizontal.'
  },
  {
    value: 'vertical-excess',
    label: 'Vertical excess',
    help: 'Profile minus mean curvature; the departure from a sphere in the vertical.'
  },
  {
    value: 'accumulation',
    label: 'Accumulation',
    help: 'Product of profile and plan curvature: where flow both converges and decelerates (deposition).'
  },
  {value: 'ring', label: 'Ring (excess product)', help: 'Product of the two excess curvatures.'},
  {value: 'rotor', label: 'Rotor', help: 'Twisting of the flow lines.'},
  {
    value: 'laplacian',
    label: 'Laplacian',
    help: 'Sum of the second derivatives; the classic convexity measure.'
  },
  {
    value: 'ring-multi-radius',
    label: 'Ring curvature, multi-radius (mt-image)',
    help: 'Weighted sum over rings of samples at the chosen radii: emphasises ridge and valley lines at the ring scales.'
  }
];
