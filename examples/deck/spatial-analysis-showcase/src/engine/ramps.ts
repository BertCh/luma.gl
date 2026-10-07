// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * The single color table of the showcase. The WGSL of every layer is generated from
 * {@link RAMP_STOPS} ({@link getRampWgsl}) and every legend reads the same table
 * ({@link getRampCssGradient}, {@link getClassColors}), so a map and its legend cannot drift apart.
 *
 * Stops are evenly spaced sRGB colors on `t` in `[0, 1]`, interpolated linearly, low value first.
 *
 * - Perceptual: `viridis`, `magma`, `inferno`, `cividis` (matplotlib, CC0) and the Crameri
 *   scientific maps `batlow`, `vik`, `roma`, `romao`, `oleron` (MIT, Crameri 2018,
 *   doi:10.5281/zenodo.1243862), sampled at 17 stops from the published 256-entry tables.
 * - ColorBrewer (Cynthia Brewer, Apache 2.0): sequential schemes at their 9-class stops and
 *   diverging schemes at their 11-class stops.
 * - `diverging` and `rdbu` are the same ColorBrewer RdBu ramp, blue (low) to red (high). `rdylbu`
 *   and `spectral` also run cool (low) to warm (high); the other diverging ramps run in the order
 *   of their name (`brbg`: brown low, teal high). Use `reverseRamp` / legend `reverse` to flip.
 * - `hypsometric`: land elevation tints (green lowlands, tan uplands, grey-white summits), the
 *   Wikipedia relief-map convention. `oleron` (Crameri) splits sea and land at its midpoint.
 * - `romao` and `twilight` are cyclic (first stop equals last) for aspect, direction, hour of day,
 *   day of year.
 * - Added for the fidelity round (17 stops sampled from the published 256-entry tables by
 *   `FID/foundation-2/colour-gen/gen_ramps.py`, never typed): `fire` (Kovesi CET-L3, black-red-
 *   yellow-white, additive dark maps), `lajolla` and `bamako` (Crameri sequentials), `berlin` and
 *   `vanimo` (Crameri diverging with a dark centre, for dark grounds), `mako` (seaborn, dark-ended
 *   blue-green), `twilight` (matplotlib, cyclic) and `isolum` (Kovesi CET-I1, isoluminant: for
 *   colour drawn over shaded relief because it does not alter the perceived relief).
 * - Ramp trim: every sampler takes `range: [t0, t1]`, applied AFTER normalising, `sqrt` and
 *   `reverse`, as `t0 + t * (t1 - t0)` (the layer prop `rampRange` does the same in WGSL).
 */

/** sRGB color with 0-255 channels. */
export type RampColor = readonly [number, number, number];

/** RGBA color with 0-255 channels (the layer `palette` and legend entry format). */
export type PaletteColor = readonly [number, number, number, number];

/** Names of the scalar color ramps. */
export type RampName =
  | 'grayscale'
  | 'viridis'
  | 'magma'
  | 'inferno'
  | 'cividis'
  | 'diverging'
  | 'blues'
  | 'greens'
  | 'purples'
  | 'oranges'
  | 'reds'
  | 'greys'
  | 'ylgnbu'
  | 'ylorrd'
  | 'ylorbr'
  | 'pubugn'
  | 'bupu'
  | 'orrd'
  | 'gnbu'
  | 'rdpu'
  | 'rdbu'
  | 'brbg'
  | 'puor'
  | 'piyg'
  | 'prgn'
  | 'rdylbu'
  | 'spectral'
  | 'batlow'
  | 'vik'
  | 'roma'
  | 'romao'
  | 'oleron'
  | 'hypsometric'
  | 'fire'
  | 'lajolla'
  | 'bamako'
  | 'berlin'
  | 'vanimo'
  | 'mako'
  | 'twilight'
  | 'isolum';

/** Measurement level a ramp is designed for. */
export type RampKind = 'sequential' | 'diverging' | 'cyclic';

/** Catalogue entry of a ramp. */
export type RampInfo = {
  /** Label for option lists. */
  label: string;
  kind: RampKind;
  /** Where the ramp comes from. */
  source: 'matplotlib' | 'colorbrewer' | 'crameri' | 'colorcet' | 'seaborn' | 'convention';
  /** Licence of the published table (see {@link getRampLicence} for the default of each source). */
  licence?: string;
  /** True when the ramp stays ordered and distinguishable under common color-vision deficiencies. */
  colorBlindSafe: boolean;
  /** Short note for authors (what the ends mean, when to use it). */
  note?: string;
};

/** Stops of every scalar ramp, low value first. The key order is the colormap-index order of new ramps. */
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
  ],
  blues: [
    [247, 251, 255],
    [222, 235, 247],
    [198, 219, 239],
    [158, 202, 225],
    [107, 174, 214],
    [66, 146, 198],
    [33, 113, 181],
    [8, 81, 156],
    [8, 48, 107]
  ],
  greens: [
    [247, 252, 245],
    [229, 245, 224],
    [199, 233, 192],
    [161, 217, 155],
    [116, 196, 118],
    [65, 171, 93],
    [35, 139, 69],
    [0, 109, 44],
    [0, 68, 27]
  ],
  purples: [
    [252, 251, 253],
    [239, 237, 245],
    [218, 218, 235],
    [188, 189, 220],
    [158, 154, 200],
    [128, 125, 186],
    [106, 81, 163],
    [84, 39, 143],
    [63, 0, 125]
  ],
  oranges: [
    [255, 245, 235],
    [254, 230, 206],
    [253, 208, 162],
    [253, 174, 107],
    [253, 141, 60],
    [241, 105, 19],
    [217, 72, 1],
    [166, 54, 3],
    [127, 39, 4]
  ],
  reds: [
    [255, 245, 240],
    [254, 224, 210],
    [252, 187, 161],
    [252, 146, 114],
    [251, 106, 74],
    [239, 59, 44],
    [203, 24, 29],
    [165, 15, 21],
    [103, 0, 13]
  ],
  greys: [
    [255, 255, 255],
    [240, 240, 240],
    [217, 217, 217],
    [189, 189, 189],
    [150, 150, 150],
    [115, 115, 115],
    [82, 82, 82],
    [37, 37, 37],
    [0, 0, 0]
  ],
  ylgnbu: [
    [255, 255, 217],
    [237, 248, 177],
    [199, 233, 180],
    [127, 205, 187],
    [65, 182, 196],
    [29, 145, 192],
    [34, 94, 168],
    [37, 52, 148],
    [8, 29, 88]
  ],
  ylorrd: [
    [255, 255, 204],
    [255, 237, 160],
    [254, 217, 118],
    [254, 178, 76],
    [253, 141, 60],
    [252, 78, 42],
    [227, 26, 28],
    [189, 0, 38],
    [128, 0, 38]
  ],
  ylorbr: [
    [255, 255, 229],
    [255, 247, 188],
    [254, 227, 145],
    [254, 196, 79],
    [254, 153, 41],
    [236, 112, 20],
    [204, 76, 2],
    [153, 52, 4],
    [102, 37, 6]
  ],
  pubugn: [
    [255, 247, 251],
    [236, 226, 240],
    [208, 209, 230],
    [166, 189, 219],
    [103, 169, 207],
    [54, 144, 192],
    [2, 129, 138],
    [1, 108, 89],
    [1, 70, 54]
  ],
  bupu: [
    [247, 252, 253],
    [224, 236, 244],
    [191, 211, 230],
    [158, 188, 218],
    [140, 150, 198],
    [140, 107, 177],
    [136, 65, 157],
    [129, 15, 124],
    [77, 0, 75]
  ],
  orrd: [
    [255, 247, 236],
    [254, 232, 200],
    [253, 212, 158],
    [253, 187, 132],
    [252, 141, 89],
    [239, 101, 72],
    [215, 48, 31],
    [179, 0, 0],
    [127, 0, 0]
  ],
  gnbu: [
    [247, 252, 240],
    [224, 243, 219],
    [204, 235, 197],
    [168, 221, 181],
    [123, 204, 196],
    [78, 179, 211],
    [43, 140, 190],
    [8, 104, 172],
    [8, 64, 129]
  ],
  rdpu: [
    [255, 247, 243],
    [253, 224, 221],
    [252, 197, 192],
    [250, 159, 181],
    [247, 104, 161],
    [221, 52, 151],
    [174, 1, 126],
    [122, 1, 119],
    [73, 0, 106]
  ],
  rdbu: [
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
  ],
  brbg: [
    [84, 48, 5],
    [140, 81, 10],
    [191, 129, 45],
    [223, 194, 125],
    [246, 232, 195],
    [245, 245, 245],
    [199, 234, 229],
    [128, 205, 193],
    [53, 151, 143],
    [1, 102, 94],
    [0, 60, 48]
  ],
  puor: [
    [45, 0, 75],
    [84, 39, 136],
    [128, 115, 172],
    [178, 171, 210],
    [216, 218, 235],
    [247, 247, 247],
    [254, 224, 182],
    [253, 184, 99],
    [224, 130, 20],
    [179, 88, 6],
    [127, 59, 8]
  ],
  piyg: [
    [142, 1, 82],
    [197, 27, 125],
    [222, 119, 174],
    [241, 182, 218],
    [253, 224, 239],
    [247, 247, 247],
    [230, 245, 208],
    [184, 225, 134],
    [127, 188, 65],
    [77, 146, 33],
    [39, 100, 25]
  ],
  prgn: [
    [64, 0, 75],
    [118, 42, 131],
    [153, 112, 171],
    [194, 165, 207],
    [231, 212, 232],
    [247, 247, 247],
    [217, 240, 211],
    [166, 219, 160],
    [90, 174, 97],
    [27, 120, 55],
    [0, 68, 27]
  ],
  rdylbu: [
    [49, 54, 149],
    [69, 117, 180],
    [116, 173, 209],
    [171, 217, 233],
    [224, 243, 248],
    [255, 255, 191],
    [254, 224, 144],
    [253, 174, 97],
    [244, 109, 67],
    [215, 48, 39],
    [165, 0, 38]
  ],
  spectral: [
    [94, 79, 162],
    [50, 136, 189],
    [102, 194, 165],
    [171, 221, 164],
    [230, 245, 152],
    [255, 255, 191],
    [254, 224, 139],
    [253, 174, 97],
    [244, 109, 67],
    [213, 62, 79],
    [158, 1, 66]
  ],
  batlow: [
    [1, 25, 89],
    [13, 49, 93],
    [17, 67, 96],
    [22, 82, 98],
    [34, 96, 97],
    [53, 106, 89],
    [77, 115, 77],
    [102, 122, 63],
    [129, 130, 50],
    [160, 138, 43],
    [191, 144, 53],
    [219, 149, 76],
    [242, 157, 108],
    [252, 168, 145],
    [253, 180, 180],
    [252, 192, 214],
    [250, 204, 250]
  ],
  vik: [
    [0, 18, 97],
    [2, 43, 113],
    [3, 68, 129],
    [12, 94, 146],
    [47, 124, 166],
    [96, 157, 188],
    [146, 189, 210],
    [197, 219, 229],
    [236, 229, 225],
    [233, 203, 185],
    [220, 171, 143],
    [207, 142, 104],
    [195, 114, 67],
    [179, 83, 31],
    [147, 46, 6],
    [116, 21, 6],
    [89, 0, 8]
  ],
  roma: [
    [126, 23, 0],
    [142, 60, 12],
    [157, 88, 24],
    [169, 114, 35],
    [182, 140, 50],
    [196, 170, 74],
    [208, 201, 113],
    [209, 225, 159],
    [193, 234, 194],
    [161, 228, 212],
    [120, 210, 215],
    [82, 184, 208],
    [57, 157, 198],
    [43, 131, 188],
    [34, 106, 177],
    [24, 79, 165],
    [3, 49, 152]
  ],
  romao: [
    [115, 57, 87],
    [126, 57, 66],
    [139, 68, 51],
    [153, 88, 44],
    [170, 116, 47],
    [189, 151, 65],
    [206, 187, 101],
    [214, 214, 143],
    [204, 225, 178],
    [177, 221, 199],
    [140, 204, 207],
    [106, 178, 203],
    [84, 148, 192],
    [79, 118, 173],
    [88, 89, 146],
    [102, 68, 116],
    [114, 57, 89]
  ],
  oleron: [
    [26, 38, 89],
    [50, 63, 114],
    [76, 88, 139],
    [103, 115, 166],
    [131, 144, 195],
    [160, 173, 223],
    [187, 200, 243],
    [209, 222, 250],
    [128, 159, 128],
    [63, 87, 0],
    [97, 100, 9],
    [134, 120, 43],
    [168, 143, 79],
    [204, 170, 115],
    [236, 199, 155],
    [248, 226, 192],
    [253, 253, 230]
  ],
  hypsometric: [
    [172, 208, 165],
    [148, 191, 139],
    [168, 198, 143],
    [189, 204, 150],
    [209, 215, 171],
    [225, 228, 181],
    [239, 235, 192],
    [232, 225, 182],
    [222, 214, 163],
    [211, 202, 157],
    [202, 185, 130],
    [195, 167, 107],
    [185, 152, 90],
    [170, 135, 83],
    [172, 154, 124],
    [186, 174, 154],
    [202, 195, 184],
    [224, 222, 216],
    [245, 244, 242]
  ],
  fire: [
    [0, 0, 0],
    [50, 0, 0],
    [74, 1, 0],
    [99, 1, 0],
    [125, 2, 0],
    [152, 4, 0],
    [180, 6, 0],
    [208, 10, 0],
    [237, 20, 0],
    [251, 63, 0],
    [255, 104, 0],
    [255, 136, 0],
    [255, 165, 1],
    [255, 192, 4],
    [255, 217, 9],
    [255, 242, 31],
    [255, 255, 255]
  ],
  lajolla: [
    [25, 25, 0],
    [39, 30, 8],
    [55, 36, 17],
    [76, 43, 28],
    [102, 52, 41],
    [133, 61, 56],
    [166, 70, 68],
    [197, 79, 74],
    [217, 95, 78],
    [225, 116, 79],
    [229, 135, 81],
    [233, 154, 82],
    [237, 173, 84],
    [241, 193, 89],
    [247, 217, 114],
    [252, 239, 159],
    [255, 254, 203]
  ],
  bamako: [
    [0, 59, 71],
    [8, 63, 67],
    [16, 69, 62],
    [25, 75, 56],
    [37, 82, 49],
    [50, 91, 41],
    [64, 100, 32],
    [81, 111, 22],
    [98, 122, 11],
    [117, 132, 1],
    [138, 137, 0],
    [159, 145, 8],
    [180, 161, 34],
    [202, 178, 64],
    [221, 196, 101],
    [239, 213, 138],
    [255, 229, 173]
  ],
  berlin: [
    [158, 176, 255],
    [121, 171, 237],
    [82, 159, 211],
    [54, 134, 173],
    [40, 104, 134],
    [29, 75, 97],
    [20, 49, 63],
    [17, 26, 32],
    [25, 12, 9],
    [42, 14, 1],
    [64, 18, 1],
    [90, 28, 7],
    [124, 51, 29],
    [156, 80, 61],
    [188, 110, 97],
    [221, 141, 134],
    [255, 173, 173]
  ],
  vanimo: [
    [255, 205, 253],
    [230, 160, 220],
    [205, 120, 189],
    [179, 88, 160],
    [146, 63, 128],
    [106, 42, 92],
    [65, 27, 55],
    [36, 20, 30],
    [26, 21, 19],
    [28, 31, 17],
    [41, 54, 22],
    [61, 83, 29],
    [81, 112, 38],
    [102, 140, 49],
    [126, 172, 69],
    [156, 210, 108],
    [190, 253, 165]
  ],
  mako: [
    [11, 4, 5],
    [28, 16, 28],
    [43, 27, 52],
    [55, 39, 79],
    [62, 53, 107],
    [65, 67, 134],
    [60, 85, 152],
    [55, 104, 159],
    [53, 122, 163],
    [52, 140, 167],
    [53, 158, 170],
    [59, 176, 173],
    [74, 193, 173],
    [104, 209, 173],
    [151, 221, 182],
    [190, 232, 202],
    [222, 245, 229]
  ],
  twilight: [
    [226, 217, 226],
    [194, 206, 212],
    [148, 180, 199],
    [114, 150, 193],
    [98, 117, 186],
    [94, 80, 172],
    [89, 42, 143],
    [69, 19, 92],
    [48, 20, 55],
    [75, 19, 66],
    [117, 30, 79],
    [153, 53, 80],
    [178, 87, 82],
    [195, 125, 99],
    [204, 164, 137],
    [217, 199, 191],
    [226, 217, 226]
  ],
  isolum: [
    [55, 183, 236],
    [64, 184, 222],
    [72, 185, 208],
    [79, 186, 194],
    [87, 187, 179],
    [96, 187, 164],
    [106, 187, 148],
    [119, 186, 132],
    [135, 184, 118],
    [152, 181, 107],
    [168, 177, 100],
    [184, 173, 95],
    [198, 168, 93],
    [212, 162, 94],
    [224, 157, 98],
    [236, 151, 103],
    [246, 144, 109]
  ]
};

/** Catalogue of every ramp: label, measurement level, source, color-blind safety. */
export const RAMP_INFO: Record<RampName, RampInfo> = {
  grayscale: {label: 'Grayscale', kind: 'sequential', source: 'convention', colorBlindSafe: true},
  viridis: {label: 'Viridis', kind: 'sequential', source: 'matplotlib', colorBlindSafe: true},
  magma: {label: 'Magma', kind: 'sequential', source: 'matplotlib', colorBlindSafe: true},
  inferno: {label: 'Inferno', kind: 'sequential', source: 'matplotlib', colorBlindSafe: true},
  cividis: {
    label: 'Cividis',
    kind: 'sequential',
    source: 'matplotlib',
    colorBlindSafe: true,
    note: 'Optimised for color-vision deficiency.'
  },
  diverging: {
    label: 'Red-blue (diverging)',
    kind: 'diverging',
    source: 'colorbrewer',
    colorBlindSafe: true,
    note: 'Alias of rdbu: blue low, red high.'
  },
  blues: {label: 'Blues', kind: 'sequential', source: 'colorbrewer', colorBlindSafe: true},
  greens: {label: 'Greens', kind: 'sequential', source: 'colorbrewer', colorBlindSafe: true},
  purples: {label: 'Purples', kind: 'sequential', source: 'colorbrewer', colorBlindSafe: true},
  oranges: {label: 'Oranges', kind: 'sequential', source: 'colorbrewer', colorBlindSafe: true},
  reds: {label: 'Reds', kind: 'sequential', source: 'colorbrewer', colorBlindSafe: true},
  greys: {label: 'Greys', kind: 'sequential', source: 'colorbrewer', colorBlindSafe: true},
  ylgnbu: {
    label: 'Yellow-green-blue',
    kind: 'sequential',
    source: 'colorbrewer',
    colorBlindSafe: true
  },
  ylorrd: {
    label: 'Yellow-orange-red',
    kind: 'sequential',
    source: 'colorbrewer',
    colorBlindSafe: true
  },
  ylorbr: {
    label: 'Yellow-orange-brown',
    kind: 'sequential',
    source: 'colorbrewer',
    colorBlindSafe: true
  },
  pubugn: {
    label: 'Purple-blue-green',
    kind: 'sequential',
    source: 'colorbrewer',
    colorBlindSafe: true
  },
  bupu: {label: 'Blue-purple', kind: 'sequential', source: 'colorbrewer', colorBlindSafe: true},
  orrd: {label: 'Orange-red', kind: 'sequential', source: 'colorbrewer', colorBlindSafe: true},
  gnbu: {label: 'Green-blue', kind: 'sequential', source: 'colorbrewer', colorBlindSafe: true},
  rdpu: {label: 'Red-purple', kind: 'sequential', source: 'colorbrewer', colorBlindSafe: true},
  rdbu: {
    label: 'Red-blue',
    kind: 'diverging',
    source: 'colorbrewer',
    colorBlindSafe: true,
    note: 'Blue low, red high (same as diverging).'
  },
  brbg: {
    label: 'Brown-teal',
    kind: 'diverging',
    source: 'colorbrewer',
    colorBlindSafe: true,
    note: 'Brown low, teal high: dry to wet, loss to gain.'
  },
  puor: {
    label: 'Purple-orange',
    kind: 'diverging',
    source: 'colorbrewer',
    colorBlindSafe: true,
    note: 'Purple low, orange high.'
  },
  piyg: {
    label: 'Pink-green',
    kind: 'diverging',
    source: 'colorbrewer',
    colorBlindSafe: true,
    note: 'Pink low, green high.'
  },
  prgn: {
    label: 'Purple-green',
    kind: 'diverging',
    source: 'colorbrewer',
    colorBlindSafe: true,
    note: 'Purple low, green high.'
  },
  rdylbu: {
    label: 'Red-yellow-blue',
    kind: 'diverging',
    source: 'colorbrewer',
    colorBlindSafe: true,
    note: 'Blue low, red high, a light yellow midpoint.'
  },
  spectral: {
    label: 'Spectral',
    kind: 'diverging',
    source: 'colorbrewer',
    colorBlindSafe: false,
    note: 'Blue low, red high. Many hues: avoid for color-blind audiences.'
  },
  batlow: {
    label: 'Batlow',
    kind: 'sequential',
    source: 'crameri',
    colorBlindSafe: true,
    note: 'Multi-hue, perceptually uniform, readable in greyscale.'
  },
  vik: {
    label: 'Vik',
    kind: 'diverging',
    source: 'crameri',
    colorBlindSafe: true,
    note: 'Blue low, brown-red high, light midpoint.'
  },
  roma: {
    label: 'Roma',
    kind: 'diverging',
    source: 'crameri',
    colorBlindSafe: true,
    note: 'Red-brown low, blue high, light yellow midpoint.'
  },
  romao: {
    label: 'Roma O (cyclic)',
    kind: 'cyclic',
    source: 'crameri',
    colorBlindSafe: true,
    note: 'Cyclic: aspect, wind direction, hour of day. Map 0 and 360 degrees to t = 0 and 1.'
  },
  oleron: {
    label: 'Oleron (sea and land)',
    kind: 'diverging',
    source: 'crameri',
    colorBlindSafe: true,
    note: 'Bathymetry below the midpoint, topography above: use a range symmetric about sea level.'
  },
  hypsometric: {
    label: 'Hypsometric tints',
    kind: 'sequential',
    source: 'convention',
    colorBlindSafe: false,
    note: 'Land elevation: green lowlands, tan uplands, grey-white summits.'
  },
  fire: {
    label: 'Fire (CET-L3)',
    kind: 'sequential',
    source: 'colorcet',
    licence: 'CC BY 4.0 (Kovesi)',
    colorBlindSafe: true,
    note: 'Black, red, yellow, white: the dark-ground ramp for additive density. Brightest = highest. Protan/deutan viewers see it ordered by lightness.'
  },
  lajolla: {
    label: 'La Jolla',
    kind: 'sequential',
    source: 'crameri',
    licence: 'MIT (Crameri)',
    colorBlindSafe: true,
    note: 'Near-black brown, red-orange, cream. Warm single family for intensity (low end dark).'
  },
  bamako: {
    label: 'Bamako',
    kind: 'sequential',
    source: 'crameri',
    licence: 'MIT (Crameri)',
    colorBlindSafe: true,
    note: 'Dark teal, olive, cream. Vegetation and nature counts (low end dark).'
  },
  berlin: {
    label: 'Berlin (dark centre)',
    kind: 'diverging',
    source: 'crameri',
    licence: 'MIT (Crameri)',
    colorBlindSafe: true,
    note: 'Light blue low, near-black centre, light red high: the diverging ramp for dark grounds.'
  },
  vanimo: {
    label: 'Vanimo (dark centre)',
    kind: 'diverging',
    source: 'crameri',
    licence: 'MIT (Crameri)',
    colorBlindSafe: true,
    note: 'Pink low, near-black centre, green high: dark-ground diverging, red-green free in lightness.'
  },
  mako: {
    label: 'Mako',
    kind: 'sequential',
    source: 'seaborn',
    licence: 'BSD-3-Clause / CC0 (seaborn, van der Walt and Smith)',
    colorBlindSafe: true,
    note: 'Near-black, blue, pale mint. Cool counterpart of magma for dark grounds; trim the first 15 %.'
  },
  twilight: {
    label: 'Twilight (cyclic)',
    kind: 'cyclic',
    source: 'matplotlib',
    licence: 'CC0 (matplotlib)',
    colorBlindSafe: true,
    note: 'Cyclic, light at the seam and dark at the opposite side: hour of day, month, bearing.'
  },
  isolum: {
    label: 'Isoluminant (CET-I1)',
    kind: 'sequential',
    source: 'colorcet',
    licence: 'CC BY 4.0 (Kovesi)',
    colorBlindSafe: false,
    note: 'Constant lightness (hue only): colour over hillshade or relief without hiding it. Not readable in greyscale, so label the ends.'
  }
};

/**
 * The six ramps of the original table, in their original order. Kept so option lists built from
 * it do not grow; use {@link ALL_RAMP_NAMES} or {@link getRampOptions} for the full catalogue.
 */
export const RAMP_NAMES: readonly RampName[] = [
  'grayscale',
  'viridis',
  'magma',
  'inferno',
  'cividis',
  'diverging'
];

/** Every ramp name, in colormap-index order. */
export const ALL_RAMP_NAMES = Object.keys(RAMP_STOPS) as RampName[];

/** Ramp names of one measurement level. */
export function getRampNames(kind: RampKind): RampName[] {
  return ALL_RAMP_NAMES.filter(name => RAMP_INFO[name].kind === kind && name !== 'diverging');
}

/**
 * `{value, label}` choices for a `select` option, for example
 * `options: getRampOptions(['viridis', 'batlow', 'ylgnbu'])` or `getRampOptions('diverging')`.
 */
export function getRampOptions(
  names: RampKind | readonly RampName[]
): {value: RampName; label: string; help?: string}[] {
  const list = typeof names === 'string' ? getRampNames(names) : names;
  return list.map(name => ({
    value: name,
    label: RAMP_INFO[name].label,
    ...(RAMP_INFO[name].note ? {help: RAMP_INFO[name].note} : {})
  }));
}

const LEGACY_INDEXES = {
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

function buildColormapIndexes(): Record<'uniform' | 'category' | 'mask' | RampName, number> {
  const indexes: Record<string, number> = {...LEGACY_INDEXES};
  let next = 9;
  for (const name of ALL_RAMP_NAMES) {
    if (!(name in indexes)) indexes[name] = next++;
  }
  return indexes as Record<'uniform' | 'category' | 'mask' | RampName, number>;
}

/**
 * Style-uniform colormap index of each non-scalar mode and ramp. The first nine are the original
 * indexes (layers written against them keep working); new ramps follow from 9.
 */
export const COLORMAP_INDEXES = buildColormapIndexes();

/**
 * Samples a ramp at `t` in `[0, 1]` (flipped when `reverse`). Returns 0-255 channels.
 *
 * @param range Optional trim `[t0, t1]` (same semantics as the layer prop `rampRange`): `t` is
 *   clamped, reversed if asked, then mapped to `t0 + t * (t1 - t0)` before sampling, so
 *   `range: [0.15, 1]` keeps a dark ramp's lowest value off pure black.
 */
export function sampleRamp(
  name: RampName,
  t: number,
  reverse = false,
  range?: readonly [number, number]
): [number, number, number] {
  const stops = RAMP_STOPS[name];
  let position01 = Math.min(Math.max(t, 0), 1);
  if (reverse) position01 = 1 - position01;
  if (range) position01 = range[0] + position01 * (range[1] - range[0]);
  const position = Math.min(Math.max(position01, 0), 1) * (stops.length - 1);
  const index = Math.min(Math.floor(position), stops.length - 2);
  const fraction = position - index;
  const from = stops[index];
  const to = stops[index + 1];
  return [0, 1, 2].map(channel =>
    Math.round(from[channel] + (to[channel] - from[channel]) * fraction)
  ) as [number, number, number];
}

/**
 * Ramp position of class `index` of `classCount` classes: the first class takes the low end, the
 * last the high end (one class takes the middle). The layer's `classBreaks` uses the same rule.
 */
export function getClassRampPosition(index: number, classCount: number): number {
  return classCount <= 1 ? 0.5 : index / (classCount - 1);
}

/**
 * One RGBA color per class of a classed (stepped) ramp, low class first: exactly the colors a
 * layer with `colormap: name` and `classBreaks` of `classCount - 1` values draws. `range` is the
 * ramp trim `[t0, t1]` (layer prop `rampRange`). These are sampled stops, not the published
 * ColorBrewer tables: for exact class tables use `getClassPalette` (`cartography/class-table.ts`).
 */
export function getClassColors(
  name: RampName,
  classCount: number,
  reverse = false,
  alpha = 255,
  range?: readonly [number, number]
): PaletteColor[] {
  return Array.from({length: classCount}, (_, index) => {
    const [r, g, b] = sampleRamp(name, getClassRampPosition(index, classCount), reverse, range);
    return [r, g, b, alpha] as const;
  });
}

/**
 * CSS `linear-gradient` of a ramp, left to right.
 *
 * @param sqrtScale Matches `sqrtScale` on the layer: the color at normalized value `x` is the ramp
 *   color at `sqrt(x)`, so the gradient is sampled that way.
 * @param reverse Matches `reverseRamp` on the layer.
 * @param range Matches `rampRange` on the layer: the gradient shows only the trimmed part of the
 *   ramp, applied after `sqrtScale` and `reverse`.
 */
export function getRampCssGradient(
  name: RampName,
  options: {
    sqrtScale?: boolean;
    direction?: string;
    reverse?: boolean;
    range?: readonly [number, number];
  } = {}
): string {
  const samples = 24;
  const parts: string[] = [];
  for (let index = 0; index <= samples; index++) {
    const x = index / samples;
    const [r, g, b] = sampleRamp(
      name,
      options.sqrtScale ? Math.sqrt(x) : x,
      options.reverse,
      options.range
    );
    parts.push(`rgb(${r} ${g} ${b}) ${(x * 100).toFixed(1)}%`);
  }
  return `linear-gradient(${options.direction ?? 'to right'}, ${parts.join(', ')})`;
}

const formatChannel = (value: number) => (value / 255).toFixed(4);

/**
 * Generates one WGSL function per ramp (`spatialAnalysisRamp_<name>(t)`) and the
 * `spatialAnalysisSampleRamp(colormap, t)` dispatcher from {@link RAMP_STOPS}.
 */
export function getRampWgsl(): string {
  const functions = ALL_RAMP_NAMES.map(name => {
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
  const branches = ALL_RAMP_NAMES.map(
    name => `    case ${COLORMAP_INDEXES[name]}u: { return spatialAnalysisRamp_${name}(t); }`
  );
  return `${functions.join('\n')}

// Maps a normalized value to a ramp color; unknown colormaps fall back to grayscale.
fn spatialAnalysisSampleRamp(colormap: u32, t: f32) -> vec3<f32> {
  switch colormap {
${branches.join('\n')}
    default: { return vec3<f32>(t); }
  }
}
`;
}

// ---------------------------------------------------------------------------------------------
// Qualitative and bivariate palettes
// ---------------------------------------------------------------------------------------------

/**
 * Okabe-Ito (2008), the color-blind-safe qualitative set, without its black (useless on a dark
 * map): orange, sky blue, bluish green, yellow, blue, vermillion, reddish purple.
 */
export const OKABE_ITO: readonly PaletteColor[] = [
  [230, 159, 0, 255],
  [86, 180, 233, 255],
  [0, 158, 115, 255],
  [240, 228, 66, 255],
  [0, 114, 178, 255],
  [213, 94, 0, 255],
  [204, 121, 167, 255]
];

/** Tableau 10 (2016 revision): 10 balanced categorical hues. */
export const TABLEAU10: readonly PaletteColor[] = [
  [78, 121, 167, 255],
  [242, 142, 44, 255],
  [225, 87, 89, 255],
  [118, 183, 178, 255],
  [89, 161, 79, 255],
  [237, 201, 73, 255],
  [175, 122, 161, 255],
  [255, 157, 167, 255],
  [156, 117, 95, 255],
  [186, 176, 171, 255]
];

/** ColorBrewer Set2: 8 soft hues, good as large fills on a light ground. */
export const SET2: readonly PaletteColor[] = [
  [102, 194, 165, 255],
  [252, 141, 98, 255],
  [141, 160, 203, 255],
  [231, 138, 195, 255],
  [166, 216, 84, 255],
  [255, 217, 47, 255],
  [229, 196, 148, 255],
  [179, 179, 179, 255]
];

/** ColorBrewer Dark2: 8 strong hues, good for points and lines. */
export const DARK2: readonly PaletteColor[] = [
  [27, 158, 119, 255],
  [217, 95, 2, 255],
  [117, 112, 179, 255],
  [231, 41, 138, 255],
  [102, 166, 30, 255],
  [230, 171, 2, 255],
  [166, 118, 29, 255],
  [102, 102, 102, 255]
];

/** ColorBrewer Paired: 6 light/dark pairs (12 colors) for two-level categories. */
export const PAIRED: readonly PaletteColor[] = [
  [166, 206, 227, 255],
  [31, 120, 180, 255],
  [178, 223, 138, 255],
  [51, 160, 44, 255],
  [251, 154, 153, 255],
  [227, 26, 28, 255],
  [253, 191, 111, 255],
  [255, 127, 0, 255],
  [202, 178, 214, 255],
  [106, 61, 154, 255],
  [255, 255, 153, 255],
  [177, 89, 40, 255]
];

/** Qualitative palettes by name. Layers take up to 16 `palette` entries. */
export const QUALITATIVE_PALETTES = {
  okabeIto: OKABE_ITO,
  tableau10: TABLEAU10,
  set2: SET2,
  dark2: DARK2,
  paired: PAIRED
} as const;

/** Name of a qualitative palette. */
export type QualitativePaletteName = keyof typeof QUALITATIVE_PALETTES;

/**
 * 3 x 3 bivariate palettes (Joshua Stevens), `colors[row * 3 + column]`: column = class of the
 * x variable (low to high), row = class of the y variable (low to high). Feed the same list to a
 * layer `palette` (class index `row * 3 + column`) and a `bivariate` legend.
 */
export const BIVARIATE_PALETTES = {
  /** Grey to teal along x, grey to pink-purple along y, dark blue-violet where both are high. */
  tealPink: [
    [232, 232, 232, 255],
    [172, 228, 228, 255],
    [90, 200, 200, 255],
    [223, 176, 214, 255],
    [165, 173, 211, 255],
    [86, 152, 185, 255],
    [190, 100, 172, 255],
    [140, 98, 170, 255],
    [59, 73, 148, 255]
  ],
  /** Grey to red along x, grey to blue along y, dark brown-grey where both are high. */
  redBlue: [
    [232, 232, 232, 255],
    [228, 172, 172, 255],
    [200, 90, 90, 255],
    [176, 213, 223, 255],
    [173, 158, 165, 255],
    [152, 83, 86, 255],
    [100, 172, 190, 255],
    [98, 127, 140, 255],
    [87, 66, 73, 255]
  ],
  /**
   * Stevens "DkBlue" (light grounds): grey to green-teal along x, grey to blue-violet along y,
   * dark teal where both are high. Best when one variable is "nature" or "green".
   */
  dkBlue: [
    [232, 232, 232, 255],
    [184, 214, 190, 255],
    [115, 174, 128, 255],
    [181, 192, 218, 255],
    [144, 178, 179, 255],
    [90, 145, 120, 255],
    [108, 131, 181, 255],
    [86, 121, 148, 255],
    [42, 90, 91, 255]
  ],
  /** Stevens "Brown" (light grounds): grey to yellow-brown along x, grey to violet along y; the sober one. */
  brown: [
    [232, 232, 232, 255],
    [228, 217, 172, 255],
    [200, 179, 90, 255],
    [203, 184, 215, 255],
    [200, 173, 160, 255],
    [175, 142, 83, 255],
    [153, 114, 175, 255],
    [151, 107, 130, 255],
    [128, 77, 54, 255]
  ],
  /**
   * DESIGN PROPOSAL, simulate colour-vision deficiency before use. Derived from Stevens' rule for
   * dark backgrounds (dark neutral for low-low, lightness rises with both variables, high-high
   * brightest): x cyan, y magenta, high-high near white. Not a published palette.
   */
  darkGround: [
    [42, 45, 54, 255],
    [31, 111, 122, 255],
    [41, 194, 214, 255],
    [122, 47, 108, 255],
    [106, 90, 154, 255],
    [74, 176, 217, 255],
    [214, 63, 176, 255],
    [197, 138, 216, 255],
    [242, 242, 255, 255]
  ]
} as const satisfies Record<string, readonly PaletteColor[]>;

/** Returns `color` with a new alpha (0-255). */
export function withAlpha(
  color: readonly [number, number, number, number?],
  alpha: number
): PaletteColor {
  return [color[0], color[1], color[2], alpha];
}

/** Legend `categories` entries from a palette and one label per entry. */
export function getPaletteLegendEntries(
  palette: readonly PaletteColor[],
  labels: readonly string[]
): {color: PaletteColor; label: string}[] {
  return labels.map((label, index) => ({color: palette[index % palette.length], label}));
}

// ---------------------------------------------------------------------------------------------
// Ground-aware direction and licences
// ---------------------------------------------------------------------------------------------

/** Where the low end of a ramp sits in lightness: `'mid'` for diverging and cyclic ramps. */
export type RampLowEnd = 'light' | 'dark' | 'mid';

/**
 * Lightness of the low end of every ramp, to drive {@link directionFor}. ColorBrewer ramps start
 * pale (`'light'`), the perceptual and Kovesi/Crameri single-family ramps start dark (`'dark'`),
 * diverging, cyclic and isoluminant ramps have no light-to-dark order (`'mid'`).
 */
export const RAMP_LOW_END: Record<RampName, RampLowEnd> = {
  grayscale: 'dark',
  viridis: 'dark',
  magma: 'dark',
  inferno: 'dark',
  cividis: 'dark',
  diverging: 'mid',
  blues: 'light',
  greens: 'light',
  purples: 'light',
  oranges: 'light',
  reds: 'light',
  greys: 'light',
  ylgnbu: 'light',
  ylorrd: 'light',
  ylorbr: 'light',
  pubugn: 'light',
  bupu: 'light',
  orrd: 'light',
  gnbu: 'light',
  rdpu: 'light',
  rdbu: 'mid',
  brbg: 'mid',
  puor: 'mid',
  piyg: 'mid',
  prgn: 'mid',
  rdylbu: 'mid',
  spectral: 'mid',
  batlow: 'dark',
  vik: 'mid',
  roma: 'mid',
  romao: 'mid',
  oleron: 'mid',
  hypsometric: 'light',
  fire: 'dark',
  lajolla: 'dark',
  bamako: 'dark',
  berlin: 'mid',
  vanimo: 'mid',
  mako: 'dark',
  twilight: 'mid',
  isolum: 'mid'
};

/**
 * Whether a sequential ramp must be reversed so that "more" reads as the stronger mark on a ground.
 *
 * Rule of thumb: on a dark ground the BRIGHTEST colour is the highest value, on a light ground the
 * DARKEST colour is the highest value. A ramp whose low end is light (ColorBrewer: `blues`,
 * `ylorrd`) is right on a light ground and must be reversed on a dark one; a ramp whose low end is
 * dark (`magma`, `inferno`, `fire`, `mako`, `viridis`) is right on a dark ground and must be
 * reversed on a light one. Diverging, cyclic and isoluminant ramps (`'mid'`) never flip: choose a
 * dark-centre diverging (`berlin`, `vanimo`) for dark grounds instead.
 *
 * ```ts
 * const {reverse} = directionFor('dark', 'ylorrd'); // true: pale-to-dark would fade out on dark
 * layer.reverseRamp = reverse;
 * ```
 *
 * @param ground The basemap ground the layer is drawn on.
 * @param name Ramp to place; omit it for a generic "dark-ended" ramp (the default is the
 *   magma-like case).
 */
export function directionFor(
  ground: 'light' | 'dark',
  name: RampName | 'sequential' = 'sequential'
): {reverse: boolean} {
  const lowEnd = name === 'sequential' ? 'dark' : RAMP_LOW_END[name];
  if (lowEnd === 'mid') return {reverse: false};
  return {reverse: ground === 'dark' ? lowEnd === 'light' : lowEnd === 'dark'};
}

const SOURCE_LICENCES: Record<RampInfo['source'], string> = {
  matplotlib: 'CC0 (matplotlib, van der Walt and Smith)',
  colorbrewer: 'Apache-2.0 (Cynthia Brewer, Mark Harrower, Penn State)',
  crameri: 'MIT (Crameri)',
  colorcet: 'CC BY 4.0 (Peter Kovesi)',
  seaborn: 'BSD-3-Clause (seaborn)',
  convention: 'Public convention, no licence'
};

/** Licence and credit string of a ramp, for the "About colours" line. */
export function getRampLicence(name: RampName): string {
  return RAMP_INFO[name].licence ?? SOURCE_LICENCES[RAMP_INFO[name].source];
}
