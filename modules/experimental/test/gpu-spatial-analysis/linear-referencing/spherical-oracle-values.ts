// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Pinned references for the spherical linear-referencing specs, generated with pyproj
 * `Geod(a=6371008.8, f=0)` (`inv`, `fwd`; dense `npts`-style sampling plus ternary refinement for
 * the closest point) on the f32-rounded inputs below, and shapely 2.1.2 for the planar normalized
 * cases. Generator: scratchpad `build/K/gen.py`.
 */

/** Longitude/latitude path vertices, f32-exact. */
export const SPHERICAL_POSITIONS = [
  8.0, 47.0, 8.75057315826416, 49.38328170776367, 10.40468692779541, 47.734527587890625,
  9.205684661865234, 49.97584533691406, 6.237276554107666, 51.90321731567383, -60.0, 72.0,
  -56.43516540527344, 71.61521911621094, -58.79877853393555, 68.95632934570312, -61.74034118652344,
  68.2972412109375, 178.0, 10.0, -179.0, 12.0, -176.0, 11.0, 100.0, 5.0, 100.00001525878906,
  5.000214099884033, 100.00199890136719, 5.001384735107422, 100.00248718261719, 5.003340244293213,
  100.00135040283203, 5.001981258392334, 100.00180053710938, 5.000156879425049, 20.0, 20.0
];
export const SPHERICAL_PATH_OFFSETS = [0, 5, 9, 12, 18, 19, 19];
/** Total great-circle length of each path in meters. */
export const SPHERICAL_PATH_LENGTHS = [1053641.67, 579435.645, 741095.397, 908.931232, 0.0, 0.0];
export const SPHERICAL_POINTS = [
  10.464241981506348, 47.59934997558594, 9.668670654296875, 47.51690673828125, 10.392180442810059,
  46.724586486816406, 8.451863288879395, 46.50413513183594, 5.039703369140625, 51.38136291503906,
  7.220621585845947, 51.29603576660156, 8.276082038879395, 51.49717712402344, 8.039163589477539,
  48.38859939575195, -57.831695556640625, 67.32231140136719, -58.63420104980469, 68.98741912841797,
  -59.353721618652344, 69.34906005859375, -63.50333786010742, 67.84777069091797, -60.19797897338867,
  70.2216796875, -56.917381286621094, 73.53021240234375, -56.01494216918945, 72.16720581054688,
  -58.09297561645508, 67.55947875976562, -180.041748046875, 11.609992980957031, -180.61318969726562,
  13.871312141418457, -175.3129425048828, 10.201680183410645, 179.49630737304688,
  10.648859024047852, -177.6197052001953, 13.779792785644531, 179.61566162109375,
  10.278876304626465, 176.7698516845703, 11.711623191833496, 178.2093048095703, 8.722209930419922,
  100.00765991210938, 5.004172325134277, 99.9968490600586, 4.99659538269043, 99.98153686523438,
  5.015262603759766, 99.99872589111328, 5.002119541168213, 100.01205444335938, 4.982392311096191,
  99.99490356445312, 4.981428146362305, 100.0204849243164, 5.006467342376709, 99.99713134765625,
  5.000949382781982, 8.0, 47.0, 0.0, -60.0
];

/** Closest point on the path network per query point: path, segment, meters, foot point, side. */
export const SPHERICAL_PROJECTIONS: {
  path: number;
  segment: number;
  distance: number;
  measure: number;
  normalized: number;
  foot: [number, number];
  side: number;
}[] = [
  {
    path: 0,
    segment: 1,
    distance: 15678.7158,
    measure: 490838.76,
    normalized: 0.465849799,
    foot: [10.4046869, 47.7345276],
    side: -1
  },
  {
    path: 0,
    segment: 1,
    distance: 59397.8298,
    measure: 480843.478,
    normalized: 0.456363384,
    foot: [10.3318544, 47.8099206],
    side: -1
  },
  {
    path: 0,
    segment: 1,
    distance: 112304.452,
    measure: 490838.76,
    normalized: 0.465849799,
    foot: [10.4046869, 47.7345276],
    side: -1
  },
  {
    path: 0,
    segment: 0,
    distance: 65002.0114,
    measure: 0.0,
    normalized: 0.0,
    foot: [8.0, 47.0],
    side: -1
  },
  {
    path: 0,
    segment: 3,
    distance: 100974.047,
    measure: 1053641.67,
    normalized: 1.0,
    foot: [6.23727655, 51.9032173],
    side: 1
  },
  {
    path: 0,
    segment: 3,
    distance: 424.488214,
    measure: 957877.267,
    normalized: 0.909111033,
    foot: [7.21626977, 51.2933584],
    side: -1
  },
  {
    path: 0,
    segment: 3,
    distance: 68560.802,
    measure: 922955.5,
    normalized: 0.875967159,
    foot: [7.5667967, 51.0689673],
    side: -1
  },
  {
    path: 0,
    segment: 0,
    distance: 28153.7803,
    measure: 151845.774,
    normalized: 0.144115194,
    foot: [8.41226581, 48.3370669],
    side: 1
  },
  {
    path: 1,
    segment: 2,
    distance: 177600.054,
    measure: 494961.376,
    normalized: 0.854212854,
    foot: [-59.9849934, 68.7013322],
    side: 1
  },
  {
    path: 1,
    segment: 1,
    distance: 5389.70154,
    measure: 434424.677,
    normalized: 0.749737578,
    foot: [-58.7643886, 69.0005158],
    side: 1
  },
  {
    path: 1,
    segment: 1,
    distance: 32719.8578,
    measure: 403212.349,
    normalized: 0.695870806,
    foot: [-58.5509222, 69.2707229],
    side: -1
  },
  {
    path: 1,
    segment: 2,
    distance: 88635.1233,
    measure: 579435.645,
    normalized: 1.0,
    foot: [-61.7403412, 68.2972412],
    side: 1
  },
  {
    path: 1,
    segment: 1,
    distance: 88664.8028,
    measure: 317552.326,
    normalized: 0.548037265,
    foot: [-57.9367769, 70.0108675],
    side: -1
  },
  {
    path: 1,
    segment: 0,
    distance: 193775.167,
    measure: 41180.7238,
    normalized: 0.0710704013,
    foot: [-58.863089, 71.8860873],
    side: 1
  },
  {
    path: 1,
    segment: 0,
    distance: 62511.9896,
    measure: 122528.928,
    normalized: 0.211462531,
    foot: [-56.6592359, 71.6418384],
    side: 1
  },
  {
    path: 1,
    segment: 2,
    distance: 149155.205,
    measure: 491706.033,
    normalized: 0.848594727,
    foot: [-59.9160883, 68.7165338],
    side: 1
  },
  {
    path: 2,
    segment: 0,
    distance: 27441.7417,
    measure: 277562.858,
    normalized: 0.374530539,
    foot: [-179.900604, 11.4056008],
    side: 1
  },
  {
    path: 2,
    segment: 0,
    distance: 270294.316,
    measure: 367532.153,
    normalized: 0.495930962,
    foot: [-179.215773, 11.8580801],
    side: 1
  },
  {
    path: 2,
    segment: 1,
    distance: 116270.846,
    measure: 741095.397,
    normalized: 1.0,
    foot: [-176.0, 11.0],
    side: -1
  },
  {
    path: 2,
    segment: 0,
    distance: 32722.0054,
    measure: 175865.24,
    normalized: 0.237304455,
    foot: [179.327914, 10.8922358],
    side: -1
  },
  {
    path: 2,
    segment: 1,
    distance: 235325.602,
    measure: 474386.935,
    normalized: 0.640115884,
    foot: [-178.315458, 11.7751687],
    side: 1
  },
  {
    path: 2,
    segment: 0,
    distance: 74088.6567,
    measure: 163550.295,
    normalized: 0.220687236,
    foot: [179.234676, 10.8299382],
    side: -1
  },
  {
    path: 2,
    segment: 0,
    distance: 232956.186,
    measure: 0.0,
    normalized: 0.0,
    foot: [178.0, 10.0],
    side: 1
  },
  {
    path: 2,
    segment: 0,
    distance: 143927.632,
    measure: 0.0,
    normalized: 0.0,
    foot: [178.0, 10.0],
    side: -1
  },
  {
    path: 3,
    segment: 2,
    distance: 580.41205,
    measure: 503.329538,
    normalized: 0.553759757,
    foot: [100.002487, 5.00334024],
    side: -1
  },
  {
    path: 3,
    segment: 0,
    distance: 514.924178,
    measure: 3.10642257e-10,
    normalized: 3.41766513e-13,
    foot: [100.0, 5.0],
    side: 1
  },
  {
    path: 3,
    segment: 0,
    distance: 2643.79434,
    measure: 23.8667806,
    normalized: 0.0262580708,
    foot: [100.000015, 5.0002141],
    side: 1
  },
  {
    path: 3,
    segment: 0,
    distance: 255.519938,
    measure: 23.8667806,
    normalized: 0.0262580708,
    foot: [100.000015, 5.0002141],
    side: 1
  },
  {
    path: 3,
    segment: 4,
    distance: 2278.62184,
    measure: 908.931232,
    normalized: 1.0,
    foot: [100.001801, 5.00015688],
    side: 1
  },
  {
    path: 3,
    segment: 0,
    distance: 2140.87591,
    measure: 0.0,
    normalized: 0.0,
    foot: [100.0, 5.0],
    side: 1
  },
  {
    path: 3,
    segment: 2,
    distance: 2023.72634,
    measure: 503.329538,
    normalized: 0.553759757,
    foot: [100.002487, 5.00334024],
    side: -1
  },
  {
    path: 3,
    segment: 0,
    distance: 329.752815,
    measure: 23.8667806,
    normalized: 0.0262580708,
    foot: [100.000015, 5.0002141],
    side: 1
  },
  {
    path: 0,
    segment: 0,
    distance: 1.15454905e-10,
    measure: 0.0,
    normalized: 0.0,
    foot: [8.0, 47.0],
    side: 0
  },
  {
    path: 3,
    segment: 0,
    distance: 11044053.5,
    measure: 1.28694659e-9,
    normalized: 1.41588994e-12,
    foot: [100.0, 5.0],
    side: 1
  }
];

/** Located events: path, fraction of the path, lateral offset in meters, expected position. */
export const SPHERICAL_EVENTS: {
  path: number;
  fraction: number;
  offset: number;
  meters: number;
  position: [number, number];
  azimuth: number;
}[] = [
  {path: 0, fraction: 0.0, offset: 0.0, meters: 0.0, position: [8.0, 47.0], azimuth: 11.5780769},
  {
    path: 0,
    fraction: 0.0,
    offset: 2500.0,
    meters: 0.0,
    position: [7.9677017, 47.0045079],
    azimuth: 11.5780769
  },
  {
    path: 0,
    fraction: 0.1,
    offset: 0.0,
    meters: 105364.167,
    position: [8.28380929, 47.9279343],
    azimuth: 11.7872095
  },
  {
    path: 0,
    fraction: 0.1,
    offset: 2500.0,
    meters: 105364.167,
    position: [8.25096043, 47.9325224],
    azimuth: 11.7872095
  },
  {
    path: 0,
    fraction: 0.37,
    offset: 0.0,
    meters: 389847.418,
    position: [9.65892579, 48.4941341],
    azimuth: 146.477903
  },
  {
    path: 0,
    fraction: 0.37,
    offset: 2500.0,
    meters: 389847.418,
    position: [9.68721636, 48.5065471],
    azimuth: 146.477903
  },
  {
    path: 0,
    fraction: 0.5,
    offset: 0.0,
    meters: 526820.835,
    position: [10.2475864, 48.0404928],
    azimuth: 340.942115
  },
  {
    path: 0,
    fraction: 0.5,
    offset: 2500.0,
    meters: 526820.835,
    position: [10.2158073, 48.0331472],
    azimuth: 340.942115
  },
  {
    path: 0,
    fraction: 0.93,
    offset: 0.0,
    meters: 979886.753,
    position: [6.99358794, 51.4342395],
    azimuth: 315.290528
  },
  {
    path: 0,
    fraction: 0.93,
    offset: 2500.0,
    meters: 979886.753,
    position: [6.96796644, 51.4184196],
    azimuth: 315.290528
  },
  {
    path: 0,
    fraction: 1.0,
    offset: 0.0,
    meters: 1053641.67,
    position: [6.23727655, 51.9032173],
    azimuth: 314.69724
  },
  {
    path: 0,
    fraction: 1.0,
    offset: 2500.0,
    meters: 1053641.67,
    position: [6.21165541, 51.8872329],
    azimuth: 314.69724
  },
  {path: 1, fraction: 0.0, offset: 0.0, meters: 0.0, position: [-60.0, 72.0], azimuth: 107.372351},
  {
    path: 1,
    fraction: 0.0,
    offset: 2500.0,
    meters: 0.0,
    position: [-59.9782513, 72.0214562],
    azimuth: 107.372351
  },
  {
    path: 1,
    fraction: 0.1,
    offset: 0.0,
    meters: 57943.5645,
    position: [-58.4043205, 71.8378256],
    azimuth: 108.889244
  },
  {
    path: 1,
    fraction: 0.1,
    offset: 2500.0,
    meters: 57943.5645,
    position: [-58.3809432, 71.8590964],
    azimuth: 108.889244
  },
  {
    path: 1,
    fraction: 0.37,
    offset: 0.0,
    meters: 214391.189,
    position: [-57.1367592, 70.8991929],
    azimuth: 197.144394
  },
  {
    path: 1,
    fraction: 0.37,
    offset: 2500.0,
    meters: 214391.189,
    position: [-57.0711273, 70.8925538],
    azimuth: 197.144394
  },
  {
    path: 1,
    fraction: 0.5,
    offset: 0.0,
    meters: 289717.822,
    position: [-57.727734, 70.2508949],
    azimuth: 196.58705
  },
  {
    path: 1,
    fraction: 0.5,
    offset: 2500.0,
    meters: 289717.822,
    position: [-57.6639857, 70.2444653],
    azimuth: 196.58705
  },
  {
    path: 1,
    fraction: 0.93,
    offset: 0.0,
    meters: 538875.15,
    position: [-60.9053518, 68.493554],
    azimuth: 237.828976
  },
  {
    path: 1,
    fraction: 0.93,
    offset: 2500.0,
    meters: 538875.15,
    position: [-60.8727256, 68.4745198],
    azimuth: 237.828976
  },
  {
    path: 1,
    fraction: 1.0,
    offset: 0.0,
    meters: 579435.645,
    position: [-61.7403412, 68.2972412],
    azimuth: 237.052644
  },
  {
    path: 1,
    fraction: 1.0,
    offset: 2500.0,
    meters: 579435.645,
    position: [-61.7073018, 68.2783709],
    azimuth: 237.052644
  },
  {path: 2, fraction: 0.0, offset: 0.0, meters: 0.0, position: [178.0, 10.0], azimuth: 55.5389468},
  {
    path: 2,
    fraction: 0.0,
    offset: 2500.0,
    meters: 0.0,
    position: [177.987081, 10.0185372],
    azimuth: 55.5389468
  },
  {
    path: 2,
    fraction: 0.1,
    offset: 0.0,
    meters: 74109.5397,
    position: [178.558655, 10.3766552],
    azimuth: 55.6377654
  },
  {
    path: 2,
    fraction: 0.1,
    offset: 2500.0,
    meters: 74109.5397,
    position: [178.545753, 10.3952144],
    azimuth: 55.6377654
  },
  {
    path: 2,
    fraction: 0.37,
    offset: 0.0,
    meters: 274205.297,
    position: [-179.926118, 11.3886837],
    azimuth: 55.923865
  },
  {
    path: 2,
    fraction: 0.37,
    offset: 2500.0,
    meters: 274205.297,
    position: [-179.938969, 11.4073059],
    azimuth: 55.923865
  },
  {
    path: 2,
    fraction: 0.5,
    offset: 0.0,
    meters: 370547.698,
    position: [-179.192781, 11.8732178],
    azimuth: 56.0717143
  },
  {
    path: 2,
    fraction: 0.5,
    offset: 2500.0,
    meters: 370547.698,
    position: [-179.205605, 11.8918725],
    azimuth: 56.0717143
  },
  {
    path: 2,
    fraction: 0.93,
    offset: 0.0,
    meters: 689218.719,
    position: [-176.449389, 11.1521839],
    azimuth: 108.994945
  },
  {
    path: 2,
    fraction: 0.93,
    offset: 2500.0,
    meters: 689218.719,
    position: [-176.44193, 11.1734426],
    azimuth: 108.994945
  },
  {
    path: 2,
    fraction: 1.0,
    offset: 0.0,
    meters: 741095.397,
    position: [-176.0, 11.0],
    azimuth: 109.081279
  },
  {
    path: 2,
    fraction: 1.0,
    offset: 2500.0,
    meters: 741095.397,
    position: [-175.992512, 11.0212476],
    azimuth: 109.081279
  },
  {path: 3, fraction: 0.0, offset: 0.0, meters: 0.0, position: [100.0, 5.0], azimuth: 4.0610863},
  {
    path: 3,
    fraction: 0.0,
    offset: 0.5,
    meters: 0.0,
    position: [99.9999955, 5.00000032],
    azimuth: 4.0610863
  },
  {
    path: 3,
    fraction: 0.1,
    offset: 0.0,
    meters: 90.8931232,
    position: [100.000536, 5.00052133],
    azimuth: 59.357492
  },
  {
    path: 3,
    fraction: 0.1,
    offset: 0.5,
    meters: 90.8931232,
    position: [100.000534, 5.00052519],
    azimuth: 59.357492
  },
  {
    path: 3,
    fraction: 0.37,
    offset: 0.0,
    meters: 336.304556,
    position: [100.002123, 5.00188257],
    azimuth: 13.9684899
  },
  {
    path: 3,
    fraction: 0.37,
    offset: 0.5,
    meters: 336.304556,
    position: [100.002119, 5.00188366],
    azimuth: 13.9684899
  },
  {
    path: 3,
    fraction: 0.5,
    offset: 0.0,
    meters: 454.465616,
    position: [100.002381, 5.0029138],
    azimuth: 13.9685123
  },
  {
    path: 3,
    fraction: 0.5,
    offset: 0.5,
    meters: 454.465616,
    position: [100.002376, 5.00291488],
    azimuth: 13.9685123
  },
  {
    path: 3,
    fraction: 0.93,
    offset: 0.0,
    meters: 845.306046,
    position: [100.001663, 5.00071254],
    azimuth: 166.19082
  },
  {
    path: 3,
    fraction: 0.93,
    offset: 0.5,
    meters: 845.306046,
    position: [100.001668, 5.00071361],
    azimuth: 166.19082
  },
  {
    path: 3,
    fraction: 1.0,
    offset: 0.0,
    meters: 908.931232,
    position: [100.001801, 5.00015688],
    azimuth: 166.190832
  },
  {
    path: 3,
    fraction: 1.0,
    offset: 0.5,
    meters: 908.931232,
    position: [100.001805, 5.00015795],
    azimuth: 166.190832
  }
];

/** shapely `line_locate_point(normalized=True)` on LineString([(0,0),(10,0),(10,10),(20,10)]) (length 30). */
export const PLANAR_NORMALIZED_QUERIES: [number, number][] = [
  [5, 2],
  [12, 5],
  [10, 0],
  [30, 10],
  [-3, -3],
  [15, 9]
];
export const PLANAR_NORMALIZED_EXPECTED = [
  0.16666666666666666, 0.5, 0.3333333333333333, 1, 0, 0.8333333333333334
];
/** shapely `line_interpolate_point(normalized=True)` at fractions 0, 0.25, 0.5, 0.8, 1. */
export const PLANAR_INTERPOLATED_FRACTIONS = [0, 0.25, 0.5, 0.8, 1];
export const PLANAR_INTERPOLATED_EXPECTED: [number, number][] = [
  [0, 0],
  [7.5, 0],
  [10, 5],
  [14, 10],
  [20, 10]
];
