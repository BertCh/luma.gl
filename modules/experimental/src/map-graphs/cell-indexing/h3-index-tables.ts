// luma.gl
// SPDX-License-Identifier: MIT AND Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileCopyrightText: Copyright 2016-2024 Uber Technologies, Inc.

// The H3 face/IJK -> base cell lookup is derived from Uber's Apache-2.0 H3 (`faceIjkBaseCells` and
// the pentagon `cwOffsetPent` data).
//
// GENERATED, not recalled. Produced by sampling 1e6 float64 points whose nearest icosahedron face is
// `f`, walking each sample's res 1..4 IJK up to the res 0 lattice with the same ijk math as H3's
// `_faceIjkToH3`, and taking the base cell and the counter-clockwise rotation count that
// reproduce h3-js 4.4.0 `latLngToCell` for every sample of that entry (300 samples per entry, all
// 320 reachable entries). For pentagons, entries are also searched over the clockwise-offset flag;
// the offset faces are the ones the data pins down. The 60 slots no point can reach (IJK beyond the
// face) are -1. `h3-index-wgsl.node.spec.ts` re-verifies the whole table by comparing a float64
// port against h3-js on random, pentagon-adjacent and edge-adjacent points at resolutions 0..15.

/**
 * Face/IJK to base cell table, indexed `face * 27 + i * 9 + j * 3 + k` for res 0 IJK components
 * 0..2. Each value is `baseCell | counterClockwiseRotations << 8`, or -1 where no point can land.
 */
export const CELL_INDEX_H3_BASE_CELL_PACKED_TABLE: readonly number[] = [
  // face 0
  16, 18, 24, 33, 30, 800, 305, 816, -1, 8, 1285, 1290, 22, -1, -1, 297, -1, -1, 4, 1280, -1, 271,
  -1, -1, -1, -1, -1,
  // face 1
  2, 6, 14, 10, 11, 785, 280, 791, -1, 0, 1281, 1289, 5, -1, -1, 274, -1, -1, 260, 1283, -1, 264,
  -1, -1, -1, -1, -1,
  // face 2
  7, 21, 38, 9, 19, 802, 270, 788, -1, 3, 1293, 1309, 1, -1, -1, 262, -1, -1, 516, 1292, -1, 256,
  -1, -1, -1, -1, -1,
  // face 3
  26, 42, 58, 29, 43, 830, 294, 815, -1, 12, 1308, 1324, 13, -1, -1, 277, -1, -1, 772, 1295, -1,
  259, -1, -1, -1, -1, -1,
  // face 4
  31, 41, 49, 44, 53, 829, 314, 833, -1, 15, 1302, 1313, 28, -1, -1, 298, -1, -1, 1028, 1288, -1,
  268, -1, -1, -1, -1, -1,
  // face 5
  50, 48, 817, 32, 798, 801, 792, 786, -1, 70, 67, 834, 820, -1, -1, 805, -1, -1, 83, 855, -1, 842,
  -1, -1, -1, -1, -1,
  // face 6
  25, 23, 792, 17, 779, 778, 782, 774, -1, 45, 39, 805, 803, -1, -1, 795, -1, -1, 63, 827, -1, 824,
  -1, -1, -1, -1, -1,
  // face 7
  36, 20, 782, 34, 787, 777, 806, 789, -1, 55, 40, 795, 822, -1, -1, 819, -1, -1, 72, 828, -1, 841,
  -1, -1, -1, -1, -1,
  // face 8
  64, 47, 806, 62, 811, 797, 826, 810, -1, 84, 69, 819, 850, -1, -1, 844, -1, -1, 97, 857, -1, 866,
  -1, -1, -1, -1, -1,
  // face 9
  75, 65, 826, 61, 821, 812, 817, 809, -1, 94, 86, 844, 849, -1, -1, 834, -1, -1, 107, 872, -1, 869,
  -1, -1, -1, -1, -1,
  // face 10
  57, 59, 831, 74, 846, 847, 851, 860, -1, 37, 807, 813, 52, -1, -1, 838, -1, -1, 24, 791, -1, 800,
  -1, -1, -1, -1, -1,
  // face 11
  46, 60, 840, 56, 836, 848, 831, 845, -1, 27, 808, 823, 35, -1, -1, 813, -1, -1, 14, 788, -1, 785,
  -1, -1, -1, -1, -1,
  // face 12
  71, 89, 865, 73, 859, 871, 840, 856, -1, 51, 837, 852, 54, -1, -1, 823, -1, -1, 38, 815, -1, 802,
  -1, -1, -1, -1, -1,
  // face 13
  96, 104, 875, 98, 878, 883, 865, 879, -1, 76, 854, 862, 82, -1, -1, 852, -1, -1, 58, 833, -1, 830,
  -1, -1, -1, -1, -1,
  // face 14
  85, 87, 851, 101, 870, 868, 875, 880, -1, 66, 835, 838, 81, -1, -1, 862, -1, -1, 49, 816, -1, 829,
  -1, -1, -1, -1, -1,
  // face 15
  95, 92, 83, 79, 78, 842, 319, 827, -1, 109, 108, 1380, 349, -1, -1, 333, -1, -1, 1141, 1398, -1,
  362, -1, -1, -1, -1, -1,
  // face 16
  90, 77, 63, 80, 68, 824, 328, 828, -1, 106, 93, 1359, 355, -1, -1, 344, -1, -1, 885, 1389, -1,
  369, -1, -1, -1, -1, -1,
  // face 17
  105, 88, 72, 103, 91, 841, 353, 857, -1, 113, 99, 1360, 372, -1, -1, 367, -1, -1, 629, 1386, -1,
  377, -1, -1, -1, -1, -1,
  // face 18
  119, 111, 97, 115, 110, 866, 363, 872, -1, 121, 116, 1383, 376, -1, -1, 368, -1, -1, 373, 1393,
  -1, 374, -1, -1, -1, -1, -1,
  // face 19
  114, 112, 107, 100, 102, 869, 339, 855, -1, 118, 120, 1395, 364, -1, -1, 348, -1, -1, 117, 1401,
  -1, 365, -1, -1, -1, -1, -1
];

/**
 * For each pentagon base cell, the two faces whose base-cell rotation is clockwise rather than
 * counter-clockwise (H3 `cwOffsetPent`), as `[baseCell, faceA, faceB]`. Pentagons 4 and 117 have none.
 */
export const CELL_INDEX_H3_CW_OFFSET_PENTAGON_FACES: readonly (readonly [
  number,
  number,
  number
])[] = [
  [4, -1, -1],
  [14, 2, 6],
  [24, 1, 5],
  [38, 3, 7],
  [49, 0, 9],
  [58, 4, 8],
  [63, 11, 15],
  [72, 12, 16],
  [83, 10, 19],
  [97, 13, 17],
  [107, 14, 18],
  [117, -1, -1]
];
