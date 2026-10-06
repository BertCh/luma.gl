// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Reference fixture generated with Shapely 2.1.2 (`LineString.intersects`, `intersection`,
 * `relate`) from two seeded sets of 40 two-vertex lines on a 7 by 7 integer lattice. Each pair is
 * `[leftSegmentId, rightSegmentId, kind]`; two-vertex lines make segment IDs `2 * lineRow`. The kinds are `proper`
 * (interior/interior relate `0`), `touch`, `collinearTouch` (parallel, one shared point) and
 * `overlap` (a shared line). Counts: 285 proper, 147 touch, 3 collinearTouch, 3 overlap.
 */
export const SHAPELY_SEGMENT_FIXTURE = {
  left: [
    [
      [4, 2],
      [5, 2]
    ],
    [
      [5, 4],
      [0, 6]
    ],
    [
      [3, 6],
      [1, 5]
    ],
    [
      [0, 1],
      [0, 2]
    ],
    [
      [3, 6],
      [1, 3]
    ],
    [
      [4, 0],
      [4, 1]
    ],
    [
      [0, 5],
      [1, 3]
    ],
    [
      [2, 1],
      [6, 6]
    ],
    [
      [3, 1],
      [6, 6]
    ],
    [
      [0, 1],
      [4, 4]
    ],
    [
      [3, 1],
      [1, 0]
    ],
    [
      [6, 0],
      [1, 6]
    ],
    [
      [1, 1],
      [6, 1]
    ],
    [
      [2, 2],
      [1, 4]
    ],
    [
      [5, 5],
      [1, 1]
    ],
    [
      [5, 1],
      [3, 2]
    ],
    [
      [0, 2],
      [3, 1]
    ],
    [
      [1, 2],
      [0, 2]
    ],
    [
      [2, 6],
      [4, 4]
    ],
    [
      [0, 4],
      [5, 5]
    ],
    [
      [2, 0],
      [2, 2]
    ],
    [
      [6, 2],
      [3, 5]
    ],
    [
      [2, 1],
      [3, 3]
    ],
    [
      [5, 1],
      [0, 2]
    ],
    [
      [0, 5],
      [2, 6]
    ],
    [
      [3, 0],
      [4, 6]
    ],
    [
      [3, 2],
      [3, 4]
    ],
    [
      [6, 0],
      [3, 0]
    ],
    [
      [5, 1],
      [4, 1]
    ],
    [
      [0, 6],
      [1, 6]
    ],
    [
      [3, 2],
      [4, 2]
    ],
    [
      [4, 2],
      [6, 3]
    ],
    [
      [0, 4],
      [5, 6]
    ],
    [
      [0, 3],
      [0, 1]
    ],
    [
      [2, 4],
      [4, 2]
    ],
    [
      [1, 2],
      [2, 5]
    ],
    [
      [4, 0],
      [2, 5]
    ],
    [
      [2, 2],
      [1, 6]
    ],
    [
      [0, 5],
      [1, 5]
    ],
    [
      [5, 2],
      [3, 1]
    ]
  ],
  right: [
    [
      [5, 0],
      [0, 4]
    ],
    [
      [4, 3],
      [0, 1]
    ],
    [
      [5, 4],
      [2, 6]
    ],
    [
      [2, 3],
      [5, 3]
    ],
    [
      [1, 0],
      [5, 0]
    ],
    [
      [6, 3],
      [2, 6]
    ],
    [
      [1, 1],
      [5, 4]
    ],
    [
      [1, 5],
      [6, 3]
    ],
    [
      [0, 1],
      [3, 2]
    ],
    [
      [1, 0],
      [6, 3]
    ],
    [
      [2, 1],
      [3, 4]
    ],
    [
      [6, 1],
      [4, 3]
    ],
    [
      [3, 5],
      [5, 2]
    ],
    [
      [3, 2],
      [2, 3]
    ],
    [
      [3, 1],
      [0, 3]
    ],
    [
      [6, 4],
      [1, 5]
    ],
    [
      [3, 6],
      [2, 1]
    ],
    [
      [0, 3],
      [2, 4]
    ],
    [
      [2, 0],
      [6, 6]
    ],
    [
      [6, 2],
      [5, 4]
    ],
    [
      [5, 0],
      [6, 2]
    ],
    [
      [2, 4],
      [5, 5]
    ],
    [
      [2, 6],
      [3, 2]
    ],
    [
      [6, 5],
      [5, 2]
    ],
    [
      [2, 5],
      [1, 4]
    ],
    [
      [6, 0],
      [3, 4]
    ],
    [
      [6, 2],
      [2, 5]
    ],
    [
      [2, 3],
      [2, 6]
    ],
    [
      [4, 5],
      [5, 6]
    ],
    [
      [2, 2],
      [2, 5]
    ],
    [
      [2, 5],
      [6, 3]
    ],
    [
      [2, 6],
      [1, 6]
    ],
    [
      [6, 5],
      [3, 2]
    ],
    [
      [2, 4],
      [1, 4]
    ],
    [
      [1, 1],
      [6, 2]
    ],
    [
      [6, 3],
      [2, 5]
    ],
    [
      [0, 5],
      [5, 5]
    ],
    [
      [3, 1],
      [4, 6]
    ],
    [
      [4, 4],
      [5, 3]
    ],
    [
      [2, 4],
      [4, 6]
    ]
  ],
  pairs: [
    [0, 18, 'proper'],
    [0, 22, 'touch'],
    [0, 24, 'touch'],
    [0, 46, 'touch'],
    [0, 50, 'proper'],
    [2, 4, 'touch'],
    [2, 10, 'proper'],
    [2, 12, 'touch'],
    [2, 24, 'proper'],
    [2, 30, 'proper'],
    [2, 32, 'proper'],
    [2, 36, 'proper'],
    [2, 38, 'touch'],
    [2, 42, 'proper'],
    [2, 44, 'proper'],
    [2, 54, 'proper'],
    [2, 64, 'touch'],
    [2, 72, 'proper'],
    [2, 74, 'proper'],
    [2, 78, 'proper'],
    [4, 4, 'proper'],
    [4, 10, 'proper'],
    [4, 14, 'touch'],
    [4, 30, 'touch'],
    [4, 32, 'touch'],
    [4, 44, 'proper'],
    [4, 54, 'proper'],
    [4, 72, 'touch'],
    [6, 2, 'touch'],
    [6, 16, 'touch'],
    [8, 0, 'proper'],
    [8, 4, 'proper'],
    [8, 10, 'proper'],
    [8, 14, 'proper'],
    [8, 30, 'proper'],
    [8, 32, 'touch'],
    [8, 34, 'proper'],
    [8, 44, 'proper'],
    [8, 52, 'proper'],
    [8, 54, 'proper'],
    [8, 58, 'proper'],
    [8, 60, 'proper'],
    [8, 66, 'proper'],
    [8, 70, 'proper'],
    [8, 72, 'proper'],
    [10, 0, 'proper'],
    [10, 8, 'touch'],
    [12, 0, 'proper'],
    [12, 34, 'proper'],
    [12, 72, 'touch'],
    [14, 0, 'proper'],
    [14, 2, 'proper'],
    [14, 4, 'proper'],
    [14, 6, 'proper'],
    [14, 10, 'proper'],
    [14, 12, 'proper'],
    [14, 14, 'proper'],
    [14, 16, 'proper'],
    [14, 20, 'touch'],
    [14, 24, 'proper'],
    [14, 26, 'proper'],
    [14, 28, 'proper'],
    [14, 30, 'proper'],
    [14, 32, 'touch'],
    [14, 36, 'touch'],
    [14, 44, 'proper'],
    [14, 50, 'proper'],
    [14, 52, 'proper'],
    [14, 60, 'proper'],
    [14, 68, 'proper'],
    [14, 70, 'proper'],
    [14, 74, 'proper'],
    [14, 76, 'proper'],
    [16, 0, 'proper'],
    [16, 4, 'proper'],
    [16, 6, 'proper'],
    [16, 10, 'proper'],
    [16, 12, 'proper'],
    [16, 14, 'proper'],
    [16, 18, 'proper'],
    [16, 22, 'proper'],
    [16, 24, 'proper'],
    [16, 28, 'touch'],
    [16, 30, 'proper'],
    [16, 36, 'touch'],
    [16, 50, 'proper'],
    [16, 52, 'proper'],
    [16, 60, 'proper'],
    [16, 64, 'proper'],
    [16, 68, 'proper'],
    [16, 70, 'proper'],
    [16, 74, 'touch'],
    [16, 76, 'proper'],
    [18, 0, 'proper'],
    [18, 2, 'touch'],
    [18, 6, 'proper'],
    [18, 14, 'proper'],
    [18, 16, 'touch'],
    [18, 20, 'proper'],
    [18, 24, 'proper'],
    [18, 26, 'proper'],
    [18, 28, 'proper'],
    [18, 32, 'proper'],
    [18, 44, 'proper'],
    [18, 50, 'proper'],
    [18, 52, 'proper'],
    [18, 58, 'proper'],
    [18, 60, 'touch'],
    [18, 70, 'touch'],
    [18, 74, 'proper'],
    [18, 76, 'touch'],
    [20, 8, 'touch'],
    [20, 18, 'touch'],
    [20, 28, 'touch'],
    [20, 36, 'proper'],
    [20, 74, 'touch'],
    [22, 2, 'proper'],
    [22, 6, 'proper'],
    [22, 12, 'proper'],
    [22, 14, 'proper'],
    [22, 18, 'proper'],
    [22, 20, 'proper'],
    [22, 30, 'proper'],
    [22, 32, 'proper'],
    [22, 36, 'proper'],
    [22, 40, 'proper'],
    [22, 42, 'proper'],
    [22, 44, 'proper'],
    [22, 48, 'proper'],
    [22, 50, 'touch'],
    [22, 54, 'proper'],
    [22, 58, 'proper'],
    [22, 62, 'touch'],
    [22, 64, 'proper'],
    [22, 68, 'proper'],
    [22, 72, 'proper'],
    [22, 74, 'proper'],
    [22, 78, 'proper'],
    [24, 0, 'proper'],
    [24, 12, 'touch'],
    [24, 18, 'proper'],
    [24, 20, 'touch'],
    [24, 22, 'touch'],
    [24, 28, 'touch'],
    [24, 32, 'touch'],
    [24, 36, 'proper'],
    [24, 40, 'proper'],
    [24, 50, 'proper'],
    [24, 68, 'touch'],
    [24, 74, 'touch'],
    [26, 0, 'proper'],
    [26, 2, 'touch'],
    [26, 34, 'proper'],
    [26, 48, 'touch'],
    [26, 58, 'touch'],
    [26, 66, 'touch'],
    [28, 0, 'proper'],
    [28, 2, 'proper'],
    [28, 4, 'proper'],
    [28, 6, 'proper'],
    [28, 10, 'proper'],
    [28, 12, 'touch'],
    [28, 14, 'proper'],
    [28, 16, 'proper'],
    [28, 20, 'proper'],
    [28, 24, 'proper'],
    [28, 26, 'proper'],
    [28, 28, 'proper'],
    [28, 30, 'proper'],
    [28, 32, 'proper'],
    [28, 42, 'touch'],
    [28, 44, 'proper'],
    [28, 50, 'proper'],
    [28, 52, 'proper'],
    [28, 58, 'touch'],
    [28, 60, 'proper'],
    [28, 68, 'touch'],
    [28, 70, 'proper'],
    [28, 72, 'touch'],
    [28, 74, 'proper'],
    [28, 76, 'touch'],
    [30, 16, 'touch'],
    [30, 18, 'proper'],
    [30, 26, 'touch'],
    [30, 36, 'proper'],
    [30, 44, 'touch'],
    [30, 64, 'touch'],
    [30, 68, 'proper'],
    [30, 74, 'proper'],
    [32, 2, 'proper'],
    [32, 12, 'proper'],
    [32, 16, 'proper'],
    [32, 18, 'proper'],
    [32, 20, 'proper'],
    [32, 28, 'touch'],
    [32, 32, 'proper'],
    [32, 36, 'proper'],
    [32, 68, 'proper'],
    [32, 74, 'touch'],
    [36, 4, 'touch'],
    [36, 10, 'touch'],
    [36, 24, 'touch'],
    [36, 30, 'proper'],
    [36, 32, 'proper'],
    [36, 42, 'proper'],
    [36, 44, 'touch'],
    [36, 54, 'touch'],
    [36, 60, 'touch'],
    [36, 62, 'touch'],
    [36, 70, 'touch'],
    [36, 72, 'proper'],
    [36, 74, 'proper'],
    [36, 76, 'collinearTouch'],
    [36, 78, 'proper'],
    [38, 0, 'touch'],
    [38, 4, 'proper'],
    [38, 10, 'proper'],
    [38, 14, 'proper'],
    [38, 24, 'proper'],
    [38, 30, 'proper'],
    [38, 32, 'proper'],
    [38, 42, 'touch'],
    [38, 44, 'proper'],
    [38, 48, 'proper'],
    [38, 52, 'proper'],
    [38, 54, 'proper'],
    [38, 58, 'proper'],
    [38, 60, 'proper'],
    [38, 70, 'proper'],
    [38, 72, 'touch'],
    [38, 74, 'proper'],
    [38, 78, 'proper'],
    [40, 2, 'touch'],
    [40, 8, 'touch'],
    [40, 12, 'proper'],
    [40, 16, 'proper'],
    [40, 18, 'proper'],
    [40, 20, 'touch'],
    [40, 28, 'proper'],
    [40, 32, 'touch'],
    [40, 36, 'touch'],
    [40, 58, 'collinearTouch'],
    [40, 68, 'proper'],
    [42, 6, 'touch'],
    [42, 12, 'proper'],
    [42, 14, 'proper'],
    [42, 18, 'proper'],
    [42, 24, 'touch'],
    [42, 30, 'proper'],
    [42, 36, 'proper'],
    [42, 38, 'touch'],
    [42, 40, 'touch'],
    [42, 42, 'proper'],
    [42, 46, 'proper'],
    [42, 52, 'touch'],
    [42, 60, 'proper'],
    [42, 64, 'proper'],
    [42, 68, 'touch'],
    [42, 70, 'proper'],
    [42, 72, 'touch'],
    [42, 74, 'proper'],
    [42, 76, 'overlap'],
    [42, 78, 'touch'],
    [44, 0, 'proper'],
    [44, 2, 'proper'],
    [44, 6, 'touch'],
    [44, 12, 'proper'],
    [44, 16, 'proper'],
    [44, 20, 'touch'],
    [44, 26, 'proper'],
    [44, 28, 'proper'],
    [44, 32, 'touch'],
    [44, 44, 'proper'],
    [44, 68, 'proper'],
    [46, 0, 'proper'],
    [46, 2, 'proper'],
    [46, 12, 'proper'],
    [46, 16, 'proper'],
    [46, 18, 'proper'],
    [46, 20, 'proper'],
    [46, 28, 'proper'],
    [46, 32, 'proper'],
    [46, 36, 'proper'],
    [46, 68, 'proper'],
    [46, 74, 'proper'],
    [48, 4, 'touch'],
    [48, 10, 'touch'],
    [48, 44, 'touch'],
    [48, 54, 'touch'],
    [48, 62, 'touch'],
    [48, 72, 'touch'],
    [50, 0, 'proper'],
    [50, 2, 'proper'],
    [50, 4, 'proper'],
    [50, 6, 'proper'],
    [50, 8, 'touch'],
    [50, 10, 'proper'],
    [50, 12, 'proper'],
    [50, 14, 'proper'],
    [50, 18, 'proper'],
    [50, 24, 'proper'],
    [50, 30, 'proper'],
    [50, 36, 'proper'],
    [50, 42, 'proper'],
    [50, 50, 'proper'],
    [50, 52, 'proper'],
    [50, 60, 'proper'],
    [50, 64, 'proper'],
    [50, 68, 'proper'],
    [50, 70, 'proper'],
    [50, 72, 'proper'],
    [50, 74, 'touch'],
    [50, 78, 'touch'],
    [52, 2, 'proper'],
    [52, 6, 'proper'],
    [52, 12, 'proper'],
    [52, 16, 'touch'],
    [52, 20, 'touch'],
    [52, 26, 'touch'],
    [52, 44, 'touch'],
    [52, 50, 'touch'],
    [52, 64, 'touch'],
    [54, 0, 'touch'],
    [54, 8, 'overlap'],
    [54, 40, 'touch'],
    [54, 50, 'touch'],
    [58, 62, 'collinearTouch'],
    [60, 16, 'touch'],
    [60, 26, 'touch'],
    [60, 36, 'proper'],
    [60, 44, 'touch'],
    [60, 64, 'touch'],
    [60, 74, 'proper'],
    [62, 10, 'touch'],
    [62, 14, 'touch'],
    [62, 18, 'touch'],
    [62, 22, 'proper'],
    [62, 24, 'proper'],
    [62, 38, 'proper'],
    [62, 46, 'proper'],
    [62, 50, 'proper'],
    [62, 52, 'proper'],
    [62, 60, 'touch'],
    [62, 70, 'touch'],
    [64, 0, 'touch'],
    [64, 4, 'proper'],
    [64, 10, 'proper'],
    [64, 14, 'proper'],
    [64, 30, 'proper'],
    [64, 32, 'proper'],
    [64, 44, 'proper'],
    [64, 48, 'proper'],
    [64, 52, 'proper'],
    [64, 54, 'proper'],
    [64, 56, 'touch'],
    [64, 58, 'proper'],
    [64, 60, 'proper'],
    [64, 70, 'proper'],
    [64, 72, 'proper'],
    [64, 74, 'proper'],
    [64, 78, 'proper'],
    [66, 2, 'touch'],
    [66, 16, 'touch'],
    [66, 28, 'touch'],
    [66, 34, 'touch'],
    [68, 2, 'proper'],
    [68, 6, 'proper'],
    [68, 12, 'proper'],
    [68, 20, 'proper'],
    [68, 32, 'proper'],
    [68, 34, 'touch'],
    [68, 36, 'proper'],
    [68, 42, 'touch'],
    [68, 44, 'proper'],
    [68, 54, 'touch'],
    [68, 58, 'touch'],
    [68, 64, 'proper'],
    [68, 66, 'touch'],
    [68, 74, 'proper'],
    [68, 78, 'touch'],
    [70, 0, 'proper'],
    [70, 14, 'proper'],
    [70, 28, 'proper'],
    [70, 30, 'proper'],
    [70, 34, 'proper'],
    [70, 48, 'touch'],
    [70, 52, 'touch'],
    [70, 54, 'touch'],
    [70, 58, 'touch'],
    [70, 60, 'touch'],
    [70, 66, 'proper'],
    [70, 70, 'touch'],
    [70, 72, 'touch'],
    [72, 0, 'proper'],
    [72, 2, 'proper'],
    [72, 6, 'proper'],
    [72, 8, 'touch'],
    [72, 12, 'proper'],
    [72, 14, 'proper'],
    [72, 18, 'proper'],
    [72, 20, 'proper'],
    [72, 30, 'proper'],
    [72, 32, 'proper'],
    [72, 36, 'proper'],
    [72, 42, 'proper'],
    [72, 44, 'proper'],
    [72, 48, 'touch'],
    [72, 52, 'touch'],
    [72, 54, 'touch'],
    [72, 58, 'touch'],
    [72, 60, 'touch'],
    [72, 64, 'proper'],
    [72, 68, 'proper'],
    [72, 70, 'touch'],
    [72, 72, 'touch'],
    [72, 74, 'proper'],
    [72, 78, 'proper'],
    [74, 0, 'proper'],
    [74, 2, 'touch'],
    [74, 14, 'proper'],
    [74, 30, 'proper'],
    [74, 34, 'proper'],
    [74, 48, 'proper'],
    [74, 58, 'touch'],
    [74, 62, 'touch'],
    [74, 66, 'proper'],
    [74, 72, 'proper'],
    [76, 14, 'touch'],
    [76, 30, 'touch'],
    [76, 72, 'overlap'],
    [78, 0, 'proper'],
    [78, 22, 'touch'],
    [78, 24, 'touch'],
    [78, 28, 'touch'],
    [78, 46, 'touch'],
    [78, 50, 'proper'],
    [78, 68, 'proper'],
    [78, 74, 'touch']
  ]
} as const;
