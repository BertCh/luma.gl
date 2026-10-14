// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {OraclePolygonFeature} from './spatial-join-oracle';

/** Axis-aligned rectangle as a polygon feature. */
function box(x0: number, y0: number, x1: number, y1: number): OraclePolygonFeature {
  return [
    [
      [
        [x0, y0],
        [x1, y0],
        [x1, y1],
        [x0, y1]
      ]
    ]
  ];
}

/** A scene of a coverage and the owner of each probe point under the half-open rule. */
export type HalfOpenScene = {
  features: OraclePolygonFeature[];
  points: [number, number][];
  /** Owning feature row per point, `null` when no feature owns the point. */
  owners: (number | null)[];
};

/**
 * Owners under the half-open rule, from `shapely.Polygon.contains(Point(x + 1e-9, y + 1e-6))`:
 * the rule assigns a boundary point to the feature containing it after an infinitesimal move toward
 * +x and +y (+y dominates). Script: scratchpad/fix/work/E/E2-spatial-join/half_open_fixtures.py,
 * shapely 2.2.0. `squares` is a 2x2 coverage of side-2 squares (A bottom-left, B bottom-right,
 * C top-left, D top-right) probed on shared edges, the shared vertex, outer edges and corners;
 * `triangles` is a square cut along its diagonal.
 */
export const HALF_OPEN_SCENES: Record<'squares' | 'triangles', HalfOpenScene> = {
  squares: {
    features: [box(0, 0, 2, 2), box(2, 0, 4, 2), box(0, 2, 2, 4), box(2, 2, 4, 4)],
    points: [
      [2, 1],
      [1, 2],
      [2, 2],
      [2, 3],
      [3, 2],
      [0, 1],
      [1, 0],
      [0, 0],
      [4, 1],
      [1, 4],
      [4, 4],
      [4, 0],
      [0, 4],
      [1, 1],
      [3, 3],
      [5, 5]
    ],
    owners: [1, 2, 3, 3, 3, 0, 0, 0, null, null, null, null, null, 0, 3, null]
  },
  triangles: {
    features: [
      [
        [
          [
            [0, 0],
            [2, 0],
            [2, 2]
          ]
        ]
      ],
      [
        [
          [
            [0, 0],
            [2, 2],
            [0, 2]
          ]
        ]
      ]
    ],
    points: [
      [1, 1],
      [0, 0],
      [2, 2],
      [2, 0],
      [0, 2],
      [1, 0],
      [2, 1],
      [0, 1],
      [1, 2],
      [1.5, 1.5],
      [0.5, 0.5],
      [3, 3]
    ],
    owners: [1, 1, null, null, null, 0, null, 1, null, 1, 1, null]
  }
};
