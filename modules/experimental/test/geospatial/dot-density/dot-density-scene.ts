// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {DotDensityCPUPolygons} from '../../../src/geospatial/dot-density/dot-density-cpu';

/** One feature: polygons, each a list of rings (shell first), each a list of `[x, y]`. */
export type SceneFeature = number[][][][];

/** Packs features into GeoArrow polygon columns. */
export function createPolygonColumns(features: SceneFeature[]): DotDensityCPUPolygons {
  const positions: number[] = [];
  const featureOffsets = [0];
  const polygonOffsets = [0];
  const ringOffsets = [0];
  for (const feature of features) {
    for (const polygon of feature) {
      for (const ring of polygon) {
        for (const [x, y] of ring) {
          positions.push(x, y);
        }
        ringOffsets.push(positions.length / 2);
      }
      polygonOffsets.push(ringOffsets.length - 1);
    }
    featureOffsets.push(polygonOffsets.length - 1);
  }
  return {
    polygonPositions: new Float32Array(positions),
    featureOffsets: new Uint32Array(featureOffsets),
    polygonOffsets: new Uint32Array(polygonOffsets),
    ringOffsets: new Uint32Array(ringOffsets)
  };
}

/** Square with a square hole (closing vertex repeated), two disjoint triangles, and a sliver. */
export const SCENE_FEATURES: SceneFeature[] = [
  [
    [
      [
        [0, 0],
        [10, 0],
        [10, 10],
        [0, 10],
        [0, 0]
      ],
      [
        [4, 4],
        [6, 4],
        [6, 6],
        [4, 6]
      ]
    ]
  ],
  [
    [
      [
        [20, 0],
        [30, 0],
        [20, 10]
      ]
    ],
    [
      [
        [30, 10],
        [30, 5],
        [25, 10]
      ]
    ]
  ],
  [
    [
      [
        [40, 0],
        [50, 10],
        [49.98, 10]
      ]
    ]
  ]
];

/** Returns whether a point lies in the hole of feature 0. */
export function isInSceneHole(x: number, y: number): boolean {
  return x > 4 && x < 6 && y > 4 && y < 6;
}
