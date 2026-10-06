// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  generateRandomFeatures,
  line,
  point,
  polygonWithRings,
  rectangle,
  triangle,
  type OracleFeature
} from './spatial-predicate-oracle';

/** Kinds in the order used by the Shapely fixtures. */
export const RELATE_KINDS: OracleFeature['kind'][] = ['points', 'lines', 'polygons'];

/** Number of random features per side in the pinned random scenes. */
export const RANDOM_RELATE_COUNT = 8;

/** Seeds of the pinned random scene of `(leftKind, rightKind)`. */
export function getRandomRelateSeeds(
  leftKind: OracleFeature['kind'],
  rightKind: OracleFeature['kind']
): [number, number] {
  const index = RELATE_KINDS.indexOf(leftKind) * 3 + RELATE_KINDS.indexOf(rightKind);
  return [1000 + index * 2, 1001 + index * 2];
}

/** Hand-built features per kind: shared edges, touching corners, collinear overlaps, holes, endpoints. */
export function getHandRelateScene(kind: OracleFeature['kind']): OracleFeature[] {
  const hole = polygonWithRings(
    [
      [0, 0],
      [6, 0],
      [6, 6],
      [0, 6]
    ],
    [
      [2, 2],
      [2, 4],
      [4, 4],
      [4, 2]
    ]
  );
  if (kind === 'points') {
    return [
      point(0, 0),
      point(3, 3),
      point(2, 2),
      point(2, 3),
      point(6, 6),
      point(7, 7),
      point(1, 1),
      point(3, 0),
      point(4.5, 4.5),
      point(10, 10)
    ];
  }
  if (kind === 'lines') {
    return [
      line([0, 0], [6, 0]),
      line([1, 0], [3, 0]),
      line([2, 2], [4, 4]),
      line([3, 1], [3, 5]),
      line([1, 1], [5, 1]),
      line([6, 6], [8, 8]),
      line([0, 0], [4, 0], [4, 4], [0, 4], [0, 0]),
      line([7, 0], [9, 0]),
      line([2, 3], [4, 3]),
      line([0, 3], [6, 3])
    ];
  }
  return [
    hole,
    rectangle(0, 0, 6, 6),
    rectangle(2, 2, 2, 2),
    rectangle(6, 0, 2, 2),
    rectangle(6, 6, 2, 2),
    triangle(0, 0, 3),
    rectangle(1, 1, 4, 4),
    rectangle(2.5, 2.5, 1, 1),
    rectangle(1, 1, 1, 1),
    rectangle(10, 10, 1, 1)
  ];
}

/** The features of one side of one pinned scene: hand scene first, then random features. */
export function getRelateSide(
  kind: OracleFeature['kind'],
  seed: number,
  includeHand: boolean
): OracleFeature[] {
  return [
    ...(includeHand ? getHandRelateScene(kind) : []),
    ...generateRandomFeatures(kind, RANDOM_RELATE_COUNT, seed)
  ];
}
