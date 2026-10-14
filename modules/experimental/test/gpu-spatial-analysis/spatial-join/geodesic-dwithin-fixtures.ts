// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Great-circle `dwithin` oracle values on the sphere R = 6371008.8 m.
 *
 * Script: scratchpad/fix/work/E/E2-spatial-join/geodesic_dwithin_fixtures.py (numpy 2.5.4
 * haversine, cross-checked with pyproj 3.8.0 `Geod(a=R, f=0)`; polygon distances from shapely
 * 2.2.0 in the azimuthal equidistant projection centered on each query point).
 *
 * The point pairs `i == j` are the close pairs: 1.11 km at the equator, at latitude 60 (where the
 * same longitude step is half as far), across the antimeridian, 2.22 km over the pole, 78.6 m at
 * latitude -45 and 54.8 km at latitude 10. Every other pair is thousands of kilometers apart.
 */
export const GEODESIC_DWITHIN_POINTS = {
  left: [
    [0, 0],
    [10, 60],
    [179.995, 0],
    [0, 89.99],
    [100, -45],
    [-70, 10]
  ] as [number, number][],
  right: [
    [0.01, 0],
    [10.02, 60],
    [-179.995, 0],
    [180, 89.99],
    [100.001, -45],
    [-70.5, 10]
  ] as [number, number][],
  diagonalDistances: [
    1111.9508023353292, 1111.9507981013535, 1111.9508023368205, 2223.9016046713, 78.62679526800973,
    54752.883316571
  ],
  /** Pairs within 1500 m, and within 2500 m (adds the pole pair). */
  pairsWithin1500: [
    [0, 0],
    [1, 1],
    [2, 2],
    [4, 4]
  ] as [number, number][],
  pairsWithin2500: [
    [0, 0],
    [1, 1],
    [2, 2],
    [3, 3],
    [4, 4]
  ] as [number, number][]
};

/** A small polygon at latitude 60 and query points around it, with distances in meters. */
export const GEODESIC_DWITHIN_POLYGON = {
  ring: [
    [10, 60],
    [10.01, 60],
    [10.01, 60.005],
    [10, 60.005]
  ] as [number, number][],
  queries: [
    [10.02, 60.0025],
    [10.005, 60.002],
    [10.005, 60.02],
    [11, 61]
  ] as [number, number][],
  distances: [555.9333803424286, 0.0, 1667.9156990472075, 123199.80868050527]
};
