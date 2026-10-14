// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Four-level polygon layout (GeoArrow) of nested features, one polygon per feature. */
export type PolygonLayout = {
  positions: Float32Array;
  ringOffsets: Uint32Array;
  /** Polygon-to-ring offsets; polygon `f` owns the rings of feature `f`. */
  polygonOffsets: Uint32Array;
  /** Feature-to-polygon offsets, `0..featureCount`. */
  featureOffsets: Uint32Array;
};

/**
 * Packs `features[feature][ring][vertex] = [x, y]` into the four-level layout. Every feature is one
 * polygon whose first ring is the shell and whose later rings are holes.
 */
export function flattenPolygonFeatures(features: number[][][][]): PolygonLayout {
  const positions: number[] = [];
  const ringOffsets = [0];
  const polygonOffsets = [0];
  const featureOffsets = [0];
  for (const rings of features) {
    for (const ring of rings) {
      for (const [x, y] of ring) {
        positions.push(x, y);
      }
      ringOffsets.push(positions.length / 2);
    }
    polygonOffsets.push(ringOffsets.length - 1);
    featureOffsets.push(polygonOffsets.length - 1);
  }
  return {
    positions: new Float32Array(positions),
    ringOffsets: new Uint32Array(ringOffsets),
    polygonOffsets: new Uint32Array(polygonOffsets),
    featureOffsets: new Uint32Array(featureOffsets)
  };
}
