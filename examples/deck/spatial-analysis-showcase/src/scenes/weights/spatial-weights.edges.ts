// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Geography} from './b4-geography';

/**
 * Flags the places on the outer edge of a polygon coverage: a place with at least one boundary
 * edge that no other place shares (a coast, a lake shore, a national border, the city limit).
 * Those are the places an edge effect starves of neighbours. One pass over the ring edges with
 * exact vertex keys, the same equality the contiguity producer relies on.
 */
export function getOuterEdgeRows(geography: Geography): Uint8Array {
  const {contiguityVertices: vertices, contiguityRingOffsets: ringOffsets} = geography;
  const featureRingOffsets = geography.featureRingOffsets;
  const edgeCounts = new Map<string, number>();
  const getEdgeKey = (from: number, to: number): string => {
    const first = `${vertices[from * 2]},${vertices[from * 2 + 1]}`;
    const second = `${vertices[to * 2]},${vertices[to * 2 + 1]}`;
    return first < second ? `${first}|${second}` : `${second}|${first}`;
  };
  const visitEdges = (visit: (feature: number, key: string) => void) => {
    for (let feature = 0; feature < geography.count; feature++) {
      for (let ring = featureRingOffsets[feature]; ring < featureRingOffsets[feature + 1]; ring++) {
        const first = ringOffsets[ring];
        const end = ringOffsets[ring + 1];
        for (let vertex = first; vertex < end; vertex++) {
          visit(feature, getEdgeKey(vertex, vertex + 1 < end ? vertex + 1 : first));
        }
      }
    }
  };
  visitEdges((_feature, key) => edgeCounts.set(key, (edgeCounts.get(key) ?? 0) + 1));
  const flags = new Uint8Array(geography.count);
  visitEdges((feature, key) => {
    if (edgeCounts.get(key) === 1) flags[feature] = 1;
  });
  return flags;
}
