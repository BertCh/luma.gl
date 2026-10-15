// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {canonicalizeNoding, nodingFixtures} from './parity-wp3-noding-fixture';

for (const fixture of nodingFixtures) {
  it(`canonical comparison integrity, not GPU parity: ${fixture.case}`, () => {
    const nodeCount = fixture.nodePositions.length / 2;
    const coordinateBits = new Uint32Array(Float32Array.from(fixture.nodePositions).buffer);
    // Reproduce the unsigned float-bit numbering behind the r6 failures, and
    // also test an unrelated permutation. Geometry and incidence are unchanged.
    const numericIds = Array.from({length: nodeCount}, (_, node) => node);
    for (const nodeOrder of [
      [...numericIds].sort(
        (first, second) =>
          coordinateBits[2 * first] - coordinateBits[2 * second] ||
          coordinateBits[2 * first + 1] - coordinateBits[2 * second + 1]
      ),
      [...numericIds].reverse()
    ]) {
      const permutedIds = new Array<number>(nodeCount);
      nodeOrder.forEach((node, index) => {
        permutedIds[node] = index;
      });
      const entries: number[] = [];
      const offsets = [0];
      for (const node of nodeOrder) {
        for (let entry = fixture.offsets[node]; entry < fixture.offsets[node + 1]; entry++) {
          entries.push(entry);
        }
        offsets.push(entries.length);
      }
      const permuted = {
        nodePositions: nodeOrder.flatMap(node =>
          fixture.nodePositions.slice(2 * node, 2 * node + 2)
        ),
        fromNodes: fixture.fromNodes.map(node => permutedIds[node]),
        toNodes: fixture.toNodes.map(node => permutedIds[node]),
        offsets,
        neighbors: entries.map(entry => permutedIds[fixture.neighbors[entry]]),
        edgeIds: entries.map(entry => fixture.edgeIds[entry]),
        weights: entries.map(entry => fixture.weights[entry])
      };
      const expected = canonicalizeNoding(fixture);
      expect(canonicalizeNoding(permuted)).toEqual(expected);
      // Relabeling must not erase bad geometry, topology, provenance or weights.
      for (const field of ['nodePositions', 'neighbors', 'edgeIds', 'weights'] as const) {
        const corrupted = {...permuted, [field]: [...permuted[field]]};
        corrupted[field][0] += 1;
        expect(canonicalizeNoding(corrupted)).not.toEqual(expected);
      }
      if (nodeCount === 1) continue;
      const rewired = {...permuted, toNodes: [...permuted.toNodes]};
      rewired.toNodes[0] = rewired.fromNodes[0];
      expect(() => {
        expect(canonicalizeNoding(rewired)).toEqual(expected);
      }).toThrow();
    }
  });

  it(`reference integrity, not GPU parity: Shapely noding ${fixture.case}`, () => {
    expect(fixture.versions['shapely']).toBeTruthy();
    expect(fixture.geosHausdorff).toBeLessThanOrEqual(Math.SQRT2 * fixture.coordinateBound);
    expect(fixture.intersectionCapacity).toBeGreaterThan(fixture.boundingBoxPairCount);
    const pieces = fixture.sourceIds.length;
    const nodes = fixture.nodePositions.length / 2;
    expect(fixture.pieceOffsets).toHaveLength(pieces + 1);
    expect(fixture.offsets).toHaveLength(nodes + 1);
    expect(fixture.offsets[nodes]).toBe(2 * pieces);
    const sourceTotals = fixture.sourceLengths.map(() => 0);
    const sourceBounds = fixture.sourceLengths.map(() => 0);
    for (let edge = 0; edge < pieces; edge++) {
      let length = 0;
      for (
        let vertex = fixture.pieceOffsets[edge];
        vertex + 1 < fixture.pieceOffsets[edge + 1];
        vertex++
      ) {
        length += Math.hypot(
          fixture.piecePositions[2 * vertex + 2] - fixture.piecePositions[2 * vertex],
          fixture.piecePositions[2 * vertex + 3] - fixture.piecePositions[2 * vertex + 1]
        );
      }
      // JS and GEOS sum identical double norms; 16 binary64 ulps per term
      // bound their differing sqrt/hypot and reduction evaluation order.
      const terms = fixture.pieceOffsets[edge + 1] - fixture.pieceOffsets[edge] - 1;
      expect(Math.abs(length - fixture.lengths[edge])).toBeLessThanOrEqual(
        16 * Number.EPSILON * terms * Math.max(1, length)
      );
      sourceTotals[fixture.sourceIds[edge]] += length;
      sourceBounds[fixture.sourceIds[edge]] += fixture.lengthBounds[edge];
      expect(fixture.fromNodes[edge]).toBeLessThan(nodes);
      expect(fixture.toNodes[edge]).toBeLessThan(nodes);
    }
    const occurrences = new Array(pieces).fill(0);
    for (let node = 0; node < nodes; node++) {
      for (let entry = fixture.offsets[node]; entry < fixture.offsets[node + 1]; entry++) {
        const edge = fixture.edgeIds[entry];
        occurrences[edge]++;
        expect(
          [fixture.fromNodes[edge], fixture.toNodes[edge]].sort((first, second) => first - second)
        ).toEqual([node, fixture.neighbors[entry]].sort((first, second) => first - second));
        expect(fixture.weights[entry]).toBe(fixture.lengths[edge]);
      }
    }
    expect(occurrences).toEqual(new Array(pieces).fill(2));
    for (const [source, length] of sourceTotals.entries())
      expect(Math.abs(length - fixture.sourceLengths[source])).toBeLessThanOrEqual(
        sourceBounds[source]
      );
  });
}
