// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {MATCH_FIXTURES, compareValues, EXACT} from './parity-wp3-map-matching-fixture';

// These are fixture/provenance checks, not an execution of GPUMapMatching.
for (const fixture of MATCH_FIXTURES) {
  it(`map matching reference integrity: ${fixture.case}`, () => {
    const nodeCount = fixture.positions.length / 2;
    expect(fixture.offsets).toHaveLength(nodeCount + 1);
    expect(fixture.offsets[0]).toBe(0);
    expect(fixture.offsets[nodeCount]).toBe(fixture.neighbors.length);
    expect(fixture.trackOffsets.at(-1)).toBe(fixture.points.length / 2);
    expect(fixture.expected).toHaveLength(fixture.trackOffsets.length - 1);
    expect(fixture.versions['networkx']).toBeTruthy();
    expect(fixture.versions['scipy']).toBeTruthy();
    expect(fixture.versions['shapely']).toBeTruthy();
    expect(fixture.diagnostics.nonNearestMatches).toBeGreaterThan(0);
    // Conservative full Dijkstra balls fit inside the GPU's route table budget.
    expect(fixture.diagnostics.maximumDijkstraBall).toBeLessThan(128);
    const sources = fixture.neighbors.map(() => 0);
    for (let node = 0; node < nodeCount; node++) {
      expect(fixture.offsets[node + 1]).toBeGreaterThanOrEqual(fixture.offsets[node]);
      for (let edge = fixture.offsets[node]; edge < fixture.offsets[node + 1]; edge++) {
        sources[edge] = node;
        expect(fixture.neighbors[edge]).toBeLessThan(nodeCount);
      }
    }
    const getLength = (source: number, target: number) =>
      Math.hypot(
        fixture.positions[target * 2] - fixture.positions[source * 2],
        fixture.positions[target * 2 + 1] - fixture.positions[source * 2 + 1]
      );
    const routeLengths: number[] = [];
    const referenceLengths: number[] = [];
    const likelihoods: number[] = [];
    const referenceLikelihoods: number[] = [];
    for (let track = 0; track < fixture.expected.length; track++) {
      const reference = fixture.expected[track];
      const pointCount = fixture.trackOffsets[track + 1] - fixture.trackOffsets[track];
      for (const alternative of reference.alternatives) {
        expect(alternative.edges).toHaveLength(pointCount);
        expect(alternative.positions).toHaveLength(pointCount * 2);
        let likelihood = 0;
        for (let point = 0; point < pointCount; point++) {
          const edge = alternative.edges[point];
          if (edge === 0xffffffff) {
            expect(fixture.candidates[track][point]).toHaveLength(0);
            continue;
          }
          expect(fixture.candidates[track][point].some(candidate => candidate.edge === edge)).toBe(
            true
          );
          likelihood -=
            Math.log(fixture.settings.sigma * Math.sqrt(2 * Math.PI)) +
            0.5 * (alternative.distances[point] / fixture.settings.sigma) ** 2;
          if (point === 0) {
            continue;
          }
          const previousEdge = alternative.edges[point - 1];
          const route = alternative.routes[point - 1];
          let length = alternative.offsets[point] - alternative.offsets[point - 1];
          if (route.nodePath.length > 0) {
            expect(route.nodePath[0]).toBe(fixture.neighbors[previousEdge]);
            expect(route.nodePath.at(-1)).toBe(sources[edge]);
            length =
              getLength(sources[previousEdge], fixture.neighbors[previousEdge]) -
              alternative.offsets[point - 1] +
              alternative.offsets[point];
            for (let hop = 1; hop < route.nodePath.length; hop++) {
              const source = route.nodePath[hop - 1];
              const target = route.nodePath[hop];
              expect(
                fixture.neighbors.slice(fixture.offsets[source], fixture.offsets[source + 1])
              ).toContain(target);
              length += getLength(source, target);
            }
          } else {
            expect(previousEdge).toBe(edge);
            expect(length).toBeGreaterThanOrEqual(0);
          }
          routeLengths.push(length);
          referenceLengths.push(route.length);
          const pointIndex = fixture.trackOffsets[track] + point;
          const straight = Math.hypot(
            fixture.points[pointIndex * 2] - fixture.points[(pointIndex - 1) * 2],
            fixture.points[pointIndex * 2 + 1] - fixture.points[(pointIndex - 1) * 2 + 1]
          );
          expect(length).toBeLessThanOrEqual(
            straight * fixture.settings.routeFactor! + fixture.settings.routeSlack! + 1e-10
          );
          likelihood -=
            Math.log(fixture.settings.beta) + Math.abs(length - straight) / fixture.settings.beta;
        }
        likelihoods.push(likelihood);
        referenceLikelihoods.push(reference.logLikelihood);
      }
    }
    compareValues(
      fixture,
      'oracle integrity: reconstructed route costs (CPU only)',
      routeLengths,
      referenceLengths,
      {absolute: 1e-10, relative: 1e-12}
    );
    compareValues(
      fixture,
      'oracle integrity: normalized likelihood (CPU only)',
      likelihoods,
      referenceLikelihoods,
      {absolute: 1e-10, relative: 1e-12}
    );
    if (fixture.case === 'chicago') {
      expect(fixture.source.nodeIds).toHaveLength(nodeCount);
      expect(fixture.source.edgeIds).toHaveLength(fixture.neighbors.length);
      expect(fixture.source.fullNodeCount).toBe(29557);
      expect(fixture.source.fullEdgeCount).toBe(77140);
      expect(fixture.source.fullGraphDistanceComparisons).toBeGreaterThan(0);
      compareValues(
        fixture,
        'oracle integrity: full graph versus subset cost error (generator evidence)',
        [fixture.source.maximumSubsetCostError!],
        [0],
        EXACT
      );
    } else {
      expect(fixture.expected[2].alternatives).toHaveLength(2);
      expect(fixture.expected.at(-1)?.alternatives[0].edges).toHaveLength(0);
    }
  });
}
