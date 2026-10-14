// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  AMBIGUOUS_180_EDGE,
  ANTARCTICA_NATURAL_EARTH,
  ARCTIC_RING,
  CHICAGO_SQUARE,
  FIJI_SPLIT,
  FIJI_TRANSMERIDIAN,
  FIJI_WITH_SEAM_HOLE,
  GREAT_CIRCLE_60_DEGREE_EDGE,
  LON_LAT_POLYGON_FIXTURES,
  SOUTHERN_POLE_RING,
  SPIRAL_WIDER_THAN_360
} from './fixtures';
import {
  capPoleRing,
  chamberlainDuquetteArea,
  classifyRing,
  densifyRing,
  flattenLonLatFeatures,
  getLonLatFeatures,
  interpolateGreatCircle,
  isAmbiguousGreatCircleEdge,
  isAmbiguousLinearEdge,
  normalizePolygon,
  pointInLonLatPolygonPrepared,
  prepareLonLatPolygons,
  ringWinding,
  sphericalPolygonArea,
  wrapLongitudeStep,
  type LonLatPolygonData,
  type LonLatPosition
} from './lon-lat-polygon-oracle';

function firstRing(fixture: LonLatPolygonData) {
  return getLonLatFeatures(fixture)[0][0][0];
}

it('classifies every shared fixture using closed short-way edges and raw pole edges', () => {
  const expected: Record<string, [boolean, boolean, boolean, boolean]> = {
    FIJI_SPLIT: [false, false, false, false],
    FIJI_TRANSMERIDIAN: [true, false, false, false],
    FIJI_WITH_SEAM_HOLE: [true, false, false, false],
    ANTARCTICA_NATURAL_EARTH: [false, false, false, false],
    ARCTIC_RING: [true, true, false, false],
    SOUTHERN_POLE_RING: [true, true, false, false],
    AMBIGUOUS_180_EDGE: [false, false, true, false],
    SPIRAL_WIDER_THAN_360: [true, true, false, true],
    CHICAGO_SQUARE: [false, false, false, false],
    GREAT_CIRCLE_60_DEGREE_EDGE: [false, false, false, false]
  };
  for (const [name, fixture] of Object.entries(LON_LAT_POLYGON_FIXTURES)) {
    for (const ring of getLonLatFeatures(fixture).flat(2)) {
      const result = classifyRing(ring, 'linear');
      expect(
        [result.crossing, result.poleRing, result.ambiguous, result.widerThan360],
        name
      ).toEqual(expected[name]);
      expect(result.invalidValue).toBe(false);
    }
  }
});

it('uses ties-to-even steps, implicit closure, and ignores rings with fewer than three distinct vertices', () => {
  expect([180, -180, 540, -540].map(wrapLongitudeStep)).toEqual([180, -180, -180, 180]);
  const ring = firstRing(ARCTIC_RING);
  expect(ringWinding(ring)).toBe(360);
  expect(ringWinding([...ring, ring[0]])).toBe(360);
  expect(ringWinding([])).toBe(0);
  expect(
    classifyRing([
      [179, 0],
      [-179, 0],
      [179, 0],
      [-179, 0]
    ]).crossing
  ).toBe(false);
  expect(
    classifyRing([
      [0, 0],
      [1, 91],
      [2, 0]
    ]).invalidValue
  ).toBe(true);
  expect(
    classifyRing([
      [0, 0],
      [1, Number.NaN],
      [2, 0]
    ]).invalidValue
  ).toBe(true);
});

it('great-circle ambiguity distinguishes antipodes, pole-crossing minor arcs and the opposite major arc', () => {
  expect(isAmbiguousGreatCircleEdge([0, 0], [180, 0])).toBe(true);
  expect(isAmbiguousGreatCircleEdge([0, 70], [180, 70])).toBe(true);
  expect(isAmbiguousGreatCircleEdge([0, -70], [180, -70])).toBe(true);
  expect(isAmbiguousGreatCircleEdge([0, 60], [0, 70])).toBe(false);
  expect(isAmbiguousGreatCircleEdge([0, 70], [179, 70])).toBe(false);
  expect(isAmbiguousGreatCircleEdge([0, 90], [0, 70])).toBe(true);
  expect(isAmbiguousGreatCircleEdge([180, -90], [-180, -90])).toBe(false);
  expect(isAmbiguousGreatCircleEdge([0, 0], [180 - 0.00001, 0])).toBe(true);
  expect(isAmbiguousGreatCircleEdge([0, 0], [180 - 0.001, 0])).toBe(false);
  expect(isAmbiguousLinearEdge([0, 0], [180 - 0.00005, 10])).toBe(true);
  expect(isAmbiguousLinearEdge([0, 0], [180 - 0.001, 10])).toBe(false);
});

it('splits Fiji into two copies, retaining original vertices bit for bit', () => {
  const prepared = prepareLonLatPolygons(FIJI_TRANSMERIDIAN);
  expect(Array.from(prepared.featureOffsets)).toEqual([0, 2]);
  expect(Array.from(prepared.polygonOffsets)).toEqual([0, 1, 2]);
  expect(Array.from(prepared.ringOffsets)).toEqual([0, 4, 8]);
  expect(Array.from(prepared.ringFlags)).toEqual([1]);
  const output = getLonLatFeatures(prepared).flat(3);
  for (const position of firstRing(FIJI_TRANSMERIDIAN)) {
    expect(
      output.some(
        candidate => Object.is(candidate[0], position[0]) && Object.is(candidate[1], position[1])
      )
    ).toBe(true);
  }
  expect(pointInLonLatPolygonPrepared([179, -18], prepared)).toBe(true);
  expect(pointInLonLatPolygonPrepared([-179, -18], prepared)).toBe(true);
  expect(pointInLonLatPolygonPrepared([0, -18], prepared)).toBe(false);
  expect(pointInLonLatPolygonPrepared([-180, -18], prepared)).toBe(true);
  expect(pointInLonLatPolygonPrepared([180, -18], prepared)).toBe(false);
});

it('aligns seam holes and preserves their roles in both copies', () => {
  const prepared = prepareLonLatPolygons(FIJI_WITH_SEAM_HOLE);
  expect(Array.from(prepared.polygonOffsets)).toEqual([0, 2, 4]);
  expect(pointInLonLatPolygonPrepared([179.5, -17.5], prepared)).toBe(false);
  expect(pointInLonLatPolygonPrepared([-179.5, -17.5], prepared)).toBe(false);
  expect(pointInLonLatPolygonPrepared([178, -17.5], prepared)).toBe(true);
  const polygon = normalizePolygon([
    [
      [177, 0],
      [182, 0],
      [182, 5]
    ],
    [
      [-179, 1],
      [-178, 1],
      [-178, 2]
    ]
  ]);
  expect(polygon[1][0][0]).toBe(181);
});

it('lays out every main polygon before every east copy within each unchanged feature row', () => {
  const prepared = prepareLonLatPolygons([
    getLonLatFeatures(FIJI_SPLIT)[0],
    getLonLatFeatures(CHICAGO_SQUARE)[0]
  ]);
  expect(Array.from(prepared.featureOffsets)).toEqual([0, 4, 6]);
  expect(Array.from(prepared.polygonOffsets)).toEqual([0, 1, 2, 3, 4, 5, 6]);
  expect(Array.from(prepared.ringOffsets)).toEqual([0, 4, 8, 8, 8, 12, 12]);
  expect(Array.from(prepared.ringFlags)).toEqual([0, 0, 0]);
});

it('caps both windings at the nearer pole, preserves closure and spans exactly 360 degrees', () => {
  for (const fixture of [ARCTIC_RING, SOUTHERN_POLE_RING]) {
    const ring = firstRing(fixture);
    const capped = capPoleRing(densifyRing(ring).unwrapped);
    const longitudes = capped.map(position => position[0]);
    expect(Math.max(...longitudes) - Math.min(...longitudes)).toBe(360);
    expect(capped.at(-1)?.[1]).toBe(Math.sign(ring[0][1]) * 90);
    expect(capped[capped.length - 3][1]).toBe(ring[0][1]);
    const prepared = prepareLonLatPolygons(fixture);
    expect(prepared.status.invalidCount).toBe(0);
    expect(prepared.ringFlags[0] & 2).toBe(2);
    for (const longitude of [-179, -100, 0, 100, 179]) {
      expect(pointInLonLatPolygonPrepared([longitude, Math.sign(ring[0][1]) * 85], prepared)).toBe(
        true
      );
      expect(pointInLonLatPolygonPrepared([longitude, 0], prepared)).toBe(false);
    }
  }
  expect(() =>
    capPoleRing([
      [0, -10],
      [120, 0],
      [240, 10]
    ])
  ).toThrow('ambiguous pole');
});

it('keeps Antarctica finite and unchanged, including its raw pole-line edge', () => {
  const prepared = prepareLonLatPolygons(ANTARCTICA_NATURAL_EARTH);
  expect(Array.from(prepared.positions)).toEqual(Array.from(ANTARCTICA_NATURAL_EARTH.positions));
  expect(prepared.status.invalidCount).toBe(0);
  expect(ringWinding(firstRing(ANTARCTICA_NATURAL_EARTH))).toBe(0);
  expect(chamberlainDuquetteArea(firstRing(ANTARCTICA_NATURAL_EARTH), 6371008.8)).toBeGreaterThan(
    1e12
  );
});

it('rejects whole features for invalid rings, option rejection and ambiguous pole choice', () => {
  for (const fixture of [AMBIGUOUS_180_EDGE, SPIRAL_WIDER_THAN_360]) {
    const prepared = prepareLonLatPolygons(fixture);
    expect(prepared.status.invalidCount).toBe(1);
    expect(Array.from(prepared.ringOffsets)).toEqual([0, 0, 0]);
  }
  expect(
    prepareLonLatPolygons(FIJI_TRANSMERIDIAN, {antimeridian: 'reject'}).status.invalidCount
  ).toBe(1);
  expect(prepareLonLatPolygons(ARCTIC_RING, {poleRings: 'reject'}).status.invalidCount).toBe(1);
  const invalid = flattenLonLatFeatures([
    [
      [
        [
          [0, 0],
          [1, Infinity],
          [2, 0]
        ]
      ]
    ]
  ]);
  expect(prepareLonLatPolygons(invalid).ringFlags[0] & 16).toBe(16);
  expect(prepareLonLatPolygons(invalid).status.invalidCount).toBe(1);
});

it('densifies closure without repeated endpoints and retains wrapped coordinates with integer turns', () => {
  const ring = firstRing(FIJI_TRANSMERIDIAN);
  const result = densifyRing([...ring, ring[0]], 'linear', 1);
  expect(result.unwrapped.length).toBeGreaterThan(ring.length);
  expect(result.unwrapped.at(-1)).not.toEqual(result.unwrapped[0]);
  expect(result.unwrapped.at(-1)?.[1]).toBeLessThan(-16);
  for (let index = 0; index < result.vertices.length; index++) {
    const vertex = result.vertices[index];
    expect(Math.abs(vertex.wrapped[0])).toBeLessThanOrEqual(180);
    expect(Number.isInteger(vertex.turn)).toBe(true);
    expect(vertex.wrapped[0] + vertex.turn * 360).toBeCloseTo(result.unwrapped[index][0], 12);
  }
});

function separationMeters(first: LonLatPosition, second: LonLatPosition): number {
  const latitudeStep = ((second[1] - first[1]) * Math.PI) / 180;
  const longitudeStep = ((second[0] - first[0]) * Math.PI) / 180;
  const haversine =
    Math.sin(latitudeStep / 2) ** 2 +
    Math.cos((first[1] * Math.PI) / 180) *
      Math.cos((second[1] * Math.PI) / 180) *
      Math.sin(longitudeStep / 2) ** 2;
  return 2 * 6371008.8 * Math.asin(Math.sqrt(haversine));
}

it('records the 60-degree edge deviation and the audit bound counterexample at step 0.1', () => {
  const ring = firstRing(GREAT_CIRCLE_60_DEGREE_EDGE);
  const result = densifyRing(ring, 'great-circle', 0.1);
  const pieces = 600;
  let maximumDeviation = 0;
  let maximumLongitudeStep = 0;
  for (let piece = 0; piece < pieces; piece++) {
    const first = result.unwrapped[piece];
    const second = result.unwrapped[piece + 1];
    maximumLongitudeStep = Math.max(maximumLongitudeStep, Math.abs(second[0] - first[0]));
    expect(
      separationMeters(first, interpolateGreatCircle(ring[0], ring[1], piece / pieces))
    ).toBeLessThan(1e-6);
    const middle: LonLatPosition = [(first[0] + second[0]) / 2, (first[1] + second[1]) / 2];
    maximumDeviation = Math.max(
      maximumDeviation,
      separationMeters(middle, interpolateGreatCircle(ring[0], ring[1], (piece + 0.5) / pieces))
    );
  }
  // Audit section 6 assumes each output dLon is <= 0.1. Its section 5.2 piece count
  // only bounds the average dLon: equal-angle slerp pieces reach larger longitude steps.
  expect(maximumLongitudeStep).toBeGreaterThan(0.1);
  expect(maximumDeviation).toBeCloseTo(1.335894, 5);
  expect(maximumDeviation).toBeLessThanOrEqual(1.34);
  expect(result.clamped).toBe(false);
});

it('reports clamping, disabled great-circle densification and bounded output status', () => {
  const clamped = prepareLonLatPolygons(GREAT_CIRCLE_60_DEGREE_EDGE, {
    edgeModel: 'great-circle',
    maximumSegmentDegrees: 0.1,
    maximumPieceCount: 2
  });
  expect(clamped.status.approximation).toBe(1);
  expect(clamped.ringFlags[0] & 32).toBe(32);
  expect(
    prepareLonLatPolygons(GREAT_CIRCLE_60_DEGREE_EDGE, {
      edgeModel: 'great-circle',
      maximumSegmentDegrees: 0
    }).status.approximation
  ).toBe(1);
  const bounded = prepareLonLatPolygons(FIJI_TRANSMERIDIAN, {positionCapacity: 3});
  expect(bounded.status).toEqual({
    count: 3,
    requiredCount: 8,
    overflow: 1,
    invalidCount: 0,
    approximation: 0
  });
  expect(Array.from(bounded.ringOffsets)).toEqual([0, 3, 3]);
});

it('measures a spherical octant by spherical excess independent of orientation', () => {
  const radius = 6371008.8;
  const octant: LonLatPosition[] = [
    [0, 0],
    [90, 0],
    [0, 90]
  ];
  expect(sphericalPolygonArea(octant, radius) / (4 * Math.PI * radius ** 2)).toBeCloseTo(1 / 8, 14);
  expect(sphericalPolygonArea([...octant].reverse(), radius)).toBeCloseTo(
    sphericalPolygonArea(octant, radius),
    0
  );
});
