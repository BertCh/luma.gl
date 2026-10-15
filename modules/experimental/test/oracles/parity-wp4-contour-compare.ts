// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {compareRoundFour} from './parity-wp4-round4-compare';
import type {ReferenceRing} from './parity-wp4-round5-fixtures';

type Point = [number, number];
type Edge = [Point, Point];
export type ActualRing = {
  positions: number[];
  area: number;
  band: number;
  isHole: number;
  shell: number;
};

/** G permits 4–16 ULPs; eight cover interpolation, centre offset and storage rounding. */
export function contourBound(width: number, height: number): number {
  return 8 * 2 ** (Math.floor(Math.log2(Math.max(width, height))) - 23);
}

export function compareLineSegments(
  name: string,
  reference: string,
  width: number,
  height: number,
  coordinates: number[],
  levels: number[],
  edgeIds: number[],
  expected: number[]
) {
  const remaining = new Map<string, number[]>();
  for (let index = 0; index < expected.length; index += 8) {
    remaining.set(
      `${expected[index + 1]}/${expected[index + 2]}/${expected[index + 3]}`,
      expected.slice(index + 4, index + 8)
    );
  }
  const receivedCoordinates: number[] = [];
  const expectedCoordinates: number[] = [];
  let unmatched = 0;
  for (let index = 0; index < levels.length; index++) {
    const key = `${levels[index]}/${edgeIds[index * 2]}/${edgeIds[index * 2 + 1]}`;
    const target = remaining.get(key);
    if (!target) {
      unmatched++;
      continue;
    }
    remaining.delete(key);
    receivedCoordinates.push(...coordinates.slice(index * 4, index * 4 + 4));
    expectedCoordinates.push(...target);
  }
  compareRoundFour(
    'GPUIsolines',
    name,
    reference,
    'complete-directed-level-edge-pairs',
    [levels.length, unmatched, remaining.size, coordinates.length, edgeIds.length],
    [expected.length / 8, 0, 0, levels.length * 4, levels.length * 2],
    'E'
  );
  // Endpoint bijection bounds the entire straight-segment Hausdorff distance
  // by sqrt(2)*bound, not just a sample of points along the contour.
  compareRoundFour(
    'GPUIsolines',
    name,
    reference,
    'every-directed-segment-endpoint',
    receivedCoordinates,
    expectedCoordinates,
    'G',
    contourBound(width, height)
  );
}

function points(positions: number[]): Point[] {
  const result: Point[] = [];
  for (let index = 0; index < positions.length; index += 2)
    result.push([positions[index], positions[index + 1]]);
  return result;
}
function edges(positions: Point[]): Edge[] {
  return positions.slice(1).map((point, index) => [positions[index], point]);
}
function cross(first: Point, second: Point): number {
  return first[0] * second[1] - first[1] * second[0];
}
function subtract(first: Point, second: Point): Point {
  return [first[0] - second[0], first[1] - second[1]];
}
function signedArea(positions: Point[]): number {
  const origin = positions[0];
  return edges(positions).reduce(
    (sum, [first, second]) => sum + cross(subtract(first, origin), subtract(second, origin)) / 2,
    0
  );
}

/** Exact piecewise-linear vertical sweep of XOR interiors, in float64 test arithmetic.
 * Split at every vertex and inter-boundary intersection: vertical overlap lengths
 * are linear inside each open slab, so midpoint integration is exact there.
 * This measures symmetric difference, not just the difference of signed areas.
 */
export function ringSymmetricDifference(first: number[], second: number[]): number {
  const firstEdges = edges(points(first));
  const secondEdges = edges(points(second));
  const breaks = [...points(first), ...points(second)].map(point => point[0]);
  for (const [firstStart, firstEnd] of firstEdges) {
    const firstDirection = subtract(firstEnd, firstStart);
    for (const [secondStart, secondEnd] of secondEdges) {
      if (
        Math.max(firstStart[0], firstEnd[0]) < Math.min(secondStart[0], secondEnd[0]) ||
        Math.max(secondStart[0], secondEnd[0]) < Math.min(firstStart[0], firstEnd[0])
      )
        continue;
      const secondDirection = subtract(secondEnd, secondStart);
      const denominator = cross(firstDirection, secondDirection);
      if (denominator === 0) continue;
      const difference = subtract(secondStart, firstStart);
      const firstFraction = cross(difference, secondDirection) / denominator;
      const secondFraction = cross(difference, firstDirection) / denominator;
      if (firstFraction > 0 && firstFraction < 1 && secondFraction > 0 && secondFraction < 1) {
        breaks.push(firstStart[0] + firstFraction * firstDirection[0]);
      }
    }
  }
  const columns = [...new Set(breaks)].sort((firstValue, secondValue) => firstValue - secondValue);
  let area = 0;
  const allEdges = [...firstEdges, ...secondEdges];
  for (let index = 1; index < columns.length; index++) {
    const middle = (columns[index - 1] + columns[index]) / 2;
    const crossings: number[] = [];
    for (const [start, end] of allEdges) {
      if (middle > Math.min(start[0], end[0]) && middle < Math.max(start[0], end[0])) {
        crossings.push(
          start[1] + ((middle - start[0]) / (end[0] - start[0])) * (end[1] - start[1])
        );
      }
    }
    crossings.sort((firstValue, secondValue) => firstValue - secondValue);
    let length = 0;
    for (let crossing = 0; crossing + 1 < crossings.length; crossing += 2)
      length += crossings[crossing + 1] - crossings[crossing];
    area += (columns[index] - columns[index - 1]) * length;
  }
  return area;
}

export function compareContourRings(
  name: string,
  reference: string,
  width: number,
  height: number,
  actual: ActualRing[],
  expected: ReferenceRing[]
) {
  const bound = contourBound(width, height);
  const compare = (
    metric: string,
    received: number[],
    wanted: number[],
    profile: 'E' | 'G',
    absolute = 0
  ) =>
    compareRoundFour(
      'GPUIsobandRings',
      name,
      reference,
      metric,
      received,
      wanted,
      profile,
      absolute
    );
  compare('ring-count', [actual.length], [expected.length], 'E');
  const mapping: number[] = [];
  const used = new Set<number>();
  const receivedCoordinates: number[] = [];
  const wantedCoordinates: number[] = [];
  for (const ring of actual) {
    const match = expected.findIndex(
      (candidate, candidateIndex) =>
        !used.has(candidateIndex) &&
        candidate.band === ring.band &&
        candidate.isHole === ring.isHole &&
        candidate.positions.length === ring.positions.length &&
        candidate.positions.some(
          (_, index) =>
            index % 2 === 0 &&
            index < candidate.positions.length - 2 &&
            Math.abs(candidate.positions[index] - ring.positions[0]) <= bound &&
            Math.abs(candidate.positions[index + 1] - ring.positions[1]) <= bound
        )
    );
    mapping.push(match);
    if (match < 0) continue;
    used.add(match);
    const target = expected[match].positions;
    const vertexCount = target.length / 2 - 1;
    const start =
      target.findIndex(
        (_, index) =>
          index % 2 === 0 &&
          index < target.length - 2 &&
          Math.abs(target[index] - ring.positions[0]) <= bound &&
          Math.abs(target[index + 1] - ring.positions[1]) <= bound
      ) / 2;
    for (let vertex = 0; vertex <= vertexCount; vertex++) {
      receivedCoordinates.push(ring.positions[vertex * 2], ring.positions[vertex * 2 + 1]);
      const referenceVertex = (start + vertex) % vertexCount;
      wantedCoordinates.push(target[referenceVertex * 2], target[referenceVertex * 2 + 1]);
    }
  }
  compare(
    'bijection-of-band-hole-and-vertex-counts',
    mapping.map(index => Number(index >= 0)),
    expected.map(() => 1),
    'E'
  );
  compare(
    'every-cyclic-directed-boundary-vertex',
    receivedCoordinates,
    wantedCoordinates,
    'G',
    bound
  );
  compare(
    'closed-finite-rings',
    actual.map(ring =>
      Number(
        ring.positions.length >= 8 &&
          ring.positions.every(Number.isFinite) &&
          ring.positions[0] === ring.positions.at(-2) &&
          ring.positions[1] === ring.positions.at(-1)
      )
    ),
    expected.map(() => 1),
    'E'
  );
  compare(
    'every-hole-owning-shell-and-shell-self-id',
    actual.map(ring => mapping[ring.shell] ?? -1),
    mapping.map(index => expected[index].shell),
    'E'
  );
  const areaRatios: number[] = [];
  const reportedAreaRatios: number[] = [];
  for (const [index, ring] of actual.entries()) {
    const target = expected[mapping[index]];
    const targetPoints = points(target.positions);
    const perimeter = edges(targetPoints).reduce(
      (sum, [first, second]) => sum + Math.hypot(...subtract(second, first)),
      0
    );
    const distance = Math.SQRT2 * bound;
    // Corresponding directed edges lie within distance d. XOR lies inside the
    // boundary tube: 2*P*d + edgeCount*pi*d²; f64 sweep adds O(E*scale²*eps).
    const areaBound =
      2 * perimeter * distance +
      targetPoints.length * Math.PI * distance ** 2 +
      64 * Number.EPSILON * targetPoints.length * Math.max(width, height) ** 2;
    areaRatios.push(ringSymmetricDifference(ring.positions, target.positions) / areaBound);
    // ringAreas is f32 shoelace. A conservative sequential gamma(4E) bound
    // covers products/subtractions/adds even though GPU uses a parallel reduction.
    const ringEdges = edges(points(ring.positions));
    const sumAbsoluteProducts =
      ringEdges.reduce(
        (sum, [first, second]) =>
          sum + Math.abs(first[0] * second[1]) + Math.abs(first[1] * second[0]),
        0
      ) / 2;
    const unitRoundoff = 2 ** -24;
    const operations = 4 * ringEdges.length;
    const roundingBound =
      ((operations * unitRoundoff) / (1 - operations * unitRoundoff)) * sumAbsoluteProducts;
    reportedAreaRatios.push(Math.abs(ring.area - target.area) / (areaBound + roundingBound));
    compare(
      `ring-${index}-orientation`,
      [Math.sign(signedArea(points(ring.positions)))],
      [target.isHole ? -1 : 1],
      'E'
    );
  }
  compare(
    'per-ring-symmetric-difference-over-derived-bound',
    areaRatios,
    areaRatios.map(() => 0),
    'G',
    1
  );
  compare(
    'per-ring-reported-area-over-derived-bound',
    reportedAreaRatios,
    reportedAreaRatios.map(() => 0),
    'G',
    1
  );
}
