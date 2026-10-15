// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {compareRoundFour} from './parity-wp4-round4-compare';

type Point = [number, number];
type Triangle = Point[];
type Edge = [Point, Point];

function cross(first: Point, second: Point, third: Point): number {
  return (
    (second[0] - first[0]) * (third[1] - first[1]) - (second[1] - first[1]) * (third[0] - first[0])
  );
}
function area(polygon: Point[]): number {
  let result = 0;
  for (let index = 1; index + 1 < polygon.length; index++)
    result += cross(polygon[0], polygon[index], polygon[index + 1]) / 2;
  return result;
}

/** Sutherland-Hodgman convex intersection, used only for measuring test geometry. */
function intersectionArea(first: Triangle, second: Triangle): number {
  let polygon = first;
  for (let edge = 0; edge < 3 && polygon.length; edge++) {
    const start = second[edge];
    const end = second[(edge + 1) % 3];
    const clipped: Point[] = [];
    for (let index = 0; index < polygon.length; index++) {
      const current = polygon[index];
      const previous = polygon[(index + polygon.length - 1) % polygon.length];
      const currentSide = cross(start, end, current);
      const previousSide = cross(start, end, previous);
      if (currentSide >= 0 !== previousSide >= 0) {
        const fraction = previousSide / (previousSide - currentSide);
        clipped.push([
          previous[0] + fraction * (current[0] - previous[0]),
          previous[1] + fraction * (current[1] - previous[1])
        ]);
      }
      if (currentSide >= 0) clipped.push(current);
    }
    polygon = clipped;
  }
  return Math.max(0, area(polygon));
}

/** Cancel only identical reversed triangle edges; no snap can change the topology. */
function boundary(triangles: Triangle[]): Edge[] {
  const edges = new Map<string, Edge>();
  for (const triangle of triangles) {
    for (let index = 0; index < 3; index++) {
      const start = triangle[index];
      const end = triangle[(index + 1) % 3];
      if (!edges.delete(`${end}/${start}`)) edges.set(`${start}/${end}`, [start, end]);
    }
  }
  return [...edges.values()];
}

export function compareIsobandGeometry(
  name: string,
  reference: string,
  width: number,
  height: number,
  actualCoordinates: number[],
  actualBands: number[],
  referenceTriangles: number[],
  referenceEdges: number[]
) {
  const coordinateBound = 8 * 2 ** (Math.floor(Math.log2(Math.max(width, height))) - 23);
  const expectedGroups = new Map<string, Triangle[]>();
  const expectedEdges = new Map<string, Edge[]>();
  const actualGroups = new Map<string, Triangle[]>();
  for (let index = 0; index < referenceTriangles.length; index += 8) {
    const key = `${referenceTriangles[index]}/${referenceTriangles[index + 1]}`;
    const triangle: Triangle = [
      [referenceTriangles[index + 2], referenceTriangles[index + 3]],
      [referenceTriangles[index + 4], referenceTriangles[index + 5]],
      [referenceTriangles[index + 6], referenceTriangles[index + 7]]
    ];
    expectedGroups.set(key, [...(expectedGroups.get(key) ?? []), triangle]);
  }
  for (let index = 0; index < referenceEdges.length; index += 6) {
    const key = `${referenceEdges[index]}/${referenceEdges[index + 1]}`;
    const edge: Edge = [
      [referenceEdges[index + 2], referenceEdges[index + 3]],
      [referenceEdges[index + 4], referenceEdges[index + 5]]
    ];
    expectedEdges.set(key, [...(expectedEdges.get(key) ?? []), edge]);
  }
  let invalidTriangles = 0;
  let degenerateTriangles = 0;
  for (let index = 0; index < actualBands.length; index++) {
    const coordinates = actualCoordinates.slice(index * 6, index * 6 + 6);
    if (coordinates.length !== 6 || coordinates.some(value => !Number.isFinite(value))) {
      invalidTriangles++;
      continue;
    }
    const triangle: Triangle = [
      [coordinates[0], coordinates[1]],
      [coordinates[2], coordinates[3]],
      [coordinates[4], coordinates[5]]
    ];
    const signedArea = area(triangle);
    if (signedArea < 0) invalidTriangles++;
    if (signedArea === 0) {
      // Exact-break fans can contain repeated vertices. These have no filled area;
      // still require each vertex to occur on an independently expected band boundary.
      degenerateTriangles++;
      for (const point of triangle) {
        const found = [...expectedEdges].some(
          ([key, edges]) =>
            Number(key.split('/')[1]) === actualBands[index] &&
            edges.some(edge =>
              edge.some(
                endpoint =>
                  Math.abs(endpoint[0] - point[0]) <= coordinateBound &&
                  Math.abs(endpoint[1] - point[1]) <= coordinateBound
              )
            )
        );
        if (!found) invalidTriangles++;
      }
      continue;
    }
    const column = Math.floor((triangle[0][0] + triangle[1][0] + triangle[2][0]) / 3 - 0.5);
    const row = Math.floor((triangle[0][1] + triangle[1][1] + triangle[2][1]) / 3 - 0.5);
    const key = `${row * (width - 1) + column}/${actualBands[index]}`;
    actualGroups.set(key, [...(actualGroups.get(key) ?? []), triangle]);
  }
  const compare = (
    metric: string,
    actual: number[],
    expected: number[],
    profile: 'E' | 'G',
    bound = 0
  ) => compareRoundFour('GPUIsobands', name, reference, metric, actual, expected, profile, bound);
  compare(
    'finite-counterclockwise-triangles-and-degenerate-boundary-membership',
    [invalidTriangles],
    [0],
    'E'
  );
  const keys = [...new Set([...expectedGroups.keys(), ...actualGroups.keys()])].sort();
  compare(
    'exact-nonempty-cell-band-set',
    keys.map(key => Number(actualGroups.has(key))),
    keys.map(key => Number(expectedGroups.has(key))),
    'E'
  );
  // G permits 4-16 ULPs at coordinate scale. Eight cover interpolation, centre
  // offset and output rounding. This fixed bound is not fitted to observed errors.
  const coordinateActual: number[] = [];
  const coordinateExpected: number[] = [];
  const missingEdges: number[] = [];
  const symmetricDifferences: number[] = [];
  const overlaps: number[] = [];
  const areaBounds: number[] = [];
  for (const key of keys) {
    const actual = actualGroups.get(key) ?? [];
    const expected = expectedGroups.get(key) ?? [];
    const remaining = [...(expectedEdges.get(key) ?? [])];
    const actualBoundary = boundary(actual);
    let missing = 0;
    for (const edge of actualBoundary) {
      const found = remaining.findIndex(candidate =>
        edge.every((point, endpoint) =>
          point.every(
            (value, axis) => Math.abs(value - candidate[endpoint][axis]) <= coordinateBound
          )
        )
      );
      if (found < 0) {
        missing++;
      } else {
        coordinateActual.push(...edge.flat());
        coordinateExpected.push(...remaining[found].flat());
        remaining.splice(found, 1);
      }
    }
    missingEdges.push(missing + remaining.length);
    const actualArea = actual.reduce((sum, triangle) => sum + area(triangle), 0);
    const expectedArea = expected.reduce((sum, triangle) => sum + area(triangle), 0);
    let intersection = 0;
    for (const first of actual)
      for (const second of expected) intersection += intersectionArea(first, second);
    let overlap = 0;
    for (let first = 0; first < actual.length; first++)
      for (let second = first + 1; second < actual.length; second++)
        overlap += intersectionArea(actual[first], actual[second]);
    overlaps.push(overlap);
    symmetricDifferences.push(Math.max(0, actualArea + expectedArea - 2 * intersection));
    const perimeter = (expectedEdges.get(key) ?? []).reduce(
      (sum, edge) => sum + Math.hypot(edge[1][0] - edge[0][0], edge[1][1] - edge[0][1]),
      0
    );
    // Matched segment endpoints imply Hausdorff <= sqrt(2)*coordinateBound.
    // Symmetric difference lies in the boundary tube, area <= 2*P*d + pi*d²
    // per edge (conservative for disconnected pieces), plus f64 clip roundoff.
    const distanceBound = Math.SQRT2 * coordinateBound;
    areaBounds.push(
      2 * perimeter * distanceBound +
        (expectedEdges.get(key)?.length ?? 0) * Math.PI * distanceBound ** 2 +
        128 * Number.EPSILON * width * height
    );
  }
  compare(
    'every-directed-cell-band-boundary-edge-matched-once',
    missingEdges,
    missingEdges.map(() => 0),
    'E'
  );
  compare(
    'all-boundary-coordinates-imply-Hausdorff-bound',
    coordinateActual,
    coordinateExpected,
    'G',
    coordinateBound
  );
  compare(
    'every-cell-band-symmetric-difference-over-derived-bound',
    symmetricDifferences.map((value, index) => value / areaBounds[index]),
    symmetricDifferences.map(() => 0),
    'G',
    1
  );
  compare(
    'no-overlapping-triangle-interiors',
    overlaps,
    overlaps.map(() => 0),
    'G',
    128 * Number.EPSILON * width * height
  );
  return {degenerateTriangles};
}
