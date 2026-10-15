// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ringSymmetricDifference} from '../../oracles/parity-wp4-contour-compare';
import {
  compareValues,
  EXACT_TOLERANCE,
  CONTOUR_REFERENCE,
  type ContourCase,
  type ContourLevel
} from '../parity-wp5-round6-fixtures';

/** Decode global grid-edge IDs, which identify topology independently of vertex rounding.
 * Fixtures keep crossing vertices away from corners; the fixed edge coordinate is exact in f32.
 */
export function decodeSegments(coordinates: number[], fixture: ContourCase): number[][] {
  const pixelOffset = fixture.point ? 0 : 0.5;
  function edgeId(horizontal: number, vertical: number): number {
    horizontal -= pixelOffset;
    vertical -= pixelOffset;
    if (Number.isInteger(vertical)) return vertical * (fixture.width - 1) + Math.floor(horizontal);
    if (Number.isInteger(horizontal))
      return (
        fixture.height * (fixture.width - 1) + Math.floor(vertical) * fixture.width + horizontal
      );
    return -1;
  }
  const segments: number[][] = [];
  for (let index = 0; index < coordinates.length; index += 4) {
    const first = coordinates.slice(index, index + 2);
    const second = coordinates.slice(index + 2, index + 4);
    const firstId = edgeId(first[0], first[1]);
    const secondId = edgeId(second[0], second[1]);
    segments.push(
      firstId < secondId
        ? [firstId, secondId, ...first, ...second]
        : [secondId, firstId, ...second, ...first]
    );
  }
  return segments.sort((first, second) => first[0] - second[0] || first[1] - second[1]);
}

function countClosedComponents(segments: number[][]): number {
  const adjacency = new Map<number, number[]>();
  for (const [first, second] of segments) {
    adjacency.set(first, [...(adjacency.get(first) ?? []), second]);
    adjacency.set(second, [...(adjacency.get(second) ?? []), first]);
  }
  let closed = 0;
  const visited = new Set<number>();
  for (const start of adjacency.keys()) {
    if (visited.has(start)) continue;
    const pending = [start];
    let allDegreeTwo = true;
    while (pending.length) {
      const vertex = pending.pop()!;
      if (visited.has(vertex)) continue;
      visited.add(vertex);
      const neighbors = adjacency.get(vertex)!;
      allDegreeTwo &&= neighbors.length === 2;
      pending.push(...neighbors.filter(neighbor => !visited.has(neighbor)));
    }
    if (allDegreeTwo) closed++;
  }
  return closed;
}

/** Every retained segment must belong to the full reference, including its coordinates. */
export function compareBoundedContours(
  fixture: ContourCase,
  level: ContourLevel,
  coordinates: number[]
): void {
  const actual = decodeSegments(coordinates, fixture);
  const matches = actual.map(segment =>
    level.segments.find(reference => reference[0] === segment[0] && reference[1] === segment[1])
  );
  compareValues(
    'GPUTerrainContours',
    fixture.name,
    CONTOUR_REFERENCE,
    `level-${level.level}/bounded-topology-membership`,
    matches.map(match => Number(Boolean(match))),
    actual.map(() => 1),
    EXACT_TOLERANCE
  );
  compareValues(
    'GPUTerrainContours',
    fixture.name,
    CONTOUR_REFERENCE,
    `level-${level.level}/bounded-all-endpoints`,
    actual.flatMap(segment => segment.slice(2)),
    matches.flatMap(match => (match ? match.slice(2) : [NaN, NaN, NaN, NaN])),
    fixture.tolerance
  );
}

export function compareContourSegments(
  fixture: ContourCase,
  level: ContourLevel,
  coordinates: number[]
): void {
  const actual = decodeSegments(coordinates, fixture);
  const expected = level.segments;
  const compare = (
    metric: string,
    received: number[],
    wanted: number[],
    tolerance = EXACT_TOLERANCE
  ) =>
    compareValues(
      'GPUTerrainContours',
      `${fixture.name}/level-${level.level}`,
      CONTOUR_REFERENCE,
      metric,
      received,
      wanted,
      tolerance
    );
  compare(
    'all-segment-edge-pairs',
    actual.flatMap(segment => segment.slice(0, 2)),
    expected.flatMap(segment => segment.slice(0, 2))
  );
  compare(
    'all-endpoint-coordinates',
    actual.flatMap(segment => segment.slice(2)),
    expected.flatMap(segment => segment.slice(2)),
    fixture.tolerance
  );
  compare('closed-ring-count', [countClosedComponents(actual)], [level.rings.length]);
  // Linear interpolation of corresponding endpoints bounds every interior point of
  // each matched straight segment. This is a symmetric Hausdorff upper bound over
  // the COMPLETE line sets, not sparse sampling. No extra tolerance is fitted.
  const distances = actual.map((segment, index) => {
    const reference = expected[index];
    if (!reference) return Infinity;
    return Math.max(
      Math.hypot(segment[2] - reference[2], segment[3] - reference[3]),
      Math.hypot(segment[4] - reference[4], segment[5] - reference[5])
    );
  });
  compare(
    'segmentwise-symmetric-Hausdorff-upper-bound',
    distances,
    expected.map(() => 0),
    {...fixture.tolerance, absolute: Math.SQRT2 * fixture.tolerance.absolute}
  );
  const referenceVertices = new Map<string, number>();
  const actualVertices = new Map<number, number[]>();
  const seenReference = new Set<number>();
  for (const segment of expected)
    for (const endpoint of [0, 1]) {
      if (seenReference.has(segment[endpoint])) continue;
      seenReference.add(segment[endpoint]);
      referenceVertices.set(
        segment.slice(2 + endpoint * 2, 4 + endpoint * 2).join('/'),
        segment[endpoint]
      );
    }
  for (const segment of actual)
    for (const endpoint of [0, 1])
      actualVertices.set(segment[endpoint], segment.slice(2 + endpoint * 2, 4 + endpoint * 2));
  for (const [ringIndex, ring] of level.rings.entries()) {
    const received = ring.map(
      vertex => actualVertices.get(referenceVertices.get(vertex.join('/')) ?? -1) ?? [NaN, NaN]
    );
    const perimeter = ring
      .slice(1)
      .reduce(
        (sum, vertex, index) =>
          sum + Math.hypot(vertex[0] - ring[index][0], vertex[1] - ring[index][1]),
        0
      );
    const distance = Math.SQRT2 * fixture.tolerance.absolute;
    // Corresponding polygon boundaries lie within distance d. XOR lies in the
    // boundary tube: 2*perimeter*d + edgeCount*pi*d². The final term bounds f64
    // sweep arithmetic at coordinate scale. This derives AREA units from G.
    const areaBound =
      2 * perimeter * distance +
      ring.length * Math.PI * distance ** 2 +
      64 * Number.EPSILON * ring.length * Math.max(fixture.width, fixture.height) ** 2;
    const area = received.flat().every(Number.isFinite)
      ? ringSymmetricDifference(received.flat(), ring.flat())
      : Infinity;
    compare(`ring-${ringIndex}/symmetric-difference-area`, [area], [0], {
      ...fixture.tolerance,
      absolute: areaBound
    });
  }
}
