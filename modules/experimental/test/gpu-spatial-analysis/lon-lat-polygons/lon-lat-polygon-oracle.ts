// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUSpatialEdgeModel} from '../../../src/analysis-core/contracts';
import {clipRing, type Point} from '../rectangle-clip/rectangle-clip-oracle';

/** Binary64 longitude/latitude position, in degrees. */
export type LonLatPosition = Point;
/** Features contain polygons, which contain shell-first rings. */
export type LonLatFeatures = Point[][][][];
/** Four-level GeoArrow polygon layout used by the fixtures and preparation reference. */
export type LonLatPolygonData = {
  positions: Float64Array;
  featureOffsets: Uint32Array;
  polygonOffsets: Uint32Array;
  ringOffsets: Uint32Array;
};
/** Test-only preparation controls, including an optional bounded output capacity. */
export type LonLatPreparationOptions = {
  edgeModel?: GPUSpatialEdgeModel;
  maximumSegmentDegrees?: number;
  maximumPieceCount?: number;
  antimeridian?: 'split' | 'reject';
  poleRings?: 'cap' | 'reject';
  positionCapacity?: number;
};
/** Full fixed-topology preparation result and readback status. */
export type PreparedLonLatPolygons = LonLatPolygonData & {
  ringFlags: Uint32Array;
  status: {
    count: number;
    overflow: number;
    requiredCount: number;
    invalidCount: number;
    approximation: number;
  };
};
/** Unwrapped reference coordinates plus the lossless representation used by GPU preparation. */
export type DensifiedLonLatRing = {
  unwrapped: Point[];
  vertices: {
    wrapped: Point;
    turn: number;
    /** Original open-ring edge, for deriving interior interpolation error bounds. */
    sourceRow: number;
    /** Zero identifies a kept input vertex. */
    fraction: number;
  }[];
  clamped: boolean;
};

const RADIANS_PER_DEGREE = Math.PI / 180;

function roundEven(value: number): number {
  const lower = Math.floor(value);
  return value - lower === 0.5 ? (lower % 2 === 0 ? lower : lower + 1) : Math.round(value);
}

/** Same short-way arithmetic as WGSL round (ties to even), using binary64 numbers. */
export function wrapLongitudeStep(rawStep: number): number {
  return rawStep - 360 * roundEven(rawStep / 360);
}

/** A degree-plane pole edge has coincident spherical endpoints and retains its raw step. */
export function isPoleEdge(first: Point, second: Point): boolean {
  return (
    Math.abs(first[1]) >= 90 - 1e-6 && Math.abs(second[1]) >= 90 - 1e-6 && first[1] * second[1] > 0
  );
}

/** Walks non-pole edges the short way and pole edges along the raw degree-plane pole line. */
export function ringStep(first: Point, second: Point): number {
  return isPoleEdge(first, second) ? second[0] - first[0] : wrapLongitudeStep(second[0] - first[0]);
}

/** Whether an edge needs an antimeridian split. Pole edges are already capped. */
export function isCrossingEdge(first: Point, second: Point): boolean {
  return !isPoleEdge(first, second) && Math.abs(second[0] - first[0]) > 180;
}

/** Linear edges within 1e-4 degrees of a half-turn have no unique short-way interpretation. */
export function isAmbiguousLinearEdge(first: Point, second: Point): boolean {
  return (
    !isPoleEdge(first, second) && Math.abs(wrapLongitudeStep(second[0] - first[0])) >= 180 - 1e-4
  );
}

type Vector = [number, number, number];
function unitVector(position: Point): Vector {
  const longitude = position[0] * RADIANS_PER_DEGREE;
  const latitude = position[1] * RADIANS_PER_DEGREE;
  const cosine = Math.sin((90 - Math.abs(position[1])) * RADIANS_PER_DEGREE);
  return [cosine * Math.cos(longitude), cosine * Math.sin(longitude), Math.sin(latitude)];
}
function dot(first: Vector, second: Vector): number {
  return first[0] * second[0] + first[1] * second[1] + first[2] * second[2];
}
function cross(first: Vector, second: Vector): Vector {
  return [
    first[1] * second[2] - first[2] * second[1],
    first[2] * second[0] - first[0] * second[2],
    first[0] * second[1] - first[1] * second[0]
  ];
}
function angle(first: Vector, second: Vector): number {
  return Math.atan2(Math.hypot(...cross(first, second)), dot(first, second));
}

/** V1 rejects near-antipodal endpoints (1e-6 rad) and minor arcs within 1e-6 degrees of a pole. */
export function isAmbiguousGreatCircleEdge(firstPosition: Point, secondPosition: Point): boolean {
  if (isPoleEdge(firstPosition, secondPosition)) {
    return false;
  }
  const first = unitVector(firstPosition);
  const second = unitVector(secondPosition);
  if (Math.hypot(...first.map((value, index) => value + second[index])) <= 1e-6) {
    return true;
  }
  if (Math.abs(firstPosition[1]) >= 90 - 1e-6 || Math.abs(secondPosition[1]) >= 90 - 1e-6) {
    return true;
  }
  const normal = cross(first, second);
  const normalLength = Math.hypot(...normal);
  if (!normalLength || Math.abs(normal[2] / normalLength) > Math.sin(1e-6 * RADIANS_PER_DEGREE)) {
    return false;
  }
  const unitNormal = normal.map(value => value / normalLength);
  for (const hemisphere of [-1, 1]) {
    const pole: Vector = [
      -unitNormal[2] * unitNormal[0] * hemisphere,
      -unitNormal[2] * unitNormal[1] * hemisphere,
      (1 - unitNormal[2] ** 2) * hemisphere
    ];
    if (dot(cross(first, pole), normal) >= 0 && dot(cross(pole, second), normal) >= 0) {
      return true;
    }
  }
  return false;
}

function openRing(ring: readonly Point[]): Point[] {
  const first = ring[0];
  const last = ring[ring.length - 1];
  return ring.length > 1 && first[0] === last[0] && first[1] === last[1]
    ? ring.slice(0, -1)
    : [...ring];
}
function hasThreePositions(ring: readonly Point[]): boolean {
  const first = ring[0];
  const second = ring.find(position => position[0] !== first[0] || position[1] !== first[1]);
  return Boolean(
    second &&
      ring.some(
        position =>
          (position[0] !== first[0] || position[1] !== first[1]) &&
          (position[0] !== second[0] || position[1] !== second[1])
      )
  );
}

/** Longitude winding including the implicit closing edge; degenerate rings contribute zero. */
export function ringWinding(input: readonly Point[]): number {
  const ring = openRing(input);
  return hasThreePositions(ring)
    ? ring.reduce(
        (total, position, index) => total + ringStep(position, ring[(index + 1) % ring.length]),
        0
      )
    : 0;
}

/** Classifies one closed ring, including the closing endpoint in its unwrapped width. */
export function classifyRing(input: readonly Point[], edgeModel: GPUSpatialEdgeModel = 'linear') {
  const ring = openRing(input);
  const invalidValue = ring.some(
    position =>
      !Number.isFinite(position[0]) || !Number.isFinite(position[1]) || Math.abs(position[1]) > 90
  );
  let crossing = false;
  let ambiguous = false;
  let longitude = ring[0]?.[0] ?? 0;
  let minimum = longitude;
  let maximum = longitude;
  if (hasThreePositions(ring)) {
    for (let index = 0; index < ring.length; index++) {
      const first = ring[index];
      const second = ring[(index + 1) % ring.length];
      crossing ||= isCrossingEdge(first, second);
      ambiguous ||=
        edgeModel === 'linear'
          ? isAmbiguousLinearEdge(first, second)
          : isAmbiguousGreatCircleEdge(first, second);
      longitude += ringStep(first, second);
      minimum = Math.min(minimum, longitude);
      maximum = Math.max(maximum, longitude);
    }
  }
  return {
    crossing,
    poleRing: Math.abs(ringWinding(ring)) > 180,
    ambiguous,
    widerThan360: maximum - minimum > 360,
    invalidValue
  };
}

/** Exact binary64 minor-arc interpolation, used independently by the deviation tests. */
export function interpolateGreatCircle(first: Point, second: Point, fraction: number): Point {
  if (fraction === 0) {
    return [...first];
  }
  if (fraction === 1) {
    return [...second];
  }
  const firstVector = unitVector(first);
  const secondVector = unitVector(second);
  const centralAngle = angle(firstVector, secondVector);
  if (centralAngle < 1e-15) {
    return [...first];
  }
  const firstWeight = Math.sin((1 - fraction) * centralAngle) / Math.sin(centralAngle);
  const secondWeight = Math.sin(fraction * centralAngle) / Math.sin(centralAngle);
  const vector = firstVector.map(
    (value, index) => value * firstWeight + secondVector[index] * secondWeight
  );
  return [
    Math.atan2(vector[1], vector[0]) / RADIANS_PER_DEGREE,
    Math.atan2(vector[2], Math.hypot(vector[0], vector[1])) / RADIANS_PER_DEGREE
  ];
}

/** Densifies every edge including closure, emitting each edge's start but never a repeated end. */
export function densifyRing(
  input: readonly Point[],
  edgeModel: GPUSpatialEdgeModel = 'linear',
  maximumSegmentDegrees = edgeModel === 'linear' ? 0 : 0.1,
  maximumPieceCount = 1024
): DensifiedLonLatRing {
  const ring = openRing(input);
  const unwrapped: Point[] = [];
  const vertices: DensifiedLonLatRing['vertices'] = [];
  let clamped = false;
  let longitude = ring[0]?.[0] ?? 0;
  for (let index = 0; index < ring.length; index++) {
    const first = ring[index];
    const second = ring[(index + 1) % ring.length];
    const step = ringStep(first, second);
    const centralDegrees = angle(unitVector(first), unitVector(second)) / RADIANS_PER_DEGREE;
    const requested =
      maximumSegmentDegrees > 0
        ? Math.max(
            1,
            Math.ceil(
              Math.max(Math.abs(wrapLongitudeStep(second[0] - first[0])), centralDegrees) /
                maximumSegmentDegrees
            )
          )
        : 1;
    const pieces = Math.min(requested, maximumPieceCount);
    clamped ||= requested > maximumPieceCount;
    for (let piece = 0; piece < pieces; piece++) {
      const fraction = piece / pieces;
      let position: Point;
      let wrapped: Point;
      if (piece === 0) {
        position = [longitude, first[1]];
        wrapped = [...first];
      } else if (
        edgeModel === 'great-circle' &&
        maximumSegmentDegrees > 0 &&
        !isPoleEdge(first, second)
      ) {
        wrapped = interpolateGreatCircle(first, second, fraction);
        position = [
          wrapped[0] + 360 * Math.round((longitude + step * fraction - wrapped[0]) / 360),
          wrapped[1]
        ];
      } else {
        position = [longitude + step * fraction, first[1] + (second[1] - first[1]) * fraction];
        wrapped = [wrapLongitudeStep(position[0]), position[1]];
      }
      // Preserve endpoints outside the canonical interval only as whole turns in test metadata.
      if (Math.abs(wrapped[0]) > 180) {
        wrapped[0] = wrapLongitudeStep(wrapped[0]);
      }
      unwrapped.push(position);
      vertices.push({
        wrapped,
        turn: Math.round((position[0] - wrapped[0]) / 360),
        sourceRow: index,
        fraction
      });
    }
    longitude += step;
  }
  return {unwrapped, vertices, clamped};
}

/**
 * Closes a pole ring through the nearer pole; rejects a zero mid-latitude.
 * Includes the unwrapped closing endpoint before the two pole vertices, so the last original
 * edge survives and both pole meridians coincide on the sphere. The longitude span is 360.
 */
export function capPoleRing(unwrapped: readonly Point[]): Point[] {
  if (!unwrapped.length) {
    return [];
  }
  const latitudes = unwrapped.map(position => position[1]);
  const hemisphere = Math.sign((Math.min(...latitudes) + Math.max(...latitudes)) / 2);
  if (!hemisphere) {
    throw new Error('ambiguous pole');
  }
  const first = unwrapped[0];
  const last = unwrapped[unwrapped.length - 1];
  const closureLongitude = last[0] + wrapLongitudeStep(first[0] - last[0]);
  return [
    ...unwrapped.map(position => [...position] as Point),
    [closureLongitude, first[1]],
    [closureLongitude, hemisphere * 90],
    [first[0], hemisphere * 90]
  ];
}

/** Shifts the shell minimum to [-180,180) and aligns each hole to the shell's bbox centre. */
export function normalizePolygon(polygon: readonly Point[][]): Point[][] {
  const shell = polygon[0] ?? [];
  if (!shell.length) {
    return polygon.map(() => []);
  }
  const minimum = Math.min(...shell.map(position => position[0]));
  const maximum = Math.max(...shell.map(position => position[0]));
  const shellTurns = Math.floor((minimum + 180) / 360);
  const centre = (minimum + maximum) / 2;
  return polygon.map((ring, index) => {
    const turns =
      index === 0 || !ring.length
        ? -shellTurns
        : Math.round((centre - ring[0][0]) / 360) - shellTurns;
    return ring.map(position => [position[0] + 360 * turns, position[1]]);
  });
}

/** Clips normalized rings into the main and east copies, keeping empty ring roles. */
export function splitPolygonAtAntimeridian(polygon: readonly Point[][]): [Point[][], Point[][]] {
  const clipCopy = (copy: number) =>
    polygon.map(ring => {
      const clipped = clipRing(
        ring.map(position => [position[0] - copy * 360, position[1]]),
        {minX: -180, minY: -90, maxX: 180, maxY: 90}
      );
      const result = openRing(clipped).filter(
        (position, index, positions) =>
          index === 0 ||
          position[0] !== positions[index - 1][0] ||
          position[1] !== positions[index - 1][1]
      );
      return hasThreePositions(result) ? result : [];
    });
  return [clipCopy(0), clipCopy(1)];
}

/** Flattens shell-first nested features to the geometry-port offset layout. */
export function flattenLonLatFeatures(features: LonLatFeatures): LonLatPolygonData {
  const positions: number[] = [];
  const featureOffsets = [0];
  const polygonOffsets = [0];
  const ringOffsets = [0];
  for (const feature of features) {
    for (const polygon of feature) {
      for (const ring of polygon) {
        for (const position of ring) {
          positions.push(...position);
        }
        ringOffsets.push(positions.length / 2);
      }
      polygonOffsets.push(ringOffsets.length - 1);
    }
    featureOffsets.push(polygonOffsets.length - 1);
  }
  return {
    positions: new Float64Array(positions),
    featureOffsets: new Uint32Array(featureOffsets),
    polygonOffsets: new Uint32Array(polygonOffsets),
    ringOffsets: new Uint32Array(ringOffsets)
  };
}

/** Reads fixture or oracle output into shell-first nested features (test-only). */
export function getLonLatFeatures(data: LonLatPolygonData): LonLatFeatures {
  return Array.from({length: data.featureOffsets.length - 1}, (_, feature) =>
    Array.from(
      {length: data.featureOffsets[feature + 1] - data.featureOffsets[feature]},
      (_, polygonIndex) => {
        const polygon = data.featureOffsets[feature] + polygonIndex;
        return Array.from(
          {length: data.polygonOffsets[polygon + 1] - data.polygonOffsets[polygon]},
          (_, ringIndex) => {
            const ring = data.polygonOffsets[polygon] + ringIndex;
            return Array.from(
              {length: data.ringOffsets[ring + 1] - data.ringOffsets[ring]},
              (_, vertexIndex): Point => {
                const vertex = data.ringOffsets[ring] + vertexIndex;
                return [data.positions[vertex * 2], data.positions[vertex * 2 + 1]];
              }
            );
          }
        );
      }
    )
  );
}

/** Full binary64 preparation reference: classify, densify, cap, normalize, clip and bound output. */
export function prepareLonLatPolygons(
  input: LonLatFeatures | LonLatPolygonData,
  options: LonLatPreparationOptions = {}
): PreparedLonLatPolygons {
  const features = Array.isArray(input) ? input : getLonLatFeatures(input);
  const edgeModel = options.edgeModel ?? 'linear';
  const maximumSegmentDegrees = options.maximumSegmentDegrees ?? (edgeModel === 'linear' ? 0 : 0.1);
  const ringFlags: number[] = [];
  const output: LonLatFeatures = [];
  let invalidCount = 0;
  let approximation = 0;
  for (const feature of features) {
    let invalid = false;
    const copies = feature.map(polygon => {
      const rings = polygon.map(ring => {
        const classification = classifyRing(ring, edgeModel);
        const densified = classification.invalidValue
          ? {unwrapped: [], clamped: false}
          : densifyRing(ring, edgeModel, maximumSegmentDegrees, options.maximumPieceCount ?? 1024);
        const reject =
          classification.ambiguous ||
          classification.widerThan360 ||
          classification.invalidValue ||
          (classification.crossing && options.antimeridian === 'reject') ||
          (classification.poleRing && options.poleRings === 'reject');
        invalid ||= reject;
        let capped = false;
        let positions = densified.unwrapped;
        if (classification.poleRing && !reject) {
          try {
            positions = capPoleRing(positions);
            capped = true;
          } catch {
            invalid = true;
          }
        }
        ringFlags.push(
          Number(classification.crossing) |
            (Number(capped) << 1) |
            (Number(classification.ambiguous) << 2) |
            (Number(classification.widerThan360) << 3) |
            (Number(classification.invalidValue) << 4) |
            (Number(densified.clamped) << 5)
        );
        if (edgeModel === 'great-circle' && (maximumSegmentDegrees === 0 || densified.clamped)) {
          approximation = 1;
        }
        return hasThreePositions(ring) ? positions : [];
      });
      return splitPolygonAtAntimeridian(normalizePolygon(rings));
    });
    if (invalid) {
      invalidCount++;
    }
    output.push(
      copies
        .map(copy => copy[0])
        .concat(copies.map(copy => copy[1]))
        .map(polygon => polygon.map(ring => (invalid ? [] : ring)))
    );
  }
  const flat = flattenLonLatFeatures(output);
  const requiredCount = flat.positions.length / 2;
  const count = Math.min(requiredCount, options.positionCapacity ?? requiredCount);
  return {
    ...flat,
    positions: flat.positions.slice(0, count * 2),
    ringOffsets: flat.ringOffsets.map(offset => Math.min(offset, count)),
    ringFlags: new Uint32Array(ringFlags),
    status: {
      count,
      requiredCount,
      overflow: Number(count < requiredCount),
      invalidCount,
      approximation
    }
  };
}

/** Half-open even-odd containment; returns true if any feature contains the point. */
export function pointInLonLatPolygonPrepared(point: Point, prepared: LonLatPolygonData): boolean {
  return getLonLatFeatures(prepared).some(feature =>
    feature.some(polygon => {
      let inside = false;
      for (const ring of polygon) {
        for (let index = 0; index < ring.length; index++) {
          const first = ring[index];
          const second = ring[(index + 1) % ring.length];
          if (
            first[1] > point[1] !== second[1] > point[1] &&
            point[0] <
              first[0] + ((point[1] - first[1]) * (second[0] - first[0])) / (second[1] - first[1])
          ) {
            inside = !inside;
          }
        }
      }
      return inside;
    })
  );
}

/** Great-circle spherical excess, triangulated with signed solid angles and returned unsigned. */
export function sphericalPolygonArea(input: readonly Point[], radius: number): number {
  const ring = openRing(input);
  if (ring.length < 3) {
    return 0;
  }
  const first = unitVector(ring[0]);
  let excess = 0;
  for (let index = 1; index + 1 < ring.length; index++) {
    const second = unitVector(ring[index]);
    const third = unitVector(ring[index + 1]);
    excess +=
      2 *
      Math.atan2(
        dot(first, cross(second, third)),
        1 + dot(first, second) + dot(second, third) + dot(third, first)
      );
  }
  return Math.abs(excess) * radius * radius;
}

/** Linear-edge Chamberlain-Duquette reference, using the shared pole-aware longitude steps. */
export function chamberlainDuquetteArea(input: readonly Point[], radius: number): number {
  const ring = openRing(input);
  let sum = 0;
  for (let index = 0; index < ring.length; index++) {
    const first = ring[index];
    const second = ring[(index + 1) % ring.length];
    sum +=
      ringStep(first, second) *
      RADIANS_PER_DEGREE *
      (Math.sin(first[1] * RADIANS_PER_DEGREE) + Math.sin(second[1] * RADIANS_PER_DEGREE));
  }
  return (Math.abs(sum) * radius * radius) / 2;
}
