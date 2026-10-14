// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Regenerate from the repository root:
 * node examples/deck/spatial-analysis-explorer/scripts/build-new-york-land.mjs
 *
 * Source: NYC Department of City Planning, Borough Boundaries (nybb, water excluded),
 * https://data.cityofnewyork.us/City-Government/Borough-Boundaries/gthc-hcne
 * The former tqmj-j8zm export is retired (404). The current Socrata GeoJSON export below
 * is the same NYC Open Data source, not a substitute boundary provider.
 * Dataset license: unspecified; NYC Open Data terms apply:
 * https://opendata.cityofnewyork.us/overview/#termsofuse
 * DCP supplies these boundaries for information without a warranty of accuracy or fitness.
 * The generated coordinates retain DCP attribution; this script's MIT license covers code.
 *
 * Preparation only: clip every ring to the deck.gl-data trips POI bbox plus 500 metres,
 * simplify at up to 5 metres in a local equirectangular approximation, round to 1e-5 degrees,
 * and drop polygon parts below 0.5 hectares. Refine intersecting rings below 5 m until no
 * shortcuts cross other edges (including holes/islands). Remove sub-25 m² loops created by
 * rounding two nearby vertices onto the same coordinate. Shells and holes retain their roles.
 * Borough parts are collected into ONE multipolygon feature (no borough attributes).
 * Rings close implicitly; offsets are vertex/ring indexes, including terminal offsets.
 * This small teaching fixture covers only the POI neighbourhood, not all NYC land.
 * The runtime projects it into the POI frame and performs window analysis on the GPU.
 */

import {writeFile} from 'node:fs/promises';

const BOUNDARY_URL = 'https://data.cityofnewyork.us/resource/gthc-hcne.geojson';
const POINTS_URL =
  'https://raw.githubusercontent.com/visgl/deck.gl-data/master/examples/trips/pois.json';
const PADDING_METERS = 500;
const SIMPLIFICATION_METERS = 5;
const MINIMUM_AREA = 5000;

async function fetchJson(url) {
  const response = await fetch(url, {signal: AbortSignal.timeout(60000)});
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response.json();
}

const [boundary, points] = await Promise.all([fetchJson(BOUNDARY_URL), fetchJson(POINTS_URL)]);
if (boundary.type !== 'FeatureCollection' || boundary.features.length !== 5) {
  throw new Error('Expected the five NYC boroughs');
}
const longitudes = points.map(point => point.coordinates[0]);
const latitudes = points.map(point => point.coordinates[1]);
const west = Math.min(...longitudes);
const east = Math.max(...longitudes);
const south = Math.min(...latitudes);
const north = Math.max(...latitudes);
const origin = [(west + east) / 2, (south + north) / 2];
const metersPerLatitudeDegree = (6371008.8 * Math.PI) / 180;
const metersPerLongitudeDegree = metersPerLatitudeDegree * Math.cos((origin[1] * Math.PI) / 180);
const bounds = [
  west - PADDING_METERS / metersPerLongitudeDegree,
  south - PADDING_METERS / metersPerLatitudeDegree,
  east + PADDING_METERS / metersPerLongitudeDegree,
  north + PADDING_METERS / metersPerLatitudeDegree
];
const project = point => [
  (point[0] - origin[0]) * metersPerLongitudeDegree,
  (point[1] - origin[1]) * metersPerLatitudeDegree
];
const equalPoints = (first, second) => first[0] === second[0] && first[1] === second[1];

function cleanRing(ring) {
  const result = ring.filter((point, index) => index === 0 || !equalPoints(point, ring[index - 1]));
  if (result.length > 1 && equalPoints(result[0], result.at(-1))) result.pop();
  return result;
}

/** Sutherland-Hodgman, preserving the ring's shell/hole role. */
function clipRing(ring) {
  let result = cleanRing(ring);
  for (const [axis, boundaryValue, direction] of [
    [0, bounds[0], 1],
    [0, bounds[2], -1],
    [1, bounds[1], 1],
    [1, bounds[3], -1]
  ]) {
    const input = result;
    result = [];
    for (let index = 0; index < input.length; index++) {
      const start = input[(index + input.length - 1) % input.length];
      const end = input[index];
      const startInside = (start[axis] - boundaryValue) * direction >= 0;
      const endInside = (end[axis] - boundaryValue) * direction >= 0;
      if (startInside !== endInside) {
        const fraction = (boundaryValue - start[axis]) / (end[axis] - start[axis]);
        const intersection = [
          start[0] + fraction * (end[0] - start[0]),
          start[1] + fraction * (end[1] - start[1])
        ];
        intersection[axis] = boundaryValue;
        result.push(intersection);
      }
      if (endInside) result.push(end);
    }
    result = cleanRing(result);
  }
  return result;
}

function squaredSegmentDistance(point, start, end) {
  const differenceX = end[0] - start[0];
  const differenceY = end[1] - start[1];
  const squaredLength = differenceX ** 2 + differenceY ** 2;
  const fraction = squaredLength
    ? Math.max(
        0,
        Math.min(
          1,
          ((point[0] - start[0]) * differenceX + (point[1] - start[1]) * differenceY) /
            squaredLength
        )
      )
    : 0;
  return (
    (point[0] - start[0] - fraction * differenceX) ** 2 +
    (point[1] - start[1] - fraction * differenceY) ** 2
  );
}

/** Iterative Douglas-Peucker; split closed rings at the most distant vertex first. */
function simplifyRing(ring, tolerance = SIMPLIFICATION_METERS) {
  if (ring.length < 3) return [];
  const metric = ring.map(project);
  let split = 1;
  for (let index = 2; index < ring.length; index++) {
    if (
      squaredSegmentDistance(metric[index], metric[0], metric[0]) >
      squaredSegmentDistance(metric[split], metric[0], metric[0])
    )
      split = index;
  }
  const keep = new Set([0, split]);
  const segments = [
    [0, split],
    [split, ring.length]
  ];
  metric.push(metric[0]);
  while (segments.length) {
    const [first, last] = segments.pop();
    let largest = tolerance ** 2;
    let furthest = -1;
    for (let index = first + 1; index < last; index++) {
      const distance = squaredSegmentDistance(metric[index], metric[first], metric[last]);
      if (distance > largest) {
        largest = distance;
        furthest = index;
      }
    }
    if (furthest >= 0) {
      keep.add(furthest);
      segments.push([first, furthest], [furthest, last]);
    }
  }
  return removeRoundingLoops(
    cleanRing(
      ring
        .filter((_, index) => keep.has(index))
        .map(point => point.map(value => Math.round(value * 1e5) / 1e5))
    )
  );
}

/** Remove tiny self-touching loops caused by the required coordinate rounding. */
function removeRoundingLoops(ring) {
  const result = [...ring];
  for (let first = 0; first < result.length; first++) {
    for (let last = first + 1; last < result.length; last++) {
      if (!equalPoints(result[first], result[last])) continue;
      if (ringArea(result.slice(first, last)) > SIMPLIFICATION_METERS ** 2) {
        throw new Error('Rounding creates a large self-touching loop; review source geometry');
      }
      result.splice(first + 1, last - first);
      last = first;
    }
  }
  return result;
}

/** Reject crossing shortcuts; retry only the affected ring at a finer tolerance. */
function prepareRing(ring, maximumTolerance = SIMPLIFICATION_METERS) {
  for (const tolerance of [maximumTolerance, maximumTolerance / 2, maximumTolerance / 4, 0]) {
    const simplified = simplifyRing(ring, tolerance);
    if (!hasCrossings(simplified)) return simplified;
  }
  throw new Error('Rounding introduces a crossing; review source geometry');
}

function segmentsCross(start, end, other, otherEnd) {
  const orientation = (first, last, point) =>
    (last[0] - first[0]) * (point[1] - first[1]) - (last[1] - first[1]) * (point[0] - first[0]);
  return (
    orientation(start, end, other) * orientation(start, end, otherEnd) < -1e-12 &&
    orientation(other, otherEnd, start) * orientation(other, otherEnd, end) < -1e-12
  );
}

function hasCrossings(ring) {
  const metric = ring.map(project);
  for (let first = 0; first < metric.length; first++) {
    const start = metric[first];
    const end = metric[(first + 1) % metric.length];
    for (let second = first + 2; second < metric.length; second++) {
      if (first === 0 && second === metric.length - 1) continue;
      const other = metric[second];
      const otherEnd = metric[(second + 1) % metric.length];
      if (segmentsCross(start, end, other, otherEnd)) return true;
    }
  }
  return false;
}

function ringArea(ring) {
  const metric = ring.map(project);
  return (
    Math.abs(
      metric.reduce((sum, point, index) => {
        const next = metric[(index + 1) % metric.length];
        return sum + point[0] * next[1] - point[1] * next[0];
      }, 0)
    ) / 2
  );
}

const preparedPolygons = [];
for (const feature of boundary.features) {
  if (feature.geometry.type !== 'MultiPolygon') throw new Error('Expected borough multipolygons');
  for (const polygon of feature.geometry.coordinates) {
    const clipped = polygon.map(clipRing);
    if (clipped[0].length < 3) continue;
    const source = [
      clipped[0],
      ...clipped.slice(1).filter(ring => ring.length >= 3 && ringArea(ring) > 0)
    ];
    const area =
      ringArea(source[0]) - source.slice(1).reduce((sum, ring) => sum + ringArea(ring), 0);
    if (area < MINIMUM_AREA) continue;
    preparedPolygons.push(
      source.map(ring => ({
        source: ring,
        ring: prepareRing(ring),
        tolerance: SIMPLIFICATION_METERS
      }))
    );
  }
}

/** Test crossings between rings too: independent shortcuts must not cut a hole or an island. */
function ringsCross(first, second) {
  for (let firstIndex = 0; firstIndex < first.length; firstIndex++) {
    const start = first[firstIndex];
    const end = first[(firstIndex + 1) % first.length];
    for (let secondIndex = 0; secondIndex < second.length; secondIndex++) {
      const other = second[secondIndex];
      const otherEnd = second[(secondIndex + 1) % second.length];
      if (segmentsCross(start, end, other, otherEnd)) return true;
    }
  }
  return false;
}

const preparedRings = preparedPolygons.flat();
while (true) {
  const refine = new Set();
  const metricRings = preparedRings.map(record => record.ring.map(project));
  for (let first = 0; first < metricRings.length; first++) {
    for (let second = first + 1; second < metricRings.length; second++) {
      if (ringsCross(metricRings[first], metricRings[second])) {
        refine.add(first);
        refine.add(second);
      }
    }
  }
  if (!refine.size) break;
  for (const index of refine) {
    const record = preparedRings[index];
    if (record.tolerance === 0) throw new Error('Rounded rings cross; review source geometry');
    record.tolerance = record.tolerance > 0.625 ? record.tolerance / 2 : 0;
    record.ring = prepareRing(record.source, record.tolerance);
  }
}
const polygons = preparedPolygons
  .map(polygon => polygon.map(record => record.ring))
  .filter(
    polygon =>
      polygon[0].length >= 3 &&
      ringArea(polygon[0]) - polygon.slice(1).reduce((sum, ring) => sum + ringArea(ring), 0) >=
        MINIMUM_AREA
  );
const lngLat = [];
const ringOffsets = [0];
const polygonOffsets = [0];
for (const polygon of polygons) {
  for (const ring of polygon) {
    for (const point of ring) lngLat.push(...point);
    ringOffsets.push(lngLat.length / 2);
  }
  polygonOffsets.push(ringOffsets.length - 1);
}
const area = polygons.reduce(
  (sum, polygon) =>
    sum +
    ringArea(polygon[0]) -
    polygon.slice(1).reduce((holes, ring) => holes + ringArea(ring), 0),
  0
);
const attribution =
  'NYC Department of City Planning, Borough Boundaries (water excluded), NYC Open Data';
const coordinates = [];
for (let index = 0; index < lngLat.length; index += 8) {
  coordinates.push(`    ${lngLat.slice(index, index + 8).join(', ')}`);
}
const source = `/**
 * Generated ${new Date().toISOString().slice(0, 10)} from NYC DCP Borough Boundaries (water excluded).
 * Source: ${BOUNDARY_URL}
 * License unspecified; terms: https://opendata.cityofnewyork.us/overview/#termsofuse
 * Coordinate data attribution: NYC Department of City Planning.
 * Regenerate: node examples/deck/spatial-analysis-explorer/scripts/build-new-york-land.mjs
 * POI extent + 500 m; up to 5 m simplification (refined at crossings); parts >= 0.5 ha;
 * 1e-5 degree precision.
 * One multipolygon feature, implicitly closed rings; see generator header for details.
 * ${lngLat.length / 2} vertices, ${polygons.length} polygons, approximate area ${(area / 1e6).toFixed(6)} km².
 */
export const NEW_YORK_LAND: {
  attribution: string;
  lngLat: number[];
  ringOffsets: number[];
  polygonOffsets: number[];
} = {
  attribution: '${attribution}',
  lngLat: [
${coordinates.join(',\n')}
  ],
  ringOffsets: [${ringOffsets.join(', ')}],
  polygonOffsets: [${polygonOffsets.join(', ')}]
};
`;
const bytes = Buffer.byteLength(source);
if (!polygons.length || lngLat.length / 2 > 3000 || bytes > 80000) {
  throw new Error(
    `Dataset budget exceeded or empty: ${lngLat.length / 2} vertices, ${bytes} bytes`
  );
}
await writeFile(new URL('../new-york-land.ts', import.meta.url), source);
console.log(
  JSON.stringify({
    vertices: lngLat.length / 2,
    polygons: polygons.length,
    rings: ringOffsets.length - 1,
    bytes,
    areaSquareKilometers: area / 1e6,
    bounds
  })
);
